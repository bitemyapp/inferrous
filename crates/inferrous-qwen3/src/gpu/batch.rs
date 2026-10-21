//! Paged, ragged execution with reusable session ownership and bounded scratch.
use super::*;
use crate::{
    cache::{CacheLease, CachePool, CacheUsage},
    config::{CacheMode, EngineConfig, PAGE_TOKENS},
};
use anyhow::ensure;
use serde::Serialize;
use std::collections::VecDeque;

pub struct Session<'engine> {
    owner: &'engine BatchEngine,
    cache: CacheLease,
    tokens: Vec<u32>,
}
impl<'engine> Session<'engine> {
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }
    pub fn cached_tokens(&self) -> usize {
        self.cache.written_tokens()
    }

    pub fn append(&mut self, tokens: &[u32]) -> Result<()> {
        ensure!(
            self.tokens
                .len()
                .checked_add(tokens.len())
                .is_some_and(|n| n < self.owner.config.max_context),
            "session exceeds context"
        );
        ensure!(
            tokens.iter().all(|&id| (id as usize) < VOCAB),
            "token outside vocabulary"
        );
        self.tokens.extend_from_slice(tokens);
        Ok(())
    }

    /// Fork immutable full pages; the unfinished tail is recomputed privately.
    pub fn fork(&self) -> Result<Session<'engine>> {
        let cached = self.cache.written_tokens() / PAGE_TOKENS * PAGE_TOKENS;
        Ok(Session {
            owner: self.owner,
            cache: self.cache.fork_prefix(cached)?,
            tokens: self.tokens.clone(),
        })
    }
}

#[derive(Clone, Debug)]
pub struct SessionOptions {
    pub max_tokens: usize,
    pub stop_ids: Vec<u32>,
    pub cache: CacheMode,
}

#[derive(Serialize)]
pub struct BatchGeneration {
    pub generations: Vec<Generation>,
    pub elapsed_seconds: f64,
    pub processed_tokens: usize,
    pub reused_tokens: usize,
}

struct Prefix {
    tokens: Vec<u32>,
    cache: CacheLease,
}

/// Partition packed rows without ever sharing a tile across sequence slots or
/// nonconsecutive positions. Each pair is (first row, count), with count <= 4.
fn query_tiles(info: &[u32]) -> Result<Vec<u32>> {
    ensure!(info.len() % 2 == 0, "incomplete query row metadata");
    ensure!(info.len() / 2 <= 1024, "too many query rows");
    let mut tiles = Vec::new();
    let mut first = 0;
    while first < info.len() / 2 {
        let mut count = 1;
        while count < 4 && first + count < info.len() / 2 {
            let row = first + count;
            if info[row * 2] != info[first * 2]
                || info[(row - 1) * 2 + 1].checked_add(1) != Some(info[row * 2 + 1])
            {
                break;
            }
            count += 1;
        }
        tiles.extend([first as u32, count as u32]);
        first += count;
    }
    Ok(tiles)
}

struct Step {
    // Captured storage must be destroyed before any of its allocations.
    graph: Option<Graph>,
    shape: (usize, usize, usize),
    w: Workspace,
    input: DeviceBuffer<u32>,
    info: DeviceBuffer<u32>,
    query_tiles: DeviceBuffer<u32>,
    query_tile_count: usize,
    table: DeviceBuffer<u32>,
    outputs: DeviceBuffer<u32>,
    stream: Arc<CudaStream>,
}
impl Step {
    fn new(
        stream: &Arc<CudaStream>,
        shape: (usize, usize, usize),
        table_len: usize,
    ) -> Result<Self> {
        let (n, output_rows, stride) = shape;
        Ok(Self {
            graph: None,
            shape,
            w: Workspace::with_shape(stream, n, output_rows.max(1), stride)?,
            input: DeviceBuffer::zeroed(stream, n)?,
            info: DeviceBuffer::zeroed(stream, n * 2)?,
            query_tiles: DeviceBuffer::zeroed(stream, n * 2)?,
            query_tile_count: 0,
            table: DeviceBuffer::zeroed(stream, table_len)?,
            outputs: DeviceBuffer::zeroed(stream, output_rows.max(1))?,
            stream: stream.clone(),
        })
    }
}
impl Drop for Step {
    fn drop(&mut self) {
        // Also fence a partially enqueued forward pass before fields are freed.
        self.stream.context().record_err(self.stream.synchronize());
    }
}

struct Execution {
    // A step can capture these caches: fence and destroy its graph first.
    decode: Option<Step>,
    prefill: Option<Step>,
    prefixes: VecDeque<Prefix>,
    caches: Vec<Cache>,
    poisoned: bool,
}

struct ExecutionFence(Option<Arc<CudaStream>>);
impl Drop for ExecutionFence {
    fn drop(&mut self) {
        // A host panic must also fence before caller-owned leases can unwind.
        if let Some(stream) = &self.0 {
            stream.context().record_err(stream.synchronize());
        }
    }
}

pub struct BatchEngine {
    execution: Mutex<Execution>,
    pool: CachePool,
    gpu: GpuModel,
    config: EngineConfig,
    model_root: String,
    device_name: String,
}

impl BatchEngine {
    pub fn new(model: &Model, config: EngineConfig) -> Result<Self> {
        config.validate(model.max_context())?;
        let device_name = CudaContext::new(0)?.device_name()?;
        config.hardware.validate_device(&device_name)?;
        let pool = CachePool::with_byte_budget(PAGE_TOKENS, config.kv_cache_bytes)?;
        let gpu = GpuModel::weights_only(model)?;
        let elements = pool.usage().total_pages * PAGE_TOKENS * 1024;
        let caches = (0..36)
            .map(|_| {
                Ok(Cache {
                    k: DeviceBuffer::zeroed(&gpu.stream, elements)?,
                    v: DeviceBuffer::zeroed(&gpu.stream, elements)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        gpu.stream.synchronize()?;
        Ok(Self {
            execution: Mutex::new(Execution {
                caches,
                decode: None,
                prefill: None,
                prefixes: VecDeque::new(),
                poisoned: false,
            }),
            pool,
            gpu,
            config,
            model_root: model.root.clone(),
            device_name,
        })
    }

    pub fn config(&self) -> &EngineConfig {
        &self.config
    }
    pub fn device_name(&self) -> &str {
        &self.device_name
    }
    pub fn cache_usage(&self) -> CacheUsage {
        self.pool.usage()
    }
    pub fn clear_prefixes(&self) -> Result<()> {
        self.execution
            .lock()
            .map_err(|_| anyhow::anyhow!("batch execution mutex poisoned"))?
            .prefixes
            .clear();
        Ok(())
    }

    pub fn session(&self, prompt: &[u32]) -> Result<Session<'_>> {
        self.make_session(prompt, true)
    }

    fn make_session(&self, prompt: &[u32], reuse: bool) -> Result<Session<'_>> {
        Request {
            prompt_ids: prompt.to_vec(),
            max_tokens: 1,
            stop_ids: vec![],
        }
        .validate_for_context(self.config.max_context)?;
        let mut execution = self
            .execution
            .lock()
            .map_err(|_| anyhow::anyhow!("batch execution mutex poisoned"))?;
        ensure!(
            !execution.poisoned,
            "batch engine failed previously; recreate it"
        );
        let best = if reuse {
            execution
                .prefixes
                .iter()
                .enumerate()
                .filter(|(_, p)| p.tokens.len() < prompt.len() && prompt.starts_with(&p.tokens))
                .max_by_key(|(_, p)| p.tokens.len())
                .map(|(i, _)| i)
        } else {
            None
        };
        let cache = if let Some(index) = best {
            let prefix = execution.prefixes.remove(index).expect("index found above");
            let lease = prefix.cache.fork_prefix(prefix.tokens.len())?;
            execution.prefixes.push_back(prefix);
            lease
        } else {
            self.pool.lease(0)?
        };
        Ok(Session {
            owner: self,
            cache,
            tokens: prompt.to_vec(),
        })
    }

    fn reserve(
        &self,
        execution: &mut Execution,
        session: &mut Session<'_>,
        tokens: usize,
    ) -> Result<()> {
        loop {
            match session.cache.reserve(tokens) {
                Ok(()) => return Ok(()),
                Err(error) => {
                    if execution.prefixes.pop_front().is_none() {
                        return Err(error);
                    }
                }
            }
        }
    }

    fn remember(&self, session: &Session<'_>, prompt_len: usize) -> Result<()> {
        if self.config.prefix_cache_entries == 0 {
            return Ok(());
        }
        let length = session
            .cache
            .written_tokens()
            .min(prompt_len.saturating_sub(1))
            / PAGE_TOKENS
            * PAGE_TOKENS;
        if length == 0 {
            return Ok(());
        }
        let tokens = session.tokens[..length].to_vec();
        let mut execution = self
            .execution
            .lock()
            .map_err(|_| anyhow::anyhow!("batch execution mutex poisoned"))?;
        if let Some(index) = execution.prefixes.iter().position(|p| p.tokens == tokens) {
            execution.prefixes.remove(index);
        }
        execution.prefixes.push_back(Prefix {
            tokens,
            cache: session.cache.fork_prefix(length)?,
        });
        while execution.prefixes.len() > self.config.prefix_cache_entries {
            execution.prefixes.pop_front();
        }
        Ok(())
    }

    /// Admit bounded waves when a submitted batch exceeds available KV capacity.
    pub fn generate(
        &self,
        model: &Model,
        requests: &[Request],
        cache: CacheMode,
    ) -> Result<BatchGeneration> {
        ensure!(!requests.is_empty(), "empty batch");
        for request in requests {
            request.validate_for_context(self.config.max_context)?;
        }
        let start = Instant::now();
        let mut result = BatchGeneration {
            generations: Vec::new(),
            elapsed_seconds: 0.0,
            processed_tokens: 0,
            reused_tokens: 0,
        };
        let mut index = 0;
        while index < requests.len() {
            let first = index;
            let mut sessions = Vec::new();
            let mut options = Vec::new();
            while index < requests.len() && sessions.len() < self.config.max_batch {
                let request = &requests[index];
                let mut session =
                    self.make_session(&request.prompt_ids, cache == CacheMode::Reuse)?;
                let mut execution = self
                    .execution
                    .lock()
                    .map_err(|_| anyhow::anyhow!("batch execution mutex poisoned"))?;
                let admitted = self.reserve(
                    &mut execution,
                    &mut session,
                    request.prompt_ids.len() + request.max_tokens,
                );
                if let Err(error) = admitted {
                    if sessions.is_empty() {
                        return Err(error);
                    }
                    break;
                }
                sessions.push(session);
                options.push(SessionOptions {
                    max_tokens: request.max_tokens,
                    stop_ids: request.stop_ids.clone(),
                    cache,
                });
                index += 1;
            }
            let queue_seconds = start.elapsed().as_secs_f64();
            let mut wave = self.generate_sessions(model, &mut sessions, &options)?;
            for generation in &mut wave.generations {
                generation.request_seconds += queue_seconds;
                generation.step_seconds[0] += queue_seconds;
            }
            result.processed_tokens += wave.processed_tokens;
            result.reused_tokens += wave.reused_tokens;
            result.generations.extend(wave.generations);
            if cache == CacheMode::Reuse {
                for (offset, session) in sessions.iter().enumerate() {
                    self.remember(session, requests[first + offset].prompt_ids.len())?;
                }
            }
        }
        result.elapsed_seconds = start.elapsed().as_secs_f64();
        Ok(result)
    }

    /// Sessions borrow their engine, so device storage outlives every page lease.
    pub fn generate_sessions(
        &self,
        model: &Model,
        sessions: &mut [Session<'_>],
        options: &[SessionOptions],
    ) -> Result<BatchGeneration> {
        ensure!(
            model.root == self.model_root,
            "session model does not match resident weights"
        );
        ensure!(
            !sessions.is_empty()
                && sessions.len() <= self.config.max_batch
                && sessions.len() == options.len(),
            "invalid session batch size"
        );
        for (session, option) in sessions.iter().zip(options) {
            ensure!(
                std::ptr::eq(session.owner, self) && self.pool.owns(&session.cache),
                "session belongs to another engine"
            );
            Request {
                prompt_ids: session.tokens.clone(),
                max_tokens: option.max_tokens,
                stop_ids: option.stop_ids.clone(),
            }
            .validate_for_context(self.config.max_context)?;
        }
        let mut execution = self
            .execution
            .lock()
            .map_err(|_| anyhow::anyhow!("batch execution mutex poisoned"))?;
        ensure!(
            !execution.poisoned,
            "batch engine failed previously; recreate it"
        );
        self.gpu.stream.synchronize()?;
        let mut lease_fence = ExecutionFence(Some(self.gpu.stream.clone()));
        for (session, option) in sessions.iter_mut().zip(options) {
            if option.cache == CacheMode::Recompute {
                session.cache.reset();
            }
            self.reserve(
                &mut execution,
                session,
                session.tokens.len() + option.max_tokens,
            )?;
        }
        let start = Instant::now();
        let reused_tokens = sessions.iter().map(|s| s.cache.written_tokens()).sum();
        let mut generations: Vec<_> = sessions
            .iter()
            .map(|_| Generation {
                token_ids: Vec::new(),
                text: String::new(),
                finish_reason: "length".into(),
                step_seconds: Vec::new(),
                request_seconds: 0.0,
                logit_sha256: Vec::new(),
            })
            .collect();
        let mut last_emission = vec![start; sessions.len()];
        let mut done = vec![false; sessions.len()];
        let mut processed_tokens = 0;
        let pages_per_sequence = self.config.max_context.div_ceil(PAGE_TOKENS);
        // All transient GPU storage belongs to Execution/Step. An error still
        // fences it before caller-owned session leases can be released.
        let outcome = (|| -> Result<()> {
            while done.iter().any(|&finished| !finished) {
                // No sequence can advance by more than the whole token chunk.
                // Bound the next visible prefix before packing rows, allowing
                // larger early chunks while keeping each score allocation bounded.
                let score_stride = sessions
                    .iter()
                    .zip(&done)
                    .filter(|(_, finished)| !**finished)
                    .map(|(session, _)| {
                        session
                            .tokens
                            .len()
                            .min(session.cache.written_tokens() + self.config.prefill_chunk)
                    })
                    .max()
                    .expect("an unfinished sequence exists")
                    .next_power_of_two()
                    .max(PAGE_TOKENS);
                let row_budget = self
                    .config
                    .prefill_chunk
                    .min(self.config.score_scratch_bytes / (32 * score_stride * 4));
                let mut input = Vec::new();
                let mut info = Vec::new();
                let mut output_rows = Vec::new();
                let mut output_slots = Vec::new();
                let mut ends = Vec::new();
                let mut remaining = done.iter().filter(|&&finished| !finished).count();
                let mut visible = 0;
                for (slot, session) in sessions.iter().enumerate() {
                    if done[slot] {
                        continue;
                    }
                    let begin = session.cache.written_tokens();
                    ensure!(
                        begin < session.tokens.len(),
                        "session has no unprocessed token"
                    );
                    let quota = (row_budget - input.len()) / remaining;
                    remaining -= 1;
                    let end = session.tokens.len().min(begin + quota);
                    session.cache.check_append(end)?;
                    for position in begin..end {
                        input.push(session.tokens[position]);
                        info.extend([slot as u32, position as u32]);
                    }
                    ends.push((slot, end));
                    visible = visible.max(end);
                    if end == session.tokens.len() {
                        output_rows.push((input.len() - 1) as u32);
                        output_slots.push(slot);
                    }
                }
                let stride = visible.next_power_of_two().max(PAGE_TOKENS);
                let shape = (input.len(), output_rows.len(), stride);
                let mut table = vec![u32::MAX; self.config.max_batch * pages_per_sequence];
                for (slot, session) in sessions.iter().enumerate() {
                    for (logical, page) in session.cache.pages().iter().enumerate() {
                        table[slot * pages_per_sequence + logical] = page.index();
                    }
                }
                let Execution {
                    caches,
                    decode,
                    prefill,
                    ..
                } = &mut *execution;
                let storage = if input.len() == output_slots.len() {
                    decode
                } else {
                    prefill
                };
                if storage.as_ref().is_none_or(|step| step.shape != shape) {
                    // Release the previous shape before allocating its replacement,
                    // avoiding a temporary doubling of the score workspace.
                    drop(storage.take());
                    *storage = Some(Step::new(&self.gpu.stream, shape, table.len())?);
                }
                let step = storage.as_mut().expect("allocated above");
                step.input.copy_from_host(&self.gpu.stream, &input)?;
                step.info.copy_from_host(&self.gpu.stream, &info)?;
                step.query_tile_count = input.len();
                if self.config.tiled_attention && input.len() != output_slots.len() {
                    let mut tiles = query_tiles(&info)?;
                    step.query_tile_count = tiles.len() / 2;
                    tiles.resize(input.len() * 2, 0);
                    step.query_tiles.copy_from_host(&self.gpu.stream, &tiles)?;
                }
                step.table.copy_from_host(&self.gpu.stream, &table)?;
                if !output_rows.is_empty() {
                    step.outputs
                        .copy_from_host(&self.gpu.stream, &output_rows)?;
                }
                if self.config.cuda_graphs && input.len() == output_slots.len() {
                    if step.graph.is_none() {
                        // SAFETY: Step and Execution own all captured buffers.
                        // Metadata copies complete before capture/replay, and the
                        // execution mutex excludes every concurrent mutation.
                        // Multirow kernels use ordinary full-completion edges.
                        step.graph = Some(unsafe {
                            Graph::capture(&self.gpu.stream, false, || {
                                self.forward(step, caches, pages_per_sequence)
                            })?
                        });
                    }
                    unsafe {
                        step.graph.as_ref().expect("captured above").launch()?;
                    }
                } else {
                    self.forward(step, caches, pages_per_sequence)?;
                }
                self.gpu.stream.synchronize()?;
                for (slot, end) in ends {
                    sessions[slot].cache.commit_append(end)?;
                }
                processed_tokens += input.len();
                if output_slots.is_empty() {
                    continue;
                }
                let next = step.w.next.to_host_vec(&self.gpu.stream)?;
                let logits = if self.config.verify_logits {
                    Some(step.w.logits.to_host_vec(&self.gpu.stream)?)
                } else {
                    None
                };
                for (row, &slot) in output_slots.iter().enumerate() {
                    let token = next[row];
                    ensure!((token as usize) < VOCAB, "nonfinite or empty logits");
                    let generation = &mut generations[slot];
                    if let Some(logits) = &logits {
                        generation
                            .logit_sha256
                            .push(logit_hash(&logits[row * VOCAB..(row + 1) * VOCAB]));
                    }
                    let now = Instant::now();
                    generation
                        .step_seconds
                        .push(now.duration_since(last_emission[slot]).as_secs_f64());
                    last_emission[slot] = now;
                    generation.token_ids.push(token);
                    sessions[slot].tokens.push(token);
                    if options[slot].stop_ids.contains(&token) {
                        generation.finish_reason = "stop".into();
                        done[slot] = true;
                    } else if generation.token_ids.len() == options[slot].max_tokens {
                        done[slot] = true;
                    }
                    if done[slot] {
                        generation.request_seconds = start.elapsed().as_secs_f64();
                    }
                    if options[slot].cache == CacheMode::Recompute {
                        sessions[slot].cache.restart()?;
                    }
                }
            }
            Ok(())
        })();
        let fence = self.gpu.stream.synchronize();
        if fence.is_ok() {
            lease_fence.0 = None;
        }
        if outcome.is_err() || fence.is_err() {
            execution.poisoned = true;
        }
        outcome?;
        fence?;
        for generation in &mut generations {
            generation.text = model
                .tokenizer
                .decode(&generation.token_ids, true)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        Ok(BatchGeneration {
            generations,
            elapsed_seconds: start.elapsed().as_secs_f64(),
            processed_tokens,
            reused_tokens,
        })
    }

    fn forward(
        &self,
        step: &mut Step,
        caches: &mut [Cache],
        pages_per_sequence: usize,
    ) -> Result<()> {
        let gpu = &self.gpu;
        let policy = if self.config.tuned_projections {
            ProjectionPolicy::TunedBatch
        } else {
            ProjectionPolicy::Original
        };
        let w = &mut step.w;
        let (nr, outputs, stride) = step.shape;
        let tuned_attention = self.config.tuned_attention;
        // Grouped PV loses occupancy at small batch sizes. QK has independent
        // key-page tiles and can profit from grouping even for long solo decode.
        let grouped_scores =
            tuned_attention && (nr >= 16 || (nr >= 8 && stride >= 512) || stride >= 4096);
        let grouped_values = tuned_attention && nr >= 16;
        // Decode has one row per session and cannot satisfy this occupancy
        // threshold. Only eager prefill can change the tile count/launch grid.
        let tiled_attention = self.config.tiled_attention && nr >= 2 * step.query_tile_count;
        // SAFETY: admission validates all tokens, row positions and page leases.
        // Every producer/consumer is serialized on one stream. Different rows
        // write distinct sequence positions, and all scratch allocations match
        // the complete row shapes supplied to the generated kernel interfaces.
        unsafe {
            gpu.module.embedding(
                &gpu.stream,
                cfg(w.hidden.len()),
                &step.input,
                &gpu.embedding,
                &mut w.hidden,
            )?;
        }
        for (layer, cache) in gpu.layers.iter().zip(caches) {
            gpu.norm_quantize(
                &w.hidden,
                &layer.gains["norm1"],
                &mut w.norm,
                &mut w.quant4,
                &mut w.scales,
                &mut w.maxima,
            )?;
            gpu.linear_quantized(
                &w.quant4,
                &w.scales,
                &layer.weights["q"],
                &mut w.q,
                w.map4.as_ref(),
                policy,
            )?;
            gpu.linear_quantized(
                &w.quant4,
                &w.scales,
                &layer.weights["k"],
                &mut w.k,
                w.map4.as_ref(),
                policy,
            )?;
            gpu.linear_quantized(
                &w.quant4,
                &w.scales,
                &layer.weights["v"],
                &mut w.v,
                w.map4.as_ref(),
                policy,
            )?;
            unsafe {
                gpu.module.prepare_qk_paged(
                    &gpu.stream,
                    rows(nr * 40, 128),
                    &mut w.q,
                    &mut w.k,
                    &w.v,
                    &layer.gains["qnorm"],
                    &layer.gains["knorm"],
                    &gpu.rope,
                    &step.info,
                    &step.table,
                    pages_per_sequence as u32,
                    &mut cache.k,
                    &mut cache.v,
                )?;
                if tiled_attention {
                    gpu.module.attention_scores_tiled(
                        &gpu.stream,
                        rows(step.query_tile_count * 8 * (stride / 64), 256),
                        &w.q,
                        &cache.k,
                        &step.info,
                        &step.table,
                        pages_per_sequence as u32,
                        stride as u32,
                        (stride / 64) as u32,
                        &step.query_tiles,
                        &mut w.scores,
                    )?;
                } else if grouped_scores {
                    gpu.module.attention_scores_grouped(
                        &gpu.stream,
                        rows(nr * 8 * (stride / 64), 128),
                        &w.q,
                        &cache.k,
                        &step.info,
                        &step.table,
                        pages_per_sequence as u32,
                        stride as u32,
                        (stride / 64) as u32,
                        &mut w.scores,
                    )?;
                } else {
                    gpu.module.attention_scores_paged(
                        &gpu.stream,
                        rows(nr * 32 * (stride / 64), 128),
                        &w.q,
                        &cache.k,
                        &step.info,
                        &step.table,
                        pages_per_sequence as u32,
                        stride as u32,
                        (stride / 64) as u32,
                        &mut w.scores,
                    )?;
                }
                gpu.module.attention_softmax_paged(
                    &gpu.stream,
                    rows(nr * 32, 128),
                    &mut w.scores,
                    &step.info,
                    stride as u32,
                )?;
                if tiled_attention {
                    gpu.module.attention_values_tiled(
                        &gpu.stream,
                        rows(step.query_tile_count * 8 * 4, 128),
                        &w.scores,
                        &cache.v,
                        &step.info,
                        &step.table,
                        pages_per_sequence as u32,
                        stride as u32,
                        &step.query_tiles,
                        &mut w.attention,
                    )?;
                } else if grouped_values {
                    gpu.module.attention_values_grouped(
                        &gpu.stream,
                        rows(nr * 8 * 4, 32),
                        &w.scores,
                        &cache.v,
                        &step.info,
                        &step.table,
                        pages_per_sequence as u32,
                        stride as u32,
                        &mut w.attention,
                    )?;
                } else {
                    gpu.module.attention_values_paged(
                        &gpu.stream,
                        rows(nr * 32 * 4, 32),
                        &w.scores,
                        &cache.v,
                        &step.info,
                        &step.table,
                        pages_per_sequence as u32,
                        stride as u32,
                        &mut w.attention,
                    )?;
                }
            }
            gpu.linear_residual(
                &w.attention,
                &layer.weights["o"],
                &mut w.quant4,
                &mut w.scales,
                &mut w.o,
                &w.maxima,
                &w.hidden,
                &mut w.residual,
                w.map4.as_ref(),
                false,
                policy,
            )?;
            gpu.norm_quantize(
                &w.residual,
                &layer.gains["norm2"],
                &mut w.norm,
                &mut w.quant4,
                &mut w.scales,
                &mut w.maxima,
            )?;
            if nr == 1 {
                let gate = &layer.weights["gate"];
                let up = &layer.weights["up"];
                unsafe {
                    gpu.module.linear_swiglu(
                        &gpu.stream,
                        rows(gate.n.div_ceil(8), 256),
                        &w.quant4,
                        &w.scales,
                        gate.k as u32,
                        &gate.w,
                        &gate.scale,
                        &mut w.gate,
                        &up.w,
                        &up.scale,
                        &mut w.up,
                        &mut w.act,
                        &mut w.maxima,
                        0,
                    )?;
                }
            } else {
                gpu.linear_quantized(
                    &w.quant4,
                    &w.scales,
                    &layer.weights["gate"],
                    &mut w.gate,
                    w.map4.as_ref(),
                    policy,
                )?;
                gpu.linear_quantized(
                    &w.quant4,
                    &w.scales,
                    &layer.weights["up"],
                    &mut w.up,
                    w.map4.as_ref(),
                    policy,
                )?;
                unsafe {
                    gpu.module.swiglu(
                        &gpu.stream,
                        cfg(w.act.len()),
                        &w.gate,
                        &w.up,
                        &mut w.act,
                        &mut w.maxima,
                    )?;
                }
            }
            gpu.linear_residual(
                &w.act,
                &layer.weights["down"],
                &mut w.quant12,
                &mut w.scales,
                &mut w.down,
                &w.maxima,
                &w.residual,
                &mut w.hidden,
                w.map12.as_ref(),
                false,
                policy,
            )?;
        }
        if outputs > 0 {
            gpu.norm(&w.hidden, &gpu.final_gain, &mut w.norm)?;
            unsafe {
                let head_group = if self.config.tuned_head {
                    match outputs {
                        0..=2 => 1,
                        3..=4 => 4,
                        _ if outputs % 16 == 0 || outputs % 16 >= 9 => 16,
                        _ => 8,
                    }
                } else {
                    1
                };
                match head_group {
                    4 => gpu.module.head_packed_four(
                        &gpu.stream,
                        rows((outputs.div_ceil(4) * VOCAB).div_ceil(8), 256),
                        &w.norm,
                        &step.outputs,
                        &gpu.head.words,
                        &gpu.head.offsets,
                        &gpu.head.exceptions,
                        &mut w.logits,
                    )?,
                    8 => gpu.module.head_packed_eight_staged(
                        &gpu.stream,
                        rows((outputs.div_ceil(8) * VOCAB).div_ceil(8), 256),
                        &w.norm,
                        &step.outputs,
                        &gpu.head.words,
                        &gpu.head.offsets,
                        &gpu.head.exceptions,
                        &mut w.logits,
                    )?,
                    16 => gpu.module.head_packed_sixteen_staged(
                        &gpu.stream,
                        rows((outputs.div_ceil(16) * VOCAB).div_ceil(8), 256),
                        &w.norm,
                        &step.outputs,
                        &gpu.head.words,
                        &gpu.head.offsets,
                        &gpu.head.exceptions,
                        &mut w.logits,
                    )?,
                    _ => gpu.module.head_packed_rows(
                        &gpu.stream,
                        cfg(outputs * VOCAB * 32),
                        &w.norm,
                        &step.outputs,
                        &gpu.head.words,
                        &gpu.head.offsets,
                        &gpu.head.exceptions,
                        &mut w.logits,
                        0,
                    )?,
                }
                gpu.module.select_token_rows(
                    &gpu.stream,
                    rows(outputs, 1024),
                    &w.logits,
                    &mut w.next,
                )?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::HardwareProfile;
    use anyhow::Context;

    #[test]
    fn query_tiles_preserve_sequence_and_position_boundaries() -> Result<()> {
        assert_eq!(query_tiles(&[])?, Vec::<u32>::new());
        assert!(query_tiles(&[0]).is_err());
        assert!(query_tiles(&vec![0; 2050]).is_err());
        let info = [
            1,
            61,
            1,
            62,
            1,
            63,
            1,
            64,
            1,
            65,
            0,
            66,
            0,
            67,
            0,
            69,
            1,
            u32::MAX,
            1,
            0,
        ];
        assert_eq!(query_tiles(&info)?, [0, 4, 4, 1, 5, 2, 7, 1, 8, 1, 9, 1]);
        Ok(())
    }

    fn request(length: usize, seed: u32, outputs: usize) -> Request {
        Request {
            prompt_ids: (0..length)
                .map(|i| 42 + (i as u32 * 17 + seed) % 1000)
                .collect(),
            max_tokens: outputs,
            stop_ids: vec![],
        }
    }

    fn test_hardware() -> Result<HardwareProfile> {
        let name = CudaContext::new(0)?.device_name()?;
        for profile in [HardwareProfile::Rtx5090, HardwareProfile::RtxPro6000] {
            if profile.validate_device(&name).is_ok() {
                return Ok(profile);
            }
        }
        anyhow::bail!("no test profile for {name}")
    }

    fn same(actual: &Generation, expected: &Generation) {
        assert_eq!(actual.token_ids, expected.token_ids);
        assert_eq!(actual.logit_sha256, expected.logit_sha256);
        assert_eq!(actual.finish_reason, expected.finish_reason);
        assert_eq!(actual.text, expected.text);
    }

    #[test]
    #[ignore = "requires CLEAN_LONG_MODEL_BUNDLE with 32768 RoPE positions and 9 GiB KV capacity"]
    fn native_context_ragged_recompute_and_chunk_invariance() -> Result<()> {
        let bundle = std::env::var_os("CLEAN_LONG_MODEL_BUNDLE")
            .context("set CLEAN_LONG_MODEL_BUNDLE to an extended 32768-token bundle")?;
        let model = Model::load(Path::new(&bundle))?;
        let mut config = EngineConfig::for_hardware(test_hardware()?, 32768);
        config.max_batch = 4;
        config.kv_cache_bytes = 9 << 30;
        let mut engine = BatchEngine::new(&model, config)?;
        let requests: Vec<_> = [8192, 16384, 32766]
            .into_iter()
            .enumerate()
            .map(|(i, length)| request(length, i as u32 + 50, 2))
            .collect();
        let cold = engine.generate(&model, &requests, CacheMode::Reuse)?;
        eprintln!("native context cold pass: {:.3}s", cold.elapsed_seconds);
        assert_eq!(cold.reused_tokens, 0);
        assert_eq!(cold.processed_tokens, 8193 + 16385 + 32767);
        let warm = engine.generate(&model, &requests, CacheMode::Reuse)?;
        eprintln!("native context warm pass: {:.3}s", warm.elapsed_seconds);
        assert_eq!(warm.reused_tokens, 8128 + 16320 + 32704);
        assert_eq!(
            warm.processed_tokens + warm.reused_tokens,
            cold.processed_tokens
        );
        for (actual, expected) in warm.generations.iter().zip(&cold.generations) {
            same(actual, expected);
        }
        engine.clear_prefixes()?;
        engine.config.prefill_chunk = 128;
        engine.config.cuda_graphs = false;
        engine.config.tiled_attention = true;
        engine.config.tuned_attention = true;
        engine.config.tuned_projections = true;
        engine.config.tuned_head = true;
        let fresh = engine.generate(&model, &requests, CacheMode::Recompute)?;
        assert_eq!(fresh.reused_tokens, 0);
        assert_eq!(fresh.processed_tokens, 2 * (8192 + 16384 + 32766) + 3);
        for (actual, expected) in fresh.generations.iter().zip(&cold.generations) {
            same(actual, expected);
        }
        assert_eq!(engine.cache_usage().used_pages, 0);
        eprintln!(
            "native context cold {:.3}s; warm {:.3}s; recompute {:.3}s",
            cold.elapsed_seconds, warm.elapsed_seconds, fresh.elapsed_seconds
        );
        Ok(())
    }

    #[test]
    fn tiled_prefill_matches_original_across_chunks_graphs_and_cache_modes() -> Result<()> {
        let bundle = std::env::var_os("CLEAN_MODEL_BUNDLE")
            .context("set CLEAN_MODEL_BUNDLE for batched model validation")?;
        let model = Model::load(Path::new(&bundle))?;
        let mut config = EngineConfig::for_hardware(test_hardware()?, 4096);
        config.max_batch = 16;
        config.kv_cache_bytes = 512 << 20;
        config.tuned_attention = false;
        config.tuned_projections = false;
        config.tuned_head = false;
        let requests: Vec<_> = [1, 5, 65, 129, 257, 33, 128, 64, 17, 256, 67, 3]
            .into_iter()
            .enumerate()
            .map(|(i, n)| request(n, i as u32, 2 + i % 3))
            .collect();
        let baseline = BatchEngine::new(&model, config.clone())?;
        let expected = baseline.generate(&model, &requests, CacheMode::Reuse)?;
        drop(baseline);
        config.tiled_attention = true;
        config.tuned_attention = true;
        config.tuned_projections = true;
        config.tuned_head = true;
        for (chunk, graphs) in [(64, false), (512, true)] {
            config.prefill_chunk = chunk;
            config.cuda_graphs = graphs;
            let engine = BatchEngine::new(&model, config.clone())?;
            for mode in [CacheMode::Reuse, CacheMode::Reuse, CacheMode::Recompute] {
                let actual = engine.generate(&model, &requests, mode)?;
                for (actual, expected) in actual.generations.iter().zip(&expected.generations) {
                    same(actual, expected);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn ragged_sessions_prefixes_and_recomputation_match_legacy() -> Result<()> {
        let bundle = std::env::var_os("CLEAN_MODEL_BUNDLE")
            .context("set CLEAN_MODEL_BUNDLE for batched model validation")?;
        let model = Model::load(Path::new(&bundle))?;
        let mut config = EngineConfig::for_hardware(test_hardware()?, 4096);
        config.max_batch = 8;
        config.kv_cache_bytes = 512 << 20;
        let mut engine = BatchEngine::new(&model, config)?;
        let requests: Vec<_> = [1, 5, 65, 129, 257]
            .into_iter()
            .enumerate()
            .map(|(i, n)| request(n, i as u32, 4))
            .collect();
        // The paged engine intentionally has no legacy execution allocation.
        // Keep the independent contiguous-cache oracle in a separate model.
        let oracle = GpuModel::new(&model)?;
        let expected = requests
            .iter()
            .map(|r| oracle.generate(&model, r, None))
            .collect::<Result<Vec<_>>>()?;
        let cold = engine.generate(&model, &requests, CacheMode::Reuse)?;
        assert_eq!(cold.reused_tokens, 0);
        assert_eq!(
            cold.processed_tokens,
            requests
                .iter()
                .map(|r| r.prompt_ids.len() + 3)
                .sum::<usize>()
        );
        for (actual, expected) in cold.generations.iter().zip(&expected) {
            same(actual, expected);
        }
        let warm = engine.generate(&model, &requests, CacheMode::Reuse)?;
        assert_eq!(warm.reused_tokens, 64 + 128 + 256);
        assert_eq!(
            warm.processed_tokens + warm.reused_tokens,
            cold.processed_tokens
        );
        for (actual, expected) in warm.generations.iter().zip(&expected) {
            same(actual, expected);
        }
        // Force the workspace budget to split these prefixes into smaller
        // chunks even though the configured token limit is much larger.
        engine.config.score_scratch_bytes = 4 << 20;
        engine.config.validate(model.max_context())?;
        let recomputed = engine.generate(&model, &requests, CacheMode::Recompute)?;
        assert_eq!(recomputed.reused_tokens, 0);
        assert_eq!(
            recomputed.processed_tokens,
            requests
                .iter()
                .map(|r| 4 * r.prompt_ids.len() + 6)
                .sum::<usize>()
        );
        for (actual, expected) in recomputed.generations.iter().zip(&expected) {
            same(actual, expected);
        }
        engine.config.score_scratch_bytes = 1 << 30;
        let reversed: Vec<_> = requests.iter().rev().cloned().collect();
        engine.config.cuda_graphs = false;
        let reordered = engine.generate(&model, &reversed, CacheMode::Reuse)?;
        for (actual, expected) in reordered.generations.iter().rev().zip(&expected) {
            same(actual, expected);
        }
        engine.config.cuda_graphs = true;

        // A stopped sequence must not change surviving members or their slot mapping.
        let mut stopping = requests.clone();
        stopping[1].stop_ids = vec![expected[1].token_ids[0]];
        let stopped = engine.generate(&model, &stopping, CacheMode::Reuse)?;
        assert_eq!(stopped.generations[1].token_ids, expected[1].token_ids[..1]);
        assert_eq!(
            stopped.generations[1].logit_sha256,
            expected[1].logit_sha256[..1]
        );
        assert_eq!(stopped.generations[1].finish_reason, "stop");
        for i in [0, 2, 3, 4] {
            same(&stopped.generations[i], &expected[i]);
        }
        engine.clear_prefixes()?;
        assert_eq!(engine.cache_usage().used_pages, 0);

        // Continuing and forking share only initialized full pages. A partial
        // tail is rebuilt, and dropping either branch cannot invalidate the other.
        let mut session = engine.session(&requests[3].prompt_ids)?;
        let option = SessionOptions {
            max_tokens: 4,
            stop_ids: vec![],
            cache: CacheMode::Reuse,
        };
        let first = engine.generate_sessions(
            &model,
            std::slice::from_mut(&mut session),
            std::slice::from_ref(&option),
        )?;
        same(&first.generations[0], &expected[3]);
        let mut fork = session.fork()?;
        assert_eq!(fork.cached_tokens(), 128);
        session.append(&[31, 32, 33])?;
        fork.append(&[71, 72])?;
        let branches = [&session, &fork]
            .into_iter()
            .map(|s| {
                oracle.generate(
                    &model,
                    &Request {
                        prompt_ids: s.tokens().to_vec(),
                        max_tokens: 4,
                        stop_ids: vec![],
                    },
                    None,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let mut sessions = [session, fork];
        let continued =
            engine.generate_sessions(&model, &mut sessions, &[option.clone(), option.clone()])?;
        for (actual, expected) in continued.generations.iter().zip(&branches) {
            same(actual, expected);
        }
        // Switching an already shared session to recompute must detach it first.
        let fork = sessions[0].fork()?;
        let continuation = oracle.generate(
            &model,
            &Request {
                prompt_ids: sessions[0].tokens().to_vec(),
                max_tokens: 4,
                stop_ids: vec![],
            },
            None,
        )?;
        let uncached = engine.generate_sessions(
            &model,
            &mut sessions[..1],
            &[SessionOptions {
                cache: CacheMode::Recompute,
                ..option
            }],
        )?;
        same(&uncached.generations[0], &continuation);
        assert_eq!(sessions[0].cached_tokens(), 0);
        drop(fork);
        drop(sessions);
        assert_eq!(engine.cache_usage().used_pages, 0);

        // Force capacity waves even though max_batch allows all eight requests.
        // Eight 512-token reservations need more than this 512 MiB pool.
        let waves = vec![request(512, 21, 2); 8];
        let reference = oracle.generate(&model, &waves[0], None)?;
        let result = engine.generate(&model, &waves, CacheMode::Reuse)?;
        assert_eq!(result.generations.len(), waves.len());
        for actual in &result.generations {
            same(actual, &reference);
        }
        engine.clear_prefixes()?;
        assert_eq!(engine.cache_usage().used_pages, 0);
        let too_large = request(4090, 1, 2);
        assert!(
            engine
                .generate(&model, &[too_large], CacheMode::Reuse)
                .is_err()
        );
        // A rejected admission must not poison the resident engine.
        let after = engine.generate(&model, &requests[..1], CacheMode::Reuse)?;
        same(&after.generations[0], &expected[0]);
        Ok(())
    }
}
