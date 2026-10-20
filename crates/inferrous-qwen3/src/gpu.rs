use crate::graph::Graph;
mod batch;
#[cfg(test)]
mod profile;
use crate::{
    CONTEXT, HIDDEN, VOCAB,
    kernels::device,
    model::{Generation, Model, Request, sha},
};
use anyhow::Result;
pub use batch::{BatchEngine, BatchGeneration, Session, SessionOptions};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, PinnedHostBuffer, simt::LaunchConfig};
use std::{
    collections::BTreeMap,
    fs,
    path::Path,
    sync::{Arc, Mutex},
    time::Instant,
};
type Buffer = DeviceBuffer<f32>;
const GRAPH_TILES: [usize; 8] = [1, 2, 3, 4, 8, 16, 32, 64];

// Decode L2 prefetch hints. They never change a value, hash or token; see
// docs/qwen3-8b-w8a8/optimization/round-6.md for the measurements behind these sizes.
/// Extra CTAs appended to attention and O quantization. Each SM issues bulk L2
/// prefetches at roughly 120 GB/s, so sixteen approximately match DRAM.
const PREFETCH_CTAS: u32 = 16;
/// O-projection bytes warmed by attention's and then quantization's extra CTAs.
const O_ATTENTION: u32 = 6 << 20;
const O_QUANTIZE: u32 = 10 << 20;
/// Leading bytes of each owned weight row a projection or head warp warms
/// before waiting on its latency-bound predecessor.
const ROW_QKV: u32 = 1024;
const ROW_O: u32 = 1024;
const ROW_MLP: u32 = 512;
const ROW_DOWN: u32 = 1024;
const ROW_HEAD: u32 = 1024;
fn grow(mut config: LaunchConfig, prefetch: bool) -> LaunchConfig {
    if prefetch {
        config.grid_dim.0 += PREFETCH_CTAS;
    }
    config
}
fn row_prefetch(prefetch: bool, bytes: u32) -> u32 {
    if prefetch { bytes } else { 0 }
}
struct Matrix {
    w: DeviceBuffer<i8>,
    map: Option<DeviceBuffer<u64>>,
    scale: Buffer,
    k: usize,
    n: usize,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum ProjectionPolicy {
    Original,
    Rtx5090,
}
struct PackedHead {
    words: DeviceBuffer<u32>,
    offsets: DeviceBuffer<u32>,
    exceptions: DeviceBuffer<u16>,
}
impl PackedHead {
    fn new(stream: &Arc<CudaStream>, head: &[u16]) -> Result<Self> {
        let packed = crate::packing::PackedBf16::new(head);
        Ok(Self {
            words: DeviceBuffer::from_host(stream, &packed.words)?,
            offsets: DeviceBuffer::from_host(stream, &packed.offsets)?,
            exceptions: DeviceBuffer::from_host(stream, &packed.exceptions)?,
        })
    }
}
struct Layer {
    gains: BTreeMap<String, Buffer>,
    weights: BTreeMap<String, Matrix>,
}
struct Cache {
    k: DeviceBuffer<u16>,
    v: DeviceBuffer<u16>,
}

// Scratch belongs to the serialized execution pool. Every producer and consumer
// uses the same stream; the logit read finishes each step before control changes.
struct Workspace {
    hidden: Buffer,
    residual: Buffer,
    norm: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    attention: Buffer,
    scores: Buffer,
    o: Buffer,
    gate: Buffer,
    up: Buffer,
    act: Buffer,
    down: Buffer,
    quant4: DeviceBuffer<i8>,
    map4: Option<DeviceBuffer<u64>>,
    map12: Option<DeviceBuffer<u64>>,
    quant12: DeviceBuffer<i8>,
    scales: Buffer,
    maxima: DeviceBuffer<u32>,
    logits: Buffer,
    next: DeviceBuffer<u32>,
}
impl Workspace {
    fn new(stream: &Arc<CudaStream>, rows: usize) -> Result<Self> {
        Self::with_shape(stream, rows, 1, CONTEXT)
    }

    fn with_shape(
        stream: &Arc<CudaStream>,
        rows: usize,
        output_rows: usize,
        score_stride: usize,
    ) -> Result<Self> {
        let f = |width| -> Result<Buffer> { Ok(DeviceBuffer::zeroed(stream, rows * width)?) };
        let mut workspace = Self {
            hidden: f(HIDDEN)?,
            residual: f(HIDDEN)?,
            norm: f(HIDDEN)?,
            q: f(HIDDEN)?,
            k: f(1024)?,
            v: f(1024)?,
            attention: f(HIDDEN)?,
            scores: f(32 * score_stride)?,
            o: f(HIDDEN)?,
            gate: f(12288)?,
            up: f(12288)?,
            act: f(12288)?,
            down: f(HIDDEN)?,
            quant4: DeviceBuffer::zeroed(stream, rows * HIDDEN)?,
            quant12: DeviceBuffer::zeroed(stream, rows * 12288)?,
            map4: None,
            map12: None,
            scales: f(1)?,
            maxima: DeviceBuffer::zeroed(stream, rows)?,
            logits: DeviceBuffer::zeroed(stream, output_rows * VOCAB)?,
            next: DeviceBuffer::zeroed(stream, output_rows)?,
        };
        if rows >= 256 {
            workspace.map4 = Some(crate::tensor::map(
                stream,
                &workspace.quant4,
                rows,
                HIDDEN,
                64,
            )?);
            workspace.map12 = Some(crate::tensor::map(
                stream,
                &workspace.quant12,
                rows,
                12288,
                64,
            )?);
        }
        Ok(workspace)
    }
}

// Two host slots let the CPU hash step N while CUDA fills step N+1.
// Drop fences even a failed enqueue before pinned allocations can be released.
struct Readback {
    logits: [PinnedHostBuffer<f32>; 2],
    next: [PinnedHostBuffer<u32>; 2],
    stream: Arc<CudaStream>,
}
impl Readback {
    fn new(stream: &Arc<CudaStream>) -> Result<Self> {
        Ok(Self {
            logits: [
                PinnedHostBuffer::zeroed(stream.context(), VOCAB)?,
                PinnedHostBuffer::zeroed(stream.context(), VOCAB)?,
            ],
            next: [
                PinnedHostBuffer::zeroed(stream.context(), 1)?,
                PinnedHostBuffer::zeroed(stream.context(), 1)?,
            ],
            stream: stream.clone(),
        })
    }
    fn submit(&mut self, logits: &Buffer, next: &DeviceBuffer<u32>, slot: usize) -> Result<()> {
        // SAFETY: the serialized pool owns both slots. The caller only reads
        // this slot after synchronizing; its other slot is disjoint and idle.
        // Device outputs are not reused until these same-stream copies finish.
        unsafe {
            logits.copy_to_pinned_host_async(&self.stream, &mut self.logits[slot])?;
            next.copy_to_pinned_host_async(&self.stream, &mut self.next[slot])?;
        }
        Ok(())
    }
}
impl Drop for Readback {
    fn drop(&mut self) {
        self.stream.context().record_err(self.stream.synchronize());
    }
}

// Device addresses stay fixed for the model lifetime. A new request overwrites
// every visible cache entry before reading it; stale future entries are masked.
struct Execution {
    graphs: Vec<Graph>,
    caches: Vec<Cache>,
    decode: Workspace,
    control: DeviceBuffer<u32>, // token ID, absolute position
    prefill: Option<Workspace>,
    readback: Readback,
}
impl Execution {
    fn new(stream: &Arc<CudaStream>) -> Result<Self> {
        Ok(Self {
            graphs: Vec::new(),
            caches: (0..36)
                .map(|_| {
                    Ok(Cache {
                        k: DeviceBuffer::zeroed(stream, CONTEXT * 1024)?,
                        v: DeviceBuffer::zeroed(stream, CONTEXT * 1024)?,
                    })
                })
                .collect::<Result<_>>()?,
            decode: Workspace::new(stream, 1)?,
            control: DeviceBuffer::from_host(stream, &[0_u32, 0])?,
            prefill: None,
            readback: Readback::new(stream)?,
        })
    }
}

pub struct GpuModel {
    // Drop graphs before captured weights/module; Execution drops graphs before scratch.
    execution: Mutex<Execution>,
    stream: Arc<CudaStream>,
    module: device::LoadedModule,
    layers: Vec<Layer>,
    embedding: DeviceBuffer<u16>,
    head: PackedHead,
    final_gain: Buffer,
    rope: Buffer,
    /// `CLEAN_PREFETCH=0` disables every L2 hint, restoring HEAD launch shapes.
    prefetch: bool,
}
fn logit_hash(logits: &[f32]) -> String {
    sha(&logits
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect::<Vec<_>>())
}
fn cfg(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(128) as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    }
}
fn rows(n: usize, threads: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n as u32, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}
impl GpuModel {
    pub fn new(model: &Model) -> Result<Self> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.new_stream()?;
        let module = device::load(&ctx)?;
        let mut layers = Vec::new();
        for l in &model.layers {
            let gains = l
                .gains
                .iter()
                .map(|(s, x)| Ok((s.clone(), DeviceBuffer::from_host(&stream, x)?)))
                .collect::<Result<_>>()?;
            let weights = l
                .weights
                .iter()
                .map(|(s, w)| {
                    let weights = DeviceBuffer::from_host(&stream, &w.w)?;
                    Ok((
                        s.clone(),
                        Matrix {
                            w: weights,
                            map: None,
                            scale: DeviceBuffer::from_host(&stream, &w.scale)?,
                            k: w.k,
                            n: w.n,
                        },
                    ))
                })
                .collect::<Result<_>>()?;
            layers.push(Layer { gains, weights });
        }
        let mut gpu = Self {
            execution: Mutex::new(Execution::new(&stream)?),
            embedding: DeviceBuffer::from_host(&stream, &model.embedding)?,
            head: PackedHead::new(&stream, &model.head)?,
            final_gain: DeviceBuffer::from_host(&stream, &model.final_gain)?,
            rope: DeviceBuffer::from_host(&stream, &model.rope)?,
            prefetch: std::env::var("CLEAN_PREFETCH").map_or(true, |v| v != "0"),
            stream,
            module,
            layers,
        };
        for layer in &mut gpu.layers {
            for weight in layer.weights.values_mut() {
                if weight.n >= 4096 {
                    weight.map = Some(crate::tensor::map(
                        &gpu.stream,
                        &weight.w,
                        weight.n,
                        weight.k,
                        128,
                    )?);
                }
            }
        }
        {
            let mut execution = gpu
                .execution
                .lock()
                .map_err(|_| anyhow::anyhow!("execution mutex poisoned"))?;
            let Execution {
                graphs,
                caches,
                decode,
                control,
                ..
            } = &mut *execution;
            gpu.stream.synchronize()?;
            // Every decode kernel waits on its predecessor before touching data,
            // so graph edges may be programmatic. CLEAN_PDL=0 restores ordinary
            // full-completion edges for comparison.
            let programmatic = std::env::var("CLEAN_PDL").map_or(true, |v| v != "0");
            for tiles in GRAPH_TILES {
                // SAFETY: pool and model own every captured allocation; the mutex
                // serializes replays, and graph fields drop before these allocations.
                // Programmatic edges preserve that order: see kernels::dependency_wait.
                let graph = unsafe {
                    Graph::capture(&gpu.stream, programmatic, || {
                        gpu.forward(decode, caches, control, control, tiles, None, 0)
                    })?
                };
                graphs.push(graph);
            }
        }
        Ok(gpu)
    }
    fn norm(&self, x: &Buffer, gain: &Buffer, out: &mut Buffer) -> Result<()> {
        // SAFETY: fixed graph supplies equal input/output lengths and full norm rows.
        unsafe {
            self.module
                .norm(&self.stream, rows(x.len() / gain.len(), 256), x, gain, out)?;
        }
        Ok(())
    }
    fn norm_quantize(
        &self,
        x: &Buffer,
        gain: &Buffer,
        out: &mut Buffer,
        q: &mut DeviceBuffer<i8>,
        scale: &mut Buffer,
        maxima: &mut DeviceBuffer<u32>,
    ) -> Result<()> {
        // SAFETY: hidden rows have exactly 4096 coordinates; all outputs are distinct.
        unsafe {
            self.module.norm_quantize(
                &self.stream,
                rows(scale.len(), 1024),
                x,
                gain,
                out,
                q,
                scale,
                maxima,
            )?;
        }
        Ok(())
    }
    fn linear_residual(
        &self,
        x: &Buffer,
        w: &Matrix,
        q: &mut DeviceBuffer<i8>,
        scale: &mut Buffer,
        out: &mut Buffer,
        maxima: &DeviceBuffer<u32>,
        residual: &Buffer,
        combined: &mut Buffer,
        map: Option<&DeviceBuffer<u64>>,
        prefetch: bool,
        policy: ProjectionPolicy,
    ) -> Result<()> {
        if scale.len() == 1 && w.k == 12288 {
            // SAFETY: the preceding SwiGLU produced this exact completed maximum;
            // model shapes supply full rows and uniquely owned output coordinates.
            unsafe {
                self.module.quantize_with_max(
                    &self.stream,
                    cfg(x.len()),
                    x,
                    w.k as u32,
                    maxima,
                    q,
                    scale,
                )?;
            }
        } else {
            // Decode's O quantization finishes warming the leading O rows.
            let range = if prefetch {
                (O_ATTENTION, O_QUANTIZE)
            } else {
                (0, 0)
            };
            self.quantize(x, w.k, q, scale, (&w.w, range.0, range.1, prefetch))?;
        }
        if scale.len() == 1 {
            // SAFETY: the residual source and both complete output rows have
            // separate allocations; each full warp owns exactly one coordinate.
            unsafe {
                self.module.linear_residual(
                    &self.stream,
                    rows(out.len().div_ceil(8), 256),
                    q,
                    scale,
                    &w.w,
                    &w.scale,
                    w.k as u32,
                    residual,
                    out,
                    combined,
                    row_prefetch(prefetch, if w.k == 12288 { ROW_DOWN } else { ROW_O }),
                )?;
            }
        } else {
            self.linear_quantized(q, scale, w, out, map, policy)?;
            self.residual(residual, out, combined)?;
        }
        Ok(())
    }
    fn quantize(
        &self,
        x: &Buffer,
        width: usize,
        q: &mut DeviceBuffer<i8>,
        scale: &mut Buffer,
        prefetch: (&DeviceBuffer<i8>, u32, u32, bool),
    ) -> Result<()> {
        let (weights, from, to, extra) = prefetch;
        // SAFETY: the fixed graph supplies complete rows and matching scratch.
        // Extra blocks only issue in-bounds L2 hints for immutable weights.
        unsafe {
            self.module.quantize(
                &self.stream,
                grow(rows(x.len() / width, 1024), extra),
                x,
                width as u32,
                q,
                scale,
                weights,
                from,
                to,
            )?;
        }
        Ok(())
    }
    fn linear_quantized(
        &self,
        q: &DeviceBuffer<i8>,
        scale: &Buffer,
        w: &Matrix,
        out: &mut Buffer,
        map: Option<&DeviceBuffer<u64>>,
        policy: ProjectionPolicy,
    ) -> Result<()> {
        let nr = scale.len();
        // Cold-weight CUDA-event measurements on the 5090 favor wider tiles
        // for small MLP batches and DP4A for two to four down-projection rows.
        // Keep the original PRO 6000/legacy dispatch independently selectable.
        let tuned_tiled = policy == ProjectionPolicy::Rtx5090
            && ((nr >= 4 && w.n == 12288) || (nr >= 5 && w.k == 12288));
        let tuned_dp4a = policy == ProjectionPolicy::Rtx5090 && w.k == 12288 && nr <= 4;
        // SAFETY: fixed graph supplies exact dimensions and distinct buffers. Scratch
        // stays alive across all queued uses. Stream order protects quantizer reuse.
        unsafe {
            if nr >= 256 && w.n >= 4096 {
                self.module.linear_large(
                    &self.stream,
                    rows(nr.div_ceil(64) * w.n.div_ceil(128), 256),
                    map.expect("large prefill activation map"),
                    scale,
                    w.map.as_ref().expect("large prefill weight map"),
                    &w.scale,
                    w.k as u32,
                    out,
                )?;
            } else if nr >= 32 || tuned_tiled {
                self.module.linear_tiled(
                    &self.stream,
                    rows(nr.div_ceil(32) * w.n.div_ceil(64), 256),
                    q,
                    scale,
                    &w.w,
                    &w.scale,
                    w.k as u32,
                    out,
                )?;
            } else if nr > 1 && !tuned_dp4a {
                self.module.linear_splitk(
                    &self.stream,
                    rows(nr.div_ceil(16) * w.n.div_ceil(8), 256),
                    q,
                    scale,
                    &w.w,
                    &w.scale,
                    w.k as u32,
                    out,
                )?;
            } else {
                self.module.linear(
                    &self.stream,
                    rows(out.len().div_ceil(8), 256),
                    q,
                    scale,
                    &w.w,
                    &w.scale,
                    w.k as u32,
                    out,
                )?;
            }
        }
        Ok(())
    }
    fn residual(&self, a: &Buffer, b: &Buffer, out: &mut Buffer) -> Result<()> {
        // SAFETY: fixed graph supplies equal shapes and distinct buffers.
        unsafe {
            self.module
                .residual(&self.stream, cfg(out.len()), a, b, out)?;
        }
        Ok(())
    }
    fn trace(&self, path: Option<&Path>, step: usize, name: &str, x: &Buffer) -> Result<()> {
        if let Some(path) = path {
            let a = x.to_host_vec(&self.stream)?;
            fs::write(
                path.join(format!("step.{step}.{name}.bin")),
                a.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<_>>(),
            )?;
        }
        Ok(())
    }
    fn forward(
        &self,
        w: &mut Workspace,
        caches: &mut [Cache],
        input: &DeviceBuffer<u32>,
        position: &DeviceBuffer<u32>,
        tiles: usize,
        trace: Option<&Path>,
        step: usize,
    ) -> Result<()> {
        let policy = ProjectionPolicy::Original;
        let nr = w.hidden.len() / HIDDEN;
        // Only single-row decode streams weights once per token; prefill reuses them.
        let pf = nr == 1 && self.prefetch;
        // SAFETY: admitted vocabulary indices; the execution mutex owns all scratch.
        unsafe {
            self.module.embedding(
                &self.stream,
                cfg(w.hidden.len()),
                input,
                &self.embedding,
                &mut w.hidden,
            )?;
        }
        for (l, layer) in self.layers.iter().enumerate() {
            self.norm_quantize(
                &w.hidden,
                &layer.gains["norm1"],
                &mut w.norm,
                &mut w.quant4,
                &mut w.scales,
                &mut w.maxima,
            )?;
            if l == 4 {
                self.trace(trace, step, "debug.norm", &w.norm)?;
            }
            let cache = &mut caches[l];
            if nr == 1 {
                let q = &layer.weights["q"];
                let k = &layer.weights["k"];
                let v = &layer.weights["v"];
                // SAFETY: sibling matrices share the same complete activation row;
                // their weights/scales match each distinct output allocation.
                unsafe {
                    self.module.linear_triple(
                        &self.stream,
                        rows((q.n + k.n + v.n).div_ceil(8), 256),
                        &w.quant4,
                        &w.scales,
                        q.k as u32,
                        &q.w,
                        &q.scale,
                        &mut w.q,
                        &k.w,
                        &k.scale,
                        &mut w.k,
                        &v.w,
                        &v.scale,
                        &mut w.v,
                        &cache.k,
                        &cache.v,
                        position,
                        row_prefetch(pf, ROW_QKV),
                    )?;
                }
            } else {
                self.linear_quantized(
                    &w.quant4,
                    &w.scales,
                    &layer.weights["q"],
                    &mut w.q,
                    w.map4.as_ref(),
                    policy,
                )?;
                self.linear_quantized(
                    &w.quant4,
                    &w.scales,
                    &layer.weights["k"],
                    &mut w.k,
                    w.map4.as_ref(),
                    policy,
                )?;
                self.linear_quantized(
                    &w.quant4,
                    &w.scales,
                    &layer.weights["v"],
                    &mut w.v,
                    w.map4.as_ref(),
                    policy,
                )?;
            }
            if l == 4 {
                self.trace(trace, step, "debug.q", &w.q)?;
                self.trace(trace, step, "debug.k", &w.k)?;
                self.trace(trace, step, "debug.v", &w.v)?;
            }

            // SAFETY: disjoint Q/K/V and cache allocations; a 128-thread CTA owns
            // each of 32 Q or eight K heads per row, within admitted positions.
            unsafe {
                self.module.prepare_qk(
                    &self.stream,
                    rows(nr * 40, 128),
                    &mut w.q,
                    &mut w.k,
                    &w.v,
                    &layer.gains["qnorm"],
                    &layer.gains["knorm"],
                    &self.rope,
                    position,
                    &mut cache.k,
                    &mut cache.v,
                )?;
            }
            if l == 4 {
                self.trace(trace, step, "debug.qpost", &w.q)?;
                self.trace(trace, step, "debug.kpost", &w.k)?;
            }
            // SAFETY: the preceding preparation appends the entire visible
            // prefix before attention reads it on the same stream.
            unsafe {
                if nr >= 32 && tiles <= 8 {
                    self.module.attention_prefill(
                        &self.stream,
                        rows(nr * 8, 512),
                        &w.q,
                        &cache.k,
                        &cache.v,
                        position,
                        &mut w.attention,
                    )?;
                } else if tiles <= 3 {
                    self.module.attention(
                        &self.stream,
                        grow(rows(nr * 32, 128), pf),
                        &w.q,
                        &cache.k,
                        &cache.v,
                        position,
                        &mut w.attention,
                        &layer.weights["o"].w,
                        0,
                        if pf { O_ATTENTION } else { 0 },
                    )?;
                } else {
                    self.module.attention_scores(
                        &self.stream,
                        rows(nr * 32 * tiles, 128),
                        &w.q,
                        &cache.k,
                        position,
                        tiles as u32,
                        &mut w.scores,
                    )?;
                    self.module.attention_softmax(
                        &self.stream,
                        grow(rows(nr * 32, 128), pf),
                        &mut w.scores,
                        position,
                        &layer.weights["o"].w,
                        0,
                        if pf { O_ATTENTION } else { 0 },
                    )?;
                    self.module.attention_values(
                        &self.stream,
                        rows(nr * 32 * 4, 32),
                        &w.scores,
                        &cache.v,
                        position,
                        &mut w.attention,
                    )?;
                }
            }
            self.linear_residual(
                &w.attention,
                &layer.weights["o"],
                &mut w.quant4,
                &mut w.scales,
                &mut w.o,
                &w.maxima,
                &w.hidden,
                &mut w.residual,
                w.map4.as_ref(),
                pf,
                policy,
            )?;
            if l == 4 {
                self.trace(trace, step, "debug.a", &w.attention)?;
                self.trace(trace, step, "debug.o", &w.o)?;
            }
            self.norm_quantize(
                &w.residual,
                &layer.gains["norm2"],
                &mut w.norm,
                &mut w.quant4,
                &mut w.scales,
                &mut w.maxima,
            )?;
            if l == 4 {
                self.trace(trace, step, "debug.norm2", &w.norm)?;
            }
            if nr == 1 {
                let gate = &layer.weights["gate"];
                let up = &layer.weights["up"];
                // SAFETY: one complete activation row, eight full warps per CTA,
                // and distinct gate/up/activation outputs. Norm reset maxima earlier.
                unsafe {
                    self.module.linear_swiglu(
                        &self.stream,
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
                        row_prefetch(pf, ROW_MLP),
                    )?;
                }
            } else {
                self.linear_quantized(
                    &w.quant4,
                    &w.scales,
                    &layer.weights["gate"],
                    &mut w.gate,
                    w.map4.as_ref(),
                    policy,
                )?;
                self.linear_quantized(
                    &w.quant4,
                    &w.scales,
                    &layer.weights["up"],
                    &mut w.up,
                    w.map4.as_ref(),
                    policy,
                )?;
            }
            if l == 4 {
                self.trace(trace, step, "debug.gate", &w.gate)?;
                self.trace(trace, step, "debug.up", &w.up)?;
            }
            // SAFETY: disjoint buffers of the same complete MLP shape.
            if nr != 1 {
                unsafe {
                    self.module.swiglu(
                        &self.stream,
                        cfg(w.act.len()),
                        &w.gate,
                        &w.up,
                        &mut w.act,
                        &mut w.maxima,
                    )?;
                }
            }
            self.linear_residual(
                &w.act,
                &layer.weights["down"],
                &mut w.quant12,
                &mut w.scales,
                &mut w.down,
                &w.maxima,
                &w.residual,
                &mut w.hidden,
                w.map12.as_ref(),
                pf,
                policy,
            )?;
            if l == 4 {
                self.trace(trace, step, "debug.act", &w.act)?;
                self.trace(trace, step, "debug.down", &w.down)?;
            }
            self.trace(trace, step, &format!("layer.{l}"), &w.hidden)?;
        }
        self.norm(&w.hidden, &self.final_gain, &mut w.norm)?;
        // SAFETY: selects the last full hidden row and writes every vocabulary logit.
        unsafe {
            self.module.head_packed(
                &self.stream,
                cfg(VOCAB * 32),
                &w.norm,
                &self.head.words,
                &self.head.offsets,
                &self.head.exceptions,
                &mut w.logits,
                row_prefetch(pf, ROW_HEAD),
            )?;
            self.module
                .select_token(&self.stream, rows(1, 1024), &w.logits, &mut w.next)?;
        }
        self.trace(trace, step, "norm", &w.norm)?;
        self.trace(trace, step, "logits", &w.logits)?;
        Ok(())
    }
    pub fn generate(
        &self,
        model: &Model,
        request: &Request,
        trace: Option<&Path>,
    ) -> Result<Generation> {
        self.generate_inner(model, request, trace, true)
    }
    fn generate_inner(
        &self,
        model: &Model,
        request: &Request,
        trace: Option<&Path>,
        replay: bool,
    ) -> Result<Generation> {
        request.validate()?;
        if let Some(path) = trace {
            fs::create_dir_all(path)?;
            fs::write(
                path.join("request.json"),
                serde_json::to_vec_pretty(request)?,
            )?;
        }
        let mut execution = self
            .execution
            .lock()
            .map_err(|_| anyhow::anyhow!("execution mutex poisoned"))?;
        // Also complete any previously failed eager request before reusing scratch.
        self.stream.synchronize()?;
        if execution
            .prefill
            .as_ref()
            .is_none_or(|w| w.hidden.len() != request.prompt_ids.len() * HIDDEN)
        {
            execution.prefill = Some(Workspace::new(&self.stream, request.prompt_ids.len())?);
        }
        let input = DeviceBuffer::from_host(&self.stream, &request.prompt_ids)?;
        let Execution {
            graphs,
            caches,
            decode,
            control,
            prefill,
            readback,
        } = &mut *execution;
        let prefill = prefill.as_mut().expect("prefill allocated above");
        self.stream.synchronize()?;
        let mut ids = request.prompt_ids.clone();
        let mut position = 0;
        let mut generated = Vec::new();
        let mut timings = Vec::new();
        let mut previous_logits: Option<usize> = None;
        let mut hashes = Vec::new();
        let mut finish = "length";
        let request_start = Instant::now();
        for step in 0..request.max_tokens {
            let start = Instant::now();
            let nr = ids.len();
            control.copy_from_host(
                &self.stream,
                &[if step == 0 { 0 } else { ids[0] }, position as u32],
            )?;
            let w = if step == 0 {
                &mut *prefill
            } else {
                &mut *decode
            };
            let tiles = (position + nr).div_ceil(64);
            if step > 0 && trace.is_none() && replay {
                // SAFETY: execution mutex owns all captured buffers; control copy
                // has completed, and the following logit read completes this replay.
                unsafe {
                    graphs[GRAPH_TILES.partition_point(|&bound| bound < tiles)].launch()?;
                }
            } else {
                self.forward(
                    w,
                    caches,
                    if step == 0 { &input } else { control },
                    control,
                    tiles,
                    trace,
                    step,
                )?;
            }
            let slot = step % 2;
            readback.submit(&w.logits, &w.next, slot)?;
            // The GPU is now computing the next step. Digest the previous
            // completed logit vector while that work is in flight.
            if let Some(logits) = previous_logits.take() {
                hashes.push(logit_hash(readback.logits[logits].as_slice()));
            }
            self.stream.synchronize()?;
            let next = readback.next[slot][0];
            anyhow::ensure!(next != u32::MAX, "nonfinite or empty logits");
            timings.push(start.elapsed().as_secs_f64());
            previous_logits = Some(slot);
            generated.push(next);
            if request.stop_ids.contains(&next) {
                finish = "stop";
                break;
            }
            position += nr;
            ids = vec![next];
        }
        let request_seconds = request_start.elapsed().as_secs_f64();
        if let Some(logits) = previous_logits {
            hashes.push(logit_hash(readback.logits[logits].as_slice()));
        }
        let text = model
            .tokenizer
            .decode(&generated, true)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(Generation {
            token_ids: generated,
            text,
            finish_reason: finish.into(),
            step_seconds: timings,
            request_seconds,
            logit_sha256: hashes,
        })
    }
}

#[cfg(test)]
mod replay_tests {
    use super::*;
    use anyhow::Context;
    #[test]
    fn reused_graphs_match_eager() -> Result<()> {
        let bundle = std::env::var_os("CLEAN_MODEL_BUNDLE")
            .context("set CLEAN_MODEL_BUNDLE to run the full-model eager/graph replay test")?;
        let model = Model::load(Path::new(&bundle))?;
        let gpu = GpuModel::new(&model)?;
        // Resize prefill scratch, cross graph/attention dispatch boundaries,
        // and overwrite a long request's cache with a short unrelated prefix.
        for (run, n) in [
            5, 5, 129, 1, 63, 64, 191, 192, 193, 255, 256, 511, 512, 1023, 1024, 2047, 2048, 4093,
            5,
        ]
        .into_iter()
        .enumerate()
        {
            let request = Request {
                prompt_ids: vec![42 + run as u32; n],
                max_tokens: 3,
                stop_ids: vec![],
            };
            let eager = gpu.generate_inner(&model, &request, None, false)?;
            let replay = gpu.generate(&model, &request, None)?;
            assert_eq!(replay.logit_sha256, eager.logit_sha256, "prompt length {n}");
            assert_eq!(replay.token_ids, eager.token_ids, "prompt length {n}");
            assert_eq!(replay.text, eager.text);
            assert_eq!(replay.finish_reason, eager.finish_reason);
            let stop = Request {
                stop_ids: vec![eager.token_ids[0]],
                ..request
            };
            let stopped = gpu.generate(&model, &stop, None)?;
            assert_eq!(stopped.token_ids.len(), 1);
            assert_eq!(stopped.finish_reason, "stop");
            assert_eq!(stopped.logit_sha256, eager.logit_sha256[..1]);
        }
        Ok(())
    }
}
