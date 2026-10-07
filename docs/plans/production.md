# Clean W8A8 inference: production plan

Prepared 2026-10-06 after optimization round 6. Scope: `crates/inferrous-qwen3`, the independent Rust/cuda-oxide Qwen3-8B W8A8 engine.

This document lists everything required to turn the engine into a production service: decisions to make first, the workstreams, acceptance criteria, milestones and capacity estimates. Estimates are marked as such; none is a measurement.

## 1. Where the engine stands

| Area | Today | Production gap |
|---|---|---|
| Model | Qwen3-8B revision `b968826d`, fixed shapes compiled in (`HIDDEN`, `VOCAB`, `CONTEXT`). Per-channel INT8 weights, per-token dynamic INT8 activations, BF16 residual/KV/head. | Quality is unvalidated (§3, W1). Shapes are constants. |
| Requests | One at a time behind an execution mutex. Raw token IDs, greedy only, stop token IDs, prompt plus output at most 4096. | Concurrency, sampling, stop strings, chat template, streaming. |
| KV cache | One fixed 4096-token K/V buffer per layer (604 MB total), overwritten by each request. | Many sequences, paging, admission, preemption, prefix reuse. |
| Decode | Eight persistent CUDA graphs (64–4096 visible tokens), programmatic edges, L2 prefetch. 182.3 / 178.2 / 170.5 tokens/s at 5 / 129 / 512 prompt tokens. | Batched decode; graph buckets by batch size. |
| Prefill | Eager launches; 44.1 ms first step for 512 tokens versus about 22 ms for stock vLLM. | Prefill throughput matters once batching exists. |
| Outputs | Every step copies all 151,936 logits to pinned host memory and SHA-256 hashes them. | Keep as a verification mode; it cannot scale to batches. |
| Correctness | Bit-exact regression against the original Rust oracle; independent operator fixtures; sanitizers; graph/eager replay. | The full-model Torch gate fails (DISC-001). No perplexity or eval data. |
| Interface | CLI binary printing JSON per request. | Network API, auth, metrics, health, lifecycle. |
| Build | nightly-2026-08-28 plus the cuda-oxide codegen backend from NVIDIA cuda-rust; tests run only on the development workstation. | Reproducible toolchain, container, GPU CI. |

Hardware assumed throughout: one RTX PRO 6000 Blackwell Max-Q (SM120, 97,887 MiB, GDDR7 at about 1.7 TB/s peak and about 1.6 TB/s sustained, 300 W) per replica.

## 2. Decisions required before implementation

These choices change the design of nearly every workstream. Each has a recommendation, but they are product decisions.

### D0. Build versus adopt

vLLM or SGLang already provide paging, continuous batching, prefix caching, sampling and an OpenAI-compatible API, and they run this exact INT8 checkpoint today (via Humming kernels on SM120). Building our own engine is justified only by properties those engines lack:

- Determinism and batch invariance (D1), which matter if outputs must be reproducible, attestable or verifiable.
- A Rust/cuda-oxide stack the team controls end to end, as a substrate for further research such as verifiable or private inference.
- Single-request performance: decode is now 26% / 24% / 19% faster than the best stock configuration tested.

**Recommendation:** proceed only if D1's batch invariance or a research roadmap that needs a controllable engine is a real requirement. Otherwise, serve production traffic with vLLM and keep this crate as a research and verification engine.

### D1. Numerical contract

Today every change must reproduce the round-0 Rust baseline's logit hashes bit for bit. That contract does not scale:

- The ordered softmax and PV accumulation is a serial chain over every key. At 32K keys that is roughly 32K dependent FP32 additions per output coordinate, on the order of 50 µs per layer and 2 ms per token across 36 layers (estimate).
- Tensor-core BF16 head GEMMs, split reductions and new fused kernels all reorder FP32 sums.

Options:

1. **Keep bit-exactness to the current oracle.** It caps context near today's limit and blocks many kernel designs.
2. **Deterministic and batch-invariant, with one re-baseline (recommended).** Every FP reduction gets a fixed tree that does not depend on batch composition, scheduling, chunking or which SMs run the work. Generate a new oracle once, justify it against Torch and evals (W1), and from then on require exact hashes against it.
3. **Tolerance-based only,** like stock engines. Simplest, but it gives up the engine's main differentiator.

Option 2 is cheap here because INT8 products accumulate exactly in INT32: every projection is batch-invariant under any tiling or split-K. The remaining FP reductions are per-row RMSNorm, per-sequence attention, the per-row BF16 head dot and selection. Each can be written with a fixed per-row order.

Batch invariance also yields the strongest load test available: every response produced under concurrent load must equal its solo replay, hash for hash (§6).

### D2. Target workload and service-level objectives

Required inputs:

- Interactive chat versus offline batch throughput.
- Peak concurrent sequences per GPU.
- Context length: P50/P99 prompt length and maximum context. Qwen3-8B is 32,768 natively and 131,072 with YaRN; we stop at 4096 today.
- Targets: time to first token, inter-token latency P50/P99, aggregate tokens/s per GPU, and the availability target.

These set the batch-size buckets, the KV budget, whether long-context attention (W5) is in scope, and how much prefill work (W11) is needed.

### D3. Model scope

Qwen3-8B only, or config-driven shapes covering the Qwen3 dense family and later models? Today's fixed-shape kernels and the unchecked-indexing safety argument both depend on compiled-in dimensions. Generalizing means shape-parameterized kernels and a new SAFETY argument per kernel.

**Recommendation:** ship Qwen3-8B first and keep the shapes fixed. Refactor shapes into one model descriptor so a second model is a contained change.

**Update (2026-10-06):** the second model will be Qwen3.6-35B-A3B FP8. See the [implementation plan](qwen3.6-35b-a3b-fp8.md) and the [candidate rankings](../research/single-gpu-model-candidates.md).

### D4. Quality bar and quantization recipe

What counts as acceptable degradation against BF16 (perplexity delta, eval deltas)? If the plain per-token/per-channel recipe misses the bar, choose between calibration (SmoothQuant-style activation smoothing, AWQ/GPTQ weight scales), mixed precision for sensitive layers, or a different checkpoint. Each changes the bundle format and the numerical contract.

### D5. API surface

**Recommendation:** OpenAI-compatible `/v1/chat/completions` and `/v1/completions` with streaming over server-sent events, plus `/health`, `/ready` and `/metrics`. Decide whether to expose logprobs, per-request seeds and a determinism or verification mode that returns output hashes.

### D6. Verifiability and privacy linkage

If outputs must be attestable or independently verifiable, define what is attested now: a per-response hash chain over tokens, optionally logits, plus bundle root, profile and seed. That determines which hashing survives into the production path and whether "verification mode" is a product feature.

### D7. Deployment target

Which machines (a single workstation, a fleet of RTX PRO 6000 hosts, cloud), how traffic reaches replicas (a load balancer or an existing service), and who operates it. Replicas, not tensor parallelism, are the scaling unit for an 8B model.

## 3. Workstreams

Each workstream lists what to build, acceptance criteria, and risks. Kernel work keeps the existing discipline: SAFETY argument, operator fixtures, checked-build parity, sanitizers, A/B/B/A timing, and committed evidence.

### W1. Quality gate

- Resolve or bound DISC-001: replay layer-4 inputs through independent operators and quantify amplification across varied prompts and long outputs.
- Measure perplexity (WikiText-2, plus a representative chat corpus) and a small eval suite against BF16 Qwen3-8B and against vLLM W8A8 on the same exported weights. Suggested suite: MMLU subset, GSM8K, HumanEval, IFEval, plus long-context retrieval if W5 is in scope.
- If D4's bar is missed, prototype calibrated or mixed-precision variants in Torch first; port only the winner.

Accept when evals are within D4's bar, results and harness are committed, and the Torch trace gate is either passing or replaced by a justified statistical gate.

Risk: the plain recipe may be too lossy for Qwen3, forcing a new bundle and kernels before any serving work pays off.

### W2. Numerical contract and re-baseline

Only if D1 picks option 2.

- Specify, per kernel, the reduction tree for norms, attention (fixed-size key blocks with a fixed combine tree), the head dot and selection. It must not depend on batch size, chunking or launch geometry.
- Build a CPU or Torch reference that implements the same trees for operator fixtures.
- Generate the new oracle once, with provenance, and document the reason for replacing the old one, including W1's evidence that quality did not regress.
- Add batch-invariance tests: the same request alone, in batches of 2 to 64 with random companions, chunked at random boundaries, and preempted and resumed. All must give identical hashes.

### W3. Paged KV cache

- Fixed-size blocks (16 or 32 tokens) per layer, holding K (coordinate-major within a block) and V (token-major), each sequence mapped by a block table on the device.
- An allocator with a global budget sized from free memory after weights, graphs and scratch. At 144 KiB per token, an 80 GiB budget holds about 580,000 tokens (Appendix A).
- Admission control: admit only when the worst-case remaining blocks fit, or admit optimistically and support preemption.
- Preemption by recompute (free the blocks, requeue, re-prefill later) is simpler than swapping to host and stays exact under D1 option 2.
- Preserve today's safety invariant: every visible KV entry is written before it is read, and freed blocks are never visible to another sequence. Validate block tables on the host. Unchecked device indexing needs a new SAFETY argument covering block-table indices, or checked indexing for those loads.

Accept when stress tests allocate, free and preempt thousands of sequences with no leak or double use, sanitizers are clean, and hashes match solo replays.

### W4. Batched execution engine

- **Scheduler.** Continuous, iteration-level batching: each step admits new sequences, mixes chunked prefill with decode under a token budget, and retires finished sequences. Policy covers FCFS with priority classes, a starvation bound and maximum batch tokens.
- **Projections.** At small batch, extend the DP4A warp-per-row kernels to reuse each weight load across B activation rows. Larger batches route to the exact INT8 tensor-core kernels the prefill path already has (split-K, tiled, TMA). Integer accumulation keeps all of them batch-invariant. Benchmark the crossover.
- **Activation quantization and norms.** Already per row; batch them as rows.
- **Attention.** Ragged, paged decode attention over many sequences, plus chunked-prefill attention against cached prefixes. The reduction order must be fixed per sequence (W2).
- **Head.** At batch B the head becomes a GEMM. A CUDA-core FP32 kernel keeps the per-row order; a tensor-core BF16 MMA head is faster but needs the W2 re-baseline.
- **Sampling** runs on the GPU (W6). Only token IDs and optional top-k logprobs come back to the host, not 608 KB of logits per sequence.
- **CUDA graphs** per batch-size bucket (for example 1, 2, 4, 8, 16, 32, 64), with sequence lengths and block tables in device buffers so one graph covers every context length. Keep the programmatic-edge graph rewrite, and retune L2 prefetch: at larger batches weight reads amortize and KV traffic dominates.
- **Host pipeline.** Enqueue step N+1 before reading step N back, using a device-resident token feed and a pinned readback ring (round 6's top remaining single-stream item, about 40 µs per step). Clamp any invalid token for already queued work, as round 5's rejected four-step design did.

Accept on throughput and latency curves versus batch size and context against Appendix A estimates and against vLLM on the same weights, with batch-invariance tests passing.

### W5. Long context (if D2 needs more than 4096 tokens)

- Raise the context cap with YaRN rope scaling as Qwen3 specifies; build RoPE tables on demand or per bucket instead of a 4096-row table.
- Replace shared-memory score arrays sized to 4096 with blocked attention whose FP reductions use W2's fixed trees, with scores in global scratch where needed.
- Revisit graph and scratch sizing, and memory admission for long prompts.
- Evaluate quality at long context (needle-in-haystack, RULER-style).

### W6. Sampling and generation controls

- Temperature, top-k, top-p, min-p, repetition, presence and frequency penalties, logit bias, and a maximum token count.
- Seeded counter-based randomness (for example Philox keyed by seed, request and step) so sampled outputs are reproducible and batch-invariant.
- Stop strings matched on incrementally detokenized text, alongside stop IDs and EOS handling.
- Optional logprobs (chosen token plus top-k).
- Defaults from the Qwen3 model card: thinking temperature 0.6, top-p 0.95, top-k 20; non-thinking 0.7 / 0.8 / 20. Confirm against the card.
- Structured or grammar-constrained output is deferred (§5).

### W7. Prefix caching

- Content-addressed KV blocks: a chained hash of block tokens and position, with LRU eviction of unreferenced blocks and reference counting for shared ones.
- This gives large time-to-first-token wins for shared system prompts and multi-turn chat.
- Tenancy: shared prefixes leak timing across tenants. Namespace the cache per tenant, or disable sharing for sensitive deployments.
- Under D1 option 2, reused-prefix results must equal recomputed results; add that as a test.

### W8. Serving layer

- An async HTTP server (for example axum) that owns the engine thread. Bounded queues with backpressure, per-request cancellation (client disconnect frees KV) and timeouts.
- An OpenAI-compatible request and response schema with server-sent-event streaming and usage accounting.
- The Qwen3 chat template, including the thinking/non-thinking switch and tool call formatting if needed. Render with a Jinja-compatible engine or hand-written code, tested against Hugging Face `apply_chat_template` output.
- Incremental detokenization that is UTF-8 safe across token boundaries.
- Limits on prompt and output length, request size and concurrency per key.
- Graceful drain on shutdown, and readiness only after model load and graph capture.

### W9. Reliability and error handling

- Sticky CUDA errors (illegal address, ECC, Xid) poison the context. The process must exit and a supervisor restart it. Fail in-flight requests with a retryable error and surface the cause.
- Startup budget: about 8 s to load and pack today, plus capture time for more graphs. Store the packed head in the bundle to cut host packing; measure cold start.
- A malformed-bundle suite (an open item today) plus fuzzing of request admission and the API schema.
- Investigate the pre-existing memcheck nondeterminism on the 512-token TMA prefill path ([round 6 record](../qwen3-8b-w8a8/optimization/round-6.md)) before relying on that kernel under load.
- Soak tests: hours of randomized mixed traffic with batch-invariance checks and memory-leak monitoring (§6).
- Watch thermals and clocks: the 300 W Max-Q card drifts about 1% across sequential runs. SLOs must hold under sustained heat.

### W10. Observability

- Prometheus metrics:
  - queue depth, running and waiting sequences
  - batch size and tokens per step
  - time to first token, inter-token latency and end-to-end latency histograms
  - prefill and decode tokens/s
  - KV blocks used, free, preemptions and prefix-cache hit rate
  - GPU memory, SM clock, temperature and power from NVML
  - error counters by class
- Structured logs with request IDs. Do not log prompt or output content by default.
- Optional OpenTelemetry traces per request phase.
- A debug and verification mode that records per-step logit hashes (today's behavior) for selected requests.

### W11. Performance carried over

Ordered by expected value once batching lands:

1. **Prefill.** Close the 2x long-prompt gap to stock (TMA pipelining, larger tiles, attention for long prefixes); this matters more under chunked prefill.
2. **Host round trip.** Overlap it (about 40 µs per step; part of W4).
3. **Selection.** Parallel greedy selection, or fusion into sampling (about 18 µs per step).
4. **Small-kernel fusions** (O quantization into the O prologue, Q/K preparation into attention), revalidated under batching.
5. **KV quantization** (FP8 or INT8 per head and block). This halves KV traffic, the dominant cost at high batch or long context, but it is a D1 contract change.
6. **Tighter head coding** (about 0.7% at batch 1; less as batches amortize the weights).
7. **Adopt cuda-rust's PDL support.** Upstream now provides `griddepcontrol` intrinsics (`grid_dependency::wait` and `trigger_dependents`) and a `cuda-core` launch helper with programmatic stream serialization; the generated SIMT launch methods do not expose it yet. Use them in place of the inline PTX, and extend the launch macros so the graph rewrite becomes unnecessary.

### W12. Build, packaging, CI and deployment

- Build the cuda-oxide codegen backend from `vendor/cuda-rust` inside the build, never from an out-of-tree checkout, and pin the toolchain and CUDA versions (12.8 for builds, 13.1 for sanitizer).
- A container image with the CUDA runtime matched to the minimum driver (580.x tested). Run `nvidia-smi` and device capability checks at startup.
- GPU CI: a dedicated SM120 host registered as a CI runner. Every pull request runs both build variants, operator fixtures, the graph/eager replay and the hash oracle. Nightly runs add sanitizers, A/B performance against the previous release, and the eval subset.
- A model bundle pipeline: `tools/prepare.py` becomes a versioned, reproducible export with manifest hashes, provenance (source revision, tool versions) and optional signing. The service refuses bundles whose root hash is not on an allowlist.
- Releases tag source, bundle root, executable hash and oracle version together.

### W13. Security and tenancy

- Authentication (API keys or mTLS), per-key rate and token quotas, and request size limits.
- No prompt or output persistence by default; a data-retention policy.
- KV isolation per W3. Prefix-cache tenancy per W7.
- Dependency and licence review: Qwen3 is Apache-2.0 and the tokenizer crate is pinned.

## 4. Milestones

| Milestone | Contents | Exit criteria |
|---|---|---|
| **M0: Decide and measure** | D0–D7 answered; W1 eval harness and first results; W2 contract specified if chosen. | Written decisions; quality numbers against BF16 and vLLM. |
| **M1: Batched engine core** | W3 paged KV, W4 scheduler and batched kernels, W6 GPU sampling, W2 re-baseline. An in-process Rust API only. | Batch-invariance and soak tests pass. Throughput curves meet Appendix A estimates within an agreed margin; hashes match the new oracle. |
| **M2: Serving MVP** | W8 API and streaming with chat template, W10 metrics, W9 crash/restart supervision, W12 container and GPU CI. | Internal users on one replica; latency and throughput SLOs measured under load; runbook written. |
| **M3: Production hardening** | W7 prefix caching, W5 long context if required, W11 prefill work, W13 auth and quotas, malformed-bundle and fuzz suites, multi-replica deployment. | SLOs held through a multi-day soak; on-call ready; release process exercised. |
| **Later** | Speculative decoding with exact batched verification, KV quantization, structured output, additional models, verifiable and private inference. | Each with its own design and evidence. |

M1 is the largest step and the one that justifies D0: if batch-invariant batched decode cannot land near the Appendix A estimates, revisit adoption of a stock engine.

## 5. Explicitly deferred

- Speculative decoding. Batched, batch-invariant verification makes exact speculation straightforward later; it still needs a draft source and acceptance-rate data from real traffic.
- Tensor or pipeline parallelism; replicas suffice for 8B.
- Grammar-constrained or JSON-schema decoding.
- LoRA adapters and multi-model serving.
- Lower-precision weights (FP8, INT4, NVFP4), pruning and alternative arithmetic. These are separate numerical contracts.

## 6. Validation strategy for production

- **Oracles.** The current hash oracle until W2, then the re-baselined oracle. Independent operator fixtures for every new kernel, and the eval suite (W1) per release.
- **Batch invariance.** Run randomized concurrent traffic, record every response's hashes, replay each response alone, and require exact equality. Run per merge request (short) and nightly (long).
- **Differential testing against vLLM** on the same weights: token agreement rate and logit-distribution distance as monitored metrics, not gates.
- **Load and soak.** Open-loop arrival processes at target and 2x target rates, with mixed prompt and output lengths from D2, for hours. Track SLOs, memory, preemptions and errors.
- **Fault injection.** Client disconnects mid-stream, KV exhaustion, forced CUDA errors via a test hook, and slow consumers.
- **Sanitizers** on operator tests and representative full-model requests per release; racecheck on new batched kernels.

## Appendix A. Capacity and throughput estimates

These are estimates from a bandwidth-only model, not measurements. They assume about 1.6 TB/s sustained, ignore attention compute and launch overhead, and assume INT8 tensor-core compute stays below memory time at these batch sizes; verify that in M1.

Memory:

| Item | Size |
|---|---|
| INT8 body weights plus scales | about 6.95 GB |
| Packed BF16 head (12-bit codes plus exceptions) | about 0.93 GB |
| BF16 embedding | 1.24 GB |
| KV per token: 36 layers × 8 KV heads × 128 × (K + V) × 2 bytes | 147,456 B (144 KiB) |
| Today's sampled process total, including a 4096-token KV and prefill scratch | 10,256 MiB |
| KV capacity with an 80 GiB budget | about 580,000 tokens |

Per decode step, bytes read ≈ 7.9 GB of weights + Σ context × 144 KiB:

| Batch | Average context | KV read per step | Step time | Aggregate tokens/s | Per sequence |
|---:|---:|---:|---:|---:|---:|
| 1 | 512 | 0.08 GB | ≈5.0 ms (measured 5.87 ms) | ≈200 (measured 170.5) | ≈200 |
| 16 | 2,048 | 4.8 GB | ≈7.9 ms | ≈2,000 | ≈125 |
| 64 | 2,048 | 19.3 GB | ≈17 ms | ≈3,800 | ≈60 |
| 32 | 8,192 | 38.7 GB | ≈29 ms | ≈1,100 | ≈34 |

At batch 1 the measured step is about 17% slower than the model, because latency-bound kernels and launch overhead are not modeled. Expect a similar fixed overhead per step at larger batches. At larger batches and longer contexts KV reads dominate, which makes FP8 or INT8 KV the next large lever (W11.5). Those rows should become measured curves in M1.

## Appendix B. Open questions for the product owner

1. Is batch-invariant determinism (D1 option 2) a requirement, a nice-to-have, or unnecessary?
2. What are the target concurrency, context lengths and latency SLOs (D2)?
3. Is Qwen3-8B the production model, and what is the quality bar against BF16 (D3, D4)?
4. Must responses be attestable, and if so with what payload (D6)?
5. Where will it run, and who operates it (D7)?
6. If the answer to 1 is "unnecessary", should production traffic go to vLLM while this engine continues as research infrastructure (D0)?
