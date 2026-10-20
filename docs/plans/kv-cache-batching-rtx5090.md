# KV cache, batching and RTX 5090 performance work

Started 2026-10-20 on `feat/kv-cache-batching-rtx5090`, from the validated local setup commit `491f1c2`. The original release executable and measurements are preserved under `clean-work/kv-batch-baseline/`. This document tracks implementation and evidence; unchecked items remain requirements, not claims of completion.

## Requested outcome

Implement reusable KV caching, single-stream and batched execution, a real no-KV-cache batch mode, longer context windows, and clean Rust resource management. Optimize on the RTX 5090, compare against the fastest eligible off-the-shelf harness in each scenario, and keep RTX PRO 6000 and RTX 5090 settings and evidence separate. Performance wins require measurements; a missing comparison, unsupported stock configuration or regression cannot be reported as a win.

## Semantic and measurement contract

- Preserve Qwen3-8B's pinned W8A8 weights, BF16 KV and output head, greedy decoding, and no EOS stopping in timing runs. Record model roots and every engine version. Faster alternative precisions may be screened separately but cannot replace the same-weight comparisons.
- Distinguish within-request autoregressive KV reuse, reusable session/prefix state across requests, and recomputation with no persistent KV. A stock engine with prefix caching disabled still uses decode KV and is not a no-KV baseline.
- Retain the existing short-context numerical oracle and checked-build comparisons. New attention arithmetic requires independent reference evidence and an explicit numerical contract; never silently replace regression hashes to make a test pass. Batched results must be invariant to unrelated batch members, scheduling and cache reuse under the declared arithmetic path.
- Cover cached single-stream, cached batching, and true uncached batching at prompt lengths 5, 129, 512, 2K, 4K, 8K, 16K and 32K; batches 1, 2, 4, 8, 16 and 32 where memory admission permits; short and sustained outputs; homogeneous and ragged prompts; cold and warm prefixes; growth, stop and cancellation/release boundaries. Probe unsupported capacity explicitly and report it. Extend the matrix when profiles reveal a missing regime.
- Report end-to-end latency, first-token latency, per-stream decode latency, aggregate throughput, memory capacity and cache hit/miss behavior. Keep warmup, loading, preprocessing, verification overhead and prefix-hit timing boundaries explicit and comparable. Measure GPU jobs serially; use paired A/B/B/A for retained tuning decisions and report desktop/clock drift.
- Native context target is 32,768 tokens per the [Qwen model card](https://huggingface.co/Qwen/Qwen3-8B). The pinned config's `max_position_embeddings` is 40,960; this does not by itself validate quality beyond the documented native context. YaRN beyond native context requires a distinct model configuration and quality evidence.

## Work and acceptance evidence

| Workstream | Required evidence | Status |
|---|---|---|
| Branch and reproducible baseline | Branch, baseline commit, saved executable/hash and raw measurements | Done: `491f1c2`, `clean-work/kv-batch-baseline/` |
| Rust cache ownership/admission | Bounded paged allocation, transactional reservation, valid-token tracking, release/reuse and randomized invariant tests | Implemented and integrated; host ownership tests and full-model release/checked admission, reuse and reclamation tests pass |
| Reusable sessions/prefixes | Cache hits avoid recomputation; misses, partial pages, reset and stale state are safe; outputs match recomputation | Implemented; short-context exact full-model tests pass in release and checked builds |
| Batched model execution | Shared-weight row batches, ragged sequence positions and attention, batched head/selection, bounded scratch | Implemented with bounded prefill and capacity waves; short-context exact model tests pass |
| True no-cache execution | Full prefix recomputed each output step; same prompt/output contract; valid batched baseline | Implemented; processed-row counts and full-logit hashes validated; stock counterpart pending |
| Longer context | Bounded/chunked prefill, attention through 32K, configurable RoPE/cache capacity, correct memory admission | Implemented; ragged full-model 8K/16K/near-32K cold/warm/recompute/chunk tests pass in release and checked release |
| Hardware profiles | Explicit legacy PRO 6000 profile, independent 5090 tuning and metadata; no unmeasured cross-GPU claims | Separate capacity configurations, actual-device validation and metadata implemented; performance tuning remains active |
| Off-the-shelf comparison | Reproducible current vLLM/SGLang screening, eligible winner per scenario, true no-cache reference | vLLM 0.32.0 container pulled; PyTorch 2.13.0+cu130 executes on the RTX 5090 through explicit device/driver mounts; timing pending |
| GPU correctness | Operator fixtures, solo/batched/cache/no-cache comparisons, checked builds, sanitizer coverage, long-context checks | Paged attention operator tests added: ragged/shuffled pages are bit-exact with contiguous attention; a 32K uniform-attention reference passes. Model-level batched/long-context checks remain pending |
| Optimization | Profiles, retained/rejected experiments, paired measurements across all scenarios | Paged PV staging retained after exact-output paired 8K experiment; decode graphs validated; full matrix pending |
| Final report and completion audit | Requirement-by-requirement evidence, complete comparison tables, remaining limitations | Pending |

## Initial findings

The existing engine caches K/V only inside one request, with a single 4,096-token buffer per layer and a global execution mutex. A new request overwrites all visible state. Decode is graph-captured but batching is absent. Prefill allocates `rows × 32 × 4096` FP32 score scratch even when a fused short attention path is used; merely raising the context constant would amplify memory use and is not a sufficient long-context implementation. Existing INT8 multirow projection kernels can be reused for shared-weight batching, while sequence mapping, cache access, output heads and scheduling need new paths.

Current stock versions were checked against upstream releases: [vLLM 0.32.0](https://github.com/vllm-project/vllm/releases/tag/v0.32.0) and [SGLang 0.5.22](https://github.com/sgl-project/sglang/releases/tag/v0.5.22). [Humming](https://github.com/vllm-project/humming) is a candidate INT8 backend. The earlier workstation result is not a substitute for testing these on the RTX 5090. Docker is available locally, but no NVIDIA runtime/CDI configuration is installed; GPU access for stock containers will be validated before timing.

## Implementation checkpoint: paged operator foundation

`cache.rs` now provides bounded page ownership, atomic reservation failure, explicit initialized-token lengths, full-page immutable prefix forks, and RAII reclamation. `kernels.rs` adds 64-token paged Q/K normalization and RoPE, ragged attention score/softmax/value kernels, indexed output-head rows and per-row token selection. Only the attention kernels have dedicated GPU validation so far; model integration and the remaining new kernels are not yet verified. `Workspace::with_shape` permits independent token-row, output-row and score-stride sizing while preserving the legacy constructor. All 15 GPU-enabled tests pass in release (28.75 seconds) and checked release (30.14 seconds), with logs in `clean-work/kv-batch-baseline/paged-tests-attempt2.log` and `paged-tests-checked.log`. This includes the original full-model eager/graph checks; it does not yet prove full-model 32K or batch execution.

Next, integrate a batched/session runtime using these interfaces: per-row metadata `[sequence_slot, absolute_position]`, a row-major `[sequence_slot, logical_page]` device page table, and page-private K/V writes. K is coordinate-major inside a page; V is token-major. The runtime must validate page ownership, publish initialized lengths only after successful completion, and fence all queued users before a lease is reset or released. Use bounded prefill chunks and score-stride buckets, rather than materializing a full prompt-by-native-context scratch matrix. Reuse existing exact INT8 multirow projections; use output-row indices to compute heads only for sequences ready to sample. Add true recomputation mode, session reuse, prefix lookup and batch admission/scheduling, then graph capture and tuning. The CLI still uses the original single-request path at this checkpoint.

## Stock screening checkpoint

The unmodified `vllm/vllm-openai:v0.32.0` image is pinned by digest `sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34` in `tools/stock_container.sh`. Explicit GPU devices and the host driver library directory work without installing a system NVIDIA container runtime. Rootless Docker must use container UID 0, which maps to the invoking host user; the first attempt using the host numeric UID failed on private checkpoint files and was retained in the logs. The wrapper detects this and confines writable binds to `clean-work/`.

The first successful screen uses Humming INT8, FLASH_ATTN, native decode graphs, 4K maximum context, one sequence, no prefix cache, memory fraction 0.70, and three warmups plus twenty measured repetitions per shape. vLLM resolves FULL graphs to FULL_DECODE_ONLY for this attention backend. It completes 69 requests: median decode 140.56 / 140.46 / 139.79 tokens/s and API latency 256.47 / 256.48 / 257.68 ms for 5 / 129 / 512 prompt tokens. This is one screened configuration, not the selected stock winner. Files are `clean-work/stock/vllm032-humming-fa2-baseline.jsonl`, `vllm032-humming-fa2-summary.json` and `vllm032-humming-fa2-baseline-attempt2.log`.

The loader reports conversion of some O/down weight-scale parameters from FP32 to BF16. Review the stock quantizer's supported scale configuration and report any effective arithmetic difference before treating this as a precision-matched comparison. Do not patch stock kernels to manufacture a favorable baseline. Future harness runs require `--hardware-profile rtx5090` or `--hardware-profile rtx-pro-6000`; their default memory fractions are 0.70 and 0.30 respectively, with explicit overrides recorded. The first screen predates that required flag but records its effective fraction and is an RTX 5090 run.

## Model integration checkpoint

The [session guide](../qwen3-8b-w8a8/sessions.md) documents the new public Rust API, CLI and separate hardware capacity configurations. All 16 tests pass in release (29.46 seconds) and checked release (27.44 seconds), including ragged batched full-model comparisons, prefix hits, full recomputation, divergent session forks, stop boundaries and bounded admission waves. Hashes match the unchanged legacy engine exactly. Native RoPE extension is available and a 32K bundle was prepared with preserved original positions; full-model long-context checks and optimized timing remain pending.

## Native model and staged-value checkpoint

The [active round 7 record](../qwen3-8b-w8a8/optimization/round-7.md) contains the paired eager 8K value-kernel experiment: 17.6% lower generation latency and 3.33× decode throughput, with all full-logit hashes preserved. Ragged native-context model tests pass in release (305.58 seconds) and checked release (321.73 seconds), with logs `native-context-tests.log` and `native-context-checked.log` under `clean-work/kv-batch-baseline/`. They validate cold/warm prefix reuse, complete recomputation, chunk-size changes and eager/graph decode through the native context limit. The ordinary GPU suite passes 16 tests with the long test explicitly ignored by default; `cuda.sh test` now forwards test-runner arguments for that separate run. Long prefill remains expensive and is a principal optimization target.
