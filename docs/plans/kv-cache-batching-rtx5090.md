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
| Rust cache ownership/admission | Bounded paged allocation, transactional reservation, valid-token tracking, release/reuse and randomized invariant tests | Host allocator implemented; four tests pass, including 10,000 randomized ownership transitions; GPU integration pending |
| Reusable sessions/prefixes | Cache hits avoid recomputation; misses, partial pages, reset and stale state are safe; outputs match recomputation | Pending |
| Batched model execution | Shared-weight row batches, ragged sequence positions and attention, batched head/selection, bounded scratch | Pending |
| True no-cache execution | Full prefix recomputed each output step; same prompt/output contract; valid batched baseline | Pending |
| Longer context | Bounded/chunked prefill, attention through 32K, configurable RoPE/cache capacity, correct memory admission | Pending |
| Hardware profiles | Explicit legacy PRO 6000 profile, independent 5090 tuning and metadata; no unmeasured cross-GPU claims | Pending |
| Off-the-shelf comparison | Reproducible current vLLM/SGLang screening, eligible winner per scenario, true no-cache reference | vLLM 0.32.0 container pulled; PyTorch 2.13.0+cu130 executes on the RTX 5090 through explicit device/driver mounts; timing pending |
| GPU correctness | Operator fixtures, solo/batched/cache/no-cache comparisons, checked builds, sanitizer coverage, long-context checks | Pending |
| Optimization | Profiles, retained/rejected experiments, paired measurements across all scenarios | Pending |
| Final report and completion audit | Requirement-by-requirement evidence, complete comparison tables, remaining limitations | Pending |

## Initial findings

The existing engine caches K/V only inside one request, with a single 4,096-token buffer per layer and a global execution mutex. A new request overwrites all visible state. Decode is graph-captured but batching is absent. Prefill allocates `rows × 32 × 4096` FP32 score scratch even when a fused short attention path is used; merely raising the context constant would amplify memory use and is not a sufficient long-context implementation. Existing INT8 multirow projection kernels can be reused for shared-weight batching, while sequence mapping, cache access, output heads and scheduling need new paths.

Current stock versions were checked against upstream releases: [vLLM 0.32.0](https://github.com/vllm-project/vllm/releases/tag/v0.32.0) and [SGLang 0.5.22](https://github.com/sgl-project/sglang/releases/tag/v0.5.22). [Humming](https://github.com/vllm-project/humming) is a candidate INT8 backend. The earlier workstation result is not a substitute for testing these on the RTX 5090. Docker is available locally, but no NVIDIA runtime/CDI configuration is installed; GPU access for stock containers will be validated before timing.
