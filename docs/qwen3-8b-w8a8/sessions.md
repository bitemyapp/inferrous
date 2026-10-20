# Paged sessions and batching

The paged runtime shares model weights across ragged requests, stores K/V in a fixed byte budget, and processes prefill in bounded chunks. It supports reusable sessions, immutable full-page prefix forks, an LRU prefix cache, capacity-based admission waves, and explicit full-prefix recomputation. This is an actively validated implementation; the performance comparison matrix in [the work tracker](../plans/kv-cache-batching-rtx5090.md) remains unfinished.

## Configuration and command line

The separate starting configurations are `configs/qwen3-8b-w8a8/rtx5090.json` (12 GiB KV, batch capacity 16, prefill chunk up to 1024, score scratch up to 1 GiB per workspace) and `configs/qwen3-8b-w8a8/rtx-pro-6000.json` (64 GiB KV, batch capacity 64, chunk up to 512, score scratch up to 4 GiB per workspace). These are capacity policies, not measured tuning optima. Only the RTX 5090 is available for current validation. The runtime validates the selected profile against the actual CUDA device name before allocating model or cache storage. Model weights, workspace and desktop use require memory in addition to the KV budget. `prefill_chunk` is a maximum token-row count: the scheduler reduces it as needed to keep score scratch within `score_scratch_bytes`. Decode and prefill retain separate workspaces, and other activations add to their memory use. Old scratch is released before a replacement shape is allocated. A submitted batch can exceed simultaneous capacity: the engine admits successive waves and includes their queue time in each request's first-token and completion latency.

Extend the prepared 4K bundle before using the 32K profiles:

```sh
nix develop
python crates/inferrous-qwen3/tools/extend_context.py \
  --source clean-work/model-bundle --output clean-work/model-bundle-32k
crates/inferrous-qwen3/tools/cuda.sh build
target/inferrous-gpu/release/inferrous-qwen3 clean-work/model-bundle-32k \
  --engine-config configs/qwen3-8b-w8a8/rtx5090.json \
  --prompt 'Explain paged KV caching.' --max-tokens 32
```

The extension preserves all original RoPE bytes and appends positions through 32,767 using the original Qwen formula. Weights and Hugging Face files are hard-linked to conserve disk space; treat both bundles as immutable and do not edit linked files in place. The new manifest has a distinct root. There is no YaRN extension or quality claim beyond native context. The model loader validates the configured table length and hashes.

`--batch FILE` accepts a JSON array of the same request objects used by `--request`: `prompt_ids`, `max_tokens`, and `stop_ids`. Batch mode requires `--engine-config FILE`. Prompt plus maximum output must fit `max_context`, which must fit the bundle's RoPE table. `--cache-mode reuse` retains autoregressive K/V and eligible prompt prefixes; `--cache-mode recompute` clears the initialized prefix after every sampled token and processes all prompt and generated tokens again at the next iteration. Its scratch KV is reused as storage but no computed K/V survives to the next output iteration.

The output records the actual GPU name, exact engine configuration, cache mode, request order, per-request token IDs and timings, processed token rows, reused prefix token rows and batch elapsed time. `verify_logits: true` additionally copies and hashes every vocabulary logit for correctness comparisons. Performance profiles use `false` and return an empty hash list. Timing reports must state which setting they used. Prefix cache hits occur only at 64-token boundaries; the last prompt token is always evaluated to obtain the output head.

## Rust ownership

`BatchEngine::session` creates a sequence borrowing its engine. `generate_sessions` accepts mutable sessions and corresponding `SessionOptions`; `append` adds new user tokens, and `fork` shares initialized complete pages while rebuilding any partial tail privately. The borrow prevents the engine's device allocations from being freed while sessions exist. Dropping a session releases its page references. `clear_prefixes` releases the automatic prefix cache; independently owned sessions remain valid.

Every admitted sequence reserves its worst-case capacity before kernels run. Allocation failure does not partially acquire pages. The execution mutex serializes GPU access, and initialized lengths are published only after a successful stream fence. An error during execution fences queued work and poisons the runtime, requiring recreation; an ordinary rejected admission leaves it usable. Generation calls are synchronous. Dropping a session between calls needs no further GPU cancellation mechanism because its previous call has already fenced all uses.

## Validation checkpoint

Release and checked release each pass all 16 tests, including full-model comparisons against the original engine. The new integration test covers ragged lengths 1/5/65/129/257, reordered batch members, different stop boundaries, warm prefixes, continuation, divergent forks, switching a shared session to recomputation, admission waves, capacity rejection and reclamation. Complete-vocabulary hashes and selected tokens match exactly. The existing original eager/graph test also remains unchanged. Operator tests cover shuffled physical pages, page boundaries and 32K attention. Full-model 32K validation and performance optimization are tracked separately and are not implied by these short-context results.

Logs: `clean-work/kv-batch-baseline/batch-integration-tests-attempt2.log` (29.46 seconds) and `batch-integration-tests-checked.log` (27.44 seconds). The first compile attempt exposed ambiguous test sum types; specifying `usize` fixed it without changing execution arithmetic.

## Native-context validation and batch benchmarks

The explicit long test requires the extended bundle and reserves 9 GiB of KV storage. It compares ragged 8K/16K/near-32K cold, warm and fully recomputed requests, changes prefill chunk size, and compares eager and graph execution. It is deliberately separate from the default suite because it takes several minutes. The [round 7 record](optimization/round-7.md) contains evidence and remaining limitations.

```sh
CLEAN_LONG_MODEL_BUNDLE="$PWD/clean-work/model-bundle-32k" \
  crates/inferrous-qwen3/tools/cuda.sh test --ignored native_context --nocapture
CLEAN_BOUNDS_CHECKS=1 CLEAN_LONG_MODEL_BUNDLE="$PWD/clean-work/model-bundle-32k" \
  crates/inferrous-qwen3/tools/cuda.sh test --ignored native_context --nocapture
```

`cuda_graphs: true` enables graph replay for decode batches; setting it to `false` provides the eager comparison. Graphs own stable step storage and are replaced when shape changes. The `bench_batch.py` tool accepts a workload JSON object mapping names to request arrays, runs repetitions with a resident engine per workload, verifies complete deterministic outputs, and checks the exact processed-row count in recomputation mode. Use `prefix_cache_entries: 0` for cold-prefix measurements. A nonzero value permits explicit warm-prefix measurements, recorded through `reused_tokens`.

```sh
python crates/inferrous-qwen3/tools/bench_batch.py \
  --binary target/inferrous-gpu/release/inferrous-qwen3 \
  --model clean-work/model-bundle-32k \
  --config configs/qwen3-8b-w8a8/rtx5090.json \
  --workloads path/to/batch-workloads.json --cache-mode reuse \
  --output clean-work/batch-results.jsonl
```
