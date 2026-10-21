# RTX 5090 KV cache and batching handoff

Paused at the user's request on 2026-10-20. Resume branch `feat/kv-cache-batching-rtx5090` in [bitemyapp/inferrous](https://github.com/bitemyapp/inferrous/tree/feat/kv-cache-batching-rtx5090). The latest implementation commit is `53ef642` (`Pipeline RTX 5090 projections and reuse staged head inputs`); this handoff follows it without changing production code. The checkout is `/home/callen/work/inferrous`. No benchmark, test or sanitizer job remained running at handoff.

The Nix setup, paged cache, reusable sessions, ragged batching, true full-prefix recomputation and native-context execution are implemented. Release and checked correctness gates pass, including the explicit long-context test. The requirement to beat the fastest eligible stock engine in every requested scenario is **not complete**: large batches and long prefill still have material gaps, and the final comparison matrix and report remain unfinished. This document is the current checkpoint; earlier chronological checkpoints in the [plan](kv-cache-batching-rtx5090.md) and [round 7](../qwen3-8b-w8a8/optimization/round-7.md) describe intermediate states.

## What is implemented

- [`cache.rs`](../../crates/inferrous-qwen3/src/cache.rs) owns a bounded pool of 64-token pages, transactional reservation, initialized-token lengths, reference counts, immutable complete-page sharing and RAII release. A 10,000-operation randomized host test exercises its invariants. BF16 K/V requires 147,456 bytes per token across all layers.
- [`gpu/batch.rs`](../../crates/inferrous-qwen3/src/gpu/batch.rs) implements `BatchEngine`, borrowing sessions, continuation, forks, bounded prefix caching, admission waves, ragged batches and graph-backed decode. Partial prefix pages are rebuilt privately and the last prompt token is always evaluated. GPU execution is serialized; execution errors fence and poison the runtime. Graphs are destroyed before captured allocations.
- `--cache-mode recompute` invalidates computed K/V after every output and submits the entire growing prefix again, retiring finished requests. Its processed-token counts are checked independently. Disabling prefix caching alone is not this mode.
- Score scratch is bounded independently of context. The 5090 policy uses at most 1 GiB per score workspace, a maximum prefill chunk of 1024 rows and power-of-two score strides; at 32K it limits prefill to 256 rows. Prefill and decode own separate workspaces.
- RTX 5090 tuning includes grouped GQA score/value kernels, shape-selected DP4A/split-K/asynchronous-copy/TMA projection paths, and lossless grouped output heads with staged inputs. The projection pipeline's dynamic shared-memory launch contract is 50,176 bytes and is prepared before graph capture. Existing FP32 accumulation, BF16 rounding and full-logit hashes are preserved.
- [`config.rs`](../../crates/inferrous-qwen3/src/config.rs) and the [hardware configurations](../../configs/qwen3-8b-w8a8) separate the RTX 5090 from RTX PRO 6000. The 5090 default is 12 GiB KV, batch capacity 16, chunk 1024 and 1 GiB score scratch; the latest benchmark explicitly used batch capacity 32. The PRO profile is 64 GiB KV, batch capacity 64, chunk 512 and 4 GiB score scratch. The actual GPU name is checked. PRO and the legacy engine retain their original kernel policy; current measurements are only for the 5090.

The [session guide](../qwen3-8b-w8a8/sessions.md) describes the API and CLI. Kernel implementations are in [`kernels.rs`](../../crates/inferrous-qwen3/src/kernels.rs); opt-in operator profiling is in [`gpu/profile.rs`](../../crates/inferrous-qwen3/src/gpu/profile.rs). One remaining capacity inefficiency is that the legacy `GpuModel` inside `BatchEngine` still owns approximately 604 MB of unused legacy KV and eight decode graphs. Removing those is a separate, unimplemented cleanup.

## Validation boundary

These are completed checks of the implementation at handoff, not promises about the draft kernels. All listed logs are local under `clean-work/kv-batch-baseline/`.

| Check | Result | Artifact |
|---|---|---|
| Ordinary release GPU suite | 16 passed, 4 ignored; 26.20 s | `pipeline-final-release-tests.log` |
| Ordinary checked release GPU suite | 16 passed, 4 ignored; 26.62 s | `pipeline-final-checked-tests.log` |
| Explicit native-context release test | 1 passed; 155.29 s | `pipeline-native-release.log` |
| Explicit native-context checked test | 1 passed; 211.94 s | `pipeline-native-checked.log` |
| Independent pipeline operators, memcheck | Zero errors | `pipeline-operators-memcheck.log` |
| Independent pipeline operators, racecheck | Zero hazards, errors or warnings | `pipeline-operators-racecheck.log` |
| Full ragged twelve-request model, memcheck | Zero errors; all 56 hashes and tokens preserved | `pipeline-ragged-memcheck.log` and `.jsonl` |
| Latest paired model verification | All 562 full-logit hashes per suite agree across both binaries and both cache modes | `pipeline-{reuse,recompute}-{av,bv}.jsonl` |
| Python benchmark and summary unit tests | Five passed | `crates/inferrous-qwen3/tools/tests/` |

The four ordinary-suite ignores are the explicit native-context test and three opt-in operator profiles. Native validation covers ragged 8K/16K/near-32K requests, cold/warm prefixes, recomputation, chunk changes and eager/graph execution. Plain `cargo test --release` does not enable GPU inference; use the wrapper below.

**Incomplete:** `pipeline-ragged-racecheck.log` contains only the sanitizer header, its JSONL is empty and its process is gone. It was interrupted and is not a pass. Updated `pipeline-8k-kernel-summary.json` and `pipeline-b32-kernel-summary.json` were never generated; their commands were queued behind that check. Restart only these missing checks with fresh artifact names. The local `validate_pipeline_tail.sh` also reruns an already-passed memcheck, so do not blindly rerun it or overwrite the old evidence.

Hash preservation establishes regression consistency. It does not resolve the earlier independent full-model PyTorch tolerance discrepancy documented in [discrepancies.md](../qwen3-8b-w8a8/discrepancies.md). Do not replace oracle hashes to make a changed arithmetic path pass.

## Measured improvements and remaining gaps

The latest A/B/B/A compared the previous grouped kernels against pipelined projections plus staged heads. Both variants used the native bundle, 12 GiB KV, maximum batch 32, chunk 1024, graphs, no prefix cache and `verify_logits: false`. Each process used three warmups and three measurements, yielding six measured samples per variant. These are tuning pilots, not the final twenty-sample matrix. Short workloads have 129 prompt tokens and eight outputs; the solo 8K case has two outputs. Separate probes enabled full-logit verification.

| Workload | Previous cached API | Current cached API | Previous recomputed API | Current recomputed API |
|---|---:|---:|---:|---:|
| Batch 1 | 54.627 ms | 53.378 ms | 119.785 ms | 110.815 ms |
| Batch 2 | 64.361 ms | 60.922 ms | 150.832 ms | 138.790 ms |
| Batch 4 | 77.231 ms | 75.593 ms | 233.652 ms | 233.618 ms |
| Batch 8 | 116.884 ms | 108.168 ms | 494.536 ms | 465.263 ms |
| Batch 16 | 182.264 ms | 167.104 ms | 931.489 ms | 884.579 ms |
| Batch 32 | 317.557 ms | 285.115 ms | 1798.407 ms | 1712.493 ms |
| Ragged 12 | 117.300 ms | 117.109 ms | 216.069 ms | 217.555 ms |
| Solo 8K, two outputs | 2019.058 ms | 1999.021 ms | 4044.103 ms | 4017.834 ms |

Treat the ragged, solo 8K and batch-four recomputation results as effectively unchanged. Prior grouped attention reduced a paired solo 8K/eight-output cached workload from 4.308 s to 2.056 s, preserving hashes. The latest pipelines improve cached batches 2/8/16/32 by 5.3/7.5/8.3/10.2% against the preceding implementation. Desktop activity remained present; active samples saw SM clocks of 1612–2865 MHz and temperatures of 58–83°C.

Stock screening used pinned vLLM 0.32.0 with FP32 weight scales, explicit Triton INT8 linear kernels, 12 GiB KV, 32K context and batch capacity 32. Graph-only Triton attention led sampled short batches 1/2/4/32; FlashInfer led batches 8/16 and solo 8K. Batch 32 was approximately 202.68 ms versus the current local 285.115 ms. FlashInfer's solo 8K/eight-output screen was 436.06 ms versus the earlier local eight-output result of 2.056 s. These are unpaired screens; the latest local two-output 8K case must not be presented as a matched eight-output comparison. The gap is substantial, but a definitive ranking still requires matching and pairing the final workloads.

Pinned SGLang 0.5.22 W8A8 and vLLM's automatic CUTLASS INT8 path reject SM120. Unsupported configurations are not wins. Humming runs but converts some FP32 scales to BF16, so it is a separate precision comparison. The optional unmodified vLLM B12X extra is pinned to `b12x==1.5.1`; its supported `--disable-cutedsl-warmup` setting bypasses a failing legacy precompile hook while retaining runtime compilation and excluded warmups. B12X graph and compiled screens completed but did not lead these cases. Image digests and launch handling are pinned in [`stock_container.sh`](../../crates/inferrous-qwen3/tools/stock_container.sh) and [`stock-vllm-b12x.Dockerfile`](../../crates/inferrous-qwen3/tools/stock-vllm-b12x.Dockerfile).

## Resume locally

The workstation is NixOS with an RTX 5090 reporting 32,607 MiB and driver 615.78.08. The pinned shell provides Rust nightly 2026-08-28, CUDA 13.4.92, LLVM 22, Compute Sanitizer, CUPTI and Python preparation dependencies. CUDA is already provided by the flake; no system toolkit installation is needed. Keep GPU jobs serial. See [local Nix development](../migrations/local-nix.md) for driver and preparation details.

```sh
git switch feat/kv-cache-batching-rtx5090
git submodule update --init --recursive
nix develop
nvidia-smi
crates/inferrous-qwen3/tools/cuda.sh build
crates/inferrous-qwen3/tools/cuda.sh test
CLEAN_BOUNDS_CHECKS=1 crates/inferrous-qwen3/tools/cuda.sh test
CLEAN_LONG_MODEL_BUNDLE="$PWD/clean-work/model-bundle-32k" \
  crates/inferrous-qwen3/tools/cuda.sh test --ignored native_context --nocapture
CLEAN_BOUNDS_CHECKS=1 CLEAN_LONG_MODEL_BUNDLE="$PWD/clean-work/model-bundle-32k" \
  crates/inferrous-qwen3/tools/cuda.sh test --ignored native_context --nocapture
python -m unittest discover -s crates/inferrous-qwen3/tools/tests -v
```

On a fresh checkout, prepare the missing bundles before tests. On this workstation, both already exist; reuse them. The original model revision is `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`. The 4K bundle root is `fad351c99221127c95303f18af560bd42dc8284179b93a9bc3360fb53286c20f`; the extended bundle has a distinct manifest root and preserves original RoPE bytes. Extension hard-links weights and Hugging Face files: **treat both bundles as immutable and never edit linked files in place**.

```sh
# Only when the corresponding prepared outputs are absent:
crates/inferrous-qwen3/tools/cuda.sh prepare
python crates/inferrous-qwen3/tools/extend_context.py \
  --source clean-work/model-bundle --output clean-work/model-bundle-32k
```

The legacy release benchmark is `crates/inferrous-qwen3/tools/cuda.sh bench clean-work/NEW-legacy.jsonl`; an optional second JSONL argument checks exact regression agreement. For batches, create a new output directory and a recorded configuration. This reproduces the latest pilot's workloads without needing the ignored local experiment script:

```sh
python - <<'PY'
import json
from pathlib import Path
p = Path('clean-work/resume-rtx5090')
p.mkdir(exist_ok=False)
config = json.loads(Path('configs/qwen3-8b-w8a8/rtx5090.json').read_text())
config.update(max_batch=32, prefix_cache_entries=0, verify_logits=False)
(p / 'config.json').write_text(json.dumps(config, indent=2))
def request(length, outputs, offset=0):
    return dict(prompt_ids=[42 + (17 * i + offset) % 1000 for i in range(length)],
                max_tokens=outputs, stop_ids=[])
workloads = {f'b{b:02}-p129': [request(129, 8, j) for j in range(b)]
             for b in [1, 2, 4, 8, 16, 32]}
workloads['b01-p8192'] = [request(8192, 2)]
lengths = [1, 5, 65, 129, 257, 33, 128, 64, 17, 256, 67, 3]
outputs = [8, 6, 4, 2, 1, 7, 5, 3, 2, 4, 6, 8]
workloads['ragged12'] = [request(n, m, j) for j, (n, m) in enumerate(zip(lengths, outputs))]
(p / 'workloads.json').write_text(json.dumps(workloads))
PY
python crates/inferrous-qwen3/tools/bench_batch.py \
  --binary target/inferrous-gpu/release/inferrous-qwen3 \
  --model clean-work/model-bundle-32k \
  --config clean-work/resume-rtx5090/config.json \
  --workloads clean-work/resume-rtx5090/workloads.json \
  --output clean-work/resume-rtx5090/reuse.jsonl --cache-mode reuse --repeat 23
python crates/inferrous-qwen3/tools/bench_batch.py \
  --binary target/inferrous-gpu/release/inferrous-qwen3 \
  --model clean-work/model-bundle-32k \
  --config clean-work/resume-rtx5090/config.json \
  --workloads clean-work/resume-rtx5090/workloads.json \
  --output clean-work/resume-rtx5090/recompute.jsonl --cache-mode recompute --repeat 23
python crates/inferrous-qwen3/tools/summarize_batch.py \
  clean-work/resume-rtx5090/reuse.jsonl clean-work/resume-rtx5090/recompute.jsonl \
  --output clean-work/resume-rtx5090/summary.json
```

Twenty-three repetitions mean three excluded warmups and twenty measured samples. Run separate verification probes with `verify_logits: true` and new output paths. Stock commands and comparable timing boundaries are in [benchmarks.md](../qwen3-8b-w8a8/benchmarks.md#paged-and-stock-batch-harnesses). Use the extended bundle's `hf` directory, `--max-context 32768`, `--max-num-seqs 32`, `--kv-cache-bytes 12884901888` and identical request arrays for matching these workloads. Rootless Docker works through the wrapper's explicit GPU devices and host-driver library mounts; no NVIDIA container runtime was installed.

## Immediate next work

1. Finish the interrupted full-model racecheck and collect new solo 8K and batch-32 CUPTI traces for the current implementation. Use fresh output paths and require complete sanitizer summaries and CUPTI footers with zero dropped/invalid records. The [profiling guide](../qwen3-8b-w8a8/profiling.md) gives commands. The earlier complete `grouped-b32-kernels.csv` had 23,436 records, but predates the current pipelines: eager time was 69.1% `linear_large`; graph time was 56.0% `linear_tiled` and 33.9% grouped head. Graph time includes the final prefill step in that workload. Instrumented timings are not benchmark measurements.
2. Evaluate attention reuse across four consecutive prompt rows within one sequence. The two source sketches are preserved as [QK draft](../qwen3-8b-w8a8/optimization/drafts/attention_scores_tiled.rs.txt) and [PV draft](../qwen3-8b-w8a8/optimization/drafts/attention_values_tiled.rs.txt). **They have never been compiled, tested or integrated.** They depend on helpers inside the current kernel module and are documentation artifacts, not build inputs. Production has no `attention_scores_tiled` or `attention_values_tiled` dispatch. The attempted integration was interrupted before edits executed.
3. Before using those drafts, build host-validated `(first_row, count)` groups of at most four contiguous rows from the same session. Own the metadata in `Step` with its captured allocations; never cross sequence slots. Use the path only with at least two queries per tile on average; decode remains on the existing kernels. QK intends to share a 64-key tile across four rows and four GQA heads. PV intends to double-buffer V and probability stripes; each row must stop arithmetic at its own causal end, even when zero-filled copies cover padded tails. Extend `grouped_attention_shapes` with partial groups, shuffled pages and ragged rows, compare every output against independent references, then perform sanitizer and paired model validation. See the [plan's design notes](kv-cache-batching-rtx5090.md#next-experiment-share-attention-tiles-across-prompt-rows).
4. Consider a separate batched-head experiment using original BF16 weights directly to avoid packed-weight reconstruction. This is only an idea; it must preserve term order/rounding and account for extra VRAM at the configured KV budget. Retain the measured packed solo path. Use the refreshed profiles to decide priority.
5. Complete the matched stock matrix and final acceptance audit: cached solo, cached batches and genuine uncached batches; prompt lengths 5/129/512/2K/4K/8K/16K/near-32K; batch sizes through 32 where admission fits; ragged, cold/warm prefixes, growth/capacity boundaries and short/sustained outputs. Reserve output tokens inside the 32,768-token total limit. Select the fastest eligible stock configuration per scenario. The uncached stock harness has smoke/submission-count validation, but no complete final uncached matrix.

## Evidence and preservation notes

The public branch contains implementation, pinned environment, maintained harnesses, docs and the two explicitly unvalidated source sketches. `clean-work/`, `target/`, model weights, container images, raw logs and frozen executables are ignored and **are not included in the push**. They remain on this workstation; back them up separately if migrating machines. Published numerical summaries are in round 7 and this handoff.

- `clean-work/model-bundle`, `clean-work/model-bundle-32k`, `clean-work/operator-fixtures`, `clean-work/workloads`: prepared inputs. Model loading validates manifests and tensor hashes.
- `clean-work/kv-batch-baseline/grouped-head-candidate`: previous frozen executable, SHA-256 `a4fea320dbe14c172840db691784639d1338f46f6f720b6bf478687a8d0767ea`.
- `clean-work/kv-batch-baseline/pipeline-candidate`: current implementation's frozen benchmark executable, SHA-256 `62272d132955545c43dc159fd3536a8a373f3d0fd41a45e944e4b7a05a7996f1`. It was frozen before final test/comment additions; do not assume a fresh rebuild has the same executable hash.
- `clean-work/kv-batch-baseline/pipeline-{reuse,recompute}-{a1,b1,b2,a2,av,bv}.jsonl`, `pipeline-abba-summary.json`, `compare_pipeline.py` and `pipeline-abba-telemetry.csv`: latest paired evidence and exact workload/configuration generation.
- `clean-work/stock/vllm032-ct-triton-*-native-screen.*`: native-context stock screens. Round 7 identifies the other screening and rejected-experiment artifacts.
- `pipeline-self-*` contains an invalid initial A/B that accidentally ran the candidate in both arms. It is excluded from all claims. The corrected script asserts distinct binary SHA-256 values and validates every record's identity. Preserve that guard; never rebuild or overwrite a frozen arm during measurement.

Keep warmup/loading/JIT, API latency, first-token time, post-first-token rate, aggregate throughput, prefix reuse and verification overhead distinct. Performance profiles disable full-logit transfer/hashing; correctness probes enable it. Use paired A/B/B/A for retained tuning decisions and record clocks, temperature and desktop activity. A failed stock worker, incomplete trace, missing comparison or changed arithmetic contract is not performance evidence. Resume the unfinished work from this checkpoint rather than marking the original goal complete.
