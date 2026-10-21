# Round 8: RTX PRO 6000 batching and tiled prefill

Work resumed from PR #1's [RTX 5090 handoff](../../plans/kv-cache-batching-rtx5090-handoff.md) on 2026-10-20. All builds, tests, sanitizer runs and benchmarks run through `ssh wx-workstation`, in `/home/callen/work/inferrous-kv-batch`. This separate worktree preserves unrelated edits in `/home/callen/work/inferrous`. Source inspection and editing may happen on the client machine. GPU jobs run serially. This is an active experiment, not the final comparison matrix or completion report.

## Environment and frozen baseline

The device reports `NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition`, 97,887 MiB, with a 300 W power limit and driver 580.159.04. The build uses the branch's pinned Rust nightly 2026-08-28 and cuda-rust submodule `6921d3ee07e28ec62895ea40fc2fef39977ca411`, with the workstation's CUDA 13.1.115 toolkit at `/usr/local/cuda-13.1`. The Nix CUDA 13.4 toolkit used on the RTX 5090 is not assumed here. Existing 4K model/operator fixtures are reused read-only; a new 32K bundle hard-links its weights and preserves original RoPE bytes. Its model root is `672c32995847c8ccf2d743e61a85ee9023231fc5cae9e1aa0af5ee1b48695d47`.

Raw artifacts live under `clean-work/pro6000-baseline/` on the workstation and are ignored by Git. The starting branch is `b82b6b9`. `baseline-binary` has SHA-256 `864a88df1cea4c4bc8fddaeca44e9835efaea57fbfe33a97a6d614f18722142b`. The tiled candidate frozen after eliminating unused decode metadata copies is `tiled-candidate-v2`, SHA-256 `636b4431f7625cee5d7e1c9e6bda2d43cb293a458b19e2f60a37a1b8228c7020`. Frozen executables must not be overwritten during measurement.

The initial configuration uses the PRO hardware policy, 12 GiB KV, batch capacity 32, 512-row prefill chunks, 4 GiB per score workspace, 32K context, graphs, no prefix cache and no logit hashing for timings. This explicitly reduced KV budget matches the stock screen; it does not validate the profile's full 64 GiB capacity. Workloads use batches 1/2/4/8/16/32 with 129-token prompts and eight outputs, a solo 8192-token prompt with eight outputs, and the handoff's ragged twelve-request workload. Earlier 5090 two-output 8K measurements are not comparable to this eight-output case.

## Profiling and implementation

The baseline batch-32 CUPTI trace contains 31,224 records, and the solo 8K trace contains 41,829 records. Both have complete footers with zero dropped or invalid records. Batch-32 eager time is 60.5% `linear_large`; its graph time is 57.7% `head_packed_rows` and 35.3% `linear_tiled`. Solo 8K eager time is 44.7% paged PV and 42.0% paged QK. Instrumented timings are diagnostics, not performance measurements. Artifacts are `b32-p129-*v2.*` and `b01-p8192-*v2.*`.

The CUPTI wrapper now supports the toolkit's `lib64` layout and its current activity record type. CUDA 13.1 declares `CUpti_ActivityKernel11`, while the original code assumed CUDA 13.4's `CUpti_ActivityKernel13`. Header/runtime API versions must agree. The first profiling attempt failed at compile time and is not counted as a trace.

The handoff's QK/PV sketches are integrated as opt-in tiled prefill kernels. Host-built groups contain at most four contiguous rows from one session; groups never cross a slot or position discontinuity. `Step` owns the captured metadata. At least two rows per group on average are required; decode keeps existing kernels. Partial groups and causal tails preserve each row's FP32 term order and BF16 rounding. See the [safety argument](../safety.md#tiled-paged-prefill-attention).

Explicit tuning flags now work on either validated hardware profile, allowing the existing projection pipelines and grouped/staged heads to be evaluated on the PRO 6000. The PRO defaults remain disabled pending the acceptance work. `tiled_attention` defaults to false on both profiles. The legacy engine's policy remains unchanged.

The operator profile compares original, grouped and tiled attention. For 256 rows, the measured median combined QK/softmax/PV time at strides 512/4096/32768 is respectively 0.340/3.355/33.905 ms for original kernels, 0.145/1.460/16.649 ms for grouped kernels, and 0.112/1.120/13.626 ms for tiled kernels. These isolated diagnostic samples do not establish model-level improvements. All operator outputs agree exactly. Artifact: `tiled-operator-profile.log`.

## Cached paired pilot

The completed cached A/B/B/A has six measured samples per variant and shape, after three warmups per process. Every one of the 151 full-logit hashes in the separate two-output cached verification suite agrees. The table uses the timing workloads described above, including eight outputs for the solo 8K case. The candidate combines tiled prefill with the opt-in projection, grouped-attention and staged-head policies; these measurements do not attribute the combined gain to any single kernel change. Artifact: `paired-reuse-summary.json`.

| Workload | Original API median | Candidate API median | Reduction |
|---|---:|---:|---:|
| Batch 1, prompt 129 | 60.951 ms | 58.686 ms | 3.7% |
| Batch 2, prompt 129 | 91.047 ms | 72.606 ms | 20.3% |
| Batch 4, prompt 129 | 120.802 ms | 94.908 ms | 21.4% |
| Batch 8, prompt 129 | 181.098 ms | 137.263 ms | 24.2% |
| Batch 16, prompt 129 | 306.638 ms | 214.756 ms | 30.0% |
| Batch 32, prompt 129 | 528.375 ms | 375.259 ms | 29.0% |
| Ragged 12 | 175.818 ms | 137.594 ms | 21.7% |
| Solo 8K, eight outputs | 5372.336 ms | 2146.259 ms | 60.0% |

These pilots do not establish the final twenty-sample stock ranking, and true-recomputation timings/verification remain in progress.

## Validation checkpoint

| Check | Evidence | Status |
|---|---|---|
| Starting implementation, release GPU suite | `release-tests.log` | 16 passed, 4 ignored; 28.96 s |
| Python benchmark harness tests | `harness-tests.log` on workstation | 11 passed, including six completeness/cross-mode regression-checker tests |
| Tiled operators against paged and independent contiguous implementations | `tiled-operator-test.log` | Passed; partial groups, shuffled pages, ragged slots, page crossings and complete output/probability tensors |
| Integrated tiled candidate, release GPU suite | `tiled-release-tests.log` | 18 passed, 4 ignored; 39.94 s |
| New full-model tiled test | Included in release suite | Exact full-logit hashes across 64/512-row chunks, eager/graph execution, cold/warm prefixes and full recomputation |
| Final metadata-copy adjustment, release/checked suites | `validation-queue.log` and named logs | Queued after paired timings; not yet a pass |
| Native-context release/checked comparison | `tiled-native-{release,checked}.log` | Queued; original cold/warm output compared with tuned/tiled full recomputation |
| Tiled operator memcheck/racecheck and full ragged-model racecheck | `tiled-operators-*.log`, `tiled-ragged-racecheck.*` | Queued; complete zero-error summaries required |

`run_paired.py` records distinct frozen binary identities, runs original/candidate/candidate/original for both cache modes, and captures GPU clocks, utilization, memory, temperature and power. Each arm has three excluded warmups and three measured samples, yielding six measured samples per variant. Separate verification probes enable full-logit hashing and limit outputs to two; their timings must not be mixed with eight-output performance samples. Recompute probes independently validate the exact submitted-token count. The final twenty-sample, full-shape matrix and stock ranking remain outstanding.

The maintained `tools/check_batch_regression.py` additionally checks the complete expected workload/repetition set and compares outputs across binaries and cache modes together, retaining raw-file SHA-256 identities in its report. Its initial cached report, `cached-hash-verification.json`, confirms all eight workloads and 151 hashes per repetition. The four-input cross-mode report remains pending until recomputation probes finish.

The pinned vLLM 0.32.0 container is downloaded and its PyTorch/vLLM imports succeed. The stock Triton INT8/Triton attention graph screen is queued after validation with matching requests, 32K context, 32 sequence slots and 12 GiB KV. Import success alone is not evidence of working model inference or competitive performance. Additional eligible stock configurations, true uncached results, larger capacity, warm prefixes, sustained outputs and the full final acceptance audit remain required.
