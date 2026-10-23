# RTX PRO 6000 exact-order softmax experiments

All work ran on `ssh wx-workstation`. A 128-thread paged-softmax variant replaces twelve block-wide barriers with warp shuffles while preserving the original reduction tree. It passes bitwise operator comparisons, scoped sanitizers and a complete original-logit model regression. Its paired model trial shows two small consistent recomputation gains, but no broad model-level benefit beyond arm drift; the variant is not retained. A single-warp follow-up is rejected for long-context regressions. A loop-unroll hint produces byte-identical machine code and is not a distinct compiled optimization. Production and the original numerical gate remain unchanged.

## Arithmetic and synchronization

The original kernel assigns positions `t, t + 128, ...` to thread `t`, computes its ordered maximum and exponential sum, then reduces 128 values with descending distances 64, 32, 16, 8, 4, 2 and 1. The candidate retains these per-thread element sequences, the approximate base-two exponential, separate FP32 additions and correctly rounded division.

For each lane `i` of warp zero, the candidate computes `(v[i] + v[i + 64]) + (v[i + 32] + v[i + 96])`, reproducing the first two levels of the original sum tree. The remaining five levels use descending shuffle-down distances within that full warp. Only lane zero publishes the result. The maximum uses the corresponding identically nested maxima. Extra calculations in upper lanes do not enter lane zero's result. This changes the communication mechanism, not the parenthesization or assignment of score elements to accumulators.

All 128 threads publish their partial values before a barrier; warp zero then reads them, reduces and publishes before another barrier. The original barrier protecting reads of the maximum before reuse of shared scratch remains. The two reductions therefore require five block-wide barriers in total instead of seventeen. Every full-mask shuffle executes with all thirty-two lanes of warp zero participating. Both original and candidate compile to 34 registers, 2,048 shared bytes and zero stack/local storage in release and checked builds.

The single-warp follow-up assigns four original 128-strided accumulators to each of thirty-two physical lanes. It combines those four values with the same parenthesization and then the same warp tree. It removes the explicit shared reduction storage but reduces the number of physical threads. Compiled resources are 38 release / 34 checked registers, 1,024 shared bytes and no stack/local storage. Its long-context regressions rule it out for model integration.

## Completed operator evidence

The first variant passes 324 comparisons per release/checked build: twenty-seven causal lengths from one through 32,768, six input patterns and eager/graph execution. Each case contains three ragged rows and thirty-two heads per row. Patterns include signed zero, mixed finite scores, adjacent FP32 values, signed subnormals, large finite magnitudes and sharply concentrated distributions. Comparisons check every output bit against the original paged kernel, including NaN-payload sentinels in causal padding and a trailing guard; valid probabilities must be finite and nonnegative. Release memcheck and racecheck complete all 324 cases with zero errors, hazards or warnings. These are operator regression and synchronization checks, not an independent model numerical gate.

The single-warp and unroll follow-ups each repeat 648 comparisons per build and sanitizer run: the same 324 cases for both the first variant and the follow-up. Every run passes. No native-length model sanitizer claim follows from these operator cases.

Each timing shape resets the input scores before every invocation, outside the timed CUDA event interval, because softmax overwrites them. A 256 MiB cache eviction is also outside timing. The first screen uses original/new/new/original ordering; follow-ups use original/first-variant/follow-up/follow-up/first-variant/original. Each arm has three excluded warmups and five measured events, yielding ten measurements per implementation and shape. Every arm checks complete bitwise output agreement. Builds finish before GPU work, and jobs run serially.

The first three timing columns below come from `softmax-tree-v1`; the last column is the single-warp follow-up divided by its own control in `softmax-lanes-v1`. Ratios from different screens are not a shared-control ranking. Small event differences do not establish an API-level gain.

| Rows | Maximum causal length | Original median, ms | Warp-reduction median, ms | Warp / original | Single-warp / own control |
|---:|---:|---:|---:|---:|---:|
| 1 | 129 | 0.006144 | 0.006144 | 1.000 | 1.333 |
| 32 | 129 | 0.008192 | 0.008192 | 1.000 | 1.000 |
| 256 | 129 | 0.018432 | 0.016384 | 0.889 | 0.888 |
| 1 | 512 | 0.006144 | 0.006144 | 1.000 | 1.333 |
| 32 | 512 | 0.012288 | 0.011776 | 0.958 | 0.905 |
| 256 | 512 | 0.032768 | 0.030720 | 0.937 | 0.938 |
| 1 | 4096 | 0.008048 | 0.007936 | 0.986 | 2.400 |
| 32 | 4096 | 0.034816 | 0.034816 | 1.000 | 1.728 |
| 256 | 4096 | 0.196608 | 0.194560 | 0.990 | 1.281 |
| 1 | 32768 | 0.082944 | 0.081920 | 0.988 | 4.488 |
| 32 | 32768 | 0.432128 | 0.434176 | 1.005 | 1.974 |
| 256 | 32768 | 3.786752 | 3.784704 | 0.999 | 3.316 |

The unroll follow-up adds `#[unroll(4)]` to the three element loops of the first variant. The compiled release `.text` sections are byte-identical: 11,008 bytes each, SHA-256 `a2d89819b533baa0bec2831d334179d4ac3c43c363b57a94d7932cb47155986a`. Its duplicate timing screen therefore does not evaluate a changed instruction sequence. `unroll-code-audit.json` has SHA-256 `b46b4fb50e368aef98a309d776b8faa4b6299ad688a02c84a7db3481d995aff1`.

## Isolated model trial

`inferrous-softmax-model`, based on `89c074f`, selects only the first 128-thread variant for the PRO profile when `tuned_attention` is enabled. The original kernel remains available as the operator control and for other dispatches. The runner built release/checked tests and a CLI, required both complete ordinary suites, then ran separate hash verification and a paired timing pilot. No native-session or full-model sanitizer campaign follows the decision not to retain this candidate; the scoped operator results above do not substitute for those checks.

Both ordinary suites pass all twenty-three tests, including the 324-case bitwise softmax test, in 70.04 seconds release and 70.53 seconds checked. The validated-build receipt is `softmax-model-v1/validation.json`, SHA-256 `e5398f3d2132c941321142cc2af21aefabb9ff1f64818cec04701237811bd513`. Separate CLI verification preserves all 1,656 complete-vocabulary hashes per repetition across seven suites, each containing two repetitions of all ten workloads. Both binaries agree across cold reuse, prefix reuse and full recomputation, including the candidate's separate 64 GiB allocation. All timing arms started after this complete regression check passed.

The completed timing pilot covers solo 8K/two outputs, batch-32 prompt-129/eight outputs, batch-32 prompt-five/thirty-two outputs and batch-32 prompt-513/two outputs, each in all three modes. Each original/candidate/candidate/original suite has three excluded warmups and ten measurements, giving twenty samples per binary and case. The original retained CLI is `da73a3b486c9704b15254430d1d9eb9ff5bc4322bd7f50e59c979e0c3ee4d74b`; candidate is `9c6a2d304b23da19b610c9cf9005a73d1088bee72e1e3af8457299eed368e008`. This is a focused model trial, not the full stock matrix or an acceptance waiver.

Pilot configurations use 12 GiB KV, thirty-two slots, 512-row prefill chunks, graphs and the retained tuning flags. Prefix-enabled timing permits 128 entries; cold/recomputed configurations disable them. Timing disables logit verification, while the separate verification probes enable it. The 64 GiB allocation is a correctness probe, not a timing configuration. These policies remain distinct from the frozen final matrix's 64 GiB / thirty-two-entry configuration.

| Mode | Batch / prompt / outputs | Original API median, ms | Candidate API median, ms | Candidate / original |
|---|---|---:|---:|---:|
| Cold | 1 / 8192 / 2 | 2116.756 | 2121.979 | 1.0025 |
| Cold | 32 / 129 / 8 | 368.455 | 368.403 | 0.9999 |
| Cold | 32 / 5 / 32 | 424.690 | 424.839 | 1.0004 |
| Cold | 32 / 513 / 2 | 1252.708 | 1248.077 | 0.9963 |
| Prefix | 1 / 8192 / 2 | 45.937 | 45.770 | 0.9964 |
| Prefix | 32 / 129 / 8 | 101.901 | 101.471 | 0.9958 |
| Prefix | 32 / 5 / 32 | 419.688 | 417.270 | 0.9942 |
| Prefix | 32 / 513 / 2 | 28.443 | 28.412 | 0.9989 |
| Recompute | 1 / 8192 / 2 | 4259.705 | 4271.604 | 1.0028 |
| Recompute | 32 / 129 / 8 | 2314.502 | 2310.147 | 0.9981 |
| Recompute | 32 / 5 / 32 | 1659.709 | 1656.829 | 0.9983 |
| Recompute | 32 / 513 / 2 | 2502.391 | 2493.919 | 0.9966 |

Both candidate arm medians beat both baseline arm medians only for batch-32 recomputation at prompts 129 and 513, with pooled gains of 0.19% and 0.34%. Preserve those observations, but do not infer a broad improvement. Baseline drift is material elsewhere: cold solo 8K moves from 2104.910 to 2125.946 ms, while prefix batch-32 prompt-129 moves from 102.558 to 101.191 ms. The corresponding candidate arms lie between those baseline arms. All pooled changes are below 0.6% in magnitude. The screen does not justify retaining another production path; the earlier short/wide operator gains do not materially reduce the model-level gaps.

Active telemetry samples across the whole campaign, including verification and process transitions, span 1462–2272 MHz SM clocks, 57–87°C and 90.13–307.53 W. These are not per-API-call measurements or proof of a particular cause of drift. The final audit checks all nineteen raw suites, original/candidate identities, complete workload/repetition order, configurations, full output lengths, every reported median, log hashes and telemetry integrity. The runner is terminal and no GPU process from this campaign remains.

## Artifacts

All paths below are relative to `clean-work/pro6000-baseline/` on the workstation. All four campaigns are terminal. The combined operator audit verifies source/binary identities, seven successful command receipts per campaign, log hashes, complete comparison counts and embedded cubin provenance. The model audit is separate.

| Artifact | SHA-256 |
|---|---|
| `softmax-tree-v1/summary.json` | `ed312a97f683865b176d5947bf989a309169787233458e724b60cbae6c450ff7` |
| `softmax-lanes-v1/summary.json` | `9357fa8209498a719a00dc815e8d08abc8cf701e48671770f9b9b87391baebe5` |
| `softmax-unroll-v1/summary.json` | `4cffed7f81dfd076198352475cea1e04207c782cc6d0a2caf48284be9bf7da3f` |
| `softmax-unroll-v1/combined-audit.json` | `9a67411dc815ac54b6d2de6c8b20c3af7b9b8fcfe2c79083307d8c5d6551cfed` |
| `softmax-model-v1/identity.json` | `1fc8c6b05ed09266ff767dda9087762f38100fddddcb01a441176cf8fd62ea47` |
| `softmax-model-v1/source.patch` | `74acd5abfac5feb0c7aeb236ea45eb037638d6a11cb5d02f7f20dbd0633d545c` |
| `softmax-model-v1/run.py` | `06558db5cc5beb902d79ed4ee5838503ba537bdb3197f59d14f671967961e3b2` |
| `softmax-model-v1/paired/full-hash-verification.json` | `7ca012b70fd670f3e5e8d81e182b3cb56c57917bbd0c6bdd6d3ede7c8efd1b78` |
| `softmax-model-v1/paired/summary.json` | `86b65fcdab66dd6dd74900d48297d969504eb6114f0c8cc6f3a3b8353dd1b690` |
| `softmax-model-v1/summary.json` | `72bbebbe6fea6a0610e644e59280f87b4835fdcdf8586204cad3ffbe11cad1e7` |
| `softmax-model-v1/audit.json` | `17b14d533d835a3d10229ea796cc91f1d55655f7fbee0657f45d36b168494882` |
