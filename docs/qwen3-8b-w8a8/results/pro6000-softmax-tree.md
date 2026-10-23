# RTX PRO 6000 exact-order softmax experiments

All work runs on `ssh wx-workstation`. A 128-thread paged-softmax variant replaces twelve block-wide barriers with warp shuffles while preserving the original reduction tree. It passes bitwise operator comparisons and scoped sanitizers, with modest short/wide operator gains. An isolated model trial is running; no production change or model-level improvement is established. A single-warp follow-up is rejected for long-context regressions. A loop-unroll hint produces byte-identical machine code and is not a distinct compiled optimization.

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

`inferrous-softmax-model`, based on `89c074f`, selects only the first 128-thread variant for the PRO profile when `tuned_attention` is enabled. The original kernel remains available as the operator control and for other dispatches. The runner builds release/checked tests and a CLI, requires both complete ordinary suites, then runs separate hash verification and a paired timing pilot. Native-session qualification, full-model sanitizer coverage and a retention decision remain separate work if the pilot supports the change.

Both ordinary suites now pass all twenty-three tests, including the 324-case bitwise softmax test, in 70.04 seconds release and 70.53 seconds checked. The validated-build receipt is `softmax-model-v1/validation.json`, SHA-256 `e5398f3d2132c941321142cc2af21aefabb9ff1f64818cec04701237811bd513`. Separate CLI hash verification is running; timing arms are gated on its complete success.

The planned verification covers the existing ten-workload manifest in cold, prefix-reuse and recompute modes, both binaries, two repetitions, and a separate 64 GiB candidate allocation. The timing pilot covers solo 8K/two outputs, batch-32 prompt-129/eight outputs, batch-32 prompt-five/thirty-two outputs and batch-32 prompt-513/two outputs, each in all three modes. Each original/candidate/candidate/original suite has three excluded warmups and ten measurements, giving twenty samples per binary and case. The original retained CLI is `da73a3b486c9704b15254430d1d9eb9ff5bc4322bd7f50e59c979e0c3ee4d74b`; candidate is `9c6a2d304b23da19b610c9cf9005a73d1088bee72e1e3af8457299eed368e008`. This is a pending focused trial, not the full stock matrix or an acceptance waiver.

Pilot configurations use 12 GiB KV, thirty-two slots, 512-row prefill chunks, graphs and the retained tuning flags. Prefix-enabled timing permits 128 entries; cold/recomputed configurations disable them. Timing disables logit verification, while the separate verification probes enable it. The 64 GiB allocation is a correctness probe, not a timing configuration. These policies remain distinct from the frozen final matrix's 64 GiB / thirty-two-entry configuration.

## Artifacts

All paths below are relative to `clean-work/pro6000-baseline/` on the workstation. The three operator campaigns are terminal. Their combined audit verifies source/binary identities, seven successful command receipts per campaign, log hashes, complete comparison counts and embedded cubin provenance. The model campaign is separate and remains live at this checkpoint.

| Artifact | SHA-256 |
|---|---|
| `softmax-tree-v1/summary.json` | `ed312a97f683865b176d5947bf989a309169787233458e724b60cbae6c450ff7` |
| `softmax-lanes-v1/summary.json` | `9357fa8209498a719a00dc815e8d08abc8cf701e48671770f9b9b87391baebe5` |
| `softmax-unroll-v1/summary.json` | `4cffed7f81dfd076198352475cea1e04207c782cc6d0a2caf48284be9bf7da3f` |
| `softmax-unroll-v1/combined-audit.json` | `9a67411dc815ac54b6d2de6c8b20c3af7b9b8fcfe2c79083307d8c5d6551cfed` |
| `softmax-model-v1/identity.json` | `1fc8c6b05ed09266ff767dda9087762f38100fddddcb01a441176cf8fd62ea47` |
| `softmax-model-v1/source.patch` | `74acd5abfac5feb0c7aeb236ea45eb037638d6a11cb5d02f7f20dbd0633d545c` |
| `softmax-model-v1/run.py` | `06558db5cc5beb902d79ed4ee5838503ba537bdb3197f59d14f671967961e3b2` |
