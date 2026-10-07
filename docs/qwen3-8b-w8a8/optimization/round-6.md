# Sixth optimization pass: keep DRAM busy between projections

Baseline: round-5 final executable, SHA-256 `e01c85e654b89913b83d0b7bd73274a78c52b015ee45949cbb0d8b2eb8ac77c2` (runtime `b88d941f`). All builds, tests and timing run on the benchmark workstation, one GPU job at a time. `baseline.jsonl` remains the complete-logit oracle. The independent full-model Torch gate remains failing; exact regression is a separate claim. Raw screening data and profiles stay in `clean-work/review-20261006.AIT0la`; retention runs are in `clean-work/round6-20261006.NMTdSj`. Compact records are under [round 6 results summary](../results/round-6.md).

## Where a decode step went

A graph-level trace of the control (p5, warm) executes each decode graph in 5754.5 us with 38.9 us between graphs (172.6 tokens/s under the profiler). A node-level trace attributes the 364 kernels per step as follows; in-graph launch gaps total only 47 us.

| Kernels | us/step | Notes |
|---|---:|---|
| gate/up (36) | 2276 | 100.7 MB each, 1592 GB/s |
| O + down with residual (72) | 1652 | O 11.94 us (1405 GB/s), down 33.94 us (1483 GB/s) |
| QKV (36) | 609 | 16.91 us, 1488 GB/s |
| packed head (1) | 574 | 92% of DRAM peak under NCU |
| latency-bound kernels (252) | 642 | norm/quantize 3.69 us x72, attention 3.90, Q/K prep 2.54, quantize 1.97, known-max quantize 1.17, selection 21.3, final norm 8.6 |

Each step reads about 7.88 GB: 6.95 GB INT8 body plus the 12-bit packed head. Under NCU the memory clock runs at 13.35 GHz, a peak of about 1.71 TB/s; streaming kernels reach 91–92% of it. The practical floor is therefore about 5.0 ms per step. Roughly 830 us per step had DRAM not streaming weights: the latency-bound kernels, plus about 1.3–2.5 us of ramp and tail per projection. The SASS of the decode loop keeps up to sixteen independent loads in flight per warp, so memory-level parallelism was not the limit.

## Microbenchmarks on SM120

[microbench](../results/round-6.md#microbenchmarks) holds the CUDA C++ sources and outputs. CUDA event timing is quantized to about 2 us on this device and even reported above-peak bandwidth; use CUPTI kernel timestamps instead.

- A latency-bound kernel followed by a cold 16/50 MB GEMV-like read: L2 prefetch issued by sixteen extra CTAs during the small kernel hides 80–85% of its duration, inside CUDA graphs (for example 14.18 -> 12.29 us with a 2.4 us small kernel and 16 MB read). Bulk (`cp.async.bulk.prefetch.L2`) and per-thread (`prefetch.global.L2`) hints perform the same.
- Bulk prefetch issue runs at about 120 GB/s per SM. Eight CTAs lengthen the small kernel; 32–64 CTAs issue faster but land fewer bytes, consistent with dropped hints. Sixteen approximately match DRAM bandwidth.
- `prefetch.global.L2::evict_last` lines survived even a 768 MB streaming flush and polluted later measurements. Do not use evict-last hints for weights that are read once.
- Programmatic dependent launch saves about 0.5 us per boundary inside a graph, but only when the predecessor triggers its dependents early. Without an early trigger it gives nothing.

## Retained: hoist cold loads off latency-bound critical paths

RMSNorm gains, Q/K-norm gains and the RoPE row are cold in L2 every step. The original kernels loaded them after their reductions, putting a DRAM miss on the critical path. Load them with the activations instead; the final norm also issues all sixteen of its per-thread loads before summing. Addition order, reduction trees and BF16 boundaries are unchanged. With weight prefetch off, fused norm/quantize falls from 3.75 to 2.26 us, Q/K preparation from 2.57 to 1.80 us, and the final norm from 8.88 to 2.49 us. The first version of the final-norm change assumed rows of at most 4096; the independent 12288-wide fixture caught it, and the retained version loads in 4096-wide chunks.

## Retained: warm the visible KV prefix during QKV

With a nonzero row budget, each QKV thread issues one `prefetch.global.L2` line over the layer's visible K (coordinate-major) and V (token-major) prefix. Attention then hits L2 instead of queueing behind weight traffic. Attention falls from 4.12 to 3.84 us with weight prefetch off.

## Retained: programmatic graph edges

After capture, every kernel-to-kernel edge becomes a programmatic edge. Every decode kernel's real CTAs execute `griddepcontrol.wait` before touching any produced, consumed or reused buffer, then `griddepcontrol.launch_dependents`. That gives one kernel of lookahead with unchanged memory ordering; see [safety contract](../safety.md). cuda-oxide has no PDL launch attribute, so the graph surgery uses the driver's 12.3+ edge-data API and the device side uses inline PTX. On their own, with prefetch disabled, programmatic edges raise screening decode from 174.40 / 163.64 to 177.44 / 167.31 tokens/s (p5 / p512).

## Retained: prefetch only where DRAM is idle

Under programmatic launch, a small kernel's extra CTAs start while its predecessor still runs. Hints from small kernels that follow a projection therefore land in already saturated time, and they no longer help. Two kinds of prefetch remain:

- Attention and split-attention softmax append sixteen CTAs that warm O[0, 6 MiB); decode O quantization's sixteen warm O[6, 10 MiB). Under programmatic launch these overlap Q/K preparation and attention, which never use DRAM fully.
- Before waiting, each projection or head warp warms the first 1024 bytes of its own weight row (512 for gate/up). Under programmatic launch this runs while the preceding latency-bound kernel executes, spread across every SM.

`CLEAN_PREFETCH=0` disables all hints and restores the earlier launch shapes; `CLEAN_PDL=0` restores ordinary edges. Neither changes arithmetic.

## Rejected or neutral

These are screening medians of ten measured requests; adjacent repeated controls drifted by up to about 1%. [screening sweeps](../results/round-6.md#screening-sweeps) has every configuration.

| Trial | Outcome |
|---|---|
| Prefetch CTAs in every latency-bound kernel (no programmatic edges, before hoisting) | +3.4% / +2.9%, but norm/quantize grew 3.75 -> 6.16 us and attention 4.12 -> 5.88 us: the hints delayed their own critical-path misses. |
| 32 or 64 prefetch CTAs instead of 16 | Worse at every budget, for example 174.16 versus 176.66 tokens/s. |
| Smaller or larger per-window budgets after hoisting | All within noise (180.4–181.3 / 167.9–169.1). |
| Selection kernel prefetching the next step's layer-0 QKV | Neutral; selection grew from 23.6 to 33.1 us. |
| Head prefetch during the final norm | Neutral to negative; the head grew by about 11 us. |
| Small-kernel prefetch in norm/quantize, Q/K prep, known-max quantize, final norm and selection under programmatic edges | Removed: these CTAs now overlap the saturated predecessor. |
| Whole-row (4 KiB+) self-prefetch | Worse: 179.54 / 168.86 versus 182.69 / 171.22, consistent with L2 thrash. |

## Results

The serial A/B/B/A protocol ran control, candidate, candidate, control, then pooled forty measured requests per implementation and shape. [final-benchmarks.sh](../results/round-6.md#final-benchmarkssh) is the exact procedure. The control is the round-5 executable above; the candidate is `7affbce7a426e33b8b37b4f6e116a8c98c4bd130fb20c481b991f610b87f4fad`.

| Prompt tokens | Control decode tok/s | Candidate decode tok/s | Control API ms | Candidate API ms |
|---:|---:|---:|---:|---:|
| 5 | 170.278 | **182.270** | 190.356 | **178.290** |
| 129 | 165.292 | **178.205** | 206.480 | **192.860** |
| 512 | 160.138 | **170.496** | 237.988 | **226.197** |

That is **7.0% / 7.8% / 6.5% more decode throughput** and **6.3% / 6.6% / 5.0% lower request latency**. Both candidate runs beat both control runs on every shape. First-step medians are unchanged (7.95 / 18.64 / 44.12 ms) because prefill is untouched. A graph-level trace puts decode-graph execution at 5430 us, against 5755 us for the control, with the roughly 40 us host round trip between graphs unchanged. Against the fastest stock decode measured in round 5 (B12X: 144.4 / 143.3 / 142.9 tokens/s), decode is now 26% / 24% / 19% faster. Stock was not rerun in this pass.

### Validation

- All 276 measured requests (138 per implementation) preserve every token ID and complete-vocabulary hash of `baseline.jsonl`, 4,416 per implementation. Every screening run was checked in the same way.
- Optimized and `CLEAN_BOUNDS_CHECKS=1` builds pass all nine tests. The checked build also matches the oracle on all three shapes.
- The graph/eager replay test now compares programmatic decode graphs with eager launches, at seventeen lengths through 4093 tokens.
- Operator fixtures run attention, softmax and O quantization with sixteen appended prefetch blocks, and the projections and head with full-row hints. Outputs match the independent references and the no-prefetch launches.
- Compute Sanitizer memcheck, racecheck and synccheck report zero errors or hazards on the operator tests. Full-model memcheck (512-token prefill, three programmatic decode steps) reports zero errors.
- Sampled process GPU memory is unchanged at 10,256 MiB. rustfmt and the Python syntax checks pass.

The full-model memcheck exposed a pre-existing oddity, recorded in [memcheck determinism notes](../results/round-6.md#memcheck-determinism). Under memcheck, the *round-5 control's* 512-token request yields different logit hashes from run to run (tokens still match), while p5 and p129 match the oracle exactly for both control and candidate. Without the sanitizer, both binaries reproduce the oracle exactly on every run. Only prompts of 256 or more rows use the CTA-local TMA projection, and that kernel passes its operator test exactly under all three sanitizer tools. The likeliest explanation is a sanitizer limitation with bulk tensor copies, but a latent timing-dependent issue in that path is not ruled out. A racecheck/synccheck of a 256-row model prefill is the next diagnostic.

## Remaining opportunities

1. **Hide the host round trip (about 40 us/step, 0.7%).** Each step waits for an 11 us logit copy, synchronization and the next launch. Let selection write the next token and position into the control record, enqueue step N+1 before reading step N, and copy logits into a three-slot pinned ring. The rejected four-step graph paid for in-graph copies; this design does not. A nonfinite result must still be clamped as an embedding index for already queued work.
2. **Parallel greedy selection (about 18 us/step).** One 1024-thread CTA scans 151,936 logits in 21–23 us. A multi-CTA pass with a last-block finish, or a fused head epilogue using an order-preserving 64-bit key, keeps the lowest-index tie rule.
3. **Fewer latency-bound kernels.** About 640 us/step remains in kernels that cannot saturate DRAM. Exact fusions remain possible: O quantization (an order-independent maximum) into the O projection prologue; Q/K preparation into attention (four query heads recompute their KV head's norm/RoPE, and one writes the cache); and attention's 128-wide reductions using warp shuffles with the identical pairing. Each must beat the prefetch overlap now hiding part of these windows.
4. **Tighter head coding (about 0.7%).** The head runs at 92% of DRAM peak with about half its issue slots idle. Its BF16 weights have roughly 10.6 bits of entropy; 11-bit codes with a three-bit exponent class and sparse exceptions would save about 7% of head bytes. Check the exponent histogram first.
5. **First-class PDL in cuda-oxide.** A launch attribute (or a `#[programmatic_launch]` kernel option) plus `griddepcontrol` intrinsics would replace the graph surgery and inline PTX used here.
