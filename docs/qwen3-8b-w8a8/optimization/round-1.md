# Measurement and optimization log

Baseline: one request, 5/129/512 prompt tokens, 32 generated tokens, EOS ignored. Three warmups and twenty retained repetitions per workload. No prompt cache or speculative decoding. Weights are resident. All work executes on the benchmark workstation, RTX PRO 6000 Blackwell Max-Q, 300 W, driver 580.159.04. GPU jobs run serially.

`generate_seconds` brackets the public generation call (including per-request KV allocation, checksums and detokenization); `generation.request_seconds` excludes KV pool reservation and final checksum/detokenization. Per-step times include CPU launch work, synchronization, logit transfer and greedy selection. Decode throughput is `(generated_count-1)/sum(step_seconds[1:])`; prefill is the first step. Comparison engines report their public generate-call boundary and retain backend timing separately. Do not mix these boundaries.

Initial profile: Nsight Systems, 129 prompt tokens, two 32-token requests. Device kernel time: linear 53.8%, attention 19.8%, quantization 12.3%, head 7.0%, norm 5.5%. CPU CUDA API time includes 72,476 allocations/frees and 16,974 stream synchronizations (model loading is included in API totals).

| Candidate | Impact | Confidence | Effort | Score |
|---|---:|---:|---:|---:|
| INT8 tensor-core prefill | 5 | 5 | 3 | 8.3 |
| Reuse scratch allocations and remove intra-layer waits | 4 | 5 | 3 | 6.7 |
| Normalize attention scores once per key | 4 | 5 | 1 | 20 |
| Share QKV and gate/up activation quantization | 3 | 5 | 2 | 7.5 |

Each retained lever gets its own commit and measurements. Integer products fit INT32 (`12288*127*127 = 198193152`); changing the exact integer reduction tree does not change the sum. Floating-point scale application, BF16 cast, token tie-breaking and request ordering must remain unchanged. No RNG is used in generation. Raw baseline logit hashes are saved for all three workloads.

The original baseline can be rebuilt from its introductory commit, and its binary is retained at `clean-work/clean-baseline` on the benchmark workstation. Large bundles, traces, Nsight captures, fixture data and engine environments remain in `clean-work`; small raw measurements and reports are summarized in the [2026-10-02 results](../results/rounds-1-2.md).

## Retained: INT8 tensor-core prefill

Use a 16x8x32 register MMA tile for at least 32 prompt rows. The initial unconditional prefill switch regressed the five-token prompt (20.2 to 35.0 ms), so small prompts retain DP4A. Final prefill medians: 20.2 / 82.8 / 305.4 ms for 5 / 129 / 512 tokens, versus 20.2 / 187.8 / 790.8 ms. Decode is unchanged.

Proof: exact integer sum independent of reduction order, accumulator bound above; identical ordered FP32 scales and BF16 cast. Token tie-breaking and request ordering unchanged; RNG N/A. Independent fixtures test both DP4A and MMA, including partial rows/columns and maximal accumulator values. All 69 requests (2,208 complete vocabulary vectors) have identical logit hashes and token IDs. Reprofile: attention now consumes 24.7% of GPU kernel time, quantization 15.3%.

## Retained: reuse normalized attention probabilities

Calculate each score/denominator once, then share it across 128 PV coordinates. The same RN division produces the same FP32 bits. A block barrier protects the shared probabilities; PV multiplication and summation order remain identical. No RNG, token or scheduling changes. Independent operators and all 69 full-model logit-hash regressions pass. At 512 prompt tokens, API latency drops from 1086.7 to 1057.4 ms and decode rises from 40.36 to 41.53 tokens/s; the short prompt is essentially unchanged. Attention's share falls from 24.7% to 23.6%.

## Retained: request-local scratch reuse

Reuse separate prefill/decode workspaces across all layers and steps. The stream orders producers, consumers and buffer reuse; the final logit read completes each step. No intermediate is read before its complete producer writes it. Residuals use two separate buffers and quantizer scratch lives through all queued reads. No arithmetic, RNG, token ties or dependent operation ordering changes. All 69 requests retain exact logit hashes and tokens. API latency becomes 375.4 / 483.3 / 895.8 ms; decode becomes 88.93 / 74.53 / 49.46 tokens/s (5 / 129 / 512 tokens). Including loading, profile allocation/free calls fall from 72,476 each to 932; stream synchronizations fall from 16,974 to 782. GPU kernel durations are stable.

## Device bounds checks in the admitted fixed-shape domain

Score 8 (impact 4, confidence 4, effort 2): attention, quantization and head are top-five device costs. An isolated checked/unchecked compiler experiment preserves all 69 requests' hashes but reduces attention GPU time from 201.7 to 61.5 ms, quantization from 132.7 to 71.8 ms and head from 75.2 to 48.5 ms in the same profile. API latency is 314.5 / 365.2 / 553.7 ms, decode 107.12 / 101.39 / 87.15 tokens/s. Register/resource inspection finds no local-memory spills.

Arithmetic, scheduling, RNG and tie-breaking remain unchanged. Eliding checks is valid only for the fixed graph's admitted indices; model tensor fields are now crate-private to enforce that construction boundary. The per-kernel proof and lifetime argument are in the safety contract. A checked build remains available with `CLEAN_BOUNDS_CHECKS=1`; malformed requests are rejected on the host. Independent fixtures and whole-model Compute Sanitizer memcheck pass. Operator memcheck, racecheck and synccheck also pass with zero errors/warnings. Change retained.

## Rejected: 64-row register MMA tiles

Sharing one weight fragment across four 16-row accumulators was numerically exact and reduced 512-token prefill from 184.3 to 170.0 ms. However, 129-token prefill regressed from 45.9 to 56.9 ms, and decode was unchanged. The small aggregate benefit does not justify another kernel and dispatch policy in this baseline. The candidate code was removed; measurements remain as `prefill64*`. Larger operator fixture cases from the experiment remain useful for partial-tile tests.

## Retained: share activation quantization between sibling projections

Q/K/V consume the same normalized hidden state; gate/up consume another shared normalized state. Quantize each of those once, then keep the bytes and scales read-only until all sibling projections finish on the same stream. This removes three quantizations per layer (108 launches per step). The source, reduction, RNE division, INT8 bytes and scale bits are identical to each removed invocation. No arithmetic, RNG, tie-breaking or dependent operation ordering changes.

All 69 requests / 2,208 vocabulary hashes and the extended independent fixtures pass. Full-model Compute Sanitizer memcheck reports zero errors. API medians are 296.6 / 347.3 / 534.9 ms; decode is 113.95 / 107.36 / 91.47 tokens/s. Compared with the previous retained version, decode improves 6.4% / 5.9% / 5.0%.

The final 129-token profile reduces quantizer launches from 16,128 to 9,216 and their total GPU time from 71.8 to 45.8 ms. Remaining device time: DP4A linear 49.9%, prefill MMA 12.5%, attention 10.4%, norm 8.3%, head 8.2%, quantization 7.8%. The next measured priorities are projection throughput and prefill; see the stock-engine comparison for the remaining gap. Further kernel changes should wait for resolution of the full-model numerical discrepancy.
