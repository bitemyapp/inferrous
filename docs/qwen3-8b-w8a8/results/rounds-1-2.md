# Results summary: Initial baseline, first optimization pass and round 2

Generated from the raw artifacts in `results/2026-10-02` of the development repository, which are not published. The narrative, rationale and conclusions are in [round 1 record](../optimization/round-1.md), [round 2 record](../optimization/round-2.md). Timing columns are medians over measured runs; decode is tokens per second for tokens 2–32; API is the public generation call in milliseconds; first step is the prefill step in milliseconds.

## Measurements

| Variant | Measured runs | Decode p5 | Decode p129 | Decode p512 | API p5 | API p129 | API p512 | First step p512 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| asynctile | 20 | 117.8 | 110.6 | 93.9 | 288.0 | 313.9 | 403.0 | 58.89 |
| attention | 20 | 66.1 | 57.9 | 41.5 | 501.6 | 631.2 | 1057.4 | 297.85 |
| attentiondispatch | 20 | 129.1 | 124.8 | 118.7 | 261.5 | 282.3 | 343.1 | 67.51 |
| attentiontile | 20 | 121.1 | 117.6 | 107.5 | 280.6 | 298.1 | 372.0 | 69.67 |
| baseline | 20 | 66.0 | 57.1 | 40.3 | 502.8 | 743.4 | 1572.6 | 790.79 |
| dp4a16 | 20 | 113.2 | 106.9 | 91.2 | 298.8 | 348.6 | 536.8 | 182.88 |
| final-checked | 0 | — | — | — | — | — | — | — |
| gpuselect | 20 | 150.9 | 144.8 | 138.3 | 214.4 | 232.0 | 270.2 | 45.75 |
| gqaprefill | 20 | 149.4 | 143.5 | 137.1 | 223.8 | 241.4 | 278.2 | 44.39 |
| graphs | 20 | 149.7 | 143.5 | 135.8 | 223.3 | 243.9 | 303.6 | 67.76 |
| groupedlinear | 20 | 160.9 | 156.4 | 151.8 | 201.2 | 217.4 | 250.4 | 45.94 |
| hashoverlap | 20 | 149.7 | 143.8 | 137.4 | 216.0 | 233.6 | 270.4 | 44.54 |
| headunroll | 20 | 156.7 | 152.4 | 148.2 | 206.3 | 222.5 | 255.1 | 45.64 |
| keytranspose | 20 | 118.2 | 111.1 | 94.3 | 287.1 | 332.1 | 435.3 | 92.62 |
| large32 | 20 | 160.9 | 156.4 | 151.9 | 201.2 | 217.4 | 256.2 | 51.84 |
| large32pipe | 20 | 161.3 | 156.8 | 152.2 | 200.7 | 216.9 | 257.5 | 53.44 |
| largepipeline | 20 | 149.7 | 143.8 | 137.4 | 223.3 | 242.2 | 289.6 | 56.28 |
| largepipeline96 | 20 | 157.5 | 153.0 | 148.7 | 205.4 | 220.5 | 259.8 | 51.08 |
| largetile | 20 | 149.7 | 143.5 | 137.3 | 223.5 | 242.6 | 287.1 | 53.53 |
| ldmatrix | 20 | 149.7 | 143.8 | 137.3 | 223.3 | 242.1 | 298.7 | 65.22 |
| mlpmax | 20 | 155.3 | 150.9 | 146.6 | 208.4 | 223.1 | 254.5 | 42.90 |
| normquant | 20 | 119.4 | 112.1 | 95.1 | 284.1 | 331.6 | 474.0 | 134.02 |
| normrope | 20 | 152.5 | 146.5 | 139.6 | 212.1 | 229.7 | 268.3 | 45.92 |
| prefill64 | 20 | 107.3 | 101.4 | 87.1 | 314.1 | 376.0 | 539.6 | 170.04 |
| prefillgraph | 20 | 156.5 | 152.2 | 148.0 | 205.9 | 222.1 | 254.9 | 45.16 |
| prepareqk | 20 | 157.5 | 153.0 | 148.6 | 205.4 | 220.7 | 252.9 | 44.11 |
| prepareqkdispatch | 20 | 157.3 | 153.0 | 148.7 | 205.6 | 221.0 | 253.3 | 44.53 |
| projection256 | 20 | 161.2 | 156.7 | 152.2 | 200.8 | 217.0 | 249.7 | 45.69 |
| pvpipeline | 20 | 149.7 | 143.8 | 137.3 | 223.4 | 243.5 | 300.2 | 66.69 |
| pvunroll | 20 | 152.9 | 148.8 | 144.7 | 211.4 | 225.8 | 257.3 | 42.88 |
| qkunroll | 20 | 160.2 | 154.1 | 152.0 | 202.1 | 220.4 | 250.2 | 45.89 |
| quant1024 | 20 | 128.9 | 124.8 | 113.5 | 264.5 | 282.3 | 356.4 | 69.42 |
| quantguard | 20 | 115.9 | 109.0 | 93.0 | 292.3 | 318.1 | 406.7 | 59.30 |
| reanchor | 20 | 156.5 | 152.3 | 148.0 | 206.6 | 222.7 | 255.5 | 45.82 |
| residualnorm | 20 | 156.8 | 152.7 | 148.3 | 206.1 | 222.0 | 254.6 | 45.35 |
| round2-baseline | 20 | 113.8 | 107.3 | 91.3 | 297.5 | 348.1 | 534.0 | 181.04 |
| round2-checked | 0 | — | — | — | — | — | — | — |
| round2-final | 20 | 161.0 | 156.5 | 152.0 | 201.1 | 217.3 | 250.1 | 45.81 |
| shared-quant | 20 | 113.9 | 107.4 | 91.5 | 296.6 | 347.3 | 534.9 | 182.26 |
| splitattention | 20 | 127.4 | 124.9 | 118.5 | 265.0 | 282.2 | 344.0 | 67.54 |
| splitk | 20 | 117.2 | 113.9 | 104.3 | 285.9 | 305.8 | 380.4 | 69.69 |
| splitkprefill | 20 | 128.9 | 124.8 | 113.6 | 261.7 | 282.1 | 356.4 | 69.69 |
| splitpv | 20 | 149.4 | 143.5 | 135.5 | 224.0 | 243.8 | 300.8 | 64.25 |
| stock-final-b12x | 20 | 144.4 | 143.4 | 142.9 | 251.7 | 253.5 | 253.8 | 36.80 |
| stock-final-native | 20 | 140.6 | 140.5 | 140.3 | 236.9 | 237.4 | 242.7 | 21.51 |
| tensorcore-gated | 20 | 66.0 | 57.2 | 40.4 | 502.4 | 637.9 | 1086.7 | 305.38 |
| tensorcore | 20 | 66.0 | 57.1 | 40.4 | 517.5 | 638.2 | 1086.1 | 305.03 |
| tiled32 | 20 | 114.0 | 107.6 | 91.6 | 296.3 | 343.3 | 486.4 | 134.62 |
| unchecked-experiment | 20 | 107.1 | 101.4 | 87.2 | 314.5 | 365.2 | 553.7 | 184.29 |
| vllm-b12x-eager-2 | 20 | 37.8 | 37.7 | 37.7 | 877.9 | 878.9 | 878.8 | 57.13 |
| vllm-eager | 20 | 113.3 | 110.9 | 110.2 | 292.3 | 298.5 | 306.5 | 25.22 |
| vllm-flash | 20 | 77.9 | 77.8 | 77.8 | 414.9 | 416.9 | 424.7 | 25.98 |
| vllm-graph1 | 20 | 77.9 | 77.8 | 77.7 | 415.1 | 417.0 | 425.2 | 26.13 |
| vllm-graphs-native | 20 | 126.6 | 130.1 | 129.3 | 257.7 | 256.5 | 267.2 | 27.82 |
| vllm-humming-b12x | 20 | 144.4 | 143.3 | 142.9 | 252.0 | 253.5 | 254.5 | 37.28 |
| vllm-humming-eager | 20 | 78.1 | 76.9 | 77.2 | 423.8 | 430.1 | 428.5 | 26.90 |
| vllm-humming-graphs-native | 20 | 140.6 | 140.4 | 140.0 | 237.2 | 236.9 | 244.0 | 22.56 |
| vllm-v1 | 20 | 78.2 | 78.2 | 78.1 | 413.1 | 415.4 | 423.5 | 26.24 |
| workspace | 20 | 88.9 | 74.5 | 49.5 | 375.4 | 483.3 | 895.8 | 255.09 |

## Regression checks

Each check compares every token ID and complete-vocabulary logit hash against the original oracle.

| Check | Result |
|---|---|
| asynctile-regression.json | pass (69 requests, 3 workloads) |
| attention-regression.json | pass (69 requests, 3 workloads) |
| attentiondispatch-regression.json | pass (69 requests, 3 workloads) |
| attentiontile-regression.json | pass (69 requests, 3 workloads) |
| dp4a16-regression.json | pass (69 requests, 3 workloads) |
| final-checked-regression.json | pass (3 requests, 3 workloads) |
| gpuselect-regression.json | pass (69 requests, 3 workloads) |
| gqaprefill-regression.json | pass (69 requests, 3 workloads) |
| graphs-regression.json | pass (69 requests, 3 workloads) |
| groupedlinear-regression.json | pass (69 requests, 3 workloads) |
| hashoverlap-regression.json | pass (69 requests, 3 workloads) |
| headunroll-regression.json | pass (69 requests, 3 workloads) |
| keytranspose-regression.json | pass (69 requests, 3 workloads) |
| large32-regression.json | pass (69 requests, 3 workloads) |
| large32pipe-regression.json | pass (69 requests, 3 workloads) |
| largepipeline-regression.json | pass (69 requests, 3 workloads) |
| largepipeline96-regression.json | pass (69 requests, 3 workloads) |
| largetile-regression.json | pass (69 requests, 3 workloads) |
| ldmatrix-regression.json | pass (69 requests, 3 workloads) |
| mlpmax-regression.json | pass (69 requests, 3 workloads) |
| normquant-regression.json | pass (69 requests, 3 workloads) |
| normrope-regression.json | pass (69 requests, 3 workloads) |
| prefill64-regression.json | pass (69 requests, 3 workloads) |
| prefillgraph-regression.json | pass (69 requests, 3 workloads) |
| prepareqk-regression.json | pass (69 requests, 3 workloads) |
| prepareqkdispatch-regression.json | pass (69 requests, 3 workloads) |
| projection256-regression.json | pass (69 requests, 3 workloads) |
| pvpipeline-regression.json | pass (69 requests, 3 workloads) |
| pvunroll-regression.json | pass (69 requests, 3 workloads) |
| qkunroll-regression.json | pass (69 requests, 3 workloads) |
| quant1024-regression.json | pass (69 requests, 3 workloads) |
| quantguard-regression.json | pass (69 requests, 3 workloads) |
| reanchor-regression.json | pass (69 requests, 3 workloads) |
| residualnorm-regression.json | pass (69 requests, 3 workloads) |
| round2-baseline-regression.json | pass (69 requests, 3 workloads) |
| round2-checked-regression.json | pass (3 requests, 3 workloads) |
| round2-final-regression.json | pass (69 requests, 3 workloads) |
| shared-quant-regression.json | pass (69 requests, 3 workloads) |
| splitattention-regression.json | pass (69 requests, 3 workloads) |
| splitk-regression.json | pass (69 requests, 3 workloads) |
| splitkprefill-regression.json | pass (69 requests, 3 workloads) |
| splitpv-regression.json | pass (69 requests, 3 workloads) |
| tensorcore-gated-regression.json | pass (69 requests, 3 workloads) |
| tensorcore-regression.json | pass (69 requests, 3 workloads) |
| tiled32-regression.json | pass (69 requests, 3 workloads) |
| unchecked-regression.json | pass (69 requests, 3 workloads) |
| workspace-regression.json | pass (69 requests, 3 workloads) |

## Tests and sanitizers

| Log | Result |
|---|---|
| gpuselect-racecheck.log | ========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings) |
| gpuselect-test.log | test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 8.92s |
| gqaprefill-racecheck.log | ========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings) |
| gqaprefill-test.log | test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 8.83s |
| graphs-memcheck.log | ========= ERROR SUMMARY: 0 errors |
| graphs-racecheck.log | ========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings) |
| graphs-test.log | test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 8.90s |
| groupedlinear-racecheck.log | ========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings) |
| groupedlinear-test.log | test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 8.85s |
| hashoverlap-test.log | test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 8.72s |
| large32-test.log | test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 8.81s |
| large32pipe-test.log | test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 8.81s |
| largepipeline-test.log | test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.45s |
| largepipeline96-test.log | test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.57s |
| largetile-racecheck.log | ========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings) |
| largetile-test.log | test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.47s |
| ldmatrix-racecheck.log | ========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings) |
| ldmatrix-test.log | test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.47s |
| memcheck-model.log | ========= ERROR SUMMARY: 0 errors |
| memcheck-operators.log | ========= ERROR SUMMARY: 0 errors |
| memcheck-shared-quant.log | ========= ERROR SUMMARY: 0 errors |
| memcheck-tiled32.log | ========= ERROR SUMMARY: 0 errors |
| mlpmax-racecheck.log | ========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings) |
| mlpmax-test.log | test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 8.77s |
| normrope-racecheck.log | ========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings) |
| normrope-test.log | test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 8.79s |
| prefillgraph-test.log | test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 8.89s |
| prepareqk-racecheck.log | ========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings) |
| prepareqk-test.log | test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 8.75s |
| pvpipeline-racecheck.log | ========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings) |
| pvpipeline-test.log | test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 8.92s |
| pvunroll-racecheck.log | ========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings) |
| pvunroll-test.log | test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.57s |
| qkunroll-test.log | test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 8.74s |
| quant1024-racecheck.log | ========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings) |
| quant1024-test.log | test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.35s |
| racecheck-asynctile.log | ========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings) |
| racecheck-attentiontile.log | ========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings) |
| racecheck-normquant.log | ========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings) |
| racecheck-operators.log | ========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings) |
| racecheck-tiled32.log | ========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings) |
| residualnorm-test.log | test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 8.82s |
| round2-final-memcheck.log | ========= ERROR SUMMARY: 0 errors |
| round2-final-model-memcheck.log | ========= ERROR SUMMARY: 0 errors |
| round2-final-racecheck.log | ========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings) |
| round2-final-synccheck.log | ========= ERROR SUMMARY: 0 errors |
| round2-final-test.log | test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 16.67s |
| splitattention-racecheck.log | ========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings) |
| splitattention-test.log | test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.33s |
| splitk-racecheck.log | ========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings) |
| splitk-test.log | test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.34s |
| splitpv-test.log | test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 8.88s |
| synccheck-operators.log | ========= ERROR SUMMARY: 0 errors |
| test-2.log | test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.32s |
| test-attention.log | test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.32s |
| test-prefill64.log | test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.32s |
| test-quantguard.log | test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.34s |
| test-shared-quant.log | test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.31s |
| test-tensorcore.log | test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.33s |
| test-unchecked.log | test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.30s |

## Memory

Sampled process GPU memory in a separate instrumented run; a sampled maximum, not an exact peak.

| File | Workload | Samples | Max sampled MiB |
|---|---|---:|---:|
| memory.json | — | — | — |
| round2-memory.json | p512 | 10 | 10548.0 |

## Profiles

Nsight Systems kernel summaries: the six kernels with the largest share of GPU time.

| Profile | Top kernels (share of GPU time) |
|---|---|
| asynctile-profile-summary | linear 55.1%, attention 11.7%, head 9.1%, norm_quantize 8.5%, quantize 5.3%, linear_tiled 5.3% |
| attention-profile-summary | linear 34.4%, attention 23.6%, quantize 15.5%, head 8.8%, linear_prefill 8.6%, norm 6.9% |
| attentiondispatch-profile-summary | linear 50.4%, attention_finish 13.2%, linear_tiled 12.2%, head 8.3%, attention_scores 3.9%, norm_quantize 3.5% |
| attentiontile-profile-summary | linear 58.5%, head 9.7%, norm_quantize 9.0%, attention 6.3%, quantize 5.7%, linear_tiled 5.5% |
| dp4a16-profile-summary | linear 49.8%, linear_prefill 12.5%, attention 10.5%, norm 8.3%, head 8.3%, quantize 7.8% |
| graphs-profile-summary | linear_tiled 53.4%, attention_finish 25.2%, attention_scores 12.0%, swiglu 1.8%, norm 1.6%, quantize 1.3% |
| groupedlinear-profile-summary | linear_pair 29.3%, linear 21.3%, head 10.1%, linear_large 8.9%, linear_triple 7.9%, attention_prefill 4.7% |
| keytranspose-profile-summary | linear 51.4%, linear_tiled 11.7%, attention 10.9%, head 8.5%, norm_quantize 7.9%, quantize 5.0% |
| normquant-profile-summary | linear 51.5%, linear_tiled 11.7%, attention 10.8%, head 8.5%, norm_quantize 7.9%, quantize 5.0% |
| normrope-profile-summary | linear 55.9%, head 9.3%, linear_large 8.0%, attention_values 6.3%, attention_prefill 5.2%, norm_quantize 3.9% |
| prefill64-profile-summary | linear 46.1%, linear_prefill64 15.0%, quantize 11.3%, attention 9.7%, norm 7.6%, head 7.6% |
| profile-summary | linear 53.8%, attention 19.8%, quantize 12.3%, head 7.0%, norm 5.5%, rope 0.5% |
| quant1024-profile-summary | linear 63.3%, head 10.4%, attention 6.8%, linear_tiled 6.0%, norm_quantize 4.3%, quantize 3.4% |
| reanchor-profile-summary | linear 59.4%, head 9.8%, linear_large 8.7%, attention_prefill 4.6%, norm_quantize 4.2%, attention_values 3.6% |
| round2-final-profile-summary | linear_pair 29.2%, linear 21.3%, head 10.1%, linear_large 8.9%, linear_triple 7.9%, attention_prefill 4.8% |
| shared-quant-profile-summary | linear 49.9%, linear_prefill 12.5%, attention 10.4%, norm 8.3%, head 8.2%, quantize 7.8% |
| tensorcore-profile-summary | linear 34.0%, attention 24.7%, quantize 15.3%, head 8.7%, linear_prefill 8.4%, norm 6.8% |
| tiled32-profile-summary | linear 50.4%, linear_tiled 11.5%, attention 10.6%, norm 8.3%, head 8.3%, quantize 7.9% |
| unchecked-profile-summary | linear 47.8%, linear_prefill 11.9%, quantize 11.7%, attention 10.0%, norm 7.9%, head 7.9% |
| workspace-profile-summary | linear 34.4%, attention 23.6%, quantize 15.5%, head 8.8%, linear_prefill 8.6%, norm 7.0% |

## Nsight Compute

Speed-of-light metrics per profiled kernel. These reports used a locked base clock unless noted in the records, so absolute durations are not comparable with uninstrumented benchmarks.

| Report | Kernel | Duration | DRAM throughput | SM throughput | SM frequency |
|---|---|---:|---:|---:|---:|
| gqa | attention_prefill | 779.94 us | 1.41 % | 84.78 % | 1.03 Ghz |
| large | linear_large | 113.06 us | 17.56 % | 40.06 % | 985.43 Mhz |
| prefill | linear_tiled | 187.01 us | 10.62 % | 64.10 % | 950.06 Mhz |

## Procedures

The exact scripts used for the retained measurements and checks. `<workdir>` is the development checkout on the benchmark workstation.

### round2-final-checks.sh

```sh
#!/usr/bin/env bash
set -euo pipefail
cd <workdir>
export CUDA_OXIDE_BACKEND=<workdir>/vendor/cuda-oxide/crates/rustc-codegen-cuda/target/release/librustc_codegen_cuda.so
export RUST_TEST_THREADS=1
export CLEAN_OPERATOR_FIXTURES=$PWD/clean-work/operator-fixtures-64
export CLEAN_MODEL_BUNDLE=$PWD/clean-work/model-bundle
cp target/inferrous-gpu/release/inferrous-qwen3 clean-work/round2-final-optimized
cp target/inferrous-gpu/release/deps/inferrous_qwen3-68bfae6b70bc8921 clean-work/round2-final-tests
CLEAN_BOUNDS_CHECKS=1 crates/inferrous-qwen3/tools/cuda.sh build > clean-work/round2-checked-build.log 2>&1
python3 crates/inferrous-qwen3/tools/bench_clean.py --binary target/inferrous-gpu/release/inferrous-qwen3 --model clean-work/model-bundle --workloads clean-work/workloads --output clean-work/round2-checked.jsonl --repeat 1 > clean-work/round2-checked.log 2>&1
python3 crates/inferrous-qwen3/tools/check_regression.py clean-work/baseline.jsonl clean-work/round2-checked.jsonl > clean-work/round2-checked-regression.json
crates/inferrous-qwen3/tools/cuda.sh build > clean-work/round2-restored-build.log 2>&1
for check in memcheck racecheck synccheck; do
    /usr/local/cuda-13.1/bin/compute-sanitizer --tool "$check" --error-exitcode=1 clean-work/round2-final-tests conformance:: --test-threads=1 > "clean-work/round2-final-$check.log" 2>&1
done
/usr/local/cuda-13.1/bin/compute-sanitizer --tool memcheck --error-exitcode=1 target/inferrous-gpu/release/inferrous-qwen3 clean-work/model-bundle --request clean-work/smoke-request.json > clean-work/round2-final-model-memcheck.log 2>&1
nsys profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none --output=clean-work/round2-final-profile target/inferrous-gpu/release/inferrous-qwen3 clean-work/model-bundle --request clean-work/workloads/p512.json --repeat 2 > clean-work/round2-final-profile.log 2>&1
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum --format csv clean-work/round2-final-profile.nsys-rep > clean-work/round2-final-profile-summary.csv 2>&1
python3 clean-work/round2-memory.py
cargo +nightly-2026-04-03 fmt -p inferrous-qwen3 -- --check
python3 -m py_compile crates/inferrous-qwen3/tools/*.py
sha256sum clean-work/round2-final-optimized target/inferrous-gpu/release/inferrous-qwen3 > clean-work/round2-binaries.sha256
nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,temperature.gpu,clocks.sm,clocks.mem --format=csv > clean-work/round2-final-device.csv
```

## Executable hashes

round2-binaries.sha256:

```text
7ccc78a314eb77e64ae133215d40d3e2e2a5eb4a3e1d5781e5f26670fad456c1  clean-work/round2-final-optimized
7ccc78a314eb77e64ae133215d40d3e2e2a5eb4a3e1d5781e5f26670fad456c1  target/inferrous-gpu/release/inferrous-qwen3
```

## Other raw artifacts

Not summarized here: 1 `.csv` file; 5 `.json` files; 45 `.log` files; 1 `.py` file; 1 `.rs` file (build and benchmark logs, helper scripts, cubin resource listings and other intermediate data).
