# Results summary: Round 4

Generated from the raw artifacts in `results/2026-10-03/round4` of the development repository, which are not published. The narrative, rationale and conclusions are in [round 4 record](../optimization/round-4.md). Timing columns are medians over measured runs; decode is tokens per second for tokens 2–32; API is the public generation call in milliseconds; first step is the prefill step in milliseconds.

## Measurements

| Variant | Measured runs | Decode p5 | Decode p129 | Decode p512 | API p5 | API p129 | API p512 | First step p512 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cacheglobal | 20 | 167.3 | 162.7 | 157.6 | 193.7 | 209.4 | 248.0 | 51.02 |
| cluster-checks-checked | 0 | — | — | — | — | — | — | — |
| combined | 20 | 166.9 | 162.2 | 157.3 | 194.2 | 210.4 | 242.0 | 44.69 |
| control-131 | 20 | 166.3 | 161.2 | 156.9 | 196.8 | 211.7 | 243.8 | 45.99 |
| control-ptx | 20 | 166.3 | 160.8 | 156.9 | 196.4 | 211.4 | 245.6 | 47.82 |
| fused-gate-control | 20 | 167.1 | 162.2 | 157.3 | 194.0 | 210.4 | 243.4 | 46.05 |
| fused-gate | 20 | 167.9 | 162.9 | 158.0 | 193.1 | 209.4 | 242.0 | 45.52 |
| gate2 | 20 | 152.6 | 148.6 | 144.5 | 211.6 | 227.8 | 260.7 | 45.93 |
| prefillpack | 20 | 166.9 | 162.3 | 157.2 | 194.1 | 210.0 | 242.5 | 45.02 |
| round4-adjacent | 20 | 166.5 | 161.7 | 156.7 | 194.6 | 210.7 | 243.7 | 45.60 |
| round4-baseline | 20 | 168.4 | 163.4 | 158.4 | 192.4 | 207.7 | 239.4 | 43.49 |
| round4-candidate-a | 20 | 167.2 | 162.3 | 157.5 | 193.8 | 210.2 | 241.5 | 44.48 |
| round4-candidate-b | 20 | 166.8 | 162.1 | 157.1 | 194.2 | 210.5 | 242.4 | 44.72 |
| round4-checked | 0 | — | — | — | — | — | — | — |
| round4-control-a | 20 | 166.7 | 161.9 | 156.8 | 194.4 | 210.5 | 243.5 | 45.56 |
| round4-control-b | 20 | 165.3 | 160.6 | 155.7 | 195.9 | 212.4 | 245.7 | 46.28 |
| round4-control | 40 | 165.9 | 161.4 | 156.3 | 195.2 | 211.3 | 244.5 | 45.91 |
| round4-final | 40 | 167.1 | 162.2 | 157.3 | 194.0 | 210.4 | 241.9 | 44.61 |
| tma-131 | 20 | 165.9 | 161.0 | 156.8 | 197.1 | 211.9 | 242.8 | 44.90 |
| tma-asm | 20 | 165.5 | 160.9 | 156.0 | 195.8 | 211.9 | 243.8 | 44.82 |
| tma-pipe | 20 | 166.2 | 161.5 | 156.3 | 194.9 | 211.1 | 244.2 | 45.57 |
| tma-ptx | 20 | 167.1 | 161.6 | 157.6 | 195.6 | 210.0 | 242.5 | 45.55 |
| tma32 | 20 | 166.3 | 161.5 | 156.5 | 194.8 | 211.0 | 250.3 | 51.97 |
| unroll4 | 20 | 167.6 | 162.6 | 157.8 | 193.4 | 209.8 | 242.6 | 45.85 |

## Regression checks

Each check compares every token ID and complete-vocabulary logit hash against the original oracle.

| Check | Result |
|---|---|
| cacheglobal-regression.json | pass (69 requests, 3 workloads) |
| cluster-checks-checked-regression.json | pass (3 requests, 3 workloads) |
| combined-regression.json | pass (69 requests, 3 workloads) |
| control-131-regression.json | pass (69 requests, 3 workloads) |
| control-ptx-regression.json | pass (69 requests, 3 workloads) |
| fused-gate-control-regression.json | pass (69 requests, 3 workloads) |
| fused-gate-regression.json | pass (69 requests, 3 workloads) |
| gate2-regression.json | pass (69 requests, 3 workloads) |
| prefillpack-regression.json | pass (69 requests, 3 workloads) |
| round4-adjacent-regression.json | pass (69 requests, 3 workloads) |
| round4-baseline-regression.json | pass (69 requests, 3 workloads) |
| round4-candidate-a-regression.json | pass (69 requests, 3 workloads) |
| round4-candidate-b-regression.json | pass (69 requests, 3 workloads) |
| round4-checked-regression.json | pass (3 requests, 3 workloads) |
| round4-control-a-regression.json | pass (69 requests, 3 workloads) |
| round4-control-b-regression.json | pass (69 requests, 3 workloads) |
| round4-final-regression.json | pass (138 requests, 3 workloads) |
| tma-131-regression.json | pass (69 requests, 3 workloads) |
| tma-asm-regression.json | pass (69 requests, 3 workloads) |
| tma-pipe-regression.json | pass (69 requests, 3 workloads) |
| tma-ptx-regression.json | pass (69 requests, 3 workloads) |
| tma32-regression.json | pass (69 requests, 3 workloads) |
| unroll4-regression.json | pass (69 requests, 3 workloads) |

## Tests and sanitizers

| Log | Result |
|---|---|
| cacheglobal-test.log | test result: ok. 7 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 17.83s |
| cluster-checks-checked-test.log | test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 18.29s |
| cluster-checks-final-memcheck.log | ========= ERROR SUMMARY: 0 errors |
| cluster-checks-final-model-memcheck.log | ========= ERROR SUMMARY: 0 errors |
| cluster-checks-final-racecheck.log | ========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings) |
| cluster-checks-final-synccheck.log | ========= ERROR SUMMARY: 0 errors |
| cluster-checks-optimized-test.log | test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 17.80s |
| combined-test.log | test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 17.87s |
| fused-gate-test.log | test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 17.78s |
| gate2-test.log | test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 17.83s |
| prefillpack-test.log | test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 19.01s |
| round4-checked-test.log | test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 18.11s |
| round4-final-memcheck.log | ========= ERROR SUMMARY: 0 errors |
| round4-final-model-memcheck.log | ========= ERROR SUMMARY: 0 errors |
| round4-final-racecheck.log | ========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings) |
| round4-final-synccheck.log | ========= ERROR SUMMARY: 0 errors |
| round4-optimized-test.log | test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 17.65s |
| tma-131-test.log | test result: ok. 7 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 17.49s |
| tma-asm-test.log | test result: ok. 7 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 17.63s |
| tma-pipe-test.log | test result: ok. 7 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 17.62s |
| tma-ptx-test.log | test result: ok. 7 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 17.95s |
| tma32-test.log | test result: ok. 7 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 18.18s |
| unroll4-test.log | test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 17.75s |

## Memory

Sampled process GPU memory in a separate instrumented run; a sampled maximum, not an exact peak.

| File | Workload | Samples | Max sampled MiB |
|---|---|---:|---:|
| cluster-checks-memory.json | p512 | 17 | 14468.0 |
| round4-memory.json | p512 | 17 | 10256.0 |

## Profiles

Nsight Systems kernel summaries: the six kernels with the largest share of GPU time.

| Profile | Top kernels (share of GPU time) |
|---|---|
| cluster-checks-final-profile-summary | linear_swiglu 30.5%, linear_residual 22.2%, linear_large 8.8%, linear_triple 8.2%, head_packed 8.1%, attention_prefill 5.0% |
| round4-baseline-profile-summary | linear_pair 30.4%, linear_residual 22.2%, linear_large 9.0%, linear_triple 8.2%, head_packed 8.0%, attention_prefill 4.8% |
| round4-final-profile-summary | linear_swiglu 30.6%, linear_residual 22.2%, linear_large 8.8%, linear_triple 8.2%, head_packed 8.1%, attention_prefill 5.0% |

## Procedures

The exact scripts used for the retained measurements and checks. `<workdir>` is the development checkout on the benchmark workstation.

### final-benchmarks.sh

```sh
#!/usr/bin/env bash
set -euo pipefail
cd <workdir>
# A/B/B/A counters temperature and clock drift without changing machine policy.
for label in control-a candidate-a candidate-b control-b; do
    binary=clean-work/round4-baseline
    if [[ "$label" == candidate-* ]]; then binary=clean-work/round4-final-optimized; fi
    python3 crates/inferrous-qwen3/tools/bench_clean.py --binary "$binary" --model clean-work/model-bundle --workloads clean-work/workloads --output "clean-work/round4-$label.jsonl" > "clean-work/round4-$label.log" 2>&1
    python3 crates/inferrous-qwen3/tools/check_regression.py clean-work/baseline.jsonl "clean-work/round4-$label.jsonl" > "clean-work/round4-$label-regression.json"
done
cat clean-work/round4-control-a.jsonl clean-work/round4-control-b.jsonl > clean-work/round4-control.jsonl
cat clean-work/round4-candidate-a.jsonl clean-work/round4-candidate-b.jsonl > clean-work/round4-final.jsonl
python3 crates/inferrous-qwen3/tools/summarize.py clean-work/round4-control-a.jsonl clean-work/round4-candidate-a.jsonl clean-work/round4-candidate-b.jsonl clean-work/round4-control-b.jsonl clean-work/round4-control.jsonl clean-work/round4-final.jsonl > clean-work/round4-summary.json
python3 crates/inferrous-qwen3/tools/check_regression.py clean-work/baseline.jsonl clean-work/round4-final.jsonl > clean-work/round4-final-regression.json
```

### final-checks.sh

```sh
#!/usr/bin/env bash
set -euo pipefail
cd <workdir>
export CUDA_OXIDE_BACKEND=<workdir>/vendor/cuda-oxide/crates/rustc-codegen-cuda/target/release/librustc_codegen_cuda.so
export RUST_TEST_THREADS=1
export CLEAN_OPERATOR_FIXTURES=$PWD/clean-work/operator-fixtures-64
export CLEAN_MODEL_BUNDLE=$PWD/clean-work/model-bundle
crates/inferrous-qwen3/tools/cuda.sh test > clean-work/round4-optimized-test.log 2>&1
crates/inferrous-qwen3/tools/cuda.sh build > clean-work/round4-optimized-build.log 2>&1
cp target/inferrous-gpu/release/inferrous-qwen3 clean-work/round4-final-optimized
cp target/inferrous-gpu/release/deps/inferrous_qwen3-68bfae6b70bc8921 clean-work/round4-final-tests
CLEAN_BOUNDS_CHECKS=1 crates/inferrous-qwen3/tools/cuda.sh test > clean-work/round4-checked-test.log 2>&1
CLEAN_BOUNDS_CHECKS=1 crates/inferrous-qwen3/tools/cuda.sh build > clean-work/round4-checked-build.log 2>&1
python3 crates/inferrous-qwen3/tools/bench_clean.py --binary target/inferrous-gpu/release/inferrous-qwen3 --model clean-work/model-bundle --workloads clean-work/workloads --output clean-work/round4-checked.jsonl --repeat 1 > clean-work/round4-checked.log 2>&1
python3 crates/inferrous-qwen3/tools/check_regression.py clean-work/baseline.jsonl clean-work/round4-checked.jsonl > clean-work/round4-checked-regression.json
crates/inferrous-qwen3/tools/cuda.sh build > clean-work/round4-restored-build.log 2>&1
for check in memcheck racecheck synccheck; do
    /usr/local/cuda-13.1/bin/compute-sanitizer --tool "$check" --error-exitcode=1 clean-work/round4-final-tests conformance:: --test-threads=1 > "clean-work/round4-final-$check.log" 2>&1
done
/usr/local/cuda-13.1/bin/compute-sanitizer --tool memcheck --error-exitcode=1 target/inferrous-gpu/release/inferrous-qwen3 clean-work/model-bundle --request clean-work/round4-memcheck-request.json > clean-work/round4-final-model-memcheck.log 2>&1
nsys profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none --output=clean-work/round4-final-profile target/inferrous-gpu/release/inferrous-qwen3 clean-work/model-bundle --request clean-work/workloads/p512.json --repeat 2 > clean-work/round4-final-profile.log 2>&1
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum --format csv clean-work/round4-final-profile.nsys-rep > clean-work/round4-final-profile-summary.csv 2>&1
python3 crates/inferrous-qwen3/results/2026-10-03/round4/memory.py
cargo +nightly-2026-04-03 fmt -p inferrous-qwen3 -- --check
python3 -m py_compile crates/inferrous-qwen3/tools/*.py
sha256sum clean-work/round4-final-optimized target/inferrous-gpu/release/inferrous-qwen3 > clean-work/round4-binaries.sha256
cmp -s clean-work/round4-final-optimized target/inferrous-gpu/release/inferrous-qwen3
nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,temperature.gpu,clocks.sm,clocks.mem --format=csv > clean-work/round4-final-device.csv
```

## Executable hashes

cluster-checks-binaries.sha256:

```text
591d12c2a26035ee15213f0498f139a69781963a526255c1a68dc8d2f833f7b9  clean-work/round4-final-optimized
591d12c2a26035ee15213f0498f139a69781963a526255c1a68dc8d2f833f7b9  target/inferrous-gpu/release/inferrous-qwen3
```

round4-binaries.sha256:

```text
96f0f0234b2f2abfe348f5675b4be0f42130ba8a47cde1f9ed7bf18c9d630b0f  clean-work/round4-final-optimized
96f0f0234b2f2abfe348f5675b4be0f42130ba8a47cde1f9ed7bf18c9d630b0f  target/inferrous-gpu/release/inferrous-qwen3
```

## Other raw artifacts

Not summarized here: 2 `.csv` files; 2 `.json` files; 31 `.log` files; 3 `.py` files; 4 `.txt` files (build and benchmark logs, helper scripts, cubin resource listings and other intermediate data).
