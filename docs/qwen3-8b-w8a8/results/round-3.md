# Results summary: Round 3

Generated from the raw artifacts in `results/2026-10-03` of the development repository, which are not published. The narrative, rationale and conclusions are in [round 3 record](../optimization/round-3.md). Timing columns are medians over measured runs; decode is tokens per second for tokens 2–32; API is the public generation call in milliseconds; first step is the prefill step in milliseconds.

## Measurements

| Variant | Measured runs | Decode p5 | Decode p129 | Decode p512 | API p5 | API p129 | API p512 | First step p512 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| attentionmax | 20 | 165.7 | 161.1 | 153.5 | 195.5 | 211.5 | 247.9 | 45.72 |
| attentionreduce | 20 | 164.5 | 159.7 | 155.0 | 196.9 | 213.2 | 245.9 | 45.66 |
| fusedresidual | 20 | 166.4 | 161.5 | 156.5 | 194.8 | 211.0 | 244.0 | 45.72 |
| headlane | 20 | 165.2 | 160.5 | 155.7 | 196.0 | 212.1 | 245.3 | 45.81 |
| headpacked | 20 | 157.5 | 153.1 | 148.8 | 205.5 | 221.7 | 254.5 | 45.94 |
| large128 | 20 | 164.5 | 159.8 | 155.0 | 196.8 | 213.1 | 246.1 | 45.78 |
| largeparts | 20 | 161.6 | 157.1 | 152.5 | 200.4 | 216.3 | 251.8 | 48.24 |
| largeparts4k | 20 | 161.7 | 157.2 | 152.6 | 200.2 | 216.1 | 249.0 | 45.52 |
| round3-baseline | 20 | 162.1 | 157.5 | 152.9 | 199.8 | 215.1 | 246.9 | 43.91 |
| round3-checked | 0 | — | — | — | — | — | — | — |
| round3-control | 20 | 161.1 | 156.5 | 151.9 | 201.0 | 217.3 | 250.1 | 45.80 |
| round3-final | 20 | 166.3 | 161.6 | 156.5 | 194.8 | 210.9 | 244.1 | 45.74 |
| round3-stock-b12x | 20 | 144.4 | 143.3 | 142.9 | 252.3 | 254.1 | 254.8 | 37.51 |
| round3-stock-fa2 | 20 | 136.1 | 140.4 | 140.0 | 243.8 | 237.2 | 244.0 | 22.26 |

## Regression checks

Each check compares every token ID and complete-vocabulary logit hash against the original oracle.

| Check | Result |
|---|---|
| attentionmax-regression.json | pass (69 requests, 3 workloads) |
| attentionreduce-regression.json | pass (69 requests, 3 workloads) |
| fusedresidual-regression.json | pass (69 requests, 3 workloads) |
| headlane-regression.json | pass (69 requests, 3 workloads) |
| headpacked-regression.json | pass (69 requests, 3 workloads) |
| large128-regression.json | pass (69 requests, 3 workloads) |
| largeparts-regression.json | pass (69 requests, 3 workloads) |
| largeparts4k-regression.json | pass (69 requests, 3 workloads) |
| round3-baseline-regression.json | pass (69 requests, 3 workloads) |
| round3-checked-regression.json | pass (3 requests, 3 workloads) |
| round3-final-regression.json | pass (69 requests, 3 workloads) |

## Tests and sanitizers

| Log | Result |
|---|---|
| attentionmax-test.log | test result: ok. 7 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 17.95s |
| attentionreduce-test.log | test result: ok. 7 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 17.72s |
| fusedresidual-test.log | test result: ok. 7 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 17.69s |
| headlane-test.log | test result: ok. 7 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 17.56s |
| headpacked-test.log | test result: ok. 7 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 17.58s |
| large128-test.log | test result: ok. 7 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 17.62s |
| largeparts-test.log | test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 16.58s |
| round3-checked-test.log | test result: ok. 7 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 18.15s |
| round3-final-memcheck.log | ========= ERROR SUMMARY: 0 errors |
| round3-final-model-memcheck.log | ========= ERROR SUMMARY: 0 errors |
| round3-final-racecheck.log | ========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings) |
| round3-final-synccheck.log | ========= ERROR SUMMARY: 0 errors |

## Memory

Sampled process GPU memory in a separate instrumented run; a sampled maximum, not an exact peak.

| File | Workload | Samples | Max sampled MiB |
|---|---|---:|---:|
| round3-memory.json | p512 | 18 | 10256.0 |

## Profiles

Nsight Systems kernel summaries: the six kernels with the largest share of GPU time.

| Profile | Top kernels (share of GPU time) |
|---|---|
| headlane-profile-summary | linear_pair 29.9%, linear 21.7%, linear_large 9.1%, linear_triple 8.0%, head_packed 8.0%, attention_prefill 4.9% |
| round3-baseline-profile-summary | linear_pair 29.4%, linear 21.4%, head 10.1%, linear_large 8.7%, linear_triple 7.9%, attention_prefill 4.6% |
| round3-final-profile-summary | linear_pair 30.1%, linear_residual 22.0%, linear_large 9.3%, linear_triple 8.1%, head_packed 8.0%, attention_prefill 5.0% |

## Nsight Compute

Speed-of-light metrics per profiled kernel. These reports used a locked base clock unless noted in the records, so absolute durations are not comparable with uninstrumented benchmarks.

| Report | Kernel | Duration | DRAM throughput | SM throughput | SM frequency |
|---|---|---:|---:|---:|---:|
| round3-pair | linear_pair | 118.85 us | 88.93 % | 16.03 % | 1.00 Ghz |

## Procedures

The exact scripts used for the retained measurements and checks. `<workdir>` is the development checkout on the benchmark workstation.

### round3-final-checks.sh

```sh
#!/usr/bin/env bash
set -euo pipefail
cd <workdir>
export CUDA_OXIDE_BACKEND=<workdir>/vendor/cuda-oxide/crates/rustc-codegen-cuda/target/release/librustc_codegen_cuda.so
export RUST_TEST_THREADS=1
export CLEAN_OPERATOR_FIXTURES=$PWD/clean-work/operator-fixtures-64
export CLEAN_MODEL_BUNDLE=$PWD/clean-work/model-bundle
cp target/inferrous-gpu/release/inferrous-qwen3 clean-work/round3-final-optimized
cp target/inferrous-gpu/release/deps/inferrous_qwen3-68bfae6b70bc8921 clean-work/round3-final-tests
CLEAN_BOUNDS_CHECKS=1 crates/inferrous-qwen3/tools/cuda.sh test > clean-work/round3-checked-test.log 2>&1
CLEAN_BOUNDS_CHECKS=1 crates/inferrous-qwen3/tools/cuda.sh build > clean-work/round3-checked-build.log 2>&1
python3 crates/inferrous-qwen3/tools/bench_clean.py --binary target/inferrous-gpu/release/inferrous-qwen3 --model clean-work/model-bundle --workloads clean-work/workloads --output clean-work/round3-checked.jsonl --repeat 1 > clean-work/round3-checked.log 2>&1
python3 crates/inferrous-qwen3/tools/check_regression.py clean-work/baseline.jsonl clean-work/round3-checked.jsonl > clean-work/round3-checked-regression.json
crates/inferrous-qwen3/tools/cuda.sh build > clean-work/round3-restored-build.log 2>&1
for check in memcheck racecheck synccheck; do
    /usr/local/cuda-13.1/bin/compute-sanitizer --tool "$check" --error-exitcode=1 clean-work/round3-final-tests conformance:: --test-threads=1 > "clean-work/round3-final-$check.log" 2>&1
done
/usr/local/cuda-13.1/bin/compute-sanitizer --tool memcheck --error-exitcode=1 target/inferrous-gpu/release/inferrous-qwen3 clean-work/model-bundle --request clean-work/smoke-request.json > clean-work/round3-final-model-memcheck.log 2>&1
nsys profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none --output=clean-work/round3-final-profile target/inferrous-gpu/release/inferrous-qwen3 clean-work/model-bundle --request clean-work/workloads/p512.json --repeat 2 > clean-work/round3-final-profile.log 2>&1
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum --format csv clean-work/round3-final-profile.nsys-rep > clean-work/round3-final-profile-summary.csv 2>&1
python3 crates/inferrous-qwen3/results/2026-10-03/round3-memory.py
cargo +nightly-2026-04-03 fmt -p inferrous-qwen3 -- --check
python3 -m py_compile crates/inferrous-qwen3/tools/*.py
sha256sum clean-work/round3-final-optimized target/inferrous-gpu/release/inferrous-qwen3 > clean-work/round3-binaries.sha256
cmp -s clean-work/round3-final-optimized target/inferrous-gpu/release/inferrous-qwen3
nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,temperature.gpu,clocks.sm,clocks.mem --format=csv > clean-work/round3-final-device.csv
```

## Executable hashes

round3-binaries.sha256:

```text
22b5511e8d3a45a94b4429e12c7fb5f3a5739933eb8ba033e157b7d16dd39f91  clean-work/round3-final-optimized
22b5511e8d3a45a94b4429e12c7fb5f3a5739933eb8ba033e157b7d16dd39f91  target/inferrous-gpu/release/inferrous-qwen3
```

## Other raw artifacts

Not summarized here: 1 `.csv` file; 2 `.json` files; 1 `.log` file; 1 `.py` file (build and benchmark logs, helper scripts, cubin resource listings and other intermediate data).
