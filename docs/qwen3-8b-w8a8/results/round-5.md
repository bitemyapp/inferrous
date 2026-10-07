# Results summary: Round 5

Generated from the raw artifacts in `results/2026-10-03/round5` of the development repository, which are not published. The narrative, rationale and conclusions are in [round 5 record](../optimization/round-5.md). Timing columns are medians over measured runs; decode is tokens per second for tokens 2–32; API is the public generation call in milliseconds; first step is the prefill step in milliseconds.

## Measurements

| Variant | Measured runs | Decode p5 | Decode p129 | Decode p512 | API p5 | API p129 | API p512 | First step p512 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| round5-baseline | 20 | 169.4 | 164.4 | 159.3 | 191.4 | 206.7 | 237.4 | 42.54 |
| round5-bf16 | 20 | 170.2 | 165.1 | 160.0 | 190.4 | 206.7 | 237.8 | 43.78 |
| round5-burst | 20 | 168.6 | 163.6 | 158.7 | 192.2 | 208.5 | 240.6 | 44.88 |
| round5-checked | 0 | — | — | — | — | — | — | — |
| round5-control-a | 20 | 168.6 | 163.6 | 158.7 | 192.3 | 208.2 | 239.4 | 43.81 |
| round5-control | 40 | 167.1 | 162.3 | 157.3 | 193.9 | 210.2 | 241.8 | 44.61 |
| round5-exact | 20 | 169.6 | 164.7 | 159.6 | 191.1 | 207.1 | 238.5 | 43.95 |
| round5-final | 40 | 169.6 | 164.6 | 159.5 | 191.1 | 207.5 | 238.9 | 44.27 |
| round5-mma | 20 | 111.3 | 109.0 | 106.2 | 286.9 | 301.8 | 334.1 | 41.83 |
| round5-native | 20 | 170.5 | 165.5 | 160.3 | 190.1 | 206.1 | 237.4 | 43.77 |
| round5-packed | 20 | 166.4 | 161.8 | 156.8 | 194.7 | 210.3 | 242.2 | 44.18 |
| round5-paired-candidate-a | 20 | 169.8 | 164.7 | 159.6 | 190.9 | 207.3 | 238.8 | 44.23 |
| round5-paired-candidate-b | 20 | 169.5 | 164.4 | 159.3 | 191.3 | 207.7 | 239.3 | 44.40 |
| round5-paired-control-a | 20 | 167.6 | 162.6 | 157.8 | 193.4 | 209.8 | 241.1 | 44.38 |
| round5-paired-control-b | 20 | 166.6 | 161.9 | 156.9 | 194.5 | 210.8 | 242.7 | 44.86 |
| round5-pinned | 20 | 169.3 | 164.3 | 159.2 | 191.5 | 207.4 | 239.0 | 44.05 |
| round5-redux | 20 | 171.0 | 165.8 | 160.7 | 189.6 | 205.8 | 237.1 | 43.96 |
| round5-stock-b12x | 20 | 144.4 | 143.3 | 142.9 | 252.4 | 254.0 | 254.8 | 37.54 |
| round5-stock-fa2 | 20 | 136.1 | 140.4 | 140.2 | 244.6 | 237.4 | 243.3 | 22.00 |
| round5-tma64 | 20 | 168.2 | 163.2 | 158.3 | 192.7 | 209.0 | 241.5 | 45.41 |

## Regression checks

Each check compares every token ID and complete-vocabulary logit hash against the original oracle.

| Check | Result |
|---|---|
| round5-checked-regression.json | pass (3 requests, 3 workloads) |
| round5-final-regression.json | pass (138 requests, 3 workloads) |
| round5-paired-candidate-a-regression.json | pass (69 requests, 3 workloads) |
| round5-paired-candidate-b-regression.json | pass (69 requests, 3 workloads) |
| round5-paired-control-a-regression.json | pass (69 requests, 3 workloads) |
| round5-paired-control-b-regression.json | pass (69 requests, 3 workloads) |

## Tests and sanitizers

| Log | Result |
|---|---|
| round5-bf16-tests.log | test result: ok. 10 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 17.98s |
| round5-burst-tests.log | test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 19.89s |
| round5-checked-test.log | test result: ok. 9 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 18.22s |
| round5-exact-tests.log | test result: ok. 10 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 18.04s |
| round5-final-memcheck.log | ========= ERROR SUMMARY: 0 errors |
| round5-final-model-memcheck.log | ========= ERROR SUMMARY: 0 errors |
| round5-final-racecheck.log | ========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings) |
| round5-final-synccheck.log | ========= ERROR SUMMARY: 0 errors |
| round5-optimized-test.log | test result: ok. 9 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 17.86s |
| round5-redux-tests.log | test result: ok. 9 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 17.79s |
| round5-tma64-tests.log | test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.90s |

## Memory

Sampled process GPU memory in a separate instrumented run; a sampled maximum, not an exact peak.

| File | Workload | Samples | Max sampled MiB |
|---|---|---:|---:|
| round5-memory.json | p512 | 17 | 10256.0 |

## Profiles

Nsight Systems kernel summaries: the six kernels with the largest share of GPU time.

| Profile | Top kernels (share of GPU time) |
|---|---|
| round5-baseline-profile-summary | linear_swiglu 30.8%, linear_residual 22.4%, linear_large 8.7%, linear_triple 8.2%, head_packed 8.0%, attention_prefill 4.8% |
| round5-final-profile-summary | linear_swiglu 30.8%, linear_residual 22.4%, linear_large 9.0%, linear_triple 8.2%, head_packed 8.2%, attention_prefill 5.1% |
| round5-packed-profile-summary | linear_swiglu_packed 31.0%, linear_residual 22.2%, linear_large 8.8%, linear_triple 8.2%, head_packed 8.0%, attention_prefill 4.9% |

## Procedures

The exact scripts used for the retained measurements and checks. `<workdir>` is the development checkout on the benchmark workstation.

### final-benchmarks.sh

```sh
#!/usr/bin/env bash
set -euo pipefail
cd <workdir>
# A/B/B/A counters temperature and clock drift without changing machine policy.
for label in control-a candidate-a candidate-b control-b; do
    binary=clean-work/round5-baseline
    if [[ "$label" == candidate-* ]]; then binary=clean-work/round5-final-optimized; fi
    python3 crates/inferrous-qwen3/tools/bench_clean.py --binary "$binary" --model clean-work/model-bundle --workloads clean-work/workloads --output "clean-work/round5-paired-$label.jsonl" > "clean-work/round5-paired-$label.log" 2>&1
    python3 crates/inferrous-qwen3/tools/check_regression.py clean-work/baseline.jsonl "clean-work/round5-paired-$label.jsonl" > "clean-work/round5-paired-$label-regression.json"
done
cat clean-work/round5-paired-control-a.jsonl clean-work/round5-paired-control-b.jsonl > clean-work/round5-control.jsonl
cat clean-work/round5-paired-candidate-a.jsonl clean-work/round5-paired-candidate-b.jsonl > clean-work/round5-final.jsonl
python3 crates/inferrous-qwen3/tools/summarize.py clean-work/round5-paired-control-a.jsonl clean-work/round5-paired-candidate-a.jsonl clean-work/round5-paired-candidate-b.jsonl clean-work/round5-paired-control-b.jsonl clean-work/round5-control.jsonl clean-work/round5-final.jsonl > clean-work/round5-summary.json
python3 crates/inferrous-qwen3/tools/check_regression.py clean-work/baseline.jsonl clean-work/round5-final.jsonl > clean-work/round5-final-regression.json
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
crates/inferrous-qwen3/tools/cuda.sh test > clean-work/round5-optimized-test.log 2>&1
crates/inferrous-qwen3/tools/cuda.sh build > clean-work/round5-optimized-build.log 2>&1
cp target/inferrous-gpu/release/inferrous-qwen3 clean-work/round5-final-optimized
cp target/inferrous-gpu/release/deps/inferrous_qwen3-68bfae6b70bc8921 clean-work/round5-final-tests
CLEAN_BOUNDS_CHECKS=1 crates/inferrous-qwen3/tools/cuda.sh test > clean-work/round5-checked-test.log 2>&1
CLEAN_BOUNDS_CHECKS=1 crates/inferrous-qwen3/tools/cuda.sh build > clean-work/round5-checked-build.log 2>&1
python3 crates/inferrous-qwen3/tools/bench_clean.py --binary target/inferrous-gpu/release/inferrous-qwen3 --model clean-work/model-bundle --workloads clean-work/workloads --output clean-work/round5-checked.jsonl --repeat 1 > clean-work/round5-checked.log 2>&1
python3 crates/inferrous-qwen3/tools/check_regression.py clean-work/baseline.jsonl clean-work/round5-checked.jsonl > clean-work/round5-checked-regression.json
crates/inferrous-qwen3/tools/cuda.sh build > clean-work/round5-restored-build.log 2>&1
for check in memcheck racecheck synccheck; do
    /usr/local/cuda-13.1/bin/compute-sanitizer --tool "$check" --error-exitcode=1 clean-work/round5-final-tests conformance:: --test-threads=1 > "clean-work/round5-final-$check.log" 2>&1
done
/usr/local/cuda-13.1/bin/compute-sanitizer --tool memcheck --error-exitcode=1 target/inferrous-gpu/release/inferrous-qwen3 clean-work/model-bundle --request clean-work/round4-memcheck-request.json > clean-work/round5-final-model-memcheck.log 2>&1
nsys profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none --output=clean-work/round5-final-profile target/inferrous-gpu/release/inferrous-qwen3 clean-work/model-bundle --request clean-work/workloads/p512.json --repeat 2 > clean-work/round5-final-profile.log 2>&1
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum --format csv clean-work/round5-final-profile.nsys-rep > clean-work/round5-final-profile-summary.csv 2>&1
python3 crates/inferrous-qwen3/results/2026-10-03/round5/memory.py
cargo +nightly-2026-04-03 fmt -p inferrous-qwen3 -- --check
python3 -m py_compile crates/inferrous-qwen3/tools/*.py
sha256sum clean-work/round5-final-optimized target/inferrous-gpu/release/inferrous-qwen3 > clean-work/round5-binaries.sha256
cmp -s clean-work/round5-final-optimized target/inferrous-gpu/release/inferrous-qwen3
nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,temperature.gpu,clocks.sm,clocks.mem --format=csv > clean-work/round5-final-device.csv
```

### stock-benchmarks.sh

```sh
#!/usr/bin/env bash
set -euo pipefail
cd <workdir>
export CUDA_HOME=/usr/local/cuda-13.1 VLLM_USE_FLASHINFER_SAMPLER=0
for backend in FLASH_ATTN B12X; do
    label=fa2
    if [[ "$backend" == B12X ]]; then label=b12x; fi
    clean-work/vllm-venv/bin/python crates/inferrous-qwen3/tools/bench_engines.py vllm \
        --model clean-work/model-bundle/hf --requests clean-work/workloads/requests.json \
        --quantization humming --graphs-without-compile --attention-backend "$backend" \
        --output "clean-work/round5-stock-$label.jsonl" --repeat 23 \
        > "clean-work/round5-stock-$label.log" 2>&1
done
python3 crates/inferrous-qwen3/tools/summarize.py clean-work/round5-stock-fa2.jsonl clean-work/round5-stock-b12x.jsonl > clean-work/round5-stock-summary.json
```

## Executable hashes

round5-binaries.sha256:

```text
e01c85e654b89913b83d0b7bd73274a78c52b015ee45949cbb0d8b2eb8ac77c2  clean-work/round5-final-optimized
e01c85e654b89913b83d0b7bd73274a78c52b015ee45949cbb0d8b2eb8ac77c2  target/inferrous-gpu/release/inferrous-qwen3
```

## Rejected implementations

Patches of rejected experiments, summarized by size. Each is described, with its measurements, in the round's optimization record.

| Patch | Files | Insertions | Deletions |
|---|---:|---:|---:|
| rejected-burst.patch | 2 | 117 | 15 |
| rejected-exact.patch | 3 | 145 | 21 |
| rejected-mma.patch | 2 | 117 | 9 |
| rejected-packed.patch | 3 | 159 | 3 |
| rejected-tma64.patch | 2 | 31 | 30 |

## Other raw artifacts

Not summarized here: 1 `.csv` file; 2 `.json` files; 37 `.log` files; 1 `.py` file (build and benchmark logs, helper scripts, cubin resource listings and other intermediate data).
