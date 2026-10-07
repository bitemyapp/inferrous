# Validation scope

Specification: pinned Qwen3-8B configuration and ordinary Qwen3 operators, with the explicit symmetric per-channel/per-token INT8 recipe in README.md. The independent oracle is PyTorch. Floating-point reduction order is not prescribed by Qwen3.

| Requirement | Evidence | Status |
|---|---|---|
| Fixed model/configuration, tensor sizes and hashes | Loader admission checks | Implemented; malformed bundle suite remains open |
| Request bounds, token admission, low-index argmax tie | Rust unit test | Pass |
| BF16 embedding and residual addition | Independent Torch fixtures | Exact pass |
| Norm at widths 128, 4096, 12288, including zero | Independent Torch fixtures | Numerical pass |
| Dynamic quantization, zero scale floor, positive/negative rounding ties | Independent Torch fixtures | Exact pass |
| Native nearest-even conversion | Every BF16 encoding, dense signed grid, all relevant half-integer neighbors and one million FP32 bit samples against host f64 nearest-even | Exact pass |
| INT8 products, channel/token scales, worst accumulator, partial rows/columns, native integer warp reductions | Independent CPU integer GEMM fixtures; fused projection/residual compared bitwise to separate operators | Exact pass |
| Fused gate/up and SwiGLU, distinct weights/scales, odd channel tail and full 4096x12288 shape | Separate projection/activation operators and exact intermediate/output bits plus activation maximum | Exact pass |
| TMA prefill staging, partial tensor tiles, swizzle and barrier reuse | Independent integer GEMM fixtures, mixed-length full-model hashes, checked build and GPU sanitizers | Exact pass |
| Ordinary RoPE, including position 4095 | Independent Torch trigonometry fixture | Exact pass |
| GQA, causal softmax, prefill/decode, position 4095 | Independent Torch attention fixtures | Numerical pass |
| SiLU multiplication, large positive/negative arguments | Independent Torch fixture | Numerical pass |
| GPU greedy selection, warp/block tails, signed-zero ties, repeated maxima and nonfinite rejection | Differential cases against existing CPU argmax | Exact pass |
| BF16 output head with FP32 accumulation | Independent Torch fixture, bitwise packed/original head comparison | Numerical pass; packing exact |
| Lossless packed BF16 head storage | Every u16 bit pattern round-trips on host and GPU, with mixed common/exception chunks | Exact pass |
| Full 36-layer inference and KV append over three generation steps | Independent Torch traces | Tokens pass; numerical gate FAIL (DISC-001) |
| CUDA graph/eager equality with repeated mixed-size requests, graph/attention boundaries and stop handling | Nineteen changing-input full-model requests at seventeen lengths through 4093, all graph capacities, exact complete-logit hashes; `CLEAN_MODEL_BUNDLE` integration test | Pass |
| Pinned result-slot reuse and lifetime | Repeated full-model hashes, alternating slots, eager/graph comparisons and full-model memcheck; synchronized RAII destruction | Pass |
| Failed graph recording releases capture state | CUDA driver integration test | Pass |
| Long-form quality / perplexity against original BF16 model | Not evaluated | Open |

Numerical operator gates are per-element `abs(error) <= 2e-4 + .01*abs(expected)` and relative L2 <= .005. Quantization/scales, integer projection outputs, RoPE, embedding and residual fixtures require exact equality. The full-model gate is relative L2 <= .05 and cosine >= .999 for **every** layer/norm/logit tensor, plus identical greedy tokens. The full-model gate is deliberately not weakened to accommodate the observed discrepancy. This is not a full conformance or accuracy certification.

The optimized device build elides redundant bounds checks inside the admitted fixed-shape graph; host admission remains checked. See the safety contract. Independent operators pass Compute Sanitizer memcheck, racecheck and synccheck with zero errors/warnings; 512-token prefill and three-step full-model inference also pass memcheck. Those tools do not certify full-model numerical agreement or quality.

Fixtures: `tools/operator_fixtures.py`, seed 824, PyTorch 2.13.0, generated on the benchmark workstation. Binary fixtures stay in `clean-work/operator-fixtures-64` on the benchmark workstation (the original directory is `operator-fixtures`; the extended set adds partial 65/129-row GEMMs). Their manifest contains tensor hashes. To update, generate a new directory, compare manifests and inspect numerical changes before replacing any expectations. No test updates its own expectations. Full reference traces record PyTorch/CUDA versions, model root and individual tensor hashes. `tools/compare_traces.py` exits nonzero when the full-model gate fails.
