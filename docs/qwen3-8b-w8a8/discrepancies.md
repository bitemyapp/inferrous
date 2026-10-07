# Known numerical differences

## DISC-001: Full-model floating-point/quantization sensitivity

Status: INVESTIGATING; reviewed 2026-10-02. Affects `smoke-request.json`, every full-model comparison; the failing report is summarized in the [2026-10-02 results](results/rounds-1-2.md).

The independent Torch composition and cuda-oxide generate the same three tokens (`12095, 13, 576`). They fail the predeclared full-tensor numerical gate. Logit relative L2 is about 9–11%, despite identical first-token argmax. Individual operator fixtures pass substantially tighter thresholds.

Tracing the first divergence found identical layer outputs through layer 3. Layer 4 has a one-value Q/K normalization/RoPE rounding difference, followed by small attention differences and larger MLP differences after dynamic INT8 quantization. Ordinary FP32 reductions, reciprocal square root, exponential approximations and BF16 rounding need not agree across implementations; repeated quantization can amplify those differences. This explains an observed onset, not proof that all discrepancies are harmless. Do not claim model quality or cross-engine numerical equivalence from the performance measurements.

Optimization regression checks compare *all* vocabulary logit hashes and token IDs against the original cuda-oxide baseline, in addition to independent operator fixtures. They do not replace this separate, still-failing oracle gate.

## DISC-002: Fused floating-point operator boundaries

Status: ACCEPTED as an explicit baseline convention; reviewed 2026-10-02. Affects comparisons against eager Hugging Face Qwen3, distinct from DISC-001.

Norm multiplies the gain before the final BF16 cast; RoPE uses FP32 coefficients and one output cast; softmax probabilities remain FP32 until PV is cast; SiLU and the up projection multiply before the final BF16 cast. These are ordinary fused inference semantics. Eager frameworks can materialize additional BF16 intermediates. The independent composition documents these same boundaries. Bitwise equality to arbitrary eager/serving frameworks is not a requirement.
