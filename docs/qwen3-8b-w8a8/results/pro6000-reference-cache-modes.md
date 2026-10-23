# Torch reference cache-mode sensitivity

The unchanged Torch composition produces different tensors under cached decode and full-prefix recomputation on identical token histories. Both modes run on `ssh wx-workstation` with the same pinned weights, Torch `2.13.0+cu130`, CUDA runtime 13.0 and declared arithmetic expressions. This confirms execution-mode sensitivity in the reference itself. It does not resolve the production engine's failing numerical gate, justify changing its thresholds, or prove that matching the existing canonical cached reference is impossible.

## Comparison boundary

The maintained `reference.py` remains unchanged, SHA-256 `ceac7c4aaa5b4172a4569d1923863a04db66aa6c84238c06ac010b54d08cec38`. The isolated diagnostic changes only execution: before every step it clears K/V, resets position to zero and submits the entire prompt plus the preceding cached-reference tokens. Those supplied tokens keep the histories identical even if recomputation selects a different argmax. All original tensor expressions, model-loading integrity checks and output boundaries remain intact. Full recomputed tensors are preserved, rather than overwriting them with slices.

The five-token request and the existing 513-token request each have three outputs and no stop tokens. At step zero, all fifty-one tensors match the cached reference byte for byte. For steps one and two, each non-logit comparison selects the last logical row of the recomputed tensor and compares it with the corresponding cached-decode tensor; logits compare all 151,936 entries. The audit requires the complete 153-tensor schema, declared shapes, file lengths, hashes and finite values. Its reported tolerance checks use the original relative-L2 ≤ 0.05 and cosine ≥ 0.999 limits. They are diagnostic comparisons between reference modes, not production acceptance results.

## Results

| Prompt tokens | Output step | Logit relative L2 | Logit cosine | Cached argmax | Recomputed argmax on the same history |
|---|---:|---:|---:|---:|---:|
| 5 | 0 | 0 | 1.000000000 | 12095 | 12095 |
| 5 | 1 | 0.050195633 | 0.999425807 | 13 | 13 |
| 5 | 2 | 0.044282620 | 0.999030084 | 576 | 576 |
| 513 | 0 | 0 | 1.000000000 | 198 | 198 |
| 513 | 1 | 0.072223603 | 0.997939819 | 197 | 200 |
| 513 | 2 | 0.170176795 | 0.985638504 | 197 | 197 |

For five tokens, 130 of 153 tensors are exact and six fail one or both numerical limits. The first inexact saved layer output is layer 27 at step one, followed by layer 28 at step two. The first tolerance failure in both steps is layer 33. For 513 tokens, only 54 tensors are exact and 78 fail the numerical limits. The first differing layer outputs are layer one at step one and layer two at step two; the first failed layer outputs are layers five and four respectively.

The long request's step-two comparison still uses the cached history ending in token 197. It does not follow the recomputation argmax of 200 from step one. Thus `[198, 200, 197]` is a list of decisions on controlled histories, not a free-running recomputed generation sequence. No inference about later divergent histories is made.

Fresh cached controls then rerun the unchanged maintained reference for both requests. All 306 tensors, metadata and greedy tokens reproduce the earlier cached artifacts exactly. Together with the exact recomputation prefill controls, this rules out a stale cached trace or a changed model/runtime as an explanation for these observed differences. The prior [operator investigations](../discrepancies.md#disc-001-full-model-floating-pointquantization-sensitivity) attribute shape-dependent reductions, QK order and other sensitive boundaries; this experiment does not separately attribute every full-model difference.

## Acceptance status and evidence

The original numerical gate remains mandatory. A separate fixed-policy diagnostic has passed independent reference and batch/cache invariance checks, but has not replaced this gate or been retained. Selecting a new acceptance contract requires an explicit decision; a reference's self-disagreement alone is not authorization to waive an existing requirement. Native-capacity, integrated sanitizers, paired performance and the complete fastest-stock matrix would still be required for any new retained arithmetic path.

Remote artifacts are under `clean-work/pro6000-baseline/original-reference-recompute-v1/`: summary SHA-256 `48ce0f7e5447838013a717ee81fa7b4d4a5393e3ed7e640dc07f91a0533e0960`, identity `3b5f18fea31c6b630952e2be3aebc0c46ccb15f0a26ccb618b9b6f9db46d7ead`, diagnostic source `7e02812b98b2bc3952fb4317956cee8c8e8f496baa1a3f488a67406ebb112c2e`, runner `ea05b05683eed90c952577e2e4ee08deaf8aba387d0f682637774d20a6045bfd`, five-token comparison `0985134dca4a955cf6e54d8d3e558fc5fb743bf4168b2480fc1aa5aa42b6a85a`, and 513-token comparison `4420982c23c2c34a21110ed2ff2b0cf53475ad48ecefd14e1bd79918927e86cb`.

Fresh cached controls are under `original-reference-cache-control-v1/`: summary `ac5433083e85a67565472f166a0c7c92aedad6522081b0abc123c266702b3483`, identity `62e47dd3234d4e168732725fb2cda15d819a422aaf789d7587ed40b360d4a0fc`, and runner `ddfd08e90a67de4d994ebf70dadf88552b287d2eee4b118c076b32e87b0cfd91`. All four GPU commands exit successfully. Source identities, execution receipts, logs, comparison reports and every generated tensor hash were rechecked after completion. Model root remains `e51d2517784b99de3802110542e831061489bf00df33db1b412ba3b3d01d21d7`.
