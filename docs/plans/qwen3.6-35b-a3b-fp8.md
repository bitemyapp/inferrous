# Implementation plan: Qwen3.6-35B-A3B FP8 on one RTX PRO 6000

Prepared 2026-10-06. The target is the official `Qwen/Qwen3.6-35B-A3B-FP8` checkpoint (Apache-2.0, 37.5 GB). Why this model was chosen is in [single-GPU model candidates](../research/single-gpu-model-candidates.md); the broader service roadmap is the [production plan](production.md).

The goal is a second model in the cuda-oxide engine that keeps the Qwen3-8B W8A8 engine's discipline (bit-level regression oracle, independent operator fixtures, safety arguments, A/B/B/A timing) and beats vLLM's single-request decode on the same card. Text first; vision later.

## 1. Architecture (verified from `config.json` and safetensors headers)

| Property | Value |
|---|---|
| Layers | 40. Pattern `LLLF` ×10: 30 Gated DeltaNet linear-attention layers and 10 gated full-attention layers (`full_attention_interval` 4) |
| Hidden size / vocabulary | 2048 / 248,320; untied BF16 embedding and LM head, each `[248320, 2048]` (1.02 GB) |
| MoE (every layer) | Router `[256, 2048]` BF16; 256 routed experts, 8 per token, intermediate size 512; one shared expert (512) with a sigmoid `shared_expert_gate` `[1, 2048]` |
| Gated DeltaNet layers | `in_proj_qkv` `[8192, 2048]` (q 16×128, k 16×128, v 32×128); `in_proj_z` `[4096, 2048]`; `in_proj_a` and `in_proj_b` `[32, 2048]` BF16; causal `conv1d` `[8192, 1, 4]`; `A_log` and `dt_bias` `[32]`; gated head norm `[128]`; `out_proj` `[2048, 4096]`. Recurrent state in FP32 (`mamba_ssm_dtype`) |
| Full-attention layers | 16 query heads and 2 KV heads, `head_dim` 256. `q_proj` `[8192, 2048]` produces query and output gate (`attn_output_gate`); q/k RMSNorm `[256]`; partial RoPE (`partial_rotary_factor` 0.25, 64 of 256 dims), θ = 10⁷, interleaved multimodal-RoPE sections `[11, 11, 10]` (equivalent to 1-D positions for text); `o_proj` `[2048, 4096]` |
| MTP | One layer (`mtp.fc` `[2048, 4096]` BF16 plus a full decoder layer with MoE), shares the LM head |
| Context | 262,144 positions (`max_position_embeddings`) |
| Weights | FP8 E4M3 with BF16 `weight_scale_inv` per 128×128 block (for example `[8192, 2048]` → `[64, 16]`) for every large projection, every expert and the shared expert. Router, gating parameters, conv, norms, embedding and head are BF16. Activations: `activation_scheme: dynamic` |
| Generation defaults | Sampling temperature 1.0, top-k 20, top-p 0.95; EOS IDs 248046 and 248044 |

Verify these against the reference modeling code (Transformers `qwen3_5_moe` and vLLM) before relying on them:

- Whether layer norms are zero-centered (`x̂ · (1 + w)`, as in Qwen3-Next).
- The exact Gated DeltaNet discretization and update: `g = -exp(A_log) · softplus(a + dt_bias)`, β = sigmoid(b), L2-normalized q/k, and the delta rule S ← α·S·(I − β·k·kᵀ) + β·v·kᵀ.
- The conv1d activation.
- The order of the output gate and o_proj.
- Router normalization (softmax, top-8, renormalize).

## 2. Decode traffic and a performance target

Per generated token at batch 1 (estimate from tensor shapes):

| Component | Bytes per token |
|---|---:|
| 30 Gated DeltaNet layers: qkv + z + out projections (FP8) plus BF16 gates | 30 × 33.6 MB ≈ 1.01 GB |
| 10 full-attention layers: q (with gate) + k + v + o (FP8) | 10 × 27.3 MB ≈ 0.27 GB |
| 40 MoE blocks: router (BF16) + shared expert + 8 routed experts (FP8) | 40 × 29.4 MB ≈ 1.18 GB |
| LM head (BF16) | 1.02 GB |
| DeltaNet state read and write (FP32, 32 heads × 128 × 128 per layer) | ≈ 0.13 GB |
| KV cache (10 layers × 2 KV heads × 256 × K/V × BF16 ≈ 20 KB per token) | 0.08 GB at 4K context |
| **Total** | **≈ 3.7 GB** |

At about 1.6 TB/s sustained, that is a ceiling of about **430 tok/s**. vLLM measures 196 tok/s plain and 219 tok/s with MTP on this card (single request, 1K context). **Target: beat 196 tok/s without speculation in the first optimized version, and aim for 300+.**

Unlike Qwen3-8B, where 7.9 GB per token made decode bandwidth-bound, this model is latency-bound at batch 1. A naive graph runs 10–15 kernels per layer (norm, projections, conv, gates, recurrent step, gated norm, router, top-k, experts, combine, residual): roughly 500 launches per token. The round-6 tools (programmatic launches, weight prefetch into idle windows, hoisted cold loads) carry over directly. Fusing within each layer is the main lever. The LM head is 28% of the bytes, and the existing lossless 12-bit packed BF16 head would cut it to about 0.77 GB without changing a single logit.

## 3. Decisions before writing kernels

1. **FP8 linear-layer contract.** The reference is W8A8: dynamic per-token, per-128-group activation quantization to E4M3, block-scaled weights, FP32 accumulation. Options:
   - **Match the reference W8A8 semantics with fixed-order FP32 accumulation.** This is closest to vLLM and Transformers quality.
   - **Exact integer accumulation (recommended to evaluate).** Every E4M3 value is an integer multiple of 2⁻⁹ below 2¹⁸ in magnitude. A 128-term block dot product is therefore an exact sum of 36-bit integers, under 2⁴³, which fits in i64. Block sums become exact and order-independent. Only the per-block scale combination (16 or 32 FP32 terms in a fixed order) needs ordering. That makes projections batch- and chunk-invariant for free, the property that made the INT8 engine cheap to keep deterministic. It costs CUDA-core integer work instead of FP8 tensor cores: fine for decode, to be measured for prefill.
   - **Weight-only FP8 with BF16 activations.** More accurate than the reference, but numerically different from every stock engine.
2. **Numerical contract.** Adopt the production plan's D1 option 2 (deterministic, batch-invariant, fixed reduction trees) from day one. This model has no prior Rust oracle, so the first validated build becomes the oracle.
3. **Weights format.** Load the Hugging Face safetensors shards directly in Rust (manifest of hashes and admitted shapes), rather than a Python-exported bundle. Skip the vision tower at load time.
4. **Code organization.** Extract shared runtime pieces from the Qwen3-8B crate into a core crate: graph capture with programmatic edges, prefetch helpers, pinned readback, the packed head, argmax/selection, and the bench and regression tools. Keep the Qwen3-8B W8A8 crate unchanged as a regression baseline. Put the new model in its own crate.

## 4. New components

| Component | Decode (batch 1) | Prefill |
|---|---|---|
| FP8 block-scaled linear | Warp-per-row GEMV. Convert E4M3 in registers (`cvt.rn.f16x2.e4m3x2` or integer decode for the exact contract), apply scales per 128-column block, fixed order | FP8 tensor-core MMA (`mma … kind::f8f6f4`, SM120a), or exact integer tiles under the exact contract |
| Gated DeltaNet | One fused kernel per layer: conv state shift and conv, SiLU, L2 norms, gates, delta-rule state update (FP32, state in L2-friendly layout), output gated RMSNorm with SiLU(z) | Chunked delta rule (WY / chunkwise-parallel form, as in flash-linear-attention) with tensor cores; produces the final recurrent state for decode |
| Gated attention | Fused q/k norm, partial RoPE, GQA 8:1 decode attention over 10 layers, sigmoid output gate | Tiled causal attention with head_dim 256 |
| MoE | Router GEMV, softmax, top-8 with a deterministic lowest-index tie rule. One grouped kernel streams the 8 selected experts plus the shared expert, prefetching expert rows as soon as routing is known. Fused SwiGLU and weighted combine with residual | Token-to-expert grouping (sort by expert), grouped GEMM, deterministic combine order |
| LM head and selection | Reuse the lossless packed BF16 head and GPU argmax; add GPU sampling (top-k 20, top-p 0.95, seeded counter-based RNG) | Same |
| MTP speculation | Later phase: draft one token per step, verify in a two-row batch; exact and batch-invariant under the contract | n/a |
| Tokenizer and template | Qwen3.6 tokenizer (`tokenizers` crate), chat template with thinking and non-thinking modes | n/a |

Likely cuda-oxide gaps:
- FP8 conversion and FP8 MMA intrinsics for SM120/SM120a (inline PTX works meanwhile).
- A first-class programmatic launch attribute and `griddepcontrol` intrinsics.
- Possibly warp-group helpers for the chunked DeltaNet kernel.

## 5. Phases

| Phase | Deliverable | Acceptance |
|---|---|---|
| **P0: Reference and setup** | Download the FP8 checkpoint (37.5 GB). Record vLLM speed and greedy outputs on fixed prompts. Write a Torch reference script that emits per-layer traces (embedding, each layer's DeltaNet state and output, attention, MoE routing, logits). Settle decisions 1–4. | Reference traces reproducible; vLLM baseline recorded with versions. |
| **P1: Correct eager model** | Loader, FP8 linear, recurrent DeltaNet (prefill run sequentially token by token at first), gated attention, MoE, head, greedy generation. | Every operator matches independent Torch fixtures. Per-layer traces within declared tolerances. Same greedy tokens as the reference on smoke prompts. Sanitizers clean. |
| **P2: Fast decode** | Persistent decode graphs with programmatic edges; fused DeltaNet step; grouped MoE kernel with expert prefetch; packed head; hoisted cold loads. | Exact hashes preserved against the P1 oracle. Single-request decode above 196 tok/s at 1K context, measured A/B/B/A. Node-level profile and per-step accounting recorded. |
| **P3: Fast prefill** | Chunked DeltaNet, FP8 (or exact integer) tensor-core GEMMs, tiled attention. | Prefill hashes equal sequential prefill under the contract. Time to first token compared with vLLM at 1K / 8K / 32K prompts. |
| **P4: MTP speculation** | Exact draft and verify. | Outputs identical to non-speculative decode; tokens/s gain and acceptance rate on representative prompts. |
| **P5: Batching** | Per the production plan: paged KV for the 10 attention layers, per-sequence DeltaNet state slots, continuous batching, GPU sampling. | Batch-invariance tests and throughput curves. |
| **P6: Quality** | Perplexity and a coding-focused eval (for example SWE-bench Verified subset, LiveCodeBench, Terminal-Bench) against vLLM on the same checkpoint. | Within an agreed delta of the reference. |

## 6. Risks

- **Chunked DeltaNet prefill** is the hardest new kernel. A correct but slow sequential prefill unblocks P1–P2, so the risk falls on time to first token, not correctness.
- **FP8 tensor cores on SM120 through cuda-oxide** may need new intrinsics or an SM120a target. The exact-integer contract avoids them at some prefill cost.
- **Latency-bound decode:** gains depend on fusion and launch overlap more than on bandwidth tricks. Profile-driven, as in rounds 2–6.
- **Unverified reference details** listed in section 1 can silently change numerics. Check them first in P0.
