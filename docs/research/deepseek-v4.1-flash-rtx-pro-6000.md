# DeepSeek-V4.1-Flash on one RTX PRO 6000: quantizations and a path to running it

Researched 2026-10-06.

- **Verified directly:** Hugging Face API, config and file-tree responses; GitHub issue and PR states; the cards for infosave, JigSawPT and diffbot; and measurements on our test workstation.
- **Not re-checked:** other repository details came from a delegated survey and are marked *(survey)*.

## Verdict

The model does not fit on one 96 GB card. Its experts are already 4-bit and about 40% of the checkpoint is an n-gram lookup table (Engram), so further quantization shrinks it less than usual.

- **One card:** feasible only with routed experts in CPU RAM or streamed from NVMe. Expect about 5–25 tokens/s decode, depending on runtime and quant.
- **Constraint:** our test workstation had 199 GB of free disk, and every complete variant is 265–527 GB.
- **Two cards:** two RTX PRO 6000s have a published, patched vLLM recipe at 74–159 tokens/s.

## The model (verified)

`deepseek-ai/DeepSeek-V4.1-Flash`: public, MIT, created 2026-09-10, `DeepseekV41ForCausalLM`, 48 safetensors shards totalling **510.31 GB**.

| Property | Value (from `config.json`) |
|---|---|
| Layers / hidden / vocab | 40 / 5120 / 129,280 |
| MoE | 384 routed experts, 1 shared, 6 per token, `moe_intermediate_size` 2304, `sqrtsoftplus` scoring, `noaux_tc` |
| Attention | 64 heads, 1 KV head, `head_dim` 512 (64 rope), `q_lora_rank` 1280, `o_lora_rank` 1024, 128-token sliding window, per-layer compression ratios (2× layers 2–19, 1× layers 20–39), sparse indexer (top-512), hierarchical candidate blocks |
| Other | Hyper-connections (`hc_mult` 4, Sinkhorn); Engram tables at layers 1 and 14 (about 384M rows × 256 each); 3 MTP/DSpark drafter layers; 32-layer vision encoder |
| Context | `max_position_embeddings` 1,048,576 (YaRN factor 16 over 65,536) |
| Precision | `quant_method: fp8`, `weight_block_size: [32, 32]`, `scale_fmt: ue8m0`, `expert_dtype: fp4`. Routed experts are MXFP4 (E2M1, one E8M0 scale per 32). Attention, shared experts and Engram are FP8 E4M3. Embeddings and router are BF16. |
| Parameter count (HF) | 763.2B as listed: BF16 2.0B, F8_E4M3 204.0B, I8 557.2B (packed FP4). The card describes a 552B backbone plus 196B of Engram *(survey)*. |
| Largest files | Shards 47–48, the Engram tables, at 101.5 GB each |

Resident memory by component, from ik_llama.cpp #2519's breakdown of the Q8_0 GGUF: routed experts (MXFP4) **288.8 GB**, Engram (Q8_0) **209.2 GB**, everything else **9.9 GB**. The diffbot recipe shrinks Engram to 111 GB with int4 and FP16 scales per 32 values.

Per-token decode traffic (estimate): 6 experts × 3 × 5120 × 2304 weights at 4.25 bits ≈ 113 MB per layer, about **4.5 GB per token** for 40 layers. All dense weights add about 8–10 GB. Engram reads only a few dozen rows per token, so it can live on NVMe through mmap with little cost. The KV cache is tiny (diffbot: 4.25 GiB of FP8 KV for 1.07M tokens).

## Quantizations available

Sizes marked ✓ were verified from repository trees. There are more than 100 quantized derivatives, many of them re-uploads or "uncensored" fine-tunes.

| Repository | Format | Size | Runtime | Notes |
|---|---|---:|---|---|
| deepseek-ai (official) | FP8 + MXFP4 experts | 510.3 GB ✓ | vLLM, SGLang | The baseline. |
| nvidia/…-NVFP4 | NVFP4 W4A4 experts | 527.3 GB ✓ | vLLM, SGLang | Tested on GB300 *(survey)*. |
| amd/…-Quark-MXFP4, intel-ai/…-MXFP4-Mixed | MXFP4/MXFP8 | about 510 GB *(survey)* | vLLM | |
| INCModel3/…-W4A16-Engram-AutoRound | INT4 g32 experts and Engram | 451.7 GB *(survey)* | vLLM (Marlin) | |
| vcruz305/…-GGUF | Q2_K / Q3_K_M / Q4_K_M / Q8_0 | 264.5 / 347.3 / 444.7 / 508.0 GB ✓ | ik_llama.cpp; the Q8_0 loads per ik #2519 | Smallest complete GGUF with a working mainline-fork runtime. |
| smalinin/…-GGUF | Q2_K+Q5 Engram, IQ2_XXS+Q8, Q2_K-protected, IQ3_XS, MXFP4 | 335.4 / 371.4 / 413.4 / 434.0 / 508.0 GB ✓ | smalinin fork (CUDA) | Includes the vision projector. |
| JigSawPT/…-GGUF | Lossless MXFP4 + raw FP8 Engram | 501.8 GB *(survey)* | JigSawPT llama.cpp fork with NVMe expert streaming | 5.1 tok/s new content, 21.4 resident on an RTX 5090 with 125.7 GiB RAM ✓ |
| infosave/…-cmf | Q4TP experts + FP8 Engram | 499.1 GB ✓ | cortiq Rust/Vulkan runtime | **One RTX PRO 6000, about 234 GiB RAM: 7.08 tok/s first request, 11.26 repeat** ✓ |
| diffbot/…-EXL3-3bpw-2x-RTX-PRO-6000 | EXL3 3 bpw experts + int4 Engram + 3-bit drafter | 332.1 GB ✓ | Patched vLLM nightly + vllm-exl3, TP2 on SM120 | 2× RTX PRO 6000, about 230 GiB RAM: 159 tok/s agentic, 74–124 synthetic ✓ |
| coolbho3k, Mia-AiLab, vcruz305 (SAGE), satgeze | EXL3 2–4.75 bpw | 210–460 GB *(survey)* | vllm-exl3 | |
| apetersson/…-MixedQ2-GGUF | 2.25 bpw experts; no Engram, MTP or vision | 168.9 GB ✓ | No proven runtime | Incomplete without Engram. |
| antirez/deepseek-v4.1-flash-gguf | Q2 / Q4 | 885 GB total for both ✓ | DwarfStar (Metal) | Not usable here. |
| mlx-community, oMLX, others | MLX | 239–553 GB *(survey)* | Apple only | Not usable here. |

## Engine status on SM120 (verified issue and PR states)

| Engine | Status |
|---|---|
| vLLM | Model merged (#56228, 2026-09-10). **#59203 open: cannot run on SM120.** FlashInfer's SM120 sparse-MLA kernels lack the 32-token page size the compressed layers need. The working 2-GPU route is diffbot's patched nightly. |
| SGLang | Integration merged (#38798, 2026-09-18). SM120 field report on 8× RTX PRO 6000 (#40877, open). Single-GPU offload is unproven. |
| ik_llama.cpp | `deepseek41` merged (#2455, 2026-09-21). **#2519 open: MXFP4 routed experts on SM120 GPUs give NaNs or illegal-address crashes (Xid 31). Correct output only with all routed experts on CPU** (about 19 tok/s on a Zen 3 EPYC with 512 GB DDR4). |
| llama.cpp upstream | #28696 open; no runtime yet. |
| ktransformers, TensorRT-LLM, ExLlamaV3 native | Not supported, or open PRs only *(survey)*. |
| DeepGEMM / FlashMLA on SM120 | No upstream support *(survey)*. |

## Test workstation constraints (measured 2026-10-06)

| Resource | Value |
|---|---|
| GPU | 1× RTX PRO 6000 Max-Q, 96 GB, driver 580.159.04 |
| Host RAM | 251 GiB total, about 231 GiB available |
| CPU memory bandwidth | 282 GB/s read, 226 GB/s triad (64 threads; likely 8-channel DDR5) |
| CPU | Threadripper PRO 9985WX, 64 cores, 1 NUMA node |
| Disk | One 4 TB NVMe, **199 GB free** at the time of writing |
| Installed | vLLM 0.30.0 (too old for V4.1); no llama.cpp |

Memory fit: VRAM plus available RAM is about 345 GB. Lossless MXFP4 experts (289 GB) plus dense weights (10 GB) fit across the two only if some experts run on the GPU, which is the configuration ik_llama.cpp currently breaks on SM120. Engram must sit on NVMe through mmap, or be int4 in RAM (111 GB) at the cost of expert space.

## Options for one card

| Option | Disk | Expected decode | Quality | Readiness |
|---|---:|---|---|---|
| A. ik_llama.cpp, vcruz305 Q2_K, all routed experts on CPU, dense on GPU, Engram mmapped | 265 GB | About 15–25 tok/s (estimate: about 2.8 GB of Q2_K experts per token against 282 GB/s) | 2-bit experts: measure | Uses the only configuration known correct on SM120. Q2_K loading in ik is not yet confirmed; Q8_0 is. |
| B. infosave CMF runtime (Vulkan, dynamic expert staging) | 500 GB | 7–11 tok/s, measured on this hardware class | Q4TP experts, FP8 Engram | Proven on 1× RTX PRO 6000; third-party runtime. |
| C. JigSawPT fork, lossless MXFP4 with NVMe streaming and a VRAM/pinned cache | 502 GB | 5–21 tok/s on an RTX 5090; likely better here with 3× the VRAM and 2× the RAM | Lossless experts | Fork; proven on CUDA consumer Blackwell. |
| D. Lossless experts split VRAM/RAM (ik_llama.cpp or vLLM offload) | 510 GB | About 20–40 tok/s if routing is skewed (estimate) | Lossless | Blocked by ik #2519 and vLLM #59203. |
| E. Second RTX PRO 6000 with the diffbot recipe | 332 GB | 74–159 tok/s, measured | 3 bpw experts, int4 Engram | Proven, but pinned to patched vLLM, AGPL-licensed patches and about 230 GiB RAM. |

## Recommended plan

1. **Free disk.** Make about 600 GB free, or attach a second NVMe. That covers one complete variant plus headroom. Converting from the official checkpoint would need source and output on disk together, about 1 TB, so prefer pre-quantized repositories.
2. **Proof of life (option A).** Build ik_llama.cpp with CUDA for sm_120. Download vcruz305 Q2_K (264.5 GB). Run with `-ot "ffn_.*_exps=CPU"` so attention and dense layers stay on the GPU, mmap Engram, use 64 threads and `-fa on`. Measure decode, prefill and memory, and spot-check quality against the official API on a fixed prompt set. If Q2_K does not load, use the Q3_K_M variant (347 GB, partly paged from NVMe) or option B.
3. **Lossless quality (option C or B).** If 2-bit experts degrade quality, run the 500 GB lossless variants with expert streaming and expect about 10–20 tok/s.
4. **Track upstream fixes.** Once ik_llama.cpp #2519 or vLLM #59203 is fixed, move hot experts into VRAM (option D).
5. **For interactive speed,** add a second RTX PRO 6000 and adopt the diffbot stack (option E), or wait for upstream SM120 support.

**Building our own engine.** A cuda-oxide implementation would be a large, separate project. It would need sparse compressed attention with an indexer and candidate blocks, hyper-connections with Sinkhorn mixing, Engram lookup, MXFP4 MoE with host offload and expert scheduling, and MTP/DSpark drafting. The reusable lessons from the Qwen engine are the bandwidth-first decode design and programmatic launches. The new hard part is expert offload across PCIe, CPU and NVMe. Do not start before Qwen production decision D0 is settled.

## Sources

- https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash (API metadata, `config.json`, file tree)
- https://huggingface.co/api/models?filter=base_model:quantized:deepseek-ai/DeepSeek-V4.1-Flash
- https://huggingface.co/infosave/DeepSeek-V4.1-Flash-cmf
- https://huggingface.co/JigSawPT/DeepSeek-V4.1-Flash-GGUF
- https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000
- https://huggingface.co/vcruz305/DeepSeek-V4.1-Flash-GGUF
- https://huggingface.co/smalinin/DeepSeek-V4.1-Flash-GGUF
- https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-NVFP4
- https://github.com/vllm-project/vllm/pull/56228, https://github.com/vllm-project/vllm/issues/59203, https://github.com/vllm-project/vllm/issues/56700
- https://github.com/sgl-project/sglang/pull/38798, https://github.com/sgl-project/sglang/issues/40877
- https://github.com/ikawrakow/ik_llama.cpp/pull/2455, https://github.com/ikawrakow/ik_llama.cpp/issues/2519
- https://github.com/ggml-org/llama.cpp/pull/28696
