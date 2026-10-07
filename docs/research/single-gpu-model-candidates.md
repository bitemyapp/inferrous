# Modern open models for one RTX PRO 6000: candidates and rankings

Researched 2026-10-06 while looking for a single-GPU alternative to DeepSeek-V4.1-Flash, which needs about 510 GB ([analysis](deepseek-v4.1-flash-rtx-pro-6000.md)). Sizes and configs come from Hugging Face API, config and file-tree responses. Speeds and scores are third-party or vendor-reported, as cited; none were measured by us.

**Decision:** implement **Qwen3.6-35B-A3B FP8** next ([plan](../plans/qwen3.6-35b-a3b-fp8.md)). It is the fastest candidate by a wide margin, a strong coder, runs at essentially lossless FP8 with room for batching, and shares the Qwen family with the existing Qwen3-8B engine.

## Candidates that fit one 96 GB card

| Model | Released | Architecture | Single-card format |
|---|---|---|---|
| DeepSeek-V4-Flash-0731 (MIT) | 2026-07-31 | V4.1-Flash's design without Engram: compressed sparse attention with indexer, 512-dim single-KV-head attention, hyper-connections, MXFP4 experts (256, 6 active), DSpark drafting, 1M context | 167 GB native. 2-bit GGUFs of 82.5–91 GB fit |
| Nemotron 3 Super 120B-A12B (NVIDIA licence) | 2026-03-10 | Hybrid Mamba-2/attention (8 of 88 layers attention), latent MoE (512 experts, 22 active), MTP, up to 1M context, pretrained in NVFP4 | NVFP4 80.3 GB (native precision) |
| Qwen3.5-122B-A10B (Apache-2.0) | 2026-02-24 | Gated DeltaNet hybrid (36 linear + 12 full), 256 experts with 8 active, MTP, 262K context, vision | NVFP4 83.5 GB; FP8 127 GB does not fit |
| Qwen3.6-35B-A3B (Apache-2.0) | 2026-04-15 | Same hybrid design (30 linear + 10 full), 256 experts with 8 active plus a shared expert, MTP, 262K context, vision | FP8 37.5 GB |
| Qwen3.8-27B (Apache-2.0) | 2026-08-13 | Hybrid attention (48 linear + 16 full), dense, MTP | FP8 30.9 GB |

Too large for one card: GLM-5.3-Flash (328 GB FP8), MiniMax-M2.7 (140 GB even in NVFP4), Step-3.5-Flash (209 GB FP8), and the 0.4–2.8T flagships.

## Ranking by decode speed on one RTX PRO 6000

Single request, short context.

| Rank | Model and format | Plain | With speculation | Long context | Source |
|---:|---|---:|---:|---:|---|
| 1 | Qwen3.6-35B-A3B FP8, vLLM | 196 tok/s | 219 (MTP) | 116 at 256K | [Millstone](https://www.millstoneai.com/inference-benchmark/qwen3-6-35b-a3b-fp8-1x-rtx-pro-6000-blackwell), [MTP](https://www.millstoneai.com/inference-benchmark/qwen3-6-35b-a3b-mtp-fp8-1x-rtx-pro-6000-blackwell) |
| 2 | Qwen3.5-122B-A10B Q4_K_M, llama.cpp | 101 | n/a | 80 at 262K | [localllm.in](https://localllm.in/blog/best-local-llms-96gb-vram) |
| 3 | Nemotron 3 Super NVFP4, vLLM | 70 | n/a | 62 at 512K | [Millstone](https://www.millstoneai.com/inference-benchmark/nemotron-3-super-120b-a12b-nvfp4-1x-rtx-pro-6000-blackwell) |
| 4 | Qwen3.8-27B FP8, vLLM | 44 | 76–82 (MTP); 125 with a DFlash2 draft model | n/a | [aseemshrey](https://aseemshrey.com/blog/running-qwen3-8-27b-uncensored/) |
| 5 | DeepSeek-V4-Flash 2-bit, DwarfStar 4 (alpha) | 44 | n/a | 31 at 50K | [loftllc](https://loftllc.dev/en/docs/tech/llm-research/dwarfstar4-rtx-pro-6000-deepseek-v4-flash-first-look/) |

Engines differ between rows. Decode is bandwidth-bound and both RTX PRO 6000 editions have the same bandwidth, so the comparison holds to within roughly 10–20%.

## Ranking by software-development performance

Weighted toward agentic coding. Vendor-reported at full precision; harness versions differ.

| Rank | Model | Agentic coding | LiveCodeBench v6 | Fidelity on one card |
|---:|---|---|---:|---|
| 1 | DeepSeek-V4-Flash-0731 | Terminal-Bench 2.1 82.7; DeepSWE 54.4; NL2Repo 54.2 | n/a | 2-bit only; loss unknown |
| 2 | Qwen3.8-27B | Terminal-Bench 2.1 73.0; SWE-bench Pro 61.7 | 90.3 | FP8, near-lossless |
| 3 | Qwen3.6-35B-A3B | SWE-bench Verified 73.4; SWE-bench Pro 49.5; Terminal-Bench 2.0 51.5 | 80.4 | FP8, near-lossless |
| 4 | Qwen3.5-122B-A10B | SWE-bench Verified 72.0 (66.4 in NVIDIA's OpenHands run); Terminal-Bench 2.0 49.4 | 78.9 | 4-bit, small loss |
| 5 | Nemotron 3 Super | SWE-bench (OpenHands) 60.5; Terminal-Bench Core 2.0 31.0 | 78.7 | Native NVFP4 |

Artificial Analysis gives Qwen3.8-27B and DeepSeek-V4-Flash-0731 the same Intelligence Index score (34) ([comparison](https://artificialanalysis.ai/models/comparisons/qwen3-8-27b-vs-deepseek-v4-flash)). NVIDIA's own table places Nemotron 3 Super behind Qwen3.5-122B on every software-engineering row.

## Sources

Model cards: [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731), [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), [Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B), [Nemotron 3 Super BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16), [Nemotron 3 Super NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4), [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash), [Unsloth V4-Flash-0731 GGUFs](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF). Benchmarks are linked in the tables above, plus the [JarvisLabs RTX PRO 6000 coding builds](https://jarvislabs.ai/blog/coding-model-rtx-pro-6000).
