"""Same-checkpoint INT8 comparisons, including ragged batches and full-prefix recomputation.

End-to-end latency uses the same host wall-clock boundary for both engines;
backend-native timing details are retained separately rather than conflated.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("engine", choices=["vllm", "sglang"])
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--requests", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--repeat", type=int, default=23)
    ap.add_argument("--hardware-profile", choices=["rtx5090", "rtx-pro-6000"], required=True)
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--max-num-seqs", type=int, default=1)
    ap.add_argument("--memory-fraction", type=float)
    ap.add_argument("--kv-cache-bytes", type=int)
    ap.add_argument("--prefix-cache", action="store_true")
    ap.add_argument("--cache-mode", choices=["reuse", "recompute"], default="reuse")
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--graph-size-one", action="store_true")
    ap.add_argument("--graph-capture-sizes", type=int, nargs="+")
    ap.add_argument("--graphs-without-compile", action="store_true")
    ap.add_argument("--disable-cutedsl-warmup", action="store_true")
    ap.add_argument("--linear-backend", choices=["auto", "triton"], default="auto")
    ap.add_argument("--quantization", choices=["compressed-tensors", "humming"], default="compressed-tensors")
    ap.add_argument("--attention-backend", choices=["FLASH_ATTN", "TRITON_ATTN", "B12X", "FLASHINFER"], default="FLASH_ATTN")
    ap.add_argument("--sglang-attention-backend", choices=["flashinfer", "triton", "flashattention"])
    ap.add_argument("--sglang-quantization", choices=["w8a8_int8", "compressed-tensors"], default="w8a8_int8")
    args = ap.parse_args()
    if args.memory_fraction is None:
        args.memory_fraction = {"rtx5090": 0.70, "rtx-pro-6000": 0.30}[args.hardware_profile]
    if args.output.exists():
        ap.error("output already exists; preserve previous measurements")
    if args.repeat < 1:
        ap.error("repeat must be positive")
    if args.max_context < 1 or args.max_num_seqs < 1 or not 0 < args.memory_fraction < 1:
        ap.error("invalid context, sequence count or memory fraction")
    if args.kv_cache_bytes is not None and (args.kv_cache_bytes < 1 or args.engine != "vllm"):
        ap.error("explicit KV byte budgets require vLLM and a positive value")
    if args.disable_cutedsl_warmup and args.engine != "vllm":
        ap.error("the CuTeDSL warmup setting requires vLLM")
    if args.eager and (args.graph_size_one or args.graph_capture_sizes or args.graphs_without_compile):
        ap.error("graph configuration cannot accompany eager execution")
    if args.cache_mode == "recompute" and args.prefix_cache:
        ap.error("recompute mode requires prefix caching disabled")
    if args.graph_capture_sizes is not None and (args.graph_size_one or any(size < 1 for size in args.graph_capture_sizes)):
        ap.error("graph capture sizes must be positive and cannot accompany --graph-size-one")
    capture_sizes = [1] if args.graph_size_one else args.graph_capture_sizes
    if args.graphs_without_compile and capture_sizes is None:
        capture_sizes = sorted({1 << power for power in range(args.max_num_seqs.bit_length())} | {args.max_num_seqs})
    workloads = json.loads(args.requests.read_text())
    if not workloads:
        ap.error("empty workloads")
    for name, value in workloads.items():
        requests = value if isinstance(value, list) else [value]
        if not requests or any(not r["prompt_ids"] or r["stop_ids"] or r["max_tokens"] < 1
                               or len(r["prompt_ids"]) + r["max_tokens"] > args.max_context for r in requests):
            ap.error(f"{name}: benchmark requires nonempty requests, no stops and lengths within context")
    import torch
    scale_dtype = "torch.float32"
    if args.engine == "vllm":
        from vllm import LLM, SamplingParams
        if args.quantization == "humming":
            # The separate Humming quantization method preserves the checkpoint's
            # input schema. The compressed-tensors *linear backend* adapter does not.
            from vllm.model_executor.layers.quantization.humming import HummingConfig
            config = json.loads((args.model / "config.json").read_text())["quantization_config"]
            schema = HummingConfig(config).get_layer_input_schema(config, "model.layers.0.self_attn.q_proj")
            schema = schema.to_humming_schema(torch.bfloat16)
            assert str(schema.a_dtype) == "int8", schema
            print("Humming input schema:", schema, flush=True)
            weight_schema = HummingConfig(config).get_layer_weight_schema(config, "model.layers.0.self_attn.o_proj")
            attrs = weight_schema.get_tensors_attrs(shape_n=4096, shape_k=4096,
                                                   param_dtype=torch.bfloat16, stack_size=1)
            scale_dtype = str(attrs["weight_scale"]["dtype"])
            print("Humming effective weight scale dtype:", scale_dtype, flush=True)
        extra = {"compilation_config": {"cudagraph_capture_sizes": capture_sizes}} if capture_sizes else {}
        if args.graphs_without_compile:
            extra["compilation_config"] = dict(mode=0, cudagraph_mode="FULL", cudagraph_capture_sizes=capture_sizes, custom_ops=["all"])
        if args.attention_backend == "B12X":
            extra["block_size"] = 128
        if args.kv_cache_bytes is not None:
            extra["kv_cache_memory_bytes"] = args.kv_cache_bytes
        kernels = {"linear_backend_per_quant": {"int8_w8a8": args.linear_backend}}
        if args.disable_cutedsl_warmup:
            kernels["enable_cutedsl_warmup"] = False
        engine = LLM(model=str(args.model.resolve()), dtype="bfloat16",
                     quantization=args.quantization, max_model_len=args.max_context,
                     max_num_seqs=args.max_num_seqs, gpu_memory_utilization=args.memory_fraction,
                     enable_prefix_caching=args.prefix_cache, enforce_eager=args.eager,
                     kernel_config=kernels,
                     attention_backend=args.attention_backend,
                     disable_log_stats=False, seed=0, **extra)
    else:
        import sglang
        extra = {}
        if args.sglang_attention_backend:
            extra["attention_backend"] = args.sglang_attention_backend
        engine = sglang.Engine(model_path=str(args.model.resolve()), dtype="bfloat16",
                              quantization=args.sglang_quantization, context_length=args.max_context,
                              max_running_requests=args.max_num_seqs, mem_fraction_static=args.memory_fraction,
                              disable_radix_cache=not args.prefix_cache, disable_cuda_graph=args.eager,
                              random_seed=0, **extra)
    # Query after workers start: initializing CUDA beforehand can break a stock
    # engine's multiprocessing launch or force a different start method.
    device_name = torch.cuda.get_device_name(0)
    expected_device = {"rtx5090": "RTX 5090", "rtx-pro-6000": "RTX PRO 6000"}[args.hardware_profile]
    if expected_device not in device_name:
        ap.error(f"hardware profile does not match {device_name}")
    versions = {name: importlib.metadata.version(name) for name in
                [args.engine, "torch", "transformers", "compressed-tensors"]}
    for name in ("humming-kernels", "b12x"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    manifest = args.model.parent / "manifest.json"
    model_root = hashlib.sha256(manifest.read_bytes()).hexdigest()
    environment = {key: os.environ.get(key) for key in
                   ["CUDA_HOME", "VLLM_USE_V2_MODEL_RUNNER", "VLLM_USE_FLASHINFER_SAMPLER", "INFERROUS_STOCK_IMAGE"]}

    def generate_batch(prompts, limits):
        if args.engine == "vllm":
            params = [SamplingParams(temperature=0, max_tokens=limit, ignore_eos=True,
                                     detokenize=False) for limit in limits]
            results = engine.generate([{"prompt_token_ids": prompt} for prompt in prompts],
                                      params, use_tqdm=False)
            return [(list(r.outputs[0].token_ids),
                     dict(vars(r.metrics) if r.metrics else {}, num_cached_tokens=getattr(r, "num_cached_tokens", None)))
                    for r in results]
        results = engine.generate(input_ids=prompts,
                                  sampling_params=[dict(temperature=0, max_new_tokens=limit,
                                                        ignore_eos=True) for limit in limits])
        return [(r.get("output_ids", []), r["meta_info"]) for r in results]

    with args.output.open("x") as f:
        for name, value in workloads.items():
            requests = value if isinstance(value, list) else [value]
            for run in range(args.repeat):
                start = time.perf_counter()
                steps = []
                if args.cache_mode == "reuse":
                    outputs = generate_batch([r["prompt_ids"] for r in requests],
                                             [r["max_tokens"] for r in requests])
                else:
                    generated = [[] for _ in requests]
                    metadata = [[] for _ in requests]
                    while any(len(g) < r["max_tokens"] for g, r in zip(generated, requests)):
                        active = [i for i, r in enumerate(requests) if len(generated[i]) < r["max_tokens"]]
                        prompts = [requests[i]["prompt_ids"] + generated[i] for i in active]
                        step_start = time.perf_counter()
                        result = generate_batch(prompts, [1] * len(active))
                        steps.append(dict(slots=active, prompt_lengths=list(map(len, prompts)),
                                          seconds=time.perf_counter() - step_start,
                                          elapsed_seconds=time.perf_counter() - start))
                        if len(result) != len(active):
                            raise RuntimeError("stock engine returned an incomplete batch")
                        for slot, (ids, meta) in zip(active, result):
                            if len(ids) != 1:
                                raise RuntimeError(f"recompute expected one token, got {ids}")
                            generated[slot].extend(ids)
                            metadata[slot].append(meta)
                    outputs = list(zip(generated, metadata))
                seconds = time.perf_counter() - start
                if len(outputs) != len(requests) or any(len(ids) != r["max_tokens"]
                                                       for (ids, _), r in zip(outputs, requests)):
                    raise RuntimeError(f"{name}: stock engine returned incomplete outputs")
                record = dict(engine=args.engine, versions=versions, eager=args.eager,
                              model_root=model_root, environment=environment,
                              linear_backend=args.linear_backend,
                              attention_backend=(args.attention_backend if args.engine == "vllm"
                                                 else args.sglang_attention_backend or "auto"),
                              graph_size_one=args.graph_size_one,
                              graph_capture_sizes=capture_sizes,
                              graphs_without_compile=args.graphs_without_compile,
                              disable_cutedsl_warmup=args.disable_cutedsl_warmup,
                              quantization=args.quantization if args.engine == "vllm" else args.sglang_quantization,
                              weight_scale_dtype=scale_dtype, device_name=device_name,
                              max_context=args.max_context, max_num_seqs=args.max_num_seqs,
                              hardware_profile=args.hardware_profile,
                              memory_fraction=args.memory_fraction, prefix_cache=args.prefix_cache,
                              kv_cache_bytes=args.kv_cache_bytes,
                              cache_mode=args.cache_mode, workload=name, requests=requests, run=run,
                              request_seconds=seconds, recompute_steps=steps,
                              generations=[dict(token_ids=ids, backend_metrics=meta) for ids, meta in outputs])
                if not isinstance(value, list):
                    # Preserve the original single-request result schema too.
                    record.update(request=value, token_ids=outputs[0][0], backend_metrics=outputs[0][1])
                f.write(json.dumps(record, default=str) + "\n")
                f.flush()
                print(name, run, round(seconds, 4), flush=True)
    if args.engine == "sglang":
        engine.shutdown()


if __name__ == "__main__":
    main()
