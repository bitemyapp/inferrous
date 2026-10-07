"""Same-checkpoint, single-request INT8 comparisons. Run engines serially on the benchmark workstation.

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
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--graph-size-one", action="store_true")
    ap.add_argument("--graphs-without-compile", action="store_true")
    ap.add_argument("--linear-backend", choices=["auto", "triton"], default="auto")
    ap.add_argument("--quantization", choices=["compressed-tensors", "humming"], default="compressed-tensors")
    ap.add_argument("--attention-backend", choices=["FLASH_ATTN", "TRITON_ATTN", "B12X", "FLASHINFER"], default="FLASH_ATTN")
    args = ap.parse_args()
    if args.output.exists():
        ap.error("output already exists; preserve previous measurements")
    if args.repeat < 1:
        ap.error("repeat must be positive")
    requests = json.loads(args.requests.read_text())
    assert all(not r["stop_ids"] and r["max_tokens"] > 0 for r in requests.values())
    if args.engine == "vllm":
        from vllm import LLM, SamplingParams
        if args.quantization == "humming":
            # The separate Humming quantization method preserves the checkpoint's
            # input schema. The compressed-tensors *linear backend* adapter does not.
            import torch
            from vllm.model_executor.layers.quantization.humming import HummingConfig
            config = json.loads((args.model / "config.json").read_text())["quantization_config"]
            schema = HummingConfig(config).get_layer_input_schema(config, "model.layers.0.self_attn.q_proj")
            schema = schema.to_humming_schema(torch.bfloat16)
            assert str(schema.a_dtype) == "int8", schema
            print("Humming input schema:", schema, flush=True)
        extra = {"compilation_config": {"cudagraph_capture_sizes": [1]}} if args.graph_size_one else {}
        if args.graphs_without_compile:
            assert not args.eager
            extra["compilation_config"] = dict(mode=0, cudagraph_mode="FULL", cudagraph_capture_sizes=[1], custom_ops=["all"])
        if args.attention_backend == "B12X":
            extra["block_size"] = 128
        engine = LLM(model=str(args.model.resolve()), dtype="bfloat16",
                     quantization=args.quantization, max_model_len=4096,
                     max_num_seqs=1, gpu_memory_utilization=0.3,
                     enable_prefix_caching=False, enforce_eager=args.eager,
                     kernel_config={"linear_backend_per_quant": {"int8_w8a8": args.linear_backend}},
                     attention_backend=args.attention_backend,
                     disable_log_stats=False, seed=0, **extra)
    else:
        import sglang
        engine = sglang.Engine(model_path=str(args.model.resolve()), dtype="bfloat16",
                              quantization="w8a8_int8", context_length=4096,
                              max_running_requests=1, mem_fraction_static=0.3,
                              disable_radix_cache=True, disable_cuda_graph=args.eager,
                              random_seed=0)
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
                   ["CUDA_HOME", "VLLM_USE_V2_MODEL_RUNNER", "VLLM_USE_FLASHINFER_SAMPLER"]}
    with args.output.open("x") as f:
        for name, request in requests.items():
            for run in range(args.repeat):
                start = time.perf_counter()
                if args.engine == "vllm":
                    params = SamplingParams(temperature=0, max_tokens=request["max_tokens"],
                                            ignore_eos=True, detokenize=False)
                    result = engine.generate([{"prompt_token_ids": request["prompt_ids"]}], params, use_tqdm=False)[0]
                    seconds = time.perf_counter() - start
                    tokens = list(result.outputs[0].token_ids)
                    meta = vars(result.metrics) if result.metrics else {}
                else:
                    result = engine.generate(input_ids=request["prompt_ids"],
                                             sampling_params=dict(temperature=0, max_new_tokens=request["max_tokens"], ignore_eos=True))
                    seconds = time.perf_counter() - start
                    tokens = result.get("output_ids", [])
                    meta = result["meta_info"]
                assert len(tokens) == request["max_tokens"], (name, tokens, meta)
                record = dict(engine=args.engine, versions=versions, eager=args.eager,
                              model_root=model_root, environment=environment,
                              linear_backend=args.linear_backend,
                              attention_backend=args.attention_backend,
                              graph_size_one=args.graph_size_one,
                              graphs_without_compile=args.graphs_without_compile,
                              quantization=args.quantization,
                              workload=name, request=request, run=run,
                              request_seconds=seconds, token_ids=tokens, backend_metrics=meta)
                f.write(json.dumps(record, default=str) + "\n")
                f.flush()
                print(name, run, round(seconds, 4), flush=True)
    if args.engine == "sglang":
        engine.shutdown()


if __name__ == "__main__":
    main()
