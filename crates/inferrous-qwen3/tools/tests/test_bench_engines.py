"""Verify that the stock recomputation harness really submits the entire prefix."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("bench_engines", Path(__file__).parents[1] / "bench_engines.py")
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


class StockHarnessTests(unittest.TestCase):
    def run_harness(self, mode):
        calls = []

        class Engine:
            def __init__(self, **kwargs):
                assert not kwargs["enable_prefix_caching"]

            def generate(self, prompts, params, **kwargs):
                calls.append([(p["prompt_token_ids"].copy(), s.max_tokens) for p, s in zip(prompts, params)])
                return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=list(range(p["prompt_token_ids"][-1] + 1,
                                                                                      p["prompt_token_ids"][-1] + s.max_tokens + 1)))],
                                        metrics=SimpleNamespace()) for p, s in zip(prompts, params)]

        fake_torch = SimpleNamespace(cuda=SimpleNamespace(get_device_name=lambda _: "NVIDIA GeForce RTX 5090"))
        fake_vllm = SimpleNamespace(LLM=Engine, SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "hf").mkdir()
            (root / "manifest.json").write_text('{}')
            requests = [dict(prompt_ids=[1, 2], max_tokens=3, stop_ids=[]),
                        dict(prompt_ids=[10], max_tokens=1, stop_ids=[])]
            (root / "requests.json").write_text(json.dumps({"ragged": requests}))
            output = root / "out.jsonl"
            argv = ["bench_engines.py", "vllm", "--model", str(root / "hf"), "--requests", str(root / "requests.json"),
                    "--output", str(output), "--hardware-profile", "rtx5090", "--repeat", "2", "--cache-mode", mode]
            with patch.dict(sys.modules, {"torch": fake_torch, "vllm": fake_vllm}), \
                 patch.object(sys, "argv", argv), \
                 patch.object(harness.importlib.metadata, "version", return_value="test"):
                harness.main()
            records = [json.loads(line) for line in output.read_text().splitlines()]
        for record in records:
            self.assertEqual([g["token_ids"] for g in record["generations"]], [[3, 4, 5], [11]])
            self.assertEqual(record["cache_mode"], mode)
        return calls, records

    def test_recompute_submits_complete_growing_prefixes_and_retires_finished_requests(self):
        calls, records = self.run_harness("recompute")
        one_run = [[([1, 2], 1), ([10], 1)], [([1, 2, 3], 1)], [([1, 2, 3, 4], 1)]]
        self.assertEqual(calls, one_run * 2)
        self.assertEqual([step["slots"] for step in records[0]["recompute_steps"]], [[0, 1], [0], [0]])
        self.assertEqual([step["prompt_lengths"] for step in records[0]["recompute_steps"]], [[2, 1], [3], [4]])

    def test_cached_batch_uses_one_generate_call(self):
        calls, records = self.run_harness("reuse")
        self.assertEqual(calls, [[([1, 2], 3), ([10], 1)]] * 2)
        self.assertEqual(records[0]["recompute_steps"], [])


if __name__ == "__main__":
    unittest.main()
