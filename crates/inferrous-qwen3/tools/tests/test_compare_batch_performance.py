"""Reject comparisons that can manufacture a win by dropping or changing work."""
import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).parents[1]))
from compare_batch_performance import compare


class PerformanceComparisonTests(unittest.TestCase):
    def setUp(self):
        self.workloads = {name: [dict(prompt_ids=[1, 2, 3], max_tokens=2, stop_ids=[])] for name in ["a", "b"]}
        common = dict(model_root="a" * 64, device_name="test-gpu", cache_mode="reuse")
        self.local, self.slow, self.fast = [], [], []
        for name, requests in self.workloads.items():
            for run in range(3):
                self.local.append(dict(**common, backend="inferrous-paged", binary_sha256="b" * 64,
                    workload=name, run=run, requests=requests, generate_seconds=2.0,
                    engine_config=dict(hardware="rtx-pro-6000", max_context=32768, max_batch=32,
                                       kv_cache_bytes=12 << 30, prefix_cache_entries=0, verify_logits=False),
                    batch=dict(processed_tokens=4, reused_tokens=0, generations=[dict(token_ids=[4, 5],
                               request_seconds=2.0, step_seconds=[1.5, 0.5])])) )
                stock = dict(**common, engine="vllm", workload=name, run=run, requests=requests,
                    hardware_profile="rtx-pro-6000", max_context=32768, max_num_seqs=32,
                    kv_cache_bytes=12 << 30, prefix_cache=False, weight_scale_dtype="torch.float32",
                    quantization="compressed-tensors", attention_backend="TRITON_ATTN",
                    request_seconds=3.0, generations=[dict(token_ids=[4, 5], backend_metrics={})])
                self.slow.append(stock)
                fast = copy.deepcopy(stock)
                fast.update(attention_backend="FLASHINFER", request_seconds=1.0 if name == "a" else 4.0)
                self.fast.append(fast)

    def report(self):
        return compare([("local", self.local)], [("slow", self.slow), ("fast", self.fast)],
                       self.workloads, warmup=1, minimum=2)

    def test_selects_per_workload_and_reports_a_real_loss(self):
        report = self.report()
        self.assertFalse(report["all_local_medians_lower"])
        self.assertEqual([row["stock_over_local"] for row in report["comparisons"]], [0.5, 1.5])
        self.assertEqual([row["local_median_lower"] for row in report["comparisons"]], [False, True])
        self.assertEqual(report["comparisons"][0]["local_api_seconds"]["n"], 2)

    def test_missing_shared_workload_and_partial_stock_arm_fail(self):
        for shared in [False, True]:
            self.setUp()
            self.fast = [r for r in self.fast if r["workload"] != "b"]
            if shared:
                self.local = [r for r in self.local if r["workload"] != "b"]
                self.slow = [r for r in self.slow if r["workload"] != "b"]
            with self.assertRaisesRegex(ValueError, "incomplete workload"):
                self.report()

    def test_capacity_precision_and_verification_mismatches_fail(self):
        for key, value in [("kv_cache_bytes", 64 << 30), ("prefix_cache", True),
                           ("weight_scale_dtype", "torch.bfloat16"), ("quantization", "humming")]:
            self.setUp()
            for record in self.fast:
                record[key] = value
            with self.assertRaises(ValueError):
                self.report()
        self.setUp()
        for record in self.local:
            record["engine_config"]["verify_logits"] = True
        with self.assertRaisesRegex(ValueError, "verification timings"):
            self.report()

    def test_insufficient_or_duplicated_repetitions_fail(self):
        with self.assertRaisesRegex(ValueError, "insufficient"):
            compare([("local", self.local)], [("stock", self.slow)], self.workloads, minimum=20)
        self.fast[1]["run"] = 0
        with self.assertRaisesRegex(ValueError, "repetitions"):
            self.report()

    def test_recompute_requires_actual_growing_prefix_submissions(self):
        for record in self.local:
            record["cache_mode"] = "recompute"
            record["batch"]["processed_tokens"] = 7
        for record in self.slow + self.fast:
            record["cache_mode"] = "recompute"
            record["recompute_steps"] = [dict(slots=[0], prompt_lengths=[3 + i], elapsed_seconds=i + 1.0) for i in range(2)]
            record["generations"][0]["backend_metrics"] = [dict(num_cached_tokens=0) for _ in range(2)]
        self.report()
        self.fast[-1]["recompute_steps"][-1]["prompt_lengths"] = [1]
        with self.assertRaisesRegex(ValueError, "complete active prefixes"):
            self.report()

    def test_changed_request_and_nonfinite_timing_fail(self):
        self.fast[0]["requests"][0]["prompt_ids"].append(9)
        with self.assertRaisesRegex(ValueError, "mismatched workload"):
            self.report()
        self.setUp()
        self.local[0]["generate_seconds"] = float("nan")
        with self.assertRaisesRegex(ValueError, "invalid timing"):
            self.report()


if __name__ == "__main__":
    unittest.main()
