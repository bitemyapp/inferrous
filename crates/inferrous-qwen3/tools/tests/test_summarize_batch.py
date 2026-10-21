"""Keep stock timestamp domains and ragged/recomputed accounting explicit."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("summarize_batch", Path(__file__).parents[1] / "summarize_batch.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class SummaryTests(unittest.TestCase):
    def record(self):
        return dict(cache_mode="reuse", request_seconds=4.0,
                    requests=[dict(max_tokens=4), dict(max_tokens=2)],
                    generations=[dict(token_ids=[1] * n, backend_metrics=dict(arrival_time=1_790_000_000,
                                     first_token_latency=0.5, first_token_ts=10.0, last_token_ts=10.0 + n - 1))
                                 for n in [4, 2]])

    def test_stock_uses_monotonic_decode_interval_and_native_first_token_latency(self):
        values = summary.measurements(self.record())
        self.assertEqual(values["aggregate_output_tokens_per_second"], [1.5])
        self.assertEqual(values["native_first_token_seconds"], [0.5, 0.5])
        self.assertEqual(values["native_completion_seconds"], [3.5, 1.5])
        self.assertEqual(values["native_per_stream_post_first_tokens_per_second"], [1.0, 1.0])

    def test_ragged_recomputation_retires_the_shorter_stream(self):
        record = self.record()
        record.update(cache_mode="recompute", recompute_steps=[dict(slots=slots, elapsed_seconds=elapsed)
                      for slots, elapsed in [([0, 1], 0.5), ([0, 1], 1.5), ([0], 2.5), ([0], 3.5)]])
        values = summary.measurements(record)
        self.assertEqual(values["native_completion_seconds"], [3.5, 1.5])
        self.assertEqual(values["native_per_stream_post_first_tokens_per_second"], [1.0, 1.0])
        record["recompute_steps"].pop()
        with self.assertRaises(ValueError):
            summary.measurements(record)

    def test_incomplete_output_cannot_be_timed_as_a_complete_request(self):
        record = self.record()
        record["generations"][0]["token_ids"].pop()
        with self.assertRaises(ValueError):
            summary.measurements(record)

    def test_invalid_local_steps_cannot_hide_inside_a_positive_decode_total(self):
        for steps in [[1.0, -0.5, 2.0], [1.0, float("nan"), 2.0],
                      [1.0, float("inf"), 2.0], [1.0, 1e308, 1e308]]:
            record = dict(backend="inferrous-paged", cache_mode="reuse", generate_seconds=4.0,
                          requests=[dict(max_tokens=3)], batch=dict(reused_tokens=0, processed_tokens=5,
                          generations=[dict(token_ids=[1, 2, 3], request_seconds=4.0,
                                            step_seconds=steps)]))
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                summary.measurements(record)

    def test_nonfinite_api_and_reversed_native_timestamps_are_rejected(self):
        for invalid in [float("nan"), float("inf"), -1.0, 0.0, True]:
            record = self.record()
            record["request_seconds"] = invalid
            with self.subTest(api=invalid), self.assertRaises(ValueError):
                summary.measurements(record)
        record = self.record()
        record["generations"][0]["backend_metrics"]["last_token_ts"] = 9.0
        with self.assertRaises(ValueError):
            summary.measurements(record)

    def test_recompute_intermediate_timestamps_must_be_monotonic(self):
        record = self.record()
        record.update(cache_mode="recompute", recompute_steps=[dict(slots=slots, elapsed_seconds=elapsed)
                      for slots, elapsed in [([0, 1], 0.5), ([0, 1], 1.5), ([0], 1.0), ([0], 3.5)]])
        # First/final timestamps look valid; the third submission moves backwards.
        with self.assertRaises(ValueError):
            summary.measurements(record)

    def test_stock_prefix_metrics_distinguish_hits_zero_and_unknown(self):
        record = self.record()
        record.update(engine="vllm", prefix_cache=True)
        for request, generation, cached in zip(record["requests"], record["generations"], [64, 0]):
            request["prompt_ids"] = [1] * 129
            generation["backend_metrics"]["num_cached_tokens"] = cached
        self.assertEqual(summary.measurements(record)["reused_tokens"], [64])
        record["generations"][0]["backend_metrics"]["num_cached_tokens"] = 0
        self.assertEqual(summary.measurements(record)["reused_tokens"], [0])
        record["generations"][1]["backend_metrics"]["num_cached_tokens"] = None
        self.assertEqual(summary.measurements(record)["reused_tokens"], [])

    def test_stock_prefix_counter_cannot_exceed_prompt_or_contradict_policy(self):
        record = self.record()
        record.update(engine="vllm", prefix_cache=False)
        for request, generation in zip(record["requests"], record["generations"]):
            request["prompt_ids"] = [1] * 5
            generation["backend_metrics"]["num_cached_tokens"] = 1
        with self.assertRaisesRegex(ValueError, "caching disabled"):
            summary.measurements(record)
        record["prefix_cache"] = True
        for invalid in [-1, 6, 1.5, True]:
            record["generations"][0]["backend_metrics"]["num_cached_tokens"] = invalid
            with self.assertRaisesRegex(ValueError, "invalid stock"):
                summary.measurements(record)

    def test_recompute_reports_all_submissions_and_rejects_cached_work(self):
        record = self.record()
        record.update(engine="vllm", cache_mode="recompute", prefix_cache=False,
                      recompute_steps=[dict(slots=slots, elapsed_seconds=elapsed)
                        for slots, elapsed in [([0, 1], 0.5), ([0, 1], 1.5), ([0], 2.5), ([0], 3.5)]])
        for request, generation in zip(record["requests"], record["generations"]):
            request["prompt_ids"] = [1] * 5
            generation["backend_metrics"] = [dict(num_cached_tokens=0) for _ in range(request["max_tokens"])]
        self.assertEqual(summary.measurements(record)["reused_tokens"], [0])
        record["generations"][0]["backend_metrics"][-1]["num_cached_tokens"] = 1
        with self.assertRaisesRegex(ValueError, "caching disabled"):
            summary.measurements(record)
        record["generations"][0]["backend_metrics"].pop()
        with self.assertRaisesRegex(ValueError, "incomplete stock cache"):
            summary.measurements(record)


if __name__ == "__main__":
    unittest.main()
