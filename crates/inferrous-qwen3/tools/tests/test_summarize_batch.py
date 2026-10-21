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


if __name__ == "__main__":
    unittest.main()
