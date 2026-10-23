"""Keep stock timestamp domains and ragged/recomputed accounting explicit."""
import importlib.util
import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("summarize_batch", Path(__file__).parents[1] / "summarize_batch.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class SummaryTests(unittest.TestCase):
    def local_record(self):
        before = dict(lookups=10, hits=3, misses=7, matched_tokens=192, capacity_evictions=0,
                      entry_limit_evictions=0, replacements=0, cleared_entries=0)
        after = dict(before, lookups=12, hits=4, misses=8, matched_tokens=256,
                     capacity_evictions=1, replacements=1)
        return dict(backend="inferrous-paged", cache_mode="reuse", generate_seconds=4.0,
                    engine_config=dict(prefix_cache_entries=32), requests=[dict(max_tokens=2)],
                    batch=dict(reused_tokens=64, processed_tokens=2,
                               generations=[dict(token_ids=[1, 2], request_seconds=4.0, step_seconds=[3.0, 1.0])]),
                    prefix_cache_stats=dict(before=dict(counters=before, resident_entries=1, resident_tokens=64),
                                            after=dict(counters=after, resident_entries=1, resident_tokens=64)))

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

    def test_ragged_streams_keep_distinct_latencies_and_slots(self):
        record = self.record()
        record["generations"][1]["backend_metrics"].update(first_token_latency=2.0, last_token_ts=12.0)
        streams = summary.stream_measurements(record)
        self.assertEqual(streams[0]["native_first_token_seconds"], [0.5])
        self.assertEqual(streams[1]["native_first_token_seconds"], [2.0])
        self.assertEqual(streams[0]["native_per_stream_post_first_seconds_per_token"], [1.0])
        self.assertEqual(streams[1]["native_per_stream_post_first_seconds_per_token"], [2.0])
        self.assertEqual(summary.measurements(record)["native_per_stream_post_first_seconds_per_token"], [1.0, 2.0])

    def test_single_output_missing_timestamps_and_zero_interval_are_distinct(self):
        record = self.record()
        record["requests"][0]["max_tokens"] = 1
        record["generations"][0]["token_ids"] = [1]
        record["generations"][1]["backend_metrics"]["last_token_ts"] = 10.0
        streams = summary.stream_measurements(record)
        self.assertEqual(streams[0]["native_per_stream_post_first_seconds_per_token"], [])
        self.assertEqual(streams[1]["native_per_stream_post_first_seconds_per_token"], [0.0])
        self.assertEqual(streams[1]["native_per_stream_post_first_tokens_per_second"], [])
        del record["generations"][1]["backend_metrics"]["last_token_ts"]
        self.assertEqual(summary.stream_measurements(record)[1]["native_per_stream_post_first_seconds_per_token"], [])

    def test_recompute_slot_latencies_follow_only_active_submissions(self):
        record = self.record()
        record.update(cache_mode="recompute", recompute_steps=[dict(slots=slots, elapsed_seconds=elapsed)
                      for slots, elapsed in [([0, 1], 0.5), ([0, 1], 2.5), ([0], 3.5), ([0], 9.5)]])
        streams = summary.stream_measurements(record)
        self.assertEqual(streams[0]["native_per_stream_post_first_seconds_per_token"], [3.0])
        self.assertEqual(streams[1]["native_per_stream_post_first_seconds_per_token"], [2.0])

    def test_cli_stream_distributions_exclude_each_files_warmups(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for arm in range(2):
                path = Path(directory) / f"arm-{arm}.jsonl"
                paths.append(str(path))
                records = []
                for run in range(3):
                    record = self.record()
                    record.update(run=run, workload="ragged")
                    for slot, request in enumerate(record["requests"]):
                        request["prompt_ids"] = [42] * (slot + 1)
                        record["generations"][slot]["backend_metrics"]["first_token_latency"] = (
                            1000.0 if run == 0 else float(1 + arm * 2 + run + slot * 10))
                    records.append(record)
                path.write_text("".join(json.dumps(record) + "\n" for record in records))
            stdout = io.StringIO()
            with patch("sys.argv", ["summarize_batch.py", *paths, "--warmup", "1", "--min-samples", "4"]), contextlib.redirect_stdout(stdout):
                summary.main()
            result, = json.loads(stdout.getvalue())
            self.assertEqual(result["metrics"]["api_seconds"]["n"], 4)
            self.assertEqual(result["metrics"]["native_first_token_seconds"]["n"], 8)
            for slot, stream in enumerate(result["streams"]):
                self.assertEqual((stream["slot"], stream["prompt_length"]), (slot, slot + 1))
                self.assertEqual(stream["metrics"]["native_first_token_seconds"]["n"], 4)
                self.assertEqual(stream["metrics"]["native_first_token_seconds"]["p50"], 3.5 + slot * 10)

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

    def test_local_prefix_deltas_are_separate_from_lifetime_totals(self):
        values = summary.measurements(self.local_record())
        self.assertEqual(values["prefix_lookups"], [2])
        self.assertEqual(values["prefix_hits"], [1])
        self.assertEqual(values["prefix_misses"], [1])
        self.assertEqual(values["prefix_matched_tokens"], [64])
        self.assertEqual(values["prefix_hit_fraction"], [0.5])
        self.assertEqual(values["prefix_capacity_evictions"], [1])
        self.assertEqual(values["prefix_resident_entries"], [1])

    def test_absent_prefix_counters_are_unknown_and_no_lookups_have_no_hit_fraction(self):
        record = self.local_record()
        record["prefix_cache_stats"]["after"] = copy.deepcopy(record["prefix_cache_stats"]["before"])
        values = summary.measurements(record)
        self.assertEqual(values["prefix_hits"], [0])
        self.assertEqual(values["prefix_hit_fraction"], [])
        del record["prefix_cache_stats"]
        self.assertEqual(summary.measurements(record)["prefix_hits"], [])
        self.assertEqual(summary.measurements(self.record())["prefix_hits"], [])

    def test_inconsistent_prefix_counters_and_disabled_policy_are_rejected(self):
        for key, value in [("lookups", 11), ("matched_tokens", 192), ("replacements", -1), ("hits", True)]:
            record = self.local_record()
            record["prefix_cache_stats"]["after"]["counters"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                summary.measurements(record)
        record = self.local_record()
        record["prefix_cache_stats"]["before"]["counters"]["cleared_entries"] = 1
        with self.assertRaisesRegex(ValueError, "backwards"):
            summary.measurements(record)
        record = self.local_record()
        record["engine_config"]["prefix_cache_entries"] = 0
        with self.assertRaisesRegex(ValueError, "disabled policy"):
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
