import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("check_batch_regression", Path(__file__).parents[1] / "check_batch_regression.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class BatchRegressionTests(unittest.TestCase):
    def setUp(self):
        self.workloads = {"ragged": [
            dict(prompt_ids=[1, 2, 3], max_tokens=2, stop_ids=[]),
            dict(prompt_ids=[4], max_tokens=1, stop_ids=[]),
        ]}
        self.record = dict(
            backend="inferrous-paged", workload="ragged", run=0,
            requests=copy.deepcopy(self.workloads["ragged"]),
            model_root="a" * 64, binary_sha256="b" * 64,
            profile="test-numerical-contract", device_name="test-gpu", cache_mode="reuse",
            engine_config=dict(verify_logits=True),
            batch=dict(processed_tokens=5, reused_tokens=0, generations=[
                dict(token_ids=[5, 6], logit_sha256=["c" * 64, "d" * 64],
                     step_seconds=[0.1, 0.2], finish_reason="length"),
                dict(token_ids=[7], logit_sha256=["e" * 64],
                     step_seconds=[0.1], finish_reason="length"),
            ]),
        )

    def compare(self, candidate, workloads=None):
        return checker.compare([("original", [self.record]), ("candidate", [candidate])],
                               workloads or self.workloads, 1)

    def test_cross_binary_and_cross_cache_mode_exact_match(self):
        candidate = copy.deepcopy(self.record)
        candidate.update(binary_sha256="f" * 64, cache_mode="recompute")
        candidate["batch"]["processed_tokens"] = 8  # (3 + 4) + 1, retiring the shorter stream.
        result = self.compare(candidate)
        self.assertTrue(result["exact_match"])
        self.assertEqual(result["hashes_per_repetition"], 3)
        self.assertEqual(result["streams_per_repetition"], 2)

    def test_unchanged_tokens_do_not_hide_changed_logits(self):
        candidate = copy.deepcopy(self.record)
        candidate["batch"]["generations"][0]["logit_sha256"][1] = "f" * 64
        with self.assertRaisesRegex(ValueError, "full-logit hash"):
            self.compare(candidate)

    def test_shared_missing_workload_is_not_a_pass(self):
        workloads = dict(self.workloads, omitted=[dict(prompt_ids=[9], max_tokens=1, stop_ids=[])])
        with self.assertRaisesRegex(ValueError, "workload/repetition"):
            self.compare(copy.deepcopy(self.record), workloads)

    def test_hashing_disabled_or_partial_hashes_are_rejected(self):
        for change in ["disabled", "short", "malformed"]:
            with self.subTest(change=change):
                candidate = copy.deepcopy(self.record)
                if change == "disabled":
                    candidate["engine_config"]["verify_logits"] = False
                elif change == "short":
                    candidate["batch"]["generations"][0]["logit_sha256"].pop()
                else:
                    candidate["batch"]["generations"][0]["logit_sha256"][0] = "bad"
                with self.assertRaises(ValueError):
                    self.compare(candidate)

    def test_false_recomputation_and_wrong_request_are_rejected(self):
        for change in ["cache_mode", "requests"]:
            with self.subTest(change=change):
                candidate = copy.deepcopy(self.record)
                if change == "cache_mode":
                    candidate["cache_mode"] = "recompute"  # Cached work counts cannot pass as uncached.
                else:
                    candidate["requests"][0]["prompt_ids"][0] = 8
                with self.assertRaises(ValueError):
                    self.compare(candidate)

    def test_model_identity_and_duplicate_runs_are_rejected(self):
        candidate = copy.deepcopy(self.record)
        candidate["model_root"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "model, numerical profile or device"):
            self.compare(candidate)
        with self.assertRaisesRegex(ValueError, "workload/repetition"):
            checker.validate([self.record, self.record], self.workloads, 2)


if __name__ == "__main__":
    unittest.main()
