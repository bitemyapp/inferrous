import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


SCRIPT = Path(__file__).parents[1] / "compare_traces.py"


class TraceComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.reference = self.root / "reference"
        self.actual = self.root / "actual"
        self.reference.mkdir()
        self.actual.mkdir()
        self.request = dict(prompt_ids=[1, 2], max_tokens=2, stop_ids=[])
        self.metadata = dict(model_root="a" * 64, token_ids=[3, 4], traces={})
        self.generation = dict(model_root="a" * 64, request=self.request,
                               generation=dict(token_ids=[3, 4]))
        # Independently enumerate the reference writer's complete Qwen3-8B
        # schema; do not derive the fixture from the comparator under test.
        for step, rows in enumerate([2, 1]):
            shapes = {f"layer.{layer}": [rows, 4096] for layer in range(36)}
            shapes.update(norm=[rows, 4096], logits=[151936])
            for name in ["norm", "q", "a", "o", "norm2", "down"]:
                shapes["debug." + name] = [rows, 4096]
            for name in ["k", "v"]:
                shapes["debug." + name] = [rows, 1024]
            for name in ["gate", "up", "act"]:
                shapes["debug." + name] = [rows, 12288]
            shapes["debug.qpost"] = [rows, 32, 128]
            shapes["debug.kpost"] = [rows, 8, 128]
            for name, shape in shapes.items():
                filename = f"step.{step}.{name}.bin"
                raw = np.ones(shape, dtype="<f4").tobytes()
                self.metadata["traces"][filename] = dict(shape=shape, sha256=hashlib.sha256(raw).hexdigest())
                (self.reference / filename).write_bytes(raw)
                (self.actual / filename).write_bytes(raw)

    def run_comparison(self):
        (self.reference / "request.json").write_text(json.dumps(self.request))
        (self.reference / "reference.json").write_text(json.dumps(self.metadata))
        generation = self.root / "generation.json"
        generation.write_text(json.dumps(self.generation) + "\n")
        return subprocess.run([sys.executable, str(SCRIPT), str(self.reference), str(self.actual), str(generation)],
                              env=os.environ | {"OPENBLAS_NUM_THREADS": "1"}, capture_output=True, text=True)

    def assert_rejected(self):
        result = self.run_comparison()
        self.assertNotEqual(result.returncode, 0, result.stdout)
        if result.stdout.strip():
            self.assertFalse(json.loads(result.stdout).get("passed"), result.stdout)

    def test_complete_trace_passes(self):
        result = self.run_comparison()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["tensors"]), 102)

    def test_empty_trace_is_not_success(self):
        self.metadata["traces"] = {}
        self.assert_rejected()

    def test_omitted_layer_is_not_success(self):
        self.metadata["traces"].pop("step.1.layer.35.bin")
        self.assert_rejected()

    def test_wrong_shape_with_same_element_count_is_rejected(self):
        self.metadata["traces"]["step.0.debug.qpost.bin"]["shape"] = [2, 128, 32]
        self.assert_rejected()

    def test_both_files_truncated_to_same_length_are_rejected(self):
        name = "step.1.layer.35.bin"
        raw = np.ones(1, dtype="<f4").tobytes()
        self.metadata["traces"][name]["sha256"] = hashlib.sha256(raw).hexdigest()
        (self.reference / name).write_bytes(raw)
        (self.actual / name).write_bytes(raw)
        self.assert_rejected()

    def test_extra_actual_step_is_rejected(self):
        (self.actual / "step.2.logits.bin").write_bytes(b"\0\0\0\0")
        self.assert_rejected()

    def test_truncated_generation_without_stop_is_rejected(self):
        self.metadata["token_ids"] = [3]
        self.generation["generation"]["token_ids"] = [3]
        for name in list(self.metadata["traces"]):
            if name.startswith("step.1."):
                self.metadata["traces"].pop(name)
                (self.reference / name).unlink()
                (self.actual / name).unlink()
        self.assert_rejected()

    def test_valid_early_stop_passes(self):
        self.request["stop_ids"] = [3]
        self.metadata["token_ids"] = [3]
        self.generation["generation"]["token_ids"] = [3]
        for name in list(self.metadata["traces"]):
            if name.startswith("step.1."):
                self.metadata["traces"].pop(name)
                (self.reference / name).unlink()
                (self.actual / name).unlink()
        result = self.run_comparison()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(json.loads(result.stdout)["tensors"]), 51)

    def test_tokens_after_stop_are_rejected(self):
        self.request["stop_ids"] = [3]
        self.assert_rejected()

    def test_missing_actual_tensor_is_rejected(self):
        (self.actual / "step.1.layer.35.bin").unlink()
        self.assert_rejected()

    def test_corrupt_reference_digest_is_rejected(self):
        self.metadata["traces"]["step.0.logits.bin"]["sha256"] = "0" * 64
        self.assert_rejected()

    def test_nonfinite_reference_is_rejected(self):
        name = "step.0.logits.bin"
        values = np.ones(151936, dtype="<f4")
        values[0] = np.nan
        raw = values.tobytes()
        (self.reference / name).write_bytes(raw)
        self.metadata["traces"][name]["sha256"] = hashlib.sha256(raw).hexdigest()
        self.assert_rejected()

    def test_numerical_failure_is_preserved(self):
        (self.actual / "step.0.logits.bin").write_bytes(np.full(151936, 2, dtype="<f4").tobytes())
        result = self.run_comparison()
        self.assertNotEqual(result.returncode, 0)
        report = json.loads(result.stdout)
        self.assertFalse(report["passed"])
        self.assertEqual(len(report["tensors"]), 102)


if __name__ == "__main__":
    unittest.main()
