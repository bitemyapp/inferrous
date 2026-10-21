import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1]))
import check_racecheck_coverage as checker


def log(kernels=None):
    text = "========= COMPUTE-SANITIZER\n"
    for index, name in enumerate(["embedding", "norm"], 1):
        filtered = index != 1 if kernels is None else name not in kernels
        suffix = " (filtered: not in include list)" if filtered else ""
        api = "cuLaunchKernel" if index == 1 else "cuGraphLaunch"
        text += (f"========= Launch #{index}{suffix}\n"
                 f"=========   Kernel: {name}\n"
                 "=========   Grid:   (2, 1, 1)\n"
                 "=========   Block:  (128, 1, 1)\n"
                 "=========   Stream: 0xabcd\n"
                 "=========     Saved host backtrace up to driver entry point at kernel launch time\n"
                 f"=========         Host Frame: {api} [0x100] in libcuda.so.1\n"
                 "========= \n")
    return text + "========= RACECHECK SUMMARY: 2 hazards displayed (0 errors, 0 warnings)\n"


class LogTests(unittest.TestCase):
    def test_complete_graph_and_eager_coverage(self):
        inventory = checker.parse_log(log())
        self.assertEqual([entry["graph"] for entry in inventory], [False, True])
        checked = [checker.validate_partition(inventory, checker.parse_log(log([name])), [name])
                   for name in ["embedding", "norm"]]
        checker.require_full_coverage(inventory, checked)

    def test_malformed_or_diagnostic_logs(self):
        text = log()
        invalid = [
            "", text.replace("COMPUTE-SANITIZER", "OTHER"),
            text.replace("Launch #2", "Launch #3"),
            text.replace("Launch #2", "Launch #1"),
            text.replace("Grid:   (2, 1, 1)", "Grid:   (0, 1, 1)"),
            text.replace("Block:  (128, 1, 1)", "Block:  (128, 1)"),
            text.replace("2 hazards displayed", "3 hazards displayed"),
            text.replace("0 errors", "1 error"), text.replace("0 warnings", "1 warning"),
            text.rsplit("========= RACECHECK SUMMARY:", 1)[0],
            text + "========= Warning: tracking overflow\n",
            text.replace("========= RACECHECK SUMMARY:", "========= Error: lost launch\n========= RACECHECK SUMMARY:"),
            text.replace("=========         Host Frame:", "========= Error: invalid access\n=========         Host Frame:", 1),
        ]
        for broken in invalid:
            with self.subTest(log=broken[-100:]), self.assertRaises(ValueError):
                checker.parse_log(broken)

    def test_missing_duplicate_or_unknown_coverage(self):
        inventory = checker.parse_log(log())
        for sets in [[], [[1]], [[1, 2], [2]], [[1, 2, 3]]]:
            with self.subTest(sets=sets), self.assertRaises(ValueError):
                checker.require_full_coverage(inventory, sets)
        for names in [[], ["absent"], ["embedding", "embedding"]]:
            with self.subTest(names=names), self.assertRaises(ValueError):
                checker.validate_partition(inventory, inventory, names)

    def test_changed_sequence_shape_or_filters(self):
        reference = checker.parse_log(log())
        for key, value in [("grid", [3, 1, 1]), ("block", [64, 1, 1]),
                           ("graph", True), ("kernel", "other"), ("filtered", True)]:
            actual = copy.deepcopy(reference)
            actual[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                checker.validate_partition(reference, actual, ["embedding"])
        with self.assertRaises(ValueError):
            checker.validate_partition(reference, reference[:-1], ["embedding"])
        with self.assertRaises(ValueError):
            checker.validate_partition(reference, checker.parse_log(log(["embedding", "norm"])), ["embedding"])


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sanitizer = self.put("sanitizer", "test sanitizer")
        self.patcher = patch.object(checker, "SANITIZER_SHA256", self.sanitizer["sha256"])
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.binary = self.put("candidate", "candidate executable")
        self.original_binary = self.put("original", "original executable")
        self.config = self.put_json("config.json", dict(verify_logits=True, cuda_graphs=True))
        self.requests = self.put_json("requests.json", [dict(prompt_ids=[1, 2], max_tokens=1, stop_ids=[])])
        self.record = dict(backend="inferrous-paged", workload="default", run=0,
                           requests=json.loads(self.path(self.requests).read_text()),
                           model_root="a" * 64, profile="test-policy", device_name="test-gpu",
                           cache_mode="reuse", engine_config=dict(verify_logits=True, cuda_graphs=True),
                           batch=dict(processed_tokens=2, reused_tokens=0, generations=[
                               dict(token_ids=[3], logit_sha256=["c" * 64], step_seconds=[0.1], finish_reason="length")]))
        original_model = self.put_json("original-model.jsonl", self.record)
        original_stderr = self.put("original-stderr.log", "")
        receipt = self.put_json("original-receipt.json", dict(
            command=self.model_command(self.original_binary), exit_code=0,
            binary_sha256=self.original_binary["sha256"], stdout_sha256=original_model["sha256"],
            stderr_sha256=original_stderr["sha256"]))
        self.manifest = dict(schema_version=1, working_directory=".", model="model-bundle",
                             sanitizer=self.sanitizer, binary=self.binary, config=self.config, requests=self.requests,
                             original=dict(binary=self.original_binary, config=self.config, model=original_model,
                                           stderr=original_stderr, receipt=receipt),
                             inventory=self.make_run("inventory", None),
                             partitions=[self.make_run("embedding", ["embedding"]), self.make_run("norm", ["norm"])])

    def path(self, artifact):
        return self.root / artifact["path"]

    def put(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return dict(path=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def put_json(self, name, value):
        return self.put(name, json.dumps(value) + "\n")

    def model_command(self, binary):
        return [binary["path"], "model-bundle", "--engine-config", self.config["path"],
                "--batch", self.requests["path"], "--repeat", "1"]

    def make_run(self, label, kernels):
        diagnostic = self.put(label + ".log", log(kernels))
        model = self.put_json(label + ".jsonl", self.record)
        stderr = self.put(label + "-stderr.log", "")
        result = self.put_json(label + "-result.json", dict(exit_code=99, memory_limit_triggered=False,
                              log_sha256=diagnostic["sha256"], model_sha256=model["sha256"], stderr_sha256=stderr["sha256"]))
        command = [self.sanitizer["path"], "--tool", "racecheck", "--error-exitcode", "99",
                   "--print-limit", "0", "--dump-kernel-launches", "--log-file", diagnostic["path"]]
        if kernels is None:
            command += ["--launch-count", "1"]
        else:
            for kernel in kernels:
                command += ["--kernel-name", "kne=" + kernel]
        command += self.model_command(self.binary)
        metadata = self.put_json(label + "-command.json", dict(command=command))
        return dict(log=diagnostic, model=model, stderr=stderr, result=result, command=metadata, kernels=kernels)

    def alter_json(self, entry, change):
        value = json.loads(self.path(entry).read_text())
        change(value)
        return self.put_json(entry["path"], value)

    def verify(self):
        return checker.verify(self.manifest, self.root)

    def test_complete_manifest(self):
        report = self.verify()
        self.assertTrue(report["passed"])
        self.assertEqual(report["covered_launches"], 2)
        self.assertEqual(report["covered_kernels"], 2)
        self.assertEqual(report["regression"]["hashes_per_repetition"], 1)
        self.assertEqual(len(report["regression"]["inputs"]), 4)

    def test_missing_partition_is_not_a_pass(self):
        self.manifest["partitions"].pop()
        with self.assertRaisesRegex(ValueError, "missing or duplicate coverage"):
            self.verify()

    def test_duplicate_partition_is_not_a_pass(self):
        self.manifest["partitions"].append(self.manifest["partitions"][0])
        with self.assertRaisesRegex(ValueError, "missing or duplicate coverage"):
            self.verify()

    def test_mutated_artifact(self):
        self.path(self.manifest["partitions"][0]["log"]).write_text(log(["embedding", "norm"]))
        with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
            self.verify()

    def test_incomplete_process_and_receipt(self):
        run = self.manifest["partitions"][0]
        for field, value in [("exit_code", -9), ("memory_limit_triggered", True), ("model_sha256", "0" * 64)]:
            original = copy.deepcopy(run["result"])
            data = self.path(original).read_text()
            run["result"] = self.alter_json(run["result"], lambda r: r.update({field: value}))
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.verify()
            self.path(original).write_text(data)
            run["result"] = original

    def test_unaudited_sanitizer(self):
        with patch.object(checker, "SANITIZER_SHA256", "f" * 64), self.assertRaisesRegex(ValueError, "not been audited"):
            self.verify()

    def test_narrowed_or_disabled_tracking_commands(self):
        run = self.manifest["partitions"][0]
        original = copy.deepcopy(run["command"])
        source = self.path(original).read_text()
        for option in [["--launch-skip", "1"], ["--launch-count", "1"], ["--racecheck-filter-cluster-id", "0"],
                       ["--racecheck-memcpy-async", "no"], ["--check-bulk-copy", "no"],
                       ["--racecheck-detect-level", "error"], ["--suppressions", "ignore.xml"],
                       ["--print-level", "warn"], ["--error-exitcode", "0"]]:
            run["command"] = self.put_json(original["path"], json.loads(source))
            run["command"] = self.alter_json(run["command"], lambda r: r["command"].__setitem__(slice(1, 1), option))
            with self.subTest(option=option), self.assertRaises(ValueError):
                self.verify()

    def test_changed_workload_command(self):
        run = self.manifest["partitions"][0]
        run["command"] = self.alter_json(run["command"], lambda r: r["command"].__setitem__(-1, "2"))
        with self.assertRaisesRegex(ValueError, "model command differs"):
            self.verify()

    def test_original_failure(self):
        original = self.manifest["original"]
        original["receipt"] = self.alter_json(original["receipt"], lambda r: r.update(exit_code=1))
        with self.assertRaisesRegex(ValueError, "original model receipt"):
            self.verify()

    def test_original_config_mismatch(self):
        original = self.manifest["original"]
        original["model"] = self.alter_json(original["model"], lambda r: r["engine_config"].update(cuda_graphs=False))
        original["receipt"] = self.alter_json(original["receipt"], lambda r: r.update(stdout_sha256=original["model"]["sha256"]))
        with self.assertRaisesRegex(ValueError, "original execution settings"):
            self.verify()

    def test_cache_mode_must_match_command(self):
        run = self.manifest["partitions"][0]
        run["model"] = self.alter_json(run["model"], lambda r: r.update(cache_mode="recompute"))
        run["result"] = self.alter_json(run["result"], lambda r: r.update(model_sha256=run["model"]["sha256"]))
        with self.assertRaisesRegex(ValueError, "cache mode differs"):
            self.verify()

    def test_changed_logits_despite_matching_tokens(self):
        run = self.manifest["partitions"][0]
        run["model"] = self.alter_json(run["model"], lambda r: r["batch"]["generations"][0].update(logit_sha256=["d" * 64]))
        run["result"] = self.alter_json(run["result"], lambda r: r.update(model_sha256=run["model"]["sha256"]))
        with self.assertRaisesRegex(ValueError, "full-logit hash"):
            self.verify()

    def test_wrong_engine_config(self):
        run = self.manifest["partitions"][0]
        run["model"] = self.alter_json(run["model"], lambda r: r["engine_config"].update(cuda_graphs=False))
        run["result"] = self.alter_json(run["result"], lambda r: r.update(model_sha256=run["model"]["sha256"]))
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
