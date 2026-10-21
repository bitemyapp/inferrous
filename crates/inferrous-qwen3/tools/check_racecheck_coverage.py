"""Require exhaustive launch coverage and exact model outputs from racecheck partitions."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from check_batch_regression import compare


# This release counts informational launch records in its hazard total and error exit code.
# Other sanitizer versions need an explicit reporting/compatibility audit before acceptance.
SANITIZER_SHA256 = "fc3b9d2abad07999a56631a6be69c320461ebdb80a5ab0044775c44d301c9488"


def parse_log(text):
    lines = iter(text.splitlines())
    if next(lines, None) != "========= COMPUTE-SANITIZER":
        raise ValueError("missing sanitizer header")
    launches = []
    line = next(lines, None)
    while line and line.startswith("========= Launch #"):
        match = re.fullmatch(r"========= Launch #(\d+)( \(filtered: [^\n]+\))?", line)
        if match is None or int(match[1]) != len(launches) + 1:
            raise ValueError("missing, duplicate or reordered launch")
        item = dict(launch=int(match[1]), filtered=match[2] is not None)
        for name, pattern in [
            ("kernel", r"=========   Kernel: (\S+)"),
            ("grid", r"=========   Grid:   \(([1-9]\d*), ([1-9]\d*), ([1-9]\d*)\)"),
            ("block", r"=========   Block:  \(([1-9]\d*), ([1-9]\d*), ([1-9]\d*)\)"),
            ("stream", r"=========   Stream: (0x[0-9a-f]+)"),
        ]:
            field = re.fullmatch(pattern, next(lines, ""))
            if field is None:
                raise ValueError(f"missing or malformed {name}")
            if name != "stream":
                item[name] = field[1] if name == "kernel" else list(map(int, field.groups()))
        if next(lines, None) != "=========     Saved host backtrace up to driver entry point at kernel launch time":
            raise ValueError("missing launch backtrace")
        frames = []
        line = next(lines, None)
        while line and line.startswith("=========         Host Frame: "):
            frames.append(line)
            line = next(lines, None)
        if not frames or line != "========= ":
            raise ValueError("unexpected diagnostic inside launch record")
        item["graph"] = any("Host Frame: cuGraphLaunch " in frame for frame in frames)
        launches.append(item)
        line = next(lines, None)
    footer = re.fullmatch(r"========= RACECHECK SUMMARY: (\d+) hazards? displayed \((\d+) errors?, (\d+) warnings?\)", line or "")
    if (not launches or footer is None or int(footer[1]) != len(launches)
            or footer.groups()[1:] != ("0", "0") or next(lines, None) is not None):
        raise ValueError("incomplete summary, extra diagnostic or hazard/count mismatch")
    return launches


def validate_partition(reference, actual, kernels):
    if not kernels or len(set(kernels)) != len(kernels):
        raise ValueError("empty or duplicate kernel selection")
    if not set(kernels) <= {item["kernel"] for item in reference}:
        raise ValueError("unknown kernel selection")
    if len(reference) != len(actual):
        raise ValueError("incomplete launch sequence")
    checked = []
    for expected, observed in zip(reference, actual):
        if {k: v for k, v in expected.items() if k != "filtered"} != {k: v for k, v in observed.items() if k != "filtered"}:
            raise ValueError("launch sequence or shape differs")
        should_check = observed["kernel"] in kernels
        if should_check == observed["filtered"]:
            raise ValueError("instrumentation selection differs")
        if should_check:
            checked.append(observed["launch"])
    return checked


def require_full_coverage(reference, checked_sets):
    counts = Counter(launch for checked in checked_sets for launch in checked)
    if counts != Counter(item["launch"] for item in reference):
        raise ValueError("missing or duplicate coverage; partial partitions are not acceptance")


def file_sha(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def check_model_command(command, workdir, binary, model, config, requests):
    resolve = lambda path: (workdir / path).resolve()
    if (len(command) != 8 or resolve(command[0]) != binary or resolve(command[1]) != model
            or command[2] != "--engine-config" or resolve(command[3]) != config
            or command[4] != "--batch" or resolve(command[5]) != requests or command[6:] != ["--repeat", "1"]):
        raise ValueError("model command differs from declared inputs")


def check_command(command, workdir, sanitizer, binary, model, config, requests, log, kernels=None):
    """Allow only the audited tracking controls, including exact kernel-name selection."""
    resolve = lambda path: (workdir / path).resolve()
    if not command or resolve(command[0]) != sanitizer:
        raise ValueError("wrong sanitizer executable")
    options = {}
    selected = []
    index = 1
    while index < len(command) and command[index].startswith("--"):
        key = command[index]
        index += 1
        if key == "--dump-kernel-launches":
            value = True
        else:
            if index >= len(command):
                raise ValueError("missing option value")
            value = command[index]
            index += 1
        if key == "--kernel-name":
            selected.append(value)
        elif key in options:
            raise ValueError("duplicate sanitizer option")
        else:
            options[key] = value
    allowed = {
        "--tool": {"racecheck"}, "--racecheck-num-workers": {"1", "4"},
        "--force-synchronization-limit": {"1"}, "--racecheck-detect-level": {"warn"},
        "--racecheck-memcpy-async": {"yes"}, "--check-bulk-copy": {"yes"},
        "--error-exitcode": {"99"}, "--print-limit": {"0"},
        "--dump-kernel-launches": {True}, "--launch-count": {"1"},
    }
    for key, value in options.items():
        if key == "--log-file":
            if resolve(value) != log:
                raise ValueError("wrong sanitizer log")
        elif key not in allowed or value not in allowed[key]:
            raise ValueError(f"unaudited sanitizer option: {key}")
    required = {"--tool", "--error-exitcode", "--print-limit", "--dump-kernel-launches", "--log-file"}
    if not required <= options.keys():
        raise ValueError("missing sanitizer tracking/reporting options")
    if kernels is None:
        if options.get("--launch-count") != "1" or selected:
            raise ValueError("wrong inventory instrumentation")
    elif "--launch-count" in options or Counter(selected) != Counter("kne=" + name for name in kernels):
        raise ValueError("partition filter differs from declared coverage")
    check_model_command(command[index:], workdir, binary, model, config, requests)


def verify(manifest, directory):
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported manifest schema")
    workdir = (directory / manifest["working_directory"]).resolve()
    hashes = {}

    def artifact(entry):
        path = (directory / entry["path"]).resolve()
        actual = file_sha(path)
        if actual != entry["sha256"]:
            raise ValueError(f"artifact hash mismatch: {path}")
        hashes[str(path)] = actual
        return path

    sanitizer = artifact(manifest["sanitizer"])
    if manifest["sanitizer"]["sha256"] != SANITIZER_SHA256:
        raise ValueError("sanitizer reporting behavior has not been audited")
    binary = artifact(manifest["binary"])
    requests = artifact(manifest["requests"])
    config = artifact(manifest["config"])
    expected_config = json.loads(config.read_text())
    model = (directory / manifest["model"]).resolve()
    workloads = {"default": json.loads(requests.read_text())}
    original = manifest["original"]
    original_binary = artifact(original["binary"])
    original_output = artifact(original["model"])
    original_config = artifact(original["config"])
    original_stderr = artifact(original["stderr"])
    original_receipt = json.loads(artifact(original["receipt"]).read_text())
    if (original_receipt["exit_code"] != 0 or original_stderr.read_bytes()
            or original_receipt["stdout_sha256"] != hashes[str(original_output)]
            or original_receipt["stderr_sha256"] != hashes[str(original_stderr)]
            or original_receipt["binary_sha256"] != hashes[str(original_binary)]):
        raise ValueError("incomplete or inconsistent original model receipt")
    check_model_command(original_receipt["command"], workdir, original_binary, model, original_config, requests)
    original_settings = json.loads(original_config.read_text())
    records = [json.loads(line) for line in original_output.read_text().splitlines()]
    for record in records:
        if record["cache_mode"] != "reuse" or any(record["engine_config"].get(key) != value for key, value in original_settings.items()):
            raise ValueError("original execution settings differ from command")
        if record.get("binary_sha256", original["binary"]["sha256"]) != original["binary"]["sha256"]:
            raise ValueError("original output executable identity differs")
        record["binary_sha256"] = hashes[str(original_binary)]
    all_inputs = [(str(original_output), records)]
    reference = None
    checked_sets = []
    reports = []
    candidate_config = None
    for run in [manifest["inventory"], *manifest["partitions"]]:
        paths = {key: artifact(run[key]) for key in ["log", "model", "stderr", "result", "command"]}
        result = json.loads(paths["result"].read_text())
        if result["exit_code"] != 99 or result["memory_limit_triggered"] is not False:
            raise ValueError("incomplete sanitizer process")
        for field, key in [("log_sha256", "log"), ("model_sha256", "model"), ("stderr_sha256", "stderr")]:
            if result[field] != hashes[str(paths[key])]:
                raise ValueError("result receipt does not match artifacts")
        if paths["stderr"].read_bytes():
            raise ValueError("unexpected application diagnostic")
        command = json.loads(paths["command"].read_text())["command"]
        kernels = None if reference is None else run["kernels"]
        check_command(command, workdir, sanitizer, binary, model, config, requests, paths["log"], kernels)
        launches = parse_log(paths["log"].read_text())
        if reference is None:
            reference = launches
            if [item["launch"] for item in launches if not item["filtered"]] != [1]:
                raise ValueError("unexpected inventory filter behavior")
        else:
            checked = validate_partition(reference, launches, kernels)
            checked_sets.append(checked)
            reports.append(dict(kernels=kernels, checked_launches=len(checked), result=result))
        records = [json.loads(line) for line in paths["model"].read_text().splitlines()]
        for record in records:
            if record["cache_mode"] != "reuse":
                raise ValueError("candidate cache mode differs from command")
            if record.get("binary_sha256", manifest["binary"]["sha256"]) != manifest["binary"]["sha256"]:
                raise ValueError("candidate output executable identity differs")
            record["binary_sha256"] = hashes[str(binary)]
            actual_config = record["engine_config"]
            if any(actual_config.get(key) != value for key, value in expected_config.items()):
                raise ValueError("candidate engine configuration differs")
            if candidate_config is not None and actual_config != candidate_config:
                raise ValueError("candidate execution configuration changed across runs")
            candidate_config = actual_config
        all_inputs.append((str(paths["model"]), records))
    require_full_coverage(reference, checked_sets)
    regression = compare(all_inputs, workloads, 1)
    return dict(passed=True, covered_launches=len(reference),
                covered_kernels=len({item["kernel"] for item in reference}),
                partitions=reports, regression=regression, input_sha256=hashes,
                scope="Aggregate complete launch coverage; not a single unfiltered process.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(json.loads(args.manifest.read_text()), args.manifest.resolve().parent)
    result["manifest_sha256"] = file_sha(args.manifest)
    result["verifier_sha256"] = file_sha(Path(__file__))
    result["regression_verifier_sha256"] = file_sha(Path(__file__).with_name("check_batch_regression.py"))
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    print(f"Complete coverage: {result['covered_launches']} launches, {result['covered_kernels']} kernels")


if __name__ == "__main__":
    main()
