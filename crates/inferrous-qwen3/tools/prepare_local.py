"""Prepare the pinned checkpoint, independent fixtures, and benchmark workloads."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from huggingface_hub import snapshot_download


TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parents[2]
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def destination(variable, default):
    return Path(os.environ.get(variable, REPO / "clean-work" / default)).resolve()


def prepare(output, marker, script, *arguments):
    if (output / marker).is_file():
        print(f"Reusing {output}", flush=True)
        return
    if output.exists():
        raise SystemExit(f"Incomplete output at {output}; move it aside before retrying.")
    subprocess.run([sys.executable, str(TOOLS / script), *map(str, arguments)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Existing pinned Hugging Face snapshot; otherwise download it")
    args = parser.parse_args()
    bundle = destination("CLEAN_MODEL_BUNDLE", "model-bundle")
    fixtures = destination("CLEAN_OPERATOR_FIXTURES", "operator-fixtures")
    workloads = destination("CLEAN_WORKLOADS", "workloads")
    if not (bundle / "manifest.json").is_file():
        if bundle.exists():
            raise SystemExit(f"Incomplete output at {bundle}; move it aside before retrying.")
        source = args.source
        if source is None:
            source = Path(snapshot_download(
                "Qwen/Qwen3-8B", revision=REVISION,
                cache_dir=REPO / "clean-work" / "hf-cache",
                allow_patterns=["*.json", "*.safetensors"],
            ))
        prepare(bundle, "manifest.json", "prepare.py", "--source", source.resolve(), "--output", bundle)
    else:
        print(f"Reusing {bundle}", flush=True)
    prepare(fixtures, "manifest.json", "operator_fixtures.py", fixtures)
    prepare(workloads, "requests.json", "workloads.py", bundle, workloads)


if __name__ == "__main__":
    main()
