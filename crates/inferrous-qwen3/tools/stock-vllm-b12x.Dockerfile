FROM vllm/vllm-openai@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34

# vLLM 0.32.0's declared b12x extra. Keep the official image's Torch, Triton,
# CUDA and model kernels intact; the package's own requirements are checked.
RUN python3 -m pip install --no-cache-dir --no-deps b12x==1.5.1

RUN python3 - <<'PY'
from importlib.metadata import requires, version
from packaging.markers import default_environment
from packaging.requirements import Requirement
environment = default_environment()
environment["extra"] = ""
for text in requires("b12x") or []:
    requirement = Requirement(text)
    if requirement.marker is None or requirement.marker.evaluate(environment):
        assert version(requirement.name) in requirement.specifier, text
PY
