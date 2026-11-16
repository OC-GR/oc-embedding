#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2023 Michael Feil
# Derived from Infinity 0.0.77 requirements_install_from_poetry.sh.
# Modified for oc-embedding: keep the lockfile's Torch/Torchvision versions and
# install CPU wheels before their dependants, without changing Infinity code.
set -euo pipefail

oc_python_command="${PYTHON:-python3}"
oc_extra_url=""
oc_no_root=0
oc_without=""
oc_with=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-root)
            oc_no_root=1
            shift
            ;;
        --without|--with)
            [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
            if [[ "$1" == --without ]]; then
                oc_without="$2"
            else
                oc_with="$2"
            fi
            shift 2
            ;;
        *)
            [[ -z "$oc_extra_url" ]] || { echo "Unexpected argument: $1" >&2; exit 2; }
            oc_extra_url="$1"
            shift
            ;;
    esac
done

[[ -n "$oc_extra_url" ]] || { echo "The CPU Torch index URL is required" >&2; exit 2; }

oc_export=(poetry export --without-hashes --format=requirements.txt --extras "${EXTRAS:-all}")
[[ -z "$oc_without" ]] || oc_export+=(--without "$oc_without")
[[ -z "$oc_with" ]] || oc_export+=(--with "$oc_with")
"${oc_export[@]}" > requirements.exported.txt

# Parse exported package pins instead of copying versions into this installer.
# Poetry may append environment markers after each version.
"$oc_python_command" - <<'PY'
import pathlib
import re

lines = pathlib.Path("requirements.exported.txt").read_text().splitlines()
pins = {}
remaining = []
for line in lines:
    match = re.match(r"^(torch|torchvision)==([^\s;]+)", line)
    if match:
        name, version = match.groups()
        if name in pins and pins[name] != version:
            raise SystemExit(f"Conflicting {name} versions in the exported lockfile")
        pins[name] = version
        continue
    if re.match(r"^(nvidia(?:[-_][A-Za-z0-9_.-]+)?|triton)==", line):
        continue
    remaining.append(line)

if set(pins) != {"torch", "torchvision"}:
    raise SystemExit("The all-extras lock export must contain Torch and Torchvision pins")

pathlib.Path("requirements.cpu-torch.txt").write_text(
    "".join(f"{name}=={pins[name]}\n" for name in ("torch", "torchvision"))
)
pathlib.Path("requirements.cpu-rest.txt").write_text("\n".join(remaining) + "\n")
PY

# Keep the upstream fresh-environment behavior for each dependency selection.
if [[ -d .venv ]]; then
    rm -rf .venv
fi
"$oc_python_command" -m venv .venv
oc_pip=(.venv/bin/python -m pip)

# CPU wheels exist for both amd64 and arm64. A version such as ==2.8.0 also
# matches the upstream +cpu local-version wheels where that suffix is used.
"${oc_pip[@]}" install --no-cache-dir --index-url "$oc_extra_url" \
    -r requirements.cpu-torch.txt
"${oc_pip[@]}" install --no-cache-dir --extra-index-url "$oc_extra_url" \
    -c requirements.cpu-torch.txt -r requirements.cpu-rest.txt

if [[ "$oc_no_root" -eq 0 ]]; then
    "${oc_pip[@]}" install --no-cache-dir --no-deps --editable .
fi

.venv/bin/python - <<'PY'
import torch
import torchvision

if torch.version.cuda is not None:
    raise SystemExit(f"Expected a CPU Torch wheel, found CUDA {torch.version.cuda}")
print(f"CPU Torch {torch.__version__}; Torchvision {torchvision.__version__}")
PY
