#!/usr/bin/env bash
set -euo pipefail

mkdir -p environment

python --version > environment/python-version.txt
python -m pip freeze > environment/pip-freeze.txt

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > environment/nvidia-smi.txt
else
  echo "nvidia-smi not available on this machine." > environment/nvidia-smi.txt
fi

if command -v conda >/dev/null 2>&1; then
  conda env export --no-builds > environment/conda-environment.yml
fi

echo "Environment information saved under environment/."
