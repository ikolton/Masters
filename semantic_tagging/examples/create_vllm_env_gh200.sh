#!/usr/bin/env bash
set -euo pipefail

module load ML-bundle/24.06a
python3.11 -m venv /net/scratch/hscra/plgrid/plgikolton/Magisterka/.venv-semantic-tagging-vllm
source /net/scratch/hscra/plgrid/plgikolton/Magisterka/.venv-semantic-tagging-vllm/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install /net/software/aarch64/el9/wheels/ML-bundle/24.06a/torch-2.8.0+cu124-cp311-cp311-linux_aarch64.whl
python -m pip install /net/software/aarch64/el9/wheels/ML-bundle/24.06a/triton-3.4.0+git11ec6354-cp311-cp311-linux_aarch64.whl
python -m pip install /net/software/aarch64/el9/wheels/ML-bundle/24.06a/flash_attn-2.8.3-cp311-cp311-linux_aarch64.whl
python -m pip install /net/software/aarch64/el9/wheels/ML-bundle/24.06a/xformers-0.0.30+cu124-cp311-cp311-linux_aarch64.whl
python -m pip install /net/software/aarch64/el9/wheels/ML-bundle/24.06a/vllm-0.10.2+cu124-cp311-cp311-linux_aarch64.whl
python -m pip install requests PyYAML jsonschema pyarrow transformers sentencepiece accelerate
