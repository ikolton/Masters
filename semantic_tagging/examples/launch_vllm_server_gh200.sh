#!/usr/bin/env bash
set -euo pipefail

module load ML-bundle/24.06a
source /net/scratch/hscra/plgrid/plgikolton/Magisterka/.venv-semantic-tagging-vllm/bin/activate

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 32768 \
  --host 0.0.0.0 \
  --port 8000
