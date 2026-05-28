#!/usr/bin/env bash
# Run quick RadGPT over the exact sampled-GREEN organ-row manifest.

set -euo pipefail

ROOT="${ROOT:-/net/scratch/hscra/plgrid/plgikolton/Magisterka}"
MASTERS="${MASTERS:-${ROOT}/Masters}"
MAIN_ENV_DIR="${MAIN_ENV_DIR:-${ROOT}/.venv}"
BENCHMARK_DIR="${BENCHMARK_DIR:-${MASTERS}/outputs/decoder/benchmark_test_full_basic}"
RUN_ID="${RUN_ID:-sampled_green_radgpt_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${BENCHMARK_DIR}/radgpt_benchmark/${RUN_ID}}"

export MODEL="${MODEL:-iqbalamo93/Meta-Llama-3.1-8B-Instruct-GPTQ-Q_8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TP="${TP:-1}"
export STUDY_LIMIT="${STUDY_LIMIT:-0}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export PORT="${PORT:-8010}"
export API_CONCURRENCY="${API_CONCURRENCY:-8}"
export PROGRESS_EVERY="${PROGRESS_EVERY:-100}"
export RUN_ID
export OUTPUT_DIR
export BENCHMARK_DIR
export SAMPLE_MANIFEST="${SAMPLE_MANIFEST:-${BENCHMARK_DIR}/sampled_green/sample_manifest.json}"
export ATTACH_SAMPLED_TO_EVALUATIONS="${ATTACH_SAMPLED_TO_EVALUATIONS:-1}"

echo "[sampled-radgpt] benchmark_dir: ${BENCHMARK_DIR}"
echo "[sampled-radgpt] sample_manifest: ${SAMPLE_MANIFEST}"
echo "[sampled-radgpt] output_dir: ${OUTPUT_DIR}"
echo "[sampled-radgpt] model: ${MODEL}"
echo "[sampled-radgpt] cuda: ${CUDA_VISIBLE_DEVICES}"
echo "[sampled-radgpt] api_concurrency: ${API_CONCURRENCY}"

bash "${MASTERS}/ops/setup/run_radgpt_benchmark_local_gh200.sh"

echo "[sampled-radgpt] refreshing comparison summary"
if ! type module >/dev/null 2>&1; then
  source /usr/share/lmod/lmod/init/bash >/dev/null 2>&1
fi
module load ML-bundle/24.06a
source "${MAIN_ENV_DIR}/bin/activate"
python "${MASTERS}/apps/refresh_decoder_benchmark_summary.py" \
  --benchmark-dir "${BENCHMARK_DIR}"

echo "[sampled-radgpt] done"
echo "[sampled-radgpt] summary: ${BENCHMARK_DIR}/comparison_summary.md"
echo "[sampled-radgpt] sampled RadGPT summary: ${OUTPUT_DIR}/summary.md"
