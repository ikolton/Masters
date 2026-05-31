#!/usr/bin/env bash
#SBATCH --job-name=merlin-radgpt-score
#SBATCH --output=/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/logs/%x-%j.out
#SBATCH --account=plgmmia-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00

# RadGPT SCORING ONLY for specified Merlin ablation runs, 8B judge.
# This job writes RadGPT results to its own sidecar OUTPUT_DIR and DOES NOT touch
# any runs/*/evaluation.json. That makes it safe to run in parallel with the
# sampled-GREEN job (which does write evaluation.json). A separate merge job
# (merge_merlin_radgpt_into_summary.sbatch) attaches these results afterwards.
#
# Required env vars (set via --export or sourced before sbatch):
#   BENCHMARK_DIR       path to the benchmark output directory
#   RADGPT_OUTPUT_DIR   deterministic sidecar dir for this RadGPT run
#   RUN_LABELS          comma-separated run labels to evaluate (default: all)
# Optional:
#   STUDY_LIMIT     (default 0 = full set)
#   API_CONCURRENCY (default 4)

set -eo pipefail
set -x

BENCHMARK_DIR="${BENCHMARK_DIR:?BENCHMARK_DIR is required}"
RADGPT_OUTPUT_DIR="${RADGPT_OUTPUT_DIR:?RADGPT_OUTPUT_DIR is required}"
RUN_LABELS="${RUN_LABELS:-}"
STUDY_LIMIT="${STUDY_LIMIT:-0}"
API_CONCURRENCY="${API_CONCURRENCY:-4}"
SEED="${SEED:-13}"
MODEL="${MODEL:-iqbalamo93/Meta-Llama-3.1-8B-Instruct-GPTQ-Q_8}"
PORT="${PORT:-8012}"

RADGPT_ENV_DIR="${RADGPT_ENV_DIR:-/net/scratch/hscra/plgrid/plgikolton/Magisterka/.venv-radgpt-vllm}"
RADGPT_ROOT="${RADGPT_ROOT:-/net/scratch/hscra/plgrid/plgikolton/Magisterka/RadGPT}"
MASTERS="/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters"

source /etc/profile >/dev/null 2>&1 || true
module purge >/dev/null 2>&1 || true
module load ML-bundle/24.06a

if [[ ! -x "${RADGPT_ENV_DIR}/bin/python" ]]; then
  echo "RADGPT_ENV_DIR missing: ${RADGPT_ENV_DIR}" >&2; exit 3
fi

mkdir -p "${MASTERS}/logs"
cd "${MASTERS}"

export PYTHONPATH="${MASTERS}/src:${PYTHONPATH:-}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/net/scratch/hscra/plgrid/plgikolton/pip_cache}"
export HF_HOME="${HF_HOME:-/net/scratch/hscra/plgrid/plgikolton/hf_models}"
export TORCH_HOME="${TORCH_HOME:-/net/scratch/hscra/plgrid/plgikolton/torch_cache}"

RUN_ID="$(basename "${RADGPT_OUTPUT_DIR}")"
OUTPUT_DIR="${RADGPT_OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# NOTE: no --attach-full-to-evaluations -> evaluation.json is never written here.
RG_ARGS=(
  --benchmark-dir "${BENCHMARK_DIR}"
  --output-dir    "${OUTPUT_DIR}"
  --run-id        "${RUN_ID}"
  --study-limit   "${STUDY_LIMIT}"
  --seed          "${SEED}"
  --base-url      "http://127.0.0.1:${PORT}/v1"
  --radgpt-root   "${RADGPT_ROOT}"
  --fast
  --progress-every 25
  --api-concurrency "${API_CONCURRENCY}"
  --launch-vllm
  --vllm-python   "${RADGPT_ENV_DIR}/bin/python"
  --model         "${MODEL}"
  --cuda-visible-devices 0
  --tensor-parallel-size 1
  --dtype half
  --gpu-memory-utilization 0.90
  --max-model-len 8192
)
if [[ -n "${RUN_LABELS}" ]]; then
  RG_ARGS+=(--run-labels "${RUN_LABELS}")
fi

"${RADGPT_ENV_DIR}/bin/python" apps/run_radgpt_benchmark_from_generations.py "${RG_ARGS[@]}"

echo "RadGPT scoring complete. Sidecar written to: ${OUTPUT_DIR}"
echo "Run merge_merlin_radgpt_into_summary.sbatch to attach + refresh the summary."
