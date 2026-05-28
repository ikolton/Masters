#!/usr/bin/env bash
#SBATCH --job-name=merlin-sgreen
#SBATCH --output=/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/logs/%x-%j.out
#SBATCH --account=plgmmia-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=18:00:00

set -eo pipefail
set -x

BENCHMARK_DIR="${BENCHMARK_DIR:-/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/outputs/models_ablations/merlin/benchmark_test_full_real5_basic_bs64}"
STUDY_FRACTION="${STUDY_FRACTION:-0.10}"
STUDY_LIMIT="${STUDY_LIMIT:-}"
SEED="${SEED:-13}"
GREEN_BATCH_SIZE="${GREEN_BATCH_SIZE:-32}"
GREEN_MAX_NEW_TOKENS="${GREEN_MAX_NEW_TOKENS:-192}"
GREEN_PROMPT_MAX_LENGTH="${GREEN_PROMPT_MAX_LENGTH:-2048}"
RUN_LABELS="${RUN_LABELS:-}"
MAIN_ENV_DIR="${MAIN_ENV_DIR:-/net/scratch/hscra/plgrid/plgikolton/Magisterka/.venv}"

source /etc/profile >/dev/null 2>&1 || true
module purge >/dev/null 2>&1 || true
module load ML-bundle/24.06a

if [[ ! -x "${MAIN_ENV_DIR}/bin/python" ]]; then
  echo "MAIN_ENV_DIR does not point to a usable venv: ${MAIN_ENV_DIR}" >&2
  exit 2
fi
source "${MAIN_ENV_DIR}/bin/activate"

cd /net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters
mkdir -p logs

export PYTHONPATH=/net/scratch/hscra/plgrid/plgikolton/Magisterka/pycocoevalcap:/net/scratch/hscra/plgrid/plgikolton/Magisterka/RadEval:${PYTHONPATH:-}
export ORGAN_SEG_CLIP_LESION_METADATA_CSV=/net/scratch/hscra/plgrid/plgikolton/Magisterka/Merlin_metadata_hf_clean.csv

GREEN_ARGS=(
  --benchmark-dir "${BENCHMARK_DIR}"
  --study-fraction "${STUDY_FRACTION}"
  --seed "${SEED}"
  --green-batch-size "${GREEN_BATCH_SIZE}"
  --green-max-new-tokens "${GREEN_MAX_NEW_TOKENS}"
  --green-prompt-max-length "${GREEN_PROMPT_MAX_LENGTH}"
  --run-labels "${RUN_LABELS}"
)
if [[ -n "${STUDY_LIMIT}" ]]; then
  GREEN_ARGS+=(--study-limit "${STUDY_LIMIT}")
fi

python -u apps/evaluate_benchmark_sampled_green.py "${GREEN_ARGS[@]}"

python -u apps/refresh_decoder_benchmark_summary.py \
  --benchmark-dir "${BENCHMARK_DIR}"
