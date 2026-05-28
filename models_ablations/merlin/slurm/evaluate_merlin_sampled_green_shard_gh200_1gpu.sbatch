#!/usr/bin/env bash
#SBATCH --job-name=merlin-sgreen-shard
#SBATCH --output=/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/logs/%x-%j.out
#SBATCH --account=plgmmia-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00

set -eo pipefail
set -x

BENCHMARK_DIR="${BENCHMARK_DIR:?BENCHMARK_DIR is required}"
RUN_LABEL="${RUN_LABEL:?RUN_LABEL is required}"
NUM_SHARDS="${NUM_SHARDS:?NUM_SHARDS is required}"
SHARD_INDEX="${SHARD_INDEX:?SHARD_INDEX is required}"
STUDY_FRACTION="${STUDY_FRACTION:-0.10}"
STUDY_LIMIT="${STUDY_LIMIT:-}"
SEED="${SEED:-13}"
GREEN_BATCH_SIZE="${GREEN_BATCH_SIZE:-32}"
GREEN_MAX_NEW_TOKENS="${GREEN_MAX_NEW_TOKENS:-192}"
GREEN_PROMPT_MAX_LENGTH="${GREEN_PROMPT_MAX_LENGTH:-2048}"
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

ARGS=(
  --benchmark-dir "${BENCHMARK_DIR}"
  --run-label "${RUN_LABEL}"
  --num-shards "${NUM_SHARDS}"
  --shard-index "${SHARD_INDEX}"
  --study-fraction "${STUDY_FRACTION}"
  --seed "${SEED}"
  --green-batch-size "${GREEN_BATCH_SIZE}"
  --green-max-new-tokens "${GREEN_MAX_NEW_TOKENS}"
  --green-prompt-max-length "${GREEN_PROMPT_MAX_LENGTH}"
)
if [[ -n "${STUDY_LIMIT}" ]]; then
  ARGS+=(--study-limit "${STUDY_LIMIT}")
fi

python -u apps/evaluate_benchmark_sampled_green_sharded.py "${ARGS[@]}"
