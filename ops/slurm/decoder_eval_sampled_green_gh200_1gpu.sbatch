#!/usr/bin/env bash
#SBATCH --job-name=osc-dec-sgreen
#SBATCH --output=/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/logs/%x-%j.out
#SBATCH --account=plgmmia-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=16:00:00

# Sampled-GREEN ONLY for a decoder benchmark directory. Writes `sampled_green`
# into each runs/*/evaluation.json via the file-locked updater in the script, so
# it is safe to run IN PARALLEL with the RadGPT scoring job (which writes only to
# its own sidecar and never touches evaluation.json). The merge job attaches
# RadGPT afterwards. This is the decoder twin of the Merlin sGreen step.
#
# Required env:
#   BENCHMARK_DIR     decoder benchmark dir (contains runs/*/generations.json)
# Optional env:
#   STUDY_FRACTION    fraction of common test studies to sample (default 0.10)
#   SEED              sampling seed (default 13)
#   GREEN_BATCH_SIZE  (default 32)
#   GREEN_MAX_NEW_TOKENS (default 192)
#   GREEN_PROMPT_MAX_LENGTH (default 2048)

set -eo pipefail
set -x

BENCHMARK_DIR="${BENCHMARK_DIR:?BENCHMARK_DIR is required}"
STUDY_FRACTION="${STUDY_FRACTION:-0.10}"
SEED="${SEED:-13}"
GREEN_BATCH_SIZE="${GREEN_BATCH_SIZE:-32}"
GREEN_MAX_NEW_TOKENS="${GREEN_MAX_NEW_TOKENS:-192}"
GREEN_PROMPT_MAX_LENGTH="${GREEN_PROMPT_MAX_LENGTH:-2048}"

MAIN_ENV_DIR="${MAIN_ENV_DIR:-/net/scratch/hscra/plgrid/plgikolton/Magisterka/.venv}"
MASTERS="/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters"

source /etc/profile >/dev/null 2>&1 || true
module purge >/dev/null 2>&1 || true
module load ML-bundle/24.06a
source "${MAIN_ENV_DIR}/bin/activate"

cd "${MASTERS}"
mkdir -p logs
export PYTHONPATH="${MASTERS}/src:/net/scratch/hscra/plgrid/plgikolton/Magisterka/pycocoevalcap:/net/scratch/hscra/plgrid/plgikolton/Magisterka/RadEval:${PYTHONPATH:-}"
export ORGAN_SEG_CLIP_LESION_METADATA_CSV="${ORGAN_SEG_CLIP_LESION_METADATA_CSV:-/net/scratch/hscra/plgrid/plgikolton/Magisterka/Merlin_metadata_hf_clean.csv}"

python -u apps/evaluate_benchmark_sampled_green.py \
  --benchmark-dir "${BENCHMARK_DIR}" \
  --study-fraction "${STUDY_FRACTION}" \
  --seed "${SEED}" \
  --green-batch-size "${GREEN_BATCH_SIZE}" \
  --green-max-new-tokens "${GREEN_MAX_NEW_TOKENS}" \
  --green-prompt-max-length "${GREEN_PROMPT_MAX_LENGTH}"

echo "Sampled-GREEN complete for ${BENCHMARK_DIR}"
