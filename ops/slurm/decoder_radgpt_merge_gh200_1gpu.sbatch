#!/usr/bin/env bash
#SBATCH --job-name=osc-dec-radgpt-merge
#SBATCH --output=/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/logs/%x-%j.out
#SBATCH --account=plgmmia-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

# Attach RadGPT sidecar results into runs/*/evaluation.json, refresh the decoder
# comparison summary, and rebuild generation examples. Run this AFTER both the
# sampled-GREEN job and the RadGPT scoring job finish
# (chain with --dependency=afterok:<sgreen>:<scoring>). It is the single writer of
# evaluation.json at this point, so there is no race. Decoder twin of
# merge_merlin_radgpt_into_summary.sbatch (plus the generation-examples rebuild).
#
# Required env:
#   BENCHMARK_DIR       decoder benchmark dir (contains runs/*/evaluation.json)
#   RADGPT_OUTPUT_DIR   sidecar dir produced by decoder_radgpt_scoring_gh200_1gpu.sbatch
# Optional env:
#   TARGET_KEY          evaluation.json key for RadGPT (default radgpt_oncology)
#   BUILD_EXAMPLES      1 to rebuild generation examples (default 1)
#   EXAMPLES_MAX        --max-examples (default 44)
#   EXAMPLES_PER_ORGAN  --per-organ (default 4)

set -eo pipefail
set -x

BENCHMARK_DIR="${BENCHMARK_DIR:?BENCHMARK_DIR is required}"
RADGPT_OUTPUT_DIR="${RADGPT_OUTPUT_DIR:?RADGPT_OUTPUT_DIR is required}"
TARGET_KEY="${TARGET_KEY:-radgpt_oncology}"
BUILD_EXAMPLES="${BUILD_EXAMPLES:-1}"
EXAMPLES_MAX="${EXAMPLES_MAX:-44}"
EXAMPLES_PER_ORGAN="${EXAMPLES_PER_ORGAN:-4}"

MAIN_ENV_DIR="${MAIN_ENV_DIR:-/net/scratch/hscra/plgrid/plgikolton/Magisterka/.venv}"
MASTERS="/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters"

source /etc/profile >/dev/null 2>&1 || true
module purge >/dev/null 2>&1 || true
module load ML-bundle/24.06a

if [[ ! -x "${MAIN_ENV_DIR}/bin/python" ]]; then
  echo "MAIN_ENV_DIR missing: ${MAIN_ENV_DIR}" >&2; exit 3
fi

cd "${MASTERS}"
source "${MAIN_ENV_DIR}/bin/activate"
export PYTHONPATH="${MASTERS}/src:/net/scratch/hscra/plgrid/plgikolton/Magisterka/pycocoevalcap:/net/scratch/hscra/plgrid/plgikolton/Magisterka/RadEval:${PYTHONPATH:-}"
export ORGAN_SEG_CLIP_LESION_METADATA_CSV="${ORGAN_SEG_CLIP_LESION_METADATA_CSV:-/net/scratch/hscra/plgrid/plgikolton/Magisterka/Merlin_metadata_hf_clean.csv}"

python apps/attach_radgpt_benchmark_results.py \
  --benchmark-dir     "${BENCHMARK_DIR}" \
  --radgpt-output-dir "${RADGPT_OUTPUT_DIR}" \
  --target-key        "${TARGET_KEY}"

python apps/refresh_decoder_benchmark_summary.py \
  --benchmark-dir "${BENCHMARK_DIR}"

if [[ "${BUILD_EXAMPLES}" == "1" ]]; then
  python apps/build_decoder_generation_examples.py \
    --benchmark-dir "${BENCHMARK_DIR}" \
    --max-examples "${EXAMPLES_MAX}" \
    --per-organ "${EXAMPLES_PER_ORGAN}" \
    --prefer-disagreements
fi

echo "RadGPT merged + summary refreshed for ${BENCHMARK_DIR}"
