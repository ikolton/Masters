#!/usr/bin/env bash
#SBATCH --job-name=merlin-radgpt-merge
#SBATCH --output=/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/logs/%x-%j.out
#SBATCH --account=plgmmia-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

# Attach RadGPT sidecar results into runs/*/evaluation.json and refresh the
# comparison summary. Run this AFTER both the sampled-GREEN job and the RadGPT
# scoring job have finished (chain with --dependency=afterok:<sgreen>:<scoring>).
# It is the single writer of evaluation.json at this point, so there is no race.
#
# Required env vars:
#   BENCHMARK_DIR       path to the benchmark output directory
#   RADGPT_OUTPUT_DIR   sidecar dir produced by run_merlin_radgpt_scoring_gh200_1gpu.sbatch

set -eo pipefail
set -x

BENCHMARK_DIR="${BENCHMARK_DIR:?BENCHMARK_DIR is required}"
RADGPT_OUTPUT_DIR="${RADGPT_OUTPUT_DIR:?RADGPT_OUTPUT_DIR is required}"
TARGET_KEY="${TARGET_KEY:-radgpt_oncology}"

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

python apps/attach_radgpt_benchmark_results.py \
  --benchmark-dir     "${BENCHMARK_DIR}" \
  --radgpt-output-dir "${RADGPT_OUTPUT_DIR}" \
  --target-key        "${TARGET_KEY}"

python apps/refresh_decoder_benchmark_summary.py \
  --benchmark-dir "${BENCHMARK_DIR}"

echo "RadGPT results merged and comparison summary refreshed."
