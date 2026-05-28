#!/usr/bin/env bash
#SBATCH --job-name=merlin-radgpt-attach
#SBATCH --output=/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/logs/%x-%j.out
#SBATCH --account=plgmmia-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -eo pipefail
set -x

BENCHMARK_DIR="${BENCHMARK_DIR:-/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/outputs/models_ablations/merlin/benchmark_test_full_real5_basic_bs64}"
RADGPT_OUTPUT_DIR="${RADGPT_OUTPUT_DIR:?Set RADGPT_OUTPUT_DIR to the completed RadGPT output directory.}"
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

python -u apps/attach_radgpt_benchmark_results.py \
  --benchmark-dir "${BENCHMARK_DIR}" \
  --radgpt-output-dir "${RADGPT_OUTPUT_DIR}" \
  --target-key radgpt_oncology

python -u apps/refresh_decoder_benchmark_summary.py \
  --benchmark-dir "${BENCHMARK_DIR}"
