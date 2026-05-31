#!/usr/bin/env bash
#SBATCH --job-name=osc-dec-green-finish
#SBATCH --output=/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/logs/%x-%j.out
#SBATCH --account=plgmmia-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=02:00:00

# One-shot finisher: compute sampled-GREEN for the 2 runs that still lack it,
# ONE PER GPU IN PARALLEL, pinned to the existing sample_manifest so they score
# the IDENTICAL 5,643-row subset as the 16 runs already in the table. Then refresh
# the comparison summary + rebuild generation examples. RadGPT (18/18) is already
# attached, so this completes the decoder table.

set -eo pipefail
set -x

MASTERS="/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters"
MAIN_ENV_DIR="/net/scratch/hscra/plgrid/plgikolton/Magisterka/.venv"
BENCHMARK_DIR="${MASTERS}/outputs/decoder/benchmark_test_full_basic"
MANIFEST="${BENCHMARK_DIR}/sampled_green/sample_manifest.json"

source /etc/profile >/dev/null 2>&1 || true
module purge >/dev/null 2>&1 || true
module load ML-bundle/24.06a
source "${MAIN_ENV_DIR}/bin/activate"

cd "${MASTERS}"
mkdir -p logs
export PYTHONPATH="${MASTERS}/src:/net/scratch/hscra/plgrid/plgikolton/Magisterka/pycocoevalcap:/net/scratch/hscra/plgrid/plgikolton/Magisterka/RadEval:${PYTHONPATH:-}"
export ORGAN_SEG_CLIP_LESION_METADATA_CSV="/net/scratch/hscra/plgrid/plgikolton/Magisterka/Merlin_metadata_hf_clean.csv"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "ERROR: pinned manifest missing: ${MANIFEST}" >&2; exit 4
fi

run_green () {  # $1=gpu_id  $2=run_label
  CUDA_VISIBLE_DEVICES="$1" python -u apps/evaluate_benchmark_sampled_green.py \
    --benchmark-dir "${BENCHMARK_DIR}" \
    --sample-manifest "${MANIFEST}" \
    --run-labels "$2" \
    --green-batch-size 32 \
    --green-max-new-tokens 192 \
    --green-prompt-max-length 2048
}

# GPU 0 and GPU 1 in parallel, one missing run each.
run_green 0 semantic-family-subtype-w005 &
PID0=$!
run_green 1 semantic-minimal-w005 &
PID1=$!

FAIL=0
wait "${PID0}" || { echo "GREEN run on GPU0 (semantic-family-subtype-w005) FAILED"; FAIL=1; }
wait "${PID1}" || { echo "GREEN run on GPU1 (semantic-minimal-w005) FAILED"; FAIL=1; }
if [[ "${FAIL}" != "0" ]]; then
  echo "One or more GREEN runs failed; NOT refreshing summary." >&2
  exit 5
fi

# Both done -> refresh the table + rebuild examples (RadGPT already attached).
python apps/refresh_decoder_benchmark_summary.py --benchmark-dir "${BENCHMARK_DIR}"
python apps/build_decoder_generation_examples.py \
  --benchmark-dir "${BENCHMARK_DIR}" \
  --max-examples 44 \
  --per-organ 4 \
  --prefer-disagreements

echo "Decoder table complete: GREEN(18/18) + RadGPT(18/18). Summary refreshed."
