#!/usr/bin/env bash
#SBATCH --job-name=merlin-metric4
#SBATCH --output=/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/logs/%x-%j.out
#SBATCH --account=plgmmia-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=220G
#SBATCH --time=48:00:00

set -eo pipefail
set -x

LABEL=""
CONFIG=""
CHECKPOINT=""
TEST_COMBINED_JSON="/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/outputs/datasets/merlin_test_native/test/combined.json"
IMAGE_CACHE_DIR="/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/outputs/models_ablations/merlin/image_embedding_cache_merlin_test_native"
OUTPUT_DIR=""
BATCH_SIZE=32
MAX_NEW_TOKENS=128
NUM_SHARDS=4
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --label) LABEL="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --test-combined-json) TEST_COMBINED_JSON="$2"; shift 2 ;;
    --image-cache-dir) IMAGE_CACHE_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --num-shards) NUM_SHARDS="$2"; shift 2 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "${LABEL}" || -z "${CONFIG}" || -z "${CHECKPOINT}" || -z "${OUTPUT_DIR}" ]]; then
  echo "Required: --label --config --checkpoint --output-dir" >&2
  exit 2
fi
if [[ "${NUM_SHARDS}" -ne 4 ]]; then
  echo "This wrapper expects --num-shards 4 to match --gres=gpu:4" >&2
  exit 2
fi

source /etc/profile >/dev/null 2>&1 || true
module load ML-bundle/24.06a
MERLIN_ABLATION_VENV="${MERLIN_ABLATION_VENV:-/net/scratch/hscra/plgrid/plgikolton/Magisterka/.venv-merlin-ablation}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/net/scratch/hscra/plgrid/plgikolton/pip_cache}"
export PYTHONUSERBASE="${PYTHONUSERBASE:-/net/scratch/hscra/plgrid/plgikolton/pip_lib}"
export HF_HOME="${HF_HOME:-/net/scratch/hscra/plgrid/plgikolton/hf_models}"
export TORCH_HOME="${TORCH_HOME:-/net/scratch/hscra/plgrid/plgikolton/torch_cache}"
export PYTHONPATH="/net/scratch/hscra/plgrid/plgikolton/Magisterka/pycocoevalcap:/net/scratch/hscra/plgrid/plgikolton/Magisterka/RadEval:${PYTHONPATH:-}"
if [[ ! -x "${MERLIN_ABLATION_VENV}/bin/python" ]]; then
  echo "MERLIN_ABLATION_VENV does not point to a usable venv: ${MERLIN_ABLATION_VENV}" >&2
  exit 2
fi
source "${MERLIN_ABLATION_VENV}/bin/activate"

cd /net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/models_ablations/merlin
mkdir -p /net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/logs

pids=()
for shard in 0 1 2 3; do
  (
    export CUDA_VISIBLE_DEVICES="${shard}"
    python -u apps/benchmark_merlin_ablation_run.py \
      --label "${LABEL}_shard${shard}of4" \
      --config "${CONFIG}" \
      --checkpoint "${CHECKPOINT}" \
      --test-combined-json "${TEST_COMBINED_JSON}" \
      --image-cache-dir "${IMAGE_CACHE_DIR}" \
      --output-dir "${OUTPUT_DIR}" \
      --batch-size "${BATCH_SIZE}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --num-shards 4 \
      --shard-index "${shard}" \
      --skip-eval \
      "${EXTRA_ARGS[@]}"
  ) &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

merge_args=()
for shard in 0 1 2 3; do
  merge_args+=(--shard-label "${LABEL}_shard${shard}of4")
done
python -u apps/merge_merlin_benchmark_shards.py \
  --output-dir "${OUTPUT_DIR}" \
  --merged-label "${LABEL}" \
  "${merge_args[@]}"
