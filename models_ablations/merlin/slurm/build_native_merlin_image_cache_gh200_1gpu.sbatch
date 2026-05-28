#!/usr/bin/env bash
#SBATCH --job-name=merlin-native-cache
#SBATCH --output=/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/logs/%x-%j.out
#SBATCH --account=plgmmia-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=220G
#SBATCH --time=24:00:00

set -eo pipefail
set -x

CONFIG_PATH="${1:?Usage: sbatch build_native_merlin_image_cache_gh200_1gpu.sbatch <config-path> <image-root> <cache-dir> [extra cache args...]}"
IMAGE_ROOT="${2:?Usage: sbatch build_native_merlin_image_cache_gh200_1gpu.sbatch <config-path> <image-root> <cache-dir> [extra cache args...]}"
CACHE_DIR="${3:?Usage: sbatch build_native_merlin_image_cache_gh200_1gpu.sbatch <config-path> <image-root> <cache-dir> [extra cache args...]}"
shift 3

source /etc/profile >/dev/null 2>&1 || true
module load ML-bundle/24.06a
MERLIN_ABLATION_VENV="${MERLIN_ABLATION_VENV:-/net/scratch/hscra/plgrid/plgikolton/Magisterka/.venv-merlin-ablation}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/net/scratch/hscra/plgrid/plgikolton/pip_cache}"
export PYTHONUSERBASE="${PYTHONUSERBASE:-/net/scratch/hscra/plgrid/plgikolton/pip_lib}"
export HF_HOME="${HF_HOME:-/net/scratch/hscra/plgrid/plgikolton/hf_models}"
export TORCH_HOME="${TORCH_HOME:-/net/scratch/hscra/plgrid/plgikolton/torch_cache}"
if [[ ! -x "${MERLIN_ABLATION_VENV}/bin/python" ]]; then
  echo "MERLIN_ABLATION_VENV does not point to a usable venv: ${MERLIN_ABLATION_VENV}" >&2
  exit 2
fi
source "${MERLIN_ABLATION_VENV}/bin/activate"

cd /net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/models_ablations/merlin
mkdir -p /net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/logs

python -u apps/build_native_image_embedding_cache.py \
  --config "${CONFIG_PATH}" \
  --image-root "${IMAGE_ROOT}" \
  --cache-dir "${CACHE_DIR}" \
  "$@"
