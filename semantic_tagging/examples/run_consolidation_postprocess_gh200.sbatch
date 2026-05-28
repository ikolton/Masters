#!/usr/bin/env bash
#SBATCH --job-name=tag-postprocess
#SBATCH --output=/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/semantic_tagging/logs/%x-%j.out
#SBATCH --account=plgmmia-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

source /etc/profile >/dev/null 2>&1 || true

set -eo pipefail

module load ML-bundle/24.06a
source /net/scratch/hscra/plgrid/plgikolton/Magisterka/.venv-semantic-tagging-vllm/bin/activate

PROJECT_ROOT=/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/semantic_tagging
CONFIG_PATH="${1:-configs/merlin_consolidation_postprocess_v3.yaml}"

set -x

cd "${PROJECT_ROOT}"
python -u apps/postprocess_consolidation.py --config "${CONFIG_PATH}"
