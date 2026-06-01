#!/usr/bin/env bash
#SBATCH --job-name=osc-enc-v2m-j1
#SBATCH --output=/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/logs/%x-%j.out
#SBATCH --account=plgmmia-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=48:00:00

set -eo pipefail
set -x

source /etc/profile >/dev/null 2>&1 || true
module load ML-bundle/24.06a

export ORGAN_SEG_CLIP_DATASET_ROOT=/net/storage/pr3/plgrid/plggjmiag/Merlin_converted
export ORGAN_SEG_CLIP_LESION_METADATA_CSV=/net/scratch/hscra/plgrid/plgikolton/Magisterka/Merlin_metadata_hf_clean.csv

source /net/scratch/hscra/plgrid/plgikolton/Magisterka/.venv/bin/activate
cd /net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters
mkdir -p logs

echo "[v2m-j1] masked cross-study same-finding, no text freeze, early stopping patience=4"

torchrun --standalone --nproc_per_node=4 apps/train_encoder.py \
  --config configs/encoder/train/train_v2_15mm_k4_4gpu_masked.yaml
