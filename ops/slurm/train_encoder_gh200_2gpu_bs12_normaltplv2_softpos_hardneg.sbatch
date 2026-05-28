#!/usr/bin/env bash
#SBATCH --job-name=osc-nrmtpl-hnsp
#SBATCH --output=logs/%x-%j.out
#SBATCH --account=plgmmia-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00

set -eo pipefail
set -x

source /etc/profile >/dev/null 2>&1 || true
echo "[slurm] sourced /etc/profile"
module load ML-bundle/24.06a
echo "[slurm] loaded ML-bundle"
export ORGAN_SEG_CLIP_DATASET_ROOT=/net/storage/pr3/plgrid/plggjmiag/Merlin_converted
export ORGAN_SEG_CLIP_LESION_METADATA_CSV=/net/scratch/hscra/plgrid/plgikolton/Magisterka/Merlin_metadata_hf_clean.csv
source /net/scratch/hscra/plgrid/plgikolton/Magisterka/.venv/bin/activate
echo "[slurm] activated .venv"

cd /net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters
mkdir -p logs
echo "[slurm] starting normaltplv2 + soft_pos + hard_neg from scratch"

torchrun --standalone --nproc_per_node=2 apps/train_encoder.py \
  --config configs/encoder/train/organsegclip_128x48_20ep_gh200_2gpu_bs12_normaltplv2_softpos_hardneg.yaml
