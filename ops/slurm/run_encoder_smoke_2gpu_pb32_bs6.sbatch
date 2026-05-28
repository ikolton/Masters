#!/usr/bin/env bash
#SBATCH --job-name=osc-2g-smoke-pb32-bs6
#SBATCH --output=logs/%x-%j.out
#SBATCH --account=plgmmia-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=04:00:00

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
echo "[slurm] entering 2gpu python smoke run"

torchrun --standalone --nproc_per_node=2 apps/run_encoder_smoke.py \
  --config configs/encoder/train/organsegclip_128x48_20ep_uniformtext_schedcos.yaml \
  --output-dir outputs/encoder/smoke/gh200_2gpu_pb32_bs6 \
  --train-limit 256 \
  --val-limit 64 \
  --epochs 1 \
  --max-train-steps 40 \
  --max-val-steps 16 \
  --batch-size 6 \
  --patch-batch-size 32 \
  --num-workers 2
