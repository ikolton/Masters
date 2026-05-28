#!/usr/bin/env bash
#SBATCH --job-name=osc-decoder-concept-lex
#SBATCH --output=logs/%x-%j.out
#SBATCH --account=plgmmia-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00

set -eo pipefail
set -x

CONFIG_PATH="${1:?Usage: sbatch train_decoder_concept_lexical_v1_5_gh200_1gpu_bs64_10ep.sbatch <config-path>}"

source /etc/profile >/dev/null 2>&1 || true
module load ML-bundle/24.06a
export ORGAN_SEG_CLIP_DATASET_ROOT=/net/storage/pr3/plgrid/plggjmiag/Merlin_converted
export ORGAN_SEG_CLIP_LESION_METADATA_CSV=/net/scratch/hscra/plgrid/plgikolton/Magisterka/Merlin_metadata_hf_clean.csv
source /net/scratch/hscra/plgrid/plgikolton/Magisterka/.venv/bin/activate

cd /net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters
mkdir -p logs

python apps/train_decoder.py --config "${CONFIG_PATH}"
