#!/usr/bin/env bash
#SBATCH --job-name=osc-decoder-sched-diag05
#SBATCH --output=logs/%x-%j.out
#SBATCH --account=plgmmia-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
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

ENCODER_CONFIG=configs/encoder/train/organsegclip_128x48_20ep_uniformtext_schedcos_logitslr1e5_gh200_2gpu_bs12.yaml
ENCODER_BEST=outputs/encoder/organsegclip_128x48_20ep_uniformtext_schedcos_logitslr1e5_gh200_2gpu_bs12/best.pt
VISUAL_EXPORT=outputs/encoder/organsegclip_128x48_20ep_uniformtext_schedcos_logitslr1e5_gh200_2gpu_bs12/visual_encoder_best.pt
DECODER_CONFIG=configs/decoder/train/train_qwen05b_per_organ_binary_diag_strong_best6_gh200_1gpu_bs64_10ep_sched_norepeat_diag05.yaml

if [[ ! -f "${VISUAL_EXPORT}" ]]; then
  python apps/export_visual_encoder.py \
    --config "${ENCODER_CONFIG}" \
    --checkpoint "${ENCODER_BEST}" \
    --output "${VISUAL_EXPORT}" \
    --map-location cpu
fi

python apps/cache_decoder_features.py --config "${DECODER_CONFIG}"
python apps/train_decoder.py --config "${DECODER_CONFIG}"
