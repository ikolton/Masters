#!/usr/bin/env bash
#SBATCH --job-name=osc-decoder-reuse
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
DECODER_CONFIG="${DECODER_CONFIG:?Set DECODER_CONFIG to a decoder YAML config path.}"

if [[ ! -f "${VISUAL_EXPORT}" ]]; then
  python apps/export_visual_encoder.py \
    --config "${ENCODER_CONFIG}" \
    --checkpoint "${ENCODER_BEST}" \
    --output "${VISUAL_EXPORT}" \
    --map-location cpu
fi

FEATURE_CACHE_DIR="$(python - "${DECODER_CONFIG}" <<'PY'
import sys
from pathlib import Path
import yaml

config_path = Path(sys.argv[1]).expanduser().resolve()
payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
cache_dir = str(payload.get("paths", {}).get("feature_cache_dir", "")).strip()
if not cache_dir:
    output_dir = Path(payload.get("paths", {}).get("output_dir", "")).expanduser()
    if not output_dir.is_absolute():
        output_dir = (config_path.parent / output_dir).resolve()
    cache_dir = str(output_dir / "features")
print(cache_dir)
PY
)"

if [[ -f "${FEATURE_CACHE_DIR}/train_features.pt" && -f "${FEATURE_CACHE_DIR}/val_features.pt" ]]; then
  echo "[decoder] reusing feature cache: ${FEATURE_CACHE_DIR}"
else
  echo "[decoder] feature cache missing or incomplete, building: ${FEATURE_CACHE_DIR}"
  python apps/cache_decoder_features.py --config "${DECODER_CONFIG}"
fi

python apps/train_decoder.py --config "${DECODER_CONFIG}"
