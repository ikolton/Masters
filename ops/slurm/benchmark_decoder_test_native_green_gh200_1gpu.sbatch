#!/usr/bin/env bash
#SBATCH --job-name=osc-test-native-green
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
source /net/scratch/hscra/plgrid/plgikolton/Magisterka/.venv/bin/activate

cd /net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters
mkdir -p logs

export PYTHONPATH=/net/scratch/hscra/plgrid/plgikolton/Magisterka/pycocoevalcap:/net/scratch/hscra/plgrid/plgikolton/Magisterka/RadEval:${PYTHONPATH}
export ORGAN_SEG_CLIP_LESION_METADATA_CSV=/net/scratch/hscra/plgrid/plgikolton/Magisterka/Merlin_metadata_hf_clean.csv

python apps/build_merlin_test_native_dataset.py
python apps/build_decoder_test_eval_configs.py

NATIVE_CACHE=outputs/decoder/feature_cache_merlin_test_native/best6_gh200
LEGACY_CACHE=outputs/decoder/feature_cache_merlin_test_full/best6_gh200
mkdir -p "${NATIVE_CACHE}"
if [[ ! -f "${NATIVE_CACHE}/test_features.pt" && -f "${LEGACY_CACHE}/val_features.pt" ]]; then
  cp "${LEGACY_CACHE}/val_features.pt" "${NATIVE_CACHE}/test_features.pt"
fi

python -u apps/benchmark_decoder_checkpoints.py \
  --output-dir outputs/decoder/benchmark_test_full_native \
  --split test \
  --study-fraction 0 \
  --seed 13 \
  --green-scope organ \
  --no-study-level \
  --green-batch-size 32 \
  --green-max-new-tokens 192 \
  --green-prompt-max-length 2048 \
  --run nodiag_ce_only::outputs/decoder/benchmark_test_full_native/configs/nodiag_ce_only.yaml::outputs/decoder/train_qwen05b_per_organ_best6_gh200_1gpu_bs64_10ep_sched_norepeat_nodiag/best.pt \
  --run semantic_minimal_v3::outputs/decoder/benchmark_test_full_native/configs/semantic_minimal_v3.yaml::outputs/decoder/train_qwen05b_per_organ_semantic_minimal_v3_clean_best6_gh200_1gpu_bs64_10ep_sched_norepeat/best.pt \
  --run semantic_family_subtype_v3::outputs/decoder/benchmark_test_full_native/configs/semantic_family_subtype_v3.yaml::outputs/decoder/train_qwen05b_per_organ_semantic_family_subtype_v3_clean_best6_gh200_1gpu_bs64_10ep_sched_norepeat/best.pt \
  --run lexical_diag_v1_w001::outputs/decoder/benchmark_test_full_native/configs/lexical_diag_v1_w001.yaml::outputs/decoder/train_qwen05b_per_organ_lexical_diag_v1_w001_best6_gh200_1gpu_bs64_10ep_sched_norepeat/best.pt \
  --run lexical_diag_v1_w002::outputs/decoder/benchmark_test_full_native/configs/lexical_diag_v1_w002.yaml::outputs/decoder/train_qwen05b_per_organ_lexical_diag_v1_w002_best6_gh200_1gpu_bs64_10ep_sched_norepeat/best.pt \
  --run lexical_diag_v1_w005::outputs/decoder/benchmark_test_full_native/configs/lexical_diag_v1_w005.yaml::outputs/decoder/train_qwen05b_per_organ_lexical_diag_v1_best6_gh200_1gpu_bs64_10ep_sched_norepeat/best.pt

python apps/build_decoder_generation_examples.py \
  --benchmark-dir outputs/decoder/benchmark_test_full_native \
  --max-examples 44 \
  --per-organ 4 \
  --prefer-disagreements
