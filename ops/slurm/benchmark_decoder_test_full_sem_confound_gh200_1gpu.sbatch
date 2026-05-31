#!/usr/bin/env bash
#SBATCH --job-name=osc-test-sem-confound
#SBATCH --output=logs/%x-%j.out
#SBATCH --account=plgmmia-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00

# Decoder benchmark for the 3 semantic weight/variant confound-breaker runs:
#   semantic_family_subtype_w005  (family_subtype @ 0.05  vs existing @ 0.25)
#   semantic_minimal_w005         (minimal @ 0.05         vs existing @ 0.25)
#   sem_family_w025               (family-only @ 0.25     vs existing @ 0.05, control)
# Adds these 3 rows to benchmark_test_full_basic (18 rows total). RadGPT runs separately.

set -eo pipefail
set -x

source /etc/profile >/dev/null 2>&1 || true
module purge >/dev/null 2>&1 || true
module load ML-bundle/24.06a
source /net/scratch/hscra/plgrid/plgikolton/Magisterka/.venv/bin/activate

cd /net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters
mkdir -p logs

export PYTHONPATH=/net/scratch/hscra/plgrid/plgikolton/Magisterka/pycocoevalcap:/net/scratch/hscra/plgrid/plgikolton/Magisterka/RadEval:${PYTHONPATH}
export ORGAN_SEG_CLIP_LESION_METADATA_CSV=/net/scratch/hscra/plgrid/plgikolton/Magisterka/Merlin_metadata_hf_clean.csv

# Generate the 3 new eval configs for the test view.
python apps/build_decoder_test_eval_configs.py \
  --dataset-root outputs/datasets/merlin_test_converted_view \
  --output-dir outputs/decoder/benchmark_test_full_basic/configs \
  --feature-cache-dir outputs/decoder/feature_cache_merlin_test_full/best6_gh200 \
  --run semantic_family_subtype_w005::configs/decoder/train/train_qwen05b_per_organ_semantic_family_subtype_w005_v3_clean_best6_gh200_1gpu_bs64_10ep_sched_norepeat.yaml \
  --run semantic_minimal_w005::configs/decoder/train/train_qwen05b_per_organ_semantic_minimal_w005_v3_clean_best6_gh200_1gpu_bs64_10ep_sched_norepeat.yaml \
  --run sem_family_w025::configs/decoder/train/train_qwen05b_per_organ_sem_family_w025_v3_clean_best6_gh200_1gpu_bs64_10ep_sched_norepeat.yaml

# Pass 1: generate + basic-eval for the 3 new runs (historical runs are cache hits).
python -u apps/benchmark_decoder_checkpoints.py \
  --output-dir outputs/decoder/benchmark_test_full_basic \
  --split val \
  --study-fraction 0 \
  --seed 13 \
  --no-green \
  --no-study-level \
  --run semantic_family_subtype_w005::outputs/decoder/benchmark_test_full_basic/configs/semantic_family_subtype_w005.yaml::outputs/decoder/train_qwen05b_per_organ_semantic_family_subtype_w005_v3_clean_best6_gh200_1gpu_bs64_10ep_sched_norepeat/best.pt \
  --run semantic_minimal_w005::outputs/decoder/benchmark_test_full_basic/configs/semantic_minimal_w005.yaml::outputs/decoder/train_qwen05b_per_organ_semantic_minimal_w005_v3_clean_best6_gh200_1gpu_bs64_10ep_sched_norepeat/best.pt \
  --run sem_family_w025::outputs/decoder/benchmark_test_full_basic/configs/sem_family_w025.yaml::outputs/decoder/train_qwen05b_per_organ_sem_family_w025_v3_clean_best6_gh200_1gpu_bs64_10ep_sched_norepeat/best.pt

# Pass 2: refresh sampled GREEN (writes sampled_green into each evaluation.json; existing manifest reused).
python -u apps/evaluate_benchmark_sampled_green.py \
  --benchmark-dir outputs/decoder/benchmark_test_full_basic \
  --positive-count 1000 \
  --negative-count 1000 \
  --seed 13 \
  --green-batch-size 32 \
  --green-max-new-tokens 192 \
  --green-prompt-max-length 2048

# Pass 3: re-emit the comparison summary across ALL runs (15 historical + 3 new = 18).
python -u apps/benchmark_decoder_checkpoints.py \
  --output-dir outputs/decoder/benchmark_test_full_basic \
  --split val \
  --study-fraction 0 \
  --seed 13 \
  --no-green \
  --no-study-level \
  --run nodiag_ce_only::outputs/decoder/benchmark_test_full_basic/configs/nodiag_ce_only.yaml::outputs/decoder/train_qwen05b_per_organ_best6_gh200_1gpu_bs64_10ep_sched_norepeat_nodiag/best.pt \
  --run semantic_minimal_v3::outputs/decoder/benchmark_test_full_basic/configs/semantic_minimal_v3.yaml::outputs/decoder/train_qwen05b_per_organ_semantic_minimal_v3_clean_best6_gh200_1gpu_bs64_10ep_sched_norepeat/best.pt \
  --run semantic_family_subtype_v3::outputs/decoder/benchmark_test_full_basic/configs/semantic_family_subtype_v3.yaml::outputs/decoder/train_qwen05b_per_organ_semantic_family_subtype_v3_clean_best6_gh200_1gpu_bs64_10ep_sched_norepeat/best.pt \
  --run lexical_diag_v1_w001::outputs/decoder/benchmark_test_full_basic/configs/lexical_diag_v1_w001.yaml::outputs/decoder/train_qwen05b_per_organ_lexical_diag_v1_w001_best6_gh200_1gpu_bs64_10ep_sched_norepeat/best.pt \
  --run lexical_diag_v1_w002::outputs/decoder/benchmark_test_full_basic/configs/lexical_diag_v1_w002.yaml::outputs/decoder/train_qwen05b_per_organ_lexical_diag_v1_w002_best6_gh200_1gpu_bs64_10ep_sched_norepeat/best.pt \
  --run lexical_diag_v1_w005::outputs/decoder/benchmark_test_full_basic/configs/lexical_diag_v1_w005.yaml::outputs/decoder/train_qwen05b_per_organ_lexical_diag_v1_best6_gh200_1gpu_bs64_10ep_sched_norepeat/best.pt \
  --run lexw002_sem_normality_w002::outputs/decoder/benchmark_test_full_basic/configs/lexw002_sem_normality_w002.yaml::outputs/decoder/train_qwen05b_per_organ_lexw002_sem_normality_w002_v3_clean_best6_gh200_1gpu_bs64_10ep_sched_norepeat/best.pt \
  --run lexw002_sem_family_w002::outputs/decoder/benchmark_test_full_basic/configs/lexw002_sem_family_w002.yaml::outputs/decoder/train_qwen05b_per_organ_lexw002_sem_family_w002_v3_clean_best6_gh200_1gpu_bs64_10ep_sched_norepeat/best.pt \
  --run lexw002_sem_primary_secondary_w002::outputs/decoder/benchmark_test_full_basic/configs/lexw002_sem_primary_secondary_w002.yaml::outputs/decoder/train_qwen05b_per_organ_lexw002_sem_primary_secondary_w002_v3_clean_best6_gh200_1gpu_bs64_10ep_sched_norepeat/best.pt \
  --run sem_normality_w005::outputs/decoder/benchmark_test_full_basic/configs/sem_normality_w005.yaml::outputs/decoder/train_qwen05b_per_organ_sem_normality_w005_v3_clean_best6_gh200_1gpu_bs64_10ep_sched_norepeat/best.pt \
  --run sem_family_w005::outputs/decoder/benchmark_test_full_basic/configs/sem_family_w005.yaml::outputs/decoder/train_qwen05b_per_organ_sem_family_w005_v3_clean_best6_gh200_1gpu_bs64_10ep_sched_norepeat/best.pt \
  --run sem_primary_secondary_w005::outputs/decoder/benchmark_test_full_basic/configs/sem_primary_secondary_w005.yaml::outputs/decoder/train_qwen05b_per_organ_sem_primary_secondary_w005_v3_clean_best6_gh200_1gpu_bs64_10ep_sched_norepeat/best.pt \
  --run concept_lexical_v1_5_w001::outputs/decoder/benchmark_test_full_basic/configs/concept_lexical_v1_5_w001.yaml::outputs/decoder/train_qwen05b_per_organ_concept_lexical_v1_5_w001_best6_gh200_1gpu_bs64_10ep_sched_norepeat/best.pt \
  --run concept_lexical_v1_5_w002::outputs/decoder/benchmark_test_full_basic/configs/concept_lexical_v1_5_w002.yaml::outputs/decoder/train_qwen05b_per_organ_concept_lexical_v1_5_w002_best6_gh200_1gpu_bs64_10ep_sched_norepeat/best.pt \
  --run concept_lexical_v1_5_w005::outputs/decoder/benchmark_test_full_basic/configs/concept_lexical_v1_5_w005.yaml::outputs/decoder/train_qwen05b_per_organ_concept_lexical_v1_5_w005_best6_gh200_1gpu_bs64_10ep_sched_norepeat/best.pt \
  --run semantic_family_subtype_w005::outputs/decoder/benchmark_test_full_basic/configs/semantic_family_subtype_w005.yaml::outputs/decoder/train_qwen05b_per_organ_semantic_family_subtype_w005_v3_clean_best6_gh200_1gpu_bs64_10ep_sched_norepeat/best.pt \
  --run semantic_minimal_w005::outputs/decoder/benchmark_test_full_basic/configs/semantic_minimal_w005.yaml::outputs/decoder/train_qwen05b_per_organ_semantic_minimal_w005_v3_clean_best6_gh200_1gpu_bs64_10ep_sched_norepeat/best.pt \
  --run sem_family_w025::outputs/decoder/benchmark_test_full_basic/configs/sem_family_w025.yaml::outputs/decoder/train_qwen05b_per_organ_sem_family_w025_v3_clean_best6_gh200_1gpu_bs64_10ep_sched_norepeat/best.pt

python apps/build_decoder_generation_examples.py \
  --benchmark-dir outputs/decoder/benchmark_test_full_basic \
  --max-examples 44 \
  --per-organ 4 \
  --prefer-disagreements
