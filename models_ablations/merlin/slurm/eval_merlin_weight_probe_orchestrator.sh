#!/usr/bin/env bash
# Chain the full eval suite for the 5 weight-probe runs onto their training jobs.
# Usage:
#   bash models_ablations/merlin/slurm/eval_merlin_weight_probe_orchestrator.sh \
#     <TRAIN_CL_W01> <TRAIN_CL_W02> <TRAIN_CL_W05> <TRAIN_LEX_W01> <TRAIN_LEX_W02>
#
# DAG (per run: benchmark afterok its training job; then one sGreen over all 5;
#      then one RadGPT over all 5 — sequential after sGreen to avoid evaluation.json races):
#   B1..B5  (4-GPU sharded benchmark)  -> afterok:TRAIN_i
#   SG      (sampled GREEN, all 5)     -> afterok:B1:B2:B3:B4:B5   (refreshes summary)
#   RG      (RadGPT 8B, all 5)         -> afterok:SG               (refreshes summary -> final table)
set -euo pipefail

TR_CL_W01="${1:?need train job id concept-lexical w01}"
TR_CL_W02="${2:?need train job id concept-lexical w02}"
TR_CL_W05="${3:?need train job id concept-lexical w05}"
TR_LEX_W01="${4:?need train job id lexical w01}"
TR_LEX_W02="${5:?need train job id lexical w02}"

MASTERS="/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters"
SLURM="${MASTERS}/models_ablations/merlin/slurm"
OUT="${MASTERS}/outputs/models_ablations/merlin"
BENCHMARK_DIR="${OUT}/benchmark_test_10pct_real2_eos_sharded"
BENCH_SBATCH="${SLURM}/benchmark_merlin_ablation_sharded_gh200_4gpu.sbatch"

cd "${MASTERS}"

# run_id (checkpoint dir + config stem)  |  table label  |  training job id
RUNS=(
  "real2_eos_concept_lexical_v1_5_w01_bs16_len512|concept-lexical-v1-5-w01-eos2|${TR_CL_W01}"
  "real2_eos_concept_lexical_v1_5_w02_bs16_len512|concept-lexical-v1-5-w02-eos2|${TR_CL_W02}"
  "real2_eos_concept_lexical_v1_5_w05_bs16_len512|concept-lexical-v1-5-w05-eos2|${TR_CL_W05}"
  "real2_eos_lexical_w01_bs16_len512|lexical-w01-eos2|${TR_LEX_W01}"
  "real2_eos_lexical_w02_bs16_len512|lexical-w02-eos2|${TR_LEX_W02}"
)

BENCH_IDS=()
LABELS=()
for entry in "${RUNS[@]}"; do
  IFS='|' read -r run_id label train_id <<< "${entry}"
  checkpoint="${OUT}/${run_id}/checkpoint_best.pt"
  config="configs/${run_id}.yaml"
  bid=$(sbatch --parsable \
    --dependency=afterok:${train_id} \
    --kill-on-invalid-dep=yes \
    "${BENCH_SBATCH}" \
      --label "${label}" \
      --config "${config}" \
      --checkpoint "${checkpoint}" \
      --output-dir "${BENCHMARK_DIR}")
  echo "  BENCH ${bid}  ${label}  [afterok:${train_id}]"
  BENCH_IDS+=("${bid}")
  LABELS+=("${label}")
done

# comma-joined labels (exported via env so sbatch --export=ALL keeps commas intact)
LABEL_CSV=$(IFS=','; echo "${LABELS[*]}")
BENCH_DEP=$(IFS=':'; echo "${BENCH_IDS[*]}")

export BENCHMARK_DIR
export RUN_LABELS="${LABEL_CSV}"

SG=$(sbatch --parsable \
  --dependency=afterok:${BENCH_DEP} \
  --kill-on-invalid-dep=yes \
  --export=ALL \
  "${SLURM}/evaluate_merlin_sampled_green_gh200_1gpu.sbatch")
echo "  SGREEN ${SG}  [afterok:${BENCH_DEP}]  labels=${LABEL_CSV}"

RG=$(sbatch --parsable \
  --dependency=afterok:${SG} \
  --kill-on-invalid-dep=yes \
  --export=ALL \
  "${SLURM}/run_merlin_radgpt_and_attach_gh200_1gpu.sbatch")
echo "  RADGPT ${RG}  [afterok:${SG}]  labels=${LABEL_CSV}  (final summary refresh)"

echo
echo "Weight-probe eval DAG submitted. Final table refresh happens at end of RadGPT job ${RG}."
