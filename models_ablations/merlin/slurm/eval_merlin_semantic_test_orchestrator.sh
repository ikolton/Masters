#!/usr/bin/env bash
# Parallel eval DAG for the semantic-family / mixed final test (3 runs).
# Usage:
#   bash models_ablations/merlin/slurm/eval_merlin_semantic_test_orchestrator.sh \
#     <TRAIN_SEM_W05> <TRAIN_SEM_W10> <TRAIN_MIX>
#
# DAG (RadGPT scoring runs IN PARALLEL with sampled-GREEN; merge waits on both):
#   B1..B3   (4-GPU sharded benchmark)   -> afterok:TRAIN_i
#   SG       (sampled GREEN, all 3)      -> afterok:B1:B2:B3      (writes GREEN -> evaluation.json)
#   RG-score (RadGPT 8B scoring, all 3)  -> afterok:B1:B2:B3      (sidecar only, no evaluation.json)
#   MERGE    (attach RadGPT + refresh)   -> afterok:SG:RG-score   (sole evaluation.json writer)
set -euo pipefail

TR_SEM_W05="${1:?need train job id sem-family-w05}"
TR_SEM_W10="${2:?need train job id sem-family-w10}"
TR_MIX="${3:?need train job id concept-w02-sem-family-w05}"

MASTERS="/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters"
SLURM="${MASTERS}/models_ablations/merlin/slurm"
OUT="${MASTERS}/outputs/models_ablations/merlin"
BENCHMARK_DIR="${OUT}/benchmark_test_10pct_real2_eos_sharded"
BENCH_SBATCH="${SLURM}/benchmark_merlin_ablation_sharded_gh200_4gpu.sbatch"
RADGPT_OUTPUT_DIR="${BENCHMARK_DIR}/radgpt_benchmark/semantic_test_parallel_$(date +%Y%m%d_%H%M%S)"

cd "${MASTERS}"

# run_id (checkpoint dir + config stem)  |  table label  |  training job id
RUNS=(
  "real2_eos_sem_family_w05_bs16_len512|sem-family-w05-eos2|${TR_SEM_W05}"
  "real2_eos_sem_family_w10_bs16_len512|sem-family-w10-eos2|${TR_SEM_W10}"
  "real2_eos_concept_w02_sem_family_w05_bs16_len512|concept-w02-sem-family-w05-eos2|${TR_MIX}"
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

LABEL_CSV=$(IFS=','; echo "${LABELS[*]}")
BENCH_DEP=$(IFS=':'; echo "${BENCH_IDS[*]}")

export BENCHMARK_DIR
export RUN_LABELS="${LABEL_CSV}"
export RADGPT_OUTPUT_DIR

# sampled GREEN (writes GREEN into evaluation.json + refresh)
SG=$(sbatch --parsable \
  --dependency=afterok:${BENCH_DEP} \
  --kill-on-invalid-dep=yes \
  --export=ALL \
  "${SLURM}/evaluate_merlin_sampled_green_gh200_1gpu.sbatch")
echo "  SGREEN ${SG}  [afterok:${BENCH_DEP}]  labels=${LABEL_CSV}"

# RadGPT scoring — PARALLEL to sGreen, writes only to sidecar (no evaluation.json)
RGS=$(sbatch --parsable \
  --dependency=afterok:${BENCH_DEP} \
  --kill-on-invalid-dep=yes \
  --export=ALL \
  "${SLURM}/run_merlin_radgpt_scoring_gh200_1gpu.sbatch")
echo "  RADGPT-SCORE ${RGS}  [afterok:${BENCH_DEP}]  (parallel to sGreen)  sidecar=${RADGPT_OUTPUT_DIR}"

# merge — sole evaluation.json writer, after BOTH finish
MERGE=$(sbatch --parsable \
  --dependency=afterok:${SG}:${RGS} \
  --kill-on-invalid-dep=yes \
  --export=ALL \
  "${SLURM}/merge_merlin_radgpt_into_summary.sbatch")
echo "  MERGE ${MERGE}  [afterok:${SG}:${RGS}]  (final summary refresh)"

echo
echo "Semantic-test eval DAG submitted. Final table refresh happens at end of MERGE job ${MERGE}."
