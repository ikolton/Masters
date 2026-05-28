#!/usr/bin/env bash
# Submit the full Merlin concept_lexical eval DAG.
# Usage: bash models_ablations/merlin/slurm/eval_merlin_concept_lex_orchestrator.sh \
#          BENCH_W001_JOB BENCH_W002_JOB BENCH_W004_JOB
#
# Example (pass the already-queued benchmark job IDs):
#   bash models_ablations/merlin/slurm/eval_merlin_concept_lex_orchestrator.sh \
#     17736686 17736687 17736688
#
# DAG (all three runs proceed in parallel after their own benchmark job):
#   M_B1 (sampled_GREEN w001)  → afterok:BENCH_W001
#   M_B2 (sampled_GREEN w002)  → afterok:BENCH_W002
#   M_B3 (sampled_GREEN w004)  → afterok:BENCH_W004
#   M_C  (RadGPT 8B all 3)     → afterok:M_B1:M_B2:M_B3
set -euo pipefail

BENCH_W001="${1:?Usage: $0 BENCH_W001_JOB BENCH_W002_JOB BENCH_W004_JOB}"
BENCH_W002="${2:?}"
BENCH_W004="${3:?}"

MASTERS="$(cd "$(dirname "$0")/../../.." && pwd)"
MERLIN_SLURM="${MASTERS}/models_ablations/merlin/slurm"
BENCHMARK_DIR="${MASTERS}/outputs/models_ablations/merlin/benchmark_test_10pct_real2_eos_sharded"

cd "${MASTERS}"

# Sampled GREEN — one job per run, all independent, each afterok its benchmark job.
M_B1=$(sbatch --parsable \
  --dependency=afterok:${BENCH_W001} \
  --kill-on-invalid-dep=yes \
  --export=ALL,BENCHMARK_DIR=${BENCHMARK_DIR},RUN_LABELS=concept_lexical_v1_5_w001_eos2 \
  "${MERLIN_SLURM}/evaluate_merlin_sampled_green_gh200_1gpu.sbatch")

M_B2=$(sbatch --parsable \
  --dependency=afterok:${BENCH_W002} \
  --kill-on-invalid-dep=yes \
  --export=ALL,BENCHMARK_DIR=${BENCHMARK_DIR},RUN_LABELS=concept_lexical_v1_5_w002_eos2 \
  "${MERLIN_SLURM}/evaluate_merlin_sampled_green_gh200_1gpu.sbatch")

M_B3=$(sbatch --parsable \
  --dependency=afterok:${BENCH_W004} \
  --kill-on-invalid-dep=yes \
  --export=ALL,BENCHMARK_DIR=${BENCHMARK_DIR},RUN_LABELS=concept_lexical_v1_5_w004_eos2 \
  "${MERLIN_SLURM}/evaluate_merlin_sampled_green_gh200_1gpu.sbatch")

# RadGPT — one job after all three GREEN jobs complete.
M_C=$(sbatch --parsable \
  --dependency=afterok:${M_B1}:${M_B2}:${M_B3} \
  --kill-on-invalid-dep=yes \
  --export=ALL,BENCHMARK_DIR=${BENCHMARK_DIR},RUN_LABELS=concept_lexical_v1_5_w001_eos2,concept_lexical_v1_5_w002_eos2,concept_lexical_v1_5_w004_eos2 \
  "${MERLIN_SLURM}/run_merlin_radgpt_and_attach_gh200_1gpu.sbatch")

echo "Merlin concept_lexical eval DAG:"
echo "  M_B1  ${M_B1}  sampled_GREEN w001  [afterok:${BENCH_W001}]"
echo "  M_B2  ${M_B2}  sampled_GREEN w002  [afterok:${BENCH_W002}]"
echo "  M_B3  ${M_B3}  sampled_GREEN w004  [afterok:${BENCH_W004}]"
echo "  M_C   ${M_C}   RadGPT 8B + attach  [afterok:${M_B1}:${M_B2}:${M_B3}]"
