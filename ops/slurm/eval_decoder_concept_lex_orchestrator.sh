#!/usr/bin/env bash
# Submit the full decoder concept_lexical eval DAG.
# Usage: bash ops/slurm/eval_decoder_concept_lex_orchestrator.sh TRAIN_W001 TRAIN_W002 TRAIN_W005
#
# Example:
#   bash ops/slurm/eval_decoder_concept_lex_orchestrator.sh 17735011 17735012 17735013
#
# DAG:
#   D1 (gen+eval+sampled_GREEN+refresh) → afterok:TRAIN_W001:TRAIN_W002:TRAIN_W005
#   D2 (RadGPT 8B + attach + refresh)   → afterok:D1
set -euo pipefail

TRAIN_W001="${1:?Usage: $0 TRAIN_W001 TRAIN_W002 TRAIN_W005}"
TRAIN_W002="${2:?}"
TRAIN_W005="${3:?}"

MASTERS="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${MASTERS}"

D1=$(sbatch --parsable \
  --dependency=afterok:${TRAIN_W001}:${TRAIN_W002}:${TRAIN_W005} \
  --kill-on-invalid-dep=yes \
  ops/slurm/benchmark_decoder_test_full_concept_lex_gh200_1gpu.sbatch)

D2=$(sbatch --parsable \
  --dependency=afterok:${D1} \
  --kill-on-invalid-dep=yes \
  ops/slurm/run_radgpt_decoder_concept_lex_gh200_1gpu.sbatch)

echo "Decoder concept_lexical eval DAG:"
echo "  D1  ${D1}  gen+eval+sampled_GREEN+refresh  [afterok:${TRAIN_W001}:${TRAIN_W002}:${TRAIN_W005}]"
echo "  D2  ${D2}  RadGPT 8B + attach + refresh    [afterok:${D1}]"
