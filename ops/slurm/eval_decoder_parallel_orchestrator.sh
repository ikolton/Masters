#!/usr/bin/env bash
# Parallel eval DAG for a decoder benchmark: sampled-GREEN runs IN PARALLEL with
# RadGPT scoring (sidecar), then a single merge job attaches RadGPT + refreshes
# the summary. This is the decoder twin of
# eval_merlin_semantic_test_orchestrator.sh and exists so GREEN || RadGPT is the
# DEFAULT for decoder evals (no more serial green-then-radgpt).
#
# Prereq: generations.json already exist for all runs in BENCHMARK_DIR/runs/*
#   (i.e. the Pass-1 generation job has finished). If generation is still queued,
#   pass its job id as GEN_JOB_ID and GREEN+RadGPT will wait afterok on it.
#
# Usage:
#   bash ops/slurm/eval_decoder_parallel_orchestrator.sh \
#     <BENCHMARK_DIR> <RADGPT_RUN_LABELS_CSV> [GEN_JOB_ID]
#
# Args:
#   BENCHMARK_DIR          decoder benchmark dir (e.g. outputs/decoder/benchmark_test_full_basic)
#   RADGPT_RUN_LABELS_CSV  dash-form run dir names RadGPT should score (the new runs).
#                          GREEN always recomputes ALL runs in the dir for subset
#                          consistency, so it ignores this.
#   GEN_JOB_ID             optional; if given, GREEN+RadGPT wait afterok on it.
#                          omit or pass "none" when generations already exist.
#
# DAG:
#   GREEN   (sampled GREEN, all runs)  -> [afterok GEN]   writes sampled_green -> evaluation.json
#   RADGPT  (8B scoring, labelled runs)-> [afterok GEN]   sidecar only, no evaluation.json
#   MERGE   (attach + refresh + examples) -> afterok GREEN:RADGPT  (sole evaluation.json writer)
set -euo pipefail

BENCHMARK_DIR="${1:?need BENCHMARK_DIR}"
RADGPT_RUN_LABELS="${2:?need RadGPT run labels CSV (dash-form run dir names)}"
GEN_JOB_ID="${3:-none}"

MASTERS="/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters"
SLURM="${MASTERS}/ops/slurm"
cd "${MASTERS}"

# Normalise BENCHMARK_DIR to an absolute path (sbatches cd into MASTERS).
case "${BENCHMARK_DIR}" in
  /*) : ;;
  *) BENCHMARK_DIR="${MASTERS}/${BENCHMARK_DIR}" ;;
esac

RADGPT_OUTPUT_DIR="${BENCHMARK_DIR}/radgpt_benchmark/parallel_$(date +%Y%m%d_%H%M%S)"

# Optional upstream dependency on the generation (Pass-1) job.
GEN_DEP=()
if [[ "${GEN_JOB_ID}" != "none" && -n "${GEN_JOB_ID}" ]]; then
  GEN_DEP=(--dependency=afterok:${GEN_JOB_ID} --kill-on-invalid-dep=yes)
fi

export BENCHMARK_DIR
export RADGPT_OUTPUT_DIR
export RUN_LABELS="${RADGPT_RUN_LABELS}"

# sampled GREEN (writes sampled_green into evaluation.json; recomputes all runs)
GREEN=$(sbatch --parsable \
  "${GEN_DEP[@]}" \
  --export=ALL \
  "${SLURM}/decoder_eval_sampled_green_gh200_1gpu.sbatch")
echo "  GREEN  ${GREEN}  [dep:${GEN_JOB_ID}]  benchmark=${BENCHMARK_DIR}"

# RadGPT scoring — PARALLEL to GREEN, writes only to sidecar (no evaluation.json)
RADGPT=$(sbatch --parsable \
  "${GEN_DEP[@]}" \
  --export=ALL \
  "${SLURM}/decoder_radgpt_scoring_gh200_1gpu.sbatch")
echo "  RADGPT ${RADGPT}  [dep:${GEN_JOB_ID}]  (parallel to GREEN)  labels=${RADGPT_RUN_LABELS}  sidecar=${RADGPT_OUTPUT_DIR}"

# merge — sole evaluation.json writer, after BOTH finish
MERGE=$(sbatch --parsable \
  --dependency=afterok:${GREEN}:${RADGPT} \
  --kill-on-invalid-dep=yes \
  --export=ALL \
  "${SLURM}/decoder_radgpt_merge_gh200_1gpu.sbatch")
echo "  MERGE  ${MERGE}  [afterok:${GREEN}:${RADGPT}]  (attach RadGPT + refresh summary + examples)"

echo
echo "Decoder parallel-eval DAG submitted. Final table refresh happens at end of MERGE job ${MERGE}."
