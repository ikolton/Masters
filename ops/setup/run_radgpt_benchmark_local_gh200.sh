#!/usr/bin/env bash
# Run a model-agnostic RadGPT benchmark on an already allocated GH200 node.
#
# Example 1-GPU interactive run:
#   MODEL=iqbalamo93/Meta-Llama-3.1-8B-Instruct-GPTQ-Q_8 STUDY_LIMIT=50 \
#     bash ops/setup/run_radgpt_benchmark_local_gh200.sh
#
# Later 4-GPU run can reuse the same script with:
#   MODEL=meta-llama/Llama-3.3-70B-Instruct CUDA_VISIBLE_DEVICES=0,1,2,3 TP=4 STUDY_LIMIT=0 ...

set -euo pipefail

ROOT="${ROOT:-/net/scratch/hscra/plgrid/plgikolton/Magisterka}"
MASTERS="${MASTERS:-${ROOT}/Masters}"
# Do not use generic names like ENV_DIR here: the Helios module stack exports
# similarly named variables for system Python. RADGPT_ENV_DIR is explicit and
# safe to override from Slurm/export when needed.
RADGPT_ENV_DIR="${RADGPT_ENV_DIR:-${OVERRIDE_RADGPT_ENV_DIR:-${ROOT}/.venv-radgpt-vllm}}"
BENCHMARK_DIR="${BENCHMARK_DIR:-${MASTERS}/outputs/decoder/benchmark_test_full_basic}"
RADGPT_ROOT="${RADGPT_ROOT:-${ROOT}/RadGPT}"
MAIN_ENV_DIR="${MAIN_ENV_DIR:-${ROOT}/.venv}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${BENCHMARK_DIR}/radgpt_benchmark/${RUN_ID}}"

MODEL="${MODEL:-iqbalamo93/Meta-Llama-3.1-8B-Instruct-GPTQ-Q_8}"
PORT="${PORT:-8010}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TP="${TP:-1}"
DTYPE="${DTYPE:-half}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
STUDY_LIMIT="${STUDY_LIMIT:-50}"
SEED="${SEED:-13}"
RUN_LABELS="${RUN_LABELS:-}"
PROGRESS_EVERY="${PROGRESS_EVERY:-25}"
API_CONCURRENCY="${API_CONCURRENCY:-1}"
SAMPLE_MANIFEST="${SAMPLE_MANIFEST:-}"
ATTACH_SAMPLED_TO_EVALUATIONS="${ATTACH_SAMPLED_TO_EVALUATIONS:-0}"
ATTACH_FULL_TO_EVALUATIONS="${ATTACH_FULL_TO_EVALUATIONS:-auto}"
FAST_FLAG="${FAST_FLAG:---fast}"
SKIP_MODULE_LOAD="${SKIP_MODULE_LOAD:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

if [[ "${ATTACH_FULL_TO_EVALUATIONS}" == "auto" ]]; then
  if [[ "${STUDY_LIMIT}" == "0" && -z "${SAMPLE_MANIFEST}" ]]; then
    ATTACH_FULL_TO_EVALUATIONS="1"
  else
    ATTACH_FULL_TO_EVALUATIONS="0"
  fi
fi

mkdir -p "${OUTPUT_DIR}" "${MASTERS}/logs"
LOG="${OUTPUT_DIR}/radgpt_benchmark.log"

echo "[radgpt-benchmark-local] output: ${OUTPUT_DIR}"
echo "[radgpt-benchmark-local] log: ${LOG}"
echo "[radgpt-benchmark-local] model: ${MODEL}"
echo "[radgpt-benchmark-local] study_limit: ${STUDY_LIMIT}"
echo "[radgpt-benchmark-local] sample_manifest: ${SAMPLE_MANIFEST}"
echo "[radgpt-benchmark-local] attach_sampled_to_evaluations: ${ATTACH_SAMPLED_TO_EVALUATIONS}"
echo "[radgpt-benchmark-local] attach_full_to_evaluations: ${ATTACH_FULL_TO_EVALUATIONS}"
echo "[radgpt-benchmark-local] api_concurrency: ${API_CONCURRENCY}"
echo "[radgpt-benchmark-local] radgpt_env: ${RADGPT_ENV_DIR}"
echo "[radgpt-benchmark-local] main_env: ${MAIN_ENV_DIR}"

if [[ ! -x "${RADGPT_ENV_DIR}/bin/python" ]]; then
  echo "[radgpt-benchmark-local] ERROR: missing RadGPT env python: ${RADGPT_ENV_DIR}/bin/python" >&2
  exit 3
fi

if [[ "${SKIP_MODULE_LOAD}" != "1" ]]; then
  set +e
  if ! type module >/dev/null 2>&1; then
    source /usr/share/lmod/lmod/init/bash >/dev/null 2>&1
  fi
  module load ML-bundle/24.06a
  module_status=$?
  set -e
  if [[ "${module_status}" != "0" ]]; then
    echo "[radgpt-benchmark-local] ERROR: failed to load ML-bundle/24.06a" >&2
    exit 2
  fi
fi

"${RADGPT_ENV_DIR}/bin/python" - <<'PY'
import sys
import torch
import vllm
print("[radgpt-benchmark-local] python:", sys.executable)
print("[radgpt-benchmark-local] torch:", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
print("[radgpt-benchmark-local] vllm:", vllm.__version__)
PY

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  echo "[radgpt-benchmark-local] preflight OK"
  exit 0
fi

export PYTHONPATH="${MASTERS}/src:${PYTHONPATH:-}"

extra_args=()
if [[ -n "${SAMPLE_MANIFEST}" ]]; then
  extra_args+=(--sample-manifest "${SAMPLE_MANIFEST}")
fi
if [[ "${ATTACH_SAMPLED_TO_EVALUATIONS}" == "1" ]]; then
  extra_args+=(--attach-sampled-to-evaluations)
fi
if [[ "${ATTACH_FULL_TO_EVALUATIONS}" == "1" ]]; then
  extra_args+=(--attach-full-to-evaluations)
fi

"${RADGPT_ENV_DIR}/bin/python" "${MASTERS}/apps/run_radgpt_benchmark_from_generations.py" \
  --benchmark-dir "${BENCHMARK_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-id "${RUN_ID}" \
  --run-labels "${RUN_LABELS}" \
  --study-limit "${STUDY_LIMIT}" \
  "${extra_args[@]}" \
  --seed "${SEED}" \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --radgpt-root "${RADGPT_ROOT}" \
  "${FAST_FLAG}" \
  --progress-every "${PROGRESS_EVERY}" \
  --api-concurrency "${API_CONCURRENCY}" \
  --launch-vllm \
  --vllm-python "${RADGPT_ENV_DIR}/bin/python" \
  --model "${MODEL}" \
  --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" \
  --tensor-parallel-size "${TP}" \
  --dtype "${DTYPE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  2>&1 | tee "${LOG}"

echo
echo "[radgpt-benchmark-local] done"
echo "[radgpt-benchmark-local] summary: ${OUTPUT_DIR}/summary.md"
echo "[radgpt-benchmark-local] log: ${LOG}"

if [[ -x "${MAIN_ENV_DIR}/bin/python" ]]; then
  echo "[radgpt-benchmark-local] refreshing benchmark comparison summary"
  "${MAIN_ENV_DIR}/bin/python" "${MASTERS}/apps/refresh_decoder_benchmark_summary.py" \
    --benchmark-dir "${BENCHMARK_DIR}" \
    2>&1 | tee -a "${LOG}"
  echo "[radgpt-benchmark-local] refreshed: ${BENCHMARK_DIR}/comparison_summary.md"
else
  echo "[radgpt-benchmark-local] WARNING: main env python missing, skipped summary refresh: ${MAIN_ENV_DIR}/bin/python" >&2
fi
