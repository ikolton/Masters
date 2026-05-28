#!/usr/bin/env bash
# Local RadGPT smoke test for an already allocated GH200 interactive node.
#
# This starts a temporary local vLLM server, runs RadGPT on a tiny CSV, and
# writes all logs/artifacts under outputs/decoder/radgpt_smoke/runs/<run_id>.
# It does not submit Slurm jobs and does not modify any environment.

set -euo pipefail

ROOT="${ROOT:-/net/scratch/hscra/plgrid/plgikolton/Magisterka}"
MASTERS="${MASTERS:-${ROOT}/Masters}"
RADGPT_ROOT="${RADGPT_ROOT:-${ROOT}/RadGPT/evaluate_reports}"
ENV_DIR="${ENV_DIR:-${ROOT}/.venv-radgpt-vllm}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${MASTERS}/outputs/decoder/radgpt_smoke}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${OUTPUT_ROOT}/runs/${RUN_ID}"
LATEST_FILE="${OUTPUT_ROOT}/latest_run.txt"

MODEL="${MODEL:-hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4}"
PORT="${PORT:-8010}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
SAMPLE_COUNT="${SAMPLE_COUNT:-5}"
FAST="${FAST:-1}"
STEP="${STEP:-tumor detection}"

mkdir -p "${RUN_DIR}" "${OUTPUT_ROOT}" "${ROOT}/logs"
printf '%s\n' "${RUN_DIR}" > "${LATEST_FILE}"

SERVER_LOG="${RUN_DIR}/vllm_server.log"
RADGPT_LOG="${RUN_DIR}/radgpt.log"
INPUT_CSV="${RUN_DIR}/radgpt_smoke_input.csv"
OUTPUT_CSV="${RUN_DIR}/radgpt_smoke_output.csv"
STEP_SLUG="${STEP// /_}"
ACTUAL_OUTPUT_CSV="${OUTPUT_CSV%.csv}_${STEP_SLUG}.csv"
PID_FILE="${RUN_DIR}/vllm_server.pid"
MANIFEST="${RUN_DIR}/manifest.txt"

cleanup() {
  if [[ -f "${PID_FILE}" ]]; then
    pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
      echo "[radgpt-smoke] stopping vLLM pid=${pid}"
      kill "${pid}" >/dev/null 2>&1 || true
      sleep 3
      kill -9 "${pid}" >/dev/null 2>&1 || true
    fi
  fi
}
trap cleanup EXIT

echo "[radgpt-smoke] run_dir: ${RUN_DIR}"
echo "[radgpt-smoke] env: ${ENV_DIR}"
echo "[radgpt-smoke] model: ${MODEL}"
echo "[radgpt-smoke] port: ${PORT}"
echo "[radgpt-smoke] cuda: ${CUDA_VISIBLE_DEVICES}"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "[radgpt-smoke] ERROR: run this on the GH200/aarch64 interactive node." >&2
  exit 2
fi
if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  echo "[radgpt-smoke] ERROR: missing env python: ${ENV_DIR}/bin/python" >&2
  exit 3
fi
if [[ ! -f "${RADGPT_ROOT}/RunRadGPT.py" ]]; then
  echo "[radgpt-smoke] ERROR: missing RadGPT RunRadGPT.py under ${RADGPT_ROOT}" >&2
  exit 4
fi

set +e
if ! type module >/dev/null 2>&1; then
  source /usr/share/lmod/lmod/init/bash >/dev/null 2>&1
fi
module load ML-bundle/24.06a
module_status=$?
set -e
if [[ "${module_status}" != "0" ]]; then
  echo "[radgpt-smoke] ERROR: failed to load ML-bundle/24.06a" >&2
  exit 5
fi

cat > "${MANIFEST}" <<EOF
run_dir=${RUN_DIR}
root=${ROOT}
masters=${MASTERS}
radgpt_root=${RADGPT_ROOT}
env_dir=${ENV_DIR}
model=${MODEL}
port=${PORT}
cuda_visible_devices=${CUDA_VISIBLE_DEVICES}
tensor_parallel_size=${TENSOR_PARALLEL_SIZE}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}
max_model_len=${MAX_MODEL_LEN}
sample_count=${SAMPLE_COUNT}
step=${STEP}
fast=${FAST}
EOF

echo "[radgpt-smoke] verifying imports"
"${ENV_DIR}/bin/python" - <<'PY'
import numpy, numba, torch, vllm, transformers, openai, pandas
print("python ok")
print("numpy", numpy.__version__)
print("numba", numba.__version__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
print("vllm", vllm.__version__)
print("transformers", transformers.__version__)
print("openai", openai.__version__)
print("pandas", pandas.__version__)
major, minor, *_ = [int(part) for part in numpy.__version__.split(".")[:2]]
if (major, minor) > (2, 2):
    raise SystemExit(f"NumPy {numpy.__version__} is too new for numba/vLLM; run: /path/to/env/bin/python -m pip install --force-reinstall numpy==2.2.6")
PY

echo "[radgpt-smoke] writing tiny input csv: ${INPUT_CSV}"
"${ENV_DIR}/bin/python" - <<PY
from pathlib import Path
import pandas as pd

source = Path("${RADGPT_ROOT}") / "report_examples.csv"
target = Path("${INPUT_CSV}")
count = int("${SAMPLE_COUNT}")
df = pd.read_csv(source).head(count).copy()
if "Encrypted Accession Number" not in df.columns or "Findings" not in df.columns:
    raise SystemExit(f"Unexpected RadGPT example columns: {list(df.columns)}")
df = df[["Encrypted Accession Number", "Findings"]]
df.to_csv(target, index=False)
print(df.head().to_string(index=False))
PY

echo "[radgpt-smoke] launching vLLM"
(
  cd "${RADGPT_ROOT}"
  export TRANSFORMERS_CACHE="${RADGPT_ROOT}/HFCache"
  export HF_HOME="${RADGPT_ROOT}/HFCache"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
  export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
  mkdir -p "${RADGPT_ROOT}/HFCache"
  exec "${ENV_DIR}/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --dtype half \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --enforce-eager
) > "${SERVER_LOG}" 2>&1 &
echo $! > "${PID_FILE}"

echo "[radgpt-smoke] waiting for API"
deadline=$((SECONDS + 1800))
until curl -fsS "http://127.0.0.1:${PORT}/v1/models" > "${RUN_DIR}/models.json"; do
  if ! kill -0 "$(cat "${PID_FILE}")" >/dev/null 2>&1; then
    echo "[radgpt-smoke] ERROR: vLLM exited early. Log tail:" >&2
    tail -80 "${SERVER_LOG}" >&2 || true
    exit 6
  fi
  if (( SECONDS > deadline )); then
    echo "[radgpt-smoke] ERROR: timed out waiting for API. Log tail:" >&2
    tail -120 "${SERVER_LOG}" >&2 || true
    exit 7
  fi
  echo "[radgpt-smoke] API not ready yet..."
  sleep 10
done
echo "[radgpt-smoke] API ready"
cat "${RUN_DIR}/models.json"
echo

echo "[radgpt-smoke] running RadGPT"
(
  cd "${RADGPT_ROOT}"
  "${ENV_DIR}/bin/python" RunRadGPT.py \
    --port "${PORT}" \
    --data_path "${INPUT_CSV}" \
    --institution UCSF \
    --step "${STEP}" \
    --save_name "${OUTPUT_CSV}" \
    --fast "${FAST}" \
    --restart
) > "${RADGPT_LOG}" 2>&1

echo "[radgpt-smoke] RadGPT done"
"${ENV_DIR}/bin/python" - <<PY
from pathlib import Path
import pandas as pd
path = Path("${ACTUAL_OUTPUT_CSV}")
print("output_exists", path.exists(), "path", path)
if not path.exists():
    raise SystemExit(f"expected RadGPT output was not created: {path}")
df = pd.read_csv(path)
print("rows", len(df), "columns", list(df.columns))
print(df.head(10).to_string(index=False))
PY

echo
echo "[radgpt-smoke] success"
echo "[radgpt-smoke] latest run file: ${LATEST_FILE}"
echo "[radgpt-smoke] server log: ${SERVER_LOG}"
echo "[radgpt-smoke] radgpt log: ${RADGPT_LOG}"
echo "[radgpt-smoke] output csv: ${ACTUAL_OUTPUT_CSV}"
