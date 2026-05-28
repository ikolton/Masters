#!/usr/bin/env bash
# Create a RadGPT-specific vLLM environment on a GH200 compute node.
#
# Run this only inside an interactive GH200 job, e.g.:
#   srun --gres=gpu:1 --mem=120G --ntasks=1 --cpus-per-task=72 \
#     --job-name=radgpt-env --account=plgmmia-gpu-gh200 -t 02:00:00 \
#     --partition=plgrid-gpu-gh200 --pty bash
#   bash /net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/ops/setup/create_radgpt_vllm_env_gh200.sh

set -uo pipefail

ROOT="${ROOT:-/net/scratch/hscra/plgrid/plgikolton/Magisterka}"
ENV_DIR="${ENV_DIR:-${ROOT}/.venv-radgpt-vllm}"
WHEEL_DIR="${WHEEL_DIR:-/net/software/aarch64/el9/wheels/ML-bundle/24.06a}"
FORCE="${FORCE:-0}"

echo "[radgpt-env] root: ${ROOT}"
echo "[radgpt-env] env:  ${ENV_DIR}"
echo "[radgpt-env] wheels: ${WHEEL_DIR}"

echo "[radgpt-env] loading cluster profile/modules"
set +e
if ! type module >/dev/null 2>&1; then
  source /usr/share/lmod/lmod/init/bash >/dev/null 2>&1
fi
lmod_status=$?
module load ML-bundle/24.06a
module_status=$?
set -e
echo "[radgpt-env] lmod init status: ${lmod_status}"
echo "[radgpt-env] module load status: ${module_status}"
if [[ "${module_status}" != "0" ]]; then
  echo "[radgpt-env] ERROR: failed to load ML-bundle/24.06a" >&2
  exit 4
fi
echo "[radgpt-env] module loaded"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "[radgpt-env] ERROR: this script must run on the GH200/aarch64 node, not login/x86." >&2
  exit 2
fi

verify_env() {
  echo "[radgpt-env] verification"
  "${ENV_DIR}/bin/python" - <<'PY'
import importlib.util
import sys

print("python", sys.executable)

import numpy
import numba
import torch
import vllm
import transformers
import openai
import pandas
import sklearn
import matplotlib
import nibabel
import skimage
import SimpleITK

print("numpy", numpy.__version__)
print("numba", numba.__version__)
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(), "gpu_count", torch.cuda.device_count())
print("vllm", vllm.__version__)
print("transformers", transformers.__version__)
print("openai", openai.__version__)
print("pandas", pandas.__version__)
print("sklearn", sklearn.__version__)
print("matplotlib", matplotlib.__version__)
print("nibabel", nibabel.__version__)
print("skimage", skimage.__version__)
print("SimpleITK", SimpleITK.Version_VersionString())

missing = []
for name in ("cv2", "scipy", "numpy", "tqdm"):
    if importlib.util.find_spec(name) is None:
        missing.append(name)
if missing:
    raise SystemExit(f"missing expected modules: {missing}")
major, minor, *_ = [int(part) for part in numpy.__version__.split(".")[:2]]
if (major, minor) > (2, 2):
    raise SystemExit(f"NumPy {numpy.__version__} is too new for this numba/vLLM stack; expected <=2.2.x")
print("[radgpt-env] OK")
PY
}

if [[ -d "${ENV_DIR}" && "${FORCE}" != "1" ]]; then
  if ! grep -q "VIRTUAL_ENV=\"${ENV_DIR}\"" "${ENV_DIR}/bin/activate" 2>/dev/null; then
    echo "[radgpt-env] ERROR: existing env looks stale/malformed:"
    echo "[radgpt-env] ${ENV_DIR}/bin/activate does not point at ${ENV_DIR}"
    echo "[radgpt-env] Recreate this env with:"
    echo "FORCE=1 bash $0"
    exit 5
  fi
  echo "[radgpt-env] existing env found; verifying without modifying it"
  verify_env
  echo
  echo "[radgpt-env] done"
  echo "[radgpt-env] activate with:"
  echo "source ${ENV_DIR}/bin/activate"
  exit 0
fi

if [[ -d "${ENV_DIR}" && "${FORCE}" == "1" ]]; then
  echo "[radgpt-env] removing existing env because FORCE=1"
  rm -rf "${ENV_DIR}"
fi

python3.11 -m venv "${ENV_DIR}"

"${ENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel

echo "[radgpt-env] installing GH200 cluster wheels first"
"${ENV_DIR}/bin/python" -m pip install "${WHEEL_DIR}/torch-2.8.0+cu124-cp311-cp311-linux_aarch64.whl"
"${ENV_DIR}/bin/python" -m pip install "${WHEEL_DIR}/triton-3.4.0+git11ec6354-cp311-cp311-linux_aarch64.whl"
"${ENV_DIR}/bin/python" -m pip install "${WHEEL_DIR}/flash_attn-2.8.3-cp311-cp311-linux_aarch64.whl"
"${ENV_DIR}/bin/python" -m pip install "${WHEEL_DIR}/xformers-0.0.30+cu124-cp311-cp311-linux_aarch64.whl"
"${ENV_DIR}/bin/python" -m pip install "${WHEEL_DIR}/vllm-0.10.2+cu124-cp311-cp311-linux_aarch64.whl"

echo "[radgpt-env] pinning transformers to vLLM-compatible 4.x"
"${ENV_DIR}/bin/python" -m pip install --upgrade --force-reinstall "transformers>=4.55.2,<5"

echo "[radgpt-env] installing RadGPT client/analysis dependencies"
"${ENV_DIR}/bin/python" -m pip install \
  pandas \
  scikit-learn \
  matplotlib \
  nibabel \
  openpyxl \
  simpleitk \
  "scikit-image==0.24.0"

echo "[radgpt-env] restoring vLLM/numba-compatible NumPy pin"
"${ENV_DIR}/bin/python" -m pip install --force-reinstall "numpy==2.2.6"

verify_env

echo
echo "[radgpt-env] done"
echo "[radgpt-env] activate with:"
echo "source ${ENV_DIR}/bin/activate"
