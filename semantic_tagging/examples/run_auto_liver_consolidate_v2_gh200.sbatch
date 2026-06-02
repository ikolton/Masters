#!/usr/bin/env bash
#SBATCH --job-name=sem-liver-cons-v2
#SBATCH --output=/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/semantic_tagging/logs/%x-%j.out
#SBATCH --account=plgmmia-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=02:00:00

# Re-consolidation only (stage 1 tagging already done in auto_liver_v1).
# Runs vLLM → consolidation (v2 prompt with stronger synonym-merge guidance)
# → postprocess. Tagging is NOT re-run.

source /etc/profile >/dev/null 2>&1 || true
set -eo pipefail
set -x

module load ML-bundle/24.06a
source /net/scratch/hscra/plgrid/plgikolton/Magisterka/.venv-semantic-tagging-vllm/bin/activate

export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export HF_TOKEN="${HF_TOKEN:-}"

PROJECT_ROOT=/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/semantic_tagging
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

CONS_CONFIG="configs/merlin_auto_consolidation_liver_v2.yaml"
POST_CONFIG="configs/merlin_auto_postprocess_liver_v2.yaml"
SERVER_LOG="${LOG_DIR}/vllm_server_${SLURM_JOB_ID}.log"

cd "${PROJECT_ROOT}"

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 32768 \
  --host 127.0.0.1 \
  --port 8000 \
  > "${SERVER_LOG}" 2>&1 &
VLLM_PID=$!

for _ in $(seq 1 120); do
  if curl -sf -H "Authorization: Bearer ${VLLM_API_KEY}" http://127.0.0.1:8000/v1/models >/dev/null; then
    break
  fi
  sleep 5
done
curl -sf -H "Authorization: Bearer ${VLLM_API_KEY}" http://127.0.0.1:8000/v1/models >/dev/null

echo "[auto-liver-v2] === stage 2: corpus-level consolidation (v2 prompt) ==="
python -u apps/run_tag_consolidation_llm.py --config "${CONS_CONFIG}" \
  | tee "${LOG_DIR}/auto_liver_consolidation_v2_${SLURM_JOB_ID}.log"

echo "[auto-liver-v2] === stage 3: min-support + locked-vocabulary assignment ==="
python -u apps/postprocess_consolidation.py --config "${POST_CONFIG}" \
  | tee "${LOG_DIR}/auto_liver_postprocess_v2_${SLURM_JOB_ID}.log"

echo "[auto-liver-v2] consolidation + postprocess complete."
