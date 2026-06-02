#!/usr/bin/env bash
#SBATCH --job-name=sem-liver-v3
#SBATCH --output=/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/semantic_tagging/logs/%x-%j.out
#SBATCH --account=plgmmia-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=08:00:00

# Clean automatic semantic-tagging pipeline, single organ (Liver), end to end:
#   1. candidate proposal + schema repair  (run_tagging_pipeline)
#   2. corpus-level consolidation           (run_tag_consolidation_llm)
#   3. min-support + locked-vocab assignment (postprocess_consolidation)
# One vLLM server serves all LLM stages. Statistical phrase extraction
# (diagnostic_lexicon, CPU-only) is run separately after inspecting outputs.

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

TAG_CONFIG="configs/merlin_auto_liver_v3.yaml"
CONS_CONFIG="configs/merlin_auto_consolidation_liver_v3.yaml"
POST_CONFIG="configs/merlin_auto_postprocess_liver_v3.yaml"
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
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 8192 \
  --host 127.0.0.1 \
  --port 8000 \
  > "${SERVER_LOG}" 2>&1 &
VLLM_PID=$!

# Wait for server readiness (up to ~10 min for the 70B to load).
for _ in $(seq 1 120); do
  if curl -sf -H "Authorization: Bearer ${VLLM_API_KEY}" http://127.0.0.1:8000/v1/models >/dev/null; then
    break
  fi
  sleep 5
done
curl -sf -H "Authorization: Bearer ${VLLM_API_KEY}" http://127.0.0.1:8000/v1/models >/dev/null

echo "[auto-liver] === stage 1: candidate proposal + schema repair ==="
python -u apps/run_tagging_pipeline.py --config "${TAG_CONFIG}" \
  | tee "${LOG_DIR}/auto_liver_v3_tagging_${SLURM_JOB_ID}.log"

echo "[auto-liver] === stage 2: corpus-level consolidation ==="
python -u apps/run_tag_consolidation_llm.py --config "${CONS_CONFIG}" \
  | tee "${LOG_DIR}/auto_liver_v3_consolidation_${SLURM_JOB_ID}.log"

echo "[auto-liver] === stage 3: min-support + locked-vocabulary assignment ==="
python -u apps/postprocess_consolidation.py --config "${POST_CONFIG}" \
  | tee "${LOG_DIR}/auto_liver_v3_postprocess_${SLURM_JOB_ID}.log"

echo "[auto-liver] pipeline complete."
