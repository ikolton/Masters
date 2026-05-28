#!/usr/bin/env bash
#SBATCH --job-name=osc-radgpt-8b
#SBATCH --output=logs/%x-%j.out
#SBATCH --account=plgmmia-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00

set -eo pipefail

cd /net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters
mkdir -p logs

source /etc/profile >/dev/null 2>&1 || true
module purge >/dev/null 2>&1 || true
module load ML-bundle/24.06a

# Fast iteration judge. The launcher attaches full-test RadGPT metrics to each
# run's evaluation.json and refreshes comparison_summary.md when STUDY_LIMIT=0.
export MODEL="${MODEL:-iqbalamo93/Meta-Llama-3.1-8B-Instruct-GPTQ-Q_8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TP="${TP:-1}"
export STUDY_LIMIT="${STUDY_LIMIT:-0}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export PORT="${PORT:-8010}"
export RUN_ID="${RUN_ID:-radgpt_8b_slurm_${SLURM_JOB_ID}}"
export RADGPT_ENV_DIR="${RADGPT_ENV_DIR:-/net/scratch/hscra/plgrid/plgikolton/Magisterka/.venv-radgpt-vllm}"
export API_CONCURRENCY="${API_CONCURRENCY:-8}"
export PROGRESS_EVERY="${PROGRESS_EVERY:-250}"
export SKIP_MODULE_LOAD=1

bash ops/setup/run_radgpt_benchmark_local_gh200.sh
