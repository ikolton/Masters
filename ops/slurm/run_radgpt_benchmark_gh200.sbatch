#!/usr/bin/env bash
#SBATCH --job-name=osc-radgpt-bench
#SBATCH --output=logs/%x-%j.out
#SBATCH --account=plgmmia-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=48:00:00

set -eo pipefail

cd /net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters
mkdir -p logs

source /etc/profile >/dev/null 2>&1 || true
module purge >/dev/null 2>&1 || true
module load ML-bundle/24.06a

# Defaults are intentionally model-agnostic and can be overridden at submit time:
#   sbatch --export=ALL,MODEL=meta-llama/Llama-3.3-70B-Instruct,TP=4,STUDY_LIMIT=0,API_CONCURRENCY=16 ops/slurm/run_radgpt_benchmark_gh200.sbatch
export MODEL="${MODEL:-meta-llama/Llama-3.3-70B-Instruct}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TP="${TP:-4}"
export STUDY_LIMIT="${STUDY_LIMIT:-0}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export PORT="${PORT:-8010}"
export RUN_ID="${RUN_ID:-slurm_${SLURM_JOB_ID}}"
export RADGPT_ENV_DIR="${RADGPT_ENV_DIR:-/net/scratch/hscra/plgrid/plgikolton/Magisterka/.venv-radgpt-vllm}"
export API_CONCURRENCY="${API_CONCURRENCY:-16}"
export PROGRESS_EVERY="${PROGRESS_EVERY:-250}"
export SKIP_MODULE_LOAD=1

bash ops/setup/run_radgpt_benchmark_local_gh200.sh
