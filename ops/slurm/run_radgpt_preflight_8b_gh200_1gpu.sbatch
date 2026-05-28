#!/usr/bin/env bash
#SBATCH --job-name=osc-radgpt-preflight
#SBATCH --output=logs/%x-%j.out
#SBATCH --account=plgmmia-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:10:00

set -eo pipefail
set -x

cd /net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters
mkdir -p logs

source /etc/profile >/dev/null 2>&1 || true
module purge >/dev/null 2>&1 || true
module load ML-bundle/24.06a

export MODEL="${MODEL:-iqbalamo93/Meta-Llama-3.1-8B-Instruct-GPTQ-Q_8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TP="${TP:-1}"
export STUDY_LIMIT="${STUDY_LIMIT:-0}"
export RUN_ID="${RUN_ID:-radgpt_preflight_${SLURM_JOB_ID}}"
export RADGPT_ENV_DIR="${RADGPT_ENV_DIR:-/net/scratch/hscra/plgrid/plgikolton/Magisterka/.venv-radgpt-vllm}"
export SKIP_MODULE_LOAD=1
export PREFLIGHT_ONLY=1

bash ops/setup/run_radgpt_benchmark_local_gh200.sh
