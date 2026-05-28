#!/usr/bin/env bash
#SBATCH --job-name=osc-radgpt-concept-lex
#SBATCH --output=logs/%x-%j.out
#SBATCH --account=plgmmia-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00

# RadGPT evaluation for the 3 concept_lexical_v1_5 decoder runs.
# Uses 8B quantized judge to match historical decoder table runs.
# STUDY_LIMIT=0 → full test set. Results are attached to each run's evaluation.json
# and the comparison summary is refreshed at the end.

set -eo pipefail
set -x

source /etc/profile >/dev/null 2>&1 || true
module purge >/dev/null 2>&1 || true
module load ML-bundle/24.06a

cd /net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters
mkdir -p logs

export MODEL="iqbalamo93/Meta-Llama-3.1-8B-Instruct-GPTQ-Q_8"
export BENCHMARK_DIR="outputs/decoder/benchmark_test_full_basic"
export RUN_LABELS="concept_lexical_v1_5_w001,concept_lexical_v1_5_w002,concept_lexical_v1_5_w005"
export STUDY_LIMIT="0"
export ATTACH_FULL_TO_EVALUATIONS="1"
export API_CONCURRENCY="4"
export SEED="13"

bash ops/setup/run_radgpt_benchmark_local_gh200.sh
