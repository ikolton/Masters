#!/bin/bash
#SBATCH --job-name=diagnostic-lexicon-vllm
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=120G
#SBATCH --time=06:00:00

# Template wrapper for the future diagnostic_lexicon pipeline.
# This mirrors the semantic_tagging sbatch shape, but is intentionally a template
# until the pipeline code is implemented.

set -euxo pipefail

echo "Template only: implement diagnostic_lexicon/apps/run_lexicon_pipeline.py before use."
exit 1
