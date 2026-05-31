#!/usr/bin/env bash
#SBATCH --job-name=osc-enc-v2-watch
#SBATCH --output=/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/logs/%x-%j.out
#SBATCH --account=plgmmia-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

# Lightweight CPU-only watcher — polls every 10 min, auto-recovers if training crashes.
# No GPU needed; runs on the CPU partition for 5 days (covers the full 2-job window).

set -eo pipefail

WATCHER=/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/ops/slurm/watch_encoder_v2.sh
LOG=/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/logs/enc_v2_watcher.log
STATE=/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/logs/enc_v2_watcher_state.json
PYBIN=/net/scratch/hscra/plgrid/plgikolton/conda-envs/codex-masters-py311/bin/python

echo "[watcher-slurm] started at $(date), pid=$$"

while true; do
    bash "${WATCHER}"
    EXIT=$?

    # Check if training is done
    DONE=$(${PYBIN} -c "
import json, pathlib
p = pathlib.Path('${STATE}')
print(json.loads(p.read_text()).get('done', False) if p.exists() else False)
" 2>/dev/null || echo "False")

    if [[ "${DONE}" == "True" ]]; then
        echo "[watcher-slurm] training complete — exiting watcher"
        exit 0
    fi

    if [[ ${EXIT} -eq 3 ]]; then
        echo "[watcher-slurm] max retries / unrecoverable — exiting, needs human"
        exit 3
    fi

    sleep 600  # poll every 10 minutes
done
