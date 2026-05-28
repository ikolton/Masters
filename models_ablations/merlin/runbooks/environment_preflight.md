# Merlin Ablation Environment Preflight

The first submitted smoke jobs failed before training because the active `.venv`
did not contain `torchvision`, which Merlin imports during module loading.

Do **not** mutate existing project envs just to fix this. Use a separate Merlin
ablation env and point jobs at it with `MERLIN_ABLATION_VENV`.

## Check The Current Venv On A GH200 Node

```bash
source /etc/profile >/dev/null 2>&1 || true
module load ML-bundle/24.06a
source /net/scratch/hscra/plgrid/plgikolton/Magisterka/.venv-merlin-ablation/bin/activate

python - <<'PY'
import importlib
for name in ("torch", "torchvision", "monai", "nibabel", "peft", "transformers", "yaml"):
    try:
        mod = importlib.import_module(name)
        print(name, getattr(mod, "__version__", "ok"))
    except Exception as exc:
        print("MISSING", name, exc)
PY
```

## Venv Override

The SLURM scripts now allow overriding the venv:

```bash
MERLIN_ABLATION_VENV=/net/scratch/hscra/plgrid/plgikolton/Magisterka/.venv-merlin-ablation \
sbatch slurm/train_merlin_smoke_gh200_1gpu.sbatch configs/smoke_ce_only.yaml
```

Use this only if that venv has all Merlin dependencies:

```text
torch, torchvision, monai, nibabel, peft, transformers, yaml
```
