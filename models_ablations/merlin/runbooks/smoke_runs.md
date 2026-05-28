# Merlin Smoke Runs

## Local Interactive Node

```bash
source ~/.bashrc
startenv
mastersenv
cd /net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/models_ablations/merlin

python -u apps/train_merlin_ablation.py --config configs/smoke_ce_only.yaml
python -u apps/train_merlin_ablation.py --config configs/smoke_lexical.yaml
python -u apps/train_merlin_ablation.py --config configs/smoke_sem_family.yaml
python -u apps/train_merlin_ablation.py --config configs/smoke_lex_sem_family.yaml
python -u apps/train_merlin_ablation.py --config configs/smoke_calibration_sem_family.yaml
```

## SLURM

```bash
cd /net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/models_ablations/merlin
sbatch slurm/train_merlin_smoke_gh200_1gpu.sbatch configs/smoke_ce_only.yaml
sbatch slurm/train_merlin_smoke_gh200_1gpu.sbatch configs/smoke_sem_family.yaml
sbatch slurm/train_merlin_smoke_gh200_1gpu.sbatch configs/smoke_calibration_sem_family.yaml
```

Tail logs:

```bash
tail -f /net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/logs/merlin-smoke-<jobid>.out
```
