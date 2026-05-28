# Merlin Model Ablations

Self-contained ablation harness for testing whether lexical and semantic diagnostic
losses transfer to Stanford Merlin report generation.

The harness imports Merlin by path instead of vendoring or rewriting it. The
goal is to keep the model as close as possible to the released implementation
while making auxiliary losses switchable and comparable.

## Layout

- `apps/`: runnable entrypoints.
- `configs/`: smoke and future full-run configs.
- `src/merlin_ablation/`: reusable dataset, model wrapper, loss, and training code.
- `slurm/`: job templates.
- `runbooks/`: manual commands and operational notes.
- `docs/`: design notes and analysis.
- `tests/`: lightweight unit tests that avoid loading Merlin weights.

## First Smoke Runs

From this folder:

```bash
python apps/train_merlin_ablation.py --config configs/smoke_ce_only.yaml
python apps/train_merlin_ablation.py --config configs/smoke_sem_family.yaml
```

On GH200 via SLURM:

```bash
sbatch slurm/train_merlin_smoke_gh200_1gpu.sbatch configs/smoke_ce_only.yaml
sbatch slurm/train_merlin_smoke_gh200_1gpu.sbatch configs/smoke_sem_family.yaml
```

Outputs are written under:

```text
/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/outputs/models_ablations/merlin/<run_id>/
```

## Frozen Image-Feature Cache

Cached training stores frozen Merlin image-encoder features before the trainable
adapter. This keeps the ablation scientifically close to online training:
the image encoder is frozen either way, while the adapter and decoder LoRA can
still train normally.

Build the cache for a config:

```bash
python apps/build_image_embedding_cache.py \
  --config configs/smoke_calibration_sem_family_cached.yaml
```

Run a cached smoke:

```bash
python apps/train_merlin_ablation.py \
  --config configs/smoke_calibration_sem_family_cached.yaml
```

Profile cached throughput and GPU use:

```bash
python apps/profile_cached_training.py \
  --config configs/smoke_calibration_sem_family_cached.yaml \
  --mode cached \
  --batch-sizes 8,12 \
  --max-steps 8
```

The profiler writes JSON results under the config output directory and reports
per-GPU utilization/memory. Current GH200 smoke result: batch 12 reached about
13 organ rows/s with about 65 GB peak CUDA allocation on one GPU.
