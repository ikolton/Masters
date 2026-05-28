Use these with `sbatch` from the repo root or any directory:

```bash
sbatch ops/slurm/run_encoder_smoke_pb12_bs6.sbatch
sbatch ops/slurm/run_encoder_smoke_pb10_bs8.sbatch
```

Both jobs:

- use the current working `startenv` + `mastersenv` setup
- keep W&B disabled by default
- run a longer smoke than the tiny profile sweep to compare early training behavior
- write logs to `logs/`

Two-GPU smoke jobs:

```bash
sbatch ops/slurm/run_encoder_smoke_2gpu_pb32_bs6.sbatch
sbatch ops/slurm/run_encoder_smoke_2gpu_pb16_bs6.sbatch
```

These launch DDP through `torchrun --nproc_per_node=2`. Note that `batch_size`
is per process, so `batch_size: 6` means global batch `12` on 2 GPUs.
