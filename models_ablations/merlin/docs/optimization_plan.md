# Merlin Ablation Optimization Plan

## Immediate Calibration

Use `configs/smoke_calibration_sem_family.yaml` after the environment preflight
passes. It is intentionally larger than the tiny smoke:

- `train_limit: 64`
- `val_limit: 16`
- `batch_size: 2`
- `max_steps: 10`
- `max_length: 768`

This should tell us:

- whether batch size 2 fits on a 120GB GH200 GPU;
- examples/sec after model loading and MONAI cache warmup;
- whether auxiliary semantic heads behave numerically;
- whether the run writes manifests and metrics cleanly.

## Main Bottleneck

The current harness is scientifically clean but computationally naive:

```text
one organ record -> one transformed CT -> one image encoder pass -> one decoder pass
```

The image encoder is frozen, so a full run wastes work by re-encoding the same
study for multiple organs. Merlin-converted train has roughly:

```text
25,489 studies x 11 organs = 280,379 organ rows
```

## Best Optimization

Add a frozen image-embedding cache:

```text
study image -> Merlin image encoder -> adapter-ready image embeddings -> .pt cache
```

Then training becomes:

```text
cached image embeddings + organ prompt/report -> decoder + auxiliary losses
```

This preserves the scientific ablation because the cached features are exactly
from Merlin's frozen image encoder. It only removes repeated computation.

## Secondary Optimization

Group all organs from the same study in one batch after caching:

```text
one cached image embedding repeated for N organ prompts
```

This improves IO and avoids repeated cache reads.

## Full Run Recommendation

Do not launch full runs before the calibration smoke. If calibration is healthy:

- start with `batch_size: 2`;
- keep `grad_accum_steps: 8`;
- use `max_length: 1024`;
- run `ce_only`, `lexical_w002`, `sem_family_w005`, `lexw002_sem_family_w002`;
- add image-embedding cache before running the complete 280k organ-row epoch.

