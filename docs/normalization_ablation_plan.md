# Normalization Ablation Plan

## Goal
Test whether brittle text targets are a major cause of encoder alignment drift.

The baseline failure pattern is:
- organ identity and segmentation improve
- within-organ finding alignment peaks early and then degrades
- long-tail organs suffer most

The normalization ablations are intended to reduce supervision noise, not to make the task easier in an arbitrary way.

## Variants

### 1. `canonical_normal_templates_v2`
Dataset root:
- `/net/scratch/hscra/plgrid/plgikolton/Magisterka/normalized_datasets/canonical_normal_templates_v2`

Change:
- collapse high-frequency organ-specific normal template sentences to `unremarkable`

Why:
- reduce fragmentation among semantically identical normal findings
- increase positive repetition for the dominant normal state

### 2. `canonical_normal_templates_and_absent_v2`
Dataset root:
- `/net/scratch/hscra/plgrid/plgikolton/Magisterka/normalized_datasets/canonical_normal_templates_and_absent_v2`

Change:
- everything from `canonical_normal_templates_v2`
- plus absent-family merges to `surgically absent`

Why:
- tests whether a slightly broader deterministic grouping helps more

## Fixed Training Setup
Keep these fixed against the current best GH200 run:
- config family: `organsegclip_128x48_20ep_uniformtext_schedcos_logitslr1e5`
- 2 GPUs
- global batch size `12`
- per-GPU batch size `6`
- patch batch size `16`
- `bf16`
- best checkpoint metric: `full_val_organ_alignment_loss`

## Primary Success Criteria

### Training / validation dynamics
- lower best `full_val_organ_alignment_loss`
- later peak epoch for `best.pt`, or slower post-peak degradation
- equal or better `full_val_organ_image_to_text_top1`
- equal or better `full_val_organ_text_to_image_top1`

### Checkpoint analyzer
Run the improved checkpoint analyzer on `best.pt` and compare against the current best run:
- same-organ retrieval `top1_excluding_self`
- same-organ retrieval `mrr_excluding_self`
- frozen diagnostic probe balanced accuracy
- frozen lesion probe balanced accuracy
- per-organ retrieval, especially whether weak long-tail organs improve without hurting strong organs

## Interpretation Rules

### If the normalization run improves alignment and analyzer metrics
Interpretation:
- brittle text target semantics are likely a real driver of the current failure mode

Next step:
- run the `and_absent_v2` follow-up

### If the normalization run improves loss stability but not downstream analyzer quality
Interpretation:
- normalization may be making the objective numerically easier without improving representation quality

Next step:
- consider less aggressive grouping or grouped-positive loss changes instead of more canonicalization

### If the normalization run does not improve or gets worse
Interpretation:
- current alignment problems are probably driven more by optimization / loss design than by obvious template fragmentation alone

Next step:
- move to grouped-positive or semantic-positive loss ablations rather than broader literal normalization

## Operational Notes
- All normalization datasets are local sidecar datasets under `/net/scratch/hscra/plgrid/plgikolton/Magisterka/normalized_datasets`
- The original dataset under `/net/storage/pr3/plgrid/plggjmiag/Merlin_converted` is not modified
- The `v1` variants are intentionally not used for training because they were too weak after accounting for training-time text normalization
