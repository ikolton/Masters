# Consolidation V2 Analysis

## Run Identity

- Slurm job: `17476782`
- State: `COMPLETED`
- Exit code: `0:0`
- Elapsed: `00:06:12`
- Output directory: `/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/outputs/semantic_tagging/merlin_converted/consolidation/consolidation_v2`
- Config: `configs/merlin_consolidation_v2.yaml`

## Why V2 Was Run

`consolidation_v1` produced parse-valid but semantically weak family placeholders such as:

- `Colon`
- `Liver`
- `Kidneys`
- `Stomach`

Those are too vague for diagnostic-loss supervision. They mostly mean "something in this organ", which is not enough semantic signal.

`consolidation_v2` therefore tightened the LLM contract:

- `training_family` must come from a controlled enum.
- organ names are invalid as families.
- `family_only` must use a useful nonzero coarse loss weight.
- zero-weight non-excluded labels are invalid.
- `merge_relation` is explicit.

## Mechanical Health

The run completed successfully.

- observed subtype labels: `592`
- valid parsed decisions: `565`
- invalid decisions: `27`
- Slurm exit: `0:0`

Mode distribution:

- `merged`: `367`
- `family_only`: `134`
- `direct`: `58`
- `exclude`: `33`

Merge relation distribution:

- `synonym`: `322`
- `not_applicable`: `168`
- `direct`: `56`
- `clinically_related`: `25`
- `parent_child`: `21`

Human-review flags:

- `needs_human_review=false`: `540`
- `needs_human_review=true`: `52`

## What Improved

The specific v1 failure was fixed.

- valid rows with organ-name family/label: `0`
- valid non-exclude rows with `loss_weight = 0`: `0`

Family-only outputs now use controlled families:

- `other_abnormal`: `31`
- `postoperative_or_device`: `21`
- `size_or_morphology`: `15`
- `anatomic_variant`: `15`
- `inflammation`: `14`
- `limited_assessment`: `9`
- `vascular`: `8`
- `obstruction`: `5`
- `fluid_or_collection`: `4`
- `mass_or_malignancy`: `3`
- `cystic_or_fluid_lesion`: `2`
- `wall_thickening`: `2`
- `ambiguous_or_indeterminate`: `2`
- `stone_or_calcification`: `1`
- `gas_or_air`: `1`
- `focal_lesion`: `1`

This is much better than using organ names.

## What V2 Revealed

V2 exposed a second schema ambiguity.

The model often used controlled family names as `training_label` under `mode=merged`.

Examples:

- `colon_colitis -> inflammation`
- `kidneys_hydronephrosis -> obstruction`
- `liver_calcification -> stone_or_calcification`
- `gallbladder_cholecystitis -> inflammation`
- `adrenal_metastasis -> mass_or_malignancy`

This is not necessarily medically wrong as a coarse target, but it is wrong structurally:

- `training_label` should mean subtype-level label.
- `training_family` should mean coarse family-level label.
- V2 still allowed these concepts to blur.

Count:

- valid `direct`/`merged` rows where `training_label` is actually a controlled family: `243`

So V2 is cleaner than V1, but it overcorrected toward coarse-family merging.

## Invalid Decisions

The `27` invalid decisions are mostly useful failures, not crashes.

Common failure:

- model chose `mode=merged`
- `training_label=null`
- `training_family=<controlled family>`

Example:

```json
{
  "mode": "merged",
  "training_label": null,
  "training_family": "inflammation",
  "merge_relation": "synonym",
  "loss_weight": 1.0
}
```

This should have been `family_only`, not `merged`.

Another discovered issue:

- `liver_laceration`
- `liver_subcapsular_laceration`

The model proposed `training_family: trauma`, but `trauma` is not in the controlled family list. This is medically reasonable, so the enum probably needs a `trauma_or_injury` family.

## Rating

Mechanical execution: `8/10`

- Clean Slurm finish.
- All artifacts produced.
- Stricter validation worked.
- Invalid outputs were captured rather than silently accepted.

Semantic interface improvement over V1: `8/10`

- Organ-name placeholders fixed.
- Coarse-family supervision became explicit.
- Review flags increased from `10` to `52`, which is more realistic.

Training-readiness: `6/10`

- Better than V1.
- Still not final, because subtype labels and coarse families are mixed inside `training_label`.

## Recommendation

Do not use `training_vocab_draft.json` from V2 directly for loss yet.

The next version should split the task more deterministically:

1. `use_for_subtype_loss`: boolean
2. `subtype_mode`: `direct | merge_to_subtype | no_subtype`
3. `subtype_label`: organ-specific subtype label or null
4. `use_for_family_loss`: boolean
5. `family_label`: controlled family enum or null
6. `exclude_from_loss`: boolean
7. `merge_relation`: only applies to subtype merges

This would let a tag contribute to:

- subtype loss only
- family loss only
- both
- neither

That is the clean diagnostic-loss structure. It avoids forcing rare labels to become fake subtypes while still preserving useful coarse information.

V3 should also add `trauma_or_injury` to the controlled family enum.

## Bottom Line

V2 was a successful diagnostic experiment.

It fixed the vague family-placeholder bug, but revealed that the schema still conflates subtype labels and coarse family labels.

The next step should not be another broad prompt tweak. It should be a cleaner output schema with separate subtype and family decisions.
