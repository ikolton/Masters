# Consolidation V3 Analysis

## Run Identity

- Slurm job: `17480089`
- State: `COMPLETED`
- Exit code: `0:0`
- Elapsed: `00:08:32`
- GPU allocation: `2` GH200 GPUs
- Output directory: `/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/outputs/semantic_tagging/merlin_converted/consolidation/consolidation_v3`
- Config: `configs/merlin_consolidation_v3.yaml`

## Why V3 Was Needed

`consolidation_v2` fixed the vague organ-name family issue, but it still mixed two different concepts:

- subtype labels, such as `colon_wall_thickening`
- coarse family labels, such as `inflammation`

V3 solves this by making subtype supervision and family supervision separate decisions.

Each observed label can now contribute to:

- subtype loss only
- family loss only
- both subtype and family loss
- neither, if excluded

## Mechanical Health

The run completed successfully.

- total observed subtype labels: `592`
- valid parsed decisions: `572`
- invalid decisions: `20`
- final rate after model startup: about `1.64` decisions/s
- end-to-end elapsed time: `8m32s`

Decision modes:

- `subtype_and_family`: `241`
- `family_only`: `329`
- `exclude`: `22`

Subtype modes among valid rows:

- `direct`: `224`
- `merge_to_subtype`: `17`
- `no_subtype`: `331`

Merge relations among valid rows:

- `direct`: `224`
- `synonym`: `10`
- `parent_child`: `7`
- `not_applicable`: `331`

## What Improved

V3 fixed the key V2 structural problem.

- controlled family labels leaked into subtype labels: `0`
- organ names leaked into subtype/family labels: `0`
- subtype and family vocabularies are now separate artifacts inside `training_vocab_draft.json`

Draft subtype vocabulary:

- total subtype labels: `235`

Subtype labels by organ:

- `Adrenal glands`: `11`
- `Colon`: `27`
- `Gallbladder`: `17`
- `Kidneys`: `34`
- `Liver`: `39`
- `Pancreas`: `20`
- `Prostate`: `8`
- `Small bowel`: `31`
- `Spleen`: `23`
- `Stomach`: `11`
- `Urinary bladder`: `14`

Draft family vocabulary:

- total organ-family labels: `171`

Family labels by organ:

- `Adrenal glands`: `15`
- `Colon`: `17`
- `Gallbladder`: `16`
- `Kidneys`: `16`
- `Liver`: `18`
- `Pancreas`: `16`
- `Prostate`: `13`
- `Small bowel`: `13`
- `Spleen`: `15`
- `Stomach`: `15`
- `Urinary bladder`: `17`

This is now aligned with the intended diagnostic-loss design:

- subtype head learns more specific organ-level concepts
- family head learns coarse but stable semantic concepts

## Remaining Problems

### 1. Invalid Rows Remain

There are `20` invalid rows.

They are mostly useful contract failures, not crashes.

Examples:

- `colon_hernia`: proposed family `hernia`, which is not controlled
- `pancreas_fistula`: proposed family `fistula`, which is not controlled
- `small_bowel_fistulous_tract`: proposed family `fistula`, which is not controlled
- `pancreas_fatty_atrophy`: proposed family `atrophy`, which is not controlled
- `colon_constipation`: used `subtype_mode=direct` with a non-direct merge relation

This suggests the controlled family enum should probably add:

- `fistula_or_sinus_tract`
- `hernia_or_prolapse`
- `atrophy_or_fatty_change`

### 2. Review Flags Are Too Conservative

V3 produced:

- `needs_human_review=false`: `572`
- `needs_human_review=true`: `0`

That is too trusting. Even if the schema is clean, some merges should be review-flagged by deterministic rules:

- all `parent_child` subtype merges
- rare subtype labels kept as direct labels
- labels mapped to `other_abnormal`
- labels with ambiguous wording
- any label whose family is useful but subtype is suppressed

This should be handled by deterministic postprocessing, not by relying on the model to self-criticize.

### 3. Some Subtype Labels Are Not Canonically Prefixed

Examples from the draft:

- `gallstones`
- `cystic_lesion`
- `kidney_stone`
- `atrophic`
- `hepatomegaly`
- `mass`
- `splenomegaly`

These are not conceptually terrible, but for a reusable training artifact they should be normalized to organ-prefixed labels:

- `gallbladder_gallstones`
- `kidneys_cystic_lesion`
- `kidneys_stone`
- `kidneys_atrophic`
- `liver_hepatomegaly`
- `pancreas_mass`
- `spleen_splenomegaly`

This should be a deterministic canonicalization step.

## 2-GPU Runtime Assessment

The 2-GPU job was successful and queued/ran correctly.

Compared with the earlier 4-GPU consolidation:

- V2 4-GPU elapsed: `6m12s`
- V3 2-GPU elapsed: `8m32s`

For this small vocabulary-level job, 2 GPUs are a good tradeoff:

- lower queue pressure
- still fast enough
- no need to reserve 4 GPUs for a few hundred prompts

## Security/Operational Note

The sbatch script used shell tracing. That can expose environment exports in logs.

The 2-GPU launcher has been patched so tracing starts after token-related exports.

## Rating

Mechanical execution: `9/10`

- clean Slurm finish
- 2-GPU vLLM worked
- all artifacts written

Schema correctness: `8.5/10`

- subtype/family split is the right abstraction
- controlled families no longer leak into subtype labels

Training-readiness: `7/10`

- much closer than V1/V2
- still needs deterministic canonicalization and review queue generation before final loss targets

## Recommendation

Do not run another broad LLM consolidation pass immediately.

The next step should be deterministic postprocessing of V3:

1. canonicalize subtype labels to organ-prefixed names
2. add deterministic review flags
3. add a few missing controlled families if we choose to repair the `20` invalid rows
4. produce:
   - `tag_consolidation_map_v3_clean.jsonl`
   - `training_vocab_v3_clean.yaml`
   - `review_queue_v3.csv`
   - `semantic_training_targets_v3.parquet`

After that, only the invalid/review queue should need a small targeted LLM repair pass.
