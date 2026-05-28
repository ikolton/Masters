# Ontology V2 Task List

## Purpose

This task list translates the finished first-pass artifact analysis into concrete `ontology v2` work for the second run.

Primary evidence source:

- [artifact_analysis.md](/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/outputs/semantic_tagging/merlin_converted/vllm_full/reports/artifact_analysis.md)

This is intentionally data-driven:

- promote subtypes because the run reused them heavily
- defer or merge subtypes when the signal looks redundant or too narrow
- keep family expansion conservative because the run produced only `4` family proposals

## Promotion Criteria

A subtype should be promoted into `ontology v2` when most of the following are true:

1. It is used repeatedly in the finished run, especially `>= 50` times.
2. It captures a stable clinical pattern rather than a wording quirk.
3. The current ontology for that organ is clearly too thin to express it.
4. It is likely to reduce provisional pressure or unresolved ontology gaps in a second pass.
5. It can be mapped to an existing family without creating a new global family.

## Phase 1: Implement Now

These are the highest-value promotions from the first run and should be present before the second pass.

### Colon

Promote now:

- `colon_wall_thickening`
- `colon_stool_burden`
- `colon_postsurgical_change`
- `colon_submucosal_edema`
- `colon_pericolonic_stranding`
- `colon_colitis`
- `colon_mucosal_hyperemia`

Reason:

- `Colon` had `8.25%` provisional rate and `12` unresolved cases.
- The base ontology only had `colon_normal` and `colon_diverticulosis`.

### Small bowel

Promote now:

- `small_bowel_dilatation`
- `small_bowel_wall_thickening`
- `small_bowel_obstruction`
- `small_bowel_submucosal_edema`
- `small_bowel_fecalization`
- `small_bowel_postoperative_change`
- `small_bowel_hyperemia`
- `small_bowel_herniation`
- `small_bowel_narrowing`

Reason:

- `Small bowel` had the highest provisional pressure: `12.94%`.
- Several unresolved cases were direct ontology gaps in inflammatory or obstruction-related findings.

### Gallbladder

Promote now:

- `gallbladder_distension`
- `gallbladder_wall_thickening`
- `gallbladder_pericholecystic_fluid`
- `gallbladder_sludge`
- `gallbladder_focal_fundal_adenomyomatosis`
- `gallbladder_cholecystitis`

Reason:

- `Gallbladder` had `12.25%` provisional rate.
- The ontology already had `normal`, `absent_postop`, and `gallstones`, but lacked the recurrent inflammatory and caliber findings that actually appear in the data.

### Kidneys

Promote now:

- `kidneys_absent_postop`
- `kidneys_stone`
- `kidneys_hydronephrosis`
- `kidneys_hypoattenuating_lesion`
- `kidneys_cortical_scarring`
- `kidneys_atrophic`
- `kidneys_delayed_excretion`
- `kidneys_perinephric_stranding`
- `kidneys_urothelial_thickening`
- `kidneys_hydroureter`

Reason:

- `Kidneys` had low provisional rate overall, but the base ontology was still much too thin.
- Several strong recurrent subtypes are already effectively acting like stable ontology members.

## Phase 2: Promote Or Merge After Review

These are valuable, but they should be reviewed alongside canonical naming cleanup.

### Colon

- `colon_gaseous_distention`
- `colon_anastomosis`
- `colon_decompression`
- `colon_inflammatory_changes`

Decision tendency:

- likely merge into broader post-op, distention, or inflammatory buckets rather than keep all as standalone subtypes

### Small bowel

- `small_bowel_anastomotic_changes`
- `small_bowel_fluid`
- `small_bowel_collapsed_loop`

Decision tendency:

- likely keep some as modifiers or secondary tags rather than primary subtype labels

### Gallbladder

- `gallbladder_contraction`
- `gallbladder_decompression`
- `gallbladder_edema`

Decision tendency:

- useful, but less urgent than the stronger inflammatory/core findings

### Kidneys

- `kidneys_hypodensities`

Decision tendency:

- merge into `kidneys_hypoattenuating_lesion` unless we later find a strong reason to keep both

### Pancreas

- `pancreas_duct_prominence`
- `pancreas_postoperative_change`
- `pancreas_fluid_collection`
- `pancreas_calcific_chronic_change`
- `pancreas_status_post_resection`
- `pancreas_fatty_atrophy`
- `pancreas_fatty_infiltration`
- `pancreas_fatty_replacement`

Decision tendency:

- worthwhile refinement set, but not rescue priority because pancreas already performed very well

### Liver

- `liver_biliary_ductal_dilatation`
- `liver_periportal_edema`
- `liver_focal_fatty_change`
- `liver_hepatomegaly`
- `liver_hemangioma`
- `liver_cirrhosis`
- `liver_metastasis`
- `liver_posttreatment_change`

Decision tendency:

- strong concepts, but liver is already stable enough that these can be handled after the main rescue organs

## Family Policy For V2

Keep the family system mostly stable.

Do not introduce new shared families before the second pass unless a later review shows much stronger evidence than we currently have.

Current policy:

- map new stable subtypes into existing families when possible
- use `other_abnormal` when the concept is real but a better shared family is still unclear
- keep proposed families only as review notes for now

## Canonicalization Work

These should be resolved before or during the second pass:

- merge `kidneys_hypodensities` into `kidneys_hypoattenuating_lesion`
- map gallbladder post-cholecystectomy variants to `gallbladder_absent_postop`
- consolidate pancreatic fatty-change variants around the existing `pancreas_atrophy` or a single pancreatic fatty-change concept
- revisit colon inflammatory near-duplicates that differ more in wording than in meaning

## Validation Logic Cleanup

The first pass showed that these flags overfire:

- `normal_with_subtypes`
- `negative_with_primary_subtype`

Before using validation flags in downstream training or evaluation:

- refine contradiction logic so canonical normal subtypes are not over-penalized
- separate true contradictions from expected schema patterns

## Second-Pass Run Scope

Recommended rerun scope after `ontology v2` is in place:

1. all unresolved unique texts
2. all accepted-provisional unique texts for:
   - `Colon`
   - `Small bowel`
   - `Gallbladder`
   - `Kidneys`
3. optional refinement pass for:
   - `Pancreas`
   - `Liver`

If we want the cleanest possible second pass, rerun the entire unique-text inventory for the four hard organs after the ontology updates land.

## Model Policy For Second Pass

Do not switch to a smaller model by assumption.

Instead:

1. keep the current 70B run as reference
2. benchmark any smaller model only on the second-pass subset
3. compare:
   - accepted / unresolved rates
   - provisional rate reduction
   - qualitative behavior in the hard organs

## Acceptance Targets For Second Pass

The second pass should improve the artifact in measurable ways:

- cut provisional rates substantially in `Colon`, `Small bowel`, and `Gallbladder`
- reduce ontology-gap unresolveds from the current baseline
- avoid increasing cross-organ or contextless failures
- preserve confidence separation between accepted, provisional, and unresolved outputs

## Current Implementation Status

Implemented now in `ontology v2` foundation:

- phase-1 promotions for `Colon`
- phase-1 promotions for `Small bowel`
- phase-1 promotions for `Gallbladder`
- phase-1 promotions for `Kidneys`

Still pending:

- canonical merge rules
- contradiction-logic refinement
- second-pass config and run scope selection
- optional pancreas and liver refinement set
