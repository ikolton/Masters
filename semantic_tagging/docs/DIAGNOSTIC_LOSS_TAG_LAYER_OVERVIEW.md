# Semantic Tag Layer And How It Replaces The Current Diagnostic Loss

## Purpose

This note explains:

1. what the current diagnostic loss is actually doing
2. why the semantic tag layer is different and better
3. what tag fields we currently have
4. what one tagged organ report looks like in practice
5. how to implement a tag-based diagnostic loss instead of the current one

This is meant to be presentation-ready for a coworker who needs to understand both the semantic layer and its role in training.

## Executive Summary

The current decoder diagnostic loss is still a **binary pathology-word loss**.

It mainly asks:

- if the sample is lesion-positive, did the decoder place probability mass on generic words like `mass`, `cyst`, `lesion`, `tumor`?
- if the sample is lesion-negative, did the decoder avoid those generic words?

The semantic tag layer is different.

It lets us supervise:

- whether an organ is normal, abnormal, absent-postop, or mixed
- whether a finding is positive, negative, or mixed
- whether the statement is definite, probable, or indeterminate
- which organ-specific abnormality type is present
- which additional secondary abnormality types co-occur

So the difference is:

- **old loss**: generic pathology-word presence
- **new loss**: organ-specific semantic state

## What The Current Diagnostic Loss Actually Does

The current loss is implemented in:

- [decoder/losses.py](/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/src/organ_seg_clip/decoder/losses.py:1)
- wired into the decoder in [decoder/model.py](/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/src/organ_seg_clip/decoder/model.py:1)
- configured in [config/schemas.py](/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/src/organ_seg_clip/config/schemas.py:557)

### Current targets

It uses:

- `lesion_labels`
- `lesion_mask`
- `small_bowel_mask`
- `target_texts`
- hard-coded `pathology_words`
- hard-coded `normal_words`

Default pathology words:

- `lesion`
- `lesions`
- `cyst`
- `cysts`
- `mass`
- `masses`
- `nodule`
- `nodules`
- `metastasis`
- `metastases`
- `tumor`
- `tumour`

Default normal words:

- `unremarkable`
- `normal`
- `within normal limits`
- `no abnormality`
- `no focal abnormality`

### Current positive behavior

For lesion-positive rows, it rewards the decoder if **any pathology token** gets probability mass and penalizes generic normal words.

Conceptually:

- lesion-positive -> “say something pathology-like”
- also avoid “normal / unremarkable”

### Current negative behavior

For lesion-negative rows, it penalizes pathology-token probability mass, unless the target text itself contains one of those pathology concepts and the helper filter removes it from the absent set.

This behavior is explicitly tested in:

- [test_decoder_stage2.py](/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/tests/unit/test_decoder_stage2.py:116)

### What the current loss cannot express

It cannot explicitly model:

- organ-specific diagnosis identity
- positive vs negative findings within the same text
- uncertainty
- postoperative absence
- composite findings
- contradiction like “distended gallbladder, no cholecystitis”

So the current diagnostic loss is useful as a weak bias, but it is semantically crude.

## Why The Tag Layer Is Different

The semantic tag layer supervises **meaning**, not just token presence.

Instead of asking:

- “did the decoder say any pathology-like word?”

it asks:

- “what exact semantic state should the decoder express for this organ?”

That includes:

- coarse state
- polarity
- certainty
- organ-specific subtype
- optional secondary subtypes

## A Concrete Example Of The Difference

Input organ text:

`The gallbladder is distended. No cholecystitis.`

### What the old loss sees

Roughly:

- some abnormality may exist
- maybe reward pathology-ish wording
- maybe penalize `normal`

It cannot represent the internal structure of the statement.

### What the tag layer sees

```json
{
  "organ": "Gallbladder",
  "normality": "mixed",
  "polarity": "mixed",
  "certainty": "definite",
  "primary_subtype": "gallbladder_distension",
  "secondary_subtypes": [],
  "confidence": 0.94,
  "decision_status": "accepted"
}
```

This is much richer:

- positive abnormality exists
- that abnormality is `gallbladder_distension`
- the report also explicitly negates a related inflammatory diagnosis
- therefore the overall semantic state is mixed

That is the kind of behavior we actually want the decoder to learn.

## What A Semantic Tag Record Contains

Each organ-text decision contains:

- `organ`
- `raw_text`
- `normalized_text`
- `normality`
- `polarity`
- `certainty`
- `primary_subtype`
- `secondary_subtypes`
- `modifiers`
- `evidence_spans`
- `confidence`
- `decision_status`
- `decision_source`
- `ontology_version`
- `proposed_new_subtype`
- `proposed_new_family`
- `validation_flags`

These fields are defined in:

- [types.py](/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/semantic_tagging/src/semantic_tagging/types.py:1)

The already-materialized training artifact keeps the most important subset:

- `normality`
- `polarity`
- `certainty`
- `primary_subtype`
- `secondary_subtypes`
- `confidence_weight`
- `contradiction_flags`
- `provenance`
- lesion / organ-abnormal metadata

This materialization step is implemented in:

- [materialize.py](/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/semantic_tagging/src/semantic_tagging/materialize.py:1)

## Core Tag Axes

### 1. Normality

Current values:

- `normal`
- `abnormal`
- `absent_postop`
- `mixed`

### 2. Polarity

Current values:

- `positive`
- `negative`
- `mixed`

### 3. Certainty

Current values:

- `definite`
- `probable`
- `indeterminate`

### 4. Subtypes

Each record can have:

- one `primary_subtype`
- zero or more `secondary_subtypes`

This matters because many organ findings are compositional and should not be flattened into one token or one class.

## Example Tagged Organ Reports

### Example 1: normal colon

Input:

`The colon is normal in caliber with no wall thickening, obstruction, or free air.`

Tag record:

```json
{
  "organ": "Colon",
  "normality": "normal",
  "polarity": "negative",
  "certainty": "definite",
  "primary_subtype": "colon_normal",
  "secondary_subtypes": [],
  "confidence": 0.98,
  "decision_status": "accepted"
}
```

### Example 2: focal colon mass

Input:

`A large heterogeneously enhancing mass arising from the descending colon near the splenic flexure measures 6.7 x 6.1 cm.`

Tag record:

```json
{
  "organ": "Colon",
  "normality": "abnormal",
  "polarity": "positive",
  "certainty": "probable",
  "primary_subtype": "colon_mass",
  "secondary_subtypes": [],
  "modifiers": ["size_present"],
  "confidence": 0.95,
  "decision_status": "accepted"
}
```

### Example 3: postoperative small-bowel leak

Input:

`Gas and fluid collection adjacent to the anastomotic site suggests an anastomotic leak.`

Tag record:

```json
{
  "organ": "Small bowel",
  "normality": "abnormal",
  "polarity": "positive",
  "certainty": "probable",
  "primary_subtype": "small_bowel_anastomotic_leak",
  "secondary_subtypes": ["small_bowel_anastomosis"],
  "modifiers": ["postoperative"],
  "confidence": 0.93,
  "decision_status": "accepted"
}
```

### Example 4: positive finding plus explicit negation

Input:

`Gallstones without CT findings of cholecystitis.`

Tag record:

```json
{
  "organ": "Gallbladder",
  "normality": "mixed",
  "polarity": "mixed",
  "certainty": "definite",
  "primary_subtype": "gallbladder_gallstones",
  "secondary_subtypes": [],
  "confidence": 0.95,
  "decision_status": "accepted"
}
```

This is exactly the kind of case where the tag layer is much better than the old pathology-word loss.

## What Tags We Currently Have

The exact subtype vocabularies are versioned in:

- baseline full-run bundle: [v1_full_run_baseline](/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/semantic_tagging/ontology_versions/v1_full_run_baseline:1)
- best finished hard-organ refinement bundle: [v3_second_pass_candidate](/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/semantic_tagging/ontology_versions/v3_second_pass_candidate:1)
- next cleanup bundle: [v4_second_pass_candidate](/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/semantic_tagging/ontology_versions/v4_second_pass_candidate:1)

Examples by organ:

### Pancreas

- `pancreas_normal`
- `pancreas_cystic_lesion_ipmn_like`
- `pancreas_mass`
- `pancreas_atrophy`

### Liver

- `liver_normal`
- `liver_steatosis`
- `liver_cystic_lesion`
- `liver_indeterminate_hypodensity`

### Adrenal glands

- `adrenal_glands_normal`
- `adrenal_glands_thickening`
- `adrenal_glands_nodule_adenoma`

### Small bowel

- `small_bowel_normal`
- `small_bowel_dilatation`
- `small_bowel_wall_thickening`
- `small_bowel_obstruction`
- `small_bowel_submucosal_edema`
- `small_bowel_fecalization`
- `small_bowel_postoperative_change`
- `small_bowel_hyperemia`
- `small_bowel_herniation`
- `small_bowel_narrowing`
- `small_bowel_mesenteric_infiltration`
- `small_bowel_anastomosis`
- `small_bowel_anastomotic_leak`
- `small_bowel_fistulous_tract`
- `small_bowel_ileus`

### Colon

- `colon_normal`
- `colon_diverticulosis`
- `colon_wall_thickening`
- `colon_stool_burden`
- `colon_postsurgical_change`
- `colon_submucosal_edema`
- `colon_pericolonic_stranding`
- `colon_colitis`
- `colon_mucosal_hyperemia`
- `colon_perirectal_stranding`
- `colon_mass`
- `colon_abscess`
- `colon_pericolonic_fluid_collection`
- `colon_perforation`
- `colon_distension`
- `colon_narrowing`
- `colon_fistula`

### Gallbladder

- `gallbladder_normal`
- `gallbladder_absent_postop`
- `gallbladder_gallstones`
- `gallbladder_distension`
- `gallbladder_wall_thickening`
- `gallbladder_pericholecystic_fluid`
- `gallbladder_sludge`
- `gallbladder_focal_fundal_adenomyomatosis`
- `gallbladder_cholecystitis`
- `gallbladder_polyp_or_stone`

`v4` additionally prepares:

- `gallbladder_collapse`
- `gallbladder_incompletely_assessed`
- `colon_perirectal_fluid_collection`

### Kidneys

Kidneys are already strong enough to use, though they still have a small provisional tail for rarer anatomy and lesion variants.

## Which Finished Artifacts To Use Right Now

### Dataset-wide base

Full run:

- summary: [vllm_full summary](/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/outputs/semantic_tagging/merlin_converted/vllm_full/reports/summary.md)
- row-level tags: [row_level_tags.parquet](/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/outputs/semantic_tagging/merlin_converted/vllm_full/row_level_tags.parquet:1)
- loss-ready targets: [loss_ready_targets.parquet](/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/outputs/semantic_tagging/merlin_converted/vllm_full/loss_ready_targets.parquet:1)

### Hard-organ override

Best finished targeted refinement:

- summary: [vllm_targeted_v3_candidate summary](/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/outputs/semantic_tagging/merlin_converted/vllm_targeted_v3_candidate/reports/summary.md)
- row-level tags: [row_level_tags.parquet](/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/outputs/semantic_tagging/merlin_converted/vllm_targeted_v3_candidate/row_level_tags.parquet:1)
- loss-ready targets: [loss_ready_targets.parquet](/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/outputs/semantic_tagging/merlin_converted/vllm_targeted_v3_candidate/loss_ready_targets.parquet:1)

### Practical assembly

Recommended current training source:

1. start from the full-run artifact
2. replace rows for:
   - `Colon`
   - `Gallbladder`
   - `Kidneys`
   - `Small bowel`
   with the targeted `v3` rows
3. materialize or rematerialize the merged loss-ready table

That is the best current semantic label layer.

## How To Replace The Current Diagnostic Loss

There are two realistic migration paths.

### Option A: full replacement

Replace the current `BinaryDiagnosticLoss` completely.

Inputs to the new semantic diagnostic loss:

- `normality`
- `polarity`
- `certainty`
- `primary_subtype`
- `secondary_subtypes`
- `confidence_weight`
- `contradiction_flags`

This removes dependence on hand-written pathology/normal word lists.

### Option B: transition run

Keep both for one experiment:

- CE loss
- current binary pathology-word loss
- new semantic tag loss

Then compare:

- old diagnostic loss only
- semantic tag loss only
- both combined

This is safer for transition, but the long-term goal should still be replacement.

## How The Tag-Based Diagnostic Loss Should Be Structured

The clean implementation is layered.

### Layer 1: normality loss

Predict:

- `normal`
- `abnormal`
- `absent_postop`
- `mixed`

Why this is better than the old loss:

- old loss cannot distinguish `normal` from `absent_postop`
- old loss cannot represent mixed statements

### Layer 2: polarity loss

Predict:

- `positive`
- `negative`
- `mixed`

Why this matters:

- old loss cannot distinguish a positive disease statement from an explicit negation
- tag loss can train the decoder not to hallucinate positive disease language into negative or mixed reports

### Layer 3: certainty loss

Predict:

- `definite`
- `probable`
- `indeterminate`

Why this matters:

- old loss treats pathology-word presence as equally good regardless of uncertainty
- tag loss can teach the decoder to preserve indeterminate wording

### Layer 4: subtype supervision

Target:

- `primary_subtype`
- `secondary_subtypes`

Implementation choices:

#### Option 1: single-label + multi-label

- one organ-specific single-label head for `primary_subtype`
- one organ-specific multi-label head for `secondary_subtypes`

This is the most faithful to the data model.

#### Option 2: one multi-label subtype head

Treat all subtypes as a multi-label target and ignore the primary/secondary distinction.

This is simpler to implement and still much better than the current pathology-word loss.

### Layer 5: contradiction penalty

Use `contradiction_flags`, `normality`, and `polarity` to penalize semantically inconsistent decoder behavior.

Examples:

- if `polarity = negative`, penalize positive disease expression
- if `normality = normal`, penalize abnormal subtype expression
- if `absent_postop`, penalize normal-present-organ wording
- if `gallbladder_distension` is present but `cholecystitis` is explicitly negated, do not reward cholecystitis-like wording

## How To Weight The New Loss

Recommended sample weighting:

- `accepted`: `1.0`
- `accepted_provisional`: `0.35 - 0.5`
- `unresolved`: `0.0`

Then optionally multiply by semantic confidence:

- `final_weight = status_weight * confidence`

This is another big advantage over the old loss:

- the old loss is basically flat over usable lesion-labeled rows
- the tag layer gives us explicit supervision quality

## Minimal Useful First Version

If we want the smallest meaningful replacement, do not try to model everything at once.

Best first semantic diagnostic loss:

1. `normality`
2. `polarity`
3. multi-label subtype presence

That already beats the old loss conceptually because it supervises:

- which organ state should be expressed
- whether the statement is positive/negative/mixed
- which organ-specific abnormality is present

Then add later:

4. `certainty`
5. contradiction penalties
6. primary vs secondary distinction

## Concrete Replacement Formula

A clean replacement would look like:

```text
total_loss = ce_loss
           + w_norm * normality_loss
           + w_pol * polarity_loss
           + w_cert * certainty_loss
           + w_sub * subtype_loss
           + w_contra * contradiction_penalty
```

This replaces:

```text
total_loss = ce_loss + current_binary_pathology_word_loss
```

## Why This Should Work Better

The current loss says:

- “say some pathology-like word if lesion-positive”

The tag-based loss says:

- “express the right organ-specific abnormality”
- “express the right negation state”
- “express the right certainty”
- “avoid contradictory semantic states”

That is the real reason the tag layer is better. It aligns supervision with the actual structure of the organ reports instead of just with token presence.

## Current Recommendation

If we were integrating this into training now, I would recommend:

1. use the full-run artifact as the global base
2. override the four hard organs with targeted `v3`
3. build a semantic diagnostic loss around:
   - `normality`
   - `polarity`
   - subtype presence
4. weight by `decision_status` and `confidence`
5. run one transition ablation against the current `BinaryDiagnosticLoss`
6. keep `v4` as the next cleanup pass, mainly for residual gallbladder behavior

That gives us a diagnostic loss based on organ-specific meaning, not just on pathology keywords.
