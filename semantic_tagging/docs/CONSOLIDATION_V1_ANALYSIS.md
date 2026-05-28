# Consolidation V1 Analysis

## Run Identity

- Slurm job: `17476090`
- State: `COMPLETED`
- Exit code: `0:0`
- Elapsed: `00:04:34`
- Output directory: `/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/outputs/semantic_tagging/merlin_converted/consolidation/consolidation_v1`
- Config: `configs/merlin_consolidation_v1.yaml`

## What This Run Did

This was not another full semantic-tagging pass over all unique report texts.
It was a vocabulary-level consolidation pass over observed subtype labels from the composed best-current semantic layer:

- base run: `vllm_full`
- organ overrides:
  - `Colon`: `vllm_targeted_v3_candidate`
  - `Gallbladder`: `vllm_targeted_v4_candidate`
  - `Kidneys`: `vllm_targeted_v3_candidate`
  - `Small bowel`: `vllm_targeted_v3_candidate`

The deterministic builder first produced:

- `composed_validated_decisions.jsonl`
- `observed_tag_stats.jsonl`
- `observed_tag_stats.csv`
- `llm_consolidation_items.jsonl`
- `manifest.json`
- `reports/consolidation_input_report.md`

Then the LLM produced:

- `llm_consolidation_raw.jsonl`
- `llm_consolidation_decisions.jsonl`
- `training_vocab_draft.json`
- `llm_consolidation_summary.json`
- `reports/llm_consolidation_report.md`

## Mechanical Health

The run is mechanically healthy.

- composed decisions: `60851`
- observed subtype labels: `592`
- LLM consolidation items: `592`
- valid parsed LLM decisions: `592`
- invalid outputs: `0`

LLM mode distribution:

- `direct`: `267`
- `merged`: `221`
- `family_only`: `87`
- `exclude`: `17`

The speed was appropriate for this stage:

- `592` vocabulary-level decisions in about `2.5` minutes after server startup
- end-to-end Slurm elapsed time: `4m34s`

## Training Vocabulary Draft Size

The draft vocabulary contains `329` organ-specific labels before deterministic cleanup.

Labels by organ:

- `Adrenal glands`: `15`
- `Colon`: `44`
- `Gallbladder`: `19`
- `Kidneys`: `43`
- `Liver`: `42`
- `Pancreas`: `25`
- `Prostate`: `18`
- `Small bowel`: `44`
- `Spleen`: `31`
- `Stomach`: `23`
- `Urinary bladder`: `25`

This is a plausible size for an auxiliary diagnostic-loss vocabulary, but it is still too raw to use directly.

## Good Signals

The run made many medically reasonable consolidation decisions.

Examples:

- `colon_colectomy`, `colon_colostomy`, `colon_ileostomy`, `colon_j-pouch` were merged into `colon_postsurgical_change`.
- `gallbladder_collapse` was merged into `gallbladder_decompressed`.
- `pancreas_status_post_resection` was merged into `pancreas_postoperative_change`.
- `pancreas_fatty_atrophy` was merged into `pancreas_atrophy`.
- obvious artifacts or adjacent-organ leakage were excluded:
  - `kidneys_streak_artifact`
  - `pancreas_streak_artifact`
  - `small_bowel_appendicitis`
  - `prostate_bladder_wall_thickening`
  - `liver_excluded_from_field_of_view`

This supports the strategy: an LLM is useful here as a constrained reviewer over real dataset-derived labels, not as an unconstrained ontology generator.

## Problems Found

### 1. Family-Only Labels Are Not Yet Normalized

Many `family_only` outputs used the organ name itself as `training_family`, for example:

- `Adrenal glands`
- `Colon`
- `Gallbladder`
- `Kidneys`
- `Liver`
- `Pancreas`
- `Prostate`
- `Small bowel`
- `Spleen`
- `Stomach`
- `Urinary bladder`

These are parse-valid but not training-ready. They should not become loss labels.

Current count:

- non-excluded rows with `loss_weight = 0.0`: `87`
- these correspond almost exactly to the `family_only` bucket

Interpretation:

- The model correctly decided many rare tags should not become subtype labels.
- But our output schema allowed a vague family placeholder.
- We need deterministic postprocessing to map these to explicit coarse families or remove them from subtype loss.

### 2. Some Merges Are Medically Questionable

The LLM made several merges that should not be accepted without review.

Examples:

- `adrenal_atrophy -> adrenal_enlargement`
  - This is semantically wrong; atrophy and enlargement are opposite morphology directions.
- `adrenal_encasement -> adrenal_metastasis`
  - Encasement can reflect malignant involvement, but it is not equivalent to metastasis.
- `liver_ill_defined_mass -> liver_metastasis`
  - An ill-defined mass is not necessarily metastatic.
- `kidneys_hydroureter -> kidneys_hydronephrosis`
  - Related but not identical; may be acceptable as a coarse obstructive-uropathy family, not as a strict subtype merge.
- `gallbladder_polyp_or_stone -> gallbladder_gallstones`
  - Reasonable for some cases, but the original label explicitly preserves ambiguity.

This is the expected failure mode of an LLM consolidation pass: it tends to make clinically plausible but sometimes overly aggressive semantic merges.

### 3. Human Review Flag Is Too Sparse

The LLM marked only `10` rows as `needs_human_review=true`.

That is too low. A robust medical workflow should force review for:

- rare direct labels
- all merges where source and target are not deterministic wording variants
- all labels with ambiguity words like `or`, `indeterminate`, `ill_defined`, `involvement`, `encasement`
- all family-only rows
- all rows where `training_family` is not from an allowed controlled family list

### 4. Draft Vocabulary Is Too Raw For Immediate Loss Use

The `training_vocab_draft.json` is a good candidate artifact, but it should not be plugged directly into training yet.

Reasons:

- It contains organ-name pseudo-families.
- It includes zero-weight labels.
- It has some aggressive merges.
- It does not yet enforce a clean separation between:
  - subtype labels
  - coarse family labels
  - excluded labels
  - review-only labels

## Rating

Mechanical execution: `9/10`

- Clean Slurm finish.
- No invalid JSON.
- All expected artifacts were produced.
- Strong reproducibility via config and manifest.

Semantic usefulness: `7/10`

- The run is very useful as a first consolidation draft.
- It correctly collapses many obvious wording variants and identifies many artifacts.
- It is not safe as the final training vocabulary without deterministic postprocessing and a focused review layer.

Training-readiness today: `5/10`

- Good enough to inspect and build rules from.
- Not good enough to replace diagnostic loss directly.

## Recommended Next Step

Do not rerun the LLM immediately.

Instead, create a deterministic consolidation postprocessor that reads `llm_consolidation_decisions.jsonl` and writes:

- `tag_consolidation_map_v1.jsonl`
- `training_vocab_v1.yaml`
- `consolidation_review_queue_v1.csv`
- `semantic_training_targets_v1.parquet`
- `reports/training_vocab_v1_report.md`

The postprocessor should:

1. Accept `direct` labels only if they meet frequency and naming rules.
2. Accept `merged` labels only if source-target relation passes deterministic safety rules or is explicitly allowlisted.
3. Convert vague `family_only` outputs into controlled coarse families or remove them from subtype loss.
4. Force questionable medical merges into a review queue.
5. Assign loss weights deterministically:
   - direct frequent clean labels: `1.0`
   - safe merged wording variants: `0.8-1.0`
   - coarse family-only labels: `0.2-0.4`
   - review-only / excluded labels: `0.0`

After this, we can decide whether a second LLM pass is needed.

The second LLM pass, if used, should be smaller and targeted only at the review queue, not the whole vocabulary.
