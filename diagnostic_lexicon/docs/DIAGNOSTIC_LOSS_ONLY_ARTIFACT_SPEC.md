# Diagnostic-Loss-Only Artifact Spec

## Purpose

This document defines the artifact shape for testing an improved lexical diagnostic loss **without** enabling the semantic auxiliary-head loss.

The goal is a clean ablation:

```text
L_total = L_CE + beta * L_lexical_diag
```

with:

```text
semantic_loss.enabled = false
```

The lexical diagnostic targets may be derived from semantic-tag artifacts offline, but the training loss itself stays token-probability based. This lets us test whether the old diagnostic-loss idea mainly suffered from weak vocabulary construction.

## Why This Is Separate From Semantic Loss

The semantic loss trains auxiliary classifiers on pooled decoder hidden states.

The lexical diagnostic loss does not need those classifiers. It operates directly on decoder token logits, like the current `BinaryDiagnosticLoss`, but with sample-specific positive and negative vocabularies.

So there are two independent experiments:

- semantic loss: structured hidden-state supervision
- lexical diagnostic loss: better token-level vocabulary regularization

They can be compared separately and combined later.

## Inputs

Primary semantic source:

```text
/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/outputs/semantic_tagging/merlin_converted/consolidation/consolidation_v3/postprocess_v3_clean/semantic_training_targets_v3.parquet
```

Use with:

```text
training_vocab_v3_clean.yaml
tag_consolidation_map_v3_clean.jsonl
review_queue_v3.csv
```

The semantic artifact is used only to build lexical targets. It does not require `SemanticDiagnosticLoss` at training time.

## Output Directory

Recommended pattern:

```text
outputs/diagnostic_lexicon/<dataset_id>/<run_id>/
```

For the first diagnostic-loss-only experiment:

```text
outputs/diagnostic_lexicon/merlin_converted/lexical_diag_v1_from_semantic_v3/
```

## Required Artifacts

### 1. `lexicon_registry_v1.json`

Subtype and family lexical resources derived from real examples.

Each entry should be keyed by organ and canonical label.

Example:

```json
{
  "organ": "Colon",
  "label_type": "subtype",
  "label": "colon_mass",
  "positive_phrases": [
    "colonic mass",
    "mass arising from the colon",
    "rectal mass",
    "enhancing colonic mass"
  ],
  "negative_phrases": [
    "no colonic mass",
    "no rectal mass",
    "without discrete mass"
  ],
  "uncertain_phrases": [
    "possible colonic mass",
    "cannot exclude colonic mass"
  ],
  "confuser_phrases": [
    "mass effect",
    "stool ball",
    "wall thickening without discrete mass"
  ],
  "source": {
    "semantic_artifact": "semantic_training_targets_v3",
    "example_count": 184,
    "review_required": false
  }
}
```

### 2. `sample_level_lexical_targets_v1.jsonl`

One row per usable `(organ, raw_text)` training target. This is the main decoder input artifact.

Example:

```json
{
  "organ": "Colon",
  "raw_text": "A large heterogeneously enhancing mass arising from the descending colon.",
  "normalized_text": "a large heterogeneously enhancing mass arising from the descending colon.",
  "positive_concepts": [
    {
      "source_label": "colon_mass",
      "label_type": "subtype",
      "phrases": ["colonic mass", "colon mass", "rectal mass", "mass"],
      "weight": 1.0
    }
  ],
  "negative_concepts": [
    {
      "source_label": "colon_normal",
      "label_type": "subtype",
      "phrases": ["normal colon", "unremarkable colon", "no abnormality"],
      "weight": 0.25
    }
  ],
  "uncertain_concepts": [],
  "sample_weight": 0.95,
  "review_required": false,
  "decision_status": "accepted",
  "source_targets": {
    "family_targets": {"mass_or_malignancy": 0.9},
    "subtype_targets": {"colon_mass": 1.0}
  },
  "provenance": {
    "semantic_source_sha256": "...",
    "lexicon_registry_sha256": "..."
  }
}
```

### 3. `tokenized_lexical_targets_v1.pt`

Tokenizer-specific cache for fast training.

Recommended structure:

```python
{
    "tokenizer_name": "Qwen/Qwen2.5-0.5B",
    "source_jsonl_sha256": "...",
    "rows": [
        {
            "key": ("Colon", "a large heterogeneously enhancing mass arising from the descending colon."),
            "positive_token_ids": [1234, 5678, ...],
            "negative_token_ids": [111, 222, ...],
            "positive_weight": 1.0,
            "negative_weight": 0.25,
            "sample_weight": 0.95,
            "review_required": False
        }
    ]
}
```

This cache is what the decoder dataset should load for a diagnostic-loss-only run.

### 4. `reports/coverage.md`

Must answer:

- how many semantic training targets received lexical targets
- how many were excluded because `review_required=true`
- top labels by lexical coverage
- labels with no lexical coverage
- average `|V_b+|` and `|V_b-|`

## Deriving `V_b+` And `V_b-`

For each semantic training target row:

1. Read `subtype_targets` and `family_targets`.
2. Expand each active subtype/family through `lexicon_registry_v1.json`.
3. Add positive phrases for active positive labels to `V_b+`.
4. Add normal/contradictory phrases to `V_b-` when they would conflict with the target.
5. For normal/negative rows, add abnormal phrases from the same organ to `V_b-`.
6. Downweight or exclude `review_required=true`.

Conservative first-run policy:

- include only `review_required=false`
- use subtype lexical targets only when the subtype label is frequent enough
- use family lexical targets broadly
- keep negative targets modestly weighted to avoid punishing valid alternate wording

## Recommended First Loss

Use a sample-specific version of the proposed diagnostic loss:

```text
p+_{b,t} = sum_{v in V_b+} softmax(logits[b,t])[v]
p-_{b,t} = sum_{v in V_b-} softmax(logits[b,t])[v]

s+_b = log(eps + sum_t p+_{b,t})
s-_b = tau^{-1} LSE_t(tau * log(p-_{b,t} + eps))

L_lexical_diag(b) = w_b * (-s+_b + alpha * s-_b)
```

Batch:

```text
L_lexical_diag = sum_b L_lexical_diag(b) / sum_b w_b
```

Total:

```text
L_total = L_CE + beta * L_lexical_diag
```

## Alternative Simpler First Loss

The current `BinaryDiagnosticLoss` already has a stable "any token probability" helper.

A lower-risk implementation can start with:

```text
P_any(V) = 1 - product_t(1 - sum_{v in V} P[t, v])

L_pos = -log(P_any(V_b+))
L_neg = -log(1 - P_any(V_b-))

L_lexical_diag = w_b * (L_pos + alpha * L_neg)
```

This is closer to the current code and easier to debug. The LSE version can be the second variant.

## Training Ablations

Run these as separate experiments:

1. `CE only`
2. `CE + old BinaryDiagnosticLoss`
3. `CE + lexical_diag_v1`, semantic loss disabled
4. `CE + semantic_loss_v3`, lexical diagnostic disabled
5. `CE + semantic_loss_v3 + lexical_diag_v1`

The third run is the key diagnostic-loss-only experiment.

## Decoder Config Shape

For the diagnostic-loss-only run:

```yaml
diagnostic_loss:
  enabled: true
  variant: sample_specific_lexical
  target_jsonl: /net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/outputs/diagnostic_lexicon/merlin_converted/lexical_diag_v1_from_semantic_v3/sample_level_lexical_targets_v1.jsonl
  tokenized_target_cache: /net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/outputs/diagnostic_lexicon/merlin_converted/lexical_diag_v1_from_semantic_v3/tokenized_lexical_targets_v1.pt
  weight: 0.05
  negative_weight: 0.25
  tau: 8.0
  epsilon: 1.0e-6
  include_review_required: false

semantic_loss:
  enabled: false
```

Start with a small `diagnostic_loss.weight`, because lexical losses can otherwise push the decoder toward keyword stuffing.

## Relationship To `Merlin_metadata_hf_clean.csv`

The CSV can remain useful for comparison and fallback:

- old binary diagnostic loss uses lesion labels from CSV
- new lexical diagnostic loss can be derived from semantic targets
- CSV lesion labels can be used as an additional consistency report

The new artifact does not require the CSV at training time if `sample_level_lexical_targets_v1.jsonl` already exists.

## Acceptance Criteria

A generated diagnostic-loss-only artifact is acceptable if:

- every target row has deterministic source provenance
- `review_required=false` can be selected cleanly
- `V_b+` and `V_b-` are non-empty only when justified
- normal rows do not get broad false positive vocabularies
- mixed rows can carry both positive and negative lexical constraints
- tokenized caches are tied to the exact decoder tokenizer
- coverage report makes it obvious which organs/subtypes are weak

## Expected Outcome

This artifact lets us test:

- whether the original diagnostic-loss idea improves when vocabularies are sample-specific and data-grounded
- whether lexical regularization alone can beat the old generic pathology-word baseline
- whether semantic loss adds value beyond better lexical supervision

