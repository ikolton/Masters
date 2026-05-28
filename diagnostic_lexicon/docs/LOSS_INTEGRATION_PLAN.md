# Lexical Diagnostic Loss Integration Plan

## Goal

Use the outputs of `diagnostic_lexicon/` to provide a lexical regularization term for the decoder that is:

- sample-specific
- organ-aware
- subtype-aware

instead of based on one global pathology-word list.

## Expected training inputs

For each sample `b`:

- semantic targets:
  - `normality`
  - `polarity`
  - `certainty`
  - subtype tags
- lexical targets:
  - `V_b^+`
  - `V_b^-`

## Final intended role

- semantic tag loss = main auxiliary diagnostic loss
- lexical target loss = smaller wording regularizer

## Diagnostic-loss-only ablation

The lexical diagnostic loss should also be testable without semantic auxiliary heads:

```text
L_total = L_CE + beta * L_lexical_diag
```

with:

```text
semantic_loss.enabled = false
```

This lets us answer whether the original diagnostic-loss idea was mostly limited by weak generic vocabularies, or whether full semantic supervision is needed.

The concrete artifact schema for that ablation is defined in:

- [DIAGNOSTIC_LOSS_ONLY_ARTIFACT_SPEC.md](/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/diagnostic_lexicon/docs/DIAGNOSTIC_LOSS_ONLY_ARTIFACT_SPEC.md:1)

## Why keep the lexical layer

Even with strong semantic supervision, wording still matters.

Example:

- `colon_mass`

may be semantically correct, but we may still want the decoder to prefer wording like:

- `mass`
- `carcinoma`
- `mass arising from the colon`

over vague alternatives.

That is the job of the lexical layer.
