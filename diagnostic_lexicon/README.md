# Diagnostic Lexicon

`diagnostic_lexicon/` is a standalone subproject for building lexical resources and sample-level lexical target sets for a future semantic-aware diagnostic loss.

It is intentionally separate from:

- [semantic_tagging](/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/semantic_tagging:1)

because it depends on finished semantic-tag artifacts, but should not modify or destabilize the semantic-tagging pipeline itself.

## What this subproject will produce

1. **Subtype-centered lexical registries**
   - positive lexicalizations
   - negative lexicalizations
   - uncertain lexicalizations
   - confusers / exclusions

2. **Family-centered lexical registries**
   - shared wording for family-level diagnostic regularization

3. **Sample-level lexical target specs**
   - `V_b^+`
   - `V_b^-`
   - optional normal-word sets
   - derived from semantic tags plus lexical registries

4. **Tokenizer-aware caches**
   - phrase-to-token-id mappings for efficient decoder loss computation

5. **Reports**
   - coverage
   - ambiguity
   - unsupported subtype tail
   - final recommended training assembly

## Why this exists

The current decoder diagnostic loss is still built around generic pathology words like:

- `mass`
- `cyst`
- `lesion`
- `tumor`

That is too coarse.

The semantic tag pipeline gives us the structured meaning layer.
This subproject builds the **lexical realization layer** on top of that semantic structure, so that a future diagnostic loss can use:

- semantic tags as the main target
- lexical target sets as an optional wording regularizer

## Design principle

This pipeline should be:

- **grounded in real data**
- **versioned**
- **subtype-centered**
- **compatible with semantic-tag outputs**
- **queueable as its own Slurm job later**

## Important note

This folder currently contains the **designed pipeline and scaffolding**.
It is meant to define the architecture, artifacts, prompts, and job shape before full code implementation.

The implementation plan is in:

- [IMPLEMENTATION_PLAN.md](/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/diagnostic_lexicon/IMPLEMENTATION_PLAN.md:1)

The diagnostic-loss-only artifact contract is in:

- [DIAGNOSTIC_LOSS_ONLY_ARTIFACT_SPEC.md](/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/diagnostic_lexicon/docs/DIAGNOSTIC_LOSS_ONLY_ARTIFACT_SPEC.md:1)

The current preparation command is documented in:

- [prepare_diagnostic_loss_only_artifacts.md](/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/diagnostic_lexicon/runbooks/prepare_diagnostic_loss_only_artifacts.md:1)
