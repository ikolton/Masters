# Diagnostic Lexicon Pipeline Architecture

## High-level flow

1. merge best semantic sources
2. build subtype example catalogs
3. mine real candidate phrases from the corpus
4. run subtype-centered LLM lexicalization
5. validate and consolidate lexical registries
6. build sample-level lexical target sets
7. export tokenizer-aware caches for training

## Key design principle

The LLM is not the first source of truth.

The first source of truth is:

- semantic tags
- real accepted examples
- mined phrases from the corpus

The LLM is used to:

- normalize
- classify
- consolidate
- lightly extend

those grounded candidates.

## Main artifacts

### Registry artifacts

- `subtype_lexicon_registry.json`
- `family_lexicon_registry.json`
- `tokenized_lexicon_registry.json`

### Training artifacts

- `sample_level_lexical_targets.parquet`

### Reports

- `summary.md`
- `coverage_analysis.md`
- `confuser_analysis.md`

## Relationship to final loss

This pipeline prepares lexical target sets for formulas of the form:

- positive lexical target set `V_b^+`
- negative lexical target set `V_b^-`

But those lexical targets should be used beside, not instead of, the semantic tag loss.
