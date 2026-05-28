# Diagnostic Lexicon Pipeline Implementation Plan

## Summary

Build a standalone subproject at:

- `Masters/diagnostic_lexicon/`

This pipeline will build lexical resources for a future semantic-aware diagnostic loss.

It is **not** a replacement for `semantic_tagging`.
It is a downstream system that consumes:

- finished semantic-tag artifacts
- ontology versions
- optional lesion metadata from:
  - `/net/scratch/hscra/plgrid/plgikolton/Magisterka/Merlin_metadata_hf_clean.csv`

and produces:

- lexical registries for organ subtypes and families
- sample-level `V_b^+` / `V_b^-` targets
- tokenizer-indexed caches
- reports for loss integration

The unit of LLM work here is primarily:

- **organ subtype**

not:

- every raw sample independently

That is the main design choice keeping this grounded and efficient.

## Why this pipeline is needed

We now have a semantic tag layer that can supervise:

- `normality`
- `polarity`
- `certainty`
- `primary_subtype`
- `secondary_subtypes`

This is the main semantic signal.

But the original proposed diagnostic loss family still has value as a **lexical regularizer** if we can define much better:

- positive word/phrase sets
- negative word/phrase sets
- uncertainty phrases
- confusers

The role of this new pipeline is therefore:

- not to replace semantic tagging
- but to build a **lexical layer on top of semantic tagging**

## Relationship to semantic tagging

### Semantic tagging provides

- structured semantic labels
- ontology versions
- confidence / provisional / unresolved status
- examples of real phrasing for each subtype

### Diagnostic lexicon provides

- lexical realizations of those subtype/family semantics
- sample-level lexical target sets for token-probability regularization

### Final loss design

Long-term expected use:

- CE loss
- semantic tag loss as main diagnostic supervision
- lexical diagnostic loss as smaller wording regularizer

## Core Design Choice

### What we will **not** do

We will **not** run another raw sample-by-sample “good words / bad words” pipeline over the whole dataset.

That would be:

- noisy
- unstable
- overly dependent on individual phrasing
- too easy to overfit to local wording accidents

### What we **will** do

We will run a **subtype-centered lexical mining pipeline**:

1. collect accepted semantic-tagged examples for a subtype
2. mine candidate phrases from the real data
3. ask an LLM to normalize and classify those candidates
4. build structured lexical registries
5. derive `V_b^+` and `V_b^-` from semantic labels plus registries

That keeps the system grounded in actual corpus evidence.

## Inputs

### Primary semantic inputs

Recommended best-current semantic source:

1. full-run base:
   - `outputs/semantic_tagging/merlin_converted/vllm_full/`
2. targeted hard-organ override:
   - `outputs/semantic_tagging/merlin_converted/vllm_targeted_v3_candidate/`

Practical merged assembly for this pipeline:

- start from the full-run row-level or unique-text decisions
- replace rows/decisions for:
  - `Colon`
  - `Gallbladder`
  - `Kidneys`
  - `Small bowel`
  with the targeted `v3` outputs

This merged semantic view becomes the lexical pipeline’s source of truth.

### Ontology inputs

Use the ontology version associated with the chosen semantic source.

For hard-organ refined work today, that means:

- `semantic_tagging/ontology_versions/v3_second_pass_candidate/`

and later `v4` if that cleanup pass wins.

### Optional metadata input

- `/net/scratch/hscra/plgrid/plgikolton/Magisterka/Merlin_metadata_hf_clean.csv`

This CSV should be treated as:

- coarse lesion metadata
- optional consistency / enrichment signal

It should **not** be treated as the main semantic source.

## Outputs

Root output pattern:

- `outputs/diagnostic_lexicon/<dataset_id>/<run_id>/`

Required artifacts:

- `semantic_source_manifest.json`
- `subtype_example_catalog.parquet`
- `family_example_catalog.parquet`
- `candidate_phrase_mining.parquet`
- `raw_llm_lexicon_decisions.jsonl`
- `validated_lexicon_entries.parquet`
- `subtype_lexicon_registry.json`
- `family_lexicon_registry.json`
- `tokenized_lexicon_registry.json`
- `sample_level_lexical_targets.parquet`
- `reports/summary.md`
- `reports/coverage_analysis.md`
- `reports/confuser_analysis.md`

## Subproject Structure

This subproject should own its own:

- docs
- prompts
- configs
- runbooks
- examples
- code

Suggested structure:

- `diagnostic_lexicon/README.md`
- `diagnostic_lexicon/IMPLEMENTATION_PLAN.md`
- `diagnostic_lexicon/pyproject.toml`
- `diagnostic_lexicon/apps/`
- `diagnostic_lexicon/configs/`
- `diagnostic_lexicon/docs/`
- `diagnostic_lexicon/examples/`
- `diagnostic_lexicon/prompts/`
- `diagnostic_lexicon/runbooks/`
- `diagnostic_lexicon/schemas/`
- `diagnostic_lexicon/src/diagnostic_lexicon/`
- `diagnostic_lexicon/tests/`

## Pipeline Stages

## Stage 1: Build semantic source manifest

Goal:

- freeze which semantic artifacts are being consumed

Output:

- a manifest recording:
  - full-run base path
  - targeted override path
  - ontology version path
  - merge rules
  - date/time

This makes the lexical build reproducible.

## Stage 2: Merge semantic sources

Goal:

- produce one merged semantic decision space

Merge rule:

- all organs from full-run base
- overwrite `Colon`, `Gallbladder`, `Kidneys`, `Small bowel` with targeted refined run

This merged view is the semantic foundation for lexical mining.

## Stage 3: Build subtype example catalogs

For each organ subtype:

- collect accepted examples
- optionally include accepted-provisional examples with lower trust
- collect contrast examples:
  - same organ, different subtype
  - negative/mixed counterparts

Catalog rows should include:

- organ
- subtype
- family
- raw text
- normalized text
- confidence
- decision_status
- lesion metadata if available

This is the real evidence packet for lexical induction.

## Stage 4: Candidate phrase mining

Before using the LLM, mine candidate phrases directly from the examples.

For each subtype:

- extract uni/bi/tri-grams
- optionally extract dependency-lite noun phrases
- compute support counts
- compute distinctiveness against contrast sets

Output candidate table should include:

- phrase
- support count
- support rate
- distinctiveness score
- subtype
- organ
- example snippets

This keeps the later LLM step grounded in real corpus candidates.

## Stage 5: LLM lexicalization pass

The LLM should operate **per organ subtype** using:

- subtype definition from ontology
- positive real examples
- contrast examples
- candidate phrases mined from the corpus

The LLM returns structured JSON that classifies lexical candidates into:

- `positive_lexicalizations`
- `negative_lexicalizations`
- `uncertain_lexicalizations`
- `confusers`
- `discouraged_lexicalizations`
- `notes`

### Important constraint

The LLM should not invent arbitrary lexical resources from memory alone.

It should be asked to:

- normalize
- deduplicate
- classify
- lightly extend

using real subtype evidence.

## Stage 6: Validation and consolidation

Validate that each LLM entry:

- belongs to the correct organ/subtype
- is structurally valid JSON
- uses allowed output categories
- does not duplicate existing canonical phrases excessively

Consolidate near-duplicates:

- singular/plural variants
- hyphenation variants
- obvious wording duplicates

The goal is a stable lexical registry, not a raw phrase dump.

## Stage 7: Build family lexicons

After subtype lexicons are stable, derive family-level registries.

Example:

- subtype lexicons:
  - `colon_mass`
  - `pancreas_mass`
  - `gallbladder_mass_invasion`

can contribute to family-level lexicalization for:

- `focal_lesion`

This enables coarser lexical regularization if needed.

## Stage 8: Build tokenizer-aware caches

Tokenize each lexical phrase using the decoder tokenizer.

Output:

- phrase text
- token ids
- organ
- subtype / family
- polarity class
- uncertainty class

This is the artifact that will make decoder loss computation efficient.

## Stage 9: Build sample-level lexical targets

This is the crucial bridge into the researcher’s diagnostic-loss idea.

For each tagged training sample `b`, derive:

- `V_b^+`
- `V_b^-`
- optional `V_b^0` / normality-related sets

### Construction logic

#### Positive lexical set `V_b^+`

Union of:

- positive lexicalizations of the sample’s `primary_subtype`
- positive lexicalizations of its `secondary_subtypes`
- optionally family-level lexicalizations of those subtypes

#### Negative lexical set `V_b^-`

Union of:

- lexicalizations of explicitly incompatible sibling states
- lexicalizations contradicted by:
  - `polarity`
  - `normality`
  - `certainty`

Example:

If sample says:

- `primary_subtype = gallbladder_distension`
- `polarity = mixed`
- text explicitly negates cholecystitis

then `V_b^-` should include:

- strong positive cholecystitis lexicalizations

and **not** include all abnormal gallbladder words indiscriminately.

### This is the main value of semantic tagging for lexical loss

The semantic tags tell us which lexical target sets should be active for each sample.

Without the tag layer, `V_b^+` and `V_b^-` are much harder to define well.

## Stage 10: Reporting

Produce reports showing:

- subtype coverage
- family coverage
- unsupported subtype tail
- most common confusers
- phrase overlap across subtypes
- tokenization coverage statistics
- sample-level target set sizes

This is necessary before using the lexical artifacts in training.

## Prompt Design

The LLM prompt should be structured and subtype-centered.

Input should include:

- organ
- subtype
- family
- subtype description from ontology
- positive examples
- contrast examples
- mined candidate phrases
- instructions to classify, not freestyle

Output JSON should contain:

- subtype metadata
- categorized phrase lists
- rationales
- confidence

See:

- [prompts/system_v1.md](/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/diagnostic_lexicon/prompts/system_v1.md:1)
- [prompts/user_v1.md](/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/diagnostic_lexicon/prompts/user_v1.md:1)
- [prompts/output_schema_v1.json](/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/diagnostic_lexicon/prompts/output_schema_v1.json:1)

## Model Strategy

### Main lexicalization pass

Use a strong instruct model, same class as semantic tagging:

- `meta-llama/Llama-3.3-70B-Instruct`

Reason:

- we want careful subtype-sensitive lexical classification

### Later optional cheaper consolidation

Once the lexical ontology is mature, a smaller model may be acceptable for:

- deduplication
- incremental subtype additions

But initial registry building should bias toward quality.

## Runtime Strategy

This job should be much cheaper than semantic tagging because:

- the unit of work is subtype, not 60k raw samples
- subtype count is much smaller
- phrase mining is local and cheap

So this is a good candidate for a dedicated background job once implemented.

## Integration With Diagnostic Loss

The lexical pipeline does **not** replace the semantic tag loss.

Expected use:

- semantic loss = main diagnostic supervision
- lexical loss = smaller wording regularizer

This subproject prepares the lexical side for formulas like the researcher’s:

- positive vocabulary set `V_b^+`
- negative vocabulary set `V_b^-`

But now these sets are constructed from:

- semantic subtype/family labels
- real corpus lexicalizations

instead of from generic pathology words alone.

## Queueing Plan

Later, once the code exists, the intended queueable entrypoint is:

- `apps/run_lexicon_pipeline.py`

with a Slurm wrapper like:

- [examples/run_diagnostic_lexicon_vllm_gh200.sbatch](/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/diagnostic_lexicon/examples/run_diagnostic_lexicon_vllm_gh200.sbatch:1)

The batch structure should mirror semantic tagging:

- allocate GPU node
- launch vLLM locally
- wait for readiness
- run lexical pipeline config
- write outputs locally

## Acceptance Criteria

This pipeline should be considered ready when it can:

1. consume merged semantic artifacts cleanly
2. build subtype-centered lexical registries
3. validate and consolidate phrase sets
4. derive per-sample `V_b^+` / `V_b^-` targets
5. export tokenizer-indexed lexical resources
6. do all this without modifying semantic-tagging artifacts
