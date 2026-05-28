# Reproduce Semantic Tagging And Consolidation Artifacts

This runbook explains how to reproduce the current local semantic-tagging artifacts from source dataset inputs through the draft diagnostic-loss vocabularies.

The goal is reproducibility, not just rerunning a notebook. Every stage is config-driven and writes versioned artifacts under `outputs/semantic_tagging/<dataset_id>/`.

## Project Root

All commands below assume:

```bash
cd /net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/semantic_tagging
```

The subproject is intentionally self-contained under:

```text
/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/semantic_tagging
```

## Environment

The environment is created manually inside a GH200 interactive job, not by the codebase.

Environment path used for the current artifacts:

```text
/net/scratch/hscra/plgrid/plgikolton/Magisterka/.venv-semantic-tagging-vllm
```

Setup documentation:

- `runbooks/create_vllm_env_gh200.md`
- `runbooks/launch_vllm_server_gh200.md`
- `runbooks/run_semantic_tagging_vllm_gh200_sbatch.md`

Minimum runtime assumptions:

- Python `3.11`
- `ML-bundle/24.06a`
- vLLM `0.10.2+cu124`
- `transformers<5`
- model access for `meta-llama/Llama-3.3-70B-Instruct`

## Dataset Inputs

Current dataset id:

```text
merlin_converted
```

Current source inputs:

```text
dataset_root: /net/storage/pr3/plgrid/plggjmiag/Merlin_converted
lesion_csv: /net/scratch/hscra/plgrid/plgikolton/Magisterka/Merlin_metadata_hf_clean.csv
```

These are referenced by the run configs, not hardcoded inside the pipeline code.

## Output Root

All outputs are local and non-destructive:

```text
/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/outputs/semantic_tagging
```

For Merlin:

```text
/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/outputs/semantic_tagging/merlin_converted
```

## Reproduction Stages

### Stage 1: Full Semantic Tagging Run

Config:

```text
configs/merlin_vllm_full.yaml
```

Output:

```text
outputs/semantic_tagging/merlin_converted/vllm_full
```

Command:

```bash
sbatch examples/run_semantic_tagging_vllm_gh200.sbatch configs/merlin_vllm_full.yaml
```

Important artifacts:

```text
source_rows.parquet
unique_texts.parquet
unique_text_stats.parquet
validated_decisions.parquet
validated_decisions.partial.jsonl
row_level_tags.parquet
loss_ready_targets.parquet
provisional_subtypes.json
proposed_families.json
reports/summary.md
```

This is the full dataset semantic layer.

### Stage 2: Targeted Hard-Organ Refinement Runs

These runs improve organs where the full-run ontology was too coarse or produced many provisional labels.

#### V2 Targeted Candidate

Config:

```text
configs/merlin_vllm_targeted_v2_candidate.yaml
```

Output:

```text
outputs/semantic_tagging/merlin_converted/vllm_targeted_v2_candidate
```

Command:

```bash
sbatch examples/run_semantic_tagging_vllm_gh200.sbatch configs/merlin_vllm_targeted_v2_candidate.yaml
```

#### V3 Targeted Candidate

Config:

```text
configs/merlin_vllm_targeted_v3_candidate.yaml
```

Output:

```text
outputs/semantic_tagging/merlin_converted/vllm_targeted_v3_candidate
```

Command:

```bash
sbatch examples/run_semantic_tagging_vllm_gh200.sbatch configs/merlin_vllm_targeted_v3_candidate.yaml
```

#### V4 Targeted Candidate

Config:

```text
configs/merlin_vllm_targeted_v4_candidate.yaml
```

Output:

```text
outputs/semantic_tagging/merlin_converted/vllm_targeted_v4_candidate
```

Command:

```bash
sbatch examples/run_semantic_tagging_vllm_gh200.sbatch configs/merlin_vllm_targeted_v4_candidate.yaml
```

### Stage 3: Compose Best-Current Semantic Layer For Consolidation

The consolidation configs define which run is used as the base and which organ-specific overrides replace it.

Current composition policy:

- base: `vllm_full`
- `Colon`: `vllm_targeted_v3_candidate`
- `Gallbladder`: `vllm_targeted_v4_candidate`
- `Kidneys`: `vllm_targeted_v3_candidate`
- `Small bowel`: `vllm_targeted_v3_candidate`

This policy is encoded in:

```text
configs/merlin_consolidation_v1.yaml
configs/merlin_consolidation_v2.yaml
configs/merlin_consolidation_v3.yaml
```

The deterministic builder writes a composed decision file and observed tag statistics before any consolidation LLM call.

Manual deterministic build command:

```bash
python apps/build_consolidation_artifacts.py --config configs/merlin_consolidation_v3.yaml
```

In normal usage, the LLM consolidation job runs this build automatically before calling the model.

## Consolidation Versions

### Consolidation V1

Config:

```text
configs/merlin_consolidation_v1.yaml
```

Output:

```text
outputs/semantic_tagging/merlin_converted/consolidation/consolidation_v1
```

Command:

```bash
sbatch examples/run_tag_consolidation_vllm_gh200.sbatch configs/merlin_consolidation_v1.yaml
```

Purpose:

- first vocabulary-level LLM consolidation
- exposed that free-form `training_family` produced vague organ placeholders

Analysis:

```text
docs/CONSOLIDATION_V1_ANALYSIS.md
```

### Consolidation V2

Config:

```text
configs/merlin_consolidation_v2.yaml
```

Output:

```text
outputs/semantic_tagging/merlin_converted/consolidation/consolidation_v2
```

Command:

```bash
sbatch examples/run_tag_consolidation_vllm_gh200.sbatch configs/merlin_consolidation_v2.yaml
```

Purpose:

- introduced controlled family enum
- made organ-name families invalid
- exposed that subtype labels and family labels were still being mixed

Analysis:

```text
docs/CONSOLIDATION_V2_ANALYSIS.md
```

### Consolidation V3

Config:

```text
configs/merlin_consolidation_v3.yaml
```

Output:

```text
outputs/semantic_tagging/merlin_converted/consolidation/consolidation_v3
```

Command:

```bash
sbatch examples/run_tag_consolidation_vllm_gh200_2gpu.sbatch configs/merlin_consolidation_v3.yaml
```

Purpose:

- separates subtype and family supervision
- prevents controlled families from becoming subtype labels
- supports subtype-only, family-only, subtype-and-family, and excluded decisions
- adds `trauma_or_injury` to controlled families

Analysis:

```text
docs/CONSOLIDATION_V3_ANALYSIS.md
```

Current status:

- best consolidation design so far
- not final training artifact yet
- needs deterministic postprocessing and canonicalization

## Deterministic Postprocessing

After `consolidation_v3`, run deterministic postprocessing to turn the LLM draft into cleaner training artifacts.

This stage does not call an LLM.
It applies fixed rules:

- canonicalize subtype labels to stable organ-prefixed labels
- keep subtype labels and coarse family labels separate
- repair obvious invalid V3 rows into conservative family-only targets
- add deterministic review flags
- write clean vocab files
- materialize semantic training targets

Config:

```text
configs/merlin_consolidation_postprocess_v3.yaml
```

Output:

```text
outputs/semantic_tagging/merlin_converted/consolidation/consolidation_v3/postprocess_v3_clean
```

Command:

```bash
sbatch examples/run_consolidation_postprocess_gh200.sbatch configs/merlin_consolidation_postprocess_v3.yaml
```

Why this runs through Slurm:

- it is CPU-style deterministic work
- but the working Python/pyarrow environment is the ARM GH200 venv
- therefore a tiny GH200 allocation is used for compatibility, not for GPU compute

Expected artifacts:

```text
tag_consolidation_map_v3_clean.jsonl
training_vocab_v3_clean.json
training_vocab_v3_clean.yaml
review_queue_v3.csv
semantic_training_targets_v3.jsonl
semantic_training_targets_v3.parquet
manifest.json
reports/deterministic_postprocessing_v3.md
```

The intended training-consumption artifact is:

```text
semantic_training_targets_v3.parquet
```

The intended vocabulary contract is:

```text
training_vocab_v3_clean.yaml
```

## V3 Expected Artifacts

After V3 finishes, expect:

```text
composed_validated_decisions.jsonl
observed_tag_stats.jsonl
observed_tag_stats.csv
llm_consolidation_items.jsonl
llm_consolidation_raw.jsonl
llm_consolidation_decisions.jsonl
llm_consolidation_summary.json
training_vocab_draft.json
manifest.json
reports/consolidation_input_report.md
reports/llm_consolidation_report.md
```

The `manifest.json` stores source file hashes and config provenance.

## Inspecting Outputs

Semantic-tagging run summary:

```bash
python apps/inspect_run_outputs.py \
  --output-dir /net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/outputs/semantic_tagging/merlin_converted/vllm_full
```

Consolidation summary:

```bash
cat /net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/outputs/semantic_tagging/merlin_converted/consolidation/consolidation_v3/llm_consolidation_summary.json
```

Consolidation report:

```bash
less /net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/outputs/semantic_tagging/merlin_converted/consolidation/consolidation_v3/reports/llm_consolidation_report.md
```

## Reusing The Setup For Another Dataset

For a new dataset, do not edit pipeline code.

Create new configs with:

```yaml
project:
  dataset_id: your_dataset_id
  run_id: your_run_id

paths:
  dataset_root: /path/to/dataset
  lesion_csv: /path/to/metadata.csv
  output_root: /path/to/local/outputs
  ontology_root: /path/to/ontology_version
  prompt_root: /path/to/prompts
  schema_root: /path/to/schemas
```

Then create a matching consolidation config:

```yaml
project:
  dataset_id: your_dataset_id
  consolidation_id: consolidation_v1

paths:
  output_root: /path/to/local/outputs

source_runs:
  base:
    run_id: your_full_run
    decision_file: validated_decisions.partial.jsonl
  organ_overrides: {}
```

The reusable parts are:

- source-row builder
- unique-text inventory builder
- ontology/prompt system
- backend abstraction
- deterministic consolidation artifact builder
- LLM consolidation runner
- sbatch launchers

Dataset-specific parts are:

- dataset adapter assumptions
- organ list
- lesion metadata columns
- ontology seed quality
- selected source runs for consolidation

## Current Best Practice

For the current Merlin dataset:

1. Use `vllm_full` as the base semantic layer.
2. Override hard organs with:
   - `Colon`: `vllm_targeted_v3_candidate`
   - `Gallbladder`: `vllm_targeted_v4_candidate`
   - `Kidneys`: `vllm_targeted_v3_candidate`
   - `Small bowel`: `vllm_targeted_v3_candidate`
3. Use `consolidation_v3` as the best draft consolidation layer.
4. Do not use `training_vocab_draft.json` directly for model training.
5. Use deterministic postprocessing to produce:
   - `training_vocab_v3_clean.yaml`
   - `semantic_training_targets_v3.parquet`

## Reproducibility Checklist

Before claiming an artifact is reproducible, record:

- config path
- config hash
- source decision file hashes
- ontology version path
- prompt version path
- model name
- vLLM version
- Slurm job id
- output directory
- summary/report path

The consolidation builder already stores config and source hashes in:

```text
outputs/semantic_tagging/<dataset_id>/consolidation/<consolidation_id>/manifest.json
```

## Important Warning

Do not overwrite old run ids.

Use new `run_id` / `consolidation_id` values for every material change:

- prompt change
- ontology change
- schema change
- source-run composition change
- model/backend change
- deterministic postprocessing change

This is how we preserve history and make the decisions auditable.
