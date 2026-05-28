# Semantic Tagging

This is a standalone subproject for building a local, versioned semantic tagging
layer over organ-specific text findings.

It is intentionally structured as a project within the project:
- its own package under `src/semantic_tagging`
- its own CLI entrypoints under `apps/`
- its own ontology, prompts, schemas, tests, and runbooks

The goal is to produce:
- local source tables
- unique organ-text inventories
- structured semantic tags from LLM inference
- online/provisional ontology updates
- row-level propagated tags
- loss-ready artifacts for future decoder diagnostic-loss redesign

The original dataset is never modified.

## Main Outputs

Outputs are written under:

- `../outputs/semantic_tagging/<dataset_id>/<run_id>/`

The main artifacts are:
- `source_rows.parquet`
- `unique_texts.parquet`
- `unique_text_stats.parquet`
- `raw_llm_decisions.jsonl`
- `validated_decisions.parquet`
- `provisional_subtypes.json`
- `row_level_tags.parquet`
- `loss_ready_targets.parquet`
- `reports/summary.md`

## Main Commands

Run from inside `semantic_tagging/`:

```bash
python apps/build_source_rows.py --config configs/merlin_local_v1.yaml
python apps/build_unique_text_inventory.py --config configs/merlin_local_v1.yaml
python apps/run_tagging_pipeline.py --config configs/merlin_local_v1.yaml
python apps/consolidate_ontology.py --config configs/merlin_local_v1.yaml
python apps/materialize_loss_targets.py --config configs/merlin_local_v1.yaml
python apps/inspect_run_outputs.py --output-dir ../outputs/semantic_tagging/merlin_converted/default_run
```

## Environment

This subproject does not create environments automatically.

Use the runbooks:
- [create_vllm_env_gh200.md](runbooks/create_vllm_env_gh200.md)
- [launch_vllm_server_gh200.md](runbooks/launch_vllm_server_gh200.md)
