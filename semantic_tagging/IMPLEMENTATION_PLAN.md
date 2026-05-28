# Semantic Tagging Implementation Plan

## Purpose

Build a self-contained semantic tagging subproject that reads the current
dataset, constructs unique organ-text inventories, uses LLM inference to assign
structured semantic tags, supports online subtype expansion, and materializes
loss-ready supervision artifacts for decoder diagnostic loss.

## Design Principles

- tag-first structured outputs, not single-label grouping
- local outputs only; original dataset remains immutable
- organ-aware ontology grounded in observed dataset distributions
- strict JSON outputs with schema validation and repair
- online subtype proposals are provisional and auditable
- backend-abstract inference with vLLM as the primary target

## Main Deliverables

- standalone code package in `src/semantic_tagging`
- CLI apps under `apps/`
- ontology under `ontology/`
- prompts under `prompts/`
- schema assets under `schemas/`
- manual GH200 runbooks under `runbooks/`
- unit and integration tests under `tests/`

## Output Contract

The final artifacts must support:
- coarse normality supervision
- organ subtype supervision
- certainty/polarity supervision
- confidence-weighted filtering
- contradiction-aware downstream loss design
