# Implementation Plan

## Goal

Test whether the lexical and semantic diagnostic-loss ideas improve a different
report-generation model family, while preserving Merlin's original model
components as much as possible.

## Scientific Constraint

This ablation should change the loss, not quietly change the model. The harness:

- imports Merlin from a configured repo path;
- uses Merlin's `DataLoader`, `ImageTransforms`, image encoder, adapter,
  tokenizer, and LoRA decoder;
- does not copy Merlin weights or source into this project;
- avoids editing the external Merlin repository;
- records config and provenance into each output directory.

## Minimal Delta From Merlin

Merlin's released report-generation `forward()` returns only CE loss and wraps
the image-to-decoder path in `torch.no_grad()`. For training ablations we need:

- gradients through the adapter and decoder LoRA parameters;
- pooled decoder hidden states for auxiliary diagnostic heads;
- switchable CE-only, lexical, semantic-normality, semantic-family, and
  semantic-subtype losses.

Therefore the local wrapper calls the same Merlin submodules directly but owns
the teacher-forcing forward pass and auxiliary heads.

## Initial Runs

Smoke:

- `smoke_ce_only`
- `smoke_lexical`
- `smoke_sem_family`
- `smoke_lex_sem_family`

Future full ablations:

- `merlin_ce_only`
- `merlin_lexical_w002`
- `merlin_sem_family_w005`
- `merlin_sem_family_w002`
- `merlin_lexical_w002_sem_family_w002`
- `merlin_lexical_w002_sem_normality_w002`

## Acceptance Criteria

- Config loads and validates.
- Dataset builds organ-level records from Merlin-converted data.
- Merlin imports from configured repo path.
- CE-only forward/backward works.
- Lexical and semantic heads can be enabled independently.
- Output manifest records paths, loss config, dataset summary, and train metrics.
- Smoke jobs can be queued without editing the external Merlin repo.

