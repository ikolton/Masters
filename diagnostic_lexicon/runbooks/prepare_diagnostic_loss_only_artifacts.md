# Prepare Diagnostic-Loss-Only Artifacts

This runbook builds lexical diagnostic-loss artifacts from the clean semantic consolidation output.

It is for the ablation:

```text
L_total = L_CE + beta * L_lexical_diag
semantic_loss.enabled = false
```

The semantic artifact is used offline to create better `V_b+` and `V_b-` targets, but no semantic auxiliary heads are required during this experiment.

## Inputs

Best current semantic source:

```text
/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/outputs/semantic_tagging/merlin_converted/consolidation/consolidation_v3/postprocess_v3_clean/semantic_training_targets_v3.jsonl
```

Vocabulary source:

```text
/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/outputs/semantic_tagging/merlin_converted/consolidation/consolidation_v3/postprocess_v3_clean/training_vocab_v3_clean.json
```

## Build Command

Use the working Python 3.11 env:

```bash
/net/scratch/hscra/plgrid/plgikolton/conda-envs/codex-masters-py311/bin/python3.11 \
  /net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/diagnostic_lexicon/apps/build_diagnostic_loss_artifacts.py \
  --tokenizer-name Qwen/Qwen2.5-0.5B
```

This writes to:

```text
/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/outputs/diagnostic_lexicon/merlin_converted/lexical_diag_v1_from_semantic_v3
```

## Outputs

Expected files:

```text
lexicon_registry_v1.json
sample_level_lexical_targets_v1.jsonl
tokenized_lexical_targets_v1.pt
manifest.json
reports/coverage.md
```

## Current Build Summary

The first generated build produced:

- semantic target rows: `60851`
- usable semantic rows: `37742`
- lexical target rows: `37742`
- registry entries: `407`
- subtype registry entries: `225`
- family registry entries: `182`
- tokenizer cache rows: `37742`
- tokenizer: `Qwen/Qwen2.5-0.5B`

The default build excludes `review_required=true` rows.

## How It Should Be Used

For the diagnostic-loss-only experiment, point the decoder diagnostic-loss code at:

```text
sample_level_lexical_targets_v1.jsonl
tokenized_lexical_targets_v1.pt
```

and keep:

```yaml
semantic_loss:
  enabled: false
```

The old binary diagnostic loss should be kept as a separate baseline, not mixed into this ablation unless explicitly testing a combined objective.

