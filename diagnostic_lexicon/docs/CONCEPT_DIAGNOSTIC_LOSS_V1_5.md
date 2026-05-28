# Concept Diagnostic Loss v1.5

This document defines the matched diagnostic-logit loss used for cross-model
ablations. A run may use this name only when the loss is applied directly to
decoder vocabulary logits. Hidden-state lexical classifiers are a separate
auxiliary-supervision intervention.

## Objective

The loss keeps the original diagnostic-loss skeleton:

```text
decoder logits -> vocabulary probabilities -> diagnostic target mass -> CE + lambda * diagnostic loss
```

The v1.5 change is that targets remain concept-indexed. Each sample has
positive and negative concepts, and each concept has its own token set and
weight. Synonyms and mined phrases cooperate inside a concept; different
clinical concepts are averaged separately.

For sample `b`, concept `c`, and decoder position `t`:

```text
p[b,c,t] = sum_{v in V[b,c]} softmax(logits[b,t])[v]
```

Positive concepts use occurrence probability:

```text
q[b,c] = 1 - product_t (1 - p[b,c,t])
L_pos[b,c] = -log(q[b,c] + eps)
```

Negative concepts use a smooth maximum over positions:

```text
L_neg[b,c] = tau^-1 * logsumexp_t(tau * log(p[b,c,t] + eps))
```

The sample diagnostic term is:

```text
L_diag[b] =
  sample_weight[b] * (
    weighted_mean_{c in C_pos[b]} L_pos[b,c]
    + alpha * weighted_mean_{c in C_neg[b]} L_neg[b,c]
  )
```

The training objective is:

```text
L_total = L_CE + lambda_diag * mean_contributing_samples(L_diag)
```

## Target Cache

The concept cache preserves the diagnostic lexicon structure instead of
flattening all phrases into one positive and one negative token bag. Each row
contains:

```text
key: (organ, normalized_text)
positive_concepts: [{source_label, label_type, weight, token_ids, phrases}]
negative_concepts: [{source_label, label_type, weight, token_ids, phrases}]
sample_weight
review_required
```

Target hygiene belongs in cache preparation, not in the loss:

```text
remove within-sample positive/negative token overlap from negative concepts
downweight mixed-row normal_wording negatives
drop concepts whose token set becomes empty
```

## Comparability Rules

Use `concept_specific_lexical` for this loss. Keep `sample_specific_lexical`
for historical flat lexical-loss baselines.

Decoder table comparisons must match the existing 10-epoch decoder setup.
Merlin table comparisons must match the existing 2-epoch Merlin setup.
