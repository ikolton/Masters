# OrganSegCLIP — Project Tasks

## Legend
- `[x]` Done
- `[-]` In progress / partially done
- `[ ]` Planned / pending
- `[A]` Ablation config prepared (not yet run)

---

## Phase 1 — Encoder Bug Fixes (Completed)

- `[x]` Skip alignment computation in stage1 when `organ_alignment_weight=0` (composer.py)
- `[x]` Fix `_build_organ_finding_counts` normalization — use shared `_normalize_finding_label` (engine.py)
- `[x]` Fix `_directional_siglip_loss` weighting inconsistency — uniform 0.5× for pos and neg (siglip.py)
- `[x]` Fix `text_positive_mask = image_positive_mask.clone()` (was reference alias) (siglip.py)
- `[x]` Add silent exclusion warnings in `_build_usable_studies` (contracts.py)
- `[x]` Fix decoder generation `.split("###")[0]` truncation removed (decoder/model.py)
- `[x]` Fix `drop_last=is_distributed()` for DDP consistency (engine.py)
- `[x]` Add `max_grad_norm: 1.0` gradient clipping in stage2 yaml

---

## Phase 2 — Improvements (Completed)

### Segmentation Metrics
- `[x]` Dual dice metric: `segmentation_dice` (all patches, legacy) + `segmentation_foreground_dice` (patches with GT foreground only)
  - Added `segmentation_foreground_dice` and `segmentation_foreground_patch_count` to `OrganSegOutput`
  - `_segmentation_supervision_with_oom_fallback` returns 6-tuple with foreground dice

### Vectorization
- `[x]` O(N²) → vectorized `_build_positive_mask`, `_build_positive_mask_against_global` (contrastive.py)
- `[x]` O(N²) → vectorized `_build_same_organ_mask`, `_build_id_positive_mask` (siglip.py)

### Config & Init
- `[x]` `organ_logit_scale_init` + `organ_logit_bias_init` added to `OrgansConfig` (schemas.py)
- `[x]` Model logit parameters now initialized from config (aggregation/model.py)
- `[A]` Ablation: neutral logit init (`scale=1.0, bias=0.0`) — `ablation_logit_init_neutral.yaml`

### Contrastive Learning
- `[x]` `siglip_soft_positive_threshold` added to `LossConfig` (schemas.py)
- `[x]` Soft positive mask expansion in `masked_organ_siglip_loss` (siglip.py)
- `[A]` Ablation: soft positive threshold=0.85 — `ablation_soft_positive.yaml`

### Training Control
- `[x]` `early_stopping_patience` added to `TrainingConfig` (schemas.py)
- `[x]` Early stopping logic with DDP broadcast in engine.py
- `[x]` Warm-start + `organ_alignment_weight=0` warning (engine.py)

### Decoder Efficiency
- `[x]` `BinaryDiagnosticLoss` vectorized with logsumexp trick — avoids full vocab softmax (decoder/losses.py)
- `[x]` `visual_projector_depth` config added to `DecoderModelConfig` (schemas.py)
- `[x]` Decoder projector depth-aware construction (decoder/model.py)
- `[A]` Ablation: projector depth=2 — decoder `ablation_projdepth2.yaml`

### Checkpoint & Validation
- `[x]` Stage2 `best_checkpoint_metric` changed to `full_val_organ_image_to_text_top1`
- `[x]` Stage2 `validation_every_epochs` changed to 2

### Data Pipeline
- `[x]` Duodenum removed from `CSV_TO_ORGAN_NAME` mapping (lesion_metadata.py)

---

## Phase 3 — Pending / Future Work

### High Impact
- `[x]` Upfront text embedding cache (on-disk persistent cache at `/net/scratch/hscra/plgrid/plgikolton/text_embedding_cache/`)
  - `disk_cache_path` field in `TextEncoderConfig`; `HFTextEncoder` loads at init, rank-0 saves after each epoch
  - All encoder train yamls updated with the cache path
  - Must NOT write to shared team dataset folder ✓
- `[A]` Hard negative mining + soft positive combined ablation — `ablation_hard_neg_soft_pos.yaml`
  - `siglip_hard_negative_weight: 2.0` + `siglip_soft_positive_threshold: 0.85`
  - Same threshold T partitions same-organ pairs: sim ≥ T → soft positive, sim < T → hard negative (2× loss weight)
- `[ ]` Contrastive batch expansion — clarify whether current architecture already benefits from cross-study organ contrast
  - Conclusion: already cross-study (132 embeddings per direction). MoCo/queue only needed if batch is limiting.

### Moderate
- `[ ]` No organ prepend — code already uses raw finding text only; verify no `organ_text_template` remnant is active in the current text encoder call path
- `[ ]` Organ query correspondence to anatomy — explore linking organ query slot i to anatomy mask i via soft supervision
- `[ ]` Text encoder projection with cosine similarity — if projection is trained, contrastive loss uses its own representation; ablate with frozen projection

### Low / Housekeeping
- `[x]` Test suite fixes: `row_probs` → `row_logits` in decoder/losses.py; duodenum test inverted; visual encoder batch test indices fixed
- `[x]` Smoke-val every epoch / full-val every 2 epochs — already implemented via `validation_every_epochs: 2` + existing smoke-val path in engine
- `[ ]` `organ_alignment_weight` schema-level validation assert (currently only a runtime warning in engine.py)
- `[ ]` `AlternateMerlinDataset` support for running on local dataset copies without symlinks
- `[ ]` Evaluation script for final checkpoint → per-organ top-1 alignment table
- `[ ]` Stage2 from long-anatomy init (stage1 long → stage2 alignment) run

---

## Active Runs (update as needed)

| Run | Config | SLURM job | Status | Notes |
|-----|--------|-----------|--------|-------|
| stage1_short | `organsegclip_128x48_stage1_short_anatomy_gh200_2gpu_bs12.yaml` | — | done | produces `best.pt` for stage2 warm-start |
| stage2_align | `organsegclip_128x48_stage2_align_from_short_gh200_2gpu_bs12.yaml` | 17110559 | running (ep 13/20) | best top-1=0.521 at epoch 3; overfitting after epoch 3 |
| ablation_hard_neg_soft_pos_scratch | `...ablation_hard_neg_soft_pos_scratch.yaml` | 17164992 | running | hard-neg(2×)+soft-pos(0.85), lr=1e-4, from scratch |
| normaltplv2_softpos_hardneg | `...normaltplv2_softpos_hardneg.yaml` | 17164993 | running | normaltplv2 dataset + hard-neg(2×)+soft-pos(0.85), lr=1e-4, from scratch |
| labelnormal_softpos_hardneg | `...labelnormal_softpos_hardneg.yaml` | 17166940 | pending | label_based_normal_v1 dataset + hard-neg(2×)+soft-pos(0.85), lr=1e-4, from scratch |

---

## Discussion Notes

### Contrastive Batch Expansion
The model already contrasts all (study, organ) pairs within the global DDP batch. With 2 GPUs × 6 studies × 11 organs = 132 embeddings per direction, the effective batch is already cross-study. "Batch expansion" via MoCo/queue would only help if the effective batch is too small — monitor `organ_positive_logit_mean` vs `organ_negative_logit_mean` during stage2 to assess whether the batch is limiting.

### Hard Negative Mining vs Soft Positive
These address opposite ends of the same problem (label noise). Soft positives reduce false negatives (semantically similar texts punished as negatives). Hard negatives reduce false positives (trivially easy negatives not providing gradient). They can be combined: use cosine similarity threshold T to split same-organ pairs into soft positives (sim ≥ T) and hard negatives (sim < T). One ablation config could cover both.

### No Organ Prepend
The `organ_text_template` is `"{organ}: {finding}"` in the config but `organ_raw_texts` is used for contrastive loss (not `organ_texts`). The distinction is already handled — `organ_raw_texts` contains the finding only. The `organ_text_template` is only used internally by the text encoder when building `organ_texts`. Since `organ_raw_texts` bypasses the template, the current setup effectively does not prepend the organ name for contrastive learning. **No code change needed unless the template is also applied to raw texts.**

### Organ Query Correspondence
Each organ query slot is initialized randomly and has no forced correspondence to a specific organ. The anatomy supervision (organ presence classification via `patch_organ_presence_head`) provides indirect supervision, but the correspondence is emergent. A harder constraint (e.g., align organ query i to its segmentation mask centroid) might improve organ localization but requires architectural changes to the aggregator.

### Text Encoder Projection Cosine Similarity
The SigLIP loss uses L2-normalized embeddings for logit computation: `scale * img @ text.T + bias`. The projection is a linear layer, so cosine similarity in projected space approximates cosine similarity in PubMedBERT space (up to rotation). Training the projection jointly with alignment means the loss gradient flows into the projection — this is the expected and correct behaviour.
