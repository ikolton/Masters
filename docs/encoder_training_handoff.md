# Encoder training handoff

This note summarizes the current OrganSegCLIP encoder training state before moving the project to a larger-GPU server. It is intentionally written as a practical handoff: what the current setup does, what problems we observed, what fixes are already in the repo, and what experiments are worth trying next.

## Current training goal

The goal is not only image-text retrieval. The main goal is to learn useful per-organ CT representations for downstream medical tasks, with emphasis on medically meaningful findings. The current encoder combines:

- a SegMamba-style patch encoder and segmentation head,
- patch/grid aggregation over 128x128x48 crops,
- organ queries,
- segmentation-derived organ attention supervision,
- per-organ templated text targets,
- same-organ SigLIP alignment between organ image embeddings and organ text embeddings,
- auxiliary diagnostic / lesion / patch-organ losses.

The current baseline configs are:

- `configs/encoder/train/organsegclip_128x48_20ep_uniformtext_schedcos.yaml`
- `configs/encoder/train/organsegclip_128x48_20ep_uniformtext_schedcos_logitslr1e5.yaml`

Both use 20 epochs, fixed 512-case smoke validation, full validation every 3 epochs, cosine schedule with warmup, bf16 AMP, patch batch size 20, and OOM retry handling.

## What was fixed recently

Recent fixes and robustness upgrades include:

- Added cosine LR scheduling with warmup.
- Increased smoke validation size and kept smoke validation fixed rather than random.
- Split W&B logging so future runs should distinguish batch-level validation metrics from aggregate epoch metrics.
- Added bf16 AMP support.
- Added safer normalization for alignment embeddings using float32 normalization with an epsilon before casting back.
- Added robust checkpoint RNG restore handling.
- Added OOM fallback/retry paths around patch encoding and segmentation supervision.
- Made supervised segmentation patches run first within patch chunks, so expensive supervised work is less likely to happen at the worst memory point.
- Fixed SLURM launch behavior around master port/address in the sbatch helper.

The OOM issue was real: some steps had unusually high patch counts and memory accumulation. The retry/fallback path allowed the run to pass the previous failure region while preserving patch batch size 20 for normal steps.

## Current observed behavior

The model is learning, but not all objectives behave equally.

Segmentation looks healthy. Full-validation dice improved strongly during the current runs and did not show the same early degradation as organ alignment. This suggests that data loading, patch encoding, segmentation backpropagation, and anatomical supervision are broadly working.

Organ alignment initially improves but then starts to overfit or degrade on validation. The training alignment loss and training top-k metrics keep improving, while validation organ alignment loss rises after the best early/mid checkpoint. The run with lower logit-scale/bias LR behaves better than the default run, but still shows the same general direction.

Because segmentation keeps improving, `full_val_total_loss` can hide degradation in the organ alignment objective. For downstream organ representation quality, checkpoint selection should not rely only on total validation loss.

## Recent clean run metrics

The two most relevant clean-from-zero runs were both trained with the current OOM retry / supervised-first / bf16 setup. Metrics below are from local `metrics.json`, not W&B.

### Default logit scale/bias LR

Output directory:
`outputs/encoder/train_joint_2gpu_128x128x48_20ep_siglip_grid_orgattn_sameorgan_a100_fast_noreport_summary332_noactckpt_prefetch4_dropout01_pubmedbert_uniformtext_schedcos_smoke512_fullval3_cleanfrom0`

| epoch | full total | full organ loss | full i2t | full t2i | full dice | full seg loss | organ scale |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.8598 | 0.5937 | 0.6189 | 0.6363 | 0.8354 | 0.1016 | 12.0914 |
| 6 | 0.8051 | 0.5710 | 0.6438 | 0.6645 | 0.8632 | 0.0824 | 12.6553 |
| 9 | 0.8184 | 0.5892 | 0.6385 | 0.6635 | 0.8739 | 0.0769 | 13.0246 |
| 12 | 0.8920 | 0.6617 | 0.6293 | 0.6519 | 0.8829 | 0.0711 | 13.5443 |
| 15 | 1.0700 | 0.8206 | 0.6227 | 0.6431 | 0.8853 | 0.0682 | 14.1130 |
| 18 | 1.2778 | 1.0044 | 0.6209 | 0.6352 | 0.8887 | 0.0662 | 14.4666 |

Best full-validation organ alignment loss and total loss were both at epoch 6. Segmentation dice kept improving until at least epoch 18. At epoch 19, train organ loss was 0.2314 with train i2t/t2i 0.8682/0.8828, showing a large train/validation alignment gap.

### Logit scale/bias LR 1e-5

Output directory:
`outputs/encoder/train_joint_2gpu_128x128x48_20ep_siglip_grid_orgattn_sameorgan_a100_fast_noreport_summary332_noactckpt_prefetch4_dropout01_pubmedbert_uniformtext_schedcos_smoke512_fullval3_logitslr1e5_cleanfrom0`

| epoch | full total | full organ loss | full i2t | full t2i | full dice | full seg loss | organ scale |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.8823 | 0.6138 | 0.6122 | 0.6185 | 0.8306 | 0.1053 | 10.7721 |
| 6 | 0.8142 | 0.5800 | 0.6361 | 0.6538 | 0.8612 | 0.0836 | 11.2669 |
| 9 | 0.8213 | 0.5915 | 0.6378 | 0.6589 | 0.8733 | 0.0774 | 11.5877 |
| 12 | 0.8835 | 0.6490 | 0.6211 | 0.6479 | 0.8831 | 0.0713 | 11.8778 |
| 15 | 1.0328 | 0.7762 | 0.6157 | 0.6405 | 0.8856 | 0.0681 | 12.1596 |

Best full-validation organ alignment loss and total loss were again at epoch 6, with the best i2t/t2i around epochs 6-9. This run slowed organ scale growth and was less extreme than the default-scale run, but it still showed the same alignment-overfit pattern. At epoch 17, train organ loss was 0.2933 with train i2t/t2i 0.8428/0.8594.

Overall interpretation: both clean runs confirm that the encoder/segmentation path learns robustly, while the organ-text alignment objective peaks early and then becomes increasingly train-specific. The logit-scale LR reduction helps, but does not solve the underlying alignment/generalization issue.

## Metrics caveat

Older W&B validation metrics were misleading because batch-level validation values were logged under names that looked like full validation aggregates. Future runs should use the fixed logging paths, but for existing runs the trusted source is `metrics.json` in the output directory.

The current top-1 retrieval-style metrics are also batch-local averages, not a global full-validation retrieval matrix. They are useful as trends but should not be interpreted as absolute dataset-level retrieval accuracy.

For future runs, prefer:

- aggregate epoch metrics from `metrics.json`,
- full validation organ alignment loss,
- validation organ image-to-text / text-to-image trends only as approximate signals,
- an offline global validation retrieval or nearest-neighbor evaluation,
- downstream probes using both pre-projection and post-projection organ vectors.

## Main code-level concerns

No obvious catastrophic training bug was found. The following pieces look correct:

- training/eval mode handling,
- `torch.no_grad` during validation,
- DDP all-gather for SigLIP embeddings,
- masking of missing organ text,
- exact duplicate same-organ texts treated as positives,
- cross-organ negatives disabled as intended,
- scheduler stepping and resume behavior,
- segmentation/attention gradients flowing through the intended model path.

The suspicious parts are mostly objective/design issues rather than broken backpropagation.

### 1. Text backbone is frozen, but the alignment text space still moves

The PubMedBERT backbone is frozen, but there are trainable projection layers after it. In practice, this means the raw text encoder is frozen, but the text representation used by SigLIP is not fully fixed. The image side, text projection, organ text projection, and logit scale can co-adapt to the dataset.

This is a likely contributor to early alignment overfit.

### 2. Same-organ SigLIP has a small effective candidate pool

The global batch is small because the 3D model is expensive. Since cross-organ negatives are intentionally disabled, each organ query only sees a few same-organ image/text candidates per step. SigLIP is more batch-efficient than CLIP, but it still benefits from useful negative diversity.

This makes the contrastive task noisy and potentially instance-discriminative.

### 3. Exact text identity may create false negatives

Per-organ text is much cleaner than full reports, which is good. However, the current positives are based on exact normalized text identity. Two different templated texts that describe similar or compatible findings may be treated as negatives for the same organ.

This can push clinically related organ states apart.

### 4. Alignment may be too sharp too early

The run with lower LR for organ logit scale/bias looked healthier. The normal run's scale increased faster and overfit more strongly. This suggests that overconfidence from the SigLIP temperature/bias dynamics matters.

### 5. The final organ representation and alignment head should be evaluated separately

The downstream organ representation may be useful even if the post-projection SigLIP embedding overfits. Future evaluation should compare:

- pre-projection organ features,
- post-projection aligned embeddings,
- diagnostic/linear-probe performance from both.

## Reference repos

Two nearby references were inspected:

- `../Merlin`: CLIP-style whole-CT/report model with Clinical Longformer text encoder and whole-image/report contrastive embeddings.
- `../SPECTRE`: SigLIP-style CT pretraining with large per-GPU batch, long warmup, early image-backbone freezing, layer-wise LR decay, projection-head stabilization, and gradient logging.

Important differences from SPECTRE:

- SPECTRE uses much larger batches (`batch_size_per_gpu: 128` in its default config).
- SPECTRE warms up for many epochs and freezes the image backbone early.
- SPECTRE uses deeper projection heads with weight norm and early last-layer gradient cancellation.
- SPECTRE aligns whole scans/reports rather than 11 organ-specific query/text pairs.

The lesson is not simply "use bigger batches". For this project, the more useful lessons are staged training, projection-head stabilization, logit-scale control, and better semantic grouping of positives.

## Recommended next experiments on bigger GPUs

Use bigger GPUs first for stability and experiment speed, not necessarily for a huge batch jump. Keep changes interpretable.

### Experiment A: stabilized alignment

Keep the current architecture but reduce alignment over-adaptation:

- keep PubMedBERT frozen,
- freeze or strongly slow the text projection and organ text projection,
- keep lower LR for organ logit scale/bias,
- consider stronger weight decay on projection heads,
- select checkpoints by organ validation metrics, not total loss.

This tests whether moving projection/logit dynamics are a main cause of validation degradation.

### Experiment B: anatomy-first staged training

Train anatomy/attention before strong text alignment:

- early phase: segmentation + organ attention + diagnostic/auxiliary losses,
- organ alignment off or very low,
- later phase: gradually turn on organ alignment,
- optionally freeze/slow the patch encoder when alignment starts.

This matches the downstream goal: first learn anatomically grounded organ features, then align them to text.

### Experiment C: semantic-positive alignment

Investigate whether exact text matching is too strict:

- count unique normalized organ texts per organ,
- inspect near-duplicate templates,
- group clinically equivalent or compatible labels,
- allow multiple positives per organ beyond exact text identity,
- optionally use text prototypes or coarse finding classes.

This is likely important if the model is learning exact sentence discrimination instead of finding semantics.

### Experiment D: larger effective alignment set without huge image batches

If bigger GPUs help but are still limited:

- accumulate image/text embeddings for alignment across microbatches,
- use text-side prototypes/templates as extra candidates,
- consider a cautious memory queue for negatives,
- avoid increasing 3D image batch until the staged/projection fixes are tested.

## Suggested evaluation additions

Before treating the next runs as final, add:

- offline global validation retrieval for organ embeddings,
- per-organ validation metrics rather than only aggregate organ metrics,
- gradient norm logging per major module and per loss family,
- separate logging for pre-projection and post-projection feature quality,
- downstream linear probes or small validation tasks if available,
- text-label entropy and near-duplicate statistics per organ.

## Migration checklist

When moving to the new server:

1. Copy the repo plus required metadata files.
2. Copy or regenerate the converted dataset path expected by `ORGAN_SEG_CLIP_DATASET_ROOT`.
3. Copy `Merlin_metadata_hf_clean.csv` or update `data.lesion_metadata_csv`.
4. Recreate the conda environment or export it from the current server.
5. Update config paths if the dataset/output roots change.
6. Run a short 2-GPU smoke test with bf16 and OOM retry enabled.
7. Run one full validation early to verify metrics/logging.
8. Start clean runs from epoch 0 rather than resuming the old A100 runs unless debugging continuity specifically.

## Cleanup notes

The repo still contains many old pilot/profile/benchmark configs. They are useful experiment history, but they make the config folder noisy. For the migration commit, the cleanest approach is:

- keep current production configs directly under `configs/encoder/train/`,
- keep historical tracked configs under `configs/encoder/train/legacy/`,
- keep temporary `debug_nan/` and `debug_profile/` configs ignored,
- do a later separate cleanup commit if we want to delete old legacy configs entirely.

Avoid committing generated SLURM configs, local logs, W&B files, or outputs.
