# Training And Evaluation Audit, 2026-05-27

This audit compares the Merlin ablation runs under:

- `outputs/models_ablations/merlin/benchmark_test_full_real5_basic_bs64`

against the previous decoder benchmark under:

- `outputs/decoder/benchmark_test_full_basic`

## Executive Judgment

The previous decoder benchmark is usable as a test-set comparison with caveats.

The Merlin ablation benchmark is not usable for final conclusions yet, because generation was capped at `max_new_tokens=128` while the Merlin prompt asks for a full radiology report. This produced severely unfinished generations:

- Merlin generated mean length: about `72` words.
- Merlin target mean length: about `13` words.
- Merlin end-punctuation rate: about `6%`.
- Merlin unclosed-generation rate: about `94%`.

Training itself appears to have run cleanly, but the Merlin evaluation is compromised by generation truncation and prompt mismatch.

## Dataset And Split Integrity

The `combined.json` files under `/net/storage/pr3/plgrid/plggjmiag/Merlin_converted/{train,val}` are identical by content:

- train combined records: `25,489`
- val combined records: `25,489`
- SHA1 for both: `6e82c13e88f5adf34310efd90e7888192bd639fa`

This is not automatically a fatal issue because both the previous decoder and Merlin loaders filter records by split-specific study directories.

Actual study directories are disjoint:

- train study dirs: `15,310`
- val study dirs: `5,056`
- train/val directory overlap: `0`
- test study dirs: `5,125`
- test overlap with train dirs: `0`
- test overlap with val dirs: `0`

Merlin records are therefore effectively:

- train organ records: `168,399`
- val organ records: `55,605`
- test organ records: `56,375`

The previous decoder test wrapper also selects the same `5,125` held-out test studies, despite the loader split name being `val`.

## Training Audit

### Previous Decoder Runs

Setup:

- Model: Qwen2.5-0.5B-style per-organ decoder.
- Prompt: `Generate the CT finding for {organ}###\n`
- Training: `10` epochs, batch size `64`, LR `2e-4`, cosine schedule.
- Generation cap in config: `128` new tokens.
- Data root: `/net/storage/pr3/plgrid/plggjmiag/Merlin_converted`
- Train/val feature caches contain `15,309` train studies and `5,055` val studies.

Training behavior:

- The previous decoder overfits after epoch `3` or `4`.
- Best checkpoints were selected from validation, not final epoch.
- This is a healthy training pattern for checkpoint selection.

Representative best validation losses:

| run | best epoch | best val total | final val total | note |
| --- | ---: | ---: | ---: | --- |
| nodiag | 3 | 0.729360 | 0.919060 | clear overfit after best |
| lexical w0.02 | 3 | 0.811854 | 1.000600 | auxiliary term large, CE worse |
| semantic minimal | 4 | 0.759223 | 0.925206 | stable, but not truly minimal; see below |
| semantic family+subtype | 4 | 0.758253 | 0.930235 | stable |
| semantic family w0.05 | 3 | 0.732038 | 0.918077 | tiny aux contribution |

Important comparability issue:

The previous decoder `semantic_minimal` name is misleading. In the old decoder implementation, family loss is computed whenever family targets exist and `family_weight > 0`, even under the `minimal` variant. Merlin's `minimal` variant does not include family loss. So `semantic_minimal` is not the same ablation across the two model families.

### Merlin Runs

Setup:

- Model: original Merlin report-generation stack imported through the local Merlin repo.
- Image encoder: frozen.
- Adapter: trainable.
- Decoder LoRA: trainable.
- Total parameters: `7,132,609,984`.
- Trainable Merlin parameters: `276,828,160`.
- Semantic target vocab loaded: `60,851` unique text targets.
- Family vocab size used by Merlin loss: `23`.
- Subtype vocab size: `235`.
- Training: `5` epochs, batch size `8`, gradient accumulation `2`, LR `1e-5`.
- Prompt: `Generate a radiology report for {organ}###\n`
- Training max length: `1024`.
- Cached image feature shape: `(490, 2048)`, leaving about `534` text-token positions after image features.

Training jobs:

| job | run | state | elapsed |
| --- | --- | --- | --- |
| 17689137 | CE | COMPLETED | 21:57:54 |
| 17689138 | lexical | COMPLETED | 21:21:56 |
| 17689139 | semantic minimal | COMPLETED | 21:21:49 |
| 17689140 | semantic family | COMPLETED | 23:40:04 |
| 17689141 | semantic family+subtype | COMPLETED | 21:29:41 |
| 17689142 | lexical+semantic family | COMPLETED | 22:04:40 |

Training behavior:

- All Merlin runs improved monotonically through epoch `5`.
- Best checkpoint is final epoch for every run.
- There is no overfit signal yet.
- This means the Merlin runs may be under-trained relative to the previous decoder, which trained for `10` epochs and selected an earlier best checkpoint.

Final validation losses:

| run | final val loss | final val CE | weighted aux |
| --- | ---: | ---: | ---: |
| CE | 0.518462 | 0.518462 | 0.000000 |
| lexical w0.002 | 0.517149 | 0.516947 | 0.000202 |
| semantic minimal w0.005 | 0.518909 | 0.518500 | 0.000409 |
| semantic family w0.002 | 0.517660 | 0.517455 | 0.000206 |
| semantic family+subtype w0.001 | 0.518500 | 0.518390 | 0.000110 |
| lexical w0.002 + semantic family w0.001 | 0.517830 | 0.517526 | 0.000304 |

The Merlin auxiliary losses are extremely small compared with CE. They may be too weak to create a meaningful training effect.

## Evaluation Audit

### Previous Decoder Evaluation

Setup:

- Held-out test studies: `5,125`
- Generated organ rows: `56,375`
- Basic benchmark job: `17608392`, completed in `06:42:25`.
- Sampled GREEN job: `17608393`, completed in `04:21:36`.
- RadGPT job: `17644190`, Slurm state failed after producing usable metrics and later attachment.
- Basic benchmark used `--no-green --no-study-level`.
- Sampled GREEN used `1,000` abnormal-positive and `1,000` abnormal-negative organ rows.

Generation quality:

- Generated mean length: about `9` to `11` words.
- Target mean length: about `13` words.
- End-punctuation rates: mostly `70%` to `94%`.
- The main concern is empty generations in two runs:
  - `sem_primary_secondary_w005`: `972 / 56,375` empty generations.
  - `lexw002_sem_family_w002`: `13 / 56,375` empty generations.

These empty rows are not enough to invalidate every result, but the affected runs should be treated with caution.

Best previous-decoder signals:

- Best RadGPT all macro-F1: `sem-family-w005`, about `0.1965`.
- Best text overlap: `lexw002-sem-primary-secondary-w002`.
- Best sampled GREEN: CE-only baseline.

This means the previous decoder results are mixed:

- Semantic family supervision looks useful for oncology-style RadGPT labels.
- Lexical/semantic mixed supervision can improve lexical overlap.
- GREEN did not clearly favor the auxiliary losses.
- Some semantic variants caused output-quality degradation.

### Merlin Evaluation

Setup:

- Held-out test studies: `5,125`
- Generated organ rows: `56,375`
- Basic metric jobs: `17694193` through `17694198`, all completed.
- Sampled GREEN job: `17698460`, completed in `08:37:01`.
- RadGPT job: `17698520`, Slurm state failed after producing metrics; attach job `17710493` completed.
- Sampled GREEN used `3,000` abnormal-positive and `3,000` abnormal-negative organ rows.

Critical flaw:

Every Merlin generation job used:

```text
--max-new-tokens 128
```

But the Merlin prompt asked:

```text
Generate a radiology report for {organ}###
```

This combination produced long, unfinished report fragments.

Generation quality:

| run | generated mean words | end punctuation | unclosed |
| --- | ---: | ---: | ---: |
| CE | 71.83 | 6.18% | 93.82% |
| lexical | 71.74 | 6.28% | 93.72% |
| semantic minimal | 71.62 | 6.31% | 93.69% |
| semantic family | 72.05 | 6.33% | 93.67% |
| semantic family+subtype | 72.14 | 6.22% | 93.78% |
| lexical+semantic family | 71.85 | 6.42% | 93.58% |

Example:

Target:

```text
Mildly thickened left adrenal gland. Normal right adrenal gland.
```

Generated:

```text
The adrenal glands have normal morphology with no nodules or masses. Mild thickening of the left adrenal gland without discrete mass lesion identified, likely related to chronic inflammation from adjacent diverticulitis. The right adrenal gland is unremarkable. A 1 cm fat attenuating nodule in the lateral limb of the left adrenal gland consistent with an adenoma. No focal nodularity within this region on delayed imaging. This may represent a small focus of residual thrombus versus calcification. Small bilater
```

This is cut off mid-word and contains extra unsupported details. Therefore the Merlin benchmark metrics are provisional only.

## Metric Reliability

Basic text metrics:

- Previous decoder text metrics are usable, except for affected runs with empty generations.
- Merlin text metrics are compromised by systematic truncation.

GREEN:

- Previous decoder sampled GREEN used `2,000` total sampled rows per run.
- Merlin sampled GREEN used `6,000` total sampled rows per run.
- Within each benchmark family, comparisons are okay.
- Cross-family GREEN comparisons are not perfectly apples-to-apples.
- Merlin GREEN is still compromised by truncated generations.

RadGPT:

- RadGPT metrics were generated for both benchmark families.
- Slurm state for the RadGPT jobs can show failed because summary refresh/attachment failed after core labels were generated.
- The attached metric artifacts are present.
- However, Merlin RadGPT is measuring truncated report fragments, so it should not be used for final loss-quality claims yet.

## Trust Levels

| component | trust level | reason |
| --- | --- | --- |
| Previous decoder train split | high | train/val dirs disjoint; feature caches match split sizes |
| Previous decoder training curves | high | clear best epochs and overfit behavior |
| Previous decoder evaluation | medium-high | full test set; some runs have empty generations |
| Merlin train split | high | dirs disjoint and loader filters by existing image files |
| Merlin training execution | medium-high | jobs clean; losses logged; final checkpoints saved |
| Merlin training conclusion | medium | no overfit yet; aux weights tiny |
| Merlin evaluation | low | systematic generation truncation |
| Cross-family comparison | low until Merlin generation is fixed | prompt and generation regime differ too much |

## Corrective Plan

1. Add benchmark guardrails:
   - Record generation settings in every generation artifact.
   - Fail or loudly warn when end-punctuation rate is below a threshold such as `70%`.
   - Fail or loudly warn when unclosed-generation rate exceeds a threshold such as `25%`.
   - Fail or loudly warn when generated mean length is more than about `3x` target mean length.
   - Report empty-generation counts prominently.

2. Rerun Merlin generation-only evaluation from existing checkpoints:
   - Use the same six checkpoints.
   - Start with `max_new_tokens=512`, not `128`.
   - Also test a CT-finding prompt diagnostic:
     - `Generate the CT finding for {organ}###\n`
   - Do not draw final conclusions until generation quality is sane.

3. Decide whether to retrain Merlin with a shorter organ-finding prompt:
   - If CT-finding prompt improves generated length/completion from existing checkpoints, retrain CE plus strongest losses with that prompt.
   - If original prompt is kept, use a much larger generation cap and possibly stopping criteria.

4. Rebalance Merlin auxiliary weights:
   - Current weighted auxiliary terms are around `0.0001` to `0.0004` against CE around `0.52`.
   - These are probably too weak.
   - Test stronger weights after generation is fixed.

5. Align semantic ablations across model families:
   - Rename or fix old decoder `semantic_minimal`, because it includes family loss when family targets exist.
   - Define exact variants:
     - `normality_polarity`
     - `family_only`
     - `family_subtype_bce`
     - `primary_secondary`
   - Use matching semantics in Merlin and the previous decoder.

6. For final claims:
   - Run at least three seeds for CE and the top two auxiliary-loss configurations.
   - Use full held-out test for basic metrics and RadGPT.
   - Use a fixed shared sampled GREEN manifest for all comparable runs.

