# V4 Edit Plan

## Purpose

`v4` is a surgical follow-up to the finished targeted `v3` run:

- run summary: [vllm_targeted_v3_candidate](/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/outputs/semantic_tagging/merlin_converted/vllm_targeted_v3_candidate/reports/summary.md)
- ontology bundle used there: [v3_second_pass_candidate](/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/semantic_tagging/ontology_versions/v3_second_pass_candidate/MANIFEST.md)

This is not a broad ontology expansion pass. It exists to resolve the very small residual tail left by `v3`.

## Grounded Starting Point

The targeted `v3` run covered the four hard organs:

- `Colon`
- `Gallbladder`
- `Kidneys`
- `Small bowel`

and finished with:

- accepted: `29837`
- accepted provisional: `50`
- unresolved: `8`

Compared to `v2`:

- accepted: `28209 -> 29837`
- accepted provisional: `1669 -> 50`
- unresolved: `17 -> 8`

So `v3` was a strong success overall.

## Decision Process

`v4` follows these rules:

1. Only fix failure modes that are still visible in the real `v3` artifacts.
2. Prefer prompt and validation cleanup over ontology growth when the problem is formatting or negation.
3. Add new subtypes only when the unresolved text clearly represents a stable missing concept.
4. Avoid rerunning broad organs just to chase a tiny tail.

## What Still Failed After V3

Residual unresolveds in `v3` were:

### Gallbladder

1. `Collapsed, limiting evaluation.`
2. `Gallbladder wall thickening is incompletely assessed on this examination.`
3. `Surgically absent gallbladder. Mild intrahepatic biliary prominence likely due to reservoir effect in the setting of cholecystectomy.`
4. `The gallbladder contains a single stone but does not exhibit any signs of cholecystitis.`
5. `The gallbladder is distended. No radiopaque gallstones, gallbladder wall thickening, or pericholecystic fluid.`
6. `The gallbladder is filled with sludge without evidence of radiopaque gallstones, gallbladder wall thickening, or pericholecystic fluid.`
7. `The gallbladder is mildly distended. No cholecystitis.`

### Colon

1. `Transition point in the proximal rectum in the region of the perirectal fluid collection. Similar appearing partial distal large bowel obstruction at the level of the perirectal abscess.`

## Interpretation Of The V3 Tail

The remaining failures are of three types:

### 1. Real missing subtypes

These should be promoted into the ontology:

- `colon_perirectal_fluid_collection`
- `gallbladder_collapse`
- `gallbladder_incompletely_assessed`

### 2. Negation handling failures

These are not ontology gaps. They are prompt/validation behavior problems:

- `gallbladder_pericholecystic_fluid_negated`
- `gallbladder_cholecystitis_negated`

The correct behavior is:

- keep the positive target-organ subtype if present
- encode the negative clause using `polarity`
- do **not** create secondary subtype names ending in `_negated`

### 3. Adjacent-system leakage

This is also not really an ontology gap:

- `gallbladder_intrahepatic_biliary_tree_prominence`

For a gallbladder-organ task, this should usually be ignored unless it directly defines the gallbladder finding.

## V4 Changes

### Ontology changes

Create a new versioned bundle:

- [v4_second_pass_candidate](/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/semantic_tagging/ontology_versions/v4_second_pass_candidate/MANIFEST.md)

Add:

- `Colon`
  - `colon_perirectal_fluid_collection`
- `Gallbladder`
  - `gallbladder_collapse`
  - `gallbladder_incompletely_assessed`

### Prompt changes

Tighten instructions so the model:

- does not invent `*_negated` subtype names
- uses `polarity` and `certainty` for negation/uncertainty
- ignores adjacent non-target-organ findings unless essential

### Few-shot changes

Teach the exact remaining behavior:

- `Colon`
  - perirectal fluid collection + secondary abscess/narrowing
- `Gallbladder`
  - positive gallstones with negative cholecystitis -> `mixed`, not `*_negated`
  - distension with explicitly absent wall thickening/pericholecystic fluid -> `mixed`
  - sludge with explicitly absent inflammatory findings -> `mixed`

### Validation changes

Normalize benign negated-secondary drift:

- if an unknown `secondary_subtype` ends with `_negated`
- and the base subtype is otherwise valid for the organ
- drop the negated secondary and keep the record

This is intentionally narrow. It is not a blanket “drop unknown subtypes” rule.

## Expected V4 Effect

If `v4` works as intended, it should:

- eliminate the remaining `Colon` unresolved case
- reduce `Gallbladder` unresolveds substantially
- keep `Colon` and `Small bowel` gains from `v3`
- leave `Kidneys` essentially stable

## Files Touched For V4

- `semantic_tagging/ontology_versions/v4_second_pass_candidate/`
- `semantic_tagging/configs/merlin_vllm_smoke_v4_candidate.yaml`
- `semantic_tagging/configs/merlin_vllm_targeted_v4_candidate.yaml`
- `semantic_tagging/prompts/fewshot/colon.jsonl`
- `semantic_tagging/prompts/fewshot/gallbladder.jsonl`
- `semantic_tagging/prompts/system_v1.md`
- `semantic_tagging/prompts/user_v1.md`
- `semantic_tagging/src/semantic_tagging/validation.py`

## Acceptance Criteria

`v4` should be accepted if it:

- keeps total unresolved at or below `8`
- reduces `Gallbladder` unresolveds below `7`
- eliminates the remaining `Colon` unresolved
- does not re-inflate provisional counts
- does not materially regress `Colon`, `Small bowel`, or `Kidneys`
