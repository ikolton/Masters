# V3 Edit Plan

## Purpose

This plan defines the next round of edits after the finished targeted `v2` run:

- run: [vllm_targeted_v2_candidate summary](/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/outputs/semantic_tagging/merlin_converted/vllm_targeted_v2_candidate/reports/summary.md)
- ontology bundle used: [v2_second_pass_candidate](/net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/semantic_tagging/ontology_versions/v2_second_pass_candidate/MANIFEST.md)

This is not a generic wishlist. It is based on what still failed after `v2` already improved the hard organs.

## What Changed In V2

Compared to the first-pass baseline on the same four organs:

- accepted: `27766 -> 28209`
- accepted provisional: `2109 -> 1669`
- unresolved: `20 -> 17`

Biggest gains:

- `Kidneys`
  - provisional rate `0.0136 -> 0.0047`
- `Small bowel`
  - provisional rate `0.1294 -> 0.0793`

Smaller gains:

- `Colon`
  - provisional rate `0.0825 -> 0.0789`
- `Gallbladder`
  - provisional rate `0.1225 -> 0.1172`

So `v2` clearly helped, but the remaining pressure is now much more concentrated.

## Decision Process

The `v3` plan uses the following rule:

1. Fix the unresolved concepts that recur directly.
2. Fix the highest-reuse provisional concepts that still signal ontology holes.
3. Prefer canonical merges over ontology explosion.
4. Only widen the ontology where `v2` data shows repeated value.
5. Do not rebuild the family system unless the data forces it.

## What Still Fails After V2

From the finished targeted `v2` run, the unresolved tail is:

- `17` total unresolved
- `10` ontology-gap unresolved
- `7` JSON-format unresolved

### Residual unresolved ontology gaps

#### Colon

Repeated direct failures:

- `colon_perirectal_stranding`
- `colon_mass`

Representative unresolved examples:

- `Perirectal fat stranding is present, with associated mild rectal wall thickening.`
- `Mass along the ascending colon with adjacent calcification approximately at the level of the ileocecal junction.`

These are the clearest `v3` colon additions.

#### Small bowel

Residual direct failure:

- `small_bowel_mesenteric_infiltration`

Representative unresolved example:

- `Mild associated mesenteric infiltration ... appearance suggests early or partial small bowel obstruction, likely due to adhesions.`

This suggests `v2` still lacks a stable post-op / mesenteric-reactive companion concept.

#### Gallbladder

Residual direct failure:

- `gallbladder_polyp_or_stone`

Representative unresolved example:

- `Small nodular densities in the gallbladder lumen, representing polyps or stones.`

This is not a random failure. It is a recurring ambiguous imaging pattern.

#### Kidneys

Only one unresolved kidney case remained, and it is a long complex transplant/native-kidney mixed description with JSON failure rather than a clean new subtype need.

Conclusion:

- kidneys are not the main ontology priority for `v3`
- kidney work in `v3` should focus more on parser/repair resilience than on broad ontology expansion

## What Still Looks Provisional After V2

This section only looks at true provisional subtype usage that remained after the `v2` targeted run.

### Colon

Most important surviving provisional concepts:

- `colon_dilation` (`30`)
- `colon_distension` (`26`)
- `colon_perforation` (`24`)
- `colon_fluid` (`23`)
- `colon_fistula` (`21`)
- `colon_narrowing` (`21`)
- `colon_abscess` (`18`)
- `colon_anastomosis` (`18`)
- `colon_obstructive_mass` (`18`)
- `colon_pericolonic_fluid_collection` (`15`)
- `colon_leak` (`13`)

Interpretation:

- colon still has two unresolved classes of missing structure:
  - rectal/perirectal inflammatory/post-op changes
  - complicated diverticulitis / post-op complication vocabulary

### Gallbladder

Most important surviving provisional concepts:

- `gallbladder_cholelithiasis_or_wall_calcification` (`24`)
- `gallbladder_vicarious_excretion` (`17`)
- `gallbladder_polyp_or_gallstone` (`16`)
- `gallbladder_wall_edema` (`13`)
- `gallbladder_decompressed` (`12`)
- `gallbladder_sludge_or_tiny_stones` (`12`)

Interpretation:

- gallbladder still has unresolved ambiguity around:
  - polyp vs stone
  - edema vs thickening/cholecystitis
  - decompressed vs absent vs contracted

### Small bowel

Most important surviving provisional concepts:

- `small_bowel_adjacent_fluid_collection` (`14`)
- `small_bowel_anastomotic_leak` (`12`)
- `small_bowel_fistulous_tract` (`11`)
- `small_bowel_anastomosis` (`11`)
- `small_bowel_ileus` (`11`)
- `small_bowel_mesenteric_stranding` (`10`)

Interpretation:

- small bowel now looks less like “general inflammatory chaos” and more like:
  - postoperative complication vocabulary
  - mesenteric reactive-change vocabulary

### Kidneys

No remaining kidney provisional concept is both frequent and obviously missing enough to justify broad `v3` ontology expansion.

Interpretation:

- kidneys are mostly good enough semantically
- focus `v3` kidney effort on stability, not major ontology growth

## V3 Editing Strategy

## 1. Create A New Versioned Bundle

Do not edit `v2_second_pass_candidate` in place.

Create:

- `semantic_tagging/ontology_versions/v3_second_pass_candidate/`

by copying:

- `semantic_tagging/ontology_versions/v2_second_pass_candidate/`

This preserves:

- `v1_full_run_baseline`
- `v2_second_pass_candidate`
- `v3_second_pass_candidate`

as separate ontology states.

## 2. Ontology Edits To Implement In V3

### Colon

#### Promote now

Add explicit colon subtypes for:

- `colon_perirectal_stranding`
- `colon_mass`
- `colon_abscess`
- `colon_pericolonic_fluid_collection`
- `colon_perforation`

Why these:

- they are directly supported by unresolveds and/or strong provisional reuse
- they cover real semantic gaps that `v2` did not close

#### Canonical merge, not blind expansion

Do not keep all these as separate long-term subtypes unless needed:

- `colon_dilation`
- `colon_distension`
- `colon_decompression`
- `colon_collapse`
- `colon_narrowing`

Preferred `v3` direction:

- choose one broad caliber abnormality anchor, likely `colon_distension`
- optionally keep `colon_narrowing` if it remains clinically distinct
- map the others to the chosen canonical subtype or to modifiers

#### Post-op complication merge

Do not keep all of these as isolated standalone ideas if one cleaner abstraction works:

- `colon_fistula`
- `colon_leak`
- `colon_anastomosis`

Preferred `v3` direction:

- keep `colon_anastomosis` only if it is needed as a stable post-op structural tag
- otherwise prefer a clearer complication concept like:
  - `colon_fistula_or_leak`
or explicitly decide to separate:
  - `colon_fistula`
  - `colon_anastomotic_leak`

This is the one place where we should make a deliberate ontology choice rather than let the model keep inventing neighboring labels.

### Small bowel

#### Promote now

Add explicit small bowel subtypes for:

- `small_bowel_mesenteric_infiltration`
- `small_bowel_anastomosis`
- `small_bowel_anastomotic_leak`
- `small_bowel_fistulous_tract`
- `small_bowel_ileus`

Why these:

- they match the surviving unresolved and high-reuse provisional patterns
- they are exactly the sort of post-op / complication language that `v2` still underrepresents

#### Canonical merge

Likely unify:

- `small_bowel_mesenteric_infiltration`
- `small_bowel_mesenteric_stranding`

Preferred `v3` direction:

- one canonical mesenteric-reactive subtype
- one wording mapped to the other rather than separate ontology members

### Gallbladder

#### Promote now

Add:

- `gallbladder_polyp_or_stone`

Why:

- it appears directly in unresolveds
- there are multiple related provisional variants
- it represents a real imaging ambiguity rather than a random wording quirk

#### Canonical merge

Merge or map:

- `gallbladder_polyp_or_gallstone`
- `gallbladder_polyp_or_stone`

to one canonical subtype.

Likewise evaluate whether these should map to existing v2 subtypes rather than become standalone:

- `gallbladder_wall_edema`
- `gallbladder_decompressed`
- `gallbladder_sludge_or_tiny_stones`

Preferred `v3` direction:

- `wall_edema` likely maps into `gallbladder_wall_thickening` or `gallbladder_cholecystitis`
- `decompressed` likely maps into a caliber/assessment state rather than a full new disease subtype
- `sludge_or_tiny_stones` likely maps into `gallbladder_sludge` unless a broader ambiguous subtype is more useful

### Kidneys

No major ontology expansion is recommended for `v3` unless review of concrete examples reveals a very stable missing pattern.

Preferred `v3` kidney work:

- improve handling of long composite transplant/native cases
- improve JSON repair on long complex kidney outputs

## 3. Prompt Edits For V3

The ontology alone is not enough. Some of the remaining pressure is naming drift.

### Add few-shot examples in the hard zones

Update organ few-shots so the model sees examples for:

- `Colon`
  - perirectal stranding
  - abscess / collection
  - broad mass wording mapped to a canonical mass subtype
- `Small bowel`
  - anastomotic leak
  - fistulous tract
  - mesenteric reactive/infiltrative changes
- `Gallbladder`
  - ambiguous polyp vs stone wording

### Tighten canonicalization instructions

Add explicit guidance such as:

- if an existing subtype already matches the concept, do not propose a wording variant
- if wording is uncertain between two existing related gallbladder lesion types, use the canonical ambiguity subtype rather than invent a new hybrid
- if the concept is a broad malignant or obstructive colonic mass, use the canonical mass subtype rather than surface-form variants like `polypoid`, `circumferential`, or `invasive` unless the ontology explicitly keeps them separate

## 4. Validation / Repair Edits For V3

`v2` unresolveds were:

- `10` ontology-gap
- `7` JSON-format

That means parser/repair work is still worth doing.

### Recommended repair changes

Add one stronger recovery path for malformed responses:

- if JSON parse fails, send a repair prompt using the raw model output and require strict schema-only JSON

This is especially worthwhile because the remaining JSON failures often come from long complex complication descriptions, not from trivial bad generations.

### Do not overread noisy flags

`negative_with_primary_subtype` and `normal_with_subtypes` still overfire and should not drive `v3` ontology decisions directly.

## 5. Proposed File Edit Scope

If we implement `v3`, the main files to touch should be:

- `semantic_tagging/ontology_versions/v3_second_pass_candidate/global_axes.yaml`
  - probably no major changes
- `semantic_tagging/ontology_versions/v3_second_pass_candidate/shared_families.yaml`
  - likely unchanged
- `semantic_tagging/ontology_versions/v3_second_pass_candidate/organs/colon.yaml`
- `semantic_tagging/ontology_versions/v3_second_pass_candidate/organs/small_bowel.yaml`
- `semantic_tagging/ontology_versions/v3_second_pass_candidate/organs/gallbladder.yaml`

Prompt support files likely to edit:

- `semantic_tagging/prompts/fewshot/colon.jsonl`
- `semantic_tagging/prompts/fewshot/small_bowel.jsonl`
- `semantic_tagging/prompts/fewshot/gallbladder.jsonl`

Likely code areas if we do repair improvements:

- `semantic_tagging/src/semantic_tagging/pipeline.py`
- `semantic_tagging/src/semantic_tagging/validation.py`
- `semantic_tagging/src/semantic_tagging/schemas.py`

## 6. Recommended V3 Run Strategy

Do not jump straight to a full all-organs rerun.

Recommended order:

1. create `v3_second_pass_candidate`
2. implement the ontology and prompt edits above
3. run a smoke test on selected hard examples
4. run a targeted hard-organ rerun again:
   - `Colon`
   - `Small bowel`
   - `Gallbladder`
   - optionally `Kidneys`
5. compare against the `v2` targeted run before deciding on any broader rerun

## 7. V3 Success Criteria

`v3` should be accepted only if it improves the specific residual pain points from `v2`.

Minimum targets:

- reduce unresolved count below `17`
- specifically reduce colon unresolveds below `13`
- cut or eliminate `colon_perirectal_stranding` unresolveds
- eliminate `gallbladder_polyp_or_stone` unresolveds
- reduce small-bowel post-op complication provisional drift
- reduce JSON-format unresolveds below `7`

## Bottom Line

`v2` proved the approach works.

`v3` should now be much more surgical:

- mostly colon
- some small bowel post-op complication vocabulary
- one gallbladder ambiguity subtype
- better repair for malformed JSON

That is the best evidence-based path from the current artifacts.
