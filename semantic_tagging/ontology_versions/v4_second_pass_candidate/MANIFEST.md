# V4 Second Pass Candidate

This ontology bundle is derived from `v3_second_pass_candidate` and is intended as a narrow cleanup pass rather than a broad redesign.

Purpose of `v4`:

- preserve the strong `v3` gains for `Colon` and `Small bowel`
- keep `Kidneys` essentially stable
- reduce the remaining `Gallbladder` unresolved tail
- remove the last remaining `Colon` unresolved edge case

Primary `v4` changes:

- add `colon_perirectal_fluid_collection`
- add `gallbladder_collapse`
- add `gallbladder_incompletely_assessed`
- tighten prompt instructions around:
  - ignoring adjacent non-target-organ findings
  - encoding negation in polarity/certainty instead of inventing `*_negated` subtypes
- normalize stray negated secondary subtype names during validation instead of failing the whole record

This bundle is versioned separately so that:

- `v1_full_run_baseline`
- `v2_second_pass_candidate`
- `v3_second_pass_candidate`
- `v4_second_pass_candidate`

remain independently auditable.
