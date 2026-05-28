# Ontology Versioning

This directory keeps explicit filesystem-level ontology versions in addition to git history and run-level ontology snapshots.

## Why this exists

The semantic tagging pipeline already writes immutable run artifacts:

- `ontology_snapshot/`
- `final_ontology_snapshot/`

under each run output directory.

That is good for provenance after a run finishes, but it is not enough for day-to-day ontology curation. We also want editable, named ontology bundles so we can:

- preserve a clean baseline
- create a candidate ontology for the next pass
- compare runs across ontology versions without overwriting the current source tree

## Current bundles

- `v1_full_run_baseline`
  - baseline copied before `ontology v2` curation work
- `v2_second_pass_candidate`
  - working copy for second-pass ontology changes

## How to use this

The pipeline already supports selecting an ontology by path using `paths.ontology_root` in the run config.

That means we can run:

- baseline runs against `ontology_versions/v1_full_run_baseline`
- candidate runs against `ontology_versions/v2_second_pass_candidate`

without losing history or mutating the original source ontology in place.

## Policy

1. Do not delete old ontology bundles after a run.
2. Prefer creating a new named ontology bundle over silently replacing an old one.
3. Record the ontology path used by each run in the config committed to the repo or saved with the run.
