# Encoder training configs

Use the configs in this directory for new encoder runs.

Current baseline configs:

- `organsegclip_128x48_20ep_uniformtext_schedcos.yaml`
- `organsegclip_128x48_20ep_uniformtext_schedcos_logitslr1e5.yaml`
- `organsegclip_128x48_20ep_uniformtext_schedcos_gh200_2gpu_bs12.yaml`

Before launching runs, export the local data paths in your shell environment:

- `ORGAN_SEG_CLIP_DATASET_ROOT`
- `ORGAN_SEG_CLIP_LESION_METADATA_CSV`

The older pilot, benchmark, smoke, and profiling configs are kept under `legacy/` for provenance, but they should not be treated as recommended starting points for new server runs.

Current GH200 recommendation for 2-GPU throughput-focused runs:

- `batch_size: 6` per GPU
- `patch_batch_size: 16`
- `num_workers: 2`
- `compile_model: false`

See `docs/encoder_training_handoff.md` for the current training status, known issues, and suggested next experiments.
