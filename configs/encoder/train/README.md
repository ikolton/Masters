# Encoder training configs

Use the configs in this directory for new encoder runs.

Current baseline configs:

- `organsegclip_128x48_20ep_uniformtext_schedcos.yaml`
- `organsegclip_128x48_20ep_uniformtext_schedcos_logitslr1e5.yaml`

The older pilot, benchmark, smoke, and profiling configs are kept under `legacy/` for provenance, but they should not be treated as recommended starting points for new server runs.

See `docs/encoder_training_handoff.md` for the current training status, known issues, and suggested next experiments.
