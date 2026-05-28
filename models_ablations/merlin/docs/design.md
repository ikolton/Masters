# Design Notes

The harness tests loss transfer while preserving Merlin's released report
generation pathway.

## What Is Reused

- `merlin.data.DataLoader`
- `merlin.data.monai_transforms.ImageTransforms`
- `Merlin(RadiologyReport=True)`
- `Clip3DForTextGeneration.encode_image`
- `Clip3DForTextGeneration.adapter`
- `TextDecoder.tokenizer`
- `TextDecoder.text_decoder`
- LoRA decoder parameters as defined by Merlin

## What Is New

- Organ-level dataset records from Merlin-converted `combined.json`.
- Teacher-forcing wrapper that returns decoder hidden states.
- Auxiliary diagnostic heads.
- Config-driven loss switches.
- Output manifests and smoke-run SLURM templates.

## Why Not Call Merlin's Original `forward()` Directly?

The released forward returns only CE loss and puts the image-to-decoder path
inside `torch.no_grad()`. That is fine for a compact demo-style model, but it
prevents us from attaching hidden-state losses and would also block adapter
training. The wrapper therefore calls the same submodules directly.

