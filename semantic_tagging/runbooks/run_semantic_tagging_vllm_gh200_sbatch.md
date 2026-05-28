# Run Semantic Tagging via `sbatch` on GH200

This is the cleanest unattended mode: one Slurm job starts the local vLLM server inside the allocation, waits for readiness, then runs the semantic tagging pipeline against `127.0.0.1:8000`.

## Files

- Batch script:
  - `semantic_tagging/examples/run_semantic_tagging_vllm_gh200.sbatch`
- Default full config:
  - `semantic_tagging/configs/merlin_vllm_full.yaml`

## Prerequisites

You must already have the vLLM environment created:

- `/net/scratch/hscra/plgrid/plgikolton/Magisterka/.venv-semantic-tagging-vllm`

If the model is gated, export `HF_TOKEN` before submission.

## Submit the default full run

```bash
export VLLM_API_KEY=EMPTY
export HF_TOKEN=your_hf_token_if_needed

cd /net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/semantic_tagging
sbatch examples/run_semantic_tagging_vllm_gh200.sbatch
```

## Submit with an explicit config

The batch script accepts the config path as its first positional argument.

```bash
export VLLM_API_KEY=EMPTY
export HF_TOKEN=your_hf_token_if_needed

cd /net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/semantic_tagging
sbatch examples/run_semantic_tagging_vllm_gh200.sbatch configs/merlin_vllm_full.yaml
```

## Logs

Slurm stdout:

- `semantic_tagging/logs/semantic-tagging-vllm-<jobid>.out`

vLLM server log:

- `semantic_tagging/logs/vllm_server_<jobid>.log`

Pipeline tee log:

- `semantic_tagging/logs/semantic_tagging_<jobid>.log`

## Output artifacts

Run outputs still go to:

- `Masters/outputs/semantic_tagging/<dataset_id>/<run_id>/`

For the default full config:

- `Masters/outputs/semantic_tagging/merlin_converted/vllm_full/`

## Notes

- The batch script keeps vLLM and the pipeline in the same job, so `127.0.0.1:8000` works reliably without a second shell.
- Current pipeline supports partial checkpointing, so rerunning the same config resumes from:
  - `raw_llm_decisions.partial.jsonl`
  - `validated_decisions.partial.jsonl`
