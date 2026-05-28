# Create vLLM Environment on GH200

Run these commands inside an interactive GH200 allocation.

## 1. Start an interactive job

```bash
srun --gres=gpu:4 --mem=240G --ntasks=1 --cpus-per-task=72 --job-name=semantic-tagging-env --account=plgdiffusion-gpu-gh200 -t 08:00:00 --partition=plgrid-gpu-gh200 --pty bash
```

## 2. Load the ML bundle and create the environment

```bash
module load ML-bundle/24.06a
python3.11 -m venv /net/scratch/hscra/plgrid/plgikolton/Magisterka/.venv-semantic-tagging-vllm
source /net/scratch/hscra/plgrid/plgikolton/Magisterka/.venv-semantic-tagging-vllm/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## 3. Install cluster wheel stack first

```bash
python -m pip install /net/software/aarch64/el9/wheels/ML-bundle/24.06a/torch-2.8.0+cu124-cp311-cp311-linux_aarch64.whl
python -m pip install /net/software/aarch64/el9/wheels/ML-bundle/24.06a/triton-3.4.0+git11ec6354-cp311-cp311-linux_aarch64.whl
python -m pip install /net/software/aarch64/el9/wheels/ML-bundle/24.06a/flash_attn-2.8.3-cp311-cp311-linux_aarch64.whl
python -m pip install /net/software/aarch64/el9/wheels/ML-bundle/24.06a/xformers-0.0.30+cu124-cp311-cp311-linux_aarch64.whl
python -m pip install /net/software/aarch64/el9/wheels/ML-bundle/24.06a/vllm-0.10.2+cu124-cp311-cp311-linux_aarch64.whl
```

## 4. Install semantic_tagging runtime dependencies

```bash
python -m pip install requests PyYAML jsonschema pyarrow
python -m pip install transformers sentencepiece accelerate
```

## 5. Verify the environment

```bash
python - <<'PY'
import torch, requests, yaml, jsonschema
print("torch", torch.__version__, "cuda", torch.version.cuda)
try:
    import vllm
    print("vllm", vllm.__version__)
except Exception as exc:
    print("vllm import failed:", exc)
PY
```
