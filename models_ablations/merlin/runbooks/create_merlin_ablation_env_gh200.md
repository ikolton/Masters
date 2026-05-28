# Create Separate Merlin Ablation Env On GH200

This is intentionally a **manual command runbook**. Do not run these commands on
the login node, and do not install into existing project envs.

## 1. Start/enter a GH200 job

Use your existing bashrc aliases/account variables if preferred. Example:

```bash
srun --job-name=merlin-env \
  --account="${ACCOUNT}" \
  --partition="${PARTITION_ID}" \
  --time=02:00:00 \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=16 \
  --mem=120G \
  --gres=gpu:1 \
  --pty bash
```

## 2. Create an isolated env

```bash
module load ML-bundle/24.06a

export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/net/scratch/hscra/plgrid/plgikolton/pip_cache}"
export PYTHONUSERBASE="${PYTHONUSERBASE:-/net/scratch/hscra/plgrid/plgikolton/pip_lib}"
export HF_HOME="${HF_HOME:-/net/scratch/hscra/plgrid/plgikolton/hf_models}"
export TORCH_HOME="${TORCH_HOME:-/net/scratch/hscra/plgrid/plgikolton/torch_cache}"
export MERLIN_ABLATION_VENV=/net/scratch/hscra/plgrid/plgikolton/Magisterka/.venv-merlin-ablation
python3.11 -m venv "${MERLIN_ABLATION_VENV}"
source "${MERLIN_ABLATION_VENV}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
```

## 3. Install wheel-stack-first dependencies

Install Torch from the cluster wheel stack first:

```bash
python -m pip install /net/software/aarch64/el9/wheels/ML-bundle/24.06a/torch-2.8.0+cu124-cp311-cp311-linux_aarch64.whl
```

Then install Merlin runtime dependencies. Prefer binary wheels and keep the env
separate from existing project envs:

```bash
python -m pip install \
  "torchvision==0.23.*" \
  "monai>=1.3.0" \
  "nibabel" \
  "peft>=0.10.0" \
  "transformers>=4.38.2,<5" \
  "huggingface_hub" \
  "nltk" \
  "pandas" \
  "rich" \
  "einops" \
  "sentencepiece" \
  "protobuf" \
  "accelerate" \
  "pyyaml"
```

If `torchvision` tries to replace Torch, stop and use:

```bash
python -m pip install --no-deps "torchvision==0.23.*"
python -m pip install "monai>=1.3.0" "nibabel" "peft>=0.10.0" "transformers>=4.38.2,<5" "pyyaml"
```

## 4. Verify

```bash
python - <<'PY'
import torch, torchvision, monai, nibabel, peft, transformers, yaml
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("torchvision", torchvision.__version__)
print("monai", monai.__version__)
print("peft", peft.__version__)
print("transformers", transformers.__version__)
print("yaml ok")
PY
```

## 5. Submit smoke with the isolated env

```bash
cd /net/scratch/hscra/plgrid/plgikolton/Magisterka/Masters/models_ablations/merlin

MERLIN_ABLATION_VENV=/net/scratch/hscra/plgrid/plgikolton/Magisterka/.venv-merlin-ablation \
sbatch slurm/train_merlin_smoke_gh200_1gpu.sbatch configs/smoke_ce_only.yaml

MERLIN_ABLATION_VENV=/net/scratch/hscra/plgrid/plgikolton/Magisterka/.venv-merlin-ablation \
sbatch slurm/train_merlin_smoke_gh200_1gpu.sbatch configs/smoke_sem_family.yaml
```
