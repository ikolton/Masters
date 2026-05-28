# Launch vLLM Server on GH200

Assumes the environment from `create_vllm_env_gh200.md` already exists.

## 1. Activate environment

```bash
module load ML-bundle/24.06a
source /net/scratch/hscra/plgrid/plgikolton/Magisterka/.venv-semantic-tagging-vllm/bin/activate
export HF_TOKEN=your_token_if_needed
```

## 2. Launch server

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 32768 \
  --host 0.0.0.0 \
  --port 8000
```

## 3. Verify from another shell

```bash
curl http://127.0.0.1:8000/v1/models
```

## 4. Use with semantic_tagging

Set the config backend section to:

```yaml
backend:
  kind: vllm
  model_name: meta-llama/Llama-3.3-70B-Instruct
  base_url: http://127.0.0.1:8000/v1
  api_key_env: VLLM_API_KEY
  timeout_seconds: 120
  temperature: 0.0
  top_p: 1.0
  max_tokens: 900
  use_response_format_json: false
  use_guided_json: false
```
