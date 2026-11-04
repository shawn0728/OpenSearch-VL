#!/usr/bin/env bash
# Local page summariser: vLLM OpenAI-compatible server on :8000, text only.
# Replaces the paid DeepSeek summariser (one call per fetched page).
# Env: /root/venvs/vllm (separate from the sft env so torch versions never clash).
set -euo pipefail
MODEL=${SUMMARY_MODEL_PATH:-/root/autodl-fs/models/Qwen3.8-27B-FP8}
export CUDA_VISIBLE_DEVICES=${SUMMARY_GPU:-1}
# Prompts are <= 6000 chars of page text + 1024 output tokens; 16k leaves headroom.
exec /root/venvs/vllm/bin/vllm serve "$MODEL" \
  --served-model-name summariser --host 127.0.0.1 --port 8000 \
  --max-model-len 16384 --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image": 0, "video": 0}'
