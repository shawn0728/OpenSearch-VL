#!/usr/bin/env bash
# PaddleOCR-VL layout-parsing API on :8080 (env from .proj-tool/env-setup.sh, stage `ocr`).
# Models are cached on the netdisk so they stay out of the instance image.
set -euo pipefail
export PADDLE_PDX_CACHE_HOME=${PADDLE_PDX_CACHE_HOME:-/root/autodl-fs/models/paddlex}
export PADDLE_PDX_MODEL_SOURCE=${PADDLE_PDX_MODEL_SOURCE:-modelscope}
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
exec /root/venvs/ocr/bin/paddlex --serve --pipeline PaddleOCR-VL --engine transformers \
  --device gpu:0 --host 127.0.0.1 --port 8080
