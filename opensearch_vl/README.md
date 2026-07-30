# OpenSearch-VL Inference

A modular Python package that drives the **Visual Investigation Agent**
across three Qwen3-VL checkpoints (8B, 32B and 30B-A3B MoE) and an
optional Claude Opus 4.5 backend. The same agent loop, tool definitions
and search/visual utilities are shared across all four backends; the
specific model is selected via a single `--model` flag.

## Layout

```
opensearch_vl/
|-- run_infer.py              # Unified entrypoint (--model 8b|32b|30b-a3b|claude)
|-- run_infer.sh              # Convenience wrapper that reads env variables
|-- run_eval.sh               # Judge harness for BC-VL / HLE / VDR-Bench
|-- eval_with_gpt4o.py        # GPT-4o judge implementation
|-- .env.example              # Environment variables consumed by everything below
`-- opensearch_infer/
    |-- __init__.py
    |-- config.py             # Env-driven settings + model registry
    |-- prompts.py            # System prompt for the Visual Investigation Agent
    |-- auth.py               # HMAC helpers + Claude gateway client
    |-- cos_upload.py         # Optional COS uploader bootstrap
    |-- image_io.py           # Image download / decode / cache utilities
    |-- image_engines.py      # PIL + OpenCV crop / OCR / enhance pipelines
    |-- search.py             # text_search / image_search / layout_parsing
    |-- tools.py              # JSON schema + parsing + tool dispatcher
    |-- messages.py           # Gemini <-> Claude / Qwen3-VL converters
    |-- runners.py            # Inference runners (Claude + MiniMax + dense + MoE)
    `-- pipeline.py           # Per-case multi-turn agent loop
```

## Quick start

1. Install dependencies into your environment:

   ```bash
   pip install torch transformers qwen-vl-utils accelerate \
               pandas pyarrow Pillow opencv-python tqdm requests httpx
   ```

   `opencv-python` is only required if you plan to use
   `perspective_correct`, `super_resolution` or `sharpen`. Everything
   else degrades gracefully when an optional dependency is missing.

2. Copy `.env.example` to a private location and fill in the entries
   that apply to your deployment:

   ```bash
   cp .env.example ~/.opensearch-vl.env
   editor ~/.opensearch-vl.env
   source ~/.opensearch-vl.env
   ```

   At minimum the runtime needs:

   - **For Qwen3-VL backends**: a checkpoint via
     `QWEN3VL_8B_PATH`, `QWEN3VL_32B_PATH` or `QWEN3VL_30B_A3B_PATH`
     (or the `--checkpoint` flag).
   - **For the Claude backend**: `CLAUDE_API_HOST`, `CLAUDE_API_USER`
     and `CLAUDE_API_KEY`.
   - **For the MiniMax backend**: `MINIMAX_API_KEY` (plus optionally
     `MINIMAX_API_REGION` to switch between the global and China
     endpoints, or an explicit `MINIMAX_ANTHROPIC_BASE_URL`).
   - **For `text_search` / `image_search`**: either the gateway triple
     (`API_HOST` + `API_USER` + `API_KEY`) or direct provider keys
     (`SERPER_API_KEY` + `JINA_API_KEY`), plus a Qwen-compatible chat
     completions server reachable at `QWEN_API_BASE`.
   - **For `layout_parsing`**: `LAYOUT_PARSING_API_URL` and (optionally)
     `LAYOUT_PARSING_TOKEN`.

3. Run inference. The unified entrypoint accepts a `--model` flag:

   ```bash
   # Dense Qwen3-VL-8B on a single GPU
   python run_infer.py --model 8b --gpus 0 \
       --data-path /data/fvqa_train.parquet \
       --output-dir ./outputs/fvqa_train_8b

   # Dense Qwen3-VL-32B with 4-way model parallel
   python run_infer.py --model 32b --gpus 0,1,2,3 \
       --data-path /data/fvqa_test.parquet \
       --output-dir ./outputs/fvqa_test_32b

   # MoE Qwen3-VL-30B-A3B-Instruct (auto-applies the scatter dtype patch)
   python run_infer.py --model 30b-a3b --gpus 0,1,2,3 \
       --checkpoint /models/Qwen3-VL-30B-A3B-Instruct \
       --data-path /data/fvqa_train.parquet \
       --output-dir ./outputs/fvqa_train_30b_a3b

   # Claude Opus 4.5 (no GPUs needed)
   python run_infer.py --model claude \
       --data-path /data/fvqa_test.parquet \
       --output-dir ./outputs/fvqa_test_claude

   # MiniMax-M3 via the Anthropic-compatible API (no GPUs needed)
   python run_infer.py --model minimax-m3 \
       --data-path /data/fvqa_test.parquet \
       --output-dir ./outputs/fvqa_test_minimax_m3
   ```

   `run_infer.sh` is a thin wrapper that sources environment variables
   such as `MODEL`, `DATA_PATH`, `OUTPUT_DIR`, `GPUS`, `LIMIT`, etc.

4. Evaluate trajectories against a GPT-4o judge:

   ```bash
   export JUDGE_API_BASE_URL=...
   export JUDGE_APP_ID=...
   export JUDGE_APP_KEY=...
   export TRAJ_HLE=./outputs/hle
   export TRAJ_BC_VL_LEVEL1=./outputs/bc_vl_level1
   export TRAJ_VDR_PRIMARY=./outputs/vdr_primary
   export VDR_ANSWER_PARQUET=/data/vdr_testmini.parquet

   bash run_eval.sh --workers 20
   ```

## Tooling

The agent has access to the following tools (defined in
`opensearch_infer/tools.py` and dispatched by `execute_tool`):

| Tool                    | Description |
|-------------------------|-------------|
| `crop`                  | Crop a region of an image and return a new image reference. |
| `layout_parsing`        | Hit the layout-parsing endpoint and surface structured text. |
| `text_search`           | Serper search + Jina reader + Qwen3-32B summarization. |
| `web_search`            | Alias for `text_search` (kept for legacy prompts). |
| `image_search`          | External visual lookup; results filtered by Qwen3-32B. |
| `perspective_correct`   | OpenCV-based document perspective correction. |
| `super_resolution`      | OpenCV `dnn_superres` upscaling (requires a model file). |
| `sharpen`               | Unsharp-mask sharpening. |

`image_search` requires a callable visual lookup. Pass one through
`pipeline.process_single_case(..., visual_lookup=fn)` if you want the
tool to call into your own backend; otherwise it returns a clean error.

## Extending

- **Add a new model** by registering a `ModelSpec` in
  `opensearch_infer/config.py` and (if needed) extending `runners.py`.
- **Add a new tool** by extending `get_tools_definition`, the
  `_FALLBACK_TAGS` table and `execute_tool` in `opensearch_infer/tools.py`.
- **Swap the search backend** by editing
  `opensearch_infer/search.py`. All upstream calls go through that
  module so models / agents are unaffected.

The package has no internal hostnames, credentials or storage paths
hard-coded; every external endpoint is read from an environment
variable, with sensible public defaults where applicable.
