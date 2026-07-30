"""OpenSearch-VL inference package.

Modular runtime that drives a Visual Investigation Agent over the
FVQA-style benchmarks. The package is split into:

* ``config``        - environment-driven settings and the model registry
* ``prompts``       - the Visual Investigation Agent system prompt
* ``auth``          - HMAC helper plus the optional Claude gateway client
* ``cos_upload``    - lazy bootstrap for an external ``upload.py`` module
* ``image_io``      - download / decode / cache utilities for images
* ``image_engines`` - PIL- and OpenCV-based crop / OCR / enhance pipelines
* ``search``        - text_search, image_search and layout_parsing clients
* ``tools``         - JSON tool schema, parsing helpers and the dispatcher
* ``messages``      - Gemini <-> Claude / Qwen3-VL message converters
* ``runners``       - inference runners (Claude API, MiniMax API, dense Qwen3-VL, MoE)
* ``pipeline``      - per-case multi-turn orchestration loop

The unified entrypoint is :mod:`run_infer` at the project root.
"""

__all__ = [
    "config",
    "prompts",
    "auth",
    "cos_upload",
    "image_io",
    "image_engines",
    "search",
    "tools",
    "messages",
    "runners",
    "pipeline",
]
