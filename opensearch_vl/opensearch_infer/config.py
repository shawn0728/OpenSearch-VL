"""Environment-driven configuration and the model registry.

Every value is read from an environment variable so the codebase does
not embed any internal hostname, credential or storage path. Callers
that want non-default behaviour set the variable before running, or
override the value via the CLI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Generic runtime knobs
# ---------------------------------------------------------------------------

# Maximum number of agent turns per case before forcing termination.
MAX_TURNS: int = int(os.environ.get("AGENT_MAX_TURNS", "50"))

# Optional retrieval server (used when the agent runs against a local index).
RETRIEVAL_SERVER_URL: str = os.environ.get(
    "RETRIEVAL_SERVER_URL", "http://127.0.0.1:8000"
)

# Local directory that holds raw benchmark images keyed by ``case_id``.
# When set, the pipeline can recover from broken URLs by loading the
# original picture from disk. Leave empty to disable the fallback.
FVQA_IMAGE_DIR: str = os.environ.get("FVQA_IMAGE_DIR", "")


# ---------------------------------------------------------------------------
# Optional API gateway that proxies Serper / Jina behind one HMAC key
# ---------------------------------------------------------------------------

API_HOST: str = os.environ.get("API_HOST", "")
API_USER: str = os.environ.get("API_USER", "")
API_KEY: str = os.environ.get("API_KEY", "")


def gateway_enabled() -> bool:
    """Return ``True`` when the Serper / Jina gateway is fully configured."""

    return bool(API_HOST and API_USER and API_KEY)


# ---------------------------------------------------------------------------
# Direct provider keys (used when the gateway is not configured)
# ---------------------------------------------------------------------------

SERPER_API_KEY: str = os.environ.get("SERPER_API_KEY", "")
SERPER_SEARCH_URL: str = os.environ.get(
    "SERPER_SEARCH_URL", "https://google.serper.dev/search"
)
JINA_API_KEY: str = os.environ.get("JINA_API_KEY", "")
JINA_READER_URL: str = os.environ.get("JINA_READER_URL", "https://r.jina.ai/")


# ---------------------------------------------------------------------------
# Summarization / image-search backbone (defaults to a local OpenAI-compatible
# server such as vLLM hosting Qwen3-32B).
# ---------------------------------------------------------------------------

QWEN_API_BASE: str = os.environ.get("QWEN_API_BASE", "http://localhost:8000/v1")
QWEN_MODEL_NAME: str = os.environ.get("QWEN_MODEL_NAME", "Qwen/Qwen3-32B")


# ---------------------------------------------------------------------------
# Layout parsing (PP-StructureV3 compatible) endpoint
# ---------------------------------------------------------------------------

LAYOUT_PARSING_API_URL: str = os.environ.get("LAYOUT_PARSING_API_URL", "")
LAYOUT_PARSING_TOKEN: str = os.environ.get("LAYOUT_PARSING_TOKEN", "")


# ---------------------------------------------------------------------------
# Claude gateway (HMAC-secured ``/data_eval`` endpoint)
# ---------------------------------------------------------------------------

CLAUDE_API_HOST: str = os.environ.get("CLAUDE_API_HOST", "")
CLAUDE_API_USER: str = os.environ.get("CLAUDE_API_USER", "")
CLAUDE_API_KEY: str = os.environ.get("CLAUDE_API_KEY", "")
CLAUDE_API_SOURCE: str = os.environ.get("CLAUDE_API_SOURCE", CLAUDE_API_USER)
CLAUDE_API_VERSION: str = os.environ.get("CLAUDE_API_VERSION", "v2.03")
CLAUDE_MODEL_MARKER: str = os.environ.get(
    "CLAUDE_MODEL_MARKER", "api_anthropic_claude-opus-4-5-20251101"
)


# ---------------------------------------------------------------------------
# MiniMax API backend (Bearer-authenticated, Anthropic-compatible endpoint)
# ---------------------------------------------------------------------------

# ``MINIMAX_API_REGION`` selects the default endpoints:
#   * ``global_en`` -> api.minimax.io  (docs at platform.minimax.io)
#   * ``cn_zh``     -> api.minimaxi.com (docs at platform.minimaxi.com)
# An explicit ``MINIMAX_*_BASE_URL`` override always wins.
MINIMAX_API_REGION: str = os.environ.get("MINIMAX_API_REGION", "global_en")

MINIMAX_REGION_ENDPOINTS: Dict[str, Dict[str, str]] = {
    "global_en": {
        "openai_base_url": "https://api.minimax.io/v1",
        "anthropic_base_url": "https://api.minimax.io/anthropic",
    },
    "cn_zh": {
        "openai_base_url": "https://api.minimaxi.com/v1",
        "anthropic_base_url": "https://api.minimaxi.com/anthropic",
    },
}


def minimax_region_endpoints(region: Optional[str] = None) -> Dict[str, str]:
    """Return the endpoint pair for ``region`` (falls back to ``global_en``)."""

    key = (region or MINIMAX_API_REGION).strip()
    return MINIMAX_REGION_ENDPOINTS.get(key, MINIMAX_REGION_ENDPOINTS["global_en"])


MINIMAX_API_KEY: str = os.environ.get("MINIMAX_API_KEY", "")
MINIMAX_ANTHROPIC_BASE_URL: str = os.environ.get(
    "MINIMAX_ANTHROPIC_BASE_URL",
    minimax_region_endpoints()["anthropic_base_url"],
)
MINIMAX_OPENAI_BASE_URL: str = os.environ.get(
    "MINIMAX_OPENAI_BASE_URL",
    minimax_region_endpoints()["openai_base_url"],
)
# Anthropic-style API version header sent with every request.
MINIMAX_API_VERSION: str = os.environ.get("MINIMAX_API_VERSION", "2023-06-01")


def minimax_enabled() -> bool:
    """Return ``True`` when the MiniMax API key is configured."""

    return bool(MINIMAX_API_KEY)


# ---------------------------------------------------------------------------
# COS uploader bootstrap. We keep this isolated so the package can run
# without the optional internal uploader.
# ---------------------------------------------------------------------------

COS_UPLOAD_PATHS: str = os.environ.get("COS_UPLOAD_PATHS", "")
COS_UPLOAD_USERID: str = os.environ.get("COS_UPLOAD_USERID", "opensearch-vl")
COS_BUCKET_HOST_TEMPLATE: str = os.environ.get(
    "COS_BUCKET_HOST_TEMPLATE", ""
)
"""Optional template used to reconstruct a public URL when the uploader only
returns an object key. Use ``{bucket}`` / ``{region}`` placeholders, e.g.
``http://{bucket}.cos.{region}.myqcloud.com``. Leave empty when the
uploader already returns a fully-qualified URL.
"""


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """Static metadata describing one supported model variant."""

    name: str
    family: str  # "qwen3_vl_dense" | "qwen3_vl_moe" | "claude" | "minimax"
    display_name: str
    default_checkpoint_env: Optional[str] = None
    default_checkpoint: str = ""
    supports_multi_gpu: bool = True
    needs_moe_patch: bool = False
    extra: Dict[str, str] = field(default_factory=dict)


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "8b": ModelSpec(
        name="8b",
        family="qwen3_vl_dense",
        display_name="Qwen3-VL-8B-Instruct",
        default_checkpoint_env="QWEN3VL_8B_PATH",
        default_checkpoint="Qwen/Qwen3-VL-8B-Instruct",
    ),
    "32b": ModelSpec(
        name="32b",
        family="qwen3_vl_dense",
        display_name="Qwen3-VL-32B-Instruct",
        default_checkpoint_env="QWEN3VL_32B_PATH",
        default_checkpoint="Qwen/Qwen3-VL-32B-Instruct",
    ),
    "30b-a3b": ModelSpec(
        name="30b-a3b",
        family="qwen3_vl_moe",
        display_name="Qwen3-VL-30B-A3B-Instruct",
        default_checkpoint_env="QWEN3VL_30B_A3B_PATH",
        default_checkpoint="Qwen/Qwen3-VL-30B-A3B-Instruct",
        needs_moe_patch=True,
    ),
    "claude": ModelSpec(
        name="claude",
        family="claude",
        display_name="Claude Opus 4.5",
        default_checkpoint="",
        supports_multi_gpu=False,
    ),
    "minimax-m3": ModelSpec(
        name="minimax-m3",
        family="minimax",
        display_name="MiniMax-M3",
        default_checkpoint="",
        supports_multi_gpu=False,
        extra={
            "api_model": "MiniMax-M3",
            "context_window": "1000000",
            "input_modalities": "text,image,video",
            "thinking": "adaptive,disabled",
        },
    ),
    "minimax-m2.7": ModelSpec(
        name="minimax-m2.7",
        family="minimax",
        display_name="MiniMax-M2.7",
        default_checkpoint="",
        supports_multi_gpu=False,
        extra={
            "api_model": "MiniMax-M2.7",
            "context_window": "204800",
            "input_modalities": "text",
            "thinking": "always_on",
        },
    ),
}


def resolve_checkpoint(spec: ModelSpec, override: Optional[str] = None) -> str:
    """Resolve a checkpoint path / HF id with CLI > env > default priority."""

    if override:
        return override
    if spec.default_checkpoint_env:
        env_value = os.environ.get(spec.default_checkpoint_env, "").strip()
        if env_value:
            return env_value
    return spec.default_checkpoint


def list_model_names() -> str:
    return ", ".join(sorted(MODEL_REGISTRY))
