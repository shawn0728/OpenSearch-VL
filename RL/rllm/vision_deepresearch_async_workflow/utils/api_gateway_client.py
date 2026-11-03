"""
LLM Judge Client — thin OpenAI-compatible Chat Completions wrapper.

Used by the query-utility reward in `deepresearch_workflow._judge_query_utility`
to call an external judge model (e.g. GPT-4o) and score search-trajectory
quality on a 0.0–1.0 scale.

The client is optional: if no API key is configured, the reward falls back
to 0.0 and training proceeds normally.

Environment variables
---------------------
JUDGE_API_BASE_URL   OpenAI-compatible endpoint (default: ``https://api.openai.com/v1``).
JUDGE_API_KEY        Bearer token for the endpoint.
JUDGE_MODEL          Default model name when the caller does not pass one.

Usage::

    from vision_deepresearch_async_workflow.utils.api_gateway_client import (
        api_gateway_chat,
        is_api_gateway_configured,
    )

    if is_api_gateway_configured():
        result = api_gateway_chat(messages, model_marker="gpt-4o")
        text = result["choices"][0]["message"]["content"]
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests

try:
    from dotenv import load_dotenv

    _env_path = Path(__file__).resolve().parents[2] / ".env"
    if _env_path.exists():
        load_dotenv(_env_path, override=False)
except ImportError:
    pass


_BASE_URL = os.getenv("JUDGE_API_BASE_URL", "https://api.openai.com/v1").rstrip("/")
_API_KEY = os.getenv("JUDGE_API_KEY", "")
_DEFAULT_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o-mini")
# Extra provider-specific body, e.g. '{"thinking": {"type": "disabled"}}' for
# DeepSeek (its default thinking mode can exhaust max_tokens before answering).
try:
    _EXTRA_BODY: dict[str, Any] = json.loads(os.getenv("LLM_EXTRA_BODY_JSON", "") or "{}")
except json.JSONDecodeError:
    _EXTRA_BODY = {}


def is_api_gateway_configured() -> bool:
    """Return True when a judge API key is available."""
    return bool(_API_KEY)


def api_gateway_chat(
    messages: list[dict],
    model_marker: str | None = None,
    max_tokens: int | None = None,
    timeout: int = 300,
    temperature: float | None = None,
) -> dict:
    """Call the OpenAI-compatible Chat Completions endpoint.

    Parameters
    ----------
    messages
        OpenAI-format list of ``{"role": ..., "content": ...}`` dicts.
    model_marker
        Model name. Falls back to ``$JUDGE_MODEL`` / ``"gpt-4o-mini"``.
    max_tokens, temperature
        Standard sampling params; passed through if provided.
    timeout
        Per-request timeout in seconds.

    Returns
    -------
    dict
        The raw OpenAI response JSON. The caller reads
        ``result["choices"][0]["message"]["content"]``.
    """

    if not is_api_gateway_configured():
        raise RuntimeError(
            "Judge API not configured. Set JUDGE_API_KEY "
            "(and optionally JUDGE_API_BASE_URL / JUDGE_MODEL)."
        )

    model = model_marker or _DEFAULT_MODEL

    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        **_EXTRA_BODY,
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if temperature is not None:
        body["temperature"] = temperature

    headers = {
        "Authorization": f"Bearer {_API_KEY}",
        "Content-Type": "application/json",
    }

    resp = requests.post(
        f"{_BASE_URL}/chat/completions",
        headers=headers,
        json=body,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()
