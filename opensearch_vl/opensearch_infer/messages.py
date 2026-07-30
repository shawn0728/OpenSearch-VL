"""Conversion helpers between Gemini-style content and runner-specific formats."""

from __future__ import annotations

import base64
import io
import logging
import os
import urllib.parse
from typing import Any, Dict, Iterable, List

from PIL import Image

from . import config
from . import image_io


logger = logging.getLogger(__name__)


def to_claude_messages(contents: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert Gemini-style ``contents`` to Claude content blocks."""

    messages: List[Dict[str, Any]] = []
    for item in contents:
        role = item.get("role", "user")
        parts = item.get("parts", []) or []
        block: List[Dict[str, Any]] = []
        for part in parts:
            if "image_url" in part:
                value = part["image_url"]
                url = value.get("url", "") if isinstance(value, dict) else str(value)
                if url:
                    block.append({"type": "image_url", "value": url})
            elif "inline_data" in part:
                data = part["inline_data"]
                payload = data.get("data", "")
                mime = data.get("mime_type", "") or image_io.detect_image_format(payload)
                block.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime,
                            "data": payload,
                        },
                    }
                )
            elif "text" in part:
                block.append({"type": "text", "text": part["text"]})
        if block:
            claude_role = "assistant" if role == "model" else role
            messages.append({"role": claude_role, "content": block})
    return messages


def to_minimax_messages(contents: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert Gemini-style ``contents`` to MiniMax content blocks.

    MiniMax exposes an Anthropic-compatible Messages API, so images are
    emitted as ``{"type": "image", "source": {...}}`` blocks with either a
    ``url`` or ``base64`` source, matching the Anthropic wire format.
    """

    messages: List[Dict[str, Any]] = []
    for item in contents:
        role = item.get("role", "user")
        parts = item.get("parts", []) or []
        block: List[Dict[str, Any]] = []
        for part in parts:
            if "image_url" in part:
                value = part["image_url"]
                url = value.get("url", "") if isinstance(value, dict) else str(value)
                if url:
                    block.append(
                        {"type": "image", "source": {"type": "url", "url": url}}
                    )
            elif "inline_data" in part:
                data = part["inline_data"]
                payload = data.get("data", "")
                if "base64," in payload:
                    payload = payload.split("base64,", 1)[1]
                mime = data.get("mime_type", "") or image_io.detect_image_format(
                    payload
                )
                block.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime,
                            "data": payload,
                        },
                    }
                )
            elif "text" in part:
                block.append({"type": "text", "text": part["text"]})
        if block:
            minimax_role = "assistant" if role == "model" else role
            messages.append({"role": minimax_role, "content": block})
    return messages


def _resolve_image_url_to_pil(url: str) -> Image.Image | None:
    """Try the network first, then fall back to ``FVQA_IMAGE_DIR``."""

    fetch_url = image_io.cos_url_to_internal(url)
    data = image_io.download_image_bytes(fetch_url)
    if data:
        try:
            return Image.open(io.BytesIO(data))
        except Exception as exc:
            logger.warning("Failed to decode downloaded image: %s", exc)

    if config.FVQA_IMAGE_DIR and os.path.isdir(config.FVQA_IMAGE_DIR):
        parsed = urllib.parse.urlparse(url)
        candidate = os.path.basename(parsed.path)
        if candidate:
            local = os.path.join(config.FVQA_IMAGE_DIR, candidate)
            if os.path.exists(local):
                try:
                    return Image.open(local)
                except Exception as exc:
                    logger.warning("Failed to read local image %s: %s", local, exc)
    return None


def to_qwen3vl_messages(contents: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert Gemini-style ``contents`` to Qwen3-VL chat messages."""

    messages: List[Dict[str, Any]] = []
    for item in contents:
        role = item.get("role", "user")
        parts = item.get("parts", []) or []
        block: List[Dict[str, Any]] = []
        for part in parts:
            if "inline_data" in part:
                data = part["inline_data"]
                payload = data.get("data", "")
                if "base64," in payload:
                    payload = payload.split("base64,", 1)[1]
                try:
                    pil_image = Image.open(io.BytesIO(base64.b64decode(payload)))
                    block.append({"type": "image", "image": pil_image})
                except Exception as exc:
                    logger.warning("Failed to decode inline base64 image: %s", exc)
                    block.append({"type": "image", "image": payload})
            elif "image_url" in part:
                value = part["image_url"]
                url = value.get("url", "") if isinstance(value, dict) else str(value)
                if not url:
                    continue
                pil_image = _resolve_image_url_to_pil(url)
                if pil_image is not None:
                    block.append({"type": "image", "image": pil_image})
            elif "text" in part:
                block.append({"type": "text", "text": part["text"]})
        if block:
            qwen_role = "assistant" if role == "model" else role
            messages.append({"role": qwen_role, "content": block})
    return messages
