"""HMAC helpers and the optional Claude gateway client."""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import uuid
from datetime import datetime
from typing import Iterable, List, Mapping, Optional

import requests

from . import config


logger = logging.getLogger(__name__)


def hmac_sha1_auth(
    source: str, secret_id: str, secret_key: str
) -> tuple[str, str]:
    """Build an ``Authorization`` header value plus its matching ``Date``.

    Returns ``(authorization, http_date)``.
    """

    http_date = datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")
    sign_str = f"date: {http_date}\nsource: {source}"
    signature_bytes = hmac.new(
        secret_key.encode(), sign_str.encode(), hashlib.sha1
    ).digest()
    signature = base64.b64encode(signature_bytes).decode()
    auth = (
        f'hmac id="{secret_id}", algorithm="hmac-sha1", '
        f'headers="date source", signature="{signature}"'
    )
    return auth, http_date


class ClaudeGatewayClient:
    """Thin client for an HMAC-secured ``/api/v1/data_eval`` Claude gateway."""

    def __init__(
        self,
        host: str,
        user: str,
        api_key: str,
        source: Optional[str] = None,
        api_version: Optional[str] = None,
        model_marker: Optional[str] = None,
        timeout: int = 3600,
    ) -> None:
        if not (host and user and api_key):
            raise ValueError(
                "ClaudeGatewayClient requires host, user and api_key. "
                "Set CLAUDE_API_HOST / CLAUDE_API_USER / CLAUDE_API_KEY."
            )
        self.host = host.rstrip("/")
        self.user = user
        self.api_key = api_key
        self.source = source or user
        self.api_version = api_version or config.CLAUDE_API_VERSION
        self.model_marker = model_marker or config.CLAUDE_MODEL_MARKER
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "ClaudeGatewayClient":
        """Build a client from the standard ``CLAUDE_API_*`` env vars."""

        return cls(
            host=config.CLAUDE_API_HOST,
            user=config.CLAUDE_API_USER,
            api_key=config.CLAUDE_API_KEY,
            source=config.CLAUDE_API_SOURCE,
        )

    def _headers(self) -> dict:
        auth, http_date = hmac_sha1_auth(self.source, self.user, self.api_key)
        return {
            "Apiversion": self.api_version,
            "Authorization": auth,
            "Date": http_date,
            "Source": self.source,
            "Content-Type": "application/json",
        }

    @staticmethod
    def _normalize_messages(messages: Iterable[Mapping]) -> List[dict]:
        """Translate Claude content blocks into the gateway's wire format."""

        normalized: List[dict] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", [])
            api_content: List[dict] = []
            for item in content:
                kind = item.get("type")
                if kind == "text":
                    api_content.append(
                        {"type": "text", "value": item.get("text", "")}
                    )
                elif kind == "image_url":
                    url = item.get("value")
                    if isinstance(url, dict):
                        url = url.get("url", "")
                    api_content.append({"type": "image_url", "value": url})
                elif kind == "image":
                    src = item.get("source", {})
                    if src.get("type") == "base64":
                        media_type = src.get("media_type", "image/jpeg")
                        data = src.get("data", "")
                        api_content.append(
                            {
                                "type": "image_url",
                                "value": f"data:{media_type};base64,{data}",
                            }
                        )
            if api_content:
                normalized.append({"role": role, "content": api_content})
        return normalized

    def call(
        self,
        messages: Iterable[Mapping],
        system_instruction: Optional[str] = None,
        max_tokens: int = 32768,
        timeout: Optional[int] = None,
    ) -> requests.Response:
        """POST to ``/api/v1/data_eval`` and return the raw response."""

        body = {
            "request_id": str(uuid.uuid4()),
            "model_marker": self.model_marker,
            "messages": self._normalize_messages(messages),
            "params": {"max_tokens": max_tokens},
            "timeout": timeout if timeout is not None else self.timeout,
        }
        if system_instruction:
            body["params"]["system"] = system_instruction

        url = f"{self.host}/api/v1/data_eval"
        headers = self._headers()
        try:
            response = requests.post(
                url, headers=headers, json=body, timeout=body["timeout"]
            )
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            logger.error("Claude gateway request failed: %s", exc)
            raise


class MiniMaxClient:
    """Thin client for the MiniMax Anthropic-compatible Messages API.

    The endpoint mirrors the Anthropic ``/v1/messages`` contract and is
    authenticated with a Bearer token. Both the global (``api.minimax.io``)
    and China (``api.minimaxi.com``) hosts share the same wire format; the
    base URL is selected via :mod:`config`.
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        timeout: int = 3600,
    ) -> None:
        if not api_key:
            raise ValueError(
                "MiniMaxClient requires an API key. Set MINIMAX_API_KEY."
            )
        self.api_key = api_key
        self.base_url = (base_url or config.MINIMAX_ANTHROPIC_BASE_URL).rstrip("/")
        self.api_version = api_version or config.MINIMAX_API_VERSION
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "MiniMaxClient":
        """Build a client from the standard ``MINIMAX_*`` env vars."""

        return cls(
            api_key=config.MINIMAX_API_KEY,
            base_url=config.MINIMAX_ANTHROPIC_BASE_URL,
            api_version=config.MINIMAX_API_VERSION,
        )

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "anthropic-version": self.api_version,
            "Content-Type": "application/json",
        }

    def call(
        self,
        model: str,
        messages: Iterable[Mapping],
        system_instruction: Optional[str] = None,
        max_tokens: int = 32768,
        timeout: Optional[int] = None,
    ) -> requests.Response:
        """POST to ``/v1/messages`` and return the raw response."""

        body: dict = {
            "model": model,
            "messages": list(messages),
            "max_tokens": max_tokens,
        }
        if system_instruction:
            body["system"] = system_instruction

        url = f"{self.base_url}/v1/messages"
        request_timeout = timeout if timeout is not None else self.timeout
        try:
            response = requests.post(
                url, headers=self._headers(), json=body, timeout=request_timeout
            )
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            logger.error("MiniMax request failed: %s", exc)
            raise
