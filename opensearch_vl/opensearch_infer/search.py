"""Search and document-parsing helpers used by the agent's tools.

Two providers are supported for ``text_search``:

1. **Gateway mode** (``API_HOST`` + ``API_USER`` + ``API_KEY``): a single
   gateway proxies Serper and Jina behind one HMAC credential.
2. **Direct mode** (``SERPER_API_KEY`` + ``JINA_API_KEY``): the public
   Serper / Jina endpoints are called directly.

Per-page summarisation is delegated to an OpenAI-compatible chat
completion endpoint serving ``QWEN_MODEL_NAME`` (typically Qwen3-32B).
``image_search`` requires an external visual lookup function
(historically named ``lens_scan``); when the function is not provided
the tool returns a clear, recoverable error message instead of raising.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional

import requests
from PIL import Image

from . import config


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layout parsing
# ---------------------------------------------------------------------------


def layout_parsing(
    file_path: str,
    use_chart_recognition: bool = False,
    use_doc_orientation_classify: bool = False,
    use_doc_unwarping: bool = False,
) -> Dict:
    """POST a local image to the configured layout-parsing endpoint."""

    if not config.LAYOUT_PARSING_API_URL:
        return {
            "error": (
                "Layout parsing endpoint is not configured. "
                "Set LAYOUT_PARSING_API_URL (and optionally LAYOUT_PARSING_TOKEN)."
            ),
            "rec_texts": [],
            "formatted_text": "",
            "blocks": [],
        }

    if not os.path.exists(file_path):
        return {
            "error": f"File not found: {file_path}",
            "rec_texts": [],
            "formatted_text": "",
            "blocks": [],
        }

    try:
        with open(file_path, "rb") as fh:
            file_data = base64.b64encode(fh.read()).decode("ascii")

        headers = {"Content-Type": "application/json"}
        if config.LAYOUT_PARSING_TOKEN:
            headers["Authorization"] = f"token {config.LAYOUT_PARSING_TOKEN}"

        payload = {
            "file": file_data,
            "fileType": 1,
            "useDocOrientationClassify": use_doc_orientation_classify,
            "useDocUnwarping": use_doc_unwarping,
            "useChartRecognition": use_chart_recognition,
        }

        response = requests.post(
            config.LAYOUT_PARSING_API_URL,
            json=payload,
            headers=headers,
            timeout=60,
        )
        if response.status_code != 200:
            return {
                "error": (
                    f"API request failed with status {response.status_code}: "
                    f"{response.text[:200]}"
                ),
                "rec_texts": [],
                "formatted_text": "",
                "blocks": [],
            }

        result = response.json().get("result", {})
        text_labels = {
            "abstract",
            "content",
            "doc_title",
            "figure_title",
            "paragraph_title",
            "reference",
            "reference_content",
            "table",
            "text",
            "vision_footnote",
        }
        blocks = (
            result.get("layoutParsingResults", [{}])[0]
            .get("prunedResult", {})
            .get("parsing_res_list", [])
        )
        texts = [
            blk["block_content"].strip()
            for blk in blocks
            if blk.get("block_label") in text_labels
            and blk.get("block_content", "").strip()
        ]
        return {
            "rec_texts": texts,
            "formatted_text": "\n".join(texts),
            "blocks": blocks,
            "error": None,
        }
    except Exception as exc:  # pragma: no cover - network-bound
        return {
            "error": f"Layout parsing error: {exc}",
            "rec_texts": [],
            "formatted_text": "",
            "blocks": [],
        }


# ---------------------------------------------------------------------------
# Summarisation backbone
# ---------------------------------------------------------------------------


def _summary_payload(prompt: str) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "model": config.QWEN_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0.3,
    }
    if config.LLM_EXTRA_BODY:
        payload.update(config.LLM_EXTRA_BODY)
    else:  # local vLLM-style server hosting Qwen
        payload["top_p"] = 0.95
        payload["extra_body"] = {
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    return payload


def _summary_headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if config.QWEN_API_KEY:
        headers["Authorization"] = f"Bearer {config.QWEN_API_KEY}"
    return headers


def summarize_with_qwen(content: str, query: str, title: str) -> str:
    """Generate a short, query-focused summary for a single page."""

    prompt = (
        f"Based on the following webpage content, provide a concise summary "
        f"that is relevant to the query: \"{query}\"\n\n"
        f"Webpage Title: {title}\n"
        f"Content:\n{content[:config.SUMMARY_CONTENT_MAX_CHARS]}\n\n"
        f"Please provide a focused summary (2-4 sentences) that directly "
        f"addresses the query. Focus on the most relevant information."
    )
    payload = _summary_payload(prompt)
    try:
        response = requests.post(
            f"{config.QWEN_API_BASE.rstrip('/')}/chat/completions",
            headers=_summary_headers(),
            json=payload,
            timeout=60,
        )
        if response.status_code == 200:
            data = response.json()
            choices = data.get("choices", [])
            if choices:
                summary = choices[0].get("message", {}).get("content", "")
                if summary:
                    return summary.strip()
    except Exception as exc:
        logger.warning("Qwen summarization failed: %s", exc)
    return content[:500] + ("..." if len(content) > 500 else "")


def summarize_image_search(result_obj: object) -> object:
    """Reduce a raw image-search payload to ``{title, source}`` records."""

    try:
        result_str = json.dumps(result_obj, ensure_ascii=False, indent=2)
    except Exception:
        result_str = str(result_obj)

    prompt = (
        "You are processing image search results. Extract and summarize only "
        "the relevant \"title\" and \"source\" information from the following "
        "image search results. Remove all irrelevant information and keep only "
        "the essential identification details.\n\n"
        f"Image Search Results:\n{result_str[:3000]}\n\n"
        "Please extract and return ONLY the relevant information in JSON "
        "format with \"title\" and \"source\" fields. If there are multiple "
        "results, return a list of objects, each with \"title\" and \"source\" "
        "fields. Remove any irrelevant details, descriptions, or metadata that "
        "are not directly related to identifying the object/entity."
    )
    payload = _summary_payload(prompt)
    try:
        response = requests.post(
            f"{config.QWEN_API_BASE.rstrip('/')}/chat/completions",
            headers=_summary_headers(),
            json=payload,
            timeout=60,
        )
        if response.status_code == 200:
            choices = response.json().get("choices", [])
            if choices:
                text = choices[0].get("message", {}).get("content", "")
                if text:
                    match = re.search(
                        r"```(?:json)?\s*(\{.*?\}|\[.*?\])", text, re.DOTALL
                    )
                    try:
                        if match:
                            return json.loads(match.group(1))
                        return json.loads(text.strip())
                    except json.JSONDecodeError:
                        return {"summary": text.strip()}
    except Exception as exc:
        logger.warning("Qwen image-search summarization failed: %s", exc)

    if isinstance(result_obj, dict):
        filtered: Dict[str, object] = {}
        for src_key, dst_key in (
            ("title", "title"),
            ("name", "title"),
            ("label", "title"),
            ("entity", "title"),
            ("source", "source"),
            ("url", "source"),
            ("link", "source"),
            ("reference", "source"),
        ):
            if src_key in result_obj and dst_key not in filtered:
                filtered[dst_key] = result_obj[src_key]
        return filtered or result_obj
    return result_obj


# ---------------------------------------------------------------------------
# Search providers
# ---------------------------------------------------------------------------


def _search_via_gateway(query: str, lang: str, top_k: int) -> List[dict]:
    headers = {
        "Authorization": f"Bearer {config.API_USER}:{config.API_KEY}?provider=serper&timeout=60",
        "Content-Type": "application/json",
    }
    body = {
        "q": query,
        "location": "United States",
        "hl": lang,
        "num": min(top_k, 20),
    }
    response = requests.post(
        f"{config.API_HOST.rstrip('/')}/search",
        headers=headers,
        json=body,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict) and data.get("code") not in (None, 0):
        raise RuntimeError(f"Gateway error: {data.get('msg', 'Unknown error')}")
    return data.get("organic", []) or []


def _search_via_serper(query: str, lang: str, top_k: int) -> List[dict]:
    if not config.SERPER_API_KEY:
        raise RuntimeError(
            "Serper is not configured. Either set API_HOST/API_USER/API_KEY "
            "for the gateway, or set SERPER_API_KEY for direct access."
        )
    headers = {
        "X-API-KEY": config.SERPER_API_KEY,
        "Content-Type": "application/json",
    }
    body = {
        "q": query,
        "hl": lang,
        "num": min(top_k, 20),
    }
    response = requests.post(
        config.SERPER_SEARCH_URL,
        headers=headers,
        json=body,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("organic", []) or []


def _read_via_gateway(url: str) -> str:
    headers = {
        "Authorization": f"Bearer {config.API_USER}:{config.API_KEY}?provider=jina_ai&timeout=60",
        "Content-Type": "application/json",
    }
    response = requests.post(
        f"{config.API_HOST.rstrip('/')}/images",
        headers=headers,
        json={"url": url},
        timeout=30,
    )
    if response.status_code != 200:
        return ""
    data = response.json()
    if isinstance(data, dict):
        if data.get("code") == 200:
            return data.get("data", {}).get("content", "")
        return data.get("content") or data.get("text") or data.get("markdown") or ""
    return ""


def _read_via_jina(url: str) -> str:
    headers = {"Accept": "application/json"}
    if config.JINA_API_KEY:
        headers["Authorization"] = f"Bearer {config.JINA_API_KEY}"
    response = requests.get(
        config.JINA_READER_URL.rstrip("/") + "/" + url,
        headers=headers,
        timeout=30,
    )
    if response.status_code != 200:
        return ""
    try:
        data = response.json()
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], dict):
                return data["data"].get("content", "")
            return data.get("content", "")
    except ValueError:
        return response.text
    return ""


# ---------------------------------------------------------------------------
# DeepSeek Search & Fetch
# ---------------------------------------------------------------------------

_RETRY_STATUS = {408, 429, 500, 502, 503, 504}
_DEEPSEEK_QUERY_MAX_CHARS = 128  # hard API limit


def _post_with_retries(url, headers, payload, attempts=5, timeout=60):
    """POST, retrying documented transient statuses with backoff."""
    last: Optional[requests.Response] = None
    for attempt in range(attempts):
        try:
            last = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if last.status_code not in _RETRY_STATUS:
                return last
        except requests.RequestException as exc:
            if attempt == attempts - 1:
                raise
            logger.debug("POST %s failed: %s", url, exc)
        time.sleep(2 * (attempt + 1))
    return last


def _deepseek_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {config.DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }


def _deepseek_fetch(url: str) -> str:
    """Page Markdown, or "" (Fetch returns 403/404 for e.g. Wikipedia, BBC)."""
    try:
        response = _post_with_retries(
            f"{config.DEEPSEEK_API_BASE}/v1/tools/fetch",
            _deepseek_headers(),
            {"url": url},
            attempts=3,
        )
    except requests.RequestException:
        return ""
    if response is None or response.status_code != 200:
        return ""
    return (response.json().get("content") or "").strip()


def _text_search_deepseek(query: str, top_k: int) -> str:
    """DeepSeek Search -> page text (search index, else Fetch) -> summaries."""
    top_k = max(1, min(int(top_k), 10))
    payload = {
        "query": query.strip()[:_DEEPSEEK_QUERY_MAX_CHARS],
        "count": top_k,
        "need_content": True,
    }
    try:
        response = _post_with_retries(
            f"{config.DEEPSEEK_API_BASE}/v1/tools/search", _deepseek_headers(), payload
        )
    except requests.RequestException as exc:
        return f"Tool execution error:\nText search failed: {exc}"
    if response is None or response.status_code != 200:
        code = None if response is None else response.status_code
        return f"Tool execution error:\nText search failed: HTTP {code}"
    results = (response.json().get("results") or [])[:top_k]
    if not results:
        return "Tool execution result:\nNo relevant web pages found for the query."

    def one(item: dict) -> str:
        url, title = item.get("url", ""), item.get("title", "")
        content = (item.get("content") or "").strip()
        if not content and url:
            content = _deepseek_fetch(url)
        if not content:
            content = item.get("summary") or item.get("snippet") or ""
        return summarize_with_qwen(content=content, query=query, title=title)

    with ThreadPoolExecutor(max_workers=len(results)) as pool:
        summaries = list(pool.map(one, results))

    out = "Tool execution result:\n\n"
    for idx, (item, summary) in enumerate(zip(results, summaries), start=1):
        out += (
            f"\n{'=' * 60}[Passage {idx}]\n"
            f"Title: {item.get('title', '')}\n"
            f"URL: {item.get('url', '')}\n"
            f"Summary:\n{summary}\n"
        )
    return out


# ---------------------------------------------------------------------------
# 豆包搜索 Global (reverse image search)
# ---------------------------------------------------------------------------

_DOUBAO_RETRY_CODES = {700429, 10500, 10501}


def _encode_for_visual_search(image_bytes: bytes) -> bytes:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    edge = config.IMAGE_SEARCH_MAX_EDGE
    if max(image.size) > edge:
        image.thumbnail((edge, edge), Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def image_search_bytes(image_bytes: bytes, attempts: int = 4) -> str:
    """Visual search by image content; no public URL or object storage needed."""
    try:
        encoded = _encode_for_visual_search(image_bytes)
    except Exception as exc:
        return f"Tool execution error:\nCannot decode image: {exc}"
    payload = {
        "SearchType": "visual",
        "ImageQuery": {"ImageBase64": base64.b64encode(encoded).decode("ascii")},
        "DocCount": max(1, min(config.IMAGE_SEARCH_DOC_COUNT, 20)),
        "EnableWaiting": True,  # queue server-side when over the QPS limit
        "MaxWaitTime": 10000,
    }
    headers = {
        "Authorization": f"Bearer {config.DOUBAO_SEARCH_API_KEY}",
        "Content-Type": "application/json",
    }
    code: object = None
    for attempt in range(attempts):
        try:
            response = requests.post(
                config.DOUBAO_SEARCH_URL, headers=headers, json=payload, timeout=60
            )
            data = response.json()
        except (requests.RequestException, ValueError) as exc:
            code = type(exc).__name__
        else:
            result = data.get("Result") or {}
            error = (data.get("ResponseMetadata") or {}).get("Error")
            code = result.get("ErrorCode") or (error or {}).get("Code")
            if response.status_code == 200 and not code:
                docs = [
                    {
                        "title": (doc.get("Title") or "").strip(),
                        "source": (doc.get("HostInfo") or {}).get("Hostname") or "",
                        "url": doc.get("Url", ""),
                    }
                    for doc in result.get("Documents") or []
                ]
                if not docs:
                    return "Tool execution result:\nNo visually similar results found."
                summary = summarize_image_search({"organic": docs})
                return "Tool execution result:\n" + json.dumps(
                    summary, ensure_ascii=False, indent=2
                )
            code = code or response.status_code
            if code not in _DOUBAO_RETRY_CODES and response.status_code < 500:
                break
        time.sleep(2 * (attempt + 1))
    # Only the code: provider messages may contain agent-directed instructions.
    return f"Tool execution error:\nimage_search failed (code {code})"


def visual_backend_enabled() -> bool:
    return bool(config.DOUBAO_SEARCH_API_KEY)


def text_search(query: str, lang: str = "en", top_k: int = 5) -> str:
    """Run a web search, read the top pages and summarise them."""

    if config.DEEPSEEK_API_KEY:
        return _text_search_deepseek(query, top_k)

    use_gateway = config.gateway_enabled()
    try:
        if use_gateway:
            organic = _search_via_gateway(query, lang, top_k)
        else:
            organic = _search_via_serper(query, lang, top_k)
    except Exception as exc:
        return f"Tool execution error:\nText search failed: {exc}"

    if not organic:
        return "Tool execution result:\nNo relevant web pages found for the query."

    pages = []
    for idx, item in enumerate(organic[:top_k], start=1):
        url = item.get("link", "")
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        content = snippet
        if url:
            try:
                if use_gateway:
                    fetched = _read_via_gateway(url)
                else:
                    fetched = _read_via_jina(url)
                if fetched:
                    content = fetched
            except Exception as exc:
                logger.debug("Jina fetch failed for %s: %s", url, exc)
        pages.append({"index": idx, "url": url, "title": title, "content": content})

    formatted = []
    for page in pages:
        summary = summarize_with_qwen(
            content=page["content"], query=query, title=page["title"]
        )
        formatted.append(
            f"[Passage {page['index']}]\n"
            f"Title: {page['title']}\n"
            f"URL: {page['url']}\n"
            f"Summary:\n{summary}"
        )

    body = ("\n\n" + "=" * 60 + "\n\n").join(formatted)
    return f"Tool execution result:\n{body}"


def image_search(
    image_url: str,
    visual_lookup: Optional[Callable[..., object]] = None,
    max_retries: int = 3,
    base_delay: int = 2,
) -> str:
    """Run an external visual lookup against ``image_url`` and summarise it."""

    if not visual_lookup:
        return (
            "Tool execution error:\n"
            "image_search requires a visual lookup callable. Configure one via "
            "the runner (e.g. an external lens / similar-image API) before "
            "invoking image_search."
        )

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            result = visual_lookup(image_url=image_url)
            if isinstance(result, dict) and "error" in result:
                raise RuntimeError(result["error"])
            summarised = summarize_image_search(result)
            payload = (
                json.dumps(summarised, ensure_ascii=False, indent=2)
                if isinstance(summarised, (dict, list))
                else str(summarised)
            )
            return f"Tool execution result:\n{payload}"
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(base_delay * (2 ** (attempt - 1)))
    return f"Tool execution error:\nimage_search failed after {max_retries} retries: {last_error}"
