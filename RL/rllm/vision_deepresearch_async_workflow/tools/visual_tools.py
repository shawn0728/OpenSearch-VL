"""
Visual Investigation Tools – ported from qwen_search_fvqa.py.

Provides: crop, layout_parsing, text_search, image_search,
          web_search, perspective_correct, super_resolution, sharpen.

All tools follow the async DeepResearchTool interface.
"""

import asyncio
import base64
import hashlib
import io
import json
import os
import re
import tempfile
import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import requests
from PIL import Image

from vision_deepresearch_async_workflow.tools.shared import (
    DeepResearchTool,
    get_cache_async,
    get_cache_key,
    log_tool_event,
    run_with_retries_async,
    set_cache_async,
)

# ---------------------------------------------------------------------------
# Optional dependency: OpenCV (for perspective_correct, super_resolution, sharpen)
# ---------------------------------------------------------------------------
try:
    import cv2
    import numpy as np

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# ---------------------------------------------------------------------------
# Optional dependency: COS upload
# ---------------------------------------------------------------------------
UPLOAD_AVAILABLE = False
upload_cos = None

# Search for an optional ``upload.py`` module (COS uploader). By default we
# only look at the project root; additional search roots can be injected via
# the ``COS_UPLOAD_PATHS`` env var (colon-separated).
_upload_paths = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."),
]
_extra_paths = os.getenv("COS_UPLOAD_PATHS", "")
if _extra_paths:
    _upload_paths.extend(p for p in _extra_paths.split(":") if p)

for _p in _upload_paths:
    _ufile = os.path.join(_p, "upload.py")
    if not os.path.exists(_ufile):
        continue
    try:
        import importlib.util

        _spec = importlib.util.spec_from_file_location("upload", _ufile)
        if _spec and _spec.loader:
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            upload_cos = _mod.upload_cos
            if callable(upload_cos):
                UPLOAD_AVAILABLE = True
                print(f"[visual_tools] COS upload loaded from {_ufile}")
                break
    except Exception as _e:
        print(f"[visual_tools] Failed to load upload.py from {_ufile}: {_e}")
        continue

if not UPLOAD_AVAILABLE:
    print("[visual_tools] WARNING: COS upload not available. image_search will not work for local images.")
    def upload_cos(*args, **kwargs):
        return None, None

# ---------------------------------------------------------------------------
# Layout Parsing API config
# ---------------------------------------------------------------------------
LAYOUT_PARSING_API_URL = os.getenv("LAYOUT_PARSING_API_URL", "")
LAYOUT_PARSING_TOKEN = os.getenv("LAYOUT_PARSING_TOKEN", "")

# ---------------------------------------------------------------------------
# Optional proxy gateway / Serper / JINA config (read from env or .env)
# ---------------------------------------------------------------------------
API_HOST = os.getenv("API_GATEWAY_HOST", "")
API_USER = os.getenv("API_GATEWAY_USER", "")
API_KEY = os.getenv("API_GATEWAY_KEY", "")

SERP_SEARCH_URL = os.getenv("TEXT_SEARCH_URL", "https://google.serper.dev/search")

# ---------------------------------------------------------------------------
# DeepSeek Search & Fetch (backs text_search / web_search)
# Docs: https://api-docs.deepseek.com/guides/search/
# ---------------------------------------------------------------------------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_QUERY_MAX_CHARS = 128          # hard API limit on `query`
DEEPSEEK_RETRY_STATUS = {408, 429, 500, 502, 503, 504}

# Summariser endpoint (OpenAI-compatible; name kept from upstream Qwen3-32B setup)
QWEN_API_BASE = os.getenv("QWEN_API_BASE", "").rstrip("/")
QWEN_MODEL = os.getenv("QWEN_MODEL", "")
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
SUMMARY_CONTENT_MAX_CHARS = int(os.getenv("SUMMARY_CONTENT_MAX_CHARS", "6000"))
# Extra provider-specific chat body, e.g. '{"thinking": {"type": "disabled"}}'
# for DeepSeek, whose default thinking mode can spend max_tokens on reasoning.
try:
    LLM_EXTRA_BODY = json.loads(os.getenv("LLM_EXTRA_BODY_JSON", "") or "{}")
except json.JSONDecodeError:
    print("[visual_tools] WARNING: LLM_EXTRA_BODY_JSON is not valid JSON; ignored")
    LLM_EXTRA_BODY = {}

# ---------------------------------------------------------------------------
# env.py lens_scan / web_search (Polaris)
# ---------------------------------------------------------------------------
_env_lens_scan = None
_env_web_search = None

try:
    import sys
    _env_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."
    )
    _search_paths = [_env_dir]
    _extra_env = os.getenv("VDR_EXTRA_ENV_PATHS", "")
    if _extra_env:
        _search_paths.extend(p for p in _extra_env.split(":") if p)
    for _try_path in _search_paths:
        try:
            if _try_path not in sys.path:
                sys.path.insert(0, _try_path)
            from env import web_search as _ws, lens_scan as _ls  # type: ignore
            _env_web_search = _ws
            _env_lens_scan = _ls
            break
        except Exception:
            continue
except Exception:
    pass


# ===================================================================
# Helper: upload PIL image to COS
# ===================================================================

def _upload_pil_to_cos(
    pil_img: Image.Image,
    tool_name: str,
    userid: str = os.getenv("COS_USERID", "opensearch-vl"),
) -> Optional[str]:
    if not UPLOAD_AVAILABLE:
        print(f"[visual_tools] _upload_pil_to_cos: COS upload not available (UPLOAD_AVAILABLE=False)")
        return None
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            pil_img.save(tmp.name, format="PNG")
            tmp_path = tmp.name
        date_str = datetime.now().strftime("%Y-%m-%d")
        ts = int(time.time() * 1000)
        filename = f"vdr_{tool_name}_{ts}.png"
        cos_key, cos_url = upload_cos(
            tmp_path, filename, date_str,
            f"vdr_{date_str}", userid, use_direct_url=True,
        )
        if cos_url:
            print(f"[visual_tools] Uploaded to COS: {cos_url}")
        else:
            print(f"[visual_tools] _upload_pil_to_cos: upload_cos returned None for {filename}")
        return cos_url or None
    except Exception as exc:
        print(f"[visual_tools] _upload_pil_to_cos failed: {type(exc).__name__}: {exc}")
        return None
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


# ===================================================================
# Helper: download image from URL
# ===================================================================

def _download_image_bytes(url: str, timeout: int = 60, retries: int = 3) -> Optional[bytes]:
    for attempt in range(retries):
        try:
            # Many image hosts (e.g. Wikimedia) reject the default python-requests UA.
            resp = requests.get(
                url, timeout=(timeout, timeout), stream=True,
                headers={"User-Agent": os.getenv(
                    "TOOL_HTTP_USER_AGENT", "Mozilla/5.0 (compatible; ImagentRL-tools/0.1)")},
            )
            resp.raise_for_status()
            return resp.content
        except Exception:
            if attempt < retries - 1:
                time.sleep((attempt + 1) * 2)
    return None


# ===================================================================
# Helper: resolve image reference → local path
# ===================================================================

def _resolve_image_ref(
    image_ref: str,
    image_paths: dict,
    intermediate_dir: str,
    tool_name: str,
) -> Optional[str]:
    """Return a local file path for *image_ref* (downloading if needed).

    Handles: local path str, URL str, bytes, PIL.Image, HuggingFace dict
    ({"bytes": ..., "path": ...}), and base64 data URIs.
    """
    if image_ref not in image_paths:
        return None
    data = image_paths[image_ref]

    # Already a local file path
    if isinstance(data, str) and len(data) < 500 and os.path.exists(data):
        return data

    # URL → download
    if isinstance(data, str) and data.startswith(("http://", "https://")):
        img_bytes = _download_image_bytes(data)
        if img_bytes is None:
            return None
        os.makedirs(intermediate_dir, exist_ok=True)
        ts = int(time.time() * 1000)
        local = os.path.join(intermediate_dir, f"vdr_{tool_name}_{ts}.png")
        with open(local, "wb") as f:
            f.write(img_bytes)
        image_paths[image_ref] = local
        return local

    # Base64 data URI → decode
    if isinstance(data, str) and data.startswith("data:image/"):
        try:
            _, encoded = data.split(",", 1)
            raw = base64.b64decode(encoded)
            os.makedirs(intermediate_dir, exist_ok=True)
            ts = int(time.time() * 1000)
            local = os.path.join(intermediate_dir, f"vdr_{tool_name}_{ts}.png")
            Image.open(io.BytesIO(raw)).save(local)
            image_paths[image_ref] = local
            return local
        except Exception:
            pass

    # Raw bytes
    if isinstance(data, bytes):
        os.makedirs(intermediate_dir, exist_ok=True)
        ts = int(time.time() * 1000)
        local = os.path.join(intermediate_dir, f"vdr_{tool_name}_{ts}.png")
        with open(local, "wb") as f:
            f.write(data)
        image_paths[image_ref] = local
        return local

    # PIL Image
    if isinstance(data, Image.Image):
        os.makedirs(intermediate_dir, exist_ok=True)
        ts = int(time.time() * 1000)
        local = os.path.join(intermediate_dir, f"vdr_{tool_name}_{ts}.png")
        data.save(local)
        image_paths[image_ref] = local
        return local

    # HuggingFace dict format: {"bytes": b"...", "path": "..."}
    if isinstance(data, dict):
        pil = None
        if "bytes" in data and data["bytes"] is not None:
            try:
                pil = Image.open(io.BytesIO(data["bytes"])).convert("RGB")
            except Exception:
                pass
        if pil is None and "path" in data and isinstance(data["path"], str):
            try:
                pil = Image.open(data["path"]).convert("RGB")
            except Exception:
                pass
        if pil is None:
            d_str = data.get("data") or data.get("url") or ""
            if isinstance(d_str, str) and d_str.startswith("data:image/"):
                try:
                    _, encoded = d_str.split(",", 1)
                    pil = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
                except Exception:
                    pass
        if pil is not None:
            os.makedirs(intermediate_dir, exist_ok=True)
            ts = int(time.time() * 1000)
            local = os.path.join(intermediate_dir, f"vdr_{tool_name}_{ts}.png")
            pil.save(local)
            image_paths[image_ref] = local
            return local

    print(f"[visual_tools] _resolve_image_ref: unhandled data type {type(data).__name__} for {image_ref}")
    return None


# ===================================================================
# Helper: ensure an image ref becomes a public URL (upload if needed)
# ===================================================================

def _ensure_image_url(
    image_ref: str,
    image_paths: dict,
    intermediate_dir: str,
    tool_name: str,
) -> Optional[str]:
    """
    Resolve *image_ref* to a public URL suitable for APIs that require one.

    Priority:
      1. Already a URL string in image_paths → return directly
      2. Local file / PIL / bytes → upload to COS → return URL
      3. image_ref itself looks like a URL → return as-is

    On success the URL is cached back into image_paths[image_ref] so
    subsequent calls skip the upload.
    """
    if image_ref in image_paths:
        data = image_paths[image_ref]
        # Already a URL
        if isinstance(data, str) and data.startswith(("http://", "https://")):
            return data

        # Resolve to a local file first
        local = _resolve_image_ref(image_ref, image_paths, intermediate_dir, tool_name)
        if local:
            try:
                pil = Image.open(local)
                cos_url = _upload_pil_to_cos(pil, tool_name)
                if cos_url:
                    image_paths[image_ref] = cos_url
                    return cos_url
            except Exception as exc:
                print(f"[visual_tools] _ensure_image_url: failed to upload {image_ref}: {exc}")
        return None

    # image_ref is not a known key — maybe it's already a raw URL
    if isinstance(image_ref, str) and image_ref.startswith(("http://", "https://")):
        return image_ref

    return None


# ===================================================================
# Layout Parsing standalone function
# ===================================================================

def layout_parsing(
    file_path: str,
    use_chart_recognition: bool = False,
    use_doc_orientation_classify: bool = False,
) -> dict:
    """Call the remote Layout Parsing API and return structured result."""
    try:
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}", "rec_texts": [], "formatted_text": "", "blocks": []}
        with open(file_path, "rb") as fh:
            file_data = base64.b64encode(fh.read()).decode("ascii")

        headers = {
            "Authorization": f"token {LAYOUT_PARSING_TOKEN}",
            "Content-Type": "application/json",
        }
        payload = {
            "file": file_data,
            "fileType": 1,
            "useDocOrientationClassify": use_doc_orientation_classify,
            "useDocUnwarping": False,
            "useChartRecognition": use_chart_recognition,
        }
        resp = requests.post(LAYOUT_PARSING_API_URL, json=payload, headers=headers, timeout=60)
        if resp.status_code != 200:
            return {"error": f"API HTTP {resp.status_code}: {resp.text[:200]}", "rec_texts": [], "formatted_text": "", "blocks": []}

        result = resp.json().get("result", {})
        TEXT_LABELS = {
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
            if blk.get("block_label") in TEXT_LABELS and blk.get("block_content", "").strip()
        ]
        return {"rec_texts": texts, "formatted_text": "\n".join(texts), "blocks": blocks, "error": None}
    except Exception as exc:
        return {"error": str(exc), "rec_texts": [], "formatted_text": "", "blocks": []}


# ===================================================================
# ImageEnhancementEngine (perspective_correct, super_resolution, sharpen)
# ===================================================================

SR_MODEL_PATH = os.getenv("SR_MODEL_PATH", "EDSR_x4.pb")
SR_MAX_INPUT_EDGE = int(os.getenv("SR_MAX_INPUT_EDGE", "384"))
_sr_local = threading.local()


def _get_sr_model(model_path: str):
    """Per-thread cached EDSR x4 net (cv2.dnn nets are not safe to share across threads)."""
    cached = getattr(_sr_local, "models", None)
    if cached is None:
        cached = _sr_local.models = {}
    if model_path not in cached:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"EDSR weights not found at {model_path!r} (set SR_MODEL_PATH)"
            )
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(model_path)
        sr.setModel("edsr", 4)
        cached[model_path] = sr
    return cached[model_path]


class ImageEnhancementEngine:
    """OpenCV-based image enhancement."""

    def __init__(self):
        self.current_image = None

    def load_from_path(self, path: str):
        self.current_image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
        return self

    def load_from_pil(self, pil_img: Image.Image):
        self.current_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return self

    # -- perspective correct --
    def auto_correct_perspective(self):
        if self.current_image is None:
            return None
        if len(self.current_image.shape) == 3 and self.current_image.shape[2] == 4:
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)
        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
        screen_cnt = None
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                screen_cnt = approx
                break
        if screen_cnt is None:
            return self.current_image
        pts = screen_cnt.reshape(4, 2).astype("float32")
        s = pts.sum(axis=1)
        rect = np.zeros((4, 2), dtype="float32")
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        (tl, tr, br, bl) = rect
        wA = np.linalg.norm(br - bl)
        wB = np.linalg.norm(tr - tl)
        maxW = max(int(wA), int(wB))
        hA = np.linalg.norm(tr - br)
        hB = np.linalg.norm(tl - bl)
        maxH = max(int(hA), int(hB))
        dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        self.current_image = cv2.warpPerspective(self.current_image, M, (maxW, maxH))
        return self.current_image

    # -- super resolution --
    def apply_super_resolution(self, model_path: str = "", scale: int = 4):
        """EDSR x4 on CPU. Raises on failure so callers never report a no-op as success.

        Input is first downscaled to SR_MAX_INPUT_EDGE (CPU cost grows with
        pixel count: ~4s for 256x256, ~17s for 512x512 on 24 cores). Only x4
        weights are shipped, so scale 2/3 are produced by x4 + area downscale.
        """
        if self.current_image is None:
            return None
        if len(self.current_image.shape) == 2:
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_GRAY2BGR)
        elif self.current_image.shape[2] == 4:
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGRA2BGR)
        scale = scale if scale in (2, 3, 4) else 4
        h, w = self.current_image.shape[:2]
        if max(h, w) > SR_MAX_INPUT_EDGE:
            f = SR_MAX_INPUT_EDGE / max(h, w)
            self.current_image = cv2.resize(
                self.current_image, (max(1, int(w * f)), max(1, int(h * f))),
                interpolation=cv2.INTER_AREA,
            )
            h, w = self.current_image.shape[:2]
        sr = _get_sr_model(model_path or SR_MODEL_PATH)
        out = sr.upsample(self.current_image)
        if scale != 4:
            out = cv2.resize(out, (w * scale, h * scale), interpolation=cv2.INTER_AREA)
        self.current_image = out
        return self.current_image

    # -- sharpen --
    def enhance_sharpness(self, amount: float = 1.5):
        if self.current_image is None:
            return None
        blurred = cv2.GaussianBlur(self.current_image, (0, 0), 3)
        self.current_image = cv2.addWeighted(self.current_image, 1.0 + amount, blurred, -amount, 0)
        return self.current_image

    def to_pil(self) -> Optional[Image.Image]:
        if self.current_image is None:
            return None
        return Image.fromarray(cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB))


# ===================================================================
# Qwen summariser helper
# ===================================================================

def _summarize_with_qwen(content: str, query: str, title: str) -> str:
    """Use an external Qwen model to summarise web-page content."""
    if not QWEN_API_BASE or not QWEN_MODEL:
        return content[:500] + ("..." if len(content) > 500 else "")
    try:
        prompt = (
            f'Based on the following webpage content, provide a concise summary '
            f'relevant to the query: "{query}"\n\n'
            f'Webpage Title: {title}\nContent:\n{content[:SUMMARY_CONTENT_MAX_CHARS]}\n\n'
            f'Provide a focused summary (2-4 sentences) that directly addresses the query.'
        )
        payload = {
            "model": QWEN_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "temperature": 0.3,
            **LLM_EXTRA_BODY,
        }
        headers = {"Content-Type": "application/json"}
        if QWEN_API_KEY:
            headers["Authorization"] = f"Bearer {QWEN_API_KEY}"
        resp = requests.post(
            f"{QWEN_API_BASE}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        if resp.status_code == 200:
            choices = resp.json().get("choices", [])
            if choices:
                text = choices[0].get("message", {}).get("content", "")
                if text:
                    return text.strip()
    except Exception:
        pass
    return content[:500] + ("..." if len(content) > 500 else "")


# ===================================================================
# Tool classes
# ===================================================================

class CropTool(DeepResearchTool):
    """Crop a specific region from an image."""

    def __init__(self):
        super().__init__(
            name="crop",
            description="Crop a specific region from an image.",
            parameters={
                "type": "object",
                "properties": {
                    "image": {"type": "string", "description": "Image reference (e.g., 'img_1')"},
                    "x": {"type": "integer", "description": "Starting X coordinate"},
                    "y": {"type": "integer", "description": "Starting Y coordinate"},
                    "width": {"type": "integer", "description": "Width of the crop region"},
                    "height": {"type": "integer", "description": "Height of the crop region"},
                },
                "required": ["image", "x", "y", "width", "height"],
            },
        )

    async def call(self, image: str = "", x: int = 0, y: int = 0,
                   width: int = 0, height: int = 0, **ctx) -> str:
        image_paths: dict = ctx.get("image_paths", {})
        intermediate_dir: str = ctx.get("intermediate_dir", "/tmp/vdr_tools")

        if not image or image not in image_paths:
            return f"Error: Image reference '{image}' not found. Available: {list(image_paths.keys())}"
        if width <= 0 or height <= 0:
            return "Error: width and height must be positive integers"

        local = _resolve_image_ref(image, image_paths, intermediate_dir, "crop")
        if not local:
            return f"Error: Failed to resolve image for {image}"

        try:
            pil_img = Image.open(local)
            cropped = pil_img.crop((x, y, x + width, y + height))
            new_id = f"img_{len(image_paths) + 1}"

            cos_url = _upload_pil_to_cos(cropped, "crop")
            if cos_url:
                image_paths[new_id] = cos_url
                return f"Image cropped successfully. New image ID: {new_id}. Uploaded to: {cos_url}"

            os.makedirs(intermediate_dir, exist_ok=True)
            save = os.path.join(intermediate_dir, f"vdr_crop_{int(time.time()*1000)}.png")
            cropped.save(save)
            image_paths[new_id] = save
            return f"Image cropped successfully. New image ID: {new_id}. Saved to: {save}"
        except Exception as exc:
            return f"Error executing crop: {exc}"


class LayoutParsingTool(DeepResearchTool):
    """Document layout parsing and text extraction via remote API."""

    def __init__(self):
        super().__init__(
            name="layout_parsing",
            description="Perform document layout parsing on an image to extract structured text.",
            parameters={
                "type": "object",
                "properties": {
                    "image": {"type": "string", "description": "Image reference (e.g., 'img_1')"},
                    "file_path": {"type": "string", "description": "Absolute path to an image file (optional)."},
                    "use_chart_recognition": {"type": "boolean", "default": False},
                    "use_doc_orientation_classify": {"type": "boolean", "default": False},
                },
                "required": [],
            },
        )

    async def call(self, image: str = "", file_path: str = "",
                   use_chart_recognition: bool = False,
                   use_doc_orientation_classify: bool = False, **ctx) -> str:
        image_paths: dict = ctx.get("image_paths", {})
        intermediate_dir: str = ctx.get("intermediate_dir", "/tmp/vdr_tools")

        if not file_path and image:
            local = _resolve_image_ref(image, image_paths, intermediate_dir, "layout_parsing")
            if local:
                file_path = local

        if not file_path:
            return "Error: 'file_path' or 'image' parameter is required for layout_parsing"

        result = await asyncio.get_event_loop().run_in_executor(
            None, layout_parsing, file_path, use_chart_recognition, use_doc_orientation_classify,
        )

        if result.get("error"):
            return f"Tool execution error:\n{result['error']}"

        rec_texts = result.get("rec_texts", [])
        formatted = result.get("formatted_text", "")
        parts = []
        if rec_texts:
            parts.append(f"Layout Parsing SUCCESS: {len(rec_texts)} text blocks detected.")
            parts.append("")
            for i, t in enumerate(rec_texts, 1):
                parts.append(f"[Text Block {i}] {t}")
            parts.append("")
            parts.append("=" * 50)
            parts.append("ALL RECOGNIZED TEXT:")
            parts.append(formatted)
            parts.append("=" * 50)
        else:
            parts.append("Layout Parsing: No text blocks detected.")
        return "\n".join(parts)


# ===================================================================
# DeepSeek Search / Fetch helpers
# ===================================================================

class _RetryableHTTPError(Exception):
    """Raised for HTTP statuses DeepSeek documents as retryable (408/429/5xx)."""


def _deepseek_headers() -> dict:
    return {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }


async def _deepseek_search(
    query: str,
    count: int,
    need_content: bool,
    proxies: Optional[dict],
    executor=None,
) -> Tuple[Optional[list], Optional[str]]:
    """POST /v1/tools/search. Returns (results, error). Cached by request."""
    if not DEEPSEEK_API_KEY:
        return None, "DEEPSEEK_API_KEY is not set"
    query = query.strip()[:DEEPSEEK_QUERY_MAX_CHARS]
    count = max(1, min(int(count), 50))
    payload = {"query": query, "count": count, "need_content": bool(need_content)}
    cache_input = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    cache_key = get_cache_key("deepseek_search:" + cache_input)

    cached = await get_cache_async("text_search", cache_key)
    if cached:
        try:
            return json.loads(cached), None
        except Exception:
            pass

    def _do():
        resp = requests.post(
            f"{DEEPSEEK_API_BASE}/v1/tools/search",
            headers=_deepseek_headers(), json=payload, timeout=60, proxies=proxies,
        )
        if resp.status_code in DEEPSEEK_RETRY_STATUS:
            raise _RetryableHTTPError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        return resp

    try:
        resp = await run_with_retries_async(_do, attempts=5, delay=2, executor=executor)
    except Exception as exc:
        return None, f"DeepSeek search failed: {exc}"
    if resp.status_code != 200:
        return None, f"DeepSeek search error: HTTP {resp.status_code}: {resp.text[:200]}"
    results = resp.json().get("results", []) or []
    if results:
        await set_cache_async(
            "text_search", cache_key, cache_input, json.dumps(results, ensure_ascii=False)
        )
    return results, None


async def _deepseek_fetch(url: str, proxies: Optional[dict], executor=None) -> str:
    """POST /v1/tools/fetch. Returns page Markdown, or "" if unavailable."""
    if not DEEPSEEK_API_KEY or not url.startswith(("http://", "https://")):
        return ""
    cache_key = get_cache_key("deepseek_fetch:" + url)
    cached = await get_cache_async("text_visit", cache_key)
    if cached:
        return cached

    def _do():
        resp = requests.post(
            f"{DEEPSEEK_API_BASE}/v1/tools/fetch",
            headers=_deepseek_headers(), json={"url": url}, timeout=60, proxies=proxies,
        )
        if resp.status_code in DEEPSEEK_RETRY_STATUS:
            raise _RetryableHTTPError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        return resp

    try:
        resp = await run_with_retries_async(_do, attempts=3, delay=2, executor=executor)
    except Exception:
        return ""
    # 403 (security policy) / 404 (no usable content) are permanent — don't retry.
    if resp.status_code != 200:
        return ""
    content = (resp.json().get("content") or "").strip()
    if content:
        await set_cache_async("text_visit", cache_key, url, content)
    return content


class TextSearchTool(DeepResearchTool):
    """DeepSeek Search → page content (search index, else DeepSeek Fetch) → LLM summary."""

    def __init__(self):
        super().__init__(
            name="text_search",
            description="Search the web and return AI-summarised passages from top-k pages.",
            parameters={
                "type": "object",
                "properties": {
                    "q": {"type": "string", "description": "Search query"},
                    "query": {"type": "string", "description": "Alternative for q"},
                    "hl": {"type": "string", "default": "en"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": [],
            },
        )

    async def call(self, q: str = "", query: str = "", hl: str = "en",
                   top_k: int = 5, lang: str = "", **ctx) -> str:
        # `hl` / `lang` are accepted for prompt compatibility; DeepSeek ranks
        # by query language, so they are not forwarded.
        q = q or query
        if not q:
            return "Error: 'q' or 'query' parameter is required"
        try:
            top_k = max(1, min(int(top_k), 10))
        except (TypeError, ValueError):
            top_k = 5

        proxies = self._get_requests_proxies()

        # Step 1 – DeepSeek search, asking for indexed page content up front.
        # (Fetch rejects some major sites, e.g. Wikipedia, while the search
        # index still carries their text.)
        results, err = await _deepseek_search(
            q, count=top_k, need_content=True, proxies=proxies, executor=self.executor,
        )
        if err:
            return f"Search error: {err}"
        if not results:
            return f"No results found for '{q}'"

        # Step 2 – fill missing content via DeepSeek Fetch, then summarise.
        async def _one(i: int, item: dict) -> str:
            url = item.get("url", "")
            title = item.get("title", "")
            page_content = (item.get("content") or "").strip()
            if not page_content and url:
                page_content = await _deepseek_fetch(url, proxies, executor=self.executor)
            if not page_content:
                page_content = item.get("summary") or item.get("snippet") or ""
            summary = await asyncio.get_event_loop().run_in_executor(
                None, _summarize_with_qwen, page_content, q, title,
            )
            return f"[Passage {i}]\nTitle: {title}\nURL: {url}\nSummary:\n{summary}"

        passages = await asyncio.gather(
            *(_one(i, it) for i, it in enumerate(results[:top_k], 1))
        )
        return "\n\n".join(passages)


class WebSearchTool(DeepResearchTool):
    """DeepSeek web search (returns snippets only, no fetch / summary)."""

    def __init__(self):
        super().__init__(
            name="web_search",
            description="Perform a web search and return snippet results.",
            parameters={
                "type": "object",
                "properties": {
                    "q": {"type": "string", "description": "Search query keywords"},
                    "hl": {"type": "string", "default": "en"},
                },
                "required": ["q"],
            },
        )

    async def call(self, q: str = "", hl: str = "en", **ctx) -> str:
        if not q:
            return "Error: 'q' parameter is required"

        proxies = self._get_requests_proxies()
        items, err = await _deepseek_search(
            q, count=10, need_content=False, proxies=proxies, executor=self.executor,
        )
        if err:
            return f"Search error: {err}"
        if not items:
            return f"No results for '{q}'"

        snippets = []
        for idx, it in enumerate(items[:10], 1):
            t = it.get("title") or "Untitled"
            link = it.get("url", "")
            snip = it.get("snippet", "")
            entry = f"{idx}. [{t}]({link})"
            if snip:
                entry += f"\n   {snip}"
            snippets.append(entry)
        return f"Search for '{q}' returned {len(snippets)} results:\n\n" + "\n\n".join(snippets)


# ===================================================================
# 豆包搜索 Global 版 — visual (image-to-image) search
# Docs: https://docs.volcengine.com/docs/Networkedsearch/doubao-search-global-edition
# ===================================================================

DOUBAO_SEARCH_API_KEY = os.getenv("DOUBAO_SEARCH_API_KEY", "")
DOUBAO_SEARCH_URL = os.getenv(
    "DOUBAO_SEARCH_URL", "https://open.feedcoopapi.com/search_api/global_search"
)
IMAGE_SEARCH_MAX_EDGE = int(os.getenv("IMAGE_SEARCH_MAX_EDGE", "1024"))
IMAGE_SEARCH_DOC_COUNT = int(os.getenv("IMAGE_SEARCH_DOC_COUNT", "10"))
# Result.ErrorCode values the docs mark as retryable (rate limit / internal).
_DOUBAO_RETRY_CODES = {700429, 10500, 10501}


def _encode_image_for_search(local_path: str) -> Optional[bytes]:
    """Load, downscale (long edge ≤ IMAGE_SEARCH_MAX_EDGE) and JPEG-encode."""
    try:
        img = Image.open(local_path).convert("RGB")
        if max(img.size) > IMAGE_SEARCH_MAX_EDGE:
            img.thumbnail((IMAGE_SEARCH_MAX_EDGE, IMAGE_SEARCH_MAX_EDGE), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()
    except Exception as exc:
        print(f"[visual_tools] _encode_image_for_search failed: {exc}")
        return None


def _format_visual_results(docs: list) -> str:
    lines = []
    for i, doc in enumerate(docs, 1):
        title = (doc.get("Title") or "Untitled").strip()
        url = doc.get("Url", "")
        host = (doc.get("HostInfo") or {}).get("Hostname") or ""
        texts = [
            s.get("Text", "").strip()
            for s in (doc.get("Snippet") or [])
            if isinstance(s, dict) and s.get("Type") == "text" and s.get("Text", "").strip()
        ]
        entry = f"{i}. {title}" + (f" ({host})" if host else "") + f"\n   URL: {url}"
        if texts:
            entry += "\n   " + " ".join(texts)[:300]
        lines.append(entry)
    return (
        f"Visual search found {len(docs)} visually similar web results "
        f"(titles indicate what the image likely shows):\n\n" + "\n\n".join(lines)
    )


class ImageSearchTool(DeepResearchTool):
    """Reverse image search via 豆包搜索 Global 版 (SearchType=visual, base64 upload)."""

    def __init__(self):
        super().__init__(
            name="image_search",
            description="Analyze an image using visual search to identify objects.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Image reference (e.g., 'img_1') or direct URL"},
                },
                "required": ["url"],
            },
        )

    async def call(self, url: str = "", **ctx) -> str:
        image_paths: dict = ctx.get("image_paths", {})
        intermediate_dir: str = ctx.get("intermediate_dir", "/tmp/vdr_tools")

        if not url:
            return "Error: 'url' parameter is required"
        if not DOUBAO_SEARCH_API_KEY:
            return "image_search error: DOUBAO_SEARCH_API_KEY is not set"

        # Resolve to a local file. A raw URL that is not a known ref is
        # registered under itself so _resolve_image_ref downloads it.
        ref = url
        if ref not in image_paths and ref.startswith(("http://", "https://")):
            image_paths = dict(image_paths)
            image_paths[ref] = ref
        local = _resolve_image_ref(ref, image_paths, intermediate_dir, "image_search")
        if not local:
            return (
                f"Error: Cannot load image '{url}' for image_search. "
                f"Known image refs: {list(ctx.get('image_paths', {}).keys())}"
            )

        img_bytes = await asyncio.get_event_loop().run_in_executor(
            None, _encode_image_for_search, local
        )
        if not img_bytes:
            return f"Error: Failed to decode image '{url}'"

        cache_key = hashlib.sha256(b"doubao_visual:" + img_bytes).hexdigest()
        cached = await get_cache_async("image_search", cache_key)
        if cached:
            return cached

        payload = {
            "SearchType": "visual",
            "ImageQuery": {"ImageBase64": base64.b64encode(img_bytes).decode("ascii")},
            "DocCount": max(1, min(IMAGE_SEARCH_DOC_COUNT, 20)),
            # Queue server-side instead of failing fast when over the QPS limit.
            "EnableWaiting": True,
            "MaxWaitTime": 10000,
        }
        headers = {
            "Authorization": f"Bearer {DOUBAO_SEARCH_API_KEY}",
            "Content-Type": "application/json",
        }
        proxies = self._get_requests_proxies()

        def _do():
            resp = requests.post(
                DOUBAO_SEARCH_URL, headers=headers, json=payload, timeout=60, proxies=proxies,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                raise _RetryableHTTPError(f"HTTP {resp.status_code}")
            data = resp.json()
            code = (data.get("Result") or {}).get("ErrorCode")
            if code in _DOUBAO_RETRY_CODES:
                raise _RetryableHTTPError(f"ErrorCode {code}")
            return resp.status_code, data

        try:
            status, data = await run_with_retries_async(_do, attempts=4, delay=2, executor=self.executor)
        except Exception as exc:
            return f"image_search error: visual search service unavailable ({exc})"

        result = data.get("Result") or {}
        err = (data.get("ResponseMetadata") or {}).get("Error")
        # Report only the code: provider ErrorMsg text can contain agent-directed
        # instructions that should not leak into the policy's observation.
        if status != 200 or err or result.get("ErrorCode"):
            code = result.get("ErrorCode") or (err or {}).get("Code") or status
            return f"image_search error: visual search failed (code {code})"

        docs = result.get("Documents") or []
        if not docs:
            return "image_search: no visually similar results found."
        out = _format_visual_results(docs)
        await set_cache_async("image_search", cache_key, f"sha256-jpeg:{len(img_bytes)}", out)
        return out


class PerspectiveCorrectTool(DeepResearchTool):
    """Auto perspective correction via OpenCV."""

    def __init__(self):
        super().__init__(
            name="perspective_correct",
            description="Correct perspective distortion in an image.",
            parameters={
                "type": "object",
                "properties": {
                    "image": {"type": "string", "description": "Image reference"},
                },
                "required": ["image"],
            },
        )

    async def call(self, image: str = "", **ctx) -> str:
        if not CV2_AVAILABLE:
            return "Error: OpenCV not available. pip install opencv-python"
        image_paths: dict = ctx.get("image_paths", {})
        intermediate_dir: str = ctx.get("intermediate_dir", "/tmp/vdr_tools")

        if not image or image not in image_paths:
            return f"Error: Image '{image}' not found. Available: {list(image_paths.keys())}"

        local = _resolve_image_ref(image, image_paths, intermediate_dir, "perspective_correct")
        if not local:
            return f"Error: Failed to resolve image for {image}"

        try:
            eng = ImageEnhancementEngine()
            eng.load_from_path(local)
            eng.auto_correct_perspective()
            pil = eng.to_pil()
            if not pil:
                return "Error: perspective correction failed"

            new_id = f"img_{len(image_paths) + 1}"
            cos = _upload_pil_to_cos(pil, "perspective_correct")
            if cos:
                image_paths[new_id] = cos
                return f"Perspective corrected. New image ID: {new_id}. Uploaded to: {cos}"
            save = os.path.join(intermediate_dir, f"vdr_persp_{int(time.time()*1000)}.png")
            os.makedirs(intermediate_dir, exist_ok=True)
            pil.save(save)
            image_paths[new_id] = save
            return f"Perspective corrected. New image ID: {new_id}. Saved to: {save}"
        except Exception as exc:
            return f"Error: {exc}"


class SuperResolutionTool(DeepResearchTool):
    """Super resolution via OpenCV DNN."""

    def __init__(self):
        super().__init__(
            name="super_resolution",
            description="Enhance image resolution using super-resolution.",
            parameters={
                "type": "object",
                "properties": {
                    "image": {"type": "string", "description": "Image reference"},
                    "scale": {"type": "integer", "default": 4},
                },
                "required": ["image"],
            },
        )

    async def call(self, image: str = "", scale: int = 4, **ctx) -> str:
        if not CV2_AVAILABLE:
            return "Error: OpenCV not available. pip install opencv-python"
        image_paths: dict = ctx.get("image_paths", {})
        intermediate_dir: str = ctx.get("intermediate_dir", "/tmp/vdr_tools")

        if not image or image not in image_paths:
            return f"Error: Image '{image}' not found."

        local = _resolve_image_ref(image, image_paths, intermediate_dir, "super_resolution")
        if not local:
            return f"Error: Failed to resolve image for {image}"

        try:
            def _run_sr():
                eng = ImageEnhancementEngine()
                eng.load_from_path(local)
                eng.apply_super_resolution(scale=int(scale or 4))
                return eng.to_pil()

            # CPU-bound for seconds: keep it off the event loop.
            pil = await asyncio.get_event_loop().run_in_executor(self.executor, _run_sr)
            if not pil:
                return "Error: super resolution failed"

            new_id = f"img_{len(image_paths) + 1}"
            cos = _upload_pil_to_cos(pil, "super_resolution")
            if cos:
                image_paths[new_id] = cos
                return f"Super resolution done. New image ID: {new_id}. Uploaded to: {cos}"
            save = os.path.join(intermediate_dir, f"vdr_sr_{int(time.time()*1000)}.png")
            os.makedirs(intermediate_dir, exist_ok=True)
            pil.save(save)
            image_paths[new_id] = save
            return f"Super resolution done. New image ID: {new_id}. Saved to: {save}"
        except Exception as exc:
            return f"Error: super resolution failed: {exc}"


class SharpenTool(DeepResearchTool):
    """Sharpen / unsharp-mask via OpenCV."""

    def __init__(self):
        super().__init__(
            name="sharpen",
            description="Sharpen an image to reduce blur.",
            parameters={
                "type": "object",
                "properties": {
                    "image": {"type": "string", "description": "Image reference"},
                    "amount": {"type": "number", "default": 1.5},
                },
                "required": ["image"],
            },
        )

    async def call(self, image: str = "", amount: float = 1.5, **ctx) -> str:
        if not CV2_AVAILABLE:
            return "Error: OpenCV not available. pip install opencv-python"
        image_paths: dict = ctx.get("image_paths", {})
        intermediate_dir: str = ctx.get("intermediate_dir", "/tmp/vdr_tools")

        if not image or image not in image_paths:
            return f"Error: Image '{image}' not found."

        local = _resolve_image_ref(image, image_paths, intermediate_dir, "sharpen")
        if not local:
            return f"Error: Failed to resolve image for {image}"

        try:
            eng = ImageEnhancementEngine()
            eng.load_from_path(local)
            eng.enhance_sharpness(amount=amount)
            pil = eng.to_pil()
            if not pil:
                return "Error: sharpening failed"

            new_id = f"img_{len(image_paths) + 1}"
            cos = _upload_pil_to_cos(pil, "sharpen")
            if cos:
                image_paths[new_id] = cos
                return f"Image sharpened. New image ID: {new_id}. Uploaded to: {cos}"
            save = os.path.join(intermediate_dir, f"vdr_sharp_{int(time.time()*1000)}.png")
            os.makedirs(intermediate_dir, exist_ok=True)
            pil.save(save)
            image_paths[new_id] = save
            return f"Image sharpened. New image ID: {new_id}. Saved to: {save}"
        except Exception as exc:
            return f"Error: {exc}"
