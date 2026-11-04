"""Per-case inference pipeline (multi-turn agent loop)."""

from __future__ import annotations

import io
import json
import logging
import os
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
from PIL import Image

from . import config, image_io, tools
from .runners import BaseRunner, InferenceConfig
from .prompts import SYSTEM_PROMPT


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trajectory serialisation
# ---------------------------------------------------------------------------


def _strip_base64_payloads(obj: Any, image_urls: Dict[str, str]) -> Any:
    """Replace large base64 image blobs with their public URLs."""

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, bytes):
        return f"<bytes: {len(obj)} bytes>"
    if isinstance(obj, list):
        return [_strip_base64_payloads(item, image_urls) for item in obj]
    if isinstance(obj, dict):
        replacement_url = next(iter(image_urls.values()), None)
        if "source" in obj and isinstance(obj["source"], dict):
            data = obj["source"].get("data", "")
            if isinstance(data, str) and len(data) > 100 and replacement_url:
                return {
                    "type": "image_url",
                    "image_url": {"url": replacement_url},
                }
        if "inline_data" in obj and isinstance(obj["inline_data"], dict):
            data = obj["inline_data"].get("data", "")
            if isinstance(data, str) and len(data) > 100 and replacement_url:
                return {
                    "type": "image_url",
                    "image_url": {"url": replacement_url},
                }
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if k == "data" and isinstance(v, str) and len(v) > 1000:
                if any(key in obj for key in ("type", "media_type", "mime_type")):
                    out[k] = (
                        replacement_url
                        if replacement_url
                        else f"<base64_image_data: {len(v)} chars>"
                    )
                    continue
            out[k] = _strip_base64_payloads(v, image_urls)
        return out
    return obj


# ---------------------------------------------------------------------------
# Image bootstrap helpers
# ---------------------------------------------------------------------------


def _add_image_url(
    image_paths_dict: Dict[str, Any],
    initial_parts: List[Dict[str, Any]],
    url: str,
) -> None:
    image_id = f"img_{len(image_paths_dict) + 1}"
    image_paths_dict[image_id] = url
    initial_parts.append({"image_url": {"url": url}})


def _add_inline_image(
    image_paths_dict: Dict[str, Any],
    initial_parts: List[Dict[str, Any]],
    payload: bytes | str,
    raw_storage: Any,
) -> None:
    encoded = image_io.image_to_base64(payload) if isinstance(payload, bytes) else payload
    if not encoded:
        return
    image_id = f"img_{len(image_paths_dict) + 1}"
    image_paths_dict[image_id] = raw_storage
    initial_parts.append(
        {
            "inline_data": {
                "mime_type": image_io.detect_image_format(encoded),
                "data": encoded,
            }
        }
    )


def _as_list(value: Any) -> list:
    """Parquet list columns come back as numpy arrays."""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value if isinstance(value, list) else [value]


def _bootstrap_images(
    row: Any,
    case_id: str,
    case_idx: int,
    filename_prefix: str,
    image_urls_dict: Optional[Dict[str, List[Any]]],
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Build the initial (image_paths_dict, parts) pair from row data.

    Resolution order:

    1. URLs supplied via ``image_urls_dict`` (cheapest, no network).
    2. The configured ``FVQA_IMAGE_DIR`` (loads bytes from disk).
    3. The ``images`` column on the row (parquet payloads).
    """

    image_paths_dict: Dict[str, Any] = {}
    initial_parts: List[Dict[str, Any]] = []

    if image_urls_dict and case_id in image_urls_dict:
        for entry in image_urls_dict[case_id]:
            if isinstance(entry, dict):
                url = entry.get("cos_url") or entry.get("image_url") or entry.get("url")
            else:
                url = str(entry)
            if url:
                _add_image_url(image_paths_dict, initial_parts, url)

    if not initial_parts and config.FVQA_IMAGE_DIR:
        local = image_io._try_local_image(case_id)
        if local:
            try:
                with open(local, "rb") as fh:
                    payload = fh.read()
                _add_inline_image(image_paths_dict, initial_parts, payload, local)
            except OSError as exc:
                logger.warning("Failed to read local image %s: %s", local, exc)

    if not initial_parts:
        images = _as_list(row.get("images", [])) if hasattr(row, "get") else []
        for entry in images:
            if entry is None:
                continue
            url: Optional[str] = None
            payload: Optional[bytes] = None
            if isinstance(entry, dict):
                url = entry.get("url") or entry.get("image_url") or entry.get("cos_url")
                payload = entry.get("bytes")
            elif isinstance(entry, str) and entry.startswith(("http://", "https://")):
                url = entry
            elif isinstance(entry, bytes):
                payload = entry

            if url:
                _add_image_url(image_paths_dict, initial_parts, url)
                continue
            if payload:
                pil_image = Image.open(io.BytesIO(payload))
                from . import cos_upload

                cos_url = cos_upload.upload_pil_image(
                    pil_image,
                    filename_prefix,
                    case_idx,
                    0,
                    "original_image",
                )
                if cos_url:
                    _add_image_url(image_paths_dict, initial_parts, cos_url)
                else:
                    _add_inline_image(
                        image_paths_dict, initial_parts, payload, payload
                    )
    return image_paths_dict, initial_parts


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def process_single_case(
    row: Any,
    runner: BaseRunner,
    output_dir: str,
    case_idx: int,
    image_urls_dict: Optional[Dict[str, List[Any]]] = None,
    dataset_type: str = "train",
    visual_lookup: Optional[Callable[..., object]] = None,
    inference_cfg: Optional[InferenceConfig] = None,
) -> Dict[str, Any]:
    """Drive one benchmark example through the agent."""

    filename_prefix = "fvqa_test" if dataset_type == "test" else "fvqa_train"

    if hasattr(row, "get"):
        case_id = row.get("data_id", f"case_{case_idx}")
        category = row.get("category", "unknown")
        data_source = row.get("data_source", "unknown")
        prompt_list = _as_list(row.get("prompt", []))
    else:
        case_id = f"case_{case_idx}"
        category = "unknown"
        data_source = "unknown"
        prompt_list = []

    logger.info(
        "Processing case %d (%s, category=%s, source=%s)",
        case_idx + 1,
        case_id,
        category,
        data_source,
    )

    if isinstance(prompt_list, list) and prompt_list:
        first = prompt_list[0]
        prompt_text = (
            first.get("content", "") if isinstance(first, dict) else str(first)
        )
    else:
        prompt_text = str(prompt_list) if prompt_list else ""

    image_paths_dict, initial_parts = _bootstrap_images(
        row, case_id, case_idx, filename_prefix, image_urls_dict
    )

    if getattr(runner, "prompt_in_system", False):
        initial_parts.append({"text": prompt_text})
    else:
        tools_text = f"<tools>\n{tools.get_tools_definition()}\n</tools>"
        initial_parts.append({"text": tools_text + "\n\n" + prompt_text})

    gemini_contents: List[Dict[str, Any]] = [
        {"role": "user", "parts": initial_parts}
    ]

    trajectory: Dict[str, Any] = {
        "case_id": case_id,
        "case_idx": case_idx,
        "category": category,
        "data_source": data_source,
        "prompt": prompt_list,
        "turns": [],
        "timestamp": datetime.now().isoformat(),
        "stop_reason": "max_turns",
    }
    if hasattr(row, "get") and row.get("answer") is not None:
        trajectory["original_data"] = {
            "question": prompt_text,
            "answers": [str(row.get("answer"))],
        }

    intermediate_dir = os.path.join(output_dir, "intermediate")
    cfg = inference_cfg or InferenceConfig()

    for turn_num in range(config.MAX_TURNS):
        try:
            response = runner.infer(
                contents=gemini_contents,
                system_instruction=SYSTEM_PROMPT,
                cfg=cfg,
            )
        except Exception as exc:
            logger.error("Inference failed on turn %d: %s", turn_num, exc, exc_info=True)
            trajectory["turns"].append({"turn": turn_num, "error": str(exc)})
            trajectory["stop_reason"] = "runner_exception"
            break

        response_text = ""
        for cand in response.get("candidates", []) or []:
            for part in cand.get("content", {}).get("parts", []):
                if "text" in part:
                    response_text += part["text"]

        turn_record: Dict[str, Any] = {
            "turn": turn_num,
            "response": response,
            "response_text": response_text,
        }
        trajectory["turns"].append(turn_record)

        candidates = response.get("candidates", []) or []
        if candidates:
            gemini_contents.append(
                {
                    "role": "model",
                    "parts": candidates[0].get("content", {}).get("parts", []),
                }
            )

        finish = (candidates[0].get("finishReason") if candidates else None) or ""
        if finish == "ERROR":
            # A prompt outgrowing the context is the agent's outcome, not a crash.
            trajectory["stop_reason"] = (
                "context_limit" if "context_limit" in response_text else "runner_error"
            )
            break

        if tools.has_response_tag(response_text):
            trajectory["stop_reason"] = "response"
            break

        tool_call_json = tools.extract_tool_call(response_text)
        if not tool_call_json:
            logger.info("Turn %d ended without tool call or response tag.", turn_num)
            trajectory["stop_reason"] = (
                "max_tokens" if finish == "MAX_TOKENS" else "no_tool_call"
            )
            break

        os.makedirs(intermediate_dir, exist_ok=True)
        # A bad tool call is the agent's error to recover from, not a lost case.
        try:
            tool_message, new_images = tools.execute_tool(
                tool_call_json,
                image_paths_dict,
                case_id,
                case_idx,
                turn_num,
                intermediate_dir,
                filename_prefix=filename_prefix,
                visual_lookup=visual_lookup,
            )
        except Exception as exc:
            logger.warning("Tool failed on turn %d: %s", turn_num, exc)
            tool_message = f"Tool execution error:\n{type(exc).__name__}: {exc}"
            new_images = {}
        turn_record["tool_call"] = tool_call_json
        observation_text = f"<observation>\n{tool_message}\n</observation>"
        turn_record["tool_output"] = observation_text

        if new_images:
            for new_id, payload in new_images.items():
                image_paths_dict[new_id] = payload
                if isinstance(payload, str) and payload.startswith(
                    ("http://", "https://")
                ):
                    gemini_contents.append(
                        {
                            "role": "user",
                            "parts": [
                                {"text": observation_text},
                                {"image_url": {"url": payload}},
                            ],
                        }
                    )
                elif isinstance(payload, str) and os.path.exists(payload):
                    try:
                        with open(payload, "rb") as fh:
                            data = fh.read()
                        encoded = image_io.image_to_base64(data) or ""
                        gemini_contents.append(
                            {
                                "role": "user",
                                "parts": [
                                    {"text": observation_text},
                                    {
                                        "inline_data": {
                                            "mime_type": image_io.detect_image_format(
                                                encoded
                                            ),
                                            "data": encoded,
                                        }
                                    },
                                ],
                            }
                        )
                    except OSError as exc:
                        logger.warning("Cannot read intermediate image %s: %s", payload, exc)
                        gemini_contents.append(
                            {"role": "user", "parts": [{"text": observation_text}]}
                        )
                elif isinstance(payload, bytes):
                    encoded = image_io.image_to_base64(payload) or ""
                    gemini_contents.append(
                        {
                            "role": "user",
                            "parts": [
                                {"text": observation_text},
                                {
                                    "inline_data": {
                                        "mime_type": image_io.detect_image_format(
                                            encoded
                                        ),
                                        "data": encoded,
                                    }
                                },
                            ],
                        }
                    )
                else:
                    gemini_contents.append(
                        {"role": "user", "parts": [{"text": observation_text}]}
                    )
        else:
            gemini_contents.append(
                {"role": "user", "parts": [{"text": observation_text}]}
            )

    # Trajectory writeout
    trajectory["final_response_text"] = "\n\n".join(
        turn.get("response_text", "")
        for turn in trajectory["turns"]
        if turn.get("response_text")
    )

    image_urls = {
        img_id: data
        for img_id, data in image_paths_dict.items()
        if isinstance(data, str) and data.startswith(("http://", "https://"))
    }
    serialised = _strip_base64_payloads(trajectory, image_urls)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{case_id}_trajectory.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(serialised, fh, ensure_ascii=False, indent=2)
    logger.info("Trajectory saved to %s", out_path)
    return trajectory
