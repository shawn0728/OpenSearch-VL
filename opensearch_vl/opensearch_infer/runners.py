"""Inference runners for the supported model families.

Runners share a tiny interface:

* :py:meth:`BaseRunner.load`  — perform any one-time setup (model load).
* :py:meth:`BaseRunner.infer` — accept Gemini-style ``contents`` plus a
  system prompt and return a Gemini-shaped response dict.

The Gemini-shaped response keeps the calling code in :mod:`pipeline`
agnostic to the underlying provider. The minimal contract is::

    {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": "..."}]},
                "finishReason": "STOP" | "ERROR",
            }
        ],
        "usageMetadata": {...},
        "modelVersion": "...",
    }
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from . import auth, config, messages
from .config import ModelSpec


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Common types
# ---------------------------------------------------------------------------


@dataclass
class InferenceConfig:
    """Tunables shared across runners."""

    temperature: float = 0.0
    max_tokens: int = 32768
    enable_thinking: bool = False


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseRunner:
    spec: ModelSpec

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec

    @property
    def display_name(self) -> str:
        return self.spec.display_name

    def load(self) -> None:  # pragma: no cover - default no-op
        """Hook for one-time model preparation."""

    def infer(
        self,
        contents: Iterable[Dict[str, Any]],
        system_instruction: Optional[str] = None,
        cfg: Optional[InferenceConfig] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError


def _empty_response(model_version: str, error: str) -> Dict[str, Any]:
    return {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": f"Error: {error}"}]},
                "finishReason": "ERROR",
            }
        ],
        "usageMetadata": {},
        "modelVersion": model_version,
        "raw_response": {"error": error},
    }


# ---------------------------------------------------------------------------
# Claude
# ---------------------------------------------------------------------------


class ClaudeRunner(BaseRunner):
    """Drives the Claude HMAC gateway."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec)
        self.client: Optional[auth.ClaudeGatewayClient] = None

    def load(self) -> None:
        self.client = auth.ClaudeGatewayClient.from_env()

    def infer(
        self,
        contents: Iterable[Dict[str, Any]],
        system_instruction: Optional[str] = None,
        cfg: Optional[InferenceConfig] = None,
    ) -> Dict[str, Any]:
        if self.client is None:
            self.load()
        cfg = cfg or InferenceConfig()
        try:
            claude_messages = messages.to_claude_messages(contents)
            response = self.client.call(  # type: ignore[union-attr]
                claude_messages,
                system_instruction=system_instruction,
                max_tokens=cfg.max_tokens,
            )
        except Exception as exc:
            logger.error("Claude inference failed: %s", exc)
            return _empty_response(self.display_name, str(exc))

        try:
            response_json = response.json()
        except ValueError as exc:
            return _empty_response(self.display_name, f"Invalid JSON: {exc}")

        if response_json.get("code", 0) != 0:
            return _empty_response(
                self.display_name,
                response_json.get("msg", "Unknown error"),
            )

        text_parts: List[Dict[str, str]] = []
        for item in response_json.get("answer", []) or []:
            if item.get("type") == "text" and item.get("value"):
                text_parts.append({"text": item["value"]})

        if not text_parts:
            request_detail = response_json.get("request_detail", {})
            response_detail = request_detail.get("response", {})
            for item in response_detail.get("content", []) or []:
                if item.get("type") == "text" and item.get("text"):
                    text_parts.append({"text": item["text"]})

        finish = response_json.get("finish_reason", "STOP")
        if finish == "STOP":
            finish = (
                response_json.get("request_detail", {})
                .get("response", {})
                .get("stop_reason", "STOP")
            )

        usage_metadata: Dict[str, int] = {}
        cost_info = response_json.get("cost_info") or {}
        if cost_info:
            usage_metadata = {
                "promptTokenCount": cost_info.get("prompt_tokens", 0),
                "candidatesTokenCount": cost_info.get("completion_tokens", 0),
                "totalTokenCount": cost_info.get("total_tokens", 0),
            }
        else:
            usage = (
                response_json.get("request_detail", {})
                .get("response", {})
                .get("usage", {})
            )
            if usage:
                usage_metadata = {
                    "promptTokenCount": usage.get("input_tokens", 0),
                    "candidatesTokenCount": usage.get("output_tokens", 0),
                    "totalTokenCount": usage.get("input_tokens", 0)
                    + usage.get("output_tokens", 0),
                }

        return {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": text_parts or [{"text": ""}],
                    },
                    "finishReason": finish or "STOP",
                }
            ],
            "usageMetadata": usage_metadata,
            "modelVersion": self.display_name,
            "raw_response": response_json,
        }


# ---------------------------------------------------------------------------
# MiniMax
# ---------------------------------------------------------------------------


class MiniMaxRunner(BaseRunner):
    """Drives the MiniMax Anthropic-compatible Messages API."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec)
        self.client: Optional[auth.MiniMaxClient] = None

    @property
    def api_model(self) -> str:
        return self.spec.extra.get("api_model", self.spec.display_name)

    def load(self) -> None:
        self.client = auth.MiniMaxClient.from_env()

    def infer(
        self,
        contents: Iterable[Dict[str, Any]],
        system_instruction: Optional[str] = None,
        cfg: Optional[InferenceConfig] = None,
    ) -> Dict[str, Any]:
        if self.client is None:
            self.load()
        cfg = cfg or InferenceConfig()
        try:
            minimax_messages = messages.to_minimax_messages(contents)
            response = self.client.call(  # type: ignore[union-attr]
                model=self.api_model,
                messages=minimax_messages,
                system_instruction=system_instruction,
                max_tokens=cfg.max_tokens,
            )
        except Exception as exc:
            logger.error("MiniMax inference failed: %s", exc)
            return _empty_response(self.display_name, str(exc))

        try:
            response_json = response.json()
        except ValueError as exc:
            return _empty_response(self.display_name, f"Invalid JSON: {exc}")

        if response_json.get("type") == "error":
            error = response_json.get("error", {}) or {}
            return _empty_response(
                self.display_name, error.get("message", "Unknown error")
            )

        text_parts: List[Dict[str, str]] = []
        for item in response_json.get("content", []) or []:
            if item.get("type") == "text" and item.get("text"):
                text_parts.append({"text": item["text"]})

        finish = response_json.get("stop_reason") or "STOP"

        usage_metadata: Dict[str, int] = {}
        usage = response_json.get("usage", {}) or {}
        if usage:
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            usage_metadata = {
                "promptTokenCount": input_tokens,
                "candidatesTokenCount": output_tokens,
                "totalTokenCount": input_tokens + output_tokens,
            }

        return {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": text_parts or [{"text": ""}],
                    },
                    "finishReason": finish,
                }
            ],
            "usageMetadata": usage_metadata,
            "modelVersion": self.display_name,
            "raw_response": response_json,
        }


# ---------------------------------------------------------------------------
# Qwen3-VL (dense and MoE)
# ---------------------------------------------------------------------------


def _patch_moe_scatter_dtype() -> None:
    """Realign router weights to the input dtype for Qwen3-VL MoE.

    Older Transformers releases scatter the router weights with the
    softmax dtype which does not match ``hidden_states`` when the model
    runs in bfloat16 / float16. This patch mirrors the upstream fix so
    the 30B-A3B checkpoint can be served immediately.
    """

    try:
        from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
            Qwen3VLMoeTextSparseMoeBlock,
        )
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - depends on transformers
        logger.warning("MoE scatter patch skipped: %s", exc)
        return

    def _patched_forward(self, hidden_states):  # type: ignore[no-untyped-def]
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, router_indices = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        routing_weights = routing_weights / routing_weights.sum(
            dim=-1, keepdim=True
        )
        routing_weights = routing_weights.to(hidden_states.dtype)
        router_weights = torch.zeros(
            router_logits.shape,
            dtype=routing_weights.dtype,
            device=router_logits.device,
        ).scatter_(1, router_indices, routing_weights)
        hidden_states = hidden_states.reshape(batch_size, -1, self.hidden_size)
        routed_out = self.experts(hidden_states, router_weights, router_indices)
        return routed_out, router_logits

    Qwen3VLMoeTextSparseMoeBlock.forward = _patched_forward


def _resolve_devices(gpus: str) -> Tuple[List[int], object, str]:
    """Return ``(valid_ids, device_map, device_str)``.

    The device map follows Hugging Face conventions:

    * ``"auto"`` for multi-GPU model parallel
    * ``{"": "cuda:N"}`` for single-GPU placement
    * ``"cpu"`` when CUDA is not available
    """

    import torch

    if not torch.cuda.is_available():
        return [], "cpu", "cpu"

    requested = [
        int(part.strip()) for part in (gpus or "0").split(",") if part.strip() != ""
    ] or [0]
    available = list(range(torch.cuda.device_count()))
    valid = [g for g in requested if g in available]
    if not valid:
        valid = available[:1]
    if len(valid) >= 2:
        return valid, "auto", f"cuda:{valid[0]}"
    return valid, {"": f"cuda:{valid[0]}"}, f"cuda:{valid[0]}"


class Qwen3VLRunner(BaseRunner):
    """Local inference for Qwen3-VL dense (8B/32B) and MoE (30B-A3B)."""

    def __init__(
        self,
        spec: ModelSpec,
        checkpoint: str,
        gpus: str = "0",
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
    ) -> None:
        super().__init__(spec)
        self.checkpoint = checkpoint
        self.gpus = gpus
        self.dtype_name = dtype
        self.trust_remote_code = trust_remote_code

        self._model = None
        self._processor = None
        self._device: Optional[object] = None
        self._uses_old_api = False
        self._process_vision_info = None

    # ------------------------------------------------------------------ load

    def load(self) -> None:
        if self._model is not None:
            return

        if not self.checkpoint:
            raise RuntimeError(
                f"Checkpoint not configured for {self.spec.name!r}. "
                "Provide --checkpoint or set the matching env variable."
            )

        if self.spec.needs_moe_patch:
            _patch_moe_scatter_dtype()

        try:
            from qwen_vl_utils import process_vision_info  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "qwen_vl_utils is required. Install with: pip install qwen-vl-utils"
            ) from exc
        self._process_vision_info = process_vision_info

        from transformers import AutoProcessor  # type: ignore

        if self.spec.family == "qwen3_vl_moe":
            try:
                from transformers import (  # type: ignore
                    Qwen3VLMoeForConditionalGeneration as ModelClass,
                )
                self._uses_old_api = False
            except ImportError:
                from transformers import (  # type: ignore
                    Qwen3VLForConditionalGeneration as ModelClass,
                )
                self._uses_old_api = True
        else:
            from transformers import (  # type: ignore
                Qwen3VLForConditionalGeneration as ModelClass,
            )
            self._uses_old_api = True

        valid_ids, device_map, device_str = _resolve_devices(self.gpus)
        self._device = device_str

        import torch

        torch_dtype = getattr(torch, self.dtype_name, torch.bfloat16)

        self._validate_checkpoint(self.checkpoint)

        load_kwargs: Dict[str, Any] = {
            "device_map": device_map,
            "trust_remote_code": self.trust_remote_code,
        }

        if self.spec.family == "qwen3_vl_moe" and not self._uses_old_api:
            load_kwargs["dtype"] = torch_dtype
        else:
            load_kwargs["torch_dtype"] = torch_dtype

        logger.info(
            "Loading %s from %s (device_map=%s, dtype=%s)",
            self.spec.display_name,
            self.checkpoint,
            device_map,
            torch_dtype,
        )
        self._model = ModelClass.from_pretrained(self.checkpoint, **load_kwargs)
        self._model.eval()

        if self.spec.family == "qwen3_vl_moe":
            self._enforce_dtype(torch_dtype)

        self._processor = self._load_processor()

        if isinstance(device_map, dict):
            logger.info("Single-GPU deployment: %s", device_str)
        elif device_map == "auto":
            logger.info(
                "Model parallel deployment across GPUs %s",
                ", ".join(map(str, valid_ids)),
            )
        else:
            logger.info("CPU deployment")

    def _validate_checkpoint(self, path: str) -> None:
        if "/" not in path or os.path.exists(path):
            if os.path.exists(path) and os.path.isdir(path):
                if not os.path.exists(os.path.join(path, "config.json")):
                    raise FileNotFoundError(
                        f"Missing config.json under checkpoint dir: {path}"
                    )
                has_weights = (
                    bool(list(Path(path).glob("*.safetensors")))
                    or bool(list(Path(path).glob("*.bin")))
                    or os.path.exists(
                        os.path.join(path, "model.safetensors.index.json")
                    )
                )
                if not has_weights:
                    logger.warning(
                        "Checkpoint dir %s has no .safetensors/.bin weights",
                        path,
                    )

    def _enforce_dtype(self, target_dtype) -> None:
        """Recursively cast every floating parameter to ``target_dtype``."""

        import torch

        def _cast(module):
            for _, param in module.named_parameters(recurse=False):
                if param is not None and param.dtype.is_floating_point:
                    param.data = param.data.to(dtype=target_dtype)
            for _, buffer in module.named_buffers(recurse=False):
                if buffer is not None and buffer.dtype.is_floating_point:
                    buffer.data = buffer.data.to(dtype=target_dtype)
            for child in module.children():
                _cast(child)

        _cast(self._model)
        self._model = self._model.to(dtype=target_dtype)

    def _load_processor(self):
        from transformers import AutoProcessor  # type: ignore

        if os.path.exists(self.checkpoint) and os.path.isdir(self.checkpoint):
            preprocessor_config = os.path.join(
                self.checkpoint, "preprocessor_config.json"
            )
            if os.path.exists(preprocessor_config):
                return AutoProcessor.from_pretrained(
                    self.checkpoint, trust_remote_code=self.trust_remote_code
                )
        # Try the bundled checkpoint first; otherwise look for the matching HF id.
        try:
            return AutoProcessor.from_pretrained(
                self.checkpoint, trust_remote_code=self.trust_remote_code
            )
        except Exception as exc:  # pragma: no cover - environment-bound
            fallback = self.spec.default_checkpoint
            if fallback and fallback != self.checkpoint:
                logger.warning(
                    "Processor for %s not found, falling back to %s (%s)",
                    self.checkpoint,
                    fallback,
                    exc,
                )
                return AutoProcessor.from_pretrained(
                    fallback, trust_remote_code=self.trust_remote_code
                )
            raise

    # --------------------------------------------------------------- infer

    def infer(
        self,
        contents: Iterable[Dict[str, Any]],
        system_instruction: Optional[str] = None,
        cfg: Optional[InferenceConfig] = None,
    ) -> Dict[str, Any]:
        if self._model is None:
            self.load()
        cfg = cfg or InferenceConfig()

        try:
            qwen_messages = messages.to_qwen3vl_messages(contents)
            qwen_messages = self._prepend_system(qwen_messages, system_instruction)
            inputs = self._build_inputs(qwen_messages)
            output_text, prompt_len, gen_len = self._generate(inputs, cfg)
        except Exception as exc:
            logger.error("Qwen3-VL inference failed: %s", exc, exc_info=True)
            return _empty_response(self.display_name, str(exc))

        return {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": output_text}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": prompt_len,
                "candidatesTokenCount": gen_len,
                "totalTokenCount": prompt_len + gen_len,
            },
            "modelVersion": self.display_name,
            "raw_response": {"text": output_text},
        }

    # ------------------------------------------------------------- helpers

    @staticmethod
    def _prepend_system(
        msgs: List[Dict[str, Any]], system_instruction: Optional[str]
    ) -> List[Dict[str, Any]]:
        if not system_instruction:
            return msgs
        if msgs and msgs[0].get("role") == "user":
            first = dict(msgs[0])
            content = first.get("content")
            if isinstance(content, list):
                first["content"] = [
                    {"type": "text", "text": system_instruction + "\n\n"},
                    *content,
                ]
            else:
                first["content"] = [
                    {"type": "text", "text": system_instruction + "\n\n"},
                    {"type": "text", "text": str(content or "")},
                ]
            return [first, *msgs[1:]]
        return [
            {
                "role": "user",
                "content": [{"type": "text", "text": system_instruction}],
            },
            *msgs,
        ]

    def _build_inputs(self, qwen_messages: List[Dict[str, Any]]):
        text = self._processor.apply_chat_template(
            qwen_messages, tokenize=False, add_generation_prompt=True
        )
        if self._uses_old_api:
            image_inputs, video_inputs = self._process_vision_info(qwen_messages)
            return self._processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        try:
            image_inputs, video_inputs, _ = self._process_vision_info(
                qwen_messages,
                image_patch_size=self._processor.image_processor.patch_size,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
        except TypeError:
            image_inputs, video_inputs = self._process_vision_info(qwen_messages)
        if video_inputs:
            return self._processor(
                text=[text],
                images=image_inputs or None,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        if image_inputs:
            return self._processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            )
        return self._processor(
            text=[text], padding=True, return_tensors="pt"
        )

    def _resolve_io_device(self):
        import torch

        device_map = getattr(self._model, "hf_device_map", None)
        if device_map:
            for value in device_map.values():
                if isinstance(value, torch.device) and value.type == "cuda":
                    return value
                if isinstance(value, str) and "cuda" in value:
                    return torch.device(value)
        return next(self._model.parameters()).device

    def _generate(self, inputs, cfg: InferenceConfig):
        import torch

        device = self._resolve_io_device()
        if hasattr(self._model, "lm_head") and hasattr(self._model.lm_head, "weight"):
            model_dtype = self._model.lm_head.weight.dtype
        else:
            model_dtype = next(self._model.parameters()).dtype

        for key in list(inputs.keys()):
            value = inputs[key]
            if isinstance(value, torch.Tensor):
                if value.dtype.is_floating_point:
                    inputs[key] = value.to(device=device, dtype=model_dtype)
                else:
                    inputs[key] = value.to(device=device)
            elif isinstance(value, (list, tuple)) and value and isinstance(
                value[0], torch.Tensor
            ):
                cast = []
                for tensor in value:
                    if tensor.dtype.is_floating_point:
                        cast.append(tensor.to(device=device, dtype=model_dtype))
                    else:
                        cast.append(tensor.to(device=device))
                inputs[key] = type(value)(cast)

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": cfg.max_tokens,
            "do_sample": cfg.temperature > 0,
        }
        if cfg.temperature > 0:
            gen_kwargs["temperature"] = cfg.temperature
            gen_kwargs["top_p"] = 0.8
            gen_kwargs["top_k"] = 20

        with torch.no_grad():
            if model_dtype == torch.bfloat16 and device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    generated = self._model.generate(**inputs, **gen_kwargs)
            else:
                generated = self._model.generate(**inputs, **gen_kwargs)

        prompt_lens = [seq.shape[0] for seq in inputs.input_ids]
        trimmed = [
            out[in_len:] for out, in_len in zip(generated, prompt_lens)
        ]
        decoded = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        text = decoded[0] if isinstance(decoded, list) else str(decoded)
        return text, int(prompt_lens[0]), int(trimmed[0].shape[0])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_runner(
    model_name: str,
    checkpoint: Optional[str] = None,
    gpus: str = "0",
    dtype: str = "bfloat16",
) -> BaseRunner:
    if model_name not in config.MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model {model_name!r}. Choose one of: {config.list_model_names()}"
        )
    spec = config.MODEL_REGISTRY[model_name]
    if spec.family == "claude":
        return ClaudeRunner(spec)
    if spec.family == "minimax":
        return MiniMaxRunner(spec)
    return Qwen3VLRunner(
        spec,
        checkpoint=config.resolve_checkpoint(spec, checkpoint),
        gpus=gpus,
        dtype=dtype,
    )
