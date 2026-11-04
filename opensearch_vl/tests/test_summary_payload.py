"""The summariser request body must actually disable thinking on a local vLLM server."""

import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opensearch_infer import config, search  # noqa: E402


def _payload(summary_extra, llm_extra):
    with mock.patch.object(config, "SUMMARY_EXTRA_BODY", summary_extra), \
            mock.patch.object(config, "LLM_EXTRA_BODY", llm_extra):
        return search._summary_payload("p")


def test_local_default_puts_vllm_fields_at_top_level():
    p = _payload(None, {})
    assert "extra_body" not in p
    assert p["chat_template_kwargs"] == {"enable_thinking": False}
    assert p["top_k"] == 20


def test_summary_body_overrides_shared_judge_body():
    local = {"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}
    p = _payload(local, {"thinking": {"type": "disabled"}})
    assert "thinking" not in p
    assert p["chat_template_kwargs"] == {"enable_thinking": False}


def test_shared_body_still_used_when_summary_body_unset():
    p = _payload(None, {"thinking": {"type": "disabled"}})
    assert p["thinking"] == {"type": "disabled"}
    assert "chat_template_kwargs" not in p
