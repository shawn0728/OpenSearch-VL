"""Offline integration checks at the serialized requests transport boundary."""

import importlib
import json
import os
import sys
import unittest
from contextlib import contextmanager
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

import run_infer
from opensearch_infer import auth, config, pipeline
from opensearch_infer.runners import InferenceConfig, MiniMaxRunner, build_runner


@contextmanager
def configured(**settings):
    try:
        with mock.patch.dict(os.environ, settings, clear=True):
            importlib.reload(config)
            yield
    finally:
        importlib.reload(config)


@contextmanager
def transport(*responses):
    """Capture PreparedRequest objects after requests performs JSON encoding."""
    captured = []

    def send(session, request, **kwargs):
        captured.append(request)
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(
            responses[min(len(captured), len(responses)) - 1]
        ).encode()
        return response

    with mock.patch.object(requests.Session, "send", send):
        yield captured


class FlashTests(unittest.TestCase):
    def test_current_choices_and_explicit_legacy_cli(self):
        parser = run_infer._build_arg_parser()
        names = config.list_model_names().split(", ")
        self.assertEqual(
            [name for name in names if name.startswith("minimax")],
            ["minimax-m3.1-flash", "minimax-m3"],
        )
        self.assertNotIn("minimax-m2.7", parser.format_help())
        required = ["--data-path", "fixture.parquet", "--output-dir", "outputs"]
        self.assertEqual(parser.parse_args(required).model, "8b")
        for name, model in (
            ("minimax-m3.1-flash", "MiniMax-M3.1-Flash"),
            ("minimax-m3", "MiniMax-M3"),
            ("minimax-m2.7", "MiniMax-M2.7"),
        ):
            with self.subTest(name=name):
                parsed = parser.parse_args(["--model", name] + required)
                self.assertEqual(build_runner(parsed.model).api_model, model)
        with self.assertRaises(ValueError):
            config.validate_model_name("unknown-model")
        flash = config.MODEL_REGISTRY["minimax-m3.1-flash"]
        self.assertEqual(flash.extra["context_window"], "1000000")
        self.assertEqual(flash.extra["max_output_tokens"], "524288")
        self.assertEqual(flash.extra["thinking"], "adaptive")

    def test_real_runner_serializes_defaults_regions_legacy_and_custom(self):
        response = {
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
        }
        for region, host in (
            ("global_en", "api.minimax.io"), ("cn_zh", "api.minimaxi.com"),
        ):
            for name, model in (
                ("minimax-m3.1-flash", "MiniMax-M3.1-Flash"),
                ("minimax-m3", "MiniMax-M3"),
                ("minimax-m2.7", "MiniMax-M2.7"),
            ):
                with (
                    self.subTest(region=region, model=model),
                    configured(MINIMAX_API_KEY="offline-fixture", MINIMAX_API_REGION=region),
                    transport(response) as captured,
                ):
                    runner = build_runner(name)
                    result = runner.infer([
                        {"role": "user", "parts": [{"text": "hello"}]},
                    ])
                    self.assertEqual(result["candidates"][0]["finishReason"], "end_turn")
                    self.assertEqual(len(captured), 1)
                    request = captured[0]
                    self.assertEqual(request.url, f"https://{host}/anthropic/v1/messages")
                    self.assertEqual(request.headers["Authorization"], "Bearer offline-fixture")
                    # No disabled thinking, effort=none or service_tier is sent.
                    self.assertEqual(json.loads(request.body), {
                        "model": model,
                        "max_tokens": 32768,
                        "messages": [{"role": "user", "content": [
                            {"type": "text", "text": "hello"},
                        ]}],
                    })

        custom = config.ModelSpec(
            name="saved-custom", family="minimax", display_name="saved-custom",
            extra={"api_model": "saved-custom-model"},
        )
        with (
            configured(
                MINIMAX_API_KEY="offline-fixture",
                MINIMAX_API_REGION="cn_zh",
                MINIMAX_ANTHROPIC_BASE_URL="https://custom.example/anthropic/",
            ),
            transport(response) as captured,
        ):
            MiniMaxRunner(custom).infer([
                {"role": "user", "parts": [{"text": "hello"}]},
            ])
            self.assertEqual(captured[0].url, "https://custom.example/anthropic/v1/messages")
            self.assertEqual(json.loads(captured[0].body)["model"], "saved-custom-model")

    def test_native_blocks_and_multimodal_history_reach_http_unchanged(self):
        blocks = [
            {"type": "thinking", "thinking": "reasoning", "signature": "opaque-signature"},
            {"type": "text", "text": "checking", "citations": []},
            {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {"q": "test"}},
        ]
        video = {
            "type": "video",
            "source": {"type": "url", "url": "https://example.com/clip.mp4"},
        }
        tool_result = {"type": "tool_result", "tool_use_id": "call_1", "content": "found"}
        runner = build_runner("minimax-m3.1-flash")
        runner.client = auth.MiniMaxClient("offline-fixture")
        contents = [{"role": "user", "parts": [
            {"text": "inspect"},
            {"image_url": {"url": "https://example.com/image.png"}},
            {"inline_data": {"mime_type": "image/png", "data": "data:image/png;base64,AAAA"}},
            {"minimax_block": video},
        ]}]
        with transport({"content": blocks, "stop_reason": "tool_use"}) as captured:
            result = runner.infer(contents, "system", InferenceConfig(max_tokens=256))
            contents.append(result["candidates"][0]["content"])
            contents.append({"role": "user", "parts": [{"minimax_block": tool_result}]})
            runner.infer(contents)
        first, second = [json.loads(request.body) for request in captured]
        self.assertEqual(first["system"], "system")
        self.assertEqual(first["max_tokens"], 256)
        self.assertEqual(first["messages"][0]["content"], [
            {"type": "text", "text": "inspect"},
            {"type": "image", "source": {"type": "url", "url": "https://example.com/image.png"}},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
            video,
        ])
        self.assertEqual(second["messages"][1], {"role": "assistant", "content": blocks})
        self.assertEqual(second["messages"][2]["content"], [tool_result])

    def test_agent_tool_loop_replays_thinking_without_parsing_it_as_tool_text(self):
        thinking = {
            "type": "thinking", "thinking": "<response>not final</response>",
            "signature": "opaque",
        }
        tool_text = '<tool_call>{"name":"text_search","arguments":{"query":"test"}}</tool_call>'
        blocks = [thinking, {"type": "text", "text": tool_text}]
        runner = build_runner("minimax-m3.1-flash")
        runner.client = auth.MiniMaxClient("offline-fixture")
        with (
            transport(
                {"content": blocks, "stop_reason": "end_turn"},
                {"content": [{"type": "text", "text": "<response>done</response>"}]},
            ) as captured,
            mock.patch.object(pipeline.tools, "execute_tool", return_value=("found", {})) as execute,
            mock.patch.object(pipeline.os, "makedirs"),
            mock.patch("builtins.open", mock.mock_open()),
        ):
            trajectory = pipeline.process_single_case(
                row={"data_id": "fixture", "prompt": [{"content": "question"}]},
                runner=runner, output_dir="outputs", case_idx=0,
            )
        execute.assert_called_once()
        self.assertEqual(len(captured), 2)
        second = json.loads(captured[1].body)
        self.assertEqual(second["messages"][1]["content"], blocks)
        self.assertEqual(second["messages"][2]["content"], [{
            "type": "text", "text": "<observation>\nfound\n</observation>",
        }])
        self.assertEqual(trajectory["turns"][0]["response_text"], tool_text)


if __name__ == "__main__":
    unittest.main()
