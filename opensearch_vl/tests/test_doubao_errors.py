"""image_search must retry provider-side failures and stop only on account/key errors."""

import io
import sys
from pathlib import Path
from unittest import mock

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opensearch_infer import search  # noqa: E402


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(buf, format="PNG")
    return buf.getvalue()


def _resp(code, http=200):
    r = mock.Mock(status_code=http)
    r.json.return_value = {"Result": {"ErrorCode": code, "Documents": []}}
    return r


@pytest.mark.parametrize("code", [500, "500", 502, 10500, 700429])
def test_server_side_codes_are_retried(code):
    with mock.patch.object(search.requests, "post", return_value=_resp(code)) as post, \
            mock.patch.object(search.time, "sleep"):
        out = search.image_search_bytes(_png(), attempts=3)
    assert post.call_count == 3
    assert out.startswith("Tool execution error")


def test_transient_then_success():
    replies = [_resp(500), _resp(0)]
    with mock.patch.object(search.requests, "post", side_effect=replies), \
            mock.patch.object(search.time, "sleep"):
        out = search.image_search_bytes(_png(), attempts=3)
    assert out == "Tool execution result:\nNo visually similar results found."


def test_account_error_stops_the_run():
    with mock.patch.object(search.requests, "post", return_value=_resp(10403)), \
            mock.patch.object(search.time, "sleep"):
        with pytest.raises(search.FatalServiceError):
            search.image_search_bytes(_png(), attempts=3)


def test_bad_input_is_a_tool_error():
    with mock.patch.object(search.requests, "post", return_value=_resp(10400)) as post, \
            mock.patch.object(search.time, "sleep"):
        out = search.image_search_bytes(_png(), attempts=3)
    assert post.call_count == 1
    assert out.startswith("Tool execution error")
