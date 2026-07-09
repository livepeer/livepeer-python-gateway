"""
Unit tests for the shared HTTP helper's handling of truncated responses.

Background: ``http.client.IncompleteRead`` subclasses ``HTTPException`` — NOT
``HTTPError`` / ``URLError`` / ``OSError`` — so a truncated response (endpoint
advertises a Content-Length but closes the connection early) slips past every
handler in ``orchestrator.request_json`` and previously surfaced as an opaque
``IncompleteRead(85, 108)`` with the partial body (which carries the real
error, e.g. ``{"error":{"message":"..."}}``) discarded.

These tests pin the hardening that generalizes PR #38's fix to the shared
helper, covering both the success-body read and the error-body read.
"""
from __future__ import annotations

import http.client
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

from livepeer_gateway.errors import LivepeerGatewayError
from livepeer_gateway.orchestrator import request_json


class _TruncatingResponse:
    """Context-manager response whose .read() raises IncompleteRead."""

    def __init__(self, partial: bytes, expected_more: int):
        self._exc = http.client.IncompleteRead(partial, expected_more)

    def read(self):
        raise self._exc

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_request_json_surfaces_truncated_success_body():
    """A truncated 200 body surfaces the partial bytes, not IncompleteRead."""
    partial = b'{"error":{"message":"ticket sender has insufficient funds'
    expected_more = 108
    url = "https://orch.test/discover"

    def _fake_urlopen(req, *args, **kwargs):
        return _TruncatingResponse(partial, expected_more)

    with patch("livepeer_gateway.orchestrator.urlopen", side_effect=_fake_urlopen):
        with pytest.raises(LivepeerGatewayError) as excinfo:
            request_json(url)

    msg = str(excinfo.value)
    assert "truncated response" in msg
    assert f"{len(partial)} of {len(partial) + expected_more} bytes" in msg
    assert "ticket sender has insufficient funds" in msg
    assert url in msg
    assert isinstance(excinfo.value.__cause__, http.client.IncompleteRead)


def test_request_json_keeps_truncated_error_body():
    """An error status whose body is ALSO truncated keeps the partial bytes."""
    partial = b'{"error":{"message":"wallet panic"}'
    url = "https://orch.test/discover"

    def _fake_urlopen(req, *args, **kwargs):
        err = HTTPError(url, 500, "internal error", {}, None)
        err.read = lambda: (_ for _ in ()).throw(
            http.client.IncompleteRead(partial, 40)
        )
        raise err

    with patch("livepeer_gateway.orchestrator.urlopen", side_effect=_fake_urlopen):
        with pytest.raises(LivepeerGatewayError) as excinfo:
            request_json(url)

    msg = str(excinfo.value)
    assert "HTTP 500" in msg
    # The real error text recovered from the partial body must be legible.
    assert "wallet panic" in msg
    assert url in msg
