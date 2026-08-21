"""
Signer/orch HTTPError bodies must never mask the status code.

Live incident (2026-08-21, cjob_0a9056941b2a): the signer rejected a dead
Daydream key at /generate-live-payment with 401 and a 193-byte JSON body.
Because the signer answers before consuming the multi-KB POST body and then
closes the connection, the client's buffered response can be truncated —
`e.read()` inside the `except HTTPError` handler raised
`http.client.IncompleteRead(84 bytes read, 109 more expected)` (84+109=193),
which escaped the handler and surfaced to users as
"payment failed: IncompleteRead(...)" — with no trace of the 401. Downstream
that classified as "GPU network briefly busy — retry", the exact opposite of
a permanent per-key auth failure.

These tests pin the two guarantees of the fix:
  1. `_read_http_error_body` never raises and salvages partial bytes.
  2. The signer paths report the status code first ("signer rejected key:
     HTTP 401: ...") even when the body read dies mid-flight.
"""

import io
import json
from http.client import IncompleteRead
from urllib.error import HTTPError

import pytest

from livepeer_gateway.byoc import _read_http_error_body, _sign_byoc_job
from livepeer_gateway.errors import LivepeerGatewayError

SIGNER_401_BODY = (
    b'{"success":false,"error":"Authentication failed","code":"AUTH/FAILED",'
    b'"status":401,"details":{"cause":"Invalid access token"}}'
)


def _http_error(code: int, fp) -> HTTPError:
    return HTTPError("https://signer.example/generate-live-payment", code, "x", {}, fp)


class _TruncatingBody(io.RawIOBase):
    """A body whose read dies mid-flight, like a connection reset."""

    def __init__(self, partial: bytes):
        self._partial = partial

    def read(self, *a):  # noqa: ANN002 - match file-like signature
        raise IncompleteRead(self._partial, expected=109)


class _ExplodingBody(io.RawIOBase):
    def read(self, *a):  # noqa: ANN002
        raise ConnectionResetError("peer reset")


class TestReadHttpErrorBody:
    def test_reads_a_healthy_body(self):
        e = _http_error(403, io.BytesIO(b'{"error":"nope"}'))
        assert _read_http_error_body(e) == '{"error":"nope"}'

    def test_salvages_incomplete_read_partial(self):
        # The real failure: 84 of 193 bytes arrive before the reset. The
        # salvaged prefix still names the failure ("Authentication failed").
        e = _http_error(401, _TruncatingBody(SIGNER_401_BODY[:84]))
        body = _read_http_error_body(e)
        assert "Authentication failed" in body

    def test_never_raises_even_with_no_salvageable_bytes(self):
        e = _http_error(401, _ExplodingBody())
        body = _read_http_error_body(e)
        assert "ConnectionResetError" in body

    def test_truncates_to_limit(self):
        e = _http_error(500, io.BytesIO(b"x" * 1000))
        assert len(_read_http_error_body(e, limit=200)) == 200


class TestSignByocJobSignerRejection:
    """End-to-end through a real except-handler: the status must survive."""

    def _run(self, monkeypatch, error: HTTPError) -> LivepeerGatewayError:
        def fake_urlopen(req, timeout=None, context=None):
            raise error

        monkeypatch.setattr("livepeer_gateway.byoc.urlopen", fake_urlopen)
        with pytest.raises(LivepeerGatewayError) as exc_info:
            _sign_byoc_job(
                signer_url="https://signer.example",
                signer_headers=None,
                job_id="job-1",
                capability="flux-schnell",
                request_json="{}",
                parameters_json="",
                timeout_seconds=30,
            )
        return exc_info.value

    def test_401_with_truncated_body_reports_signer_rejected_key(self, monkeypatch):
        err = self._run(
            monkeypatch, _http_error(401, _TruncatingBody(SIGNER_401_BODY[:84]))
        )
        msg = str(err)
        assert "signer rejected key" in msg
        assert "HTTP 401" in msg
        assert "IncompleteRead" not in msg.split("HTTP 401")[0]  # status leads

    def test_403_keeps_the_existing_message_shape(self, monkeypatch):
        # Downstream classifiers match "failed: HTTP 403" for the
        # out-of-credits case — that shape must not change.
        err = self._run(
            monkeypatch,
            _http_error(403, io.BytesIO(b'{"error":{"message":"signer auth rejected request with status 403"}}')),
        )
        msg = str(err)
        assert "HTTP 403" in msg
        assert "signer auth rejected" in msg

    def test_500_reports_status_and_body(self, monkeypatch):
        err = self._run(monkeypatch, _http_error(500, io.BytesIO(b"boom")))
        msg = str(err)
        assert "HTTP 500" in msg and "boom" in msg
