"""
Unit tests for the diagnostic surfacing of a truncated signer response in
``_create_byoc_payment``.

Background: the remote signer (/generate-live-payment) can advertise a
Content-Length but close the connection early, so urllib raises
``http.client.IncompleteRead`` and the partial body — which carries the real
signer error, e.g. ``{"error":{"message":"..."}}`` — is discarded. These tests
pin the behaviour that the partial bytes are captured and surfaced in the raised
``LivepeerGatewayError`` instead of the opaque ``IncompleteRead(85, 108)``.
"""
from __future__ import annotations

import http.client
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

import pytest

from livepeer_gateway.byoc import _create_byoc_payment
from livepeer_gateway.errors import LivepeerGatewayError


def _stub_orch_info():
    """OrchestratorInfo with a non-zero face_value so payment is attempted."""
    info = MagicMock()
    tp = MagicMock()
    tp.face_value = b"\x01\x00"  # non-zero
    info.ticket_params = tp
    info.HasField = lambda field: field == "ticket_params"
    info.SerializeToString = lambda: b"stub-orch-info-protobuf"
    return info


def _call():
    return _create_byoc_payment(
        orch_origin="https://orch.test:8936",
        capability="nano-banana",
        livepeer_hdr="stub-livepeer-header",
        signer_url="https://signer.test",
        signer_headers={"Authorization": "Bearer sk_test"},
    )


def test_incomplete_read_surfaces_partial_signer_body():
    """A truncated 200 response surfaces the partial body, not IncompleteRead."""
    partial = b'{"error":{"message":"ticket sender has insufficient funds'
    expected_more = 108

    def _fake_urlopen(req, *args, **kwargs):
        raise http.client.IncompleteRead(partial, expected_more)

    with patch("livepeer_gateway.byoc.urlopen", side_effect=_fake_urlopen), \
         patch("livepeer_gateway.orch_info.get_orch_info",
               side_effect=lambda *a, **k: _stub_orch_info()):
        with pytest.raises(LivepeerGatewayError) as excinfo:
            _call()

    msg = str(excinfo.value)
    assert "signer truncated response" in msg
    assert f"{len(partial)} of {len(partial) + expected_more} bytes" in msg
    # The real signer error text must now be legible in the raised error.
    assert "ticket sender has insufficient funds" in msg
    # Chained from the original IncompleteRead for traceability.
    assert isinstance(excinfo.value.__cause__, http.client.IncompleteRead)


def test_http_error_with_truncated_body_still_surfaces_partial():
    """An error status whose body is ALSO truncated keeps the partial bytes."""
    partial = b'{"error":{"message":"wallet panic"}'

    def _fake_urlopen(req, *args, **kwargs):
        err = HTTPError(req.full_url, 500, "internal error", {}, None)
        err.read = lambda: (_ for _ in ()).throw(
            http.client.IncompleteRead(partial, 40)
        )
        raise err

    with patch("livepeer_gateway.byoc.urlopen", side_effect=_fake_urlopen), \
         patch("livepeer_gateway.orch_info.get_orch_info",
               side_effect=lambda *a, **k: _stub_orch_info()):
        with pytest.raises(LivepeerGatewayError) as excinfo:
            _call()

    msg = str(excinfo.value)
    assert "HTTP 500" in msg
    assert "wallet panic" in msg
