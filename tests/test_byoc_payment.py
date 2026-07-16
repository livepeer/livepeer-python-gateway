"""
Unit tests for _create_byoc_payment orch-discovery capabilities threading.

When ByocPerCapPricing is enabled on the signer, TicketParams must be issued
via PriceInfoForCaps — that requires passing capabilities into get_orch_info().
"""

from __future__ import annotations

import base64
import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch
from urllib.request import Request

from livepeer_gateway.byoc import _create_byoc_payment, _payment_type_for_signer
from livepeer_gateway.capabilities import CapabilityId, byoc_capabilities_from_app


def _stub_orch_info():
    info = MagicMock()
    tp = MagicMock()
    tp.face_value = b"\x01\x00"
    info.ticket_params = tp
    info.HasField = lambda field: field == "ticket_params"
    info.SerializeToString = lambda: b"stub-orch-info-protobuf"
    return info


@contextmanager
def _mock_payment_http(*, get_orch_info_side_effect):
    captured: list[Request] = []
    get_orch_info_calls: list[dict] = []

    class _MockResponse:
        def __init__(self, body: bytes, status: int = 200):
            self._body = body
            self.status = status
            self.headers = {}

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def _fake_get_orch_info(*args, **kwargs):
        get_orch_info_calls.append(kwargs)
        return get_orch_info_side_effect(*args, **kwargs)

    def _fake_urlopen(req, *args, **kwargs):
        captured.append(req)
        return _MockResponse(
            json.dumps({"payment": "TICKETS_B64", "segCreds": "SEG_B64"}).encode()
        )

    with patch("livepeer_gateway.byoc.urlopen", side_effect=_fake_urlopen), \
         patch("livepeer_gateway.orch_info.get_orch_info", side_effect=_fake_get_orch_info):
        yield captured, get_orch_info_calls


def test_create_byoc_payment_passes_capabilities_to_get_orch_info():
    capability = "flux-schnell"
    expected_caps = byoc_capabilities_from_app(capability)

    with _mock_payment_http(get_orch_info_side_effect=lambda *a, **k: _stub_orch_info()) as (
        _reqs,
        orch_calls,
    ):
        result = _create_byoc_payment(
            orch_origin="https://byoc-staging-1.daydream.monster:8936",
            capability=capability,
            livepeer_hdr="ignored",
            signer_url="https://signer.test",
            signer_headers={"Authorization": "Bearer sk_test"},
        )

    assert result["Livepeer-Payment"] == "TICKETS_B64"
    assert len(orch_calls) == 1
    passed_caps = orch_calls[0]["capabilities"]
    assert passed_caps is not None
    assert passed_caps.capacities[int(CapabilityId.BYOC)] == 1
    assert (
        "flux-schnell"
        in passed_caps.constraints.PerCapability[int(CapabilityId.BYOC)].models
    )
    assert passed_caps.SerializeToString() == expected_caps.SerializeToString()


def test_create_byoc_payment_includes_capabilities_on_signer_payload():
    capability = "flux-dev"

    with _mock_payment_http(get_orch_info_side_effect=lambda *a, **k: _stub_orch_info()) as (
        reqs,
        _orch_calls,
    ):
        _create_byoc_payment(
            orch_origin="https://byoc-staging-1.daydream.monster:8936",
            capability=capability,
            livepeer_hdr="ignored",
            signer_url="https://signer.test",
            signer_headers={"Authorization": "Bearer sk_test"},
        )

    signer_req = [r for r in reqs if "generate-live-payment" in r.full_url][0]
    payload = json.loads(signer_req.data.decode("utf-8"))
    assert payload["type"] == "byoc"
    assert "capabilities" in payload

    caps = byoc_capabilities_from_app(capability)
    assert payload["capabilities"] == base64.b64encode(
        caps.SerializeToString()
    ).decode("ascii")


def test_payment_type_for_signer_legacy_daydream():
    assert _payment_type_for_signer("https://signer.daydream.live") == "lv2v"
    assert _payment_type_for_signer("https://signer.daydream.live/generate-live-payment") == "lv2v"


def test_payment_type_for_signer_pymthouse_dmz():
    assert (
        _payment_type_for_signer(
            "https://pymthouse-production.up.railway.app"
        )
        == "byoc"
    )
    assert _payment_type_for_signer("https://signer.test") == "byoc"


def test_create_byoc_payment_legacy_daydream_signer_uses_lv2v():
    capability = "flux-schnell"

    with _mock_payment_http(get_orch_info_side_effect=lambda *a, **k: _stub_orch_info()) as (
        reqs,
        orch_calls,
    ):
        _create_byoc_payment(
            orch_origin="https://byoc-staging-1.daydream.monster:8936",
            capability=capability,
            livepeer_hdr="ignored",
            signer_url="https://signer.daydream.live",
            signer_headers={"Authorization": "Bearer sk_test"},
        )

    assert len(orch_calls) == 1
    assert orch_calls[0]["capabilities"] is None

    signer_req = [r for r in reqs if "generate-live-payment" in r.full_url][0]
    payload = json.loads(signer_req.data.decode("utf-8"))
    assert payload["type"] == "lv2v"
    assert payload["capability"] == capability
    assert "capabilities" not in payload


def test_create_byoc_payment_pymthouse_signer_uses_byoc():
    capability = "flux-dev"

    with _mock_payment_http(get_orch_info_side_effect=lambda *a, **k: _stub_orch_info()) as (
        reqs,
        orch_calls,
    ):
        _create_byoc_payment(
            orch_origin="https://byoc-staging-1.daydream.monster:8936",
            capability=capability,
            livepeer_hdr="ignored",
            signer_url="https://pymthouse-production.up.railway.app",
            signer_headers={"Authorization": "Bearer app.pmth_test"},
        )

    assert len(orch_calls) == 1
    passed_caps = orch_calls[0]["capabilities"]
    assert passed_caps is not None
    assert (
        capability
        in passed_caps.constraints.PerCapability[int(CapabilityId.BYOC)].models
    )

    signer_req = [r for r in reqs if "generate-live-payment" in r.full_url][0]
    payload = json.loads(signer_req.data.decode("utf-8"))
    assert payload["type"] == "byoc"
    assert "capabilities" in payload
    assert "capability" not in payload
