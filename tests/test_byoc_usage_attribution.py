"""
Unit tests for BYOC usage attribution (real capability + model_id).

Root cause being fixed: ``_create_byoc_payment`` used to send only a bare
``capability`` field (which the remote signer dropped) and no model id, so the
signer's ``create_signed_ticket`` metering event recorded
``pipeline=live-video-to-video`` / ``model_id=unknown`` for every BYOC job.

These tests prove the additive wire contract on the gateway side:
- the ``/generate-live-payment`` body now carries the REAL ``capability`` and
  ``model_id`` for a representative BYOC capability (e.g. ``nano-banana``);
- ``type`` stays ``"lv2v"`` (fee/pixel routing is unchanged);
- when no model id can be determined the field is omitted (backward-compatible,
  byte-identical to today so the signer falls back).
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch
from urllib.request import Request

from livepeer_gateway.byoc import (
    ByocJobRequest,
    _create_byoc_payment,
    _extract_model_id,
)


def _stub_orch_info():
    info = MagicMock()
    tp = MagicMock()
    tp.face_value = b"\x01\x00"  # non-zero → payment required
    info.ticket_params = tp
    info.HasField = lambda field: field == "ticket_params"
    info.SerializeToString = lambda: b"stub-orch-info-protobuf"
    return info


@contextmanager
def _mock_signer(*, payment_body=b'{"payment":"TICKETS","segCreds":"SEG"}'):
    """Mock the signer HTTP call and capture the request sent to it."""
    captured: list[Request] = []

    class _MockResponse:
        def __init__(self, body: bytes):
            self._body = body

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _fake_urlopen(req, *args, **kwargs):
        captured.append(req)
        return _MockResponse(payment_body)

    with patch("livepeer_gateway.byoc.urlopen", side_effect=_fake_urlopen), patch(
        "livepeer_gateway.orch_info.get_orch_info",
        side_effect=lambda *a, **k: _stub_orch_info(),
    ):
        yield captured


def _signer_body(captured: list[Request]) -> dict:
    signer_reqs = [
        r for r in captured if "generate-live-payment" in r.full_url
    ]
    assert len(signer_reqs) == 1, f"expected 1 signer call, got {len(signer_reqs)}"
    return json.loads(signer_reqs[0].data.decode("utf-8"))


# ---------------------------------------------------------------------------
# _extract_model_id
# ---------------------------------------------------------------------------


def test_extract_model_id_from_payload_model_id_key():
    assert _extract_model_id({"model_id": "google/nano-banana"}) == "google/nano-banana"


def test_extract_model_id_from_payload_model_key():
    assert _extract_model_id({"prompt": "a cat", "model": "flux-dev"}) == "flux-dev"


def test_extract_model_id_prefers_payload_over_parameters():
    assert (
        _extract_model_id({"model_id": "from-payload"}, {"model_id": "from-params"})
        == "from-payload"
    )


def test_extract_model_id_falls_back_to_parameters():
    assert _extract_model_id({"prompt": "x"}, {"model": "from-params"}) == "from-params"


def test_extract_model_id_empty_when_absent():
    assert _extract_model_id({"prompt": "no model here"}) == ""
    assert _extract_model_id(None, None) == ""
    assert _extract_model_id({"model_id": "   "}) == ""  # whitespace-only ignored


# ---------------------------------------------------------------------------
# _create_byoc_payment wire contract
# ---------------------------------------------------------------------------


def test_payment_body_carries_real_capability_and_model_id():
    """A representative BYOC capability sends the REAL capability + model_id."""
    with _mock_signer() as captured:
        result = _create_byoc_payment(
            orch_origin="https://orch.test:8936",
            capability="nano-banana",
            livepeer_hdr="",
            signer_url="https://signer.test",
            signer_headers={"Authorization": "Bearer sk_test"},
            model_id="google/nano-banana",
        )

    body = _signer_body(captured)
    assert body["capability"] == "nano-banana"
    assert body["model_id"] == "google/nano-banana"
    # type stays lv2v so fee/pixel routing is unchanged
    assert body["type"] == "lv2v"
    assert body["orchestrator"]  # base64 orch info present
    # payment headers still returned as before
    assert result["Livepeer-Payment"] == "TICKETS"
    assert result["Livepeer-Segment"] == "SEG"


def test_payment_body_omits_model_id_when_absent():
    """Backward-compatible: no model id → field omitted (signer falls back)."""
    with _mock_signer() as captured:
        _create_byoc_payment(
            orch_origin="https://orch.test:8936",
            capability="nano-banana",
            livepeer_hdr="",
            signer_url="https://signer.test",
            signer_headers={"Authorization": "Bearer sk_test"},
            # model_id omitted → defaults to ""
        )

    body = _signer_body(captured)
    assert body["capability"] == "nano-banana"
    assert "model_id" not in body, "empty model_id must be omitted for byte-identical body"
    assert body["type"] == "lv2v"


# ---------------------------------------------------------------------------
# End-to-end through submit_byoc_job
# ---------------------------------------------------------------------------


def test_submit_byoc_job_threads_model_id_from_payload():
    """submit_byoc_job extracts the model id from the request payload and
    forwards it (and the capability) to the signer payment request."""
    from livepeer_gateway import byoc as byoc_mod

    captured: list[Request] = []

    class _MockResponse:
        def __init__(self, body: bytes, status: int = 200):
            self._body = body
            self.status = status
            self.headers = {}

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _fake_urlopen(req, *args, **kwargs):
        captured.append(req)
        url = req.full_url if hasattr(req, "full_url") else req.get_full_url()
        if "generate-live-payment" in url:
            return _MockResponse(b'{"payment":"TICKETS","segCreds":"SEG"}')
        if "sign-byoc-job" in url:
            return _MockResponse(b'{"sender":"0xabc","signature":"0xsig"}')
        # orchestrator /process/request/<cap>
        return _MockResponse(b'{"images":[{"url":"https://img"}]}', 200)

    req = ByocJobRequest(
        capability="nano-banana",
        payload={"prompt": "a dragon", "model_id": "google/nano-banana"},
    )

    with patch("livepeer_gateway.byoc.urlopen", side_effect=_fake_urlopen), patch(
        "livepeer_gateway.orch_info.get_orch_info",
        side_effect=lambda *a, **k: _stub_orch_info(),
    ):
        byoc_mod.submit_byoc_job(
            req,
            orch_url="https://orch.test:8936",
            signer_url="https://signer.test",
            signer_headers={"Authorization": "Bearer sk_test"},
        )

    body = _signer_body(captured)
    assert body["capability"] == "nano-banana"
    assert body["model_id"] == "google/nano-banana"
    assert body["type"] == "lv2v"


# ---------------------------------------------------------------------------
# End-to-end through submit_training_job
# ---------------------------------------------------------------------------


def test_submit_training_job_uses_top_level_model_id_over_params():
    """submit_training_job must attribute payment to the explicit top-level
    training ``model_id`` (which also builds the orchestrator body), not a
    ``model_id`` nested in ``params`` that would otherwise win the payload
    merge and send the wrong model to the signer."""
    from livepeer_gateway import byoc as byoc_mod
    from livepeer_gateway.byoc import ByocTrainingRequest

    captured: list[Request] = []

    class _MockResponse:
        def __init__(self, body: bytes, status: int = 200):
            self._body = body
            self.status = status
            self.headers = {}

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _fake_urlopen(req, *args, **kwargs):
        captured.append(req)
        url = req.full_url if hasattr(req, "full_url") else req.get_full_url()
        if "generate-live-payment" in url:
            return _MockResponse(b'{"payment":"TICKETS","segCreds":"SEG"}')
        if "sign-byoc-job" in url:
            return _MockResponse(b'{"sender":"0xabc","signature":"0xsig"}')
        # orchestrator /process/train/<cap>
        return _MockResponse(b'{"job_id":"train-123","status":"submitted"}', 200)

    # Top-level training model differs from a stray params["model_id"].
    req = ByocTrainingRequest(
        capability="flux-lora-trainer",
        model_id="fal-ai/flux-lora-fast-training",
        params={"model_id": "wrong/params-model", "trigger_word": "TOK"},
    )

    with patch("livepeer_gateway.byoc.urlopen", side_effect=_fake_urlopen), patch(
        "livepeer_gateway.orch_info.get_orch_info",
        side_effect=lambda *a, **k: _stub_orch_info(),
    ):
        byoc_mod.submit_training_job(
            req,
            orch_url="https://orch.test:8936",
            signer_url="https://signer.test",
            signer_headers={"Authorization": "Bearer sk_test"},
        )

    body = _signer_body(captured)
    assert body["capability"] == "flux-lora-trainer"
    # The explicit training model_id wins — NOT the nested params value.
    assert body["model_id"] == "fal-ai/flux-lora-fast-training"
    assert body["type"] == "lv2v"
