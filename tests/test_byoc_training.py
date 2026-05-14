"""
Unit tests for submit_training_job sign + payment flow (PR-1).

Per design doc §11.1, P1-P3:
- P1: submit_training_job emits the same 4 headers as submit_byoc_job
- P2: submit_training_job with signer_url=None proceeds with empty creds
- P3: refresh_training_payment uses same signer key (covered in test_byoc_refresh.py)

These tests mock urllib.request.urlopen to capture the outgoing Request and
inspect headers/body without touching network.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch
from urllib.request import Request

import pytest

from livepeer_gateway.byoc import (
    ByocJobRequest,
    ByocTrainingRequest,
    submit_byoc_job,
    submit_training_job,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _stub_orch_info():
    """OrchestratorInfo-like object with non-zero ticket params."""
    info = MagicMock()
    tp = MagicMock()
    tp.face_value = b"\x01\x00"  # non-zero → payment generation proceeds
    info.ticket_params = tp
    info.HasField = lambda field: field == "ticket_params"
    info.SerializeToString = lambda: b"stub-orch-info-protobuf"
    return info


@contextmanager
def _mock_http(*, signer_responses, orch_response_status=200,
               orch_response_body=b'{"status":"submitted","job_id":"orch-123","status_url":"/process/job/orch-123"}'):
    """
    Mock urllib.request.urlopen used inside byoc.py. signer_responses are
    consumed in order on signer-host calls. Orch call returns the response_*.

    Captures every Request object on the yielded list for assertion.
    """
    captured_requests: list[Request] = []

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

    signer_iter = iter(signer_responses)

    def _fake_urlopen(req, *args, **kwargs):
        captured_requests.append(req)
        url = req.full_url if hasattr(req, "full_url") else req.get_full_url()

        if "signer" in url or "/sign-byoc-job" in url or "/generate-live-payment" in url:
            try:
                payload = next(signer_iter)
            except StopIteration:
                payload = {"sender": "0xMOCKSENDER", "signature": "0xMOCKSIG"}
            return _MockResponse(json.dumps(payload).encode())
        return _MockResponse(orch_response_body, orch_response_status)

    with patch("livepeer_gateway.byoc.urlopen", side_effect=_fake_urlopen), \
         patch("livepeer_gateway.byoc._resolve_orchestrators",
               return_value=["https://orch.test:8935"]), \
         patch("livepeer_gateway.orch_info.get_orch_info",
               side_effect=lambda *a, **k: _stub_orch_info()):
        yield captured_requests


# ---------------------------------------------------------------------------
# P1 — header parity between training and inference paths
# ---------------------------------------------------------------------------


def test_p1_training_job_emits_same_headers_as_inference():
    """
    Submit a training job AND an inference job with the same signer config.
    Both must produce the same 4 critical headers on the orch request.
    """
    # 1. inference path
    with _mock_http(signer_responses=[
        {"sender": "0xWALLET1", "signature": "0xSIGABC"},
        {"payment": "TICKETS_B64", "segCreds": "SEG_B64"},
    ]) as inf_reqs:
        submit_byoc_job(
            req=ByocJobRequest(capability="flux-dev", payload={"prompt": "x"}, job_id="job-inf-1"),
            orch_url="https://orch.test:8935",
            signer_url="https://signer.test",
            signer_headers={"Authorization": "Bearer sk_test"},
        )

    inf_orch_req = [r for r in inf_reqs if "process/request" in r.full_url][0]
    inf_headers = {k.lower(): v for k, v in inf_orch_req.header_items()}

    # 2. training path
    with _mock_http(
        signer_responses=[
            {"sender": "0xWALLET1", "signature": "0xSIGABC"},
            {"payment": "TICKETS_B64", "segCreds": "SEG_B64"},
        ],
        orch_response_status=202,
    ) as tr_reqs:
        submit_training_job(
            req=ByocTrainingRequest(
                capability="flux-lora-training",
                model_id="flux-dev",
                params={"images_data_url": "https://x/zip", "trigger_word": "TOK", "steps": 10},
            ),
            orch_url="https://orch.test:8935",
            signer_url="https://signer.test",
            signer_headers={"Authorization": "Bearer sk_test"},
        )

    tr_orch_req = [r for r in tr_reqs if "process/train" in r.full_url][0]
    tr_headers = {k.lower(): v for k, v in tr_orch_req.header_items()}

    required = {"livepeer", "livepeer-capability", "livepeer-payment", "livepeer-segment"}
    missing_inf = required - inf_headers.keys()
    missing_tr = required - tr_headers.keys()

    assert not missing_inf, f"inference path missing headers: {missing_inf}"
    assert not missing_tr, f"training path missing headers: {missing_tr}"

    assert inf_headers["livepeer-payment"], "inference Livepeer-Payment was empty"
    assert tr_headers["livepeer-payment"], "training Livepeer-Payment was empty"
    assert inf_headers["livepeer-segment"], "inference Livepeer-Segment was empty"
    assert tr_headers["livepeer-segment"], "training Livepeer-Segment was empty"


# ---------------------------------------------------------------------------
# P2 — offchain mode (no signer_url) does NOT emit payment headers
# ---------------------------------------------------------------------------


def test_p2_training_no_signer_proceeds_unsigned():
    """
    With signer_url=None, training submit must skip signing AND payment,
    and still POST to orch. Mirrors submit_byoc_job's behavior.
    """
    with _mock_http(
        signer_responses=[],
        orch_response_status=202,
    ) as reqs:
        submit_training_job(
            req=ByocTrainingRequest(
                capability="flux-lora-training",
                model_id="flux-dev",
                params={"images_data_url": "https://x/zip", "trigger_word": "TOK", "steps": 10},
            ),
            orch_url="https://orch.test:8935",
            signer_url=None,
        )

    signer_calls = [r for r in reqs if "signer" in r.full_url or "sign-byoc-job" in r.full_url]
    assert signer_calls == [], (
        f"unexpected signer calls in offchain mode: {[r.full_url for r in signer_calls]}"
    )

    orch_calls = [r for r in reqs if "process/train" in r.full_url]
    assert len(orch_calls) == 1, "expected exactly one /process/train POST"

    headers = {k.lower(): v for k, v in orch_calls[0].header_items()}
    assert "livepeer" in headers
    assert headers.get("livepeer-capability") == "flux-lora-training"
    assert "livepeer-payment" not in headers, "Livepeer-Payment leaked into offchain submit"
    assert "livepeer-segment" not in headers, "Livepeer-Segment leaked into offchain submit"


# ---------------------------------------------------------------------------
# Bonus — body shape
# ---------------------------------------------------------------------------


def test_training_body_includes_model_id_and_params():
    """submit_training_job sends model_id + params at top level of body."""
    with _mock_http(
        signer_responses=[
            {"sender": "0xW", "signature": "0xS"},
            {"payment": "T", "segCreds": "S"},
        ],
        orch_response_status=202,
    ) as reqs:
        submit_training_job(
            req=ByocTrainingRequest(
                capability="flux-lora-training",
                model_id="flux-dev",
                params={"images_data_url": "https://x/zip", "trigger_word": "PULSEX1", "steps": 1000},
            ),
            orch_url="https://orch.test:8935",
            signer_url="https://signer.test",
            signer_headers={"Authorization": "Bearer sk_test"},
        )

    orch_req = [r for r in reqs if "process/train" in r.full_url][0]
    body = json.loads(orch_req.data.decode())
    assert body["model_id"] == "flux-dev"
    assert body["params"]["images_data_url"] == "https://x/zip"
    assert body["params"]["trigger_word"] == "PULSEX1"
    assert body["params"]["steps"] == 1000
