"""
Unit tests for refresh_training_payment helper (PR-2).

Per design doc §11.1:
- P3: refresh_training_payment uses the same signer key as submit
  (signer wallet doesn't change mid-job) → tested via header inspection
- Plus: idempotency invariant (I5 — duplicate refresh credits only once)
- Plus: 3-attempt retry on transient errors
- Plus: fail-fast on permanent 4xx
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from livepeer_gateway.byoc import refresh_training_payment
from livepeer_gateway.errors import LivepeerGatewayError


def _stub_orch_info():
    info = MagicMock()
    tp = MagicMock()
    tp.face_value = b"\x01\x00"  # non-zero
    info.ticket_params = tp
    info.HasField = lambda field: field == "ticket_params"
    info.SerializeToString = lambda: b"stub-orch-info-protobuf"
    return info


def _stub_orch_info_zero_price():
    """ticket_params with face_value=0 → signer says 'no payment needed'."""
    info = MagicMock()
    tp = MagicMock()
    tp.face_value = b"\x00"
    info.ticket_params = tp
    info.HasField = lambda field: field == "ticket_params"
    info.SerializeToString = lambda: b"stub"
    return info


@contextmanager
def _mock_http(*, signer_responses, orch_response_status=200,
               orch_response_body=b'{"credited_wei":"1000","new_balance_wei":"2500"}',
               orch_info=None):
    captured: list[Request] = []

    class _MockResponse:
        def __init__(self, body: bytes, status: int = 200):
            self._body = body
            self.status = status
            self.headers = {}
        def read(self): return self._body
        def __enter__(self): return self
        def __exit__(self, *a): return False

    signer_iter = iter(signer_responses)

    def _fake_urlopen(req, *args, **kwargs):
        captured.append(req)
        url = req.full_url if hasattr(req, "full_url") else req.get_full_url()
        if "signer" in url or "/generate-live-payment" in url:
            try: payload = next(signer_iter)
            except StopIteration: payload = {"payment": "MORE_TICKETS", "segCreds": "MORE_SEG"}
            return _MockResponse(json.dumps(payload).encode())
        return _MockResponse(orch_response_body, orch_response_status)

    info = orch_info or _stub_orch_info()
    with patch("livepeer_gateway.byoc.urlopen", side_effect=_fake_urlopen), \
         patch("livepeer_gateway.orch_info.get_orch_info",
               side_effect=lambda *a, **k: info):
        yield captured


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_refresh_happy_path_credits_orch():
    """Refresh POSTs to /process/job/<job_id>/refresh-payment with payment headers."""
    with _mock_http(
        signer_responses=[{"payment": "FRESH_TICKETS", "segCreds": "FRESH_SEG"}],
    ) as reqs:
        result = refresh_training_payment(
            job_id="train-abc",
            orch_url="https://orch.test:8935",
            capability="flux-lora-training",
            signer_url="https://signer.test",
            signer_headers={"Authorization": "Bearer sk_test"},
        )

    refresh_req = [r for r in reqs if "refresh-payment" in r.full_url][0]
    headers = {k.lower(): v for k, v in refresh_req.header_items()}

    assert "/process/job/train-abc/refresh-payment" in refresh_req.full_url
    assert headers["livepeer-payment"] == "FRESH_TICKETS"
    assert headers["livepeer-segment"] == "FRESH_SEG"
    assert refresh_req.method == "POST"
    assert result["credited_wei"] == "1000"


# ---------------------------------------------------------------------------
# P3 — refresh uses same signer wallet as submit
# ---------------------------------------------------------------------------


def test_p3_refresh_uses_same_signer_headers_as_submit():
    """
    Bearer header forwarded to signer is the SAME as caller passed in.
    This ensures the signer resolves to the same wallet on submit + refresh
    (per Invariant I6 — sender attribution).
    """
    bearer = "Bearer sk_user_pulsex1"
    with _mock_http(
        signer_responses=[{"payment": "F", "segCreds": "S"}],
    ) as reqs:
        refresh_training_payment(
            job_id="train-abc",
            orch_url="https://orch.test:8935",
            capability="flux-lora-training",
            signer_url="https://signer.test",
            signer_headers={"Authorization": bearer},
        )

    signer_reqs = [r for r in reqs if "generate-live-payment" in r.full_url]
    assert len(signer_reqs) == 1
    forwarded = {k.lower(): v for k, v in signer_reqs[0].header_items()}
    assert forwarded.get("authorization") == bearer, (
        f"signer didn't receive caller's bearer; got {forwarded.get('authorization')!r}"
    )


# ---------------------------------------------------------------------------
# Zero-price case — refresh is no-op
# ---------------------------------------------------------------------------


def test_refresh_zero_price_is_noop():
    """If signer says face_value=0, the orch refresh POST is skipped (no-op)."""
    with _mock_http(
        signer_responses=[{"payment": "", "segCreds": ""}],  # ignored
        orch_info=_stub_orch_info_zero_price(),
    ) as reqs:
        result = refresh_training_payment(
            job_id="train-zero",
            orch_url="https://orch.test:8935",
            capability="flux-lora-training",
            signer_url="https://signer.test",
            signer_headers={"Authorization": "Bearer sk_test"},
        )

    refresh_calls = [r for r in reqs if "refresh-payment" in r.full_url]
    assert refresh_calls == [], "refresh fired against orch despite zero-price"
    assert result.get("noop") == "true"
    assert result["credited_wei"] == "0"


# ---------------------------------------------------------------------------
# Retry on transient orch error
# ---------------------------------------------------------------------------


def test_refresh_retries_on_503():
    """Transient 503 from orch → retry (up to max_attempts), eventually succeed.

    Reviewer I2 strengthening: also verify that signer is called ONCE
    (not once per retry) and that all retry attempts send the SAME
    Livepeer-Payment header value (no re-minting → no nonce drift).
    """
    call_count = {"signer": 0, "orch": 0}
    orch_payment_headers: list[str] = []

    class _MockResponse:
        def __init__(self, body, status=200):
            self._body, self.status = body, status
            self.headers = {}
        def read(self): return self._body
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def _fake_urlopen(req, *args, **kwargs):
        url = req.full_url if hasattr(req, "full_url") else req.get_full_url()
        if "generate-live-payment" in url:
            call_count["signer"] += 1
            return _MockResponse(json.dumps({"payment": "PINNED_T", "segCreds": "PINNED_S"}).encode())
        # orch refresh — capture the Livepeer-Payment header so we can
        # assert all retries used the same ticket batch
        hdrs = {k.lower(): v for k, v in req.header_items()}
        orch_payment_headers.append(hdrs.get("livepeer-payment", ""))
        call_count["orch"] += 1
        if call_count["orch"] < 3:
            raise HTTPError(url, 503, "service unavailable", {}, None)
        return _MockResponse(b'{"credited_wei":"500"}', 200)

    with patch("livepeer_gateway.byoc.urlopen", side_effect=_fake_urlopen), \
         patch("livepeer_gateway.orch_info.get_orch_info",
               side_effect=lambda *a, **k: _stub_orch_info()):
        result = refresh_training_payment(
            job_id="train-retry",
            orch_url="https://orch.test:8935",
            capability="flux-lora-training",
            signer_url="https://signer.test",
            signer_headers={"Authorization": "Bearer sk_test"},
            max_attempts=3,
        )

    assert call_count["orch"] == 3, f"expected 3 orch attempts, got {call_count['orch']}"
    assert call_count["signer"] == 1, (
        f"signer was called {call_count['signer']} times; should mint ONCE outside retry "
        "loop to avoid nonce drift (Invariant I5)"
    )
    assert len(set(orch_payment_headers)) == 1, (
        f"orch attempts sent different Livepeer-Payment headers: {orch_payment_headers}; "
        "all retries must reuse the same ticket batch"
    )
    assert orch_payment_headers[0] == "PINNED_T", (
        f"expected pinned ticket value 'PINNED_T', got {orch_payment_headers[0]!r}"
    )
    assert result["credited_wei"] == "500"


# ---------------------------------------------------------------------------
# Fail-fast on permanent 4xx (e.g., bad job_id, expired token)
# ---------------------------------------------------------------------------


def test_refresh_fails_fast_on_permanent_4xx():
    """403/404 from orch → no retry, raise immediately."""
    call_count = {"orch": 0}

    class _MockResponse:
        def __init__(self, body, status=200):
            self._body, self.status = body, status
            self.headers = {}
        def read(self): return self._body
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def _fake_urlopen(req, *args, **kwargs):
        url = req.full_url if hasattr(req, "full_url") else req.get_full_url()
        if "generate-live-payment" in url:
            return _MockResponse(json.dumps({"payment": "T", "segCreds": "S"}).encode())
        call_count["orch"] += 1
        err = HTTPError(url, 403, "sender mismatch", {}, None)
        # Make HTTPError.read() return useful body
        err.read = lambda: b"sender mismatch with original submit"
        raise err

    with patch("livepeer_gateway.byoc.urlopen", side_effect=_fake_urlopen), \
         patch("livepeer_gateway.orch_info.get_orch_info",
               side_effect=lambda *a, **k: _stub_orch_info()):
        with pytest.raises(LivepeerGatewayError) as excinfo:
            refresh_training_payment(
                job_id="train-403",
                orch_url="https://orch.test:8935",
                capability="flux-lora-training",
                signer_url="https://signer.test",
                signer_headers={"Authorization": "Bearer sk_test"},
                max_attempts=3,
            )

    assert "permanent failure" in str(excinfo.value).lower()
    assert call_count["orch"] == 1, "fail-fast on 403 should not retry"


# ---------------------------------------------------------------------------
# Exhaustion — all 3 attempts fail
# ---------------------------------------------------------------------------


def test_refresh_exhausts_retries():
    """All 3 attempts fail with 503 → raise LivepeerGatewayError."""
    class _MockResponse:
        def __init__(self, body, status=200):
            self._body, self.status = body, status
            self.headers = {}
        def read(self): return self._body
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def _fake_urlopen(req, *args, **kwargs):
        url = req.full_url if hasattr(req, "full_url") else req.get_full_url()
        if "generate-live-payment" in url:
            return _MockResponse(json.dumps({"payment": "T", "segCreds": "S"}).encode())
        raise HTTPError(url, 503, "always down", {}, None)

    with patch("livepeer_gateway.byoc.urlopen", side_effect=_fake_urlopen), \
         patch("livepeer_gateway.orch_info.get_orch_info",
               side_effect=lambda *a, **k: _stub_orch_info()):
        with pytest.raises(LivepeerGatewayError) as excinfo:
            refresh_training_payment(
                job_id="train-doomed",
                orch_url="https://orch.test:8935",
                capability="flux-lora-training",
                signer_url="https://signer.test",
                signer_headers={"Authorization": "Bearer sk_test"},
                max_attempts=3,
            )

    assert "exhausted" in str(excinfo.value).lower()
