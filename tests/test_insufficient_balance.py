"""HTTP 483 (insufficient balance) must fail fast — no orch fallback."""
from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

import pytest

from livepeer_gateway.errors import InsufficientBalance, NoOrchestratorAvailableError
from livepeer_gateway.lv2v import StartJobRequest, start_lv2v
from livepeer_gateway.orchestrator import request_json


def test_request_json_maps_483_to_insufficient_balance() -> None:
    body = b'{"error":"Starter allowance exhausted"}'
    err = HTTPError(
        "https://signer.example/generate-live-payment",
        483,
        "Insufficient Balance",
        hdrs=None,  # type: ignore[arg-type]
        fp=BytesIO(body),
    )
    with patch("livepeer_gateway.orchestrator.urlopen", side_effect=err):
        with pytest.raises(InsufficientBalance, match="Starter allowance exhausted"):
            request_json(
                "https://signer.example/generate-live-payment",
                payload={},
            )


def test_start_lv2v_fails_fast_on_483_without_orch_fallback() -> None:
    info_a = MagicMock()
    info_a.transcoder = "https://orch-a.example:8935"
    info_b = MagicMock()
    info_b.transcoder = "https://orch-b.example:8935"

    cursor = MagicMock()
    cursor.next.side_effect = [
        ("https://orch-a.example:8935", info_a),
        ("https://orch-b.example:8935", info_b),
        NoOrchestratorAvailableError("exhausted", rejections=[]),
    ]

    payment_session = MagicMock()
    payment_session.get_payment.side_effect = InsufficientBalance(
        "Signer returned HTTP 483 (insufficient balance) "
        "(url=https://signer.example/generate-live-payment); "
        "body='Starter allowance exhausted'"
    )

    with (
        patch("livepeer_gateway.lv2v.orchestrator_selector", return_value=cursor),
        patch("livepeer_gateway.lv2v.PaymentSession", return_value=payment_session),
        patch("livepeer_gateway.lv2v.build_capabilities"),
    ):
        with pytest.raises(InsufficientBalance, match="Starter allowance exhausted"):
            start_lv2v(
                None,
                StartJobRequest(model_id="noop"),
                signer_url="https://signer.example",
                discovery_url="https://discovery.example/raw",
            )

    assert cursor.next.call_count == 1
    assert payment_session.get_payment.call_count == 1
