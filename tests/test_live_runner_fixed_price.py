"""Unit tests for fixed-price live-runner payment signalling.

A fixed-price single-shot live-runner generation advertises
``price_info.unit == "fixed"`` and the v0.9.0 orchestrator debits exactly one
unit per generation (``PixelsPerUnit == 1``). The gateway must signal that unit
count to the remote signer as ``inPixels:1`` on the ``lv2v``
``/generate-live-payment`` request; otherwise the signer falls back to the
continuous 720p30 estimate (720*1280*30*60 = 1,658,880,000 pixels) and inflates
the fee ~1.66e9x, blowing past the signer's max-100 ticket guard (observed e2e:
HTTP 400 "numTickets 2721947758 exceeds maximum of 100").

Pairs with go-livepeer PR #4006 (commit cd99507), which teaches the signer to
honor ``req.InPixels`` on the ``lv2v`` path. Both must ship together.

Continuous runners (``720p``/``hour``) must NOT set ``inPixels`` so the payload
stays byte-identical and the signer keeps its automatic estimate.
"""
from __future__ import annotations

import asyncio
from typing import Any, Optional
from unittest.mock import AsyncMock, patch

from livepeer_gateway.live_runner import (
    LiveRunnerInstance,
    _RunnerPaymentChallenge,
    _fixed_price_in_pixels,
    _get_runner_payment,
    _runner_price_unit,
)


def _runner(unit: Optional[str]) -> LiveRunnerInstance:
    """Build a discovered runner whose price_info advertises ``unit``.

    Passing ``unit=None`` omits ``price_info`` entirely to model a discovery
    entry that carries no pricing unit.
    """
    raw: dict[str, Any] = {
        "url": "http://runner",
        "app": "storyboard/fal-flux-schnell",
        "runner_id": "r1",
        "mode": "single-shot",
    }
    if unit is not None:
        raw["price_info"] = {
            "price_per_unit": 1284088677165,
            "pixels_per_unit": 1,
            "unit": unit,
        }
    return LiveRunnerInstance(
        url="http://runner",
        app="storyboard/fal-flux-schnell",
        runner_id="r1",
        mode="single-shot",
        orchestrator_url="http://orch",
        raw=raw,
    )


def _challenge() -> _RunnerPaymentChallenge:
    return _RunnerPaymentChallenge(
        payment_params="orch-payment-params-b64",
        orchestrator_url="http://orch",
        manifest_id="manifest-1",
    )


def _signer_response() -> dict[str, Any]:
    return {"payment": "payment-b64", "segCreds": "seg-creds-b64", "state": {}}


def _capture_payment(runner: Optional[LiveRunnerInstance]) -> dict[str, Any]:
    """Run ``_get_runner_payment`` and return the payload POSTed to the signer."""

    async def go() -> dict[str, Any]:
        post_json = AsyncMock(return_value=_signer_response())
        with patch("livepeer_gateway.http.post_json", post_json):
            _, payment = await _get_runner_payment(
                _challenge(),
                signer_url="http://signer",
                signer_headers=None,
                runner=runner,
            )
        assert payment.payment == "payment-b64"
        assert payment.seg_creds == "seg-creds-b64"
        post_json.assert_awaited_once()
        # post_json(url, payload, headers=...) -> payload is positional arg 1.
        return post_json.await_args.args[1]

    return asyncio.run(go())


def test_fixed_unit_runner_sets_in_pixels_1() -> None:
    payload = _capture_payment(_runner("fixed"))
    assert payload["type"] == "lv2v"
    assert payload["inPixels"] == 1


def test_fixed_unit_is_case_insensitive() -> None:
    payload = _capture_payment(_runner("Fixed"))
    assert payload["inPixels"] == 1


def test_continuous_720p_runner_omits_in_pixels() -> None:
    payload = _capture_payment(_runner("720p"))
    assert payload["type"] == "lv2v"
    assert "inPixels" not in payload


def test_continuous_hour_runner_omits_in_pixels() -> None:
    payload = _capture_payment(_runner("hour"))
    assert "inPixels" not in payload


def test_missing_price_info_omits_in_pixels() -> None:
    payload = _capture_payment(_runner(None))
    assert "inPixels" not in payload


def test_runner_none_omits_in_pixels() -> None:
    payload = _capture_payment(None)
    assert "inPixels" not in payload


def test_runner_price_unit_helper() -> None:
    assert _runner_price_unit(_runner("fixed")) == "fixed"
    assert _runner_price_unit(_runner("  Fixed  ")) == "fixed"
    assert _runner_price_unit(_runner("720p")) == "720p"
    assert _runner_price_unit(_runner(None)) == ""
    assert _runner_price_unit(None) == ""


def test_fixed_price_in_pixels_helper() -> None:
    assert _fixed_price_in_pixels(_runner("fixed")) == 1
    assert _fixed_price_in_pixels(_runner("720p")) is None
    assert _fixed_price_in_pixels(_runner("hour")) is None
    assert _fixed_price_in_pixels(_runner(None)) is None
    assert _fixed_price_in_pixels(None) is None
