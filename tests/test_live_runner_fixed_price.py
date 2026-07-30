"""Unit tests for fixed-price live-runner payment signalling.

A fixed-price single-shot live-runner generation advertises
``price_info.unit == "fixed"`` and the v0.9.0 orchestrator debits exactly one
unit per generation (``PixelsPerUnit == 1``). The gateway must therefore bill it
under the ``fixed`` job type, for which the signer sets ``billableUnits = 1``.

Billing it as ``lv2v`` instead is what broke: on that path the signer DISCARDS
``req.InPixels`` and substitutes its continuous 720p30-over-60s estimate
(1280*720*30*60 = 1,658,880,000 units), inflating the fee ~1.66e9x so
``numTickets`` blows past the orchestrator's max-100 guard (observed e2e:
HTTP 400 "numTickets 2721947758 exceeds maximum of 100").

Using ``fixed`` works against stock go-livepeer v0.9.0 — it needs no signer-side
change, because ``fixed`` already means "bill exactly one unit". (An earlier
attempt kept ``lv2v`` and sent ``inPixels:1``, which required go-livepeer PR
#4006 to teach the signer to honour ``InPixels`` on the lv2v path; that pairing
is no longer needed for this fix. ``inPixels:1`` is retained as harmless
belt-and-braces.)

Continuous runners (``720p``/``hour``) must keep ``lv2v`` and must NOT set
``inPixels``, so streaming payloads stay byte-identical.
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
    _payment_job_type,
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


def test_fixed_unit_runner_bills_as_fixed() -> None:
    # The job type is what makes the fee right: under `fixed` the signer sets
    # billableUnits = 1. Under `lv2v` it DISCARDS inPixels and substitutes its
    # continuous 720p30-over-60s estimate (1,658,880,000 units), inflating the fee
    # ~1.66e9x so numTickets overflows the orchestrator's 100-ticket guard.
    payload = _capture_payment(_runner("fixed"))
    assert payload["type"] == "fixed"
    assert payload["inPixels"] == 1


def test_fixed_unit_is_case_insensitive() -> None:
    payload = _capture_payment(_runner("Fixed"))
    assert payload["type"] == "fixed"
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


def test_payment_job_type_helper() -> None:
    # Only a fixed-price runner switches job type; everything else keeps lv2v so
    # continuous/streaming billing is untouched.
    assert _payment_job_type(_runner("fixed")) == "fixed"
    assert _payment_job_type(_runner("  Fixed  ")) == "fixed"
    assert _payment_job_type(_runner("720p")) == "lv2v"
    assert _payment_job_type(_runner("hour")) == "lv2v"
    assert _payment_job_type(_runner(None)) == "lv2v"
    assert _payment_job_type(None) == "lv2v"


def test_continuous_runners_still_bill_as_lv2v() -> None:
    # Regression guard for the streaming path: a non-fixed runner must keep the
    # lv2v job type (and its signer-side pixel estimate) exactly as before.
    for unit in ("720p", "hour", None):
        assert _capture_payment(_runner(unit))["type"] == "lv2v"
    assert _capture_payment(None)["type"] == "lv2v"
