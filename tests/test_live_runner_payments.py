"""Unit tests for live-runner session payment lifecycle.

Covers the session-owned payment loop added on top of run_session_payments:
- run_session_payments pays immediately, then on interval, and is a no-op offchain
- session-scoped payment endpoint ({control_url}/payment) and liveness handling:
  404 stops the loop and marks the session released, 409 (fixed-price) stops it
- LiveRunnerSession.start_payments is idempotent, offchain-safe, loop-aware
- stop_payments cancels only the loop; aclose / async-context-manager also stop
  the session (skipping the stop call when the orchestrator already released it)
- payment challenge parsing picks up payment_interval_ms
- reserve_session auto-starts payments and derives the cadence from the challenge
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from livepeer_gateway import live_runner, selection
from livepeer_gateway.errors import LivepeerHTTPError, SkipPaymentCycle
from livepeer_gateway.live_runner import (
    LiveRunnerCallResult,
    LiveRunnerSession,
    _parse_runner_payment_challenge,
    run_session_payments,
)
from livepeer_gateway.selection import reserve_session


class _FakePaymentSession:
    """Duck-typed stand-in for LivePaymentSession (only send_payment is used)."""

    def __init__(self) -> None:
        self.calls = 0
        self.payment_urls: list[str | None] = []
        self.paid = asyncio.Event()

    async def send_payment(
        self,
        orchestrator_url: str | None = None,
        *,
        payment_url: str | None = None,
    ) -> None:
        self.calls += 1
        self.payment_urls.append(payment_url)
        self.paid.set()


def _session(payment_session=None, interval: float = 10.0, control_url: str = "") -> LiveRunnerSession:
    return LiveRunnerSession(
        session_id="sess-1",
        app_url="http://app",
        runner_url="http://runner",
        control_url=control_url,
        payment_session=payment_session,
        payment_interval=interval,
    )


def _http_error(status: int) -> LivepeerHTTPError:
    return LivepeerHTTPError(status, "http://orch/payment")


def test_run_session_payments_noop_offchain() -> None:
    async def go() -> None:
        # Returns immediately when there is no payment_session.
        await asyncio.wait_for(run_session_payments(_session(None), interval=0.01), timeout=1.0)

    asyncio.run(go())


def test_run_session_payments_pays_immediately() -> None:
    async def go() -> None:
        ps = _FakePaymentSession()
        # Long interval: only the immediate first payment should land before cancel.
        task = asyncio.create_task(run_session_payments(_session(ps), interval=10.0))
        await asyncio.wait_for(ps.paid.wait(), timeout=1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert ps.calls >= 1

    asyncio.run(go())


def test_run_session_payments_uses_session_scoped_endpoint() -> None:
    async def go() -> None:
        ps = _FakePaymentSession()
        sess = _session(ps, control_url="https://orch/apps/r1/session/sess-1")
        task = asyncio.create_task(run_session_payments(sess, interval=10.0))
        await asyncio.wait_for(ps.paid.wait(), timeout=1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert ps.payment_urls[0] == "https://orch/apps/r1/session/sess-1/payment"

    asyncio.run(go())


def test_run_session_payments_without_control_url_uses_generic_endpoint() -> None:
    async def go() -> None:
        ps = _FakePaymentSession()
        task = asyncio.create_task(run_session_payments(_session(ps), interval=10.0))
        await asyncio.wait_for(ps.paid.wait(), timeout=1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # Old orchestrators without control_url fall back to send_payment's default.
        assert ps.payment_urls[0] is None

    asyncio.run(go())


def test_run_session_payments_survives_payment_error() -> None:
    async def go() -> None:
        ps = _FakePaymentSession()
        original = ps.send_payment
        attempts = {"n": 0}

        async def flaky(orchestrator_url: str | None = None, *, payment_url: str | None = None) -> None:
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("transient signer error")
            await original(orchestrator_url, payment_url=payment_url)

        ps.send_payment = flaky  # type: ignore[assignment]
        task = asyncio.create_task(run_session_payments(_session(ps), interval=0.01))
        await asyncio.wait_for(ps.paid.wait(), timeout=1.0)  # set on the 2nd, successful cycle
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert attempts["n"] >= 2

    asyncio.run(go())


def test_run_session_payments_treats_skip_cycle_as_paid_up() -> None:
    async def go() -> None:
        ps = _FakePaymentSession()
        original = ps.send_payment
        attempts = {"n": 0}

        async def skip_then_pay(orchestrator_url: str | None = None, *, payment_url: str | None = None) -> None:
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise SkipPaymentCycle("HTTP 482 (skip payment cycle)")  # orchestrator: balance current
            await original(orchestrator_url, payment_url=payment_url)

        ps.send_payment = skip_then_pay  # type: ignore[assignment]
        task = asyncio.create_task(run_session_payments(_session(ps), interval=0.01))
        await asyncio.wait_for(ps.paid.wait(), timeout=1.0)  # set on the 2nd cycle, after the skip
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # The skip did not kill the loop; it kept going and paid on the next cycle.
        assert attempts["n"] >= 2

    asyncio.run(go())


def test_run_session_payments_stops_and_marks_released_on_404() -> None:
    async def go() -> None:
        ps = _FakePaymentSession()

        async def gone(orchestrator_url: str | None = None, *, payment_url: str | None = None) -> None:
            raise _http_error(404)

        ps.send_payment = gone  # type: ignore[assignment]
        sess = _session(ps, control_url="https://orch/apps/r1/session/sess-1")
        # The loop must return on its own (no cancel) once the orchestrator 404s.
        await asyncio.wait_for(run_session_payments(sess, interval=0.01), timeout=1.0)
        assert sess.released
        await asyncio.wait_for(sess.wait_released(), timeout=1.0)

    asyncio.run(go())


def test_run_session_payments_stops_on_409_fixed_price() -> None:
    async def go() -> None:
        ps = _FakePaymentSession()

        async def fixed(orchestrator_url: str | None = None, *, payment_url: str | None = None) -> None:
            raise _http_error(409)

        ps.send_payment = fixed  # type: ignore[assignment]
        sess = _session(ps, control_url="https://orch/apps/r1/session/sess-1")
        await asyncio.wait_for(run_session_payments(sess, interval=0.01), timeout=1.0)
        # Fixed-price sessions simply need no more payments; they are not released.
        assert not sess.released

    asyncio.run(go())


def test_run_session_payments_retries_other_http_errors() -> None:
    async def go() -> None:
        ps = _FakePaymentSession()
        original = ps.send_payment
        attempts = {"n": 0}

        async def flaky(orchestrator_url: str | None = None, *, payment_url: str | None = None) -> None:
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise _http_error(500)
            await original(orchestrator_url, payment_url=payment_url)

        ps.send_payment = flaky  # type: ignore[assignment]
        task = asyncio.create_task(run_session_payments(_session(ps), interval=0.01))
        await asyncio.wait_for(ps.paid.wait(), timeout=1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert attempts["n"] >= 2

    asyncio.run(go())


def test_start_payments_noop_offchain() -> None:
    async def go() -> None:
        sess = _session(None)
        assert sess.start_payments() is None
        assert sess._payment_task is None

    asyncio.run(go())


def test_start_payments_without_running_loop_returns_none() -> None:
    # No running loop: logs a warning and skips rather than raising.
    sess = _session(_FakePaymentSession())
    assert sess.start_payments() is None
    assert sess._payment_task is None


def test_start_payments_is_idempotent() -> None:
    async def go() -> None:
        sess = _session(_FakePaymentSession())
        t1 = sess.start_payments()
        t2 = sess.start_payments()
        assert t1 is not None
        assert t1 is t2
        t1.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t1

    asyncio.run(go())


def test_stop_payments_cancels_only_loop() -> None:
    async def go() -> None:
        sess = _session(_FakePaymentSession())
        task = sess.start_payments()
        assert task is not None
        with patch.object(live_runner, "stop_runner_session", new=AsyncMock()) as stop:
            await sess.stop_payments()
            stop.assert_not_awaited()
        assert task.done()
        assert sess._payment_task is None
        # A stopped loop can be restarted (drain / hand-off flows).
        restarted = sess.start_payments()
        assert restarted is not None and restarted is not task
        restarted.cancel()
        with pytest.raises(asyncio.CancelledError):
            await restarted

    asyncio.run(go())


def test_aclose_cancels_loop_and_stops_session() -> None:
    async def go() -> None:
        sess = _session(_FakePaymentSession())
        sess.start_payments()
        task = sess._payment_task
        assert task is not None
        with patch.object(live_runner, "stop_runner_session", new=AsyncMock()) as stop:
            await sess.aclose()
            stop.assert_awaited_once()
        assert task.done()

    asyncio.run(go())


def test_aclose_skips_stop_when_released() -> None:
    async def go() -> None:
        sess = _session(_FakePaymentSession())
        sess._mark_released()
        with patch.object(live_runner, "stop_runner_session", new=AsyncMock()) as stop:
            await sess.aclose()
            stop.assert_not_awaited()

    asyncio.run(go())


def test_async_context_manager_starts_and_stops() -> None:
    async def go() -> None:
        ps = _FakePaymentSession()
        sess = _session(ps)
        with patch.object(live_runner, "stop_runner_session", new=AsyncMock()) as stop:
            async with sess as entered:
                assert entered is sess
                await asyncio.wait_for(ps.paid.wait(), timeout=1.0)
                assert sess._payment_task is not None
            stop.assert_awaited_once()
        assert sess._payment_task is None

    asyncio.run(go())


def _challenge_error(payload: dict) -> LivepeerHTTPError:
    return LivepeerHTTPError(402, "http://runner/session", body=json.dumps(payload))


def test_parse_challenge_reads_payment_interval_ms() -> None:
    challenge = _parse_runner_payment_challenge(
        _challenge_error(
            {
                "payment_params": "params",
                "orchestrator": "https://orch",
                "manifest_id": "sess-1",
                "payment_interval_ms": 5000,
            }
        )
    )
    assert challenge.payment_interval_s == 5.0


def test_parse_challenge_without_payment_interval_ms() -> None:
    challenge = _parse_runner_payment_challenge(
        _challenge_error(
            {
                "payment_params": "params",
                "orchestrator": "https://orch",
                "manifest_id": "sess-1",
            }
        )
    )
    assert challenge.payment_interval_s is None


class _FakeCursor:
    def __init__(self, result: LiveRunnerCallResult) -> None:
        self._result = result

    async def next(self) -> LiveRunnerCallResult:
        return self._result


def _reserve_result(ps: _FakePaymentSession | None, server_payment_interval: float | None = None) -> LiveRunnerCallResult:
    return LiveRunnerCallResult(
        {
            "session_id": "sess-1",
            "app_url": "https://orch/apps/r1/session/sess-1/app",
            "control_url": "https://orch/apps/r1/session/sess-1",
        },
        runner_url="https://orch/apps/r1/session",
        payment_session=ps,  # type: ignore[arg-type]
        server_payment_interval=server_payment_interval,
    )


def test_reserve_session_auto_starts_payments_and_parses_control_url() -> None:
    async def go() -> None:
        ps = _FakePaymentSession()
        cursor = _FakeCursor(_reserve_result(ps))
        with patch.object(selection, "runner_selector", new=AsyncMock(return_value=cursor)):
            sess = await reserve_session(signer_url="https://signer")
        assert sess.control_url == "https://orch/apps/r1/session/sess-1"
        assert sess._payment_task is not None
        await asyncio.wait_for(ps.paid.wait(), timeout=1.0)
        assert ps.payment_urls[0] == "https://orch/apps/r1/session/sess-1/payment"
        await sess.stop_payments()

    asyncio.run(go())


def test_reserve_session_auto_pay_false_does_not_start() -> None:
    async def go() -> None:
        cursor = _FakeCursor(_reserve_result(_FakePaymentSession()))
        with patch.object(selection, "runner_selector", new=AsyncMock(return_value=cursor)):
            sess = await reserve_session(signer_url="https://signer", auto_pay=False)
        assert sess._payment_task is None

    asyncio.run(go())


def test_reserve_session_derives_interval_from_challenge() -> None:
    async def go() -> None:
        cursor = _FakeCursor(_reserve_result(_FakePaymentSession(), server_payment_interval=5.0))
        with patch.object(selection, "runner_selector", new=AsyncMock(return_value=cursor)):
            sess = await reserve_session(signer_url="https://signer", auto_pay=False)
        # 60% of the orchestrator's 5s debit tick.
        assert sess.payment_interval == pytest.approx(3.0)

    asyncio.run(go())


def test_reserve_session_explicit_interval_wins() -> None:
    async def go() -> None:
        cursor = _FakeCursor(_reserve_result(_FakePaymentSession(), server_payment_interval=5.0))
        with patch.object(selection, "runner_selector", new=AsyncMock(return_value=cursor)):
            sess = await reserve_session(
                signer_url="https://signer",
                payment_interval=1.5,
                auto_pay=False,
            )
        assert sess.payment_interval == 1.5

    asyncio.run(go())
