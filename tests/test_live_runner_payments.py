"""Unit tests for live-runner session payment lifecycle.

Covers the session-owned payment loop added on top of run_session_payments:
- run_session_payments pays immediately, then on interval, and is a no-op offchain
- LiveRunnerSession.start_payments is idempotent, offchain-safe, loop-aware
- aclose / async-context-manager cancel the loop and stop the session
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from livepeer_gateway import live_runner
from livepeer_gateway.errors import SkipPaymentCycle
from livepeer_gateway.live_runner import LiveRunnerSession, run_session_payments


class _FakePaymentSession:
    """Duck-typed stand-in for LivePaymentSession (only send_payment is used)."""

    def __init__(self) -> None:
        self.calls = 0
        self.paid = asyncio.Event()

    async def send_payment(self, orchestrator_url: str | None = None) -> None:
        self.calls += 1
        self.paid.set()


def _session(payment_session=None, interval: float = 10.0) -> LiveRunnerSession:
    return LiveRunnerSession(
        session_id="sess-1",
        app_url="http://app",
        runner_url="http://runner",
        payment_session=payment_session,
        payment_interval=interval,
    )


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


def test_run_session_payments_survives_payment_error() -> None:
    async def go() -> None:
        ps = _FakePaymentSession()
        original = ps.send_payment
        attempts = {"n": 0}

        async def flaky(orchestrator_url: str | None = None) -> None:
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RuntimeError("transient signer error")
            await original(orchestrator_url)

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

        async def skip_then_pay(orchestrator_url: str | None = None) -> None:
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise SkipPaymentCycle("HTTP 482 (skip payment cycle)")  # orchestrator: balance current
            await original(orchestrator_url)

        ps.send_payment = skip_then_pay  # type: ignore[assignment]
        task = asyncio.create_task(run_session_payments(_session(ps), interval=0.01))
        await asyncio.wait_for(ps.paid.wait(), timeout=1.0)  # set on the 2nd cycle, after the skip
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # The skip did not kill the loop; it kept going and paid on the next cycle.
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
        assert sess._payment_task.done()

    asyncio.run(go())
