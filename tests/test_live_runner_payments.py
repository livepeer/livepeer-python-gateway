from __future__ import annotations

import asyncio
from unittest import mock

import pytest

from livepeer_gateway import live_runner, remote_signer, selection
from livepeer_gateway.errors import (
    LivepeerGatewayError,
    LivepeerHTTPError,
    SkipPaymentCycle,
)
from livepeer_gateway.live_runner import LiveRunnerCallResult, LiveRunnerSession
from livepeer_gateway.remote_signer import LivePaymentChallenge, LivePaymentSession


_CONTROL_URL = "https://orch.example.com/apps/runner-1/session/session-1"
_PAYMENT_URL = f"{_CONTROL_URL}/payment"


def _http_error(status: int) -> LivepeerHTTPError:
    return LivepeerHTTPError(status, _PAYMENT_URL)


def _live_payment_session() -> LivePaymentSession:
    return LivePaymentSession(
        "https://signer.example.com",
        type="live",
        challenge=LivePaymentChallenge(
            payment_params="opaque",
            manifest_id="session-1",
            payment_url=_PAYMENT_URL,
        ),
    )


class _FundingSession:
    def __init__(self, *, released: bool = False) -> None:
        self.released = released
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def run_payments(self) -> bool:
        self.started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return self.released


def _session(*, control_url: str = _CONTROL_URL) -> LiveRunnerSession:
    return LiveRunnerSession(
        session_id="session-1",
        app_url=f"{_CONTROL_URL}/app",
        runner_url="https://orch.example.com/apps/runner-1/session",
        control_url=control_url,
    )


class TestPaymentLoop:
    @pytest.mark.parametrize(
        "status, released", [(403, False), (404, True), (409, False)]
    )
    async def test_terminal_status_stops_loop(
        self, status: int, released: bool
    ) -> None:
        payment_session = _live_payment_session()
        with (
            mock.patch.object(
                payment_session,
                "send_payment",
                new=mock.AsyncMock(side_effect=_http_error(status)),
            ) as send_payment,
            mock.patch.object(remote_signer, "PAYMENT_INTERVAL_S", 0),
        ):
            result = await asyncio.wait_for(
                payment_session.run_payments(),
                timeout=1.0,
            )

        assert result is released
        send_payment.assert_awaited_once_with()

    @pytest.mark.parametrize(
        "first_error",
        [RuntimeError("network"), _http_error(408), SkipPaymentCycle("paid up")],
    )
    async def test_retryable_error_reaches_next_cycle(
        self, first_error: Exception
    ) -> None:
        payment_session = _live_payment_session()
        send_payment = mock.AsyncMock(side_effect=[first_error, _http_error(404)])
        with (
            mock.patch.object(payment_session, "send_payment", new=send_payment),
            mock.patch.object(remote_signer, "PAYMENT_INTERVAL_S", 0),
        ):
            released = await asyncio.wait_for(
                payment_session.run_payments(),
                timeout=1.0,
            )

        assert released
        assert send_payment.await_count == 2


class TestSessionPaymentLifecycle:
    @pytest.mark.parametrize("control_url", ["", "ftp://orch/session/session-1"])
    def test_session_rejects_missing_or_invalid_control_url(
        self, control_url: str
    ) -> None:
        with pytest.raises(LivepeerGatewayError):
            _session(control_url=control_url)

    async def test_start_payments_starts_challenge_owned_session(self) -> None:
        payment_session = _FundingSession()
        session = _session()

        session._start_payments(payment_session)  # type: ignore[arg-type]
        await asyncio.wait_for(payment_session.started.wait(), timeout=1.0)

        await session.stop_payments()

    async def test_start_payments_is_idempotent(self) -> None:
        payment_session = _FundingSession()
        session = _session()

        session._start_payments(payment_session)  # type: ignore[arg-type]
        task = session._payment_task
        session._start_payments(payment_session)  # type: ignore[arg-type]

        assert session._payment_task is task
        await session.stop_payments()

    async def test_stop_payments_cancels_and_clears_task(self) -> None:
        payment_session = _FundingSession()
        session = _session()
        session._start_payments(payment_session)  # type: ignore[arg-type]
        await asyncio.wait_for(payment_session.started.wait(), timeout=1.0)

        await session.stop_payments()

        assert session._payment_task is None
        assert payment_session.cancelled.is_set()

    async def test_aclose_delegates_to_stop_runner_session(self) -> None:
        session = _session()

        stop = mock.AsyncMock()
        with mock.patch.object(live_runner, "stop_runner_session", stop):
            await session.aclose()

        stop.assert_awaited_once_with(session)

    async def test_async_context_manager_closes_session(self) -> None:
        session = _session()
        stop = mock.AsyncMock()

        with mock.patch.object(live_runner, "stop_runner_session", stop):
            async with session as entered:
                assert entered is session

        stop.assert_awaited_once_with(session)


class TestStopRunnerSession:
    async def test_stops_payment_task_and_uses_control_url(self) -> None:
        payment_session = _FundingSession()
        session = _session()
        session._start_payments(payment_session)  # type: ignore[arg-type]
        await asyncio.wait_for(payment_session.started.wait(), timeout=1.0)
        post_empty = mock.AsyncMock()

        with mock.patch.object(live_runner, "_post_empty", post_empty):
            await live_runner.stop_runner_session(session, timeout=12.0)

        assert payment_session.cancelled.is_set()
        assert session._payment_task is None
        assert session.released
        post_empty.assert_awaited_once_with(
            f"{_CONTROL_URL}/stop",
            headers={},
            timeout=12.0,
        )

    async def test_remote_stop_failure_still_stops_payment_task(self) -> None:
        payment_session = _FundingSession()
        session = _session()
        session._start_payments(payment_session)  # type: ignore[arg-type]
        await asyncio.wait_for(payment_session.started.wait(), timeout=1.0)

        with (
            mock.patch.object(
                live_runner,
                "_post_empty",
                new=mock.AsyncMock(side_effect=LivepeerGatewayError("stop failed")),
            ),
            pytest.raises(LivepeerGatewayError, match="stop failed"),
        ):
            await live_runner.stop_runner_session(session)

        assert payment_session.cancelled.is_set()
        assert session._payment_task is None
        assert not session.released

    async def test_already_released_session_only_stops_local_funding(self) -> None:
        payment_session = _FundingSession()
        session = _session()
        session._start_payments(payment_session)  # type: ignore[arg-type]
        await asyncio.wait_for(payment_session.started.wait(), timeout=1.0)
        session.released = True
        post_empty = mock.AsyncMock()

        with mock.patch.object(live_runner, "_post_empty", post_empty):
            await live_runner.stop_runner_session(session)

        assert payment_session.cancelled.is_set()
        post_empty.assert_not_awaited()


class _Cursor:
    def __init__(self, *results: LiveRunnerCallResult) -> None:
        self.results = list(results)
        self.rejections = []

    async def next(self) -> LiveRunnerCallResult:
        if self.results:
            return self.results.pop(0)
        raise AssertionError("unexpected extra runner selection")


def _reservation(
    name: str,
    *,
    control_url: str | None,
    payment_session: object | None = None,
) -> LiveRunnerCallResult:
    data = {
        "session_id": f"session-{name}",
        "app_url": f"https://orch.example.com/session-{name}/app",
    }
    if control_url is not None:
        data["control_url"] = control_url
    return LiveRunnerCallResult(
        data,
        runner_url=f"https://orch.example.com/runner-{name}/session",
        payment_session=payment_session,  # type: ignore[arg-type]
    )


class TestReservationSelection:
    async def test_paid_reservation_starts_scoped_funding(self) -> None:
        payment_session = _FundingSession()
        cursor = _Cursor(
            _reservation(
                "1",
                control_url=_CONTROL_URL,
                payment_session=payment_session,
            )
        )

        with mock.patch.object(
            selection, "runner_selector", new=mock.AsyncMock(return_value=cursor)
        ):
            session = await selection.reserve_session()

        await asyncio.wait_for(payment_session.started.wait(), timeout=1.0)
        await session.stop_payments()

    @pytest.mark.parametrize(
        "bad_control_url",
        [None, "ftp://orch.example.com/session/bad"],
    )
    async def test_invalid_control_url_fails_immediately(
        self, bad_control_url: str | None
    ) -> None:
        cursor = _Cursor(
            _reservation("bad", control_url=bad_control_url),
            _reservation("good", control_url=_CONTROL_URL),
        )

        with mock.patch.object(
            selection,
            "runner_selector",
            new=mock.AsyncMock(return_value=cursor),
        ):
            with pytest.raises(LivepeerGatewayError):
                await selection.reserve_session()

        assert len(cursor.results) == 1

    async def test_missing_control_url_is_contract_error(self) -> None:
        cursor = _Cursor(_reservation("bad", control_url=None))
        with mock.patch.object(
            selection,
            "runner_selector",
            new=mock.AsyncMock(return_value=cursor),
        ):
            with pytest.raises(LivepeerGatewayError, match="missing control_url"):
                await selection.reserve_session()
