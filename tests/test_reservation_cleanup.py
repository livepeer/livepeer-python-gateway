from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest import mock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livepeer_gateway import selection
from livepeer_gateway.errors import LivepeerGatewayError
from livepeer_gateway.live_runner import LiveRunnerCallResult


@pytest.fixture
async def reservation():
    state = SimpleNamespace(stops=0, statuses=[], started=asyncio.Event(), finish=None)

    async def stop(request: web.Request) -> web.Response:
        state.stops += 1
        state.started.set()
        if state.finish is not None:
            await state.finish.wait()
        return web.json_response({}, status=state.statuses.pop(0) if state.statuses else 200)

    app = web.Application()
    app.router.add_post("/apps/a/session/sid/stop", stop)
    async with TestServer(app) as server:
        base = str(server.make_url("/apps/a/session"))
        state.result = LiveRunnerCallResult(
            {"session_id": "sid", "app_url": base + "/sid/app", "control_url": base + "/sid"},
            runner_url=base,
            session_id="sid",
        )
        cursor = SimpleNamespace(next=mock.AsyncMock(return_value=state.result))
        with mock.patch.object(selection, "runner_selector", new=mock.AsyncMock(return_value=cursor)):
            yield state


@pytest.mark.parametrize("field,value", [
    ("app_url", None), ("app_url", 123), ("session_id", None),
    ("control_url", None), ("control_url", "ftp://invalid/session"),
])
async def test_malformed_response_releases_identifiable_reservation(reservation, field, value):
    reservation.result.data[field] = value
    with pytest.raises(LivepeerGatewayError):
        await selection.reserve_session()
    assert reservation.stops == 1


@pytest.mark.parametrize("statuses", [[503, 200], [503, 404], [410]])
async def test_validation_cleanup_retries_transient_errors(reservation, statuses):
    reservation.result.data.pop("app_url")
    reservation.statuses = list(statuses)
    with pytest.raises(LivepeerGatewayError, match="missing app_url"):
        await selection.reserve_session()
    assert reservation.stops == len(statuses)


async def test_exhausted_cleanup_preserves_validation_error_and_handle(reservation):
    reservation.result.data.pop("app_url")
    reservation.statuses = [503, 503, 503]
    with pytest.raises(LivepeerGatewayError, match="missing app_url") as raised:
        await selection.reserve_session()
    assert reservation.stops == 3
    assert raised.value.cleanup_session.session_id == "sid"
    assert any("Reservation cleanup failed" in note for note in raised.value.__notes__)


async def test_cancellation_during_validation_cleanup_waits_for_release(reservation):
    reservation.result.data.pop("app_url")
    reservation.finish = asyncio.Event()
    task = asyncio.create_task(selection.reserve_session())
    await reservation.started.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    reservation.finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert reservation.stops == 1


async def test_success_transfers_reservation_without_stopping_it(reservation):
    session = await selection.reserve_session()
    assert session.session_id == "sid" and reservation.stops == 0
    await session.aclose()
    assert reservation.stops == 1
