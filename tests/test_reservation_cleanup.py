from __future__ import annotations

import asyncio
from dataclasses import replace
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
        state.query = dict(request.query)
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
        cursor = SimpleNamespace(next=mock.AsyncMock(side_effect=lambda: state.result))
        with mock.patch.object(selection, "runner_selector", new=mock.AsyncMock(return_value=cursor)):
            yield state


@pytest.mark.parametrize("field,value", [
    ("app_url", None), ("app_url", 123), ("session_id", None),
    ("control_url", None), ("control_url", "ftp://invalid/session"),
    ("control_url", "http://["),
])
async def test_malformed_response_releases_identifiable_reservation(reservation, field, value):
    reservation.result.data[field] = value
    with pytest.raises(LivepeerGatewayError):
        await selection.reserve_session()
    assert reservation.stops == 1


@pytest.mark.parametrize("paid", [True, False])
async def test_cleanup_recovers_canonical_reservation_identity(reservation, paid):
    if paid:
        reservation.result.data.update(session_id="wrong", control_url=None)
    else:
        reservation.result.data.pop("session_id")
        reservation.result = replace(reservation.result, session_id="")
    with pytest.raises(LivepeerGatewayError, match="session_id"):
        await selection.reserve_session()
    assert reservation.stops == 1


async def test_cleanup_uses_canonical_route_when_control_url_targets_app(reservation):
    reservation.result.data["control_url"] = reservation.result.data.pop("app_url")
    with pytest.raises(LivepeerGatewayError, match="missing app_url"):
        await selection.reserve_session()
    assert reservation.stops == 1


async def test_cleanup_preserves_verified_control_url_query(reservation):
    reservation.result.data.pop("app_url")
    reservation.result.data["control_url"] += "?lease=opaque"
    with pytest.raises(LivepeerGatewayError, match="missing app_url"):
        await selection.reserve_session()
    assert reservation.query == {"lease": "opaque"}


def test_cleanup_recovers_session_across_equivalent_origins():
    result = LiveRunnerCallResult({}, runner_url="https://ORCH.test:443/apps/a/session")
    control_url = "https://orch.test/apps/a/session/sid?lease=opaque"
    session = selection._invalid_reservation_handle(result, "", control_url)
    assert session is not None
    assert session.session_id == "sid" and session.control_url == control_url


def test_aliased_reservation_cleanup_rejects_a_different_session_id():
    result = LiveRunnerCallResult({}, runner_url="https://orch.test/session", session_id="paid-id")
    control_url = "https://orch.test/apps/a/session/paid-id"
    session = selection._invalid_reservation_handle(result, result.session_id, control_url)
    assert session is not None and session.control_url == control_url
    assert selection._invalid_reservation_handle(result, result.session_id, control_url.replace("paid-id", "body-id")) is None


@pytest.mark.parametrize("reported_app", [None, "wrong", "a"])
def test_metered_alias_recovers_control_route_from_payment_challenge(reported_app):
    control_url = "https://orch.test/apps/a/session/paid-id"
    payment = SimpleNamespace(
        manifest_id="paid-id", _challenge=SimpleNamespace(payment_url=control_url + "/payment?lease=opaque"),
    )
    result = LiveRunnerCallResult(
        {"session_id": "paid-id"}, runner_url="https://orch.test/alias", session_id="paid-id",
        payment_session=payment,
    )
    reported_url = f"https://orch.test/apps/{reported_app}/session/paid-id?lease=returned" if reported_app else ""
    session = selection._invalid_reservation_handle(result, "paid-id", reported_url)
    expected = reported_url if reported_app == "a" else control_url + "?lease=opaque"
    assert session is not None and session.control_url == expected


def test_offchain_alias_recovers_control_route_only_with_matching_app_url():
    control_url = "https://orch.test/apps/a/session/sid"
    result = LiveRunnerCallResult({}, runner_url="https://orch.test/alias")
    session = selection._invalid_reservation_handle(result, "", control_url, control_url + "/app")
    assert session is not None and session.session_id == "sid"
    assert selection._invalid_reservation_handle(result, "", control_url, control_url + "-other/app") is None


async def test_conflicting_offchain_ids_without_corroboration_do_not_stop_either(reservation):
    data = {**reservation.result.data, "session_id": "body-id", "app_url": None}
    reservation.result = replace(reservation.result, data=data, session_id="body-id")
    with pytest.raises(LivepeerGatewayError, match="missing app_url") as raised:
        await selection.reserve_session()
    assert reservation.stops == 0
    assert any("cleanup unresolved" in note for note in raised.value.__notes__)


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


@pytest.mark.parametrize("status", [200, 401])
async def test_cancellation_during_validation_cleanup_waits_for_release(reservation, status):
    reservation.result.data.pop("app_url")
    reservation.finish = asyncio.Event()
    reservation.statuses = [status]
    task = asyncio.create_task(selection.reserve_session())
    await reservation.started.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    reservation.finish.set()
    with pytest.raises(asyncio.CancelledError) as raised:
        await task
    assert reservation.stops == 1
    if status == 401:
        assert raised.value.cleanup_session.session_id == "sid"


async def test_success_transfers_reservation_without_stopping_it(reservation):
    session = await selection.reserve_session()
    assert session.session_id == "sid" and reservation.stops == 0
    await session.aclose()
    assert reservation.stops == 1
