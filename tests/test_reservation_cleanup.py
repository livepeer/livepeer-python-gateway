from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livepeer_gateway import live_runner, selection
from livepeer_gateway.errors import LivepeerGatewayError, NoRunnerAvailableError


@pytest.mark.parametrize(
    "paid,body,error,expected_stops",
    [
        (True, b"{", "did not return valid JSON", ["paid-id"]),
        (True, [], "expected JSON object", ["paid-id"]),
        (True, b'{"huge":' + b"1" * 5000 + b"}", "did not return valid JSON", ["paid-id"]),
        (True, {}, "missing session_id", ["paid-id"]),
        (
            True,
            {"session_id": "wrong-id", "app_url": "https://orch.test/app", "control_url": "https://orch.test/control"},
            "does not match payment manifest",
            ["paid-id"],
        ),
        (False, {"session_id": "free-id"}, "missing app_url", ["free-id"]),
        (False, {}, "missing session_id", []),
        (False, {"session_id": "../other"}, "missing app_url", []),
        (False, {"session_id": " free-id "}, "missing app_url", []),
        (True, {"session_id": " paid-id "}, "missing app_url", ["paid-id"]),
        (False, b"{", "did not return valid JSON", []),
    ],
)
async def test_malformed_reservation_cleanup(paid, body, error, expected_stops):
    stops = []

    async def reserve(request: web.Request) -> web.Response:
        if paid and not request.headers.get("Livepeer-Payment"):
            return web.json_response(
                {"manifest_id": "paid-id", "payment_params": "test", "payment_url": f"{base}/paid-id/payment"},
                status=402,
            )
        if isinstance(body, bytes):
            return web.Response(body=body, content_type="application/json")
        return web.json_response(body)

    async def stop(request: web.Request) -> web.Response:
        stops.append(request.match_info["session_id"])
        return web.Response(status=503)

    app = web.Application()
    app.router.add_post("/apps/a/session", reserve)
    app.router.add_post("/apps/a/session/{session_id}/stop", stop)
    async with TestServer(app) as server:
        base = str(server.make_url("/apps/a/session"))
        entries = [{"runners": [{"url": base, "app": "a", "price_info": {"price": "1", "unit": "fixed"}}]}]
        with (
            mock.patch.object(selection, "discover_runners", return_value=entries),
            mock.patch.object(live_runner, "get_signer_info", return_value=SimpleNamespace(address="payer")),
            mock.patch.object(
                live_runner.LivePaymentSession, "get_payment",
                return_value=SimpleNamespace(payment="payment", seg_creds="segment"),
            ),
            pytest.raises(LivepeerGatewayError) as raised,
        ):
            await selection.reserve_session(signer_url="https://signer.test" if paid else None)

    failure = raised.value
    reason = failure.rejections[0].reason if isinstance(failure, NoRunnerAvailableError) else str(failure)
    assert error in reason
    assert stops == expected_stops


@pytest.mark.parametrize(
    "prefix,path,should_stop",
    [
        ("", "/apps/a/app/apps/b/session", False),
        ("/mount", "/apps/a/session", False),
        ("", "/apps/a/session;other", False),
        ("/mount", "/mount/apps/a/session", True),
        ("/mount;v=2", "/mount/apps/a/session", False),
        ("/mount;v=2", "/mount;v=2/apps/a/session", True),
    ],
)
async def test_cleanup_only_targets_discovered_reservation_route(prefix, path, should_stop):
    runner = SimpleNamespace(orchestrator_url="https://orch.test" + prefix)
    with mock.patch.object(live_runner, "_request_body") as request:
        await live_runner._release_reservation("https://orch.test" + path, "paid-id", runner)
    assert request.await_count == int(should_stop)
