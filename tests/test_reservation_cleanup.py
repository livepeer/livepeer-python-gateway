from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from livepeer_gateway import live_runner, selection
from livepeer_gateway.errors import LivepeerGatewayError, NoRunnerAvailableError

_SESSION_URLS = {"app_url": "https://orch.test/app", "control_url": "https://orch.test/control"}
_ID_MISMATCH = "does not match payment manifest"


@pytest.mark.parametrize(
    "manifest_id,body,error,expected_stops",
    [
        ("onchain-id", b"{", "did not return valid JSON", ["onchain-id"]),
        ("onchain-id", [], "expected JSON object", ["onchain-id"]),
        ("onchain-id", b'{"huge":' + b"1" * 5000 + b"}", "did not return valid JSON", ["onchain-id"]),
        ("onchain-id", {}, "missing session_id", ["onchain-id"]),
        ("onchain-id", {**_SESSION_URLS, "session_id": "wrong-id"}, _ID_MISMATCH, ["onchain-id"]),
        ("onchain-id", {**_SESSION_URLS, "session_id": " onchain-id "}, _ID_MISMATCH, ["onchain-id"]),
        (" onchain-id ", {**_SESSION_URLS, "session_id": " onchain-id "}, _ID_MISMATCH, []),
        ("", {"session_id": "offchain-id"}, "missing app_url", []),
        ("", b"{", "did not return valid JSON", []),
    ],
)
async def test_malformed_reservation_cleanup(manifest_id, body, error, expected_stops):
    stops = []

    async def reserve(request: web.Request) -> web.Response:
        if manifest_id and not request.headers.get("Livepeer-Payment"):
            return web.json_response(
                {"manifest_id": manifest_id, "payment_params": "test", "payment_url": f"{base}/onchain-id/payment"},
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
            await selection.reserve_session(signer_url="https://signer.test" if manifest_id else None)

    failure = raised.value
    if isinstance(body, (bytes, list)):
        assert isinstance(failure, NoRunnerAvailableError)
        reason = failure.rejections[0].reason
    else:
        assert type(failure) is LivepeerGatewayError
        reason = str(failure)
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
        await live_runner._release_reservation("https://orch.test" + path, "onchain-id", runner)
    assert request.await_count == int(should_stop)
