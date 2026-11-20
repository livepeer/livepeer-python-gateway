from __future__ import annotations

from types import SimpleNamespace
from unittest import mock
from urllib.parse import urljoin

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
        ("onchain-id", {}, None, ["onchain-id"]),
        ("onchain-id", {**_SESSION_URLS, "session_id": "onchain-id", "control_url": "https://orch.test:bad/s1"}, None, ["onchain-id"]),
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

    async def reserve_next(request: web.Request) -> web.Response:
        assert stops == expected_stops
        return web.json_response({**_SESSION_URLS, "session_id": "next-id"})

    app = web.Application()
    app.router.add_post("/apps/a/session", reserve)
    app.router.add_post("/apps/a/session/{session_id}/stop", stop)
    app.router.add_post("/apps/next/session", reserve_next)
    async with TestServer(app) as server:
        base = str(server.make_url("/apps/a/session"))
        entries = [{"runners": [{"url": base, "app": "a", "price_info": {"price": "1", "unit": "fixed"}}]}]
        if error is None:
            entries[0]["runners"].append({"url": str(server.make_url("/apps/next/session")), "app": "a"})
        signer_url = "https://signer.test" if manifest_id else None
        with (
            mock.patch.object(selection, "discover_runners", return_value=entries),
            mock.patch.object(live_runner, "get_signer_info", return_value=SimpleNamespace(address="payer")),
            mock.patch.object(
                live_runner.LivePaymentSession, "get_payment",
                return_value=SimpleNamespace(payment="payment", seg_creds="segment"),
            ),
        ):
            if error is None:
                session = await selection.reserve_session(signer_url=signer_url)
                assert session.session_id == "next-id"
            else:
                with pytest.raises(NoRunnerAvailableError) as raised:
                    await selection.reserve_session(signer_url=signer_url)
                assert len(raised.value.rejections) == 1
                assert error in raised.value.rejections[0].reason
    assert stops == expected_stops


@pytest.mark.parametrize(
    "prefix,path,is_session",
    [
        ("", "/apps/a/app/apps/b/session", False),
        ("/mount", "/apps/a/session", False),
        ("", "/apps/a/session;other", False),
        ("", "https://proxy.test/apps/a/session", False),
        ("", "https://ORCH.test:443/apps/a/session", True),
        ("/mount", "/mount/apps/a/session", True),
        ("/mount;v=2", "/mount/apps/a/session", False),
        ("/mount;v=2", "/mount;v=2/apps/a/session", True),
    ],
)
def test_session_endpoint_matches_the_discovery_mount(prefix, path, is_session):
    runner = SimpleNamespace(mode="persistent", orchestrator_url="https://orch.test" + prefix)
    assert live_runner._is_session_endpoint(urljoin("https://orch.test", path), runner) is is_session


def test_session_endpoint_requires_http_origin():
    for url in ("/apps/a/session", "ftp://orch.test/apps/a/session", "https://orch.test:0/apps/a/session"):
        assert not live_runner._is_session_endpoint(url, None)


@pytest.mark.parametrize(
    "body,error",
    [
        (b"{}", "missing session_id"),
        (b'{"session_id":"s1","app_url":"https://orch.test/app","control_url":"https://orch.test:bad/s1"}', "Invalid orchestrator URL"),
    ],
)
async def test_call_runner_validates_session_with_implicit_post(body, error):
    runner = live_runner.LiveRunnerInstance(
        url="https://orch.test/apps/proxy/session",
        app="proxy",
        runner_id="proxy",
        mode="single-shot",
        orchestrator_url="https://orch.test",
        raw={},
    )
    with mock.patch.object(live_runner, "_request_body", return_value=(body, "application/json")) as request:
        with pytest.raises(LivepeerGatewayError, match=error):
            await live_runner.call_runner("https://orch.test/apps/a/session", runner=runner, method="", payload={})

    assert request.call_args.kwargs["method"] == "POST"
