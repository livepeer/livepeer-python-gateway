from __future__ import annotations

import ssl
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web

from livepeer_gateway.errors import LivepeerGatewayError
from livepeer_gateway.live_runner import call_runner
from livepeer_gateway.trickle_publisher import TricklePublisher
from livepeer_gateway.trickle_subscriber import TrickleSubscriber


_FIXTURES = Path(__file__).with_name("fixtures")


@asynccontextmanager
async def _untrusted_tls_server() -> AsyncIterator[str]:
    runner = web.AppRunner(web.Application())
    await runner.setup()
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(
        _FIXTURES / "untrusted-localhost-cert.pem",
        _FIXTURES / "untrusted-localhost-key.pem",
    )
    site = web.TCPSite(runner, "127.0.0.1", 0, ssl_context=ssl_context)
    await site.start()
    server = site._server
    assert server is not None
    port = server.sockets[0].getsockname()[1]
    try:
        yield f"https://127.0.0.1:{port}"
    finally:
        await runner.cleanup()


@pytest.mark.parametrize("stream", [False, True])
async def test_call_runner_rejects_untrusted_certificate(stream: bool) -> None:
    async with _untrusted_tls_server() as base_url:
        with pytest.raises(LivepeerGatewayError) as raised:
            result = await call_runner(f"{base_url}/runner", stream=stream)
            if stream:
                await result.aclose()

    assert isinstance(raised.value.__cause__, aiohttp.ClientConnectorCertificateError)


async def test_trickle_publisher_rejects_untrusted_certificate() -> None:
    async with _untrusted_tls_server() as base_url:
        async with TricklePublisher(
            f"{base_url}/trickle", "application/octet-stream"
        ) as publisher:
            with pytest.raises(aiohttp.ClientConnectorCertificateError):
                await publisher.create()


async def test_trickle_subscriber_rejects_untrusted_certificate(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async with _untrusted_tls_server() as base_url:
        async with TrickleSubscriber(
            f"{base_url}/trickle", start_seq=0, max_retries=1
        ) as subscriber:
            assert await subscriber.next() is None

    assert any(
        record.exc_info
        and isinstance(record.exc_info[1], aiohttp.ClientConnectorCertificateError)
        for record in caplog.records
    )
