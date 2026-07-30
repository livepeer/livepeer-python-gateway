"""Tests for non-JSON (raw byte) responses in call_runner.

Single-document JSON responses (``application/json`` or an RFC 6839 ``+json``
suffix) keep today's behavior: parsed into ``result.data``, strict about being an
object. Anything else — binary, or a multi-document format like ndjson — returns
the body unparsed in ``result.content`` with ``result.content_type`` set.
"""

from __future__ import annotations

import asyncio

import pytest
from aiohttp import web

from livepeer_gateway.errors import LivepeerGatewayError, LivepeerHTTPError
from livepeer_gateway.live_runner import call_runner

FAKE_JPEG = b"\xff\xd8\xff\xe0" + b"jpeg-bytes" * 100


def _run(app: web.Application, scenario):
    """Serve `app` on an ephemeral port and run `scenario(base_url)`."""

    async def main():
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            return await scenario(f"http://127.0.0.1:{port}")
        finally:
            await runner.cleanup()

    return asyncio.run(main())


def test_json_response_unchanged():
    async def handler(request):
        return web.json_response({"message": "hello", "session_id": " s1 "})

    app = web.Application()
    app.router.add_post("/call", handler)

    async def scenario(base):
        return await call_runner(f"{base}/call", payload={"x": 1})

    result = _run(app, scenario)
    assert result.data == {"message": "hello", "session_id": " s1 "}
    assert result.session_id == "s1"
    assert result.content is None
    assert result.content_type == "application/json"


def test_binary_response_returns_raw():
    async def handler(request):
        return web.Response(body=FAKE_JPEG, content_type="image/jpeg")

    app = web.Application()
    app.router.add_post("/img", handler)

    async def scenario(base):
        return await call_runner(f"{base}/img", payload={"prompt": "x"})

    result = _run(app, scenario)
    assert result.content == FAKE_JPEG
    assert result.content_type == "image/jpeg"
    assert result.data == {}


def test_json_suffix_content_type_is_parsed():
    """RFC 6839 ``+json`` types are single JSON documents, so they parse."""

    async def handler(request):
        return web.Response(
            text='{"message": "hello"}', content_type="application/vnd.acme.v1+json"
        )

    app = web.Application()
    app.router.add_post("/vnd", handler)

    async def scenario(base):
        return await call_runner(f"{base}/vnd", payload={})

    result = _run(app, scenario)
    assert result.data == {"message": "hello"}
    assert result.content is None
    assert result.content_type == "application/vnd.acme.v1+json"


def test_ndjson_returns_raw():
    """Multi-document formats json.loads can't parse come back as bytes."""

    body = b'{"token": "Hello"}\n{"token": " world"}\n'

    async def handler(request):
        return web.Response(body=body, content_type="application/x-ndjson")

    app = web.Application()
    app.router.add_post("/ndjson", handler)

    async def scenario(base):
        return await call_runner(f"{base}/ndjson", payload={})

    result = _run(app, scenario)
    assert result.content == body
    assert result.content_type == "application/x-ndjson"
    assert result.data == {}


def test_invalid_json_with_json_content_type_raises():
    async def handler(request):
        return web.Response(text="not json", content_type="application/json")

    app = web.Application()
    app.router.add_post("/bad", handler)

    async def scenario(base):
        return await call_runner(f"{base}/bad", payload={})

    with pytest.raises(LivepeerGatewayError, match="did not return valid JSON") as excinfo:
        _run(app, scenario)
    # The content type is in the message: it is what routed us into parsing.
    assert "content_type=application/json" in str(excinfo.value)


def test_json_array_still_rejected():
    async def handler(request):
        return web.json_response([1, 2, 3])

    app = web.Application()
    app.router.add_post("/arr", handler)

    async def scenario(base):
        return await call_runner(f"{base}/arr", payload={})

    with pytest.raises(LivepeerGatewayError, match="expected JSON object"):
        _run(app, scenario)


def test_http_error_still_raises_with_binary_endpoint():
    async def handler(request):
        return web.Response(status=404, text="nope")

    app = web.Application()
    app.router.add_post("/missing", handler)

    async def scenario(base):
        return await call_runner(f"{base}/missing", payload={})

    with pytest.raises(LivepeerHTTPError):
        _run(app, scenario)
