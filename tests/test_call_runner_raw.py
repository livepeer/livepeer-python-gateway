"""Tests for raw bytes in call_runner, in both directions.

Responses: single-document JSON (``application/json`` or an RFC 6839 ``+json``
suffix) keeps today's behavior: parsed into ``result.data``, strict about being an
object. Anything else — binary, or a multi-document format like ndjson — returns
the body unparsed in ``result.content`` with ``result.content_type`` set.

Requests: ``payload`` keeps sending JSON. ``content`` sends the bytes as is with
``content_type`` (default ``application/octet-stream``), and the two are mutually
exclusive.
"""

from __future__ import annotations

import asyncio

import pytest
from aiohttp import web

from livepeer_gateway.errors import LivepeerGatewayError, LivepeerHTTPError
from livepeer_gateway.http import request_json
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


# --- Responses: what the runner sends back --------------------------------------


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


def test_invalid_json_encoding_raises_gateway_error():
    async def handler(request):
        return web.Response(
            body=b'{"message": "\xff"}',
            content_type="application/json",
        )

    app = web.Application()
    app.router.add_route("*", "/bad-encoding", handler)

    async def scenario(base):
        errors = []
        for request in (
            request_json(f"{base}/bad-encoding"),
            call_runner(f"{base}/bad-encoding", payload={}),
        ):
            try:
                await request
            except Exception as exc:
                errors.append(exc)
        return errors

    errors = _run(app, scenario)
    assert len(errors) == 2
    assert all(isinstance(error, LivepeerGatewayError) for error in errors)
    assert all("did not return valid JSON" in str(error) for error in errors)


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


# --- Requests: what the client sends ---------------------------------------------


def _echo_app(seen: dict) -> web.Application:
    """Record the request body, method and content type the server receives."""

    async def handler(request):
        seen["body"] = await request.read()
        seen["content_type"] = request.headers.get("Content-Type", "")
        seen["method"] = request.method
        return web.json_response({"received": len(seen["body"])})

    app = web.Application()
    app.router.add_route("*", "/in", handler)
    return app


def test_binary_request_sent_raw():
    seen: dict = {}

    async def scenario(base):
        return await call_runner(f"{base}/in", content=FAKE_JPEG, content_type="image/jpeg")

    result = _run(_echo_app(seen), scenario)
    assert seen["body"] == FAKE_JPEG
    assert seen["content_type"] == "image/jpeg"
    assert seen["method"] == "POST"
    assert result.data == {"received": len(FAKE_JPEG)}


def test_binary_request_defaults_to_octet_stream():
    seen: dict = {}

    async def scenario(base):
        return await call_runner(f"{base}/in", content=b"\x00\x01\x02")

    _run(_echo_app(seen), scenario)
    assert seen["body"] == b"\x00\x01\x02"
    assert seen["content_type"] == "application/octet-stream"


def test_empty_binary_request_is_posted():
    seen: dict = {}

    async def scenario(base):
        return await call_runner(f"{base}/in", content=b"")

    _run(_echo_app(seen), scenario)
    assert seen["method"] == "POST"
    assert seen["body"] == b""


def test_json_request_unchanged():
    seen: dict = {}

    async def scenario(base):
        return await call_runner(f"{base}/in", payload={"x": 1})

    _run(_echo_app(seen), scenario)
    assert seen["content_type"] == "application/json"
    assert seen["body"] == b'{"x": 1}'


def test_binary_request_with_stream():
    seen: dict = {}

    async def scenario(base):
        async with await call_runner(
            f"{base}/in", content=FAKE_JPEG, content_type="image/jpeg", stream=True
        ) as stream:
            return b"".join([chunk async for chunk in stream.aiter_bytes()])

    body = _run(_echo_app(seen), scenario)
    assert seen["body"] == FAKE_JPEG
    assert seen["content_type"] == "image/jpeg"
    assert b"received" in body


def test_payload_and_content_are_mutually_exclusive():
    async def scenario(base):
        return await call_runner(f"{base}/in", payload={"x": 1}, content=b"y")

    with pytest.raises(LivepeerGatewayError):
        _run(_echo_app({}), scenario)
