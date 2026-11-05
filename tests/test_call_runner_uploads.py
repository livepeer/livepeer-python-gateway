from __future__ import annotations

import asyncio
import io
import json
from contextlib import ExitStack, asynccontextmanager
from types import SimpleNamespace
from unittest import mock

import pytest

from livepeer_gateway import (
    LiveRunnerInstance,
    LiveRunnerPriceInfo,
    RunnerSelectionCursor,
    StreamPayload,
    call_runner,
)
from livepeer_gateway import live_runner
from livepeer_gateway.errors import (
    LivepeerGatewayError,
    LivepeerHTTPError,
    NoRunnerAvailableError,
    SignerRefreshRequired,
)


CHALLENGE = {
    "payment_params": "opaque",
    "manifest_id": "session",
    "payment_url": "http://payment.example/session",
}


@pytest.fixture
def signer(monkeypatch):
    monkeypatch.setattr(live_runner, "get_signer_info", mock.AsyncMock(return_value=SimpleNamespace(address="payer")))
    payment = mock.AsyncMock(return_value={"payment": "paid", "segCreds": "segment", "state": {}})
    monkeypatch.setattr("livepeer_gateway.http.post_json", payment)
    return payment


@pytest.fixture
async def no_loop_errors():
    """A failed asyncio callback is logged, rather than failing the awaiting task."""
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()
    errors = []
    loop.set_exception_handler(lambda _, context: errors.append(context))
    try:
        yield
        # Future cancellation can queue callbacks after request cleanup returns.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous)
    assert not errors, errors


@asynccontextmanager
async def serve(handler):
    tasks = set()
    errors = []

    async def connected(reader, writer):
        task = asyncio.current_task()
        tasks.add(task)
        try:
            await handler(reader, writer)
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        except Exception as exc:
            errors.append(exc)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass
            tasks.discard(task)

    server = await asyncio.start_server(connected, "127.0.0.1", 0)
    async with server:
        port = server.sockets[0].getsockname()[1]
        try:
            yield f"http://127.0.0.1:{port}"
        finally:
            if tasks:
                await asyncio.wait_for(asyncio.gather(*tasks), 2)
    if errors:
        raise errors[0]


async def headers(reader, *, expected_request=None):
    raw = await reader.readuntil(b"\r\n\r\n")
    lines = raw.decode().split("\r\n")
    if expected_request is not None:
        assert lines[0] == expected_request
    result = dict(line.split(": ", 1) for line in lines[1:] if line)
    assert "Expect" not in result
    return result


async def body(reader, request_headers):
    if request_headers.get("Transfer-Encoding") == "chunked":
        chunks = []
        while True:
            chunk = await read_chunk(reader)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    return await reader.readexactly(int(request_headers.get("Content-Length", 0)))


async def read_chunk(reader):
    line = await reader.readline()
    if not line:
        raise asyncio.IncompleteReadError(b"", 1)
    size = int(line, 16)
    chunk = await reader.readexactly(size)
    assert await reader.readexactly(2) == b"\r\n"
    return chunk


async def respond(writer, status=200, data=None, *, fragment=False, location=None):
    data = {"ok": True} if data is None else data
    raw = json.dumps(data).encode()
    redirect = f"Location: {location}\r\n" if location else ""
    head = (
        f"HTTP/1.1 {status} Result\r\nContent-Type: application/json\r\n"
        f"Content-Length: {len(raw)}\r\n{redirect}Connection: close\r\n\r\n".encode()
    )
    writer.write(head if fragment else head + raw)
    await writer.drain()
    if fragment:
        await asyncio.sleep(0.02)
        writer.write(raw)
        await writer.drain()


class Source:
    def __init__(self, *chunks):
        self.chunks = iter(chunks)
        self.iterations = 0
        self.reads = 0
        self.closed = False

    def __aiter__(self):
        self.iterations += 1
        return self

    async def __anext__(self):
        self.reads += 1
        try:
            return next(self.chunks)
        except StopIteration:
            raise StopAsyncIteration from None

    async def aclose(self):
        self.closed = True


def candidate(url, *, price=None, mode="single-shot"):
    return LiveRunnerInstance(
        url, "test", "runner", mode, url, {},
        price_info=LiveRunnerPriceInfo(price, "wei", "fixed") if price is not None else None,
    )


CUSTOM_TYPE = "application/vnd.test+json; charset=utf-8"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("kwargs, expected, content_type", [
    ({}, b"", None),
    ({"payload": None}, b"", None),
    ({"payload": {}}, b"{}", "application/json"),
    ({"payload": {"x": 1}}, b'{"x": 1}', "application/json"),
    ({"payload": "héllo"}, "héllo".encode(), "application/octet-stream"),
    ({"payload": b"\x00\xff"}, b"\x00\xff", "application/octet-stream"),
    ({"payload": bytearray(b"abc")}, b"abc", "application/octet-stream"),
    ({"payload": memoryview(b"abc")}, b"abc", "application/octet-stream"),
    ({"payload": b""}, b"", "application/octet-stream"),
    ({"payload": ""}, b"", "application/octet-stream"),
    ({"payload": {"x": 1}, "content_type": CUSTOM_TYPE}, b'{"x": 1}', CUSTOM_TYPE),
    ({"payload": b"abc", "content_type": CUSTOM_TYPE}, b"abc", CUSTOM_TYPE),
    ({"payload": "abc", "content_type": CUSTOM_TYPE}, b"abc", CUSTOM_TYPE),
    ({"payload": None, "content_type": CUSTOM_TYPE}, b"", CUSTOM_TYPE),
])
async def test_request_bodies(kwargs, expected, content_type, stream):
    async def handler(reader, writer):
        request_headers = await headers(reader)
        assert request_headers["Accept"] == "*/*"
        assert request_headers.get("Content-Type") == content_type
        assert await body(reader, request_headers) == expected
        await respond(writer)

    async with serve(handler) as url:
        result = await call_runner(url, runner=candidate(url, price=10), stream=stream, **kwargs)
        if stream:
            async with result:
                assert json.loads(b"".join([chunk async for chunk in result.aiter_bytes()])) == {"ok": True}
        else:
            assert result.data == {"ok": True}


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("refresh", [False, True])
async def test_payment_challenge_does_not_touch_source(stream, refresh, signer, no_loop_errors):
    source = Source(b"one", b"two")
    attempts = 0
    sessions = []

    async def sign(*args, **kwargs):
        assert source.iterations == source.reads == 0
        if refresh and signer.await_count == 1:
            raise SignerRefreshRequired("refresh")
        return signer.return_value

    signer.side_effect = sign

    async def handler(reader, writer):
        nonlocal attempts
        request_headers = await headers(reader, expected_request="PATCH /upload?model=test HTTP/1.1")
        attempts += 1
        assert request_headers["Livepeer-Payer-Address"] == "payer"
        if "Livepeer-Payment" not in request_headers:
            assert source.iterations == source.reads == 0
            assert request_headers["Content-Length"] == "0"
            assert "Transfer-Encoding" not in request_headers
            assert await body(reader, request_headers) == b""
            writer.write(b"HTTP/1.1 100 Continue\r\n\r\n")
            await writer.drain()
            await respond(writer, 402, CHALLENGE, fragment=True)
            assert await reader.read() == b""
            return
        assert request_headers["Livepeer-Payment"] == "paid"
        assert request_headers["Livepeer-Segment"] == "segment"
        assert [session.manifest_id for session in sessions] == [CHALLENGE["manifest_id"]]
        assert await body(reader, request_headers) == b"onetwo"
        await respond(writer)

    async with serve(handler) as url:
        result = await asyncio.wait_for(
            call_runner(
                url + "/upload?model=test", runner=candidate("http://unused", price="10"),
                payload=StreamPayload(source=source, retry_buffer=False), method="PATCH",
                signer_url="http://signer.example",
                on_payment_session=sessions.append,
                stream=stream,
            ),
            2,
        )
        if stream:
            await result.aclose()
    assert attempts == (3 if refresh else 2)
    assert source.reads == 3
    assert not source.closed
    assert signer.await_count == (2 if refresh else 1)
    payment_request = signer.call_args.args[1]
    assert payment_request["orchestrator"] == CHALLENGE["payment_params"]
    assert payment_request["ManifestID"] == CHALLENGE["manifest_id"]
    assert payment_request["maxPrice"] == {"price": 10.12, "currency": "wei", "unit": "fixed"}


@pytest.mark.parametrize("status,kind", [(402, "nonseekable"), (503, "iterator"), (307, "file"), (200, "iterator")])
async def test_early_rejection_preserves_source_for_next_candidate(tmp_path, status, kind):
    class NonSeekable(io.BytesIO):
        def seekable(self):
            return False

    with ExitStack() as sources:
        if kind == "file":
            path = tmp_path / "upload.bin"
            path.write_bytes(b"skip-body")
            source = sources.enter_context(path.open("rb"))
            source.seek(5)
        elif kind == "nonseekable":
            source = sources.enter_context(NonSeekable(b"body"))
        else:
            source = Source(b"bo", b"dy")

        def assert_unread():
            assert not source.closed
            if isinstance(source, Source):
                assert source.iterations == source.reads == 0
            else:
                assert source.tell() == (5 if kind == "file" else 0)

        async def reject(reader, writer):
            request_headers = await headers(reader)
            assert await body(reader, request_headers) == b""
            assert_unread()
            await respond(writer, status, location=second if status == 307 else None)
            assert await reader.read() == b""

        async def accept(reader, writer):
            request_headers = await headers(reader)
            assert await body(reader, request_headers) == b"body"
            await respond(writer)

        async with serve(reject) as first, serve(accept) as second:
            payload = source if kind == "file" else StreamPayload(source=source, retry_buffer=False)
            async with RunnerSelectionCursor([candidate(first, price=10), candidate(second)], body=payload) as cursor:
                result = await cursor.next()
                assert result.data == {"ok": True} and result.runner_url == second
                assert len(cursor.rejections) == 1
        assert not source.closed


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("status,match", [(None, "failed to reach endpoint"), (200, "expected HTTP 402")])
async def test_incomplete_challenge_leaves_source_unread(stream, status, match, no_loop_errors):
    source = Source(b"never")

    async def handler(reader, writer):
        assert await body(reader, await headers(reader)) == b""
        if status == 200:
            writer.write(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n")
            await writer.drain()
        assert await reader.read() == b""

    async with serve(handler) as url:
        with pytest.raises(LivepeerGatewayError, match=match):
            await asyncio.wait_for(
                call_runner(
                    runner=candidate(url, price=10), payload=source, stream=stream,
                    timeout=0.05 if status is None else 10,
                ),
                1,
            )
    assert source.iterations == source.reads == 0
    assert not source.closed


@pytest.mark.parametrize("stream", [False, True])
async def test_closed_source_is_rejected_before_challenge(stream, signer):
    source = io.BytesIO(b"data")
    source.close()
    with mock.patch.object(live_runner, "_request_body", new_callable=mock.AsyncMock) as request:
        with pytest.raises(LivepeerGatewayError, match="no longer replayable"):
            await call_runner(
                runner=candidate("http://unused", price=10), payload=source,
                signer_url="http://signer.example", stream=stream,
            )
        request.assert_not_called()
    signer.assert_not_called()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("price,mode", [
    (None, "single-shot"), (0, "single-shot"), ("0.0", "single-shot"),
    (-1, "single-shot"), ("unknown", "single-shot"), ("NaN", "single-shot"),
    ("Infinity", "single-shot"), (10, "persistent"), (10, ""), (10, "unknown"),
])
async def test_ordinary_upload_starts_immediately(stream, price, mode):
    source = Source(b"direct")

    async def handler(reader, writer):
        request_headers = await headers(reader)
        assert await body(reader, request_headers) == b"direct"
        await respond(writer)

    async with serve(handler) as url:
        result = await call_runner(
            runner=candidate(url, price=price, mode=mode),
            payload=StreamPayload(source=source), stream=stream,
        )
        if stream:
            await result.aclose()
    assert not source.closed


async def test_buffered_upload_can_be_reused_by_cursor_until_close():
    source = Source(b"one", b"two")
    received = []

    async def handler(reader, writer):
        request_headers = await headers(reader)
        assert request_headers["Content-Type"] == "video/test"
        received.append(await body(reader, request_headers))
        await respond(writer)

    async with serve(handler) as url:
        async with RunnerSelectionCursor(
            [candidate(url), candidate(url)], body=source, content_type="video/test"
        ) as cursor:
            await cursor.next()
            state = cursor._payload_state
            cache = state.cache
            assert cache is not None and not cache.closed
            await cursor.next()
        assert cache.closed and state.closed
        with pytest.raises(LivepeerGatewayError, match="closed"):
            await cursor.next()
    assert received == [b"onetwo", b"onetwo"]
    assert source.reads == 3 and not source.closed


async def test_seekable_file_replays_from_initial_position_without_buffering():
    source = io.BytesIO(b"prefix-data")
    source.seek(7)
    received = []

    async def handler(reader, writer):
        request_headers = await headers(reader)
        received.append(await body(reader, request_headers))
        await respond(writer, 503 if len(received) == 1 else 200)

    async with serve(handler) as url:
        async with RunnerSelectionCursor(
            [candidate(url), candidate(url)],
            body=StreamPayload(source=source, retry_buffer=False),
        ) as cursor:
            await cursor.next()
            assert cursor._payload_state.cache is None
    assert received == [b"data", b"data"]
    assert not source.closed


@pytest.mark.parametrize("options", [
    {"retry_buffer": False},
    {"retry_buffer_memory_limit": 3, "retry_buffer_limit": 6},
])
async def test_unreplayable_upload_finishes_but_stops_failover(options):
    source = Source(b"prefix", b"over-limit", b"tail")
    attempts = 0

    async def handler(reader, writer):
        nonlocal attempts
        attempts += 1
        request_headers = await headers(reader)
        assert await body(reader, request_headers) == b"prefixover-limittail"
        await respond(writer, 503)

    async with serve(handler) as url:
        cursor = RunnerSelectionCursor(
            [candidate(url), candidate(url)], body=StreamPayload(source=source, **options)
        )
        with pytest.raises(LivepeerGatewayError, match="no longer replayable") as error:
            await cursor.next()
        assert isinstance(error.value.__cause__, LivepeerHTTPError)
        assert cursor._payload_state.closed and cursor._payload_state.cache is None
    assert attempts == 1 and not source.closed


@pytest.mark.parametrize("success", [False, True])
async def test_exhausted_cursor_releases_cache(success):
    source = Source(b"data")

    async def handler(reader, writer):
        request_headers = await headers(reader)
        await body(reader, request_headers)
        await respond(writer, 200 if success else 503)

    async with serve(handler) as url:
        cursor = RunnerSelectionCursor([candidate(url)], body=StreamPayload(source=source, retry_buffer=not success))
        if success:
            await cursor.next()
        with pytest.raises(NoRunnerAvailableError):
            await cursor.next()
        assert cursor._payload_state.closed
        assert cursor._payload_state.cache is None
    assert not source.closed


async def test_disconnected_upload_failover_adopts_pending_source_read():
    read_started = asyncio.Event()
    finish_read = asyncio.Event()
    source_cancelled = False
    source_reads = []
    rejections = 0

    async def source():
        nonlocal source_cancelled
        try:
            source_reads.append(b"prefix")
            yield b"prefix"
            read_started.set()
            await finish_read.wait()
            source_reads.append(b"middle")
            yield b"middle"
            source_reads.append(b"tail")
            yield b"tail"
        except asyncio.CancelledError:
            source_cancelled = True
            raise

    async def reject(reader, writer):
        nonlocal rejections
        rejections += 1
        await headers(reader)
        assert await read_chunk(reader) == b"prefix"
        await read_started.wait()

    async def accept(reader, writer):
        request_headers = await headers(reader)
        assert not source_cancelled
        assert await read_chunk(reader) == b"prefix"
        finish_read.set()
        assert await body(reader, request_headers) == b"middletail"
        await respond(writer)

    async with serve(reject) as first, serve(accept) as second:
        async with RunnerSelectionCursor(
            [candidate(first), candidate(second)], body=source(), method="PUT"
        ) as cursor:
            assert (await asyncio.wait_for(cursor.next(), 2)).data == {"ok": True}
    assert not source_cancelled
    assert source_reads == [b"prefix", b"middle", b"tail"]
    assert rejections == 1


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("known_onchain", [False, True])
async def test_partial_upload_payment_retry_replays_prefix(stream, known_onchain, signer):
    read_started = asyncio.Event()
    finish_read = asyncio.Event()
    attempts = 0

    async def source():
        yield b"prefix"
        read_started.set()
        await finish_read.wait()
        yield b"tail"

    async def handler(reader, writer):
        nonlocal attempts
        request_headers = await headers(reader)
        attempts += 1
        if known_onchain and attempts == 1:
            assert await body(reader, request_headers) == b""
            await respond(writer, 402, CHALLENGE, fragment=True)
            return
        assert await read_chunk(reader) == b"prefix"
        if attempts == (2 if known_onchain else 1):
            await read_started.wait()
            await respond(writer, 402, CHALLENGE, fragment=True)
        else:
            assert request_headers["Livepeer-Payment"] == "paid"
            finish_read.set()
            assert await body(reader, request_headers) == b"tail"
            await respond(writer)

    async with serve(handler) as url:
        result = await asyncio.wait_for(
            call_runner(
                runner=candidate(url, price=10 if known_onchain else None),
                payload=source(),
                signer_url="http://signer.example",
                payment_unit="fixed",
                stream=stream,
            ),
            2,
        )
        if stream:
            await result.aclose()
    assert signer.await_count == (2 if known_onchain else 1)
    assert attempts == (3 if known_onchain else 2)


@pytest.mark.parametrize("mode", ["buffered", "stream", "cursor"])
@pytest.mark.parametrize("known_onchain", [False, True])
async def test_unbuffered_consumed_body_is_not_paid_again_or_retried(mode, known_onchain, signer):
    source = Source(b"data")

    async def handler(reader, writer):
        request_headers = await headers(reader)
        if known_onchain and "Livepeer-Payment" not in request_headers:
            assert await body(reader, request_headers) == b""
            await respond(writer, 402, CHALLENGE)
            return
        assert await body(reader, request_headers) == b"data"
        await respond(writer, 402, CHALLENGE)

    async with serve(handler) as url:
        with pytest.raises(
            LivepeerGatewayError, match="no longer replayable"
        ) as error:
            payload = StreamPayload(source=source, retry_buffer=False)
            runner = candidate(url, price=10 if known_onchain else None)
            if mode == "cursor":
                await RunnerSelectionCursor([runner], body=payload, signer_url="http://signer.example").next()
            else:
                await call_runner(runner=runner, payload=payload, signer_url="http://signer.example", payment_unit="fixed", stream=mode == "stream")
        assert isinstance(error.value.__cause__, LivepeerHTTPError)
    assert signer.await_count == (1 if known_onchain else 0)


@pytest.mark.parametrize("stream", [False, True])
async def test_upload_cancellation_releases_cache(stream):
    read_started = asyncio.Event()

    async def source():
        yield b"prefix"
        read_started.set()
        await asyncio.Future()

    async def handler(reader, writer):
        await headers(reader)
        assert await read_chunk(reader) == b"prefix"
        await reader.read()

    states = []
    make_state = live_runner._stream_state

    def capture_state(payload):
        state = make_state(payload)
        states.append(state)
        return state

    with mock.patch.object(live_runner, "_stream_state", side_effect=capture_state):
        async with serve(handler) as url:
            task = asyncio.create_task(
                call_runner(url, payload=source(), stream=stream)
            )
            await read_started.wait()
            cache = states[0].cache
            assert cache is not None
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert states[0].closed and cache.closed
            assert states[0]._writer is states[0]._pending is None


async def test_response_stream_owns_active_upload_until_close():
    read_started = asyncio.Event()
    response_closed = asyncio.Event()

    async def source():
        yield b"prefix"
        read_started.set()
        await asyncio.Future()

    async def handler(reader, writer):
        await headers(reader)
        assert await read_chunk(reader) == b"prefix"
        await read_started.wait()
        # Send response headers while the request source is still live.
        writer.write(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n")
        await writer.drain()
        await reader.read()
        response_closed.set()

    async with serve(handler) as url:
        result = await call_runner(url, payload=source(), stream=True)
        state = result._payload_state
        cache = state.cache
        assert not state.closed
        await result.aclose()
        assert state.closed and cache.closed
        await asyncio.wait_for(response_closed.wait(), 1)


@pytest.mark.parametrize("stream", [False, True])
async def test_response_mode_controls_upload_timeout(stream):
    async def source():
        await asyncio.sleep(0.1)
        yield b"delayed"

    async def handler(reader, writer):
        request_headers = await headers(reader)
        assert await body(reader, request_headers) == b"delayed"
        await respond(writer)

    async with serve(handler) as url:
        if stream:
            async with await call_runner(
                url, payload=source(), stream=True, timeout=0.05
            ) as result:
                assert json.loads(b"".join([chunk async for chunk in result.aiter_bytes()])) == {"ok": True}
        else:
            with pytest.raises(LivepeerGatewayError):
                await call_runner(url, payload=source(), timeout=0.05)
