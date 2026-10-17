from __future__ import annotations

import asyncio
import io
import tempfile
import threading
from unittest import mock

import pytest

from livepeer_gateway import StreamPayload
from livepeer_gateway.errors import LivepeerGatewayError
from livepeer_gateway.runner_payload import _StreamState


@pytest.fixture
async def make_state():
    states = []

    def make(source, **options):
        state = _StreamState(StreamPayload(source=source, **options))
        states.append(state)
        return state

    yield make
    for state in states:
        await state.aclose()


async def _chunks(*chunks):
    for chunk in chunks:
        yield chunk


async def collect(state):
    return b"".join([chunk async for chunk in state.chunks()])


@pytest.mark.parametrize("memory,total", [(0, 1), (1, 0), (2, 1), (-1, 10)])
def test_invalid_buffer_limits(memory, total):
    with pytest.raises(ValueError, match="limits"):
        StreamPayload(
            source=io.BytesIO(),
            retry_buffer_memory_limit=memory,
            retry_buffer_limit=total,
        )


def test_text_files_are_rejected():
    with pytest.raises(TypeError, match="binary"):
        StreamPayload(source=io.StringIO("text"))


async def test_spills_and_replays_without_rereading_source(make_state):
    chunks = [b"one", b"two", b"three"]
    reads = []

    async def source():
        for chunk in chunks:
            reads.append(chunk)
            yield chunk

    state = make_state(source(), retry_buffer_memory_limit=4, retry_buffer_limit=100)
    assert await collect(state) == b"onetwothree"
    cache = state.cache
    assert cache._rolled
    assert await collect(state) == b"onetwothree"
    assert reads == chunks
    await state.aclose()
    assert cache.closed


@pytest.mark.parametrize("length,replayable", [(5, True), (6, False)])
async def test_total_limit_is_checked_before_writing(length, replayable, make_state):
    state = make_state(_chunks(b"x" * length), retry_buffer_memory_limit=2, retry_buffer_limit=5)
    assert await collect(state) == b"x" * length
    assert state.replayable is replayable
    if replayable:
        assert state.cache_size == 5
        assert await collect(state) == b"x" * length
    else:
        assert state.cache is None
        with pytest.raises(LivepeerGatewayError, match="no longer replayable"):
            await collect(state)


async def test_cancelled_attempt_replays_prefix_and_adopts_pending_read(make_state):
    read_started = asyncio.Event()
    finish_read = asyncio.Event()
    reads = []

    async def source():
        for chunk in (b"prefix", b"middle", b"end"):
            reads.append(chunk)
            if chunk == b"middle":
                read_started.set()
                await finish_read.wait()
            yield chunk

    state = make_state(source())
    first = state.chunks()
    assert await anext(first) == b"prefix"
    upload = asyncio.create_task(anext(first))
    await read_started.wait()
    upload.cancel()
    with pytest.raises(asyncio.CancelledError):
        await upload
    assert not state._pending.cancelled()
    # More than one interrupted attempt may replay the same prefix while
    # the *one* source read remains pending.
    second = state.chunks()
    assert await anext(second) == b"prefix"
    await second.aclose()
    third = state.chunks()
    assert await anext(third) == b"prefix"
    finish_read.set()
    assert b"".join([chunk async for chunk in third]) == b"middleend"
    assert reads == [b"prefix", b"middle", b"end"]


async def test_pending_overflow_preserves_prefix_for_current_retry(make_state):
    read_started = asyncio.Event()
    finish_read = asyncio.Event()
    prefix = b"x" * (64 * 1024 + 1)

    async def source():
        yield prefix
        read_started.set()
        await finish_read.wait()
        yield b"new-data"
        yield b"remainder"

    state = make_state(source(), retry_buffer_memory_limit=4, retry_buffer_limit=len(prefix))
    first = state.chunks()
    assert await anext(first) == prefix
    task = asyncio.create_task(anext(first))
    await read_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    retry = state.chunks()
    assert await anext(retry) == prefix[:-1]
    finish_read.set()
    # The pending chunk may complete before this retry finishes its cached prefix.
    await asyncio.shield(state._pending)
    assert b"".join([chunk async for chunk in retry]) == b"xnew-dataremainder"
    assert not state.replayable
    assert state.cache is None


@pytest.mark.parametrize("failure", ["create", "write", "rollover"])
async def test_cache_failure_allows_current_upload_to_finish(failure, make_state):
    state = make_state(_chunks(b"first", b"second"), retry_buffer_memory_limit=6, retry_buffer_limit=100)
    factory = tempfile.SpooledTemporaryFile

    def make_cache(*args, **kwargs):
        if failure == "create":
            raise OSError("no space")
        cache = factory(*args, **kwargs)
        setattr(cache, failure, mock.Mock(side_effect=OSError("no space")))
        return cache

    with mock.patch(
        "livepeer_gateway.runner_payload.tempfile.SpooledTemporaryFile",
        side_effect=make_cache,
    ):
        assert await collect(state) == b"firstsecond"
    assert not state.replayable
    assert state.cache is None


async def test_failed_source_is_not_retried(make_state):
    async def source():
        yield b"prefix"
        raise RuntimeError("source failed")

    state = make_state(source())
    with pytest.raises(RuntimeError, match="source failed"):
        await collect(state)
    assert not state.replayable


async def test_closed_unread_file_is_not_retried(make_state):
    file = io.BytesIO(b"data")
    state = make_state(file)
    file.close()
    assert not state.replayable
    with pytest.raises(LivepeerGatewayError, match="closed"):
        await collect(state)


async def test_non_bytes_source_fails_clearly(make_state):
    state = make_state(_chunks("text"))
    with pytest.raises(TypeError, match="produce bytes"):
        await collect(state)


async def test_replay_read_failure_disables_future_retries(make_state):
    state = make_state(_chunks(b"data"))
    assert await collect(state) == b"data"
    state.cache.read = mock.Mock(side_effect=OSError("storage unavailable"))
    with pytest.raises(
        LivepeerGatewayError, match="Cannot read upload replay buffer"
    ):
        await collect(state)
    assert not state.replayable and state.cache is None


async def test_failed_rewind_disables_future_retries(make_state):
    source = io.BytesIO(b"data")
    state = make_state(source, retry_buffer=False)
    assert await collect(state) == b"data"
    source.seek = mock.Mock(side_effect=OSError("seek failed"))
    with pytest.raises(LivepeerGatewayError, match="Cannot rewind upload source"):
        await collect(state)
    assert not state.replayable
    source.close()


@pytest.mark.parametrize("fail", [False, True])
async def test_cancellation_joins_cache_write_before_retry(make_state, fail):
    state = make_state(_chunks(b"prefix", b"middle", b"tail"), retry_buffer_memory_limit=2)
    attempt = state.chunks()
    assert await anext(attempt) == b"prefix"
    cache = state.cache
    entered, finish = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    write = cache.write

    def blocked_write(chunk):
        loop.call_soon_threadsafe(entered.set)
        assert finish.wait(2), "test did not release the cache write"
        if fail:
            raise OSError("disk full")
        return write(chunk)

    with mock.patch.object(cache, "write", side_effect=blocked_write):
        upload = asyncio.create_task(anext(attempt))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            upload.cancel()
            await asyncio.sleep(0)
            assert not upload.done()
        finally:
            finish.set()
        with pytest.raises(asyncio.CancelledError):
            await upload
    if fail:
        assert not state.replayable and cache.closed
    else:
        assert await collect(state) == b"prefixmiddletail"


async def test_memory_cache_does_not_dispatch_file_io(make_state):
    state = make_state(_chunks(b"one", b"two"))
    with mock.patch("asyncio.to_thread", side_effect=AssertionError("unexpected disk I/O")):
        assert await collect(state) == await collect(state) == b"onetwo"
        await state.aclose()
