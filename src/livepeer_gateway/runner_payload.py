"""Runner upload configuration and operation-scoped replay state."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import io
import logging
import tempfile
from collections.abc import AsyncIterable, AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any, BinaryIO

from aiohttp.payload import Payload

from ._http_continue import _cancel_writer
from .errors import LivepeerGatewayError

_LOG = logging.getLogger(__name__)
_CHUNK_SIZE = 64 * 1024


@dataclass(frozen=True, kw_only=True)
class StreamPayload:
    """A caller-owned source with 100-continue and bounded replay buffering.

    Limits are decimal bytes. Overflow or storage failure disables later retries
    without interrupting the upload. Seekable files rewind instead of buffering,
    even with ``retry_buffer=False``. Bare files and iterables use these defaults.
    """

    source: AsyncIterable[bytes] | BinaryIO
    expect100: bool = True
    retry_buffer: bool = True
    retry_buffer_memory_limit: int = 25_000_000
    retry_buffer_limit: int = 250_000_000

    def __post_init__(self) -> None:
        if not (0 < self.retry_buffer_memory_limit <= self.retry_buffer_limit):
            raise ValueError("Replay buffer limits must satisfy 0 < memory limit <= total limit")
        read = getattr(self.source, "read", None)
        if isinstance(self.source, io.TextIOBase) or (
            not isinstance(self.source, AsyncIterable)
            and (not callable(read) or inspect.iscoroutinefunction(read))
        ):
            raise TypeError("StreamPayload requires a synchronous binary file or async byte iterable")


type RunnerPayload = (
    StreamPayload | AsyncIterable[bytes] | BinaryIO | dict[str, Any]
    | bytes | bytearray | memoryview | str | None
)


class _ReplayUnavailableError(LivepeerGatewayError):
    pass


async def _file_io[T](operation: Callable[[], T], *, threaded: bool = True) -> T:
    """Join worker I/O before cancellation can hand its file to another attempt."""
    if not threaded:
        return operation()
    task = asyncio.create_task(asyncio.to_thread(operation))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        with contextlib.suppress(Exception):
            await task
        raise


class _StreamState:
    def __init__(self, config: StreamPayload) -> None:
        self.config = config
        self.source = config.source
        self.started = self.closed = self.eof = self.failed = False
        self.cache_size = 0
        self.cache: tempfile.SpooledTemporaryFile[bytes] | None = None
        self._iterator: AsyncIterator[bytes] | None = None
        self._pending: asyncio.Task[bytes] | None = None
        self._writer: asyncio.Task[None] | None = None
        self._start_position: int | None = None
        if not isinstance(self.source, AsyncIterable):
            with contextlib.suppress(AttributeError, OSError, ValueError):
                if self.source.seekable():
                    self._start_position = self.source.tell()
        self._buffering = config.retry_buffer and self._start_position is None

    @property
    def replayable(self) -> bool:
        if self.closed:
            return False
        source_open = not (
            getattr(self.source, "closed", False)
            or (hasattr(self.source, "ag_frame") and self.source.ag_frame is None)
        )
        if self._start_position is not None:
            return source_open
        return not self.failed and (
            (self.eof and self._buffering)
            or (source_open and (not self.started or self._buffering))
        )

    def require_replayable(self, cause: Exception | None = None) -> None:
        if not self.replayable:
            raise _ReplayUnavailableError(
                "Cannot retry runner call: upload source is closed or its consumed data is no longer replayable"
            ) from cause

    async def _drop_cache(self) -> None:
        cache, self.cache = self.cache, None
        self.cache_size = 0
        if cache is not None:
            with contextlib.suppress(OSError):
                await _file_io(cache.close, threaded=cache._rolled)

    async def _retain(self, chunk: bytes) -> None:
        size = self.cache_size + len(chunk)
        retained = False
        storage_error = None
        try:
            if size > self.config.retry_buffer_limit:
                self._buffering = False
                await self._drop_cache()
                return
            if self.cache is None:
                self.cache = tempfile.SpooledTemporaryFile(
                    max_size=self.config.retry_buffer_memory_limit, mode="w+b"
                )
            cache = self.cache
            on_disk = size > self.config.retry_buffer_memory_limit

            def append() -> None:
                nonlocal retained, storage_error
                try:
                    if on_disk:
                        cache.rollover()  # Spill before a large chunk grows the RAM buffer.
                    cache.seek(self.cache_size)
                    cache.write(chunk)
                    retained = True
                except OSError as exc:
                    storage_error = exc
                    raise

            await _file_io(append, threaded=on_disk)
        except OSError as exc:
            storage_error = exc
        finally:
            # _file_io joins a cancelled write. Commit it exactly once even if
            # cancellation prevents chunks() from receiving the result.
            if retained:
                self.cache_size = size
                self._pending = None
            elif storage_error is not None:
                self._buffering = False
                await self._drop_cache()
                _LOG.warning("Upload replay storage failed; continuing without replay", exc_info=storage_error)

    async def _read_source(self) -> bytes:
        try:
            if isinstance(self.source, AsyncIterable):
                if self._iterator is None:
                    self._iterator = aiter(self.source)
                try:
                    chunk = await anext(self._iterator)
                except StopAsyncIteration:
                    self.eof = True
                    return b""
            else:
                chunk = await _file_io(lambda: self.source.read(_CHUNK_SIZE))
                if not chunk:
                    self.eof = True
            if not isinstance(chunk, bytes):
                raise TypeError("Upload streams must produce bytes")
            return chunk
        except BaseException:
            self.failed = True
            raise

    async def chunks(self) -> AsyncIterator[bytes]:
        self.require_replayable()
        if self._start_position is not None and self.started:
            if self._pending is not None:
                with contextlib.suppress(Exception):
                    await asyncio.shield(self._pending)
                self._pending = None
            try:
                await _file_io(lambda: self.source.seek(self._start_position))
            except (OSError, ValueError) as exc:
                self._start_position = None
                self.failed = True
                raise _ReplayUnavailableError("Cannot rewind upload source for retry") from exc
            self.eof = self.failed = False

        # Only this attempt touches the cache. The pending read cannot append or
        # discard it while we replay the prefix, so no lock or offsets are needed.
        position = 0
        while position < self.cache_size:
            cache = self.cache

            def read_cached() -> bytes:
                cache.seek(position)
                return cache.read(min(_CHUNK_SIZE, self.cache_size - position))

            try:
                chunk = await _file_io(read_cached, threaded=cache._rolled)
            except OSError as exc:
                self._buffering = False
                await self._drop_cache()
                raise _ReplayUnavailableError("Cannot read upload replay buffer") from exc
            position += len(chunk)
            yield chunk

        while self._pending is not None or not self.eof:
            if self._pending is None:
                self.started = True
                self._pending = asyncio.create_task(self._read_source())
            # A cancelled attempt leaves the result available to its successor.
            chunk = await asyncio.shield(self._pending)
            if chunk and self._buffering:
                await self._retain(chunk)
            self._pending = None
            if chunk:
                yield chunk

    async def aclose(self) -> None:
        if self.closed:
            return
        self.closed = True
        await self.finish_attempt()
        if self._pending is not None:
            self._pending.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._pending
            self._pending = None
        await self._drop_cache()

    async def finish_attempt(self) -> None:
        """Join the HTTP writer without cancelling the shared source read."""
        writer, self._writer = self._writer, None
        await _cancel_writer(writer)


def _stream_state(payload: RunnerPayload | _StreamState) -> _StreamState | None:
    if isinstance(payload, _StreamState):
        return payload
    if isinstance(payload, StreamPayload):
        return _StreamState(payload)
    if isinstance(payload, AsyncIterable) or callable(getattr(payload, "read", None)):
        return _StreamState(StreamPayload(source=payload))
    if payload is not None and not isinstance(payload, (dict, bytes, bytearray, memoryview, str)):
        raise TypeError(f"Unsupported runner payload: {type(payload).__name__}")
    return None


class _UploadBody(Payload):
    """An attempt borrows the source and cache; aiohttp must close neither."""

    _autoclose = True

    def __init__(self, state: _StreamState) -> None:
        super().__init__(state, content_type="application/octet-stream")
        self.state = state

    def decode(self, encoding: str = "utf-8", errors: str = "strict") -> str:
        raise TypeError("Streaming request bodies cannot be decoded")

    async def write(self, writer: Any) -> None:
        self.state._writer = asyncio.current_task()
        async for chunk in self.state.chunks():
            await writer.write(chunk)
