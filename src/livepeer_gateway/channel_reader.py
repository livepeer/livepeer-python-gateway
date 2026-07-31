from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from .errors import LivepeerGatewayError
from .segment_reader import SegmentReader
from .trickle_subscriber import TrickleSubscriber

_LOG = logging.getLogger(__name__)

ChannelEventCallback = Callable[[dict[str, Any]], None | Awaitable[None]]
"""
Callback invoked for each decoded channel event.

Callbacks may be synchronous or asynchronous. Async callback results are awaited
before the next event is delivered.
"""


async def _maybe_await(value: object) -> None:
    if inspect.isawaitable(value):
        return await value


class _ChannelReaderCallback:
    def _init_callback(
        self,
        events_url: str,
        *,
        start_seq: int = -2,
        max_retries: int = 5,
        max_event_bytes: int = 1_048_576,
        on_event: ChannelEventCallback | None = None,
    ) -> None:
        self.events_url = events_url
        self.start_seq = start_seq
        self.max_retries = max_retries
        self.max_event_bytes = max_event_bytes
        self.on_event = on_event
        self._event_callback_task: asyncio.Task[None] | None = None
        self._callback_error: BaseException | None = None
        if self.on_event is not None:
            self.start_callback()

    def __call__(
        self,
        *,
        start_seq: int = -2,
        max_retries: int = 5,
        max_event_bytes: int = 1_048_576,
    ) -> AsyncIterator[dict[str, Any]]:
        raise NotImplementedError

    def start_callback(
        self,
    ) -> asyncio.Task[None] | None:
        """
        Start the configured event callback consumer.

        This is idempotent. If called without a running event loop, no task is
        started and callers may retry later from async code.

        Callback consumption uses the start_seq, max_retries, and
        max_event_bytes values supplied to the reader constructor. Those
        constructor values do not affect explicit iterator calls via __call__.
        """
        if self.on_event is None:
            return None
        if self._event_callback_task is not None and not self._event_callback_task.done():
            return self._event_callback_task
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            _LOG.warning(
                "No running event loop; %s callback not started. "
                "Call reader.start_callback() from async code or use async with the reader.",
                type(self).__name__,
            )
            return None

        task = loop.create_task(
            self._run_event_callback_loop(
                self.on_event,
                start_seq=self.start_seq,
                max_retries=self.max_retries,
                max_event_bytes=self.max_event_bytes,
            ),
            name=f"{type(self).__name__}.on_event",
        )
        self._callback_error = None
        task.add_done_callback(self._record_callback_task_result)
        self._event_callback_task = task
        return task

    def callback_task(self) -> asyncio.Task[None] | None:
        """
        Return the active or completed callback task, if one has been created.
        """
        return self._event_callback_task

    async def wait_callback(self, timeout: float | None = None) -> object:
        """
        Wait for the configured event callback consumer to finish.

        Raises the first callback error, matching close().
        """
        task = self.callback_task()
        if task is None:
            return None
        try:
            result = await asyncio.wait_for(task, timeout=timeout)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            self._record_callback_error(exc)
            raise
        return result

    def _record_callback_error(self, error: BaseException) -> None:
        if isinstance(error, asyncio.CancelledError):
            return
        if self._callback_error is None:
            self._callback_error = error

    async def _run_event_callback_loop(
        self,
        callback: ChannelEventCallback,
        *,
        start_seq: int,
        max_retries: int,
        max_event_bytes: int,
    ) -> None:
        async for event in self(
            start_seq=start_seq,
            max_retries=max_retries,
            max_event_bytes=max_event_bytes,
        ):
            await _maybe_await(callback(event))

    def _record_callback_task_result(self, task: asyncio.Task[None]) -> None:
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        if exc is None:
            return
        self._record_callback_error(exc)
        _LOG.error(
            "%s callback task failed",
            type(self).__name__,
            exc_info=(type(exc), exc, exc.__traceback__),
        )

    async def close(
        self,
        *,
        wait_callback: bool = True,
        timeout: float | None = 10.0,
    ) -> None:
        """
        Stop callback consumption and surface callback errors.

        If wait_callback is true, close waits up to timeout for the callback
        task to finish naturally before cancelling it. Any callback exception is
        raised from close(), matching wait_callback().
        """
        task = self.callback_task()
        if task is not None:
            if wait_callback and (timeout is None or timeout > 0):
                try:
                    await self.wait_callback(timeout=timeout)
                except TimeoutError:
                    _LOG.debug(
                        "%s callback did not finish before shutdown timeout; cancelling",
                        type(self).__name__,
                    )
            if not task.done():
                task.cancel()
            (result,) = await asyncio.gather(task, return_exceptions=True)
            if isinstance(result, BaseException):
                self._record_callback_error(result)
        if self._callback_error is not None:
            raise self._callback_error

    async def __aenter__(self):
        self.start_callback()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.close()


class ChannelReader(_ChannelReaderCallback):
    """
    Read a trickle channel containing one JSON object per segment.

    Iterator usage is lazy and configured per call:

        async for event in ChannelReader(url)(start_seq=-2):
            ...

    Callback usage is configured on the instance:

        reader = ChannelReader(url, start_seq=-2, on_event=handle_event)
        reader.start_callback()

    The constructor's start_seq, max_retries, and max_event_bytes values apply
    only to callback consumption. Explicit calls to reader(...) keep their own
    arguments and defaults.
    """

    def __init__(
        self,
        events_url: str,
        *,
        start_seq: int = -2,
        max_retries: int = 5,
        max_event_bytes: int = 1_048_576,
        on_event: ChannelEventCallback | None = None,
    ) -> None:
        """
        Create a JSON channel reader.

        Args:
            events_url: Trickle subscribe URL.
            start_seq: Initial server sequence for callback consumption only.
            max_retries: Retry count for callback consumption only.
            max_event_bytes: Per-segment byte limit for callback consumption only.
            on_event: Optional callback invoked for each decoded JSON object.

        If on_event is provided while an event loop is running, callback
        consumption starts immediately. If no loop is running, call
        start_callback() later from async code or use async with the reader.
        """
        self._init_callback(
            events_url,
            start_seq=start_seq,
            max_retries=max_retries,
            max_event_bytes=max_event_bytes,
            on_event=on_event,
        )

    def __call__(
        self,
        *,
        start_seq: int = -2,
        max_retries: int = 5,
        max_event_bytes: int = 1_048_576,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Subscribe to the trickle events channel.

        Each yielded item is a decoded JSON object (dict). The underlying network
        subscription starts lazily on first iteration.

        These arguments configure this iterator only. They do not change the
        instance settings used by callback consumption.

        max_event_bytes applies per segment (per JSON message), not across
        the entire stream.
        """
        url = self.events_url

        async def _read_all(segment: "SegmentReader", *, chunk_size: int = 33 * 1024) -> bytes:
            parts = []
            try:
                reader = segment.make_reader()
                while True:
                    chunk = await reader.read(chunk_size=chunk_size)
                    if not chunk:
                        break
                    parts.append(chunk)
            finally:
                await segment.close()
            return b"".join(parts)

        async def _iter() -> AsyncIterator[dict[str, Any]]:
            if max_event_bytes < 1:
                raise ValueError("max_event_bytes must be >= 1")

            try:
                async with TrickleSubscriber(
                    url,
                    start_seq=start_seq,
                    max_retries=max_retries,
                    max_bytes=max_event_bytes,
                ) as subscriber:
                    while (segment := await subscriber.next()) is not None:
                        payload = await _read_all(segment)
                        if not payload.strip():
                            continue

                        try:
                            data = json.loads(payload.decode("utf-8"))
                        except Exception as e:
                            snippet = payload[:256].decode("utf-8", errors="replace")
                            raise LivepeerGatewayError(
                                f"Trickle event JSON decode failed: {e} (payload={snippet!r})"
                            ) from e

                        if not isinstance(data, dict):
                            raise LivepeerGatewayError(
                                f"Trickle event must be JSON, got {type(data).__name__}"
                            )

                        yield data
            except LivepeerGatewayError:
                raise
            except Exception as e:
                raise LivepeerGatewayError(
                    f"Trickle events subscription error: {e.__class__.__name__}: {e}"
                ) from e

        return _iter()


class JSONLReader(_ChannelReaderCallback):
    """
    Read a trickle channel containing newline-delimited JSON objects.

    Iterator usage is lazy and configured per call:

        async for event in JSONLReader(url)(start_seq=-2):
            ...

    Callback usage is configured on the instance:

        reader = JSONLReader(url, start_seq=-2, on_event=handle_event)
        reader.start_callback()

    The constructor's start_seq, max_retries, and max_event_bytes values apply
    only to callback consumption. Explicit calls to reader(...) keep their own
    arguments and defaults.
    """

    def __init__(
        self,
        events_url: str,
        *,
        start_seq: int = -2,
        max_retries: int = 5,
        max_event_bytes: int = 1_048_576,
        on_event: ChannelEventCallback | None = None,
    ) -> None:
        """
        Create a JSONL channel reader.

        Args:
            events_url: Trickle subscribe URL.
            start_seq: Initial server sequence for callback consumption only.
            max_retries: Retry count for callback consumption only.
            max_event_bytes: Per-segment byte limit for callback consumption only.
            on_event: Optional callback invoked for each decoded JSON object.

        If on_event is provided while an event loop is running, callback
        consumption starts immediately. If no loop is running, call
        start_callback() later from async code or use async with the reader.
        """
        self._init_callback(
            events_url,
            start_seq=start_seq,
            max_retries=max_retries,
            max_event_bytes=max_event_bytes,
            on_event=on_event,
        )

    def __call__(
        self,
        *,
        start_seq: int = -2,
        max_retries: int = 5,
        max_event_bytes: int = 1_048_576,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Subscribe to a trickle channel containing newline-delimited JSON (JSONL).

        Events are yielded incrementally as newline-terminated lines arrive, without
        buffering the entire segment in memory first. max_event_bytes applies per
        segment, not across the entire stream.

        These arguments configure this iterator only. They do not change the
        instance settings used by callback consumption.
        """
        url = self.events_url

        def _decode_line(line: bytearray) -> dict[str, Any]:
            try:
                data = json.loads(line)
            except Exception as e:
                snippet = bytes(line[:256]).decode("utf-8", errors="replace")
                raise LivepeerGatewayError(
                    f"Trickle event JSONL decode failed: {e} (line={snippet!r})"
                ) from e

            if not isinstance(data, dict):
                raise LivepeerGatewayError(
                    f"Trickle event must be JSON object, got {type(data).__name__}"
                )
            return data

        async def _iter() -> AsyncIterator[dict[str, Any]]:
            if max_event_bytes < 1:
                raise ValueError("max_event_bytes must be >= 1")

            try:
                async with TrickleSubscriber(
                    url,
                    start_seq=start_seq,
                    max_retries=max_retries,
                    max_bytes=max_event_bytes,
                ) as subscriber:
                    while (segment := await subscriber.next()) is not None:
                        reader = segment.make_reader()
                        buf = bytearray()
                        start = 0
                        try:
                            while True:
                                chunk = await reader.read(chunk_size=33 * 1024)
                                if not chunk:
                                    break

                                buf.extend(chunk)

                                while True:
                                    nl = buf.find(b"\n", start)
                                    if nl < 0:
                                        break

                                    line = buf[start:nl]
                                    start = nl + 1
                                    if not line:
                                        continue

                                    yield _decode_line(line)

                                if start == len(buf):
                                    buf.clear()
                                    start = 0
                                elif start > 64 * 1024 and start > len(buf) // 2:
                                    del buf[:start]
                                    start = 0

                            tail = bytes(buf[start:]).strip()
                            if tail:
                                data = _decode_line(bytearray(tail))
                                yield data
                        finally:
                            await segment.close()
            except LivepeerGatewayError:
                raise
            except Exception as e:
                raise LivepeerGatewayError(
                    f"Trickle JSONL subscription error: {e.__class__.__name__}: {e}"
                ) from e

        return _iter()
