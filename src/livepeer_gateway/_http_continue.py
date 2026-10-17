"""Bound aiohttp's native continue handshake without replacing response parsing."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import aiohttp


class _ContinueResponse(aiohttp.ClientResponse):
    def __init__(self, *args: Any, continue_timeout: float, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._continue_timeout = continue_timeout

    async def start(self, connection: Any) -> aiohttp.ClientResponse:
        continued = self._continue
        if continued is None:
            return await super().start(connection)

        # aiohttp releases this waiter on any informational response except 101.
        # Only the wait is bounded; uploads/final headers can take longer once
        # permission arrives. Final headers also end the wait through start().
        waiting = True
        async with asyncio.timeout(self._continue_timeout) as handshake:
            def allow_upload(future: asyncio.Future[bool]) -> None:
                if waiting and not future.cancelled() and not handshake.expired():
                    handshake.reschedule(None)

            continued.add_done_callback(allow_upload)
            try:
                return await super().start(connection)
            finally:
                waiting = False
                continued.remove_done_callback(allow_upload)


async def _close_response(response: aiohttp.ClientResponse) -> None:
    """Stop and join an attempt's writer before another attempt borrows its state."""
    writer = response._writer
    response.close()
    await _cancel_writer(writer)


async def _cancel_writer(writer: asyncio.Task[None] | None) -> None:
    if writer is not None:
        writer.cancel()
        # aiohttp 3.9 tries write_eof() on cancellation. Its failure after the
        # peer closes must not mask a final response that was already received.
        with contextlib.suppress(
            asyncio.CancelledError, ConnectionError, aiohttp.ClientError
        ):
            await writer
