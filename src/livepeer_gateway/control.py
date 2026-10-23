from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from .trickle_publisher import (
    TricklePublisher,
    TricklePublisherTerminalError,
    TrickleSegmentWriteError,
)

_LOG = logging.getLogger(__name__)


class Control:
    def __init__(
        self,
        control_url: str,
        mime_type: str = "application/json",
    ) -> None:
        self.control_url = control_url
        self._publisher = TricklePublisher(control_url, mime_type)
        self._keepalive_interval_s = 10.0
        self._keepalive_task: asyncio.Task[None] | None = None

    async def write(self, msg: dict[str, Any]) -> None:
        """
        Publish an unstructured JSON message onto the trickle channel.

        One `write()` call sends one message per trickle segment.

        Raises:
            TrickleSegmentWriteError: current segment failed but publisher may continue.
            TricklePublisherTerminalError: publisher entered terminal failure state.
        """
        if not isinstance(msg, dict):
            raise TypeError(f"write expects dict, got {type(msg).__name__}")

        payload = json.dumps(msg).encode("utf-8")
        async with await self._publisher.next() as segment:
            await segment.write(payload)

    async def close(self) -> None:
        """
        Close the control-channel publisher (best-effort).
        """
        task = self._keepalive_task
        self._keepalive_task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                _LOG.exception("Control keepalive task failed during shutdown")
        await self._publisher.close()

    def start_keepalive(self) -> asyncio.Task[None] | None:
        """
        Start periodic keepalive messages.

        Returns the running task, or ``None`` if no task is started.
        """
        if self._keepalive_task is not None and not self._keepalive_task.done():
            return self._keepalive_task
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            _LOG.warning(
                "No running event loop; control keepalive not started. "
                "Call control.start_keepalive() from async code to enable."
            )
            return None
        self._keepalive_task = loop.create_task(self._keepalive_loop())
        return self._keepalive_task

    async def _keepalive_loop(self) -> None:
        while True:
            await asyncio.sleep(self._keepalive_interval_s)
            try:
                await self.write({"keep": "alive"})
            except TrickleSegmentWriteError:
                _LOG.warning("Control keepalive message failed; retrying on next interval", exc_info=True)
            except TricklePublisherTerminalError:
                _LOG.error("Control keepalive stopped due to terminal publisher failure", exc_info=True)
                return
