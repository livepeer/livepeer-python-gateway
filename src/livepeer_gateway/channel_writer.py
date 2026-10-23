from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Protocol

from .trickle_publisher import SegmentWriter, TricklePublisher


_LOG = logging.getLogger(__name__)


class ChannelWriter(Protocol):
    async def write(self, msg: dict[str, Any]) -> None: ...

    async def close(self) -> None: ...


class JSONLWriter:
    def __init__(
        self,
        url: str,
        mime_type: str = "application/jsonl",
        *,
        segment_interval: float = 10.0,
    ) -> None:
        if segment_interval <= 0:
            raise ValueError("segment_interval must be > 0")

        self.url = url
        self._publisher = TricklePublisher(url, mime_type)
        self._segment_interval = segment_interval
        self._writer: SegmentWriter | None = None
        self._lock = asyncio.Lock()
        self._rotation_task: asyncio.Task[None] | None = None
        self.start_rotation()

    def start_rotation(self) -> asyncio.Task[None] | None:
        """
        Start periodic segment rotation if running in an event loop.

        If no loop is running, log a warning and return None; callers can
        invoke this method later from async code.
        """
        if self._rotation_task is not None and not self._rotation_task.done():
            return self._rotation_task
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            _LOG.warning(
                "No running event loop; JSONL writer rotation not started. "
                "Call writer.start_rotation() from async code to enable."
            )
            return None
        self._rotation_task = loop.create_task(self._rotation_loop())
        return self._rotation_task

    async def _rotation_loop(self) -> None:
        while True:
            await asyncio.sleep(self._segment_interval)
            async with self._lock:
                if self._writer is not None:
                    await self._writer.close()
                    self._writer = None

    async def write(self, msg: dict[str, Any]) -> None:
        """
        Publish an unstructured JSON message as JSONL in a time-windowed segment.
        """
        if not isinstance(msg, dict):
            raise TypeError(f"write expects dict, got {type(msg).__name__}")

        payload = json.dumps(msg).encode("utf-8") + b"\n"
        async with self._lock:
            if self._writer is None:
                self._writer = await self._publisher.next()
            try:
                await self._writer.write(payload)
            except Exception:
                # Error writing; reset writer and recreate it on the next write.
                # NB: this could be a `TricklePublisherTerminalError` which
                # the application might want to catch and handle specifically
                await self._writer.close()
                self._writer = None
                raise

    async def close(self) -> None:
        """
        Close the time-windowed JSONL publisher (best-effort).
        """
        task = self._rotation_task
        self._rotation_task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                _LOG.exception("JSONLWriter rotation task failed during shutdown")

        async with self._lock:
            if self._writer is not None:
                await self._writer.close()
                self._writer = None

        await self._publisher.close()
