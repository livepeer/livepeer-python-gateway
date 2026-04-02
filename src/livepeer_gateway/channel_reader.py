from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import aiohttp

from .errors import LivepeerGatewayError
from .trickle_subscriber import TrickleSubscriber

_LOG = logging.getLogger(__name__)


class ChannelReader:
    def __init__(self, events_url: str) -> None:
        self.events_url = events_url

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
                        if not payload:
                            raise LivepeerGatewayError("Trickle event segment was empty")

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
            except aiohttp.ClientPayloadError as e:
                # Orchestrator truncated the transfer mid-stream (e.g. TransferEncodingError 400)
                # or went unreachable. Treat as a clean network disconnect — stop iterating
                # rather than propagating as an application error.
                _LOG.warning("Trickle events channel disconnected (network): %s: %s", e.__class__.__name__, e)
                return
            except Exception as e:
                raise LivepeerGatewayError(
                    f"Trickle events subscription error: {e.__class__.__name__}: {e}"
                ) from e

        return _iter()


class JSONLReader:
    def __init__(self, events_url: str) -> None:
        self.events_url = events_url

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
            except aiohttp.ClientPayloadError as e:
                # Orchestrator truncated the transfer mid-stream (e.g. TransferEncodingError 400)
                # or went unreachable. Treat as a clean network disconnect — stop iterating
                # rather than propagating as an application error.
                _LOG.warning("Trickle JSONL channel disconnected (network): %s: %s", e.__class__.__name__, e)
                return
            except Exception as e:
                raise LivepeerGatewayError(
                    f"Trickle JSONL subscription error: {e.__class__.__name__}: {e}"
                ) from e

        return _iter()

