from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import time
from typing import Optional

import aiohttp

from .segment_reader import SegmentReader


_LOG = logging.getLogger(__name__)

@dataclass(frozen=True)
class TrickleSubscriberStats:
    elapsed_s: float
    get_attempts: int
    get_retries: int
    get_404_eos: int
    get_470_reset: int
    get_failures: int
    segments_delivered: int
    seq_gap_events: int
    wait_ms_total: int
    latest_seq: int

    def __str__(self) -> str:
        return (
            "TrickleSubscriberStats("
            f"elapsed_s={self.elapsed_s:.1f}, "
            f"get_attempts={self.get_attempts}, "
            f"get_retries={self.get_retries}, "
            f"get_failures={self.get_failures}, "
            f"get_404_eos={self.get_404_eos}, "
            f"get_470_reset={self.get_470_reset}, "
            f"segments_delivered={self.segments_delivered}, "
            f"seq_gap_events={self.seq_gap_events}, "
            f"wait_ms_total={self.wait_ms_total}, "
            f"latest_seq={self.latest_seq}"
            ")"
        )


class TrickleSubscriber:
    """
    Trickle subscriber that streams bytes from a sequence of HTTP GET endpoints:
      - Read segment: GET {base_url}/{seq}

    Note: the runtime (lock/session) is created lazily to allow construction
    in sync code without opening sockets.

    max_bytes (if set) limits the total bytes read per segment, not the entire subscription.
    """

    def __init__(
        self,
        url: str,
        *,
        start_seq: int = -2,
        max_retries: int = 5,
        connection_close: bool = False,
        max_bytes: Optional[int] = None,
    ):
        if max_bytes is not None and max_bytes <= 0:
            raise ValueError("max_bytes must be > 0")
        self.base_url = url.rstrip("/")
        self._seq = start_seq
        self._max_retries = max_retries
        self._connection_close = connection_close
        self._max_bytes = max_bytes

        self._pending_get: Optional[aiohttp.ClientResponse] = None
        self._lock: Optional[asyncio.Lock] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._errored = False
        self._closing = False
        self._closed = False
        self._prefetch_task: Optional[asyncio.Task[None]] = None
        self._started_at = time.time()
        self._stats: dict[str, int] = {
            "get_attempts": 0,
            "get_retries": 0,
            "get_404_eos": 0,
            "get_470_reset": 0,
            "get_failures": 0,
            "segments_delivered": 0,
            "seq_gap_events": 0,
            "wait_ms_total": 0,
            "latest_seq": start_seq,
        }

    async def __aenter__(self) -> "TrickleSubscriber":
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.close()

    async def _ensure_runtime(self) -> None:
        # Prevent late background tasks from recreating a session after close.
        if self._closing or self._closed:
            raise RuntimeError("TrickleSubscriber is closing")
        if self._lock is None:
            self._lock = asyncio.Lock()
        if self._session is None:
            self._session = aiohttp.ClientSession()

    def _segment_url(self, seq: int) -> str:
        return f"{self.base_url}/{seq}"

    @staticmethod
    def _latest_seq(headers: "aiohttp.typedefs.LooseHeaders", current_seq: int) -> int:
        latest_header = headers.get("Lp-Trickle-Latest")
        if latest_header is None:
            return current_seq
        try:
            return int(latest_header)
        except ValueError:
            return current_seq

    async def _preconnect(self) -> Optional[aiohttp.ClientResponse]:
        """
        Preconnect to the server by making a GET request to fetch the next segment.

        For non-200 responses, retries up to max_retries unless a 404 is encountered.
        """
        if self._closing or self._closed or self._errored:
            return None
        try:
            await self._ensure_runtime()
        except RuntimeError:
            if self._closing or self._closed:
                return None
            raise
        assert self._session is not None

        seq = self._seq
        url = self._segment_url(seq)
        headers = {"Connection": "close"} if self._connection_close else None

        for attempt in range(0, self._max_retries):
            if self._closing or self._closed or self._errored:
                return None
            started = time.time()
            self._stats["get_attempts"] += 1
            _LOG.debug("Trickle sub preconnect attempt=%s url=%s", attempt, url)
            try:
                resp = await self._session.get(url, headers=headers)
                self._stats["wait_ms_total"] += int((time.time() - started) * 1000)

                if resp.status == 200:
                    # Return the response for later processing
                    return resp

                if resp.status == 404:
                    _LOG.debug("Trickle sub got 404, terminating %s", url)
                    self._stats["get_404_eos"] += 1
                    resp.release()
                    self._errored = True
                    return None

                if resp.status == 470:
                    # Channel exists but no data at this index. If the reported
                    # leading edge is still behind the requested index, we are
                    # polling ahead of the live edge and should retry the same
                    # segment instead of rewinding and replaying the latest one.
                    self._stats["get_470_reset"] += 1
                    latest_seq = self._latest_seq(resp.headers, seq)
                    self._stats["latest_seq"] = latest_seq
                    if latest_seq < seq:
                        _LOG.debug(
                            "Trickle sub ahead of live edge, retrying current index %s "
                            "(latest=%s)",
                            url,
                            latest_seq,
                        )
                        resp.release()
                        await asyncio.sleep(0.25)
                        continue
                    seq = latest_seq
                    self._seq = seq
                    url = self._segment_url(seq)
                    _LOG.debug("Trickle sub resetting index to leading edge %s", url)
                    resp.release()
                    continue

                body = await resp.text()
                resp.release()
                self._stats["get_failures"] += 1
                _LOG.error("Trickle sub failed GET %s status=%s msg=%s", url, resp.status, body)

            except Exception:
                self._stats["wait_ms_total"] += int((time.time() - started) * 1000)
                self._stats["get_failures"] += 1
                _LOG.exception("Trickle sub failed to complete GET %s", url)

            if attempt < self._max_retries - 1:
                self._stats["get_retries"] += 1
                await asyncio.sleep(0.5)

        _LOG.error("Trickle sub hit max retries, exiting %s", url)
        self._errored = True
        return None

    async def next(self) -> Optional["SegmentReader"]:
        """Retrieve data from the current segment and set up the next segment concurrently."""
        if self._closing or self._closed:
            return None
        try:
            await self._ensure_runtime()
        except RuntimeError:
            if self._closing or self._closed:
                return None
            raise
        assert self._lock is not None

        async with self._lock:
            # We intentionally serialize preconnect/next under one lock to avoid
            # overlapping fetches that could race and stomp segment ordering.
            if self._errored:
                _LOG.debug("Trickle subscription closed or errored for %s", self.base_url)
                return None

            # If we don't have a pending GET request, preconnect
            if self._pending_get is None:
                _LOG.debug("Trickle sub no pending connection, preconnecting...")
                self._pending_get = await self._preconnect()

            # Extract the current connection to use for reading
            resp = self._pending_get
            self._pending_get = None

            # Preconnect has failed, notify caller
            if resp is None:
                return None

            # Extract and set the next index from the response headers
            segment = SegmentReader(resp, max_bytes=self._max_bytes)

            if segment.eos():
                await segment.close()
                return None

            seq = segment.seq()
            expected_seq = self._seq
            if seq >= 0:
                if expected_seq >= 0 and seq != expected_seq:
                    self._stats["seq_gap_events"] += 1
                self._seq = seq + 1
            current_seq = seq if seq >= 0 else expected_seq
            self._stats["latest_seq"] = self._latest_seq(segment.headers(), current_seq)
            self._stats["segments_delivered"] += 1

            # Set up the next connection in the background
            if self._prefetch_task is None or self._prefetch_task.done():
                prefetch_task = asyncio.create_task(self._preconnect_next_segment())
                self._prefetch_task = prefetch_task

                def _clear_prefetch(task: asyncio.Task[None]) -> None:
                    if self._prefetch_task is task:
                        self._prefetch_task = None

                prefetch_task.add_done_callback(_clear_prefetch)

        return segment

    async def _preconnect_next_segment(self) -> None:
        """Preconnect to the next segment in the background."""
        if self._closing or self._closed or self._errored:
            return
        try:
            await self._ensure_runtime()
        except RuntimeError:
            if self._closing or self._closed:
                return
            raise
        assert self._lock is not None

        async with self._lock:
            if self._closing or self._closed or self._errored:
                return
            if self._pending_get is not None:
                return
            next_conn = await self._preconnect()
            # Discard late-arriving responses instead of leaking them.
            if next_conn and (self._closing or self._closed or self._errored):
                next_conn.close()
                return
            if next_conn:
                self._pending_get = next_conn

    async def close(self) -> None:
        """Close the session when done."""
        if self._closed:
            return

        self._closing = True
        self._errored = True
        # Cancel and await the background prefetch before taking the lock.
        prefetch_task = self._prefetch_task
        self._prefetch_task = None
        if prefetch_task is not None and not prefetch_task.done():
            prefetch_task.cancel()
            await asyncio.gather(prefetch_task, return_exceptions=True)

        _LOG.debug("Trickle sub closing %s", self.base_url)
        lock = self._lock
        if lock is not None:
            async with lock:
                if self._pending_get:
                    self._pending_get.close()
                    self._pending_get = None
                session = self._session
                self._session = None
                if session:
                    try:
                        await session.close()
                    except Exception:
                        _LOG.error("Error closing trickle subscriber", exc_info=True)
        else:
            if self._pending_get:
                self._pending_get.close()
                self._pending_get = None
            session = self._session
            self._session = None
            if session:
                try:
                    await session.close()
                except Exception:
                    _LOG.error("Error closing trickle subscriber", exc_info=True)
        self._closed = True

    def get_stats(self) -> TrickleSubscriberStats:
        return TrickleSubscriberStats(
            elapsed_s=max(0.0, time.time() - self._started_at),
            get_attempts=self._stats["get_attempts"],
            get_retries=self._stats["get_retries"],
            get_404_eos=self._stats["get_404_eos"],
            get_470_reset=self._stats["get_470_reset"],
            get_failures=self._stats["get_failures"],
            segments_delivered=self._stats["segments_delivered"],
            seq_gap_events=self._stats["seq_gap_events"],
            wait_ms_total=self._stats["wait_ms_total"],
            latest_seq=self._stats["latest_seq"],
        )


