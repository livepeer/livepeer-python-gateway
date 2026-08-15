"""
Helpers for consuming trickle media outputs as segments, bytes, or frames.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import inspect
import logging
import time
from enum import Enum
from contextlib import suppress
from collections.abc import AsyncIterator, Awaitable, Callable, Collection
from typing import Optional

import aiohttp

from .errors import LivepeerGatewayError
from .media_decode import (
    AudioDecodedMediaFrame,
    DecodedMediaFrame,
    DemuxedMediaPacket,
    DecoderQueueStats,
    MpegTsDecoder,
    MpegTsPacketDemuxer,
    VideoDecodedMediaFrame,
    decoder_error,
    is_decoder_end,
)

from .segment_reader import SegmentReader
from .trickle_subscriber import TrickleSubscriber, TrickleSubscriberStats

_LOG = logging.getLogger(__name__)
_DEFAULT_ACCEPTED_CONTENT_TYPES = frozenset({"video/mp2t", "audio/mp2t"})

MediaFrameCallback = Callable[
    [AudioDecodedMediaFrame | VideoDecodedMediaFrame],
    None | Awaitable[None],
]
MediaPacketCallback = Callable[[DemuxedMediaPacket], None | Awaitable[None]]
MediaBytesCallback = Callable[[bytes], None | Awaitable[None]]


class LagPolicy(Enum):
    """
    Policy for handling consumers that fall behind the segment window.
    """
    FAIL = "fail"
    LATEST = "latest"
    EARLIEST = "earliest"

@dataclass(frozen=True)
class MediaOutputStats:
    elapsed_s: float
    segments_consumed: int
    bytes_read: int
    chunks_read: int
    content_type_errors: int
    segment_read_errors: int
    segment_max_bytes_exceeded: int
    consumer_lag_skip_latest: int
    consumer_lag_retry_earliest: int
    consumer_lag_fail: int
    video_packets_demuxed: int
    audio_packets_demuxed: int
    other_packets_demuxed: int
    video_frames_decoded: int
    audio_frames_decoded: int
    packet_errors: int
    decode_errors: int
    # decoder: decode pipeline queue metrics, including the distinction between
    # queued_bytes in the cross-thread input queue and buffered_bytes already
    # staged in the internal bytearray used to satisfy read() calls.
    decoder: Optional[DecoderQueueStats]
    subscriber: Optional[TrickleSubscriberStats]

    def __str__(self) -> str:
        return (
            "MediaOutputStats("
            f"elapsed_s={self.elapsed_s:.1f}, "
            f"segments_consumed={self.segments_consumed}, "
            f"bytes_read={self.bytes_read}, "
            f"chunks_read={self.chunks_read}, "
            f"content_type_errors={self.content_type_errors}, "
            f"segment_read_errors={self.segment_read_errors}, "
            f"segment_max_bytes_exceeded={self.segment_max_bytes_exceeded}, "
            f"consumer_lag_skip_latest={self.consumer_lag_skip_latest}, "
            f"consumer_lag_retry_earliest={self.consumer_lag_retry_earliest}, "
            f"consumer_lag_fail={self.consumer_lag_fail}, "
            f"video_packets_demuxed={self.video_packets_demuxed}, "
            f"audio_packets_demuxed={self.audio_packets_demuxed}, "
            f"other_packets_demuxed={self.other_packets_demuxed}, "
            f"video_frames_decoded={self.video_frames_decoded}, "
            f"audio_frames_decoded={self.audio_frames_decoded}, "
            f"packet_errors={self.packet_errors}, "
            f"decode_errors={self.decode_errors}"
            f"{', decoder=' + str(self.decoder) if self.decoder is not None else ''}"
            f"{', subscriber=' + str(self.subscriber) if self.subscriber is not None else ''}"
            ")"
        )


class MediaOutput:
    """
    Access a trickle media output

    Exposes:
      - per-segment iteration (SegmentReader objects)
      - continuous byte stream (bytes chunks)
      - demuxed MPEG-TS packets
      - individual audio and video frames

    Segments are sourced from a single shared subscriber so that multiple
    iterators can consume the same output concurrently without duplicate
    network requests.

    Attributes:
        subscribe_url: Trickle subscribe URL for this output.
        start_seq: Initial server sequence when subscribing.
        max_retries: Max retries for segment fetches.
        max_segment_bytes: Safety bound for a single segment size.
        connection_close: Whether to close connections after each segment.
        chunk_size: Byte chunk size yielded by bytes()/packets()/frames().
        max_segments: Max number of segments retained in memory.
        on_lag: Behavior when a consumer falls behind the segment window.
            - LagPolicy.FAIL: raise LivepeerGatewayError.
            - LagPolicy.LATEST: skip to the newest available segment.
            - LagPolicy.EARLIEST: retry from the oldest available segment.
        _sub: Shared trickle subscriber.
        _segments: In-memory window of SegmentReader objects.
        _lock: Coroutine-level lock for segment fetching/eviction.
        _eos: End-of-stream indicator.
        _next_local_seq: Local sequence counter for fetched segments.
        _base_seq: Local sequence of _segments[0].
    """

    def __init__(
        self,
        subscribe_url: str,
        *,
        start_seq: int = -2,
        max_retries: int = 5,
        max_segment_bytes: Optional[int] = None,
        connection_close: bool = False,
        chunk_size: int = 64 * 1024,
        max_segments: int = 5,
        on_lag: LagPolicy = LagPolicy.LATEST,
        accepted_content_types: Collection[str] = _DEFAULT_ACCEPTED_CONTENT_TYPES,
        on_bytes: MediaBytesCallback | None = None,
        on_frame: MediaFrameCallback | None = None,
        on_packet: MediaPacketCallback | None = None,
    ) -> None:
        if max_segments < 1:
            raise ValueError("max_segments must be >= 1")
        self.subscribe_url = subscribe_url
        self.start_seq = start_seq
        self.max_retries = max_retries
        self.max_segment_bytes = max_segment_bytes
        self.connection_close = connection_close
        self.chunk_size = chunk_size
        self.max_segments = max_segments
        self.on_lag = on_lag
        self.accepted_content_types = _normalize_accepted_content_types(accepted_content_types)
        self.on_bytes = on_bytes
        self.on_frame = on_frame
        self.on_packet = on_packet

        self._sub: Optional[TrickleSubscriber] = None
        self._segments: list[SegmentReader] = []
        self._lock = asyncio.Lock()
        self._eos = False
        self._next_local_seq = 0
        self._base_seq = 0
        self._started_at = time.time()
        self._processor: Optional[MpegTsDecoder | MpegTsPacketDemuxer] = None
        self._last_decoder_stats: Optional[DecoderQueueStats] = None
        self._bytes_callback_task: asyncio.Task[None] | None = None
        self._frame_callback_task: asyncio.Task[None] | None = None
        self._packet_callback_task: asyncio.Task[None] | None = None
        self._callback_errors: list[BaseException] = []
        self._stats: dict[str, int] = {
            "segments_consumed": 0,
            "bytes_read": 0,
            "chunks_read": 0,
            "content_type_errors": 0,
            "segment_read_errors": 0,
            "segment_max_bytes_exceeded": 0,
            "consumer_lag_skip_latest": 0,
            "consumer_lag_retry_earliest": 0,
            "consumer_lag_fail": 0,
            "video_packets_demuxed": 0,
            "audio_packets_demuxed": 0,
            "other_packets_demuxed": 0,
            "video_frames_decoded": 0,
            "audio_frames_decoded": 0,
            "packet_errors": 0,
            "decode_errors": 0,
        }
        if self.on_bytes is not None or self.on_frame is not None or self.on_packet is not None:
            self.start_callbacks()

    def start_callbacks(self) -> list[asyncio.Task[None]]:
        """
        Start configured frame/packet callback consumers.

        This is idempotent. If called without a running event loop, no tasks are
        started and callers may retry later from async code.
        """
        if self.on_bytes is None and self.on_frame is None and self.on_packet is None:
            return []
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            _LOG.warning(
                "No running event loop; MediaOutput callbacks not started. "
                "Call start_callbacks() from async code or use async with MediaOutput(...)."
            )
            return []

        started: list[asyncio.Task[None]] = []
        if self.on_bytes is not None and self._bytes_callback_task is None:
            task = loop.create_task(
                self._run_bytes_callback_loop(self.on_bytes),
                name="MediaOutput.on_bytes",
            )
            task.add_done_callback(self._record_callback_task_result)
            self._bytes_callback_task = task
            started.append(task)
        if self.on_frame is not None and self._frame_callback_task is None:
            task = loop.create_task(
                self._run_frame_callback_loop(self.on_frame),
                name="MediaOutput.on_frame",
            )
            task.add_done_callback(self._record_callback_task_result)
            self._frame_callback_task = task
            started.append(task)
        if self.on_packet is not None and self._packet_callback_task is None:
            task = loop.create_task(
                self._run_packet_callback_loop(self.on_packet),
                name="MediaOutput.on_packet",
            )
            task.add_done_callback(self._record_callback_task_result)
            self._packet_callback_task = task
            started.append(task)
        return started

    def callback_tasks(self) -> tuple[asyncio.Task[None], ...]:
        tasks = []
        if self._bytes_callback_task is not None:
            tasks.append(self._bytes_callback_task)
        if self._frame_callback_task is not None:
            tasks.append(self._frame_callback_task)
        if self._packet_callback_task is not None:
            tasks.append(self._packet_callback_task)
        return tuple(tasks)

    async def wait_callbacks(self, timeout: float | None = None) -> tuple[object, ...]:
        """
        Wait for configured callback consumers to finish.

        Raises the first callback error, matching close().
        """
        callback_tasks = self.callback_tasks()
        if not callback_tasks:
            return ()
        results = await asyncio.wait_for(
            asyncio.gather(*callback_tasks, return_exceptions=True),
            timeout=timeout,
        )
        self._collect_callback_errors(results)
        if self._callback_errors:
            raise self._callback_errors[0]
        return tuple(results)

    def _collect_callback_errors(self, results: Collection[object]) -> None:
        for result in results:
            if isinstance(result, BaseException) and not isinstance(
                result, asyncio.CancelledError
            ):
                if not any(error is result for error in self._callback_errors):
                    self._callback_errors.append(result)

    async def _run_bytes_callback_loop(self, callback: MediaBytesCallback) -> None:
        async for chunk in self.bytes():
            await _maybe_await(callback(chunk))

    async def _run_frame_callback_loop(self, callback: MediaFrameCallback) -> None:
        async for frame in self.frames():
            await _maybe_await(callback(frame))

    async def _run_packet_callback_loop(self, callback: MediaPacketCallback) -> None:
        async for packet in self.packets():
            await _maybe_await(callback(packet))

    def _record_callback_task_result(self, task: asyncio.Task[None]) -> None:
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        if exc is None:
            return
        self._callback_errors.append(exc)
        _LOG.error(
            "MediaOutput callback task failed",
            exc_info=(type(exc), exc, exc.__traceback__),
        )

    def segments(
        self,
    ) -> AsyncIterator[SegmentReader]:
        """
        Read the trickle media channel and yield SegmentReader objects.

        Segments are shared across iterators.
        """
        async def _iter() -> AsyncIterator[SegmentReader]:
            seq = 0
            segment = await self._next_segment(seq)
            while segment is not None:
                yield segment
                # Use the returned segment's local seq in case we skipped ahead.
                seq = segment._local_seq + 1
                segment = await self._next_segment(seq)

        return _iter()

    def bytes(
        self,
    ) -> AsyncIterator[bytes]:
        """
        Read the trickle media channel and yield a continuous byte stream.
        """

        async def _iter() -> AsyncIterator[bytes]:
            async for chunk in self._iter_bytes():
                yield chunk

        return _iter()

    def packets(
        self,
    ) -> AsyncIterator[DemuxedMediaPacket]:
        """
        Read the trickle media channel, demux MPEG-TS, and yield packets.
        """

        async def _iter() -> AsyncIterator[DemuxedMediaPacket]:
            demuxer = MpegTsPacketDemuxer()
            self._processor = demuxer
            demuxer.start()

            async def _feed() -> None:
                async for chunk in self._iter_bytes():
                    demuxer.feed(chunk)
                demuxer.close()

            producer_task = asyncio.create_task(_feed())
            try:
                while True:
                    item = await asyncio.to_thread(demuxer.get)
                    err = decoder_error(item)
                    if err is not None:
                        self._stats["packet_errors"] += 1
                        raise LivepeerGatewayError(
                            f"Media demux error: {err.__class__.__name__}: {err}"
                        ) from err
                    if is_decoder_end(item):
                        if producer_task.done():
                            exc = producer_task.exception()
                            if exc:
                                raise exc
                        break
                    if isinstance(item, DemuxedMediaPacket):
                        if item.kind == "video":
                            self._stats["video_packets_demuxed"] += 1
                        elif item.kind == "audio":
                            self._stats["audio_packets_demuxed"] += 1
                        else:
                            self._stats["other_packets_demuxed"] += 1
                        yield item
            finally:
                demuxer.stop()
                if not producer_task.done():
                    producer_task.cancel()
                with suppress(asyncio.CancelledError):
                    await producer_task
                await asyncio.to_thread(demuxer.join)
                self._last_decoder_stats = demuxer.get_stats()
                if self._processor is demuxer:
                    self._processor = None

        return _iter()

    def frames(
        self,
    ) -> AsyncIterator[AudioDecodedMediaFrame | VideoDecodedMediaFrame]:
        """
        Read the trickle media channel, decode MPEG-TS, and yield raw frames.
        """

        async def _iter() -> AsyncIterator[AudioDecodedMediaFrame | VideoDecodedMediaFrame]:
            decoder = MpegTsDecoder()
            self._processor = decoder
            decoder.start()

            async def _feed() -> None:
                async for chunk in self._iter_bytes():
                    decoder.feed(chunk)
                decoder.close()

            producer_task = asyncio.create_task(_feed())
            try:
                while True:
                    item = await asyncio.to_thread(decoder.get)
                    err = decoder_error(item)
                    if err is not None:
                        self._stats["decode_errors"] += 1
                        raise LivepeerGatewayError(
                            f"Media decode error: {err.__class__.__name__}: {err}"
                        ) from err
                    if is_decoder_end(item):
                        if producer_task.done():
                            exc = producer_task.exception()
                            if exc:
                                raise exc
                        break
                    if isinstance(item, DecodedMediaFrame):
                        if item.kind == "video":
                            self._stats["video_frames_decoded"] += 1
                        elif item.kind == "audio":
                            self._stats["audio_frames_decoded"] += 1
                        yield item
            finally:
                decoder.stop()
                if not producer_task.done():
                    producer_task.cancel()
                with suppress(asyncio.CancelledError, aiohttp.ClientConnectionError):
                    await producer_task
                await asyncio.to_thread(decoder.join)
                self._last_decoder_stats = decoder.get_stats()
                if self._processor is decoder:
                    self._processor = None

        return _iter()

    async def _iter_bytes(
        self,
    ) -> AsyncIterator[bytes]:
        checked_content_type = False
        seq = 0
        segment = await self._next_segment(seq)
        while segment is not None:
            reader = segment.make_reader()
            self._stats["segments_consumed"] += 1
            while True:
                chunk = await reader.read(chunk_size=self.chunk_size)
                if not chunk:
                    break
                if not checked_content_type:
                    try:
                        _require_content_type(
                            segment.headers().get("Content-Type"),
                            self.accepted_content_types,
                        )
                    except Exception:
                        self._stats["content_type_errors"] += 1
                        raise
                    checked_content_type = True
                self._stats["chunks_read"] += 1
                self._stats["bytes_read"] += len(chunk)
                yield chunk
            segment_stats = segment.get_stats()
            self._stats["segment_read_errors"] += segment_stats.read_errors
            self._stats["segment_max_bytes_exceeded"] += segment_stats.max_bytes_exceeded
            # Use the returned segment's local seq in case we skipped ahead.
            seq = segment._local_seq + 1
            segment = await self._next_segment(seq)

    async def _next_segment(
        self,
        seq: int,
    ) -> Optional[SegmentReader]:
        """
        Return the segment at seq, lazily advancing the subscriber if needed.
        """
        # Safe lock-free read: asyncio only context-switches on awaits, and this
        # block has no awaits. That means _segments/_base_seq cannot change
        # until we return or enter the locked slow path below.
        relative = seq - self._base_seq
        if 0 <= relative < len(self._segments):
            return self._segments[relative]

        async with self._lock:
            relative = seq - self._base_seq
            if relative < 0:
                if self.on_lag is LagPolicy.FAIL:
                    self._stats["consumer_lag_fail"] += 1
                    raise LivepeerGatewayError(
                        "consumer fell behind segment window"
                    )
                if self._segments:
                    if self.on_lag is LagPolicy.EARLIEST:
                        self._stats["consumer_lag_retry_earliest"] += 1
                        _LOG.warning(
                            "MediaOutput consumer fell behind segment window; "
                            "retrying from earliest"
                        )
                        return self._segments[0]
                    self._stats["consumer_lag_skip_latest"] += 1
                    _LOG.warning(
                        "MediaOutput consumer fell behind segment window; "
                        "skipping to latest"
                    )
                    return self._segments[-1]
            elif relative < len(self._segments):
                return self._segments[relative]

            while (seq - self._base_seq) >= len(self._segments):
                if self._eos:
                    return None
                if self._sub is None:
                    self._sub = TrickleSubscriber(
                        self.subscribe_url,
                        start_seq=self.start_seq,
                        max_retries=self.max_retries,
                        max_bytes=self.max_segment_bytes,
                        connection_close=self.connection_close,
                    )
                segment = await self._sub.next()
                if segment is None:
                    self._eos = True
                    return None
                segment._local_seq = self._next_local_seq
                self._next_local_seq += 1
                self._segments.append(segment)

                prev = len(self._segments) - 2
                if prev >= 0:
                    await self._segments[prev].close()

                while len(self._segments) > self.max_segments:
                    self._segments.pop(0)
                    self._base_seq += 1

            relative = seq - self._base_seq
            if 0 <= relative < len(self._segments):
                return self._segments[relative]
            return None

    async def close(self, *, wait_callbacks: bool = True, timeout: float | None = 10.0) -> None:
        callback_tasks = self.callback_tasks()
        if callback_tasks:
            if wait_callbacks and (timeout is None or timeout > 0):
                try:
                    await self.wait_callbacks(timeout=timeout)
                except TimeoutError:
                    _LOG.debug(
                        "Media output callbacks did not finish before shutdown timeout; cancelling"
                    )
            for task in callback_tasks:
                if not task.done():
                    task.cancel()
            results = await asyncio.gather(*callback_tasks, return_exceptions=True)
            self._collect_callback_errors(results)
        for segment in self._segments:
            await segment.close()
        if self._sub is not None:
            await self._sub.close()
        if self._callback_errors:
            raise self._callback_errors[0]

    async def __aenter__(self) -> "MediaOutput":
        self.start_callbacks()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.close()

    def get_stats(self) -> MediaOutputStats:
        decoder_stats = (
            self._processor.get_stats() if self._processor is not None else self._last_decoder_stats
        )
        return MediaOutputStats(
            elapsed_s=max(0.0, time.time() - self._started_at),
            segments_consumed=self._stats["segments_consumed"],
            bytes_read=self._stats["bytes_read"],
            chunks_read=self._stats["chunks_read"],
            content_type_errors=self._stats["content_type_errors"],
            segment_read_errors=self._stats["segment_read_errors"],
            segment_max_bytes_exceeded=self._stats["segment_max_bytes_exceeded"],
            consumer_lag_skip_latest=self._stats["consumer_lag_skip_latest"],
            consumer_lag_retry_earliest=self._stats["consumer_lag_retry_earliest"],
            consumer_lag_fail=self._stats["consumer_lag_fail"],
            video_packets_demuxed=self._stats["video_packets_demuxed"],
            audio_packets_demuxed=self._stats["audio_packets_demuxed"],
            other_packets_demuxed=self._stats["other_packets_demuxed"],
            video_frames_decoded=self._stats["video_frames_decoded"],
            audio_frames_decoded=self._stats["audio_frames_decoded"],
            packet_errors=self._stats["packet_errors"],
            decode_errors=self._stats["decode_errors"],
            decoder=decoder_stats,
            subscriber=(self._sub.get_stats() if self._sub is not None else None),
        )


def _normalize_content_type(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return value.split(";", 1)[0].strip().lower()


def _normalize_accepted_content_types(
    accepted_content_types: Collection[str],
) -> frozenset[str]:
    normalized = frozenset(
        content_type
        for value in accepted_content_types
        if (content_type := _normalize_content_type(value)) is not None
    )
    if not normalized:
        raise ValueError("accepted_content_types must contain at least one content type")
    return normalized


def _require_content_type(value: Optional[str], accepted: frozenset[str]) -> None:
    normalized = _normalize_content_type(value)
    if normalized not in accepted:
        raise LivepeerGatewayError(
            f"Expected Content-Type in {sorted(accepted)!r}, got {value!r}"
        )


async def _maybe_await(value: None | Awaitable[None]) -> None:
    if inspect.isawaitable(value):
        _ = await value
    return None
