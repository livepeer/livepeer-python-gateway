from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Union

import av


@dataclass(frozen=True)
class DecodedMediaFrame:
    """A single decoded media frame with its source timing metadata.

    Base type for :class:`VideoDecodedMediaFrame` and :class:`AudioDecodedMediaFrame`.

    Attributes:
        kind: Stream kind this frame came from, e.g. ``"video"`` or ``"audio"``.
        stream_index: Index of the source stream within the container.
        frame: The underlying decoded PyAV frame.
        pts: Presentation timestamp in ``time_base`` units, or ``None`` if unset.
        time_base: Stream time base (seconds per ``pts`` unit), or ``None``.
        pts_time: Presentation timestamp in seconds (``pts * time_base``). Media time
            relative to the stream start, not wall-clock time.
        demuxed_at: Wall-clock time (``time.time()``) when the packet was demuxed.
        decoded_at: Wall-clock time (``time.time()``) when the frame was decoded.
    """

    kind: str
    stream_index: int
    frame: Union[av.VideoFrame, av.AudioFrame]
    pts: Optional[int]
    time_base: Optional[Fraction]
    # pts_time: presentation timestamp in seconds (pts * time_base). Not wall-clock time.
    pts_time: Optional[float]
    demuxed_at: float
    decoded_at: float


@dataclass(frozen=True)
class VideoDecodedMediaFrame(DecodedMediaFrame):
    """A decoded video frame.

    Extends :class:`DecodedMediaFrame` with video-specific attributes.

    Attributes:
        width: Frame width in pixels.
        height: Frame height in pixels.
        pix_fmt: Pixel format name (e.g. ``"yuv420p"``), or ``None`` if unknown.
    """

    width: int
    height: int
    pix_fmt: Optional[str]


@dataclass(frozen=True)
class AudioDecodedMediaFrame(DecodedMediaFrame):
    """A decoded audio frame.

    Extends :class:`DecodedMediaFrame` with audio-specific attributes.

    Attributes:
        sample_rate: Samples per second, or ``None`` if unknown.
        layout: Channel layout name (e.g. ``"stereo"``), or ``None`` if unknown.
        format: Sample format name (e.g. ``"fltp"``), or ``None`` if unknown.
        samples: Number of samples per channel in this frame, or ``None``.
    """

    sample_rate: Optional[int]
    layout: Optional[str]
    format: Optional[str]
    samples: Optional[int]


@dataclass(frozen=True)
class DemuxedMediaPacket:
    """A single demuxed media packet with its source timing metadata.

    Yielded by ``MediaOutput.packets()`` (and passed to ``on_packet``) before
    decoding.

    Attributes:
        kind: Stream kind this packet came from, e.g. ``"video"`` or ``"audio"``.
        stream_index: Index of the source stream within the container.
        packet: The underlying demuxed PyAV packet.
        pts: Presentation timestamp in ``time_base`` units, or ``None`` if unset.
        dts: Decode timestamp in ``time_base`` units, or ``None`` if unset.
        time_base: Stream time base (seconds per timestamp unit), or ``None``.
        pts_time: Presentation timestamp in seconds (``pts * time_base``), or ``None``.
            Media time relative to the stream start, not wall-clock time.
        dts_time: Decode timestamp in seconds (``dts * time_base``), or ``None``.
        is_keyframe: Whether the packet is a keyframe.
        size: Packet size in bytes.
        demuxed_at: Wall-clock time (``time.time()``) when the packet was demuxed.
    """

    kind: str
    stream_index: int
    packet: av.Packet
    pts: Optional[int]
    dts: Optional[int]
    time_base: Optional[Fraction]
    pts_time: Optional[float]
    dts_time: Optional[float]
    is_keyframe: bool
    size: int
    demuxed_at: float


_EOF = object()
_END = object()


@dataclass(frozen=True)
class DecoderQueueStats:
    # These queue metrics are intentionally best-effort snapshots. They are
    # derived from single-writer totals that are updated from different threads
    # without per-operation locks, and rely on CPython's implementation details
    # for cross-thread integer reads and writes being safe enough for telemetry.
    # They should not be treated as exact, atomic queue state.

    # queued_chunks: current number of chunk objects still waiting in the
    # cross-thread input queue before they are moved into the internal buffer.
    queued_chunks: int

    # queued_bytes: current number of bytes still waiting in the cross-thread
    # input queue. Unlike buffered_bytes, these bytes have not yet been moved
    # into the internal bytearray used to satisfy read(size) calls.
    queued_bytes: int

    # buffered_bytes: current number of bytes already removed from the
    # cross-thread queue and staged in the internal bytearray for future reads.
    buffered_bytes: int

    # total_chunks_dequeued: lifetime count of chunk objects moved out of the
    # cross-thread queue and into the internal bytearray.
    total_chunks_dequeued: int

    # total_bytes_dequeued: lifetime count of bytes moved out of the
    # cross-thread queue and into the internal bytearray.
    total_bytes_dequeued: int

    # total_bytes_read: lifetime count of bytes returned from read() to PyAV.
    total_bytes_read: int

    # output_items_queued: current number of objects waiting in the decoder
    # output queue. This includes frames plus terminal/error markers.
    output_items_queued: int

    # total_output_items_dequeued: lifetime count of objects removed from the
    # decoder output queue by MediaOutput.frames().
    total_output_items_dequeued: int

    # output_wait_s: cumulative wall-clock time spent blocked waiting for the
    # next decoder output item to become available.
    output_wait_s: float

    # queue_s: aggregate span of source PTS media-time between the most-
    # recently enqueued decoded frame and the most-recently dequeued decoded
    # frame. Approximates how much media time is sitting in the decoder output
    # queue. Aggregated across all streams (audio+video); noisy when streams
    # diverge in PTS but still useful as a rough depth.
    queue_s: float

    # processed_s: aggregate span of source PTS media-time between the first
    # and most-recently dequeued decoded frames. Approximates total media time
    # the decoder consumer has pulled out since start.
    processed_s: float


# Internal adapter that turns async-fed byte chunks into a blocking, file-like
# stream for PyAV.
class _BlockingByteStream:
    def __init__(self) -> None:
        # TODO: add backpressure (bounded queue + blocking feed) to avoid unbounded
        # memory growth if the decoder thread can't keep up with the producer.
        self._queue: "queue.Queue[object]" = queue.Queue()
        self._buffer = bytearray()
        self._closed = False
        self._total_chunks_fed = 0
        self._total_bytes_fed = 0
        self._buffered_bytes = 0
        self._total_chunks_dequeued = 0
        self._total_bytes_dequeued = 0
        self._total_bytes_read = 0

    def feed(self, data: bytes) -> None:
        # Called from the async producer to enqueue raw bytes for decoding.
        if not data:
            return
        self._total_chunks_fed += 1
        self._total_bytes_fed += len(data)
        self._queue.put(data)

    def close(self) -> None:
        # Signal EOF to the blocking reader.
        self._queue.put(_EOF)

    def read(self, size: int = -1) -> bytes:
        # Blocking read interface consumed by PyAV demuxer.
        if size is None or size < 0:
            size = 64 * 1024
        if size == 0:
            return b""

        if self._buffer:
            out = self._buffer[:size]
            del self._buffer[:size]
            self._buffered_bytes -= len(out)
            self._total_bytes_read += len(out)
            return bytes(out)

        while not self._buffer and not self._closed:
            item = self._queue.get()
            if item is _EOF:
                self._closed = True
                break
            if not item:
                continue
            self._buffer.extend(item)  # type: ignore[arg-type]
            chunk_len = len(item)
            self._buffered_bytes += chunk_len
            self._total_chunks_dequeued += 1
            self._total_bytes_dequeued += chunk_len

        if not self._buffer and self._closed:
            return b""

        out = self._buffer[:size]
        del self._buffer[:size]
        self._buffered_bytes -= len(out)
        self._total_bytes_read += len(out)
        return bytes(out)

    def get_stats(self) -> DecoderQueueStats:
        total_chunks_fed = self._total_chunks_fed
        total_bytes_fed = self._total_bytes_fed
        total_chunks_dequeued = self._total_chunks_dequeued
        total_bytes_dequeued = self._total_bytes_dequeued
        total_bytes_read = self._total_bytes_read
        buffered_bytes = self._buffered_bytes
        return DecoderQueueStats(
            queued_chunks=max(0, total_chunks_fed - total_chunks_dequeued),
            queued_bytes=max(0, total_bytes_fed - total_bytes_dequeued),
            buffered_bytes=max(0, buffered_bytes),
            total_chunks_dequeued=total_chunks_dequeued,
            total_bytes_dequeued=total_bytes_dequeued,
            total_bytes_read=total_bytes_read,
            output_items_queued=0,
            total_output_items_dequeued=0,
            output_wait_s=0.0,
            queue_s=0.0,
            processed_s=0.0,
        )


class _DecoderError:
    __slots__ = ("error",)

    def __init__(self, error: BaseException) -> None:
        self.error = error


def _fraction_from_time_base(time_base: object) -> Optional[Fraction]:
    numerator = getattr(time_base, "numerator", None)
    denominator = getattr(time_base, "denominator", None)
    if numerator is not None and denominator is not None:
        try:
            return Fraction(int(numerator), int(denominator))
        except Exception:
            return None
    try:
        return Fraction(time_base)  # type: ignore[arg-type]
    except Exception:
        return None


def _time_from_pts(pts: Optional[int], time_base: Optional[Fraction]) -> Optional[float]:
    if pts is None or time_base is None:
        return None
    try:
        return float(Fraction(pts) * time_base)
    except Exception:
        return None


def _build_decoded_frame(
    frame: Union[av.VideoFrame, av.AudioFrame],
    *,
    stream_index: int,
    demuxed_at: float,
    decoded_at: float,
) -> AudioDecodedMediaFrame | VideoDecodedMediaFrame:
    pts = frame.pts
    time_base = _fraction_from_time_base(frame.time_base) if frame.time_base is not None else None
    pts_time = _time_from_pts(pts, time_base)

    if isinstance(frame, av.VideoFrame):
        return VideoDecodedMediaFrame(
            kind="video",
            stream_index=stream_index,
            frame=frame,
            pts=pts,
            time_base=time_base,
            pts_time=pts_time,
            demuxed_at=demuxed_at,
            decoded_at=decoded_at,
            width=frame.width,
            height=frame.height,
            pix_fmt=frame.format.name if frame.format else None,
        )

    return AudioDecodedMediaFrame(
        kind="audio",
        stream_index=stream_index,
        frame=frame,
        pts=pts,
        time_base=time_base,
        pts_time=pts_time,
        demuxed_at=demuxed_at,
        decoded_at=decoded_at,
        sample_rate=frame.sample_rate,
        layout=frame.layout.name if frame.layout else None,
        format=frame.format.name if frame.format else None,
        samples=frame.samples,
    )


def _stream_kind(packet: av.Packet) -> str:
    stream = getattr(packet, "stream", None)
    kind = getattr(stream, "type", None)
    if isinstance(kind, str) and kind:
        return kind
    return "unknown"


def _packet_time_base(packet: av.Packet) -> Optional[Fraction]:
    time_base = getattr(packet, "time_base", None)
    if time_base is None:
        stream = getattr(packet, "stream", None)
        time_base = getattr(stream, "time_base", None)
    if time_base is None:
        return None
    return _fraction_from_time_base(time_base)


def _build_demuxed_packet(
    packet: av.Packet,
    *,
    demuxed_at: float,
) -> DemuxedMediaPacket:
    time_base = _packet_time_base(packet)
    pts = getattr(packet, "pts", None)
    dts = getattr(packet, "dts", None)
    return DemuxedMediaPacket(
        kind=_stream_kind(packet),
        stream_index=getattr(getattr(packet, "stream", None), "index", -1),
        packet=packet,
        pts=pts,
        dts=dts,
        time_base=time_base,
        pts_time=_time_from_pts(pts, time_base),
        dts_time=_time_from_pts(dts, time_base),
        is_keyframe=bool(getattr(packet, "is_keyframe", False)),
        size=int(getattr(packet, "size", 0) or 0),
        demuxed_at=demuxed_at,
    )


def _item_pts_time(item: object) -> Optional[float]:
    if isinstance(item, (DecodedMediaFrame, DemuxedMediaPacket)):
        return item.pts_time
    return None


class _MpegTsOutputWorker:
    def __init__(self, *, thread_name: str) -> None:
        self._reader = _BlockingByteStream()
        # TODO: add backpressure here too (bounded queue) if callers are slow to consume.
        self._output: "queue.Queue[object]" = queue.Queue()
        self._total_output_items_enqueued = 0
        self._total_output_items_dequeued = 0
        self._output_wait_s = 0.0
        # Aggregate source-PTS media-time watermarks across all streams for
        # output-queue telemetry. Best-effort snapshots updated from producer
        # (decoder thread) and consumer (get()) without per-op locks.
        self._max_pts_time_enqueued: Optional[float] = None
        self._max_pts_time_dequeued: Optional[float] = None
        self._min_pts_time_dequeued: Optional[float] = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=thread_name, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def feed(self, data: bytes) -> None:
        self._reader.feed(data)

    def close(self) -> None:
        self._reader.close()

    def stop(self) -> None:
        self._stop.set()
        self._reader.close()

    def join(self) -> None:
        self._thread.join()

    def get(self) -> object:
        started_wait = time.monotonic()
        item = self._output.get()
        self._output_wait_s += max(0.0, time.monotonic() - started_wait)
        self._total_output_items_dequeued += 1
        pts_time = _item_pts_time(item)
        if pts_time is not None:
            if (
                self._max_pts_time_dequeued is None
                or pts_time > self._max_pts_time_dequeued
            ):
                self._max_pts_time_dequeued = pts_time
            if (
                self._min_pts_time_dequeued is None
                or pts_time < self._min_pts_time_dequeued
            ):
                self._min_pts_time_dequeued = pts_time
        return item

    def get_stats(self) -> DecoderQueueStats:
        reader_stats = self._reader.get_stats()
        total_output_items_enqueued = self._total_output_items_enqueued
        total_output_items_dequeued = self._total_output_items_dequeued
        max_enq = self._max_pts_time_enqueued
        max_deq = self._max_pts_time_dequeued
        min_deq = self._min_pts_time_dequeued
        if max_enq is None or max_deq is None:
            queue_s = 0.0
        else:
            queue_s = max(0.0, max_enq - max_deq)
        if max_deq is None or min_deq is None:
            processed_s = 0.0
        else:
            processed_s = max(0.0, max_deq - min_deq)
        return DecoderQueueStats(
            queued_chunks=reader_stats.queued_chunks,
            queued_bytes=reader_stats.queued_bytes,
            buffered_bytes=reader_stats.buffered_bytes,
            total_chunks_dequeued=reader_stats.total_chunks_dequeued,
            total_bytes_dequeued=reader_stats.total_bytes_dequeued,
            total_bytes_read=reader_stats.total_bytes_read,
            output_items_queued=max(0, total_output_items_enqueued - total_output_items_dequeued),
            total_output_items_dequeued=total_output_items_dequeued,
            output_wait_s=self._output_wait_s,
            queue_s=queue_s,
            processed_s=processed_s,
        )

    def _put_output_item(self, item: object) -> None:
        self._total_output_items_enqueued += 1
        pts_time = _item_pts_time(item)
        if pts_time is not None and (
            self._max_pts_time_enqueued is None
            or pts_time > self._max_pts_time_enqueued
        ):
            self._max_pts_time_enqueued = pts_time
        self._output.put(item)

    def _run(self) -> None:
        raise NotImplementedError


class MpegTsDecoder(_MpegTsOutputWorker):
    def __init__(self) -> None:
        super().__init__(thread_name="MpegTsDecoder")

    def _run(self) -> None:
        container: Optional[av.container.InputContainer] = None
        try:
            container = av.open(self._reader, format="mpegts", mode="r")
            for packet in container.demux():
                if self._stop.is_set():
                    break
                demuxed_at = time.time()
                if packet is None:
                    continue
                stream_index = getattr(packet.stream, "index", -1)
                try:
                    frames = packet.decode()
                except Exception as e:
                    self._put_output_item(_DecoderError(e))
                    break
                for frame in frames:
                    decoded_at = time.time()
                    decoded = _build_decoded_frame(
                        frame,
                        stream_index=stream_index,
                        demuxed_at=demuxed_at,
                        decoded_at=decoded_at,
                    )
                    self._put_output_item(decoded)
        except Exception as e:
            self._put_output_item(_DecoderError(e))
        finally:
            if container is not None:
                try:
                    container.close()
                except Exception:
                    pass
            self._put_output_item(_END)


class MpegTsPacketDemuxer(_MpegTsOutputWorker):
    def __init__(self) -> None:
        super().__init__(thread_name="MpegTsPacketDemuxer")

    def _run(self) -> None:
        container: Optional[av.container.InputContainer] = None
        try:
            container = av.open(self._reader, format="mpegts", mode="r")
            for packet in container.demux():
                if self._stop.is_set():
                    break
                if packet is None:
                    continue
                self._put_output_item(
                    _build_demuxed_packet(
                        packet,
                        demuxed_at=time.time(),
                    )
                )
        except Exception as e:
            self._put_output_item(_DecoderError(e))
        finally:
            if container is not None:
                try:
                    container.close()
                except Exception:
                    pass
            self._put_output_item(_END)


def is_decoder_end(item: object) -> bool:
    return item is _END


def decoder_error(item: object) -> Optional[BaseException]:
    if isinstance(item, _DecoderError):
        return item.error
    return None
