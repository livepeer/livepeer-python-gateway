from __future__ import annotations

import asyncio
import os
import queue
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import av
import pytest

from livepeer_gateway.media_decode import MpegTsDecoder
from livepeer_gateway.media_output import MediaOutput


_CHUNK_SIZE = 188 * 8


@dataclass(frozen=True)
class _Cadence:
    default_delay_s: float
    active_count: int = 0
    idle_count: int = 0
    active_delay_s: float = 0.0
    idle_delay_s: float = 0.0
    stall_after: int | None = None
    stall_delay_s: float = 0.0

    def delay_for(self, completed_count: int) -> float:
        if self.stall_after is not None and completed_count == self.stall_after:
            return self.stall_delay_s
        period = self.active_count + self.idle_count
        if period <= 0:
            return self.default_delay_s
        position = completed_count % period
        if position < self.active_count:
            return self.active_delay_s
        return self.idle_delay_s


@dataclass(frozen=True)
class _DriftReport:
    decoded_frames: int
    sample_count: int
    max_abs_drift_queued_chunks: int
    max_abs_drift_queued_bytes: int
    max_abs_drift_buffered_bytes: int
    max_abs_drift_output_items_queued: int
    final_queue_snapshot: tuple[int, int, int, int]


class _SyntheticMediaOutput(MediaOutput):
    def __init__(
        self,
        payload: bytes,
        *,
        producer_cadence: _Cadence,
    ) -> None:
        super().__init__("memory://decoder-metrics")
        self._payload = payload
        self._producer_cadence = producer_cadence

    async def _iter_bytes(self):  # type: ignore[override]
        offset = 0
        completed_chunks = 0
        while offset < len(self._payload):
            delay_s = self._producer_cadence.delay_for(completed_chunks)
            if delay_s > 0.0:
                await asyncio.sleep(delay_s)
            next_offset = min(len(self._payload), offset + _CHUNK_SIZE)
            yield self._payload[offset:next_offset]
            offset = next_offset
            completed_chunks += 1


class _PermanentlyStalledMediaOutput(MediaOutput):
    def __init__(self) -> None:
        super().__init__("memory://stalled-producer")
        self.producer_started = asyncio.Event()
        self.producer_cancelled = asyncio.Event()
        self._never_resume = asyncio.Event()

    async def _iter_bytes(self):  # type: ignore[override]
        self.producer_started.set()
        try:
            await self._never_resume.wait()
        except asyncio.CancelledError:
            self.producer_cancelled.set()
            raise
        yield b""  # pragma: no cover - makes this an async generator


def _render_video_frame(width: int, height: int, frame_index: int) -> av.VideoFrame:
    row = bytearray()
    marker_left = (frame_index * 5) % max(1, width - 20)
    marker_right = min(width, marker_left + 20)
    for y in range(height):
        for x in range(width):
            if marker_left <= x < marker_right and 12 <= y < min(height, 32):
                row.extend((32, 224, 64))
            elif y < 8:
                row.extend((240, 240, 240))
            else:
                row.extend(
                    ((frame_index * 7) % 255, (x * 2) % 255, (y * 3) % 255)
                )
    frame = av.VideoFrame(width, height, "rgb24")
    frame.planes[0].update(bytes(row))
    frame.pts = frame_index
    frame.time_base = Fraction(1, 30)
    return frame


def _generate_mpegts_payload(
    *,
    frame_count: int = 90,
    width: int = 160,
    height: int = 90,
    fps: int = 30,
) -> bytes:
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".ts", delete=False) as tmp:
            tmp_path = tmp.name

        container = av.open(tmp_path, mode="w", format="mpegts")
        stream = container.add_stream("mpeg2video", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for frame_index in range(frame_count):
            frame = _render_video_frame(width, height, frame_index)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
        container.close()
        return Path(tmp_path).read_bytes()
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _actual_decoder_snapshot(decoder: object) -> tuple[int, int, int, int]:
    reader = getattr(decoder, "_reader")
    input_queue = getattr(reader, "_queue")
    with input_queue.mutex:
        input_items = list(input_queue.queue)
    queued_payloads = [
        item
        for item in input_items
        if isinstance(item, (bytes, bytearray, memoryview))
    ]
    output_queue = getattr(decoder, "_output")
    with output_queue.mutex:
        output_items_queued = len(output_queue.queue)
    return (
        len(queued_payloads),
        sum(len(item) for item in queued_payloads),
        len(getattr(reader, "_buffer")),
        output_items_queued,
    )


async def _simulate_decoder_metric_drift(
    *,
    producer_cadence: _Cadence,
    consumer_cadence: _Cadence,
    frame_count: int = 90,
    sample_interval_s: float = 0.0005,
) -> _DriftReport:
    output = _SyntheticMediaOutput(
        _generate_mpegts_payload(frame_count=frame_count),
        producer_cadence=producer_cadence,
    )
    stop_sampling = asyncio.Event()
    maxima = [0, 0, 0, 0]
    sample_count = 0

    async def _sample() -> None:
        nonlocal sample_count
        while True:
            decoder = output._processor
            if isinstance(decoder, MpegTsDecoder):
                stats = output.get_stats().decoder
                if stats is not None:
                    actual = _actual_decoder_snapshot(decoder)
                    reported = (
                        stats.queued_chunks,
                        stats.queued_bytes,
                        stats.buffered_bytes,
                        stats.output_items_queued,
                    )
                    for index, (reported_value, actual_value) in enumerate(
                        zip(reported, actual, strict=True)
                    ):
                        maxima[index] = max(
                            maxima[index], abs(reported_value - actual_value)
                        )
                    sample_count += 1
            if stop_sampling.is_set():
                return
            await asyncio.sleep(sample_interval_s)

    sampler_task = asyncio.create_task(_sample())
    decoded_frames = 0
    try:
        async for _decoded in output.frames():
            decoded_frames += 1
            delay_s = consumer_cadence.delay_for(decoded_frames)
            if delay_s > 0.0:
                await asyncio.sleep(delay_s)
    finally:
        stop_sampling.set()
        await sampler_task

    final_stats = output.get_stats().decoder
    assert final_stats is not None
    return _DriftReport(
        decoded_frames=decoded_frames,
        sample_count=sample_count,
        max_abs_drift_queued_chunks=maxima[0],
        max_abs_drift_queued_bytes=maxima[1],
        max_abs_drift_buffered_bytes=maxima[2],
        max_abs_drift_output_items_queued=maxima[3],
        final_queue_snapshot=(
            final_stats.queued_chunks,
            final_stats.queued_bytes,
            final_stats.buffered_bytes,
            final_stats.output_items_queued,
        ),
    )


_STEADY_PRODUCER = _Cadence(default_delay_s=0.0005)
_STEADY_CONSUMER = _Cadence(default_delay_s=0.002)
_BURSTY_PRODUCER = _Cadence(
    default_delay_s=0.0,
    active_count=8,
    idle_count=2,
    active_delay_s=0.0,
    idle_delay_s=0.002,
)
_BURSTY_CONSUMER = _Cadence(
    default_delay_s=0.0,
    active_count=16,
    idle_count=4,
    active_delay_s=0.0,
    idle_delay_s=0.003,
)
_STALLED_PRODUCER = _Cadence(
    default_delay_s=0.0005,
    stall_after=12,
    stall_delay_s=0.050,
)
_STALLED_CONSUMER = _Cadence(
    default_delay_s=0.002,
    stall_after=20,
    stall_delay_s=0.050,
)


@pytest.mark.parametrize(
    ("producer_cadence", "consumer_cadence"),
    [
        pytest.param(_STEADY_PRODUCER, _STEADY_CONSUMER, id="steady"),
        pytest.param(_BURSTY_PRODUCER, _STEADY_CONSUMER, id="bursty-producer"),
        pytest.param(_STEADY_PRODUCER, _BURSTY_CONSUMER, id="bursty-consumer"),
        pytest.param(_BURSTY_PRODUCER, _BURSTY_CONSUMER, id="combined-bursts"),
        pytest.param(_STALLED_PRODUCER, _STEADY_CONSUMER, id="stalled-producer"),
        pytest.param(_STEADY_PRODUCER, _STALLED_CONSUMER, id="stalled-consumer"),
    ],
)
def test_decoder_queue_metrics_track_real_queues_during_recovery(
    producer_cadence: _Cadence,
    consumer_cadence: _Cadence,
) -> None:
    report = asyncio.run(
        _simulate_decoder_metric_drift(
            producer_cadence=producer_cadence,
            consumer_cadence=consumer_cadence,
        )
    )

    assert report.decoded_frames == 90
    assert report.sample_count > 0
    assert report.max_abs_drift_queued_chunks <= 1
    assert report.max_abs_drift_queued_bytes <= _CHUNK_SIZE
    assert report.max_abs_drift_buffered_bytes <= _CHUNK_SIZE
    assert report.max_abs_drift_output_items_queued <= 1
    assert report.final_queue_snapshot == (0, 0, 0, 0)


def test_actual_decoder_snapshot_counts_only_payload_chunks() -> None:
    input_queue: queue.Queue[object] = queue.Queue()
    input_queue.put(b"abc")
    input_queue.put(object())
    output_queue: queue.Queue[object] = queue.Queue()
    output_queue.put(object())
    decoder = SimpleNamespace(
        _reader=SimpleNamespace(_queue=input_queue, _buffer=bytearray(b"de")),
        _output=output_queue,
    )
    assert _actual_decoder_snapshot(decoder) == (1, 3, 2, 1)


def test_cancelling_consumer_unblocks_permanently_stalled_producer() -> None:
    async def _run() -> None:
        output = _PermanentlyStalledMediaOutput()
        consumer_task = asyncio.create_task(anext(output.frames()))
        await asyncio.wait_for(output.producer_started.wait(), timeout=1.0)
        decoder = output._processor
        assert isinstance(decoder, MpegTsDecoder)

        consumer_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(consumer_task, timeout=1.0)

        assert output.producer_cancelled.is_set()
        assert output._processor is None
        assert not decoder._thread.is_alive()

    asyncio.run(_run())


def test_closing_stalled_consumer_joins_decoder_with_backlog() -> None:
    async def _run() -> None:
        output = _SyntheticMediaOutput(
            _generate_mpegts_payload(frame_count=90),
            producer_cadence=_Cadence(default_delay_s=0.0),
        )
        frames = output.frames()
        await asyncio.wait_for(anext(frames), timeout=1.0)
        decoder = output._processor
        assert isinstance(decoder, MpegTsDecoder)

        deadline = asyncio.get_running_loop().time() + 1.0
        while decoder.get_stats().output_items_queued == 0:
            if asyncio.get_running_loop().time() >= deadline:
                pytest.fail("decoder output queue did not build while consumer was stalled")
            await asyncio.sleep(0.005)

        stats = decoder.get_stats()
        actual = _actual_decoder_snapshot(decoder)
        assert stats.output_items_queued > 0
        assert abs(stats.output_items_queued - actual[3]) <= 1

        await asyncio.wait_for(frames.aclose(), timeout=1.0)
        assert output._processor is None
        assert not decoder._thread.is_alive()

    asyncio.run(_run())
