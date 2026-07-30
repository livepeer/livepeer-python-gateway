import asyncio
import importlib
import threading
import time
import types
from dataclasses import asdict
from unittest import mock

import pytest

media_output_mod = importlib.import_module("livepeer_gateway.media_output")
media_publish_mod = importlib.import_module("livepeer_gateway.media_publish")
segment_reader_mod = importlib.import_module("livepeer_gateway.segment_reader")
trickle_publisher_mod = importlib.import_module("livepeer_gateway.trickle_publisher")
trickle_subscriber_mod = importlib.import_module("livepeer_gateway.trickle_subscriber")
media_decode_mod = importlib.import_module("livepeer_gateway.media_decode")
lv2v_mod = importlib.import_module("livepeer_gateway.lv2v")

MediaOutput = media_output_mod.MediaOutput
MediaOutputStats = media_output_mod.MediaOutputStats
LiveVideoToVideo = lv2v_mod.LiveVideoToVideo
MediaPublish = media_publish_mod.MediaPublish
MediaPublishConfig = media_publish_mod.MediaPublishConfig
MediaPublishStats = media_publish_mod.MediaPublishStats
LivepeerGatewayError = importlib.import_module(
    "livepeer_gateway.errors"
).LivepeerGatewayError
SegmentReaderStats = segment_reader_mod.SegmentReaderStats
TricklePublisher = trickle_publisher_mod.TricklePublisher
TricklePublisherStats = trickle_publisher_mod.TricklePublisherStats
TrickleSubscriber = trickle_subscriber_mod.TrickleSubscriber
TrickleSubscriberStats = trickle_subscriber_mod.TrickleSubscriberStats
AudioDecodedMediaFrame = media_decode_mod.AudioDecodedMediaFrame
DemuxedMediaPacket = media_decode_mod.DemuxedMediaPacket
DecoderQueueStats = media_decode_mod.DecoderQueueStats
MpegTsDecoder = media_decode_mod.MpegTsDecoder
BlockingByteStream = media_decode_mod._BlockingByteStream
FrameQueue = media_publish_mod._FrameQueue
_new_track_stats = media_publish_mod._new_track_stats


class _FakeReader:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)

    async def read(self, chunk_size: int = 32 * 1024):
        if not self._chunks:
            return b""
        chunk = self._chunks.pop(0)
        return chunk[:chunk_size]


class _FakeSegment:
    def __init__(self, content_type: str | None, chunks: list[bytes]) -> None:
        self._headers = {"Lp-Trickle-Seq": "1"}
        if content_type is not None:
            self._headers["Content-Type"] = content_type
        self._chunks = list(chunks)
        self._local_seq = 0

    def headers(self):
        return self._headers

    def make_reader(self):
        return _FakeReader(self._chunks)

    async def close(self) -> None:
        return None

    def get_stats(self):
        return SegmentReaderStats(
            chunks_read=len(self._chunks),
            bytes_read=sum(len(chunk) for chunk in self._chunks),
            read_errors=0,
            max_bytes_exceeded=0,
            segment_seq=1,
        )


class _FakeDecoder:
    def __init__(self, items: list[object]) -> None:
        self._items = list(items)

    def start(self) -> None:
        return None

    def feed(self, _data: bytes) -> None:
        return None

    def close(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def join(self) -> None:
        return None

    def get(self) -> object:
        return self._items.pop(0)

    def get_stats(self):
        return DecoderQueueStats(
            queued_chunks=0,
            queued_bytes=0,
            buffered_bytes=0,
            total_chunks_dequeued=0,
            total_bytes_dequeued=0,
            total_bytes_read=0,
            output_items_queued=max(0, len(self._items)),
            total_output_items_dequeued=0,
            output_wait_s=0.0,
            queue_s=0.0,
            processed_s=0.0,
        )


class _FakePacket:
    def __init__(
        self,
        *,
        kind: str,
        stream_index: int,
        pts: int | None,
        dts: int | None = None,
        pts_time: float | None = None,
        dts_time: float | None = None,
        is_keyframe: bool = False,
        size: int = 0,
    ) -> None:
        self.stream = types.SimpleNamespace(index=stream_index, type=kind)
        self.pts = pts
        self.dts = dts
        self.time_base = None
        self.is_keyframe = is_keyframe
        self.size = size
        self._pts_time = pts_time
        self._dts_time = dts_time


def _fake_demuxed_packet(
    *,
    kind: str,
    stream_index: int,
    pts: int | None,
    dts: int | None = None,
    pts_time: float | None = None,
    dts_time: float | None = None,
    is_keyframe: bool = False,
    size: int = 0,
    demuxed_at: float = 0.0,
) -> DemuxedMediaPacket:
    packet = _FakePacket(
        kind=kind,
        stream_index=stream_index,
        pts=pts,
        dts=dts,
        pts_time=pts_time,
        dts_time=dts_time,
        is_keyframe=is_keyframe,
        size=size,
    )
    return DemuxedMediaPacket(
        kind=kind,
        stream_index=stream_index,
        packet=packet,
        pts=pts,
        dts=dts,
        time_base=None,
        pts_time=pts_time,
        dts_time=dts_time,
        is_keyframe=is_keyframe,
        size=size,
        demuxed_at=demuxed_at,
    )


class _FakePacketDemuxer:
    def __init__(self, items: list[object]) -> None:
        self._items = list(items)

    def start(self) -> None:
        return None

    def feed(self, _data: bytes) -> None:
        return None

    def close(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def join(self) -> None:
        return None

    def get(self) -> object:
        return self._items.pop(0)

    def get_stats(self):
        return DecoderQueueStats(
            queued_chunks=0,
            queued_bytes=0,
            buffered_bytes=0,
            total_chunks_dequeued=0,
            total_bytes_dequeued=0,
            total_bytes_read=0,
            output_items_queued=max(0, len(self._items)),
            total_output_items_dequeued=0,
            output_wait_s=0.0,
            queue_s=0.0,
            processed_s=0.0,
        )


class _TrackingPacketDemuxer:
    instances: list["_TrackingPacketDemuxer"] = []

    def __init__(self, items: list[object] | None = None) -> None:
        self._items = list(items or [])
        self.started = False
        self.closed = False
        self.stopped = False
        self.joined = False
        self.feed_count = 0
        _TrackingPacketDemuxer.instances.append(self)

    def start(self) -> None:
        self.started = True

    def feed(self, _data: bytes) -> None:
        self.feed_count += 1

    def close(self) -> None:
        self.closed = True

    def stop(self) -> None:
        self.stopped = True

    def join(self) -> None:
        self.joined = True

    def get(self) -> object:
        return self._items.pop(0)

    def get_stats(self):
        return DecoderQueueStats(
            queued_chunks=0,
            queued_bytes=0,
            buffered_bytes=0,
            total_chunks_dequeued=0,
            total_bytes_dequeued=0,
            total_bytes_read=0,
            output_items_queued=max(0, len(self._items)),
            total_output_items_dequeued=0,
            output_wait_s=0.0,
            queue_s=0.0,
            processed_s=0.0,
        )


class _TrackingDecoder:
    instances: list["_TrackingDecoder"] = []

    def __init__(self, items: list[object] | None = None) -> None:
        self._items = list(items or [])
        self.started = False
        self.closed = False
        self.stopped = False
        self.joined = False
        self.feed_count = 0
        _TrackingDecoder.instances.append(self)

    def start(self) -> None:
        self.started = True

    def feed(self, _data: bytes) -> None:
        self.feed_count += 1

    def close(self) -> None:
        self.closed = True

    def stop(self) -> None:
        self.stopped = True

    def join(self) -> None:
        self.joined = True

    def get(self) -> object:
        return self._items.pop(0)

    def get_stats(self):
        return DecoderQueueStats(
            queued_chunks=0,
            queued_bytes=0,
            buffered_bytes=0,
            total_chunks_dequeued=0,
            total_bytes_dequeued=0,
            total_bytes_read=0,
            output_items_queued=max(0, len(self._items)),
            total_output_items_dequeued=0,
            output_wait_s=0.0,
            queue_s=0.0,
            processed_s=0.0,
        )


async def _collect_bytes(media_output: MediaOutput) -> list[bytes]:
    return [chunk async for chunk in media_output.bytes()]


async def _collect_frames(media_output: MediaOutput) -> list[AudioDecodedMediaFrame]:
    return [frame async for frame in media_output.frames()]


async def _collect_packets(media_output: MediaOutput) -> list[DemuxedMediaPacket]:
    return [packet async for packet in media_output.packets()]


class TestStatsPull:
    def _make_media_output_for_segments(self, *segments, **kwargs):
        media_output = MediaOutput("http://example.test/trickle", **kwargs)
        pending = list(segments)

        async def _next_segment(_seq: int):
            if pending:
                return pending.pop(0)
            return None

        media_output._next_segment = _next_segment  # type: ignore[method-assign]
        return media_output

    def test_publisher_stats_are_typed_and_stringable(self) -> None:
        publisher = TricklePublisher("http://example.test/trickle", "video/mp2t")
        stats = publisher.get_stats()
        assert isinstance(stats, TricklePublisherStats)
        assert "TricklePublisherStats(" in str(stats)
        payload = asdict(stats)
        assert "post_attempts" in payload
        assert "terminal_error" in payload

    def test_subscriber_stats_are_typed_and_stringable(self) -> None:
        subscriber = TrickleSubscriber("http://example.test/trickle")
        stats = subscriber.get_stats()
        assert isinstance(stats, TrickleSubscriberStats)
        assert "TrickleSubscriberStats(" in str(stats)
        payload = asdict(stats)
        assert "get_attempts" in payload
        assert "segments_delivered" in payload
        assert "latest_seq" in payload

    def test_media_publish_stats_include_nested_publisher(self) -> None:
        media_publish = MediaPublish(
            "http://example.test/trickle", config=MediaPublishConfig()
        )
        stats = media_publish.get_stats()
        assert isinstance(stats, MediaPublishStats)
        assert isinstance(stats.publisher, TricklePublisherStats)
        assert "MediaPublishStats(" in str(stats)
        payload = asdict(stats)
        assert "publisher" in payload
        assert "segments_started" in payload

    def test_media_output_stats_include_optional_subscriber(self) -> None:
        media_output = MediaOutput("http://example.test/trickle")
        stats = media_output.get_stats()
        assert isinstance(stats, MediaOutputStats)
        assert stats.decoder is None
        assert stats.subscriber is None
        assert "MediaOutputStats(" in str(stats)
        payload = asdict(stats)
        assert "packet_errors" in payload
        assert "decode_errors" in payload
        assert "decoder" in payload
        assert "subscriber" in payload

    def test_live_video_to_video_media_output_passes_callbacks_through(self) -> None:
        def _on_frame(_frame) -> None:
            return None

        def _on_packet(_packet) -> None:
            return None

        job = LiveVideoToVideo(raw={}, subscribe_url="http://example.test/trickle")

        media_output = job.media_output(on_frame=_on_frame, on_packet=_on_packet)

        assert media_output.on_frame is _on_frame
        assert media_output.on_packet is _on_packet

    def test_media_output_bytes_accepts_video_mpegts_content_type(self) -> None:
        media_output = self._make_media_output_for_segments(
            _FakeSegment("video/mp2t", [b"video-bytes"])
        )

        chunks = asyncio.run(_collect_bytes(media_output))

        assert chunks == [b"video-bytes"]
        assert media_output.get_stats().content_type_errors == 0

    def test_media_output_bytes_accepts_audio_mpegts_content_type(self) -> None:
        media_output = self._make_media_output_for_segments(
            _FakeSegment("audio/mp2t", [b"audio-bytes"])
        )

        chunks = asyncio.run(_collect_bytes(media_output))

        assert chunks == [b"audio-bytes"]
        assert media_output.get_stats().content_type_errors == 0

    def test_media_output_bytes_rejects_non_mpegts_content_type(self) -> None:
        media_output = self._make_media_output_for_segments(
            _FakeSegment("audio/aac", [b"not-ts"])
        )

        with pytest.raises(LivepeerGatewayError, match="Expected Content-Type in"):
            asyncio.run(_collect_bytes(media_output))

        assert media_output.get_stats().content_type_errors == 1

    def test_media_output_bytes_tolerates_empty_first_segment_without_content_type(
        self,
    ) -> None:
        media_output = self._make_media_output_for_segments(
            _FakeSegment(None, [b""]),
            _FakeSegment("video/mp2t", [b"video-bytes"]),
        )

        chunks = asyncio.run(_collect_bytes(media_output))

        assert chunks == [b"video-bytes"]
        assert media_output.get_stats().content_type_errors == 0

    def test_media_output_bytes_rejects_invalid_type_on_first_non_empty_segment(
        self,
    ) -> None:
        media_output = self._make_media_output_for_segments(
            _FakeSegment(None, [b""]),
            _FakeSegment("audio/aac", [b"not-ts"]),
        )

        with pytest.raises(LivepeerGatewayError, match="Expected Content-Type in"):
            asyncio.run(_collect_bytes(media_output))

        assert media_output.get_stats().content_type_errors == 1

    def test_media_output_frames_decodes_audio_only_ts_without_content_type_errors(
        self,
    ) -> None:
        audio_frame = AudioDecodedMediaFrame(
            kind="audio",
            stream_index=0,
            frame=object(),
            pts=123,
            time_base=None,
            pts_time=1.23,
            demuxed_at=10.0,
            decoded_at=10.1,
            sample_rate=48000,
            layout="stereo",
            format="fltp",
            samples=1024,
        )
        media_output = self._make_media_output_for_segments(
            _FakeSegment("audio/mp2t", [b"audio-ts-payload"])
        )
        original_decoder = media_output_mod.MpegTsDecoder
        media_output_mod.MpegTsDecoder = lambda: _FakeDecoder(
            [audio_frame, media_decode_mod._END]
        )  # type: ignore[assignment]
        try:
            frames = asyncio.run(_collect_frames(media_output))
        finally:
            media_output_mod.MpegTsDecoder = original_decoder  # type: ignore[assignment]

        assert frames == [audio_frame]
        stats = media_output.get_stats()
        assert stats.audio_frames_decoded == 1
        assert stats.video_frames_decoded == 0
        assert stats.content_type_errors == 0

    def test_media_output_packets_yield_demuxed_packets_and_track_stats(self) -> None:
        video_packet = _fake_demuxed_packet(
            kind="video",
            stream_index=0,
            pts=100,
            pts_time=1.0,
            is_keyframe=True,
            size=512,
        )
        audio_packet = _fake_demuxed_packet(
            kind="audio",
            stream_index=1,
            pts=200,
            pts_time=2.0,
            size=256,
        )
        data_packet = _fake_demuxed_packet(
            kind="data",
            stream_index=2,
            pts=None,
            pts_time=None,
            size=64,
        )
        media_output = self._make_media_output_for_segments(
            _FakeSegment("video/mp2t", [b"packet-ts-payload"])
        )
        original_demuxer = media_output_mod.MpegTsPacketDemuxer
        media_output_mod.MpegTsPacketDemuxer = lambda: _FakePacketDemuxer(  # type: ignore[assignment]
            [video_packet, audio_packet, data_packet, media_decode_mod._END]
        )
        try:
            packets = asyncio.run(_collect_packets(media_output))
        finally:
            media_output_mod.MpegTsPacketDemuxer = original_demuxer  # type: ignore[assignment]

        assert packets == [video_packet, audio_packet, data_packet]
        stats = media_output.get_stats()
        assert stats.video_packets_demuxed == 1
        assert stats.audio_packets_demuxed == 1
        assert stats.other_packets_demuxed == 1
        assert stats.packet_errors == 0
        assert stats.content_type_errors == 0

    def test_media_output_packets_reject_invalid_type_on_first_non_empty_segment(
        self,
    ) -> None:
        media_output = self._make_media_output_for_segments(
            _FakeSegment(None, [b""]),
            _FakeSegment("audio/aac", [b"not-ts"]),
        )
        original_demuxer = media_output_mod.MpegTsPacketDemuxer
        media_output_mod.MpegTsPacketDemuxer = lambda: _FakePacketDemuxer(
            [media_decode_mod._END]
        )  # type: ignore[assignment]
        try:
            with pytest.raises(LivepeerGatewayError, match="Expected Content-Type in"):
                asyncio.run(_collect_packets(media_output))
        finally:
            media_output_mod.MpegTsPacketDemuxer = original_demuxer  # type: ignore[assignment]

        assert media_output.get_stats().content_type_errors == 1

    def test_media_output_packets_surface_demux_errors(self) -> None:
        media_output = self._make_media_output_for_segments(
            _FakeSegment("video/mp2t", [b"packet-ts-payload"])
        )
        original_demuxer = media_output_mod.MpegTsPacketDemuxer
        media_output_mod.MpegTsPacketDemuxer = lambda: _FakePacketDemuxer(  # type: ignore[assignment]
            [media_decode_mod._DecoderError(RuntimeError("demux boom"))]
        )
        try:
            with pytest.raises(LivepeerGatewayError, match="Media demux error"):
                asyncio.run(_collect_packets(media_output))
        finally:
            media_output_mod.MpegTsPacketDemuxer = original_demuxer  # type: ignore[assignment]

        assert media_output.get_stats().packet_errors == 1

    def test_media_output_packets_cleanup_stops_and_joins_demuxer(self) -> None:
        media_output = self._make_media_output_for_segments(
            _FakeSegment("video/mp2t", [b"packet-ts-payload"])
        )
        _TrackingPacketDemuxer.instances.clear()
        original_demuxer = media_output_mod.MpegTsPacketDemuxer
        media_output_mod.MpegTsPacketDemuxer = lambda: _TrackingPacketDemuxer(  # type: ignore[assignment]
            [media_decode_mod._END]
        )
        try:
            packets = asyncio.run(_collect_packets(media_output))
        finally:
            media_output_mod.MpegTsPacketDemuxer = original_demuxer  # type: ignore[assignment]

        assert packets == []
        assert len(_TrackingPacketDemuxer.instances) == 1
        demuxer = _TrackingPacketDemuxer.instances[0]
        assert demuxer.started
        assert demuxer.closed
        assert demuxer.stopped
        assert demuxer.joined

    def test_media_output_on_frame_starts_background_consumer_in_running_loop(
        self,
    ) -> None:
        audio_frame = AudioDecodedMediaFrame(
            kind="audio",
            stream_index=0,
            frame=object(),
            pts=123,
            time_base=None,
            pts_time=1.23,
            demuxed_at=10.0,
            decoded_at=10.1,
            sample_rate=48000,
            layout="stereo",
            format="fltp",
            samples=1024,
        )

        async def _run() -> tuple[
            list[AudioDecodedMediaFrame], tuple[asyncio.Task[None], ...]
        ]:
            seen: list[AudioDecodedMediaFrame] = []
            got_frame = asyncio.Event()

            def _on_frame(frame) -> None:
                seen.append(frame)
                got_frame.set()

            media_output = self._make_media_output_for_segments(
                _FakeSegment("audio/mp2t", [b"audio-ts-payload"]),
                on_frame=_on_frame,
            )
            await asyncio.wait_for(got_frame.wait(), timeout=1.0)
            tasks = media_output.callback_tasks()
            await media_output.close()
            return seen, tasks

        original_decoder = media_output_mod.MpegTsDecoder
        media_output_mod.MpegTsDecoder = lambda: _FakeDecoder(
            [audio_frame, media_decode_mod._END]
        )  # type: ignore[assignment]
        try:
            seen, tasks = asyncio.run(_run())
        finally:
            media_output_mod.MpegTsDecoder = original_decoder  # type: ignore[assignment]

        assert seen == [audio_frame]
        assert len(tasks) == 1

    def test_media_output_on_packet_starts_background_consumer_in_running_loop(
        self,
    ) -> None:
        packet = _fake_demuxed_packet(
            kind="video",
            stream_index=0,
            pts=100,
            pts_time=1.0,
            size=512,
        )

        async def _run() -> tuple[
            list[DemuxedMediaPacket], tuple[asyncio.Task[None], ...]
        ]:
            seen: list[DemuxedMediaPacket] = []
            got_packet = asyncio.Event()

            def _on_packet(item) -> None:
                seen.append(item)
                got_packet.set()

            media_output = self._make_media_output_for_segments(
                _FakeSegment("video/mp2t", [b"packet-ts-payload"]),
                on_packet=_on_packet,
            )
            await asyncio.wait_for(got_packet.wait(), timeout=1.0)
            tasks = media_output.callback_tasks()
            await media_output.close()
            return seen, tasks

        original_demuxer = media_output_mod.MpegTsPacketDemuxer
        media_output_mod.MpegTsPacketDemuxer = lambda: _FakePacketDemuxer(  # type: ignore[assignment]
            [packet, media_decode_mod._END]
        )
        try:
            seen, tasks = asyncio.run(_run())
        finally:
            media_output_mod.MpegTsPacketDemuxer = original_demuxer  # type: ignore[assignment]

        assert seen == [packet]
        assert len(tasks) == 1

    def test_media_output_on_bytes_starts_background_consumer_in_running_loop(
        self,
    ) -> None:
        async def _run() -> tuple[list[bytes], tuple[asyncio.Task[None], ...]]:
            seen: list[bytes] = []
            media_output = self._make_media_output_for_segments(
                _FakeSegment("video/mp2t", [b"abc", b"def"]),
                on_bytes=seen.append,
            )
            await media_output.wait_callbacks(timeout=1.0)
            tasks = media_output.callback_tasks()
            await media_output.close()
            return seen, tasks

        seen, tasks = asyncio.run(_run())

        assert seen == [b"abc", b"def"]
        assert len(tasks) == 1

    def test_media_output_callbacks_can_start_later_from_async_context(self) -> None:
        audio_frame = AudioDecodedMediaFrame(
            kind="audio",
            stream_index=0,
            frame=object(),
            pts=123,
            time_base=None,
            pts_time=1.23,
            demuxed_at=10.0,
            decoded_at=10.1,
            sample_rate=48000,
            layout="stereo",
            format="fltp",
            samples=1024,
        )
        seen: list[AudioDecodedMediaFrame] = []

        media_output = self._make_media_output_for_segments(
            _FakeSegment("audio/mp2t", [b"audio-ts-payload"]),
            on_frame=seen.append,
        )
        assert media_output.callback_tasks() == ()

        async def _run() -> None:
            async with media_output:
                while not seen:
                    await asyncio.sleep(0)

        original_decoder = media_output_mod.MpegTsDecoder
        media_output_mod.MpegTsDecoder = lambda: _FakeDecoder(
            [audio_frame, media_decode_mod._END]
        )  # type: ignore[assignment]
        try:
            asyncio.run(_run())
        finally:
            media_output_mod.MpegTsDecoder = original_decoder  # type: ignore[assignment]

        assert seen == [audio_frame]

    def test_media_output_async_frame_callback_is_awaited(self) -> None:
        audio_frame = AudioDecodedMediaFrame(
            kind="audio",
            stream_index=0,
            frame=object(),
            pts=123,
            time_base=None,
            pts_time=1.23,
            demuxed_at=10.0,
            decoded_at=10.1,
            sample_rate=48000,
            layout="stereo",
            format="fltp",
            samples=1024,
        )

        async def _run() -> list[str]:
            order: list[str] = []
            callback_done = asyncio.Event()

            async def _on_frame(_frame) -> None:
                order.append("start")
                await asyncio.sleep(0)
                order.append("done")
                callback_done.set()

            media_output = self._make_media_output_for_segments(
                _FakeSegment("audio/mp2t", [b"audio-ts-payload"]),
                on_frame=_on_frame,
            )
            await asyncio.wait_for(callback_done.wait(), timeout=1.0)
            await media_output.close()
            return order

        original_decoder = media_output_mod.MpegTsDecoder
        media_output_mod.MpegTsDecoder = lambda: _FakeDecoder(
            [audio_frame, media_decode_mod._END]
        )  # type: ignore[assignment]
        try:
            order = asyncio.run(_run())
        finally:
            media_output_mod.MpegTsDecoder = original_decoder  # type: ignore[assignment]

        assert order == ["start", "done"]

    def test_media_output_async_bytes_callback_is_awaited(self) -> None:
        async def _run() -> list[str]:
            order: list[str] = []

            async def _on_bytes(_chunk) -> None:
                order.append("start")
                await asyncio.sleep(0)
                order.append("done")

            media_output = self._make_media_output_for_segments(
                _FakeSegment("video/mp2t", [b"payload"]),
                on_bytes=_on_bytes,
            )
            await media_output.wait_callbacks(timeout=1.0)
            await media_output.close()
            return order

        assert asyncio.run(_run()) == ["start", "done"]

    def test_media_output_close_waits_for_bytes_callback_to_finish(self) -> None:
        async def _run() -> list[bytes]:
            seen: list[bytes] = []
            media_output = self._make_media_output_for_segments(
                _FakeSegment("video/mp2t", [b"payload"]),
                on_bytes=seen.append,
            )
            await media_output.close(timeout=1.0)
            return seen

        assert asyncio.run(_run()) == [b"payload"]

    def test_media_output_close_timeout_zero_cancels_callback_immediately(self) -> None:
        async def _run() -> tuple[bool, bool]:
            started = asyncio.Event()
            release = asyncio.Event()

            async def _on_bytes(_chunk) -> None:
                started.set()
                await release.wait()

            media_output = self._make_media_output_for_segments(
                _FakeSegment("video/mp2t", [b"payload"]),
                on_bytes=_on_bytes,
            )
            await asyncio.wait_for(started.wait(), timeout=1.0)
            task = media_output.callback_tasks()[0]
            await media_output.close(timeout=0)
            return task.cancelled(), release.is_set()

        cancelled, released = asyncio.run(_run())
        assert cancelled
        assert not released

    def test_media_output_wait_callbacks_returns_empty_without_callbacks(self) -> None:
        async def _run() -> tuple[object, ...]:
            media_output = self._make_media_output_for_segments(
                _FakeSegment("video/mp2t", [b"payload"]),
            )
            return await media_output.wait_callbacks(timeout=1.0)

        assert asyncio.run(_run()) == ()

    def test_media_output_bytes_callback_exception_raises_from_wait_and_close(
        self,
    ) -> None:
        async def _run() -> None:
            callback_called = asyncio.Event()

            def _on_bytes(_chunk) -> None:
                callback_called.set()
                raise RuntimeError("bytes callback boom")

            media_output = self._make_media_output_for_segments(
                _FakeSegment("video/mp2t", [b"payload"]),
                on_bytes=_on_bytes,
            )
            await asyncio.wait_for(callback_called.wait(), timeout=1.0)
            with pytest.raises(RuntimeError, match="bytes callback boom"):
                await media_output.wait_callbacks(timeout=1.0)
            with pytest.raises(RuntimeError, match="bytes callback boom"):
                await media_output.close()

        asyncio.run(_run())

    def test_media_output_callback_exception_stops_loop_and_close_raises(self) -> None:
        audio_frame = AudioDecodedMediaFrame(
            kind="audio",
            stream_index=0,
            frame=object(),
            pts=123,
            time_base=None,
            pts_time=1.23,
            demuxed_at=10.0,
            decoded_at=10.1,
            sample_rate=48000,
            layout="stereo",
            format="fltp",
            samples=1024,
        )

        async def _run() -> None:
            callback_called = asyncio.Event()

            def _on_frame(_frame) -> None:
                callback_called.set()
                raise RuntimeError("frame callback boom")

            media_output = self._make_media_output_for_segments(
                _FakeSegment("audio/mp2t", [b"audio-ts-payload"]),
                on_frame=_on_frame,
            )
            await asyncio.wait_for(callback_called.wait(), timeout=1.0)
            task = media_output.callback_tasks()[0]
            while not task.done():
                await asyncio.sleep(0)
            with pytest.raises(RuntimeError, match="frame callback boom"):
                await media_output.close()

        _TrackingDecoder.instances.clear()
        original_decoder = media_output_mod.MpegTsDecoder
        media_output_mod.MpegTsDecoder = lambda: _TrackingDecoder(  # type: ignore[assignment]
            [audio_frame, media_decode_mod._END]
        )
        try:
            asyncio.run(_run())
        finally:
            media_output_mod.MpegTsDecoder = original_decoder  # type: ignore[assignment]

        assert len(_TrackingDecoder.instances) == 1
        decoder = _TrackingDecoder.instances[0]
        assert decoder.stopped
        assert decoder.joined

    def test_media_output_one_callback_failure_does_not_stop_other_callback_task(
        self,
    ) -> None:
        audio_frame = AudioDecodedMediaFrame(
            kind="audio",
            stream_index=0,
            frame=object(),
            pts=123,
            time_base=None,
            pts_time=1.23,
            demuxed_at=10.0,
            decoded_at=10.1,
            sample_rate=48000,
            layout="stereo",
            format="fltp",
            samples=1024,
        )
        packet = _fake_demuxed_packet(
            kind="audio",
            stream_index=0,
            pts=123,
            pts_time=1.23,
            size=188,
        )

        async def _run() -> list[DemuxedMediaPacket]:
            packet_seen: list[DemuxedMediaPacket] = []
            frame_failed = asyncio.Event()
            packet_called = asyncio.Event()

            def _on_frame(_frame) -> None:
                frame_failed.set()
                raise RuntimeError("frame callback boom")

            def _on_packet(item) -> None:
                packet_seen.append(item)
                packet_called.set()

            media_output = self._make_media_output_for_segments(
                _FakeSegment("video/mp2t", [b"shared-ts-payload"]),
                on_frame=_on_frame,
                on_packet=_on_packet,
            )
            await asyncio.wait_for(frame_failed.wait(), timeout=1.0)
            await asyncio.wait_for(packet_called.wait(), timeout=1.0)
            with pytest.raises(RuntimeError, match="frame callback boom"):
                await media_output.close()
            return packet_seen

        original_decoder = media_output_mod.MpegTsDecoder
        original_demuxer = media_output_mod.MpegTsPacketDemuxer
        media_output_mod.MpegTsDecoder = lambda: _FakeDecoder(
            [audio_frame, media_decode_mod._END]
        )  # type: ignore[assignment]
        media_output_mod.MpegTsPacketDemuxer = lambda: _FakePacketDemuxer(  # type: ignore[assignment]
            [packet, media_decode_mod._END]
        )
        try:
            packet_seen = asyncio.run(_run())
        finally:
            media_output_mod.MpegTsDecoder = original_decoder  # type: ignore[assignment]
            media_output_mod.MpegTsPacketDemuxer = original_demuxer  # type: ignore[assignment]

        assert packet_seen == [packet]

    def test_media_output_stats_str_includes_decoder_and_subscriber_when_present(
        self,
    ) -> None:
        decoder_stats = DecoderQueueStats(
            queued_chunks=2,
            queued_bytes=1024,
            buffered_bytes=256,
            total_chunks_dequeued=5,
            total_bytes_dequeued=4096,
            total_bytes_read=3840,
            output_items_queued=1,
            total_output_items_dequeued=8,
            output_wait_s=0.75,
            queue_s=0.5,
            processed_s=4.25,
        )
        subscriber_stats = TrickleSubscriberStats(
            elapsed_s=1.2,
            get_attempts=3,
            get_retries=1,
            get_404_eos=0,
            get_470_reset=1,
            get_failures=0,
            segments_delivered=2,
            seq_gap_events=0,
            wait_ms_total=45,
            latest_seq=9,
        )
        stats = MediaOutputStats(
            elapsed_s=2.3,
            segments_consumed=2,
            bytes_read=1024,
            chunks_read=4,
            content_type_errors=0,
            segment_read_errors=0,
            segment_max_bytes_exceeded=0,
            consumer_lag_skip_latest=0,
            consumer_lag_retry_earliest=0,
            consumer_lag_fail=0,
            video_packets_demuxed=4,
            audio_packets_demuxed=2,
            other_packets_demuxed=1,
            video_frames_decoded=10,
            audio_frames_decoded=0,
            packet_errors=1,
            decode_errors=0,
            decoder=decoder_stats,
            subscriber=subscriber_stats,
        )
        rendered = str(stats)
        assert "decoder=DecoderQueueStats(" in rendered
        assert "queued_bytes=1024" in rendered
        assert "video_packets_demuxed=4" in rendered
        assert "packet_errors=1" in rendered
        assert "subscriber=TrickleSubscriberStats(" in rendered
        assert "latest_seq=9" in rendered

    def test_blocking_byte_stream_tracks_queue_and_buffer_metrics(self) -> None:
        stream = BlockingByteStream()

        stream.feed(b"abcdef")
        stream.feed(b"ghi")
        stats = stream.get_stats()
        assert stats.queued_chunks == 2
        assert stats.queued_bytes == 9
        assert stats.buffered_bytes == 0
        assert stats.total_chunks_dequeued == 0
        assert stats.total_bytes_dequeued == 0
        assert stats.total_bytes_read == 0
        assert stats.output_wait_s == 0.0

        assert stream.read(4) == b"abcd"
        stats = stream.get_stats()
        assert stats.queued_chunks == 1
        assert stats.queued_bytes == 3
        assert stats.buffered_bytes == 2
        assert stats.total_chunks_dequeued == 1
        assert stats.total_bytes_dequeued == 6
        assert stats.total_bytes_read == 4

        assert stream.read(10) == b"ef"
        stats = stream.get_stats()
        assert stats.queued_chunks == 1
        assert stats.queued_bytes == 3
        assert stats.buffered_bytes == 0
        assert stats.total_chunks_dequeued == 1
        assert stats.total_bytes_dequeued == 6
        assert stats.total_bytes_read == 6

        assert stream.read(10) == b"ghi"
        stats = stream.get_stats()
        assert stats.queued_chunks == 0
        assert stats.queued_bytes == 0
        assert stats.buffered_bytes == 0
        assert stats.total_chunks_dequeued == 2
        assert stats.total_bytes_dequeued == 9
        assert stats.total_bytes_read == 9

    def test_decoder_output_metrics_track_items_removed_by_frames(self) -> None:
        decoder = MpegTsDecoder()
        decoder._put_output_item(object())
        decoder._put_output_item(object())

        stats = decoder.get_stats()
        assert stats.output_items_queued == 2
        assert stats.total_output_items_dequeued == 0
        assert stats.output_wait_s == 0.0

        decoder.get()
        stats = decoder.get_stats()
        assert stats.output_items_queued == 1
        assert stats.total_output_items_dequeued == 1
        assert stats.output_wait_s >= 0.0

        decoder.get()
        stats = decoder.get_stats()
        assert stats.output_items_queued == 0
        assert stats.total_output_items_dequeued == 2
        assert stats.output_wait_s >= 0.0

    def test_decoder_output_wait_metrics_accumulate_blocked_get_time(self) -> None:
        decoder = MpegTsDecoder()

        def _put_later() -> None:
            time.sleep(0.03)
            decoder._put_output_item(object())

        producer = threading.Thread(target=_put_later, daemon=True)
        producer.start()
        got = decoder.get()
        producer.join()

        assert got is not None
        stats = decoder.get_stats()
        assert stats.output_wait_s >= 0.02
        assert stats.output_wait_s < 0.25

    def test_decoder_output_wait_metrics_stay_zero_for_immediate_get(self) -> None:
        decoder = MpegTsDecoder()
        decoder._put_output_item(object())

        with mock.patch.object(
            media_decode_mod.time,
            "monotonic",
            side_effect=[10.0, 10.0],
        ):
            decoder.get()

        stats = decoder.get_stats()
        assert stats.output_items_queued == 0
        assert stats.output_wait_s == 0.0

    def test_media_output_packets_and_frames_share_one_underlying_subscriber(
        self,
    ) -> None:
        audio_frame = AudioDecodedMediaFrame(
            kind="audio",
            stream_index=0,
            frame=object(),
            pts=123,
            time_base=None,
            pts_time=1.23,
            demuxed_at=10.0,
            decoded_at=10.1,
            sample_rate=48000,
            layout="stereo",
            format="fltp",
            samples=1024,
        )
        packet = _fake_demuxed_packet(
            kind="audio",
            stream_index=0,
            pts=123,
            pts_time=1.23,
            size=188,
        )

        class _FakeSubscriber:
            init_count = 0

            def __init__(self, *_args, **_kwargs) -> None:
                _FakeSubscriber.init_count += 1
                self._segments = [_FakeSegment("video/mp2t", [b"shared-ts-payload"])]

            async def next(self):
                if self._segments:
                    return self._segments.pop(0)
                return None

            async def close(self) -> None:
                return None

            def get_stats(self):
                return None

        async def _run() -> tuple[
            list[DemuxedMediaPacket], list[AudioDecodedMediaFrame]
        ]:
            media_output = MediaOutput("http://example.test/trickle")
            packets_task = asyncio.create_task(_collect_packets(media_output))
            frames_task = asyncio.create_task(_collect_frames(media_output))
            return await asyncio.gather(packets_task, frames_task)

        original_subscriber = media_output_mod.TrickleSubscriber
        original_demuxer = media_output_mod.MpegTsPacketDemuxer
        original_decoder = media_output_mod.MpegTsDecoder
        media_output_mod.TrickleSubscriber = _FakeSubscriber  # type: ignore[assignment]
        media_output_mod.MpegTsPacketDemuxer = lambda: _FakePacketDemuxer(  # type: ignore[assignment]
            [packet, media_decode_mod._END]
        )
        media_output_mod.MpegTsDecoder = lambda: _FakeDecoder(
            [audio_frame, media_decode_mod._END]
        )  # type: ignore[assignment]
        try:
            packets, frames = asyncio.run(_run())
        finally:
            media_output_mod.TrickleSubscriber = original_subscriber  # type: ignore[assignment]
            media_output_mod.MpegTsPacketDemuxer = original_demuxer  # type: ignore[assignment]
            media_output_mod.MpegTsDecoder = original_decoder  # type: ignore[assignment]

        assert _FakeSubscriber.init_count == 1
        assert packets == [packet]
        assert frames == [audio_frame]

    def test_subscriber_470_latest_seq_uses_header(self) -> None:
        class _Resp:
            def __init__(self, status: int, headers: dict[str, str]) -> None:
                self.status = status
                self.headers = headers

            async def text(self) -> str:
                return ""

            def release(self) -> None:
                return None

        class _Session:
            def __init__(self, responses: list[_Resp]) -> None:
                self._responses = list(responses)

            async def get(self, *_args, **_kwargs):
                return self._responses.pop(0)

        subscriber = TrickleSubscriber(
            "http://example.test/trickle",
            start_seq=7,
            max_retries=2,
        )
        subscriber._session = _Session(  # type: ignore[assignment]
            [
                _Resp(470, {"Lp-Trickle-Latest": "11"}),
                _Resp(404, {}),
            ]
        )
        asyncio.run(subscriber._preconnect())
        stats = subscriber.get_stats()
        assert stats.latest_seq == 11
        assert subscriber._seq == 11

    def test_subscriber_470_ahead_of_edge_retries_same_seq(self) -> None:
        class _Resp:
            def __init__(self, status: int, headers: dict[str, str]) -> None:
                self.status = status
                self.headers = headers

            async def text(self) -> str:
                return ""

            def release(self) -> None:
                return None

        class _Session:
            def __init__(self, responses: list[_Resp]) -> None:
                self._responses = list(responses)
                self.urls: list[str] = []

            async def get(self, url: str, *_args, **_kwargs):
                self.urls.append(url)
                return self._responses.pop(0)

        subscriber = TrickleSubscriber(
            "http://example.test/trickle",
            start_seq=12,
            max_retries=2,
        )
        session = _Session(
            [
                _Resp(470, {"Lp-Trickle-Latest": "11"}),
                _Resp(404, {}),
            ]
        )
        subscriber._session = session  # type: ignore[assignment]
        with mock.patch(
            "livepeer_gateway.trickle_subscriber.asyncio.sleep",
            new=mock.AsyncMock(),
        ) as sleep_mock:
            asyncio.run(subscriber._preconnect())
        stats = subscriber.get_stats()
        assert stats.latest_seq == 11
        assert subscriber._seq == 12
        sleep_mock.assert_any_await(0.25)
        assert session.urls == [
            "http://example.test/trickle/12",
            "http://example.test/trickle/12",
        ]

    def test_subscriber_470_latest_seq_falls_back_to_current_seq(self) -> None:
        class _Resp:
            def __init__(self, status: int, headers: dict[str, str]) -> None:
                self.status = status
                self.headers = headers

            async def text(self) -> str:
                return ""

            def release(self) -> None:
                return None

        class _Session:
            def __init__(self, responses: list[_Resp]) -> None:
                self._responses = list(responses)

            async def get(self, *_args, **_kwargs):
                return self._responses.pop(0)

        subscriber = TrickleSubscriber(
            "http://example.test/trickle",
            start_seq=7,
            max_retries=2,
        )
        subscriber._session = _Session(  # type: ignore[assignment]
            [
                _Resp(470, {}),
                _Resp(404, {}),
            ]
        )
        asyncio.run(subscriber._preconnect())
        stats = subscriber.get_stats()
        assert stats.latest_seq == 7
        assert subscriber._seq == 7

    def test_segment_reader_stats_dataclass_supports_asdict_and_str(self) -> None:
        stats = SegmentReaderStats(
            chunks_read=3,
            bytes_read=4096,
            read_errors=1,
            max_bytes_exceeded=0,
            segment_seq=7,
        )
        assert "SegmentReaderStats(" in str(stats)
        payload = asdict(stats)
        assert payload["segment_seq"] == 7
        assert payload["bytes_read"] == 4096

    def test_summary_logging_helpers_removed(self) -> None:
        assert not hasattr(MediaPublish, "_maybe_log_publish_summary")
        assert not hasattr(MediaPublish, "_log_publish_summary")
        assert not hasattr(MediaOutput, "_maybe_log_summary")
        assert not hasattr(MediaOutput, "_log_summary")

    def test_frame_queue_tracks_queue_and_processed_media_time_fifo(self) -> None:
        from fractions import Fraction

        class _Frame:
            def __init__(self, pts: int) -> None:
                self.pts = pts
                self.time_base = Fraction(1, 1000)

        stats = _new_track_stats()
        q = FrameQueue(maxsize=8, stats=stats)

        q.put(_Frame(0))
        q.put(_Frame(500))
        q.put(_Frame(1500))
        # After enqueues only: nothing has been dequeued yet so queue span is
        # reported as 0 until the consumer side has a watermark to subtract.
        assert q.queue_media_time_s == 0.0
        assert q.total_media_time_processed_s == 0.0

        q.get()  # pts=0 -> first/last_get = 0.0
        assert q.queue_media_time_s == pytest.approx(1.5)
        assert q.total_media_time_processed_s == pytest.approx(0.0)

        q.get()  # pts=500 -> last_get = 0.5
        assert q.queue_media_time_s == pytest.approx(1.0)
        assert q.total_media_time_processed_s == pytest.approx(0.5)

        q.get()  # pts=1500 -> last_get = 1.5
        assert q.queue_media_time_s == pytest.approx(0.0)
        assert q.total_media_time_processed_s == pytest.approx(1.5)

    def test_frame_queue_overflow_drops_advance_consumed_watermark(self) -> None:
        from fractions import Fraction

        class _Frame:
            def __init__(self, pts: int) -> None:
                self.pts = pts
                self.time_base = Fraction(1, 1000)

        stats = _new_track_stats()
        q = FrameQueue(maxsize=2, stats=stats)

        q.put(_Frame(0))
        q.put(_Frame(500))
        q.put(_Frame(1000))  # overflow: drops pts=0 from head
        assert stats["frames_dropped_overflow"] == 1
        # One dropped frame counts as a "get" for watermark purposes.
        assert q.total_media_time_processed_s == pytest.approx(0.0)
        # Now last_put=1.0, last_get=0.0 -> queue span = 1.0.
        assert q.queue_media_time_s == pytest.approx(1.0)

        q.get()  # accepted pts=500 -> last_get = 0.5
        assert q.total_media_time_processed_s == pytest.approx(0.5)
        assert q.queue_media_time_s == pytest.approx(0.5)

    def test_frame_queue_debt_skip_tracks_dropped_and_accepted(self) -> None:
        from fractions import Fraction

        class _Frame:
            def __init__(self, pts: int) -> None:
                self.pts = pts
                self.time_base = Fraction(1, 1000)

        stats = _new_track_stats()
        q = FrameQueue(maxsize=8, stats=stats, debt_skip=True)

        q.put(_Frame(0))
        q.put(_Frame(500))
        q.put(_Frame(1500))

        # Seed debt so the first candidate gets skipped. Call
        # update_after_encode twice so _last_encoded_media_time_s is not None
        # and we accrue real debt.
        q.update_after_encode(encoded_media_time_s=0.0, encode_duration_s=0.0)
        q.update_after_encode(encoded_media_time_s=0.01, encode_duration_s=1.0)
        assert q.time_debt_s > 0.0

        got = q.get()
        # Debt-skip path should have moved past pts=0 (and maybe pts=500) to a
        # frame that advances media time enough. Either way, the tracked
        # watermarks should cover every frame that left the queue.
        assert got is not None
        assert q.total_media_time_processed_s == pytest.approx(1.5)
        assert q.queue_media_time_s == pytest.approx(0.0)

    def test_decoder_output_media_time_metrics(self) -> None:
        decoder = MpegTsDecoder()

        def _frame(pts_time: float) -> AudioDecodedMediaFrame:
            return AudioDecodedMediaFrame(
                kind="audio",
                stream_index=0,
                frame=object(),
                pts=0,
                time_base=None,
                pts_time=pts_time,
                demuxed_at=0.0,
                decoded_at=0.0,
                sample_rate=48000,
                layout="mono",
                format="fltp",
                samples=1024,
            )

        decoder._put_output_item(_frame(0.25))
        decoder._put_output_item(_frame(0.50))
        decoder._put_output_item(_frame(1.75))

        stats = decoder.get_stats()
        # Nothing dequeued yet; both metrics remain 0 because there's no
        # consumer watermark.
        assert stats.output_wait_s == 0.0
        assert stats.queue_s == 0.0
        assert stats.processed_s == 0.0

        decoder.get()  # 0.25
        stats = decoder.get_stats()
        assert stats.queue_s == pytest.approx(1.5)
        assert stats.processed_s == pytest.approx(0.0)

        decoder.get()  # 0.50
        decoder.get()  # 1.75
        stats = decoder.get_stats()
        assert stats.queue_s == pytest.approx(0.0)
        assert stats.processed_s == pytest.approx(1.5)

    def test_decoder_ignores_non_frame_items_for_media_time(self) -> None:
        decoder = MpegTsDecoder()

        decoder._put_output_item(object())
        decoder._put_output_item(object())

        stats = decoder.get_stats()
        assert stats.output_items_queued == 2
        assert stats.queue_s == 0.0
        assert stats.processed_s == 0.0

        decoder.get()
        stats = decoder.get_stats()
        assert stats.output_wait_s >= 0.0
        assert stats.queue_s == 0.0
        assert stats.processed_s == 0.0
