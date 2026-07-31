from __future__ import annotations

import asyncio
import io
import os
import threading
from unittest import mock

import pytest

from livepeer_gateway import media_publish as media_publish_mod


class _Format:
    def __init__(self, name: str) -> None:
        self.name = name


class _Layout:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeVideoFrame:
    def __init__(
        self, *, width: int = 640, height: int = 360, fmt: str = "yuv420p"
    ) -> None:
        self.width = width
        self.height = height
        self.format = _Format(fmt)
        self.pts = None
        self.time_base = None
        self.pict_type = None

    def reformat(self, *, format: str) -> _FakeVideoFrame:
        self.format = _Format(format)
        return self


class _FakeAudioFrame:
    def __init__(
        self,
        *,
        sample_rate: int = 48_000,
        layout: str = "mono",
        fmt: str = "flt",
        samples: int = 960,
    ) -> None:
        self.sample_rate = sample_rate
        self.layout = _Layout(layout)
        self.format = _Format(fmt)
        self.samples = samples
        self.pts = None
        self.time_base = None


class _FakeStream:
    def __init__(self, *, codec: str, rate: int, kwargs: dict[str, object]) -> None:
        self.codec = codec
        self.rate = rate
        self.kwargs = kwargs
        self.time_base = None
        self.layout = None
        self.format = None

    def encode(self, _frame: object) -> list[object]:
        return []


class _FakeContainer:
    def __init__(self) -> None:
        self.added_streams: list[_FakeStream] = []

    def add_stream(
        self,
        codec: str,
        rate: int,
        options: dict[str, str] | None = None,
        **kwargs: object,
    ) -> _FakeStream:
        stream = _FakeStream(
            codec=codec, rate=rate, kwargs={"options": options, **kwargs}
        )
        self.added_streams.append(stream)
        return stream

    def mux(self, _packet: object) -> None:
        return None

    def close(self) -> None:
        return None


class _CloseOnlyContainer:
    def close(self) -> None:
        return None


class _FakeResampler:
    last_init: dict[str, object] | None = None

    def __init__(self, *, format: str, layout: str, rate: int) -> None:
        _FakeResampler.last_init = {
            "format": format,
            "layout": layout,
            "rate": rate,
        }

    def resample(self, frame: object) -> list[object]:
        if frame is None:
            return []
        return [frame]


class TestMediaPublishInit:
    @pytest.fixture(autouse=True)
    def fake_av_frames(self):
        with (
            mock.patch.object(media_publish_mod.av, "VideoFrame", _FakeVideoFrame),
            mock.patch.object(media_publish_mod.av, "AudioFrame", _FakeAudioFrame),
        ):
            yield

    def _build_media(self, *, timeout_s: float = 5.0) -> media_publish_mod.MediaPublish:
        config = media_publish_mod.MediaPublishConfig(
            tracks=[
                media_publish_mod.VideoOutputConfig(),
                media_publish_mod.AudioOutputConfig(),
            ],
            track_wait_timeout_s=timeout_s,
        )
        media = media_publish_mod.MediaPublish(
            "http://example.test/trickle", config=config
        )
        media._loop = object()  # bypass _open_container loop check in unit tests
        return media

    def test_four_track_publish_initializes_every_configured_stream(self) -> None:
        configs = [
            media_publish_mod.VideoOutputConfig(
                queue_size=2,
                fps=12,
                codec="video-codec-0",
            ),
            media_publish_mod.VideoOutputConfig(
                queue_size=3,
                fps=24,
                codec="video-codec-1",
            ),
            media_publish_mod.AudioOutputConfig(
                queue_size=4,
                codec="audio-codec-0",
                sample_rate=48_000,
                layout="mono",
                format="fltp",
            ),
            media_publish_mod.AudioOutputConfig(
                queue_size=5,
                codec="audio-codec-1",
                sample_rate=44_100,
                layout="stereo",
                format="flt",
            ),
        ]
        media = media_publish_mod.MediaPublish(
            "http://example.test/trickle",
            config=media_publish_mod.MediaPublishConfig(tracks=configs),
        )
        media._loop = object()

        video_tracks = media.get_tracks("video")
        audio_tracks = media.get_tracks("audio")
        assert len(media.tracks) == 4
        assert [track.index for track in video_tracks] == [0, 1]
        assert [track.index for track in audio_tracks] == [0, 1]
        assert [track._label for track in media.tracks] == [
            "video_0",
            "video_1",
            "audio_0",
            "audio_1",
        ]
        assert [track.config for track in media.tracks] == configs
        assert len({id(track._queue) for track in media.tracks}) == 4
        assert [track._queue.maxsize for track in media.tracks] == [2, 3, 4, 5]

        media._stage_frame_before_open(
            video_tracks[0], _FakeVideoFrame(width=160, height=90)
        )
        media._stage_frame_before_open(
            video_tracks[1], _FakeVideoFrame(width=320, height=180)
        )
        media._stage_frame_before_open(
            audio_tracks[0], _FakeAudioFrame(sample_rate=48_000, layout="mono")
        )
        media._stage_frame_before_open(
            audio_tracks[1], _FakeAudioFrame(sample_rate=44_100, layout="stereo")
        )
        assert media._can_open_container()

        fake_container = _FakeContainer()
        with mock.patch.object(
            media_publish_mod.av, "open", return_value=fake_container
        ):
            media._open_container()

        assert [stream.codec for stream in fake_container.added_streams] == [
            "video-codec-0",
            "video-codec-1",
            "audio-codec-0",
            "audio-codec-1",
        ]
        assert [stream.rate for stream in fake_container.added_streams] == [
            12,
            24,
            48_000,
            44_100,
        ]
        assert [stream.layout for stream in fake_container.added_streams[2:]] == [
            "mono",
            "stereo",
        ]
        assert [stream.format for stream in fake_container.added_streams[2:]] == [
            "fltp",
            "flt",
        ]

    def test_multi_track_writes_require_an_explicit_track_handle(self) -> None:
        media = media_publish_mod.MediaPublish(
            "http://example.test/trickle",
            config=media_publish_mod.MediaPublishConfig(
                tracks=[
                    media_publish_mod.VideoOutputConfig(),
                    media_publish_mod.VideoOutputConfig(),
                    media_publish_mod.AudioOutputConfig(),
                    media_publish_mod.AudioOutputConfig(),
                ]
            ),
        )
        video_frame = _FakeVideoFrame()
        audio_frame = _FakeAudioFrame()

        with pytest.raises(TypeError, match="ambiguous with multiple video tracks"):
            asyncio.run(media.write_frame(video_frame))
        with pytest.raises(TypeError, match="ambiguous with multiple audio tracks"):
            asyncio.run(media.write_frame(audio_frame))

        with mock.patch.object(
            media, "_write_frame_to_track", new_callable=mock.AsyncMock
        ) as write_to_track:
            selected_video = media.get_tracks("video")[1]
            asyncio.run(selected_video.write_frame(video_frame))
            write_to_track.assert_awaited_once_with(selected_video, video_frame)

            write_to_track.reset_mock()
            selected_audio = media.get_tracks("audio")[0]
            asyncio.run(selected_audio.write_frame(audio_frame))
            write_to_track.assert_awaited_once_with(selected_audio, audio_frame)

    def test_delayed_audio_arrives_before_timeout(self) -> None:
        media = self._build_media(timeout_s=5.0)
        video_track = media._tracks[0]
        audio_track = media._tracks[1]

        with mock.patch.object(media_publish_mod.time, "monotonic", return_value=100.0):
            media._stage_frame_before_open(video_track, _FakeVideoFrame())
        with mock.patch.object(media_publish_mod.time, "monotonic", return_value=102.0):
            assert not media._can_open_container()

        media._stage_frame_before_open(audio_track, _FakeAudioFrame())
        with mock.patch.object(media_publish_mod.time, "monotonic", return_value=102.1):
            assert media._can_open_container()

    def test_missing_audio_is_dropped_after_timeout(self) -> None:
        media = self._build_media(timeout_s=5.0)
        video_track = media._tracks[0]
        audio_track = media._tracks[1]
        fake_container = _FakeContainer()

        with mock.patch.object(media_publish_mod.time, "monotonic", return_value=10.0):
            media._stage_frame_before_open(video_track, _FakeVideoFrame())
        with mock.patch.object(media_publish_mod.time, "monotonic", return_value=14.0):
            assert not media._can_open_container()
        with mock.patch.object(media_publish_mod.time, "monotonic", return_value=16.0):
            assert media._can_open_container()

        assert audio_track._stopped
        assert audio_track._dropped_timeout
        assert audio_track._first_frame is None

        with mock.patch.object(
            media_publish_mod.av, "open", return_value=fake_container
        ):
            media._open_container()
        assert len(fake_container.added_streams) == 1
        assert fake_container.added_streams[0].codec == "libx264"

    def test_audio_only_publish_opens_on_first_audio_frame(self) -> None:
        config = media_publish_mod.MediaPublishConfig(
            tracks=[media_publish_mod.AudioOutputConfig()],
            track_wait_timeout_s=5.0,
        )
        media = media_publish_mod.MediaPublish(
            "http://example.test/trickle", config=config
        )
        media._loop = object()
        track = media._tracks[0]

        with mock.patch.object(media_publish_mod.time, "monotonic", return_value=50.0):
            media._stage_frame_before_open(
                track,
                _FakeAudioFrame(sample_rate=44_100, layout="stereo"),
            )
        with mock.patch.object(media_publish_mod.time, "monotonic", return_value=50.0):
            assert media._can_open_container()

    def test_audio_stream_uses_first_frame_properties_when_config_unset(self) -> None:
        config = media_publish_mod.MediaPublishConfig(
            tracks=[
                media_publish_mod.AudioOutputConfig(
                    format="flt",
                )
            ],
            track_wait_timeout_s=5.0,
        )
        media = media_publish_mod.MediaPublish(
            "http://example.test/trickle", config=config
        )
        media._loop = object()
        track = media._tracks[0]
        fake_container = _FakeContainer()

        with mock.patch.object(media_publish_mod.time, "monotonic", return_value=90.0):
            media._stage_frame_before_open(
                track,
                _FakeAudioFrame(sample_rate=44_100, layout="stereo"),
            )
        with mock.patch.object(
            media_publish_mod.av, "open", return_value=fake_container
        ):
            media._open_container()

        assert track._audio_sample_rate == 44100
        assert track._audio_layout == "stereo"
        assert fake_container.added_streams[0].rate == 44100
        assert fake_container.added_streams[0].layout == "stereo"

    def test_later_audio_drift_resamples_to_first_frame_targets_when_config_unset(
        self,
    ) -> None:
        config = media_publish_mod.MediaPublishConfig(
            tracks=[
                media_publish_mod.AudioOutputConfig(
                    format="flt",
                )
            ],
            track_wait_timeout_s=5.0,
        )
        media = media_publish_mod.MediaPublish(
            "http://example.test/trickle", config=config
        )
        media._loop = object()
        track = media._tracks[0]
        fake_container = _FakeContainer()

        with mock.patch.object(media_publish_mod.time, "monotonic", return_value=120.0):
            media._stage_frame_before_open(
                track,
                _FakeAudioFrame(sample_rate=44_100, layout="stereo"),
            )
        with mock.patch.object(
            media_publish_mod.av, "open", return_value=fake_container
        ):
            media._open_container()
        media._container = fake_container

        converted_frames: list[object] = []

        def _capture_converted(_track: object, frame: object) -> None:
            converted_frames.append(frame)

        with mock.patch.object(media_publish_mod.av, "AudioResampler", _FakeResampler):
            with mock.patch.object(
                media, "_encode_audio_frame_converted", side_effect=_capture_converted
            ):
                media._encode_audio_frame(
                    track,
                    _FakeAudioFrame(sample_rate=48_000, layout="mono"),
                )

        assert _FakeResampler.last_init == {
            "format": "flt",
            "layout": "stereo",
            "rate": 44100,
        }
        assert len(converted_frames) == 1

    def test_audio_stream_enforces_explicit_config_targets(self) -> None:
        config = media_publish_mod.MediaPublishConfig(
            tracks=[
                media_publish_mod.AudioOutputConfig(
                    sample_rate=48_000,
                    layout="mono",
                    format="flt",
                )
            ],
            track_wait_timeout_s=5.0,
        )
        media = media_publish_mod.MediaPublish(
            "http://example.test/trickle", config=config
        )
        media._loop = object()
        track = media._tracks[0]
        fake_container = _FakeContainer()

        with mock.patch.object(media_publish_mod.time, "monotonic", return_value=90.0):
            media._stage_frame_before_open(
                track,
                _FakeAudioFrame(sample_rate=44_100, layout="stereo"),
            )
        with mock.patch.object(
            media_publish_mod.av, "open", return_value=fake_container
        ):
            media._open_container()

        assert track._audio_sample_rate == 48000
        assert track._audio_layout == "mono"
        assert fake_container.added_streams[0].rate == 48000
        assert fake_container.added_streams[0].layout == "mono"

    def test_audio_stream_falls_back_to_internal_defaults_when_unset_and_missing_frame_metadata(
        self,
    ) -> None:
        config = media_publish_mod.MediaPublishConfig(
            tracks=[media_publish_mod.AudioOutputConfig(format="flt")],
            track_wait_timeout_s=5.0,
        )
        media = media_publish_mod.MediaPublish(
            "http://example.test/trickle", config=config
        )
        media._loop = object()
        track = media._tracks[0]
        fake_container = _FakeContainer()

        frame = _FakeAudioFrame(sample_rate=0, layout="stereo")
        frame.layout = None
        with mock.patch.object(media_publish_mod.time, "monotonic", return_value=90.0):
            media._stage_frame_before_open(track, frame)
        with mock.patch.object(
            media_publish_mod.av, "open", return_value=fake_container
        ):
            media._open_container()

        assert track._audio_sample_rate == 48000
        assert track._audio_layout == "mono"
        assert fake_container.added_streams[0].rate == 48000
        assert fake_container.added_streams[0].layout == "mono"

    def test_encoder_loop_opens_after_timeout_without_new_frames(self) -> None:
        media = self._build_media(timeout_s=1.0)
        video_track = media._tracks[0]
        audio_track = media._tracks[1]

        # First frame arrives only for video; audio remains missing.
        with mock.patch.object(media_publish_mod.time, "monotonic", return_value=0.0):
            media._stage_frame_before_open(video_track, _FakeVideoFrame())

        open_calls = {"count": 0}

        def _open() -> None:
            open_calls["count"] += 1
            media._container = _CloseOnlyContainer()

        def _next_item() -> tuple[object, object] | None:
            # First polling cycle returns no frame; timeout path should open.
            if open_calls["count"] == 0:
                return None
            # Then stop both tracks to terminate the encoder loop.
            if not video_track._stopped:
                return video_track, media_publish_mod._STOP
            if not audio_track._stopped:
                return audio_track, media_publish_mod._STOP
            return None

        with mock.patch.object(media_publish_mod.time, "monotonic", return_value=2.0):
            with mock.patch.object(media, "_open_container", side_effect=_open):
                with mock.patch.object(
                    media, "_flush_staged_frames", return_value=None
                ):
                    with mock.patch.object(
                        media, "_next_encoder_item", side_effect=_next_item
                    ):
                        media._run_encoder()

        assert open_calls["count"] == 1
        assert audio_track._dropped_timeout

    def test_writes_to_timed_out_track_raise_error(self) -> None:
        media = self._build_media(timeout_s=1.0)
        video_track = media._tracks[0]
        audio_track = media.get_tracks("audio")[0]

        with mock.patch.object(media_publish_mod.time, "monotonic", return_value=0.0):
            media._stage_frame_before_open(video_track, _FakeVideoFrame())
        with mock.patch.object(media_publish_mod.time, "monotonic", return_value=2.0):
            assert media._can_open_container()
        assert audio_track._dropped_timeout

        with pytest.raises(media_publish_mod.LivepeerGatewayError):
            asyncio.run(media._write_frame_to_track(audio_track, _FakeAudioFrame()))

    def test_track_resize_grows_capacity(self) -> None:
        media = media_publish_mod.MediaPublish(
            "http://example.test/trickle",
            config=media_publish_mod.MediaPublishConfig(
                tracks=[media_publish_mod.VideoOutputConfig(queue_size=2)]
            ),
        )
        track = media.get_tracks("video")[0]

        track.resize(8)

        assert track._queue.maxsize == 8

    def test_track_resize_same_capacity_succeeds(self) -> None:
        media = media_publish_mod.MediaPublish(
            "http://example.test/trickle",
            config=media_publish_mod.MediaPublishConfig(
                tracks=[media_publish_mod.VideoOutputConfig(queue_size=3)]
            ),
        )
        track = media.get_tracks("video")[0]

        track.resize(3)

        assert track._queue.maxsize == 3

    def test_track_resize_rejects_unknown_track(self) -> None:
        media = media_publish_mod.MediaPublish("http://example.test/trickle")
        unknown_track = media_publish_mod.MediaPublishTrack(
            media,
            kind="video",
            config=media_publish_mod.VideoOutputConfig(),
            index=99,
            queue=media_publish_mod._FrameQueue(
                maxsize=8,
                stats=media_publish_mod._new_track_stats(),
            ),
            stats=media_publish_mod._new_track_stats(),
        )

        with pytest.raises(TypeError):
            unknown_track.resize(4)

    def test_track_resize_rejects_non_positive_size(self) -> None:
        media = media_publish_mod.MediaPublish("http://example.test/trickle")
        track = media.get_tracks("video")[0]

        with pytest.raises(ValueError):
            track.resize(0)
        with pytest.raises(ValueError):
            track.resize(-1)

    def test_track_resize_rejects_shrink_below_depth(self) -> None:
        media = media_publish_mod.MediaPublish(
            "http://example.test/trickle",
            config=media_publish_mod.MediaPublishConfig(
                tracks=[media_publish_mod.VideoOutputConfig(queue_size=4)]
            ),
        )
        track = media.get_tracks("video")[0]
        track._queue.put("f0")
        track._queue.put("f1")
        track._queue.put("f2")

        with pytest.raises(ValueError):
            track.resize(2)

        assert track._queue.maxsize == 4
        assert track._queue.qsize == 3

    def test_track_resize_preserves_fifo_order(self) -> None:
        media = media_publish_mod.MediaPublish(
            "http://example.test/trickle",
            config=media_publish_mod.MediaPublishConfig(
                tracks=[media_publish_mod.VideoOutputConfig(queue_size=3)]
            ),
        )
        track = media.get_tracks("video")[0]
        track._queue.put("f0")
        track._queue.put("f1")
        track._queue.put("f2")

        track.resize(6)

        assert track._queue.get_nowait() == "f0"
        assert track._queue.get_nowait() == "f1"
        assert track._queue.get_nowait() == "f2"

    def test_rejects_negative_min_segment_wallclock(self) -> None:
        with pytest.raises(ValueError):
            media_publish_mod.MediaPublish(
                "http://example.test/trickle",
                config=media_publish_mod.MediaPublishConfig(
                    min_segment_wallclock_s=-0.1,
                ),
            )

    def test_stream_pipe_reuses_segment_across_invocations_until_min_wallclock(
        self,
    ) -> None:
        class _FakeSegment:
            def __init__(self) -> None:
                self.writes: list[bytes] = []
                self.close_calls = 0

            def seq(self) -> int:
                return 3

            async def write(self, chunk: bytes) -> None:
                self.writes.append(chunk)

            async def close(self) -> None:
                self.close_calls += 1

        class _FakePublisher:
            def __init__(self, segment: _FakeSegment) -> None:
                self._segment = segment
                self.next_calls = 0

            async def next(self) -> _FakeSegment:
                self.next_calls += 1
                return self._segment

        media = media_publish_mod.MediaPublish(
            "http://example.test/trickle",
            config=media_publish_mod.MediaPublishConfig(
                min_segment_wallclock_s=1.0,
            ),
        )
        segment = _FakeSegment()
        media._publisher = _FakePublisher(segment)  # type: ignore[assignment]
        read_file = io.BytesIO(b"abc")

        async def _inline_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        with mock.patch.object(
            media_publish_mod.asyncio, "to_thread", side_effect=_inline_to_thread
        ):
            with mock.patch.object(
                media_publish_mod, "_MONOTONIC", side_effect=[10.0, 10.2, 11.1]
            ):
                asyncio.run(media._stream_pipe_to_trickle(read_file))
                assert segment.close_calls == 0
                assert media._stats["segments_started"] == 1
                assert media._stats["segments_completed"] == 0
                assert media._active_segment is not None
                asyncio.run(media._stream_pipe_to_trickle(io.BytesIO(b"def")))

        assert segment.writes == [b"abc", b"def"]
        assert segment.close_calls == 1
        assert media._stats["segments_started"] == 1
        assert media._stats["segments_completed"] == 1
        assert media._publisher.next_calls == 1  # type: ignore[union-attr]
        assert media._active_segment is None

    def test_stream_pipe_closes_active_segment_promptly_when_closed(self) -> None:
        class _FakeSegment:
            def __init__(self) -> None:
                self.writes: list[bytes] = []
                self.close_calls = 0

            def seq(self) -> int:
                return 4

            async def write(self, chunk: bytes) -> None:
                self.writes.append(chunk)

            async def close(self) -> None:
                self.close_calls += 1

        class _FakePublisher:
            def __init__(self, segment: _FakeSegment) -> None:
                self._segment = segment
                self.next_calls = 0

            async def next(self) -> _FakeSegment:
                self.next_calls += 1
                return self._segment

        media = media_publish_mod.MediaPublish(
            "http://example.test/trickle",
            config=media_publish_mod.MediaPublishConfig(
                min_segment_wallclock_s=5.0,
            ),
        )
        segment = _FakeSegment()
        media._publisher = _FakePublisher(segment)  # type: ignore[assignment]

        async def _inline_to_thread(func, *args, **kwargs):
            return func(*args, **kwargs)

        with mock.patch.object(
            media_publish_mod.asyncio, "to_thread", side_effect=_inline_to_thread
        ):
            with mock.patch.object(
                media_publish_mod, "_MONOTONIC", side_effect=[20.0, 20.1]
            ):
                asyncio.run(media._stream_pipe_to_trickle(io.BytesIO(b"xyz")))
            assert media._active_segment is not None
            assert segment.close_calls == 0
            media._closed = True
            asyncio.run(media._stream_pipe_to_trickle(io.BytesIO(b"")))

        assert segment.writes == [b"xyz"]
        assert segment.close_calls == 1
        assert media._publisher.next_calls == 1  # type: ignore[union-attr]
        assert media._active_segment is None


class TestMediaPublishStall:
    """Cover mid-stream stall failure modes that used to kill the encoder.

    The historical failure mode:
      1. Pipeline stalls for a long time (e.g. 150s CUDA synchronize hang).
      2. Orchestrator / LB closes the idle segment POST connection.
      3. aiohttp raises ServerDisconnectedError, which becomes
         TrickleSegmentWriteError on the next SegmentWriter.write().
      4. The old _stream_pipe_to_trickle path closed read_file immediately,
         but PyAV's segment muxer was still holding the write end of the
         same OS pipe. The next muxed packet triggered BrokenPipe on the
         encoder thread and killed the stream.

    Fix 1 decouples the OS pipe lifecycle from segment POST failure:
    after a write error we keep reading-and-discarding from the pipe
    until PyAV closes its write end (EOF), and only then close the
    segment. These tests lock that in.
    """

    def _build_drain_media(
        self,
        *,
        fail_after: int,
        min_segment_wallclock_s: float = 0.0,
    ) -> tuple[media_publish_mod.MediaPublish, object]:
        class _FakeSegment:
            def __init__(self, *, fail_after: int) -> None:
                self.writes: list[bytes] = []
                self.close_calls = 0
                self._fail_after = fail_after

            def seq(self) -> int:
                return 2

            async def write(self, chunk: bytes) -> None:
                if len(self.writes) >= self._fail_after:
                    raise media_publish_mod.TrickleSegmentWriteError(
                        "simulated mid-segment disconnect",
                        seq=2,
                        url="http://example.test/trickle/2",
                    )
                self.writes.append(chunk)

            async def close(self) -> None:
                self.close_calls += 1

        class _FakePublisher:
            def __init__(self, segment: _FakeSegment) -> None:
                self._segment = segment
                self.next_calls = 0

            async def next(self) -> _FakeSegment:
                self.next_calls += 1
                return self._segment

        media = media_publish_mod.MediaPublish(
            "http://example.test/trickle",
            config=media_publish_mod.MediaPublishConfig(
                min_segment_wallclock_s=min_segment_wallclock_s,
            ),
        )
        segment = _FakeSegment(fail_after=fail_after)
        media._publisher = _FakePublisher(segment)  # type: ignore[assignment]
        return media, segment

    @staticmethod
    async def _inline_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    def test_mid_segment_disconnect_drains_remaining_chunks(self) -> None:
        media, segment = self._build_drain_media(fail_after=1)

        class _CountingReader:
            def __init__(self, chunks: list[bytes]) -> None:
                self._chunks = list(chunks)
                self.reads_returning_data = 0
                self.reads_returning_eof = 0
                self.close_calls = 0

            def read(self, _size: int) -> bytes:
                if self._chunks:
                    self.reads_returning_data += 1
                    return self._chunks.pop(0)
                self.reads_returning_eof += 1
                return b""

            def close(self) -> None:
                self.close_calls += 1

        reader = _CountingReader([b"a", b"b", b"c", b"d"])

        with mock.patch.object(
            media_publish_mod.asyncio, "to_thread", side_effect=self._inline_to_thread
        ):
            asyncio.run(media._stream_pipe_to_trickle(reader))

        # All four chunks were read (chunks 3 and 4 drained post-failure),
        # proving the task did not abandon the read fd after the write error.
        assert reader.reads_returning_data == 4
        assert reader.reads_returning_eof >= 1
        # Only the first chunk succeeded; chunks 3 and 4 were discarded.
        assert segment.writes == [b"a"]

    def test_mid_segment_disconnect_closes_segment_exactly_once(self) -> None:
        media, segment = self._build_drain_media(fail_after=1)

        class _Reader:
            def __init__(self, chunks: list[bytes]) -> None:
                self._chunks = list(chunks)

            def read(self, _size: int) -> bytes:
                return self._chunks.pop(0) if self._chunks else b""

            def close(self) -> None:
                return None

        reader = _Reader([b"a", b"b", b"c"])

        with mock.patch.object(
            media_publish_mod.asyncio, "to_thread", side_effect=self._inline_to_thread
        ):
            asyncio.run(media._stream_pipe_to_trickle(reader))

        # Segment close is called exactly once, from the post-loop
        # _close_active_segment_locked call, with mark_completed=False
        # (reflected in stats below).
        assert segment.close_calls == 1
        assert media._active_segment is None

    def test_mid_segment_disconnect_updates_stats(self) -> None:
        media, segment = self._build_drain_media(fail_after=1)

        class _Reader:
            def __init__(self, chunks: list[bytes]) -> None:
                self._chunks = list(chunks)

            def read(self, _size: int) -> bytes:
                return self._chunks.pop(0) if self._chunks else b""

            def close(self) -> None:
                return None

        # Two successful bytes attempted before failure is recorded: the
        # first write succeeds, and the second is counted in the "attempted"
        # bytes stat the moment before the write raises. The remaining
        # drained chunks are not counted.
        reader = _Reader([b"a", b"b", b"c", b"d"])

        with mock.patch.object(
            media_publish_mod.asyncio, "to_thread", side_effect=self._inline_to_thread
        ):
            asyncio.run(media._stream_pipe_to_trickle(reader))

        assert media._stats["segments_started"] == 1
        assert media._stats["segments_completed"] == 0
        assert media._stats["segments_failed"] == 1
        assert media._stats["bytes_streamed_to_trickle"] == 2
        # Crucially: the encoder thread did not error; that counter is
        # incremented only when _run_encoder raises. Fix 1 is what keeps
        # it at zero on mid-stream disconnects.
        assert media._stats["encoder_errors"] == 0
        assert media._stats["terminal_failures"] == 0

    def test_mid_segment_disconnect_respects_piggyback_window(self) -> None:
        """Write failure within an open min_segment_wallclock_s window
        keeps the wall-clock segment assigned and drains bytes to /dev/null
        for the rest of the window. Subsequent PyAV-segment invocations
        piggy-back as drain-only (no duplicate segments_failed bumps, no
        duplicate log spam). The segment is only finalized when the
        wall-clock window actually expires.
        """
        media, segment = self._build_drain_media(
            fail_after=1, min_segment_wallclock_s=60.0
        )

        class _Reader:
            def __init__(self, chunks: list[bytes]) -> None:
                self._chunks = list(chunks)

            def read(self, _size: int) -> bytes:
                return self._chunks.pop(0) if self._chunks else b""

            def close(self) -> None:
                return None

        # Invocation 1: PyAV-segment pipe fails on chunk 2, rest drained.
        with mock.patch.object(
            media_publish_mod.asyncio, "to_thread", side_effect=self._inline_to_thread
        ):
            asyncio.run(media._stream_pipe_to_trickle(_Reader([b"a", b"b", b"c"])))

        assert segment.close_calls == 0  # piggy-back window still open
        assert media._active_segment is segment
        assert media._segment_draining
        assert media._stats["segments_started"] == 1
        assert media._stats["segments_failed"] == 1
        publisher = media._publisher  # type: ignore[assignment]

        # Invocation 2: fresh PyAV-segment pipe arrives while the same
        # wall-clock window is still open. All bytes drained silently;
        # segment.write is not called again, so no second failure is
        # logged or counted.
        writes_before = len(segment.writes)
        with mock.patch.object(
            media_publish_mod.asyncio, "to_thread", side_effect=self._inline_to_thread
        ):
            asyncio.run(media._stream_pipe_to_trickle(_Reader([b"d", b"e"])))

        assert segment.close_calls == 0
        assert media._active_segment is segment
        assert len(segment.writes) == writes_before  # no new writes
        assert media._stats["segments_failed"] == 1  # not re-bumped
        assert publisher.next_calls == 1  # no new POST opened

        # Simulate wall-clock expiry: backdate the segment start so the
        # next invocation finalizes. Empty pipe = immediate EOF.
        assert media._active_segment_started_at is not None
        media._active_segment_started_at -= 120.0
        with mock.patch.object(
            media_publish_mod.asyncio, "to_thread", side_effect=self._inline_to_thread
        ):
            asyncio.run(media._stream_pipe_to_trickle(_Reader([])))

        # Segment finalized with mark_completed=False.
        assert segment.close_calls == 1
        assert media._active_segment is None
        assert media._active_segment_started_at is None
        assert not media._segment_draining
        assert media._stats["segments_completed"] == 0
        assert media._stats["segments_failed"] == 1

    def test_broken_pipe_regression_with_real_os_pipe(self) -> None:
        """Regression test: prove the OS pipe write end stays usable until EOF.

        Before Fix 1, a single TrickleSegmentWriteError caused the read end
        of the OS pipe to close while PyAV still owned the write end, and
        the encoder thread died with [Errno 32] Broken pipe on the very next
        mux. This test reproduces that scenario with a real os.pipe():
        simulate the encoder by writing bytes from a background thread,
        simulate a mid-segment disconnect on the nth chunk, and assert that
        the background writer can continue to write without a BrokenPipe
        until it voluntarily closes its write end (modeling PyAV segment
        rotation).
        """
        failure_event = threading.Event()

        class _FakeSegment:
            def __init__(self) -> None:
                self.writes = 0
                self.closed = False

            def seq(self) -> int:
                return 2

            async def write(self, _chunk: bytes) -> None:
                # Fail on the very first write so the test doesn't depend on
                # how the OS coalesces small writes into a single read chunk.
                self.writes += 1
                failure_event.set()
                raise media_publish_mod.TrickleSegmentWriteError(
                    "simulated mid-segment disconnect",
                    seq=2,
                )

            async def close(self) -> None:
                self.closed = True

        class _FakePublisher:
            def __init__(self, segment: _FakeSegment) -> None:
                self._segment = segment

            async def next(self) -> _FakeSegment:
                return self._segment

        media = media_publish_mod.MediaPublish(
            "http://example.test/trickle",
            config=media_publish_mod.MediaPublishConfig(
                min_segment_wallclock_s=0.0,
            ),
        )
        segment = _FakeSegment()
        media._publisher = _FakePublisher(segment)  # type: ignore[assignment]

        read_fd, write_fd = os.pipe()
        read_file = os.fdopen(read_fd, "rb", buffering=0)
        write_file = os.fdopen(write_fd, "wb", buffering=0)

        writer_errors: list[BaseException] = []
        writes_after_failure = 5

        def _simulated_encoder() -> None:
            try:
                # One initial write; the reader picks it up and segment.write
                # raises, setting failure_event.
                write_file.write(b"x" * 64)
                write_file.flush()
                if not failure_event.wait(timeout=3.0):
                    raise RuntimeError("reader never reached the failure point")
                # Post-failure writes. Pre-fix, these would hit EPIPE because
                # the read end had already been closed by the segment task's
                # drop-and-cleanup path. Post-fix, the read end stays open
                # and drains these bytes to /dev/null.
                for _ in range(writes_after_failure):
                    write_file.write(b"y" * 64)
                    write_file.flush()
            except Exception as e:
                writer_errors.append(e)
            finally:
                # Modeling PyAV rotating to the next segment: close our write
                # end on our own terms. The reader task sees EOF and exits.
                try:
                    write_file.close()
                except Exception:
                    pass

        writer_thread = threading.Thread(target=_simulated_encoder, daemon=True)
        writer_thread.start()

        try:
            asyncio.run(media._stream_pipe_to_trickle(read_file))
        finally:
            writer_thread.join(timeout=5.0)

        # The simulated encoder kept writing through the post-failure batch
        # and closed on its own terms. Pre-fix this would populate
        # writer_errors with a BrokenPipeError.
        assert writer_errors == []
        assert failure_event.is_set()
        assert segment.closed
        assert media._stats["segments_failed"] == 1
        assert media._stats["encoder_errors"] == 0


class TestMediaPublishIdleCutover:
    """Cover trickle idle cutover while one PyAV segment is open."""

    @staticmethod
    def _build_media(
        *,
        idle_timeout_s: float,
        min_segment_wallclock_s: float = 0.0,
    ) -> tuple[media_publish_mod.MediaPublish, object]:
        class _Segment:
            def __init__(self, seq: int) -> None:
                self._seq = seq
                self.writes: list[bytes] = []
                self.close_calls = 0

            def seq(self) -> int:
                return self._seq

            async def write(self, chunk: bytes) -> None:
                self.writes.append(chunk)

            async def close(self) -> None:
                self.close_calls += 1

        class _Publisher:
            def __init__(self) -> None:
                self.segments: list[_Segment] = []
                self.next_calls = 0

            async def next(self) -> _Segment:
                self.next_calls += 1
                segment = _Segment(seq=self.next_calls + 1)
                self.segments.append(segment)
                return segment

        media = media_publish_mod.MediaPublish(
            "http://example.test/trickle",
            config=media_publish_mod.MediaPublishConfig(
                min_segment_wallclock_s=min_segment_wallclock_s,
                segment_post_idle_timeout_s=idle_timeout_s,
            ),
        )
        publisher = _Publisher()
        media._publisher = publisher  # type: ignore[assignment]
        return media, publisher

    def test_default_idle_timeout_is_25s(self) -> None:
        """Locked in so the cutover budget stays below typical orch/LB
        idle-close budgets without further configuration."""
        assert (
            media_publish_mod.MediaPublishConfig().segment_post_idle_timeout_s == 25.0
        )

    def test_rejects_negative_idle_timeout(self) -> None:
        with pytest.raises(ValueError):
            media_publish_mod.MediaPublish(
                "http://example.test/trickle",
                config=media_publish_mod.MediaPublishConfig(
                    segment_post_idle_timeout_s=-1.0,
                ),
            )

    def test_rejects_zero_idle_timeout(self) -> None:
        with pytest.raises(ValueError):
            media_publish_mod.MediaPublish(
                "http://example.test/trickle",
                config=media_publish_mod.MediaPublishConfig(
                    segment_post_idle_timeout_s=0.0,
                ),
            )

    def test_idle_cutover_before_first_byte_keeps_late_bytes_writable(self) -> None:
        """If the PyAV segment has not emitted bytes yet, cutover is safe."""
        media, publisher = self._build_media(idle_timeout_s=0.05)

        late_chunk_ready = threading.Event()
        pipe_closed = threading.Event()

        class _BlockingReader:
            def __init__(self) -> None:
                self._state = "initial"

            def read(self, _size: int) -> bytes:
                if self._state == "initial":
                    self._state = "waiting_for_late_chunk"
                    late_chunk_ready.wait(timeout=2.0)
                    return b"late-mid-gop-chunk"
                if self._state == "waiting_for_late_chunk":
                    self._state = "eof"
                    pipe_closed.wait(timeout=2.0)
                    return b""
                return b""

        reader = _BlockingReader()

        async def _runner() -> None:
            task = asyncio.create_task(media._stream_pipe_to_trickle(reader))

            for _ in range(60):
                await asyncio.sleep(0.02)
                if publisher.next_calls >= 2:
                    break
            assert publisher.next_calls == 2
            assert publisher.segments[0].close_calls == 1
            assert media._active_segment is publisher.segments[1]
            assert publisher.segments[1].close_calls == 0

            late_chunk_ready.set()
            pipe_closed.set()
            await asyncio.wait_for(task, timeout=2.0)

        asyncio.run(_runner())

        assert publisher.segments[0].writes == []
        assert publisher.segments[1].writes == [b"late-mid-gop-chunk"]
        assert publisher.segments[1].close_calls == 1
        assert media._active_segment is None
        assert not media._segment_draining
        assert media._stats["segments_failed"] == 0
        assert media._stats["segments_completed"] == 1
        assert media._stats["segments_started"] == 2

    def test_idle_cutover_after_first_byte_drains_rest_of_pyav_segment(self) -> None:
        media, publisher = self._build_media(idle_timeout_s=0.05)

        late_chunk_ready = threading.Event()
        pipe_closed = threading.Event()

        class _Reader:
            def __init__(self) -> None:
                self._state = "first"

            def read(self, _size: int) -> bytes:
                if self._state == "first":
                    self._state = "waiting_for_late_chunk"
                    return b"opening-chunk"
                if self._state == "waiting_for_late_chunk":
                    self._state = "eof"
                    late_chunk_ready.wait(timeout=2.0)
                    return b"late-mid-gop-chunk"
                pipe_closed.wait(timeout=2.0)
                return b""

            def close(self) -> None:
                return None

        reader = _Reader()

        async def _runner() -> None:
            task = asyncio.create_task(media._stream_pipe_to_trickle(reader))

            for _ in range(60):
                await asyncio.sleep(0.02)
                if publisher.next_calls >= 2:
                    break
            assert publisher.next_calls == 2
            assert publisher.segments[0].writes == [b"opening-chunk"]
            assert publisher.segments[0].close_calls == 1
            assert media._active_segment is publisher.segments[1]

            late_chunk_ready.set()
            pipe_closed.set()
            await asyncio.wait_for(task, timeout=2.0)

        asyncio.run(_runner())

        assert publisher.segments[0].writes == [b"opening-chunk"]
        assert publisher.segments[1].writes == []
        assert publisher.segments[1].close_calls == 1
        assert media._active_segment is None
        assert media._stats["segments_failed"] == 0
        assert media._stats["segments_completed"] == 0
        assert media._stats["segments_started"] == 2

    def test_idle_cutover_repeats_until_pyav_rotation(self) -> None:
        """Repeated idle cutovers keep opening empty transport segments."""
        media, publisher = self._build_media(idle_timeout_s=0.03)

        class _SilentReader:
            def __init__(self) -> None:
                self.eof_gate = threading.Event()

            def read(self, _size: int) -> bytes:
                # Block until the test releases the gate, then return
                # EOF to simulate PyAV rotating.
                self.eof_gate.wait(timeout=2.0)
                return b""

        reader = _SilentReader()

        async def _runner() -> None:
            task = asyncio.create_task(media._stream_pipe_to_trickle(reader))
            for _ in range(100):
                await asyncio.sleep(0.02)
                if publisher.next_calls >= 4:
                    break
            assert publisher.next_calls >= 4
            reader.eof_gate.set()
            await asyncio.wait_for(task, timeout=2.0)

        asyncio.run(_runner())

        for seg in publisher.segments[:-1]:
            assert seg.close_calls == 1
            assert seg.writes == []
        assert publisher.segments[-1].close_calls == 1
        assert publisher.segments[-1].writes == []
        assert media._active_segment is None
        assert media._stats["segments_completed"] == 1
        assert media._stats["segments_failed"] == 0

    def test_idle_cutover_does_not_leak_bytes_into_thread_pool(self) -> None:
        """A pending read survives cutover and still delivers late bytes."""
        media, publisher = self._build_media(idle_timeout_s=0.05)

        produced_bytes: list[bytes] = []
        encoder_gate = threading.Event()

        class _Reader:
            def __init__(self) -> None:
                self._returned_chunk = False

            def read(self, _size: int) -> bytes:
                if not self._returned_chunk:
                    self._returned_chunk = True
                    encoder_gate.wait(timeout=2.0)
                    chunk = b"bytes-after-idle"
                    produced_bytes.append(chunk)
                    return chunk
                return b""

        reader = _Reader()

        async def _runner() -> None:
            task = asyncio.create_task(media._stream_pipe_to_trickle(reader))
            for _ in range(60):
                await asyncio.sleep(0.02)
                if publisher.next_calls >= 2:
                    break
            assert publisher.next_calls == 2
            encoder_gate.set()
            await asyncio.wait_for(task, timeout=2.0)

        asyncio.run(_runner())

        assert produced_bytes == [b"bytes-after-idle"]
        assert publisher.segments[0].writes == []
        assert publisher.segments[1].writes == [b"bytes-after-idle"]

    def test_idle_cutover_still_applies_after_write_failure(self) -> None:
        media = media_publish_mod.MediaPublish(
            "http://example.test/trickle",
            config=media_publish_mod.MediaPublishConfig(
                min_segment_wallclock_s=60.0,
                segment_post_idle_timeout_s=0.05,
            ),
        )

        class _Segment:
            def __init__(self, seq: int, *, fail_after: int | None = None) -> None:
                self._seq = seq
                self._fail_after = fail_after
                self.writes: list[bytes] = []
                self.close_calls = 0

            def seq(self) -> int:
                return self._seq

            async def write(self, chunk: bytes) -> None:
                if (
                    self._fail_after is not None
                    and len(self.writes) >= self._fail_after
                ):
                    raise media_publish_mod.TrickleSegmentWriteError(
                        "simulated mid-segment disconnect",
                        seq=self._seq,
                        url=f"http://example.test/trickle/{self._seq}",
                    )
                self.writes.append(chunk)

            async def close(self) -> None:
                self.close_calls += 1

        class _Publisher:
            def __init__(self) -> None:
                self.segments = [
                    _Segment(seq=2, fail_after=1),
                    _Segment(seq=3),
                ]
                self.next_calls = 0

            async def next(self) -> _Segment:
                segment = self.segments[self.next_calls]
                self.next_calls += 1
                return segment

        publisher = _Publisher()
        media._publisher = publisher  # type: ignore[assignment]

        late_chunk_ready = threading.Event()
        pipe_closed = threading.Event()

        class _Reader:
            def __init__(self) -> None:
                self._state = "first"

            def read(self, _size: int) -> bytes:
                if self._state == "first":
                    self._state = "fails-on-write"
                    return b"opening-chunk"
                if self._state == "fails-on-write":
                    self._state = "late-drain"
                    return b"chunk-that-fails"
                if self._state == "late-drain":
                    self._state = "eof"
                    late_chunk_ready.wait(timeout=2.0)
                    return b"late-after-failure"
                pipe_closed.wait(timeout=2.0)
                return b""

            def close(self) -> None:
                return None

        reader = _Reader()

        async def _runner() -> None:
            task = asyncio.create_task(media._stream_pipe_to_trickle(reader))
            for _ in range(60):
                await asyncio.sleep(0.02)
                if publisher.next_calls >= 2:
                    break
            assert publisher.next_calls == 2
            assert publisher.segments[0].close_calls == 1
            assert media._active_segment is publisher.segments[1]
            late_chunk_ready.set()
            pipe_closed.set()
            await asyncio.wait_for(task, timeout=2.0)

        asyncio.run(_runner())

        assert publisher.segments[0].writes == [b"opening-chunk"]
        assert publisher.segments[1].writes == []
        assert publisher.segments[1].close_calls == 1
        assert media._active_segment is None
        assert not media._segment_draining
        assert not media._eof_close_pending
        assert media._stats["segments_failed"] == 1
        assert media._stats["segments_completed"] == 0
        assert media._stats["segments_started"] == 2
