from __future__ import annotations

import asyncio
import logging
import math
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional, Set, Awaitable, Any, BinaryIO

import av
from av.video.frame import PictureType

from .errors import LivepeerGatewayError
from .trickle_publisher import (
    TricklePublisher,
    TricklePublisherStats,
    TricklePublisherTerminalError,
    TrickleSegmentWriteError,
)

_LOG = logging.getLogger(__name__)

_OUT_TIME_BASE = Fraction(1, 90_000)
_READ_CHUNK = 64 * 1024
_DRAIN_TIMEOUT_S = 5.0
_STOP = object()
_DEFAULT_AUDIO_SAMPLE_RATE = 48_000
_DEFAULT_AUDIO_LAYOUT = "mono"
_MONOTONIC = time.monotonic


def _fraction_from_time_base(time_base: object) -> Fraction:
    numerator = getattr(time_base, "numerator", None)
    denominator = getattr(time_base, "denominator", None)
    if numerator is not None and denominator is not None:
        return Fraction(int(numerator), int(denominator))
    return Fraction(time_base)


def _rescale_pts(pts: int, src_tb: Fraction, dst_tb: Fraction) -> int:
    if src_tb == dst_tb:
        return int(pts)
    return int(round(float((Fraction(pts) * src_tb) / dst_tb)))


def _normalize_fps(fps: Optional[float]) -> int:
    if fps is None or not math.isfinite(fps) or fps <= 0:
        fps = 30.0
    return max(1, int(round(fps)))


@dataclass(frozen=True)
class VideoOutputConfig:
    """
    Output settings for one video track.

    Stream creation is intentionally config-first where possible, but video
    stream initialization still waits for the first frame because width/height
    are not part of this config and are derived from the incoming frame.

    If the API later grows explicit output dimensions, the muxer could open
    video streams without waiting for the first frame.
    """

    # Local queue depth for this track before frames are dropped.
    queue_size: int = 8
    # Target output FPS hint passed to the encoder.
    fps: Optional[float] = None
    # Target GOP/keyframe cadence used for segment boundaries.
    keyframe_interval_s: float = 2.0
    # FFmpeg/PyAV encoder name for this output stream.
    codec: str = "libx264"
    # Output pixel format presented to the encoder.
    pix_fmt: str = "yuv420p"


@dataclass(frozen=True)
class AudioOutputConfig:
    """
    Output settings for one audio track.

    Audio stream creation is first-frame driven, like video.

    `sample_rate` and `layout` are optional:
    - when set, they are enforced as output targets
    - when unset (`None`), values are derived from the first audio frame
    - if unset and first-frame metadata is missing, internal defaults apply
    """

    # Local queue depth for this track before frames are dropped.
    queue_size: int = 32
    # FFmpeg/PyAV encoder name for this output stream.
    codec: str = "libopus"
    # Target output sample rate. None derives from first frame.
    sample_rate: Optional[int] = None
    # Target output channel layout. None derives from first frame.
    layout: Optional[str] = None
    # Output sample format presented to the encoder.
    format: str = "flt"


@dataclass(frozen=True)
class MediaPublishConfig:
    """
    Top-level media publish configuration.

    `tracks` defines the full set of output tracks for the muxed stream. Each
    entry becomes its own runtime track handle, queue, and encoder state inside
    one shared output container.
    """

    mime_type: str = "video/mp2t"
    tracks: list[VideoOutputConfig | AudioOutputConfig] = field(
        default_factory=lambda: [VideoOutputConfig()]
    )
    # Max seconds to wait for missing tracks after the first frame arrives.
    # Tracks with no first frame by the deadline are dropped.
    track_wait_timeout_s: float = 5.0
    # Best-effort lower bound on wall-clock lifetime for each trickle segment.
    min_segment_wallclock_s: float = 1.0

@dataclass(frozen=True)
class TrackQueueStats:
    """Per-track queue statistics."""
    label: str
    frames_in: int
    frames_dropped_overflow: int
    frames_dropped_debt: int
    frames_dropped_non_monotonic_pts: int
    time_debt_s: float
    queue_depth: int
    # queue_media_time_s: span of source PTS media-time between the most-
    # recently enqueued frame and the most-recently dequeued frame on this
    # track. Approximates how much media time is currently buffered in the
    # per-track queue waiting to be encoded.
    queue_media_time_s: float
    # total_media_time_processed_s: span of source PTS media-time between
    # the first and most-recently dequeued frame. Approximates total media
    # time the encoder has pulled off this track's queue since start.
    total_media_time_processed_s: float

    def __str__(self) -> str:
        return (
            f"{self.label}("
            f"in={self.frames_in}, "
            f"overflow={self.frames_dropped_overflow}, "
            f"debt={self.frames_dropped_debt}, "
            f"nm_pts={self.frames_dropped_non_monotonic_pts}, "
            f"debt_s={self.time_debt_s:.4f}, "
            f"depth={self.queue_depth}, "
            f"queue_s={self.queue_media_time_s:.4f}, "
            f"processed_s={self.total_media_time_processed_s:.4f})"
        )


@dataclass(frozen=True)
class MediaPublishStats:
    elapsed_s: float
    segments_started: int
    segments_completed: int
    segments_failed: int
    bytes_streamed_to_trickle: int
    segment_writer_put_timeouts: int
    terminal_failures: int
    encoder_errors: int
    publisher: TricklePublisherStats
    track_queue_stats: tuple[TrackQueueStats, ...] = ()

    def __str__(self) -> str:
        tracks = ", ".join(str(t) for t in self.track_queue_stats)
        return (
            "MediaPublishStats("
            f"elapsed_s={self.elapsed_s:.1f}, "
            f"segments_started={self.segments_started}, "
            f"segments_completed={self.segments_completed}, "
            f"segments_failed={self.segments_failed}, "
            f"bytes_streamed_to_trickle={self.bytes_streamed_to_trickle}, "
            f"segment_writer_put_timeouts={self.segment_writer_put_timeouts}, "
            f"terminal_failures={self.terminal_failures}, "
            f"encoder_errors={self.encoder_errors}, "
            f"tracks=[{tracks}]"
            ")"
        )


_TRACK_STAT_KEYS = ("frames_in", "frames_dropped_overflow", "frames_dropped_debt", "frames_dropped_non_monotonic_pts")


def _new_track_stats() -> dict[str, int]:
    return {k: 0 for k in _TRACK_STAT_KEYS}


class MediaPublishTrack:
    """
    Runtime handle for one configured output track.

    Owns its queue and encoder state. Call `write_frame()` to enqueue frames
    for encoding. Use `resize()` to adjust queue capacity at runtime.
    """

    def __init__(
        self,
        owner: "MediaPublish",
        *,
        kind: str,
        config: VideoOutputConfig | AudioOutputConfig,
        index: int,
        queue: "_FrameQueue",
        stats: dict[str, int],
    ) -> None:
        self._owner = owner
        self.kind = kind
        self.config = config
        self.index = index

        # Queue and stats.
        self._queue: _FrameQueue = queue
        self._stats: dict[str, int] = stats

        # Encoder-thread-owned state.
        self._label: str = ""
        self._stream: Any = None
        self._pending_frames: list[av.VideoFrame | av.AudioFrame] = []
        self._first_frame: Optional[av.VideoFrame | av.AudioFrame] = None
        self._audio_sample_rate: Optional[int] = None
        self._audio_layout: Optional[str] = None
        self._dropped_timeout: bool = False
        self._last_out_pts: Optional[int] = None
        self._wallclock_start: Optional[float] = None
        self._next_out_pts: Optional[int] = None
        self._last_keyframe_time: Optional[float] = None
        self._stopped: bool = False
        self._audio_resampler: Any = None

    async def write_frame(self, frame: av.VideoFrame | av.AudioFrame) -> None:
        await self._owner._write_frame_to_track(self, frame)

    def resize(self, queue_size: int) -> None:
        """Resize this track's queue at runtime."""
        self._owner.resize_track_queue(self, queue_size)

    def __repr__(self) -> str:
        return f"MediaPublishTrack(kind={self.kind!r}, index={self.index}, config={self.config!r})"


class MediaPublish:
    """
    Publish muxed media as segmented MPEG-TS over trickle.

    One `MediaPublish` owns a single output container and one runtime track
    state per configured output track. Both audio and video tracks are
    first-frame driven. Once the first frame is observed for any track, a
    startup timeout begins; tracks that do not deliver their first frame before
    the deadline are dropped before container initialization.
    """

    def __init__(
        self,
        publish_url: str,
        *,
        config: MediaPublishConfig = MediaPublishConfig(),
    ) -> None:
        self.publish_url = publish_url
        if not config.tracks:
            raise ValueError("MediaPublishConfig.tracks must include at least one track")

        self._publisher = TricklePublisher(
            publish_url,
            config.mime_type,
        )

        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._segment_tasks: Set[asyncio.Task[None]] = set()
        self._segment_stream_lock = asyncio.Lock()
        self._start_lock = threading.Lock()
        self._track_wait_timeout_s = float(config.track_wait_timeout_s)
        if self._track_wait_timeout_s < 0:
            raise ValueError("MediaPublishConfig.track_wait_timeout_s must be >= 0")
        self._min_segment_wallclock_s = float(config.min_segment_wallclock_s)
        if self._min_segment_wallclock_s < 0:
            raise ValueError("MediaPublishConfig.min_segment_wallclock_s must be >= 0")
        self._first_frame_arrived_at: Optional[float] = None

        self._closed = False
        self._error: Optional[BaseException] = None
        self._started_at = time.time()
        self._stats: dict[str, int] = {
            "segments_started": 0,
            "segments_completed": 0,
            "segments_failed": 0,
            "terminal_failures": 0,
            "bytes_streamed_to_trickle": 0,
            "segment_writer_put_timeouts": 0,
            "encoder_errors": 0,
        }
        self._tracks: list[MediaPublishTrack] = []
        self._video_tracks: list[MediaPublishTrack] = []
        self._audio_tracks: list[MediaPublishTrack] = []
        for track_config in config.tracks:
            track_stats = _new_track_stats()
            if isinstance(track_config, VideoOutputConfig):
                queue_obj = _FrameQueue(
                    maxsize=track_config.queue_size,
                    stats=track_stats,
                    debt_skip=True,
                )
                track_writer = MediaPublishTrack(
                    self,
                    kind="video",
                    config=track_config,
                    index=len(self._video_tracks),
                    queue=queue_obj,
                    stats=track_stats,
                )
                self._video_tracks.append(track_writer)
            elif isinstance(track_config, AudioOutputConfig):
                queue_obj = _FrameQueue(
                    maxsize=track_config.queue_size,
                    stats=track_stats,
                )
                track_writer = MediaPublishTrack(
                    self,
                    kind="audio",
                    config=track_config,
                    index=len(self._audio_tracks),
                    queue=queue_obj,
                    stats=track_stats,
                )
                self._audio_tracks.append(track_writer)
            else:
                raise TypeError(f"Unsupported track config type: {type(track_config).__name__}")
            self._tracks.append(track_writer)

        video_count = len(self._video_tracks)
        audio_count = len(self._audio_tracks)
        for track in self._tracks:
            count = video_count if track.kind == "video" else audio_count
            track._label = track.kind if count == 1 else f"{track.kind}_{track.index}"

        video_configs = [track.config for track in self._video_tracks if isinstance(track.config, VideoOutputConfig)]
        self._segment_time_s = (
            min(float(track.keyframe_interval_s) for track in video_configs)
            if video_configs
            else 2.0
        )
        self._next_state_index = 0

        # Encoder state (owned by the encoder thread).
        self._container: Optional[av.container.OutputContainer] = None
        self._active_segment: Any = None
        self._active_segment_started_at: Optional[float] = None

    @property
    def tracks(self) -> tuple[MediaPublishTrack, ...]:
        return tuple(self._tracks)

    def get_tracks(self, kind: Optional[str] = None) -> list[MediaPublishTrack]:
        if kind is None:
            return list(self._tracks)
        normalized = kind.strip().lower()
        if normalized == "video":
            return list(self._video_tracks)
        if normalized == "audio":
            return list(self._audio_tracks)
        raise ValueError(f"Unsupported track kind: {kind!r}")

    def resize_track_queue(self, track: MediaPublishTrack, queue_size: int) -> None:
        """Resize one track queue at runtime.

        Resizing to a smaller size than the current queue depth is rejected.
        """
        if self._closed:
            raise LivepeerGatewayError("MediaPublish is closed")
        if self._error:
            raise LivepeerGatewayError(f"MediaPublish failed: {self._error}") from self._error
        if track not in self._tracks:
            raise TypeError("MediaPublish track is not recognized")
        track._queue.resize(queue_size)

    async def write_frame(self, frame: av.VideoFrame | av.AudioFrame) -> None:
        track = self._resolve_track_for_frame(frame)
        await self._write_frame_to_track(track, frame)

    def _resolve_track_for_frame(self, frame: av.VideoFrame | av.AudioFrame) -> MediaPublishTrack:
        if isinstance(frame, av.VideoFrame):
            tracks = self._video_tracks
            kind = "video"
        elif isinstance(frame, av.AudioFrame):
            tracks = self._audio_tracks
            kind = "audio"
        else:
            raise TypeError(f"write_frame expects av.VideoFrame or av.AudioFrame, got {type(frame).__name__}")
        if not tracks:
            raise TypeError(f"MediaPublish {kind} track is not enabled")
        if len(tracks) > 1:
            raise TypeError(
                f"MediaPublish.write_frame is ambiguous with multiple {kind} tracks; "
                f"use MediaPublish.get_tracks({kind!r}) and call write_frame() on the selected track"
            )
        return tracks[0]

    async def _write_frame_to_track(
        self,
        track: MediaPublishTrack,
        frame: av.VideoFrame | av.AudioFrame,
    ) -> None:
        if self._closed:
            raise LivepeerGatewayError("MediaPublish is closed")
        if self._error:
            raise LivepeerGatewayError(f"MediaPublish failed: {self._error}") from self._error

        if self._loop is None:
            self._loop = asyncio.get_running_loop()

        if track not in self._tracks:
            raise TypeError("MediaPublish track is not recognized")
        if track.kind == "video":
            if not isinstance(frame, av.VideoFrame):
                raise TypeError(f"Track kind {track.kind!r} expects av.VideoFrame, got {type(frame).__name__}")
            if track not in self._video_tracks:
                raise TypeError("MediaPublish video track is not enabled")
        elif track.kind == "audio":
            if not isinstance(frame, av.AudioFrame):
                raise TypeError(f"Track kind {track.kind!r} expects av.AudioFrame, got {type(frame).__name__}")
            if track not in self._audio_tracks:
                raise TypeError("MediaPublish audio track is not enabled")
        else:
            raise TypeError(f"Unsupported MediaPublish track kind: {track.kind!r}")
        if track._dropped_timeout:
            raise LivepeerGatewayError(
                f"MediaPublish {track._label} track dropped before initialization timeout"
            )
        self._ensure_thread()
        track._stats["frames_in"] += 1
        track._queue.put(frame)

    async def _suppress_close_step(self, step_name: str, awaitable: Awaitable[Any]) -> None:
        try:
            await awaitable
        except Exception:
            _LOG.warning("MediaPublish close suppressed %s failure", step_name, exc_info=True)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        # Close is best-effort; capture any errors, log them and move on.
        # Intentionally step-wise: each shutdown action has its own
        # suppression so one failure does not prevent later cleanup.
        if self._thread is not None:
            for track in self._tracks:
                await self._suppress_close_step(
                    f"{track.kind} sentinel enqueue",
                    asyncio.to_thread(track._queue.put, _STOP),
                )
            await self._suppress_close_step("encoder join", asyncio.to_thread(self._thread.join, 2.0))
            if self._thread.is_alive():
                _LOG.warning("MediaPublish encoder thread still alive after join timeout")

        # Segment tasks may be blocked writing into trickle when the network
        # path is unhealthy; cancel them first so the close stays bounded.
        for task in list(self._segment_tasks):
            task.cancel()
        if self._segment_tasks:
            await self._suppress_close_step(
                "segment task gather",
                asyncio.gather(*list(self._segment_tasks), return_exceptions=True),
            )
        await self._suppress_close_step(
            "active segment close",
            self._close_active_segment(mark_completed=False),
        )

        await self._suppress_close_step("publisher close", self._publisher.close())

        if self._error:
            _LOG.warning(
                "MediaPublish close suppressed prior publish failure: %s",
                self._error,
                exc_info=(type(self._error), self._error, self._error.__traceback__),
            )

    def _ensure_thread(self) -> None:
        with self._start_lock:
            if self._thread is not None:
                return
            self._thread = threading.Thread(
                target=self._run_encoder,
                name="MediaPublishEncoder",
                daemon=True,
            )
            self._thread.start()

    def _run_encoder(self) -> None:
        try:
            while True:
                # Evaluate startup readiness even when no new frames arrive, so
                # timeout-based drops can still unblock container open.
                if self._container is None and self._can_open_container():
                    self._open_container()
                    self._flush_staged_frames()

                selected = self._next_encoder_item()
                if self._error is not None:
                    break
                if selected is None:
                    active = [t for t in self._tracks if not t._stopped]
                    if not active:
                        break
                    continue
                track, item = selected
                if item is _STOP:
                    track._stopped = True
                    continue

                if self._container is None:
                    self._stage_frame_before_open(track, item)
                    continue

                self._encode_track_frame(track, item)

            self._flush_encoder()
        except Exception as e:
            self._error = e
            self._stats["encoder_errors"] += 1
            _LOG.error("MediaPublish encoder error", exc_info=True)
        finally:
            if self._container is not None:
                try:
                    self._container.close()
                except Exception:
                    _LOG.exception("MediaPublish failed to close container")
            self._container = None
            for track in self._tracks:
                track._stream = None

    def _next_encoder_item(self) -> Optional[tuple[MediaPublishTrack, object]]:
        active = [t for t in self._tracks if not t._stopped]
        if not active:
            return None

        track_count = len(self._tracks)
        for offset in range(track_count):
            idx = (self._next_state_index + offset) % track_count
            track = self._tracks[idx]
            if track._stopped:
                continue
            item = track._queue.get_nowait()
            if item is not None:
                self._next_state_index = (idx + 1) % track_count
                return track, item

        for offset in range(track_count):
            idx = (self._next_state_index + offset) % track_count
            track = self._tracks[idx]
            if track._stopped:
                continue
            item = track._queue.get(timeout=0.05)
            if item is not None:
                self._next_state_index = (idx + 1) % track_count
                return track, item
        return None

    def _stage_frame_before_open(self, track: MediaPublishTrack, frame: av.VideoFrame | av.AudioFrame) -> None:
        if track._first_frame is None:
            track._first_frame = frame
        if self._first_frame_arrived_at is None:
            self._first_frame_arrived_at = time.monotonic()
        track._pending_frames.append(frame)

    def _can_open_container(self) -> bool:
        # The container can open when all active tracks are either initialized
        # (first frame seen), explicitly stopped, or timed out. Timeout starts
        # when the first frame arrives on any track.
        if self._first_frame_arrived_at is None:
            return False
        deadline_expired = (
            time.monotonic() - self._first_frame_arrived_at
        ) >= self._track_wait_timeout_s
        for track in self._tracks:
            if track._stopped:
                continue
            if track._first_frame is not None:
                continue
            if deadline_expired:
                _LOG.warning(
                    "MediaPublish dropping late track %s after %.3fs without first frame",
                    track._label,
                    self._track_wait_timeout_s,
                )
                track._dropped_timeout = True
                track._stopped = True
                track._pending_frames.clear()
                continue
            return False
        return any(t._first_frame is not None for t in self._tracks)

    def _open_container(self) -> None:
        if self._loop is None:
            raise RuntimeError("MediaPublish loop is not set")

        def custom_io_open(url: str, flags: int, options: dict) -> object:
            read_fd, write_fd = os.pipe()
            read_file = os.fdopen(read_fd, "rb", buffering=0)
            write_file = os.fdopen(write_fd, "wb", buffering=0)
            self._schedule_pipe_reader(read_file)
            return write_file

        segment_options = {
            "segment_time": str(self._segment_time_s),
            "segment_format": "mpegts",
        }

        self._container = av.open(
            "%d.ts",
            format="segment",
            mode="w",
            io_open=custom_io_open,
            options=segment_options,
        )

        for track in self._tracks:
            if track._stopped:
                continue
            if track._first_frame is None:
                continue
            if track.kind == "video":
                config = track.config
                assert isinstance(config, VideoOutputConfig)
                if not isinstance(track._first_frame, av.VideoFrame):
                    continue
                video_opts = {
                    "bf": "0",
                    "preset": "superfast",
                    "tune": "zerolatency",
                    "forced-idr": "1",
                }
                video_kwargs = {
                    "time_base": _OUT_TIME_BASE,
                    "width": track._first_frame.width,
                    "height": track._first_frame.height,
                    "pix_fmt": config.pix_fmt,
                }
                rounded_fps = _normalize_fps(config.fps)
                track._stream = self._container.add_stream(
                    config.codec,
                    rate=rounded_fps,
                    options=video_opts,
                    **video_kwargs,
                )
                continue

            config = track.config
            assert isinstance(config, AudioOutputConfig)
            if not isinstance(track._first_frame, av.AudioFrame):
                continue
            first_audio_frame = track._first_frame
            sample_rate = int(
                config.sample_rate
                if config.sample_rate is not None
                else (getattr(first_audio_frame, "sample_rate", 0) or _DEFAULT_AUDIO_SAMPLE_RATE)
            )
            layout = (
                config.layout
                if config.layout is not None
                else (
                str(first_audio_frame.layout.name)
                if getattr(first_audio_frame, "layout", None) is not None
                and getattr(first_audio_frame.layout, "name", None)
                else _DEFAULT_AUDIO_LAYOUT
                )
            )
            track._audio_sample_rate = sample_rate
            track._audio_layout = layout
            track._stream = self._container.add_stream(
                config.codec,
                rate=sample_rate,
            )
            try:
                track._stream.time_base = _OUT_TIME_BASE
            except Exception:
                pass
            for attr, value in (
                ("layout", layout),
                ("format", config.format),
            ):
                try:
                    setattr(track._stream, attr, value)
                except Exception:
                    pass

    def _flush_staged_frames(self) -> None:
        for track in self._tracks:
            pending = list(track._pending_frames)
            track._pending_frames.clear()
            for frame in pending:
                self._encode_track_frame(track, frame)

    def _encode_track_frame(self, track: MediaPublishTrack, frame: av.VideoFrame | av.AudioFrame) -> None:
        if track.kind == "video":
            assert isinstance(frame, av.VideoFrame)
            encode_started = time.monotonic()
            encoded, encoded_media_time_s = self._encode_video_frame(track, frame)
            encode_duration_s = max(0.0, time.monotonic() - encode_started)
            if encoded:
                track._queue.update_after_encode(
                    encoded_media_time_s=encoded_media_time_s,
                    encode_duration_s=encode_duration_s,
                )
            return

        assert isinstance(frame, av.AudioFrame)
        self._encode_audio_frame(track, frame)

    def _encode_video_frame(self, track: MediaPublishTrack, frame: av.VideoFrame) -> tuple[bool, float]:
        if track._stream is None or self._container is None:
            raise RuntimeError("MediaPublish encoder is not initialized")
        config = track.config
        assert isinstance(config, VideoOutputConfig)

        source_pts = frame.pts
        source_tb = frame.time_base

        output_pix_fmt = config.pix_fmt
        if frame.format.name != output_pix_fmt:
            frame = frame.reformat(format=output_pix_fmt)

        current_time_s, out_pts = self._compute_pts(track, source_pts, source_tb)
        if track._last_out_pts is not None and out_pts <= track._last_out_pts:
            # Timestamp would overlap with previous frame, so drop. This can
            # happen if frames arrive faster than the effective encode rate.
            track._stats["frames_dropped_non_monotonic_pts"] += 1
            return False, current_time_s
        track._last_out_pts = out_pts
        frame.pts = out_pts
        frame.time_base = _OUT_TIME_BASE

        if (
            track._last_keyframe_time is None
            or current_time_s - track._last_keyframe_time >= float(config.keyframe_interval_s)
        ):
            frame.pict_type = PictureType.I
            track._last_keyframe_time = current_time_s
        else:
            frame.pict_type = PictureType.NONE

        packets = track._stream.encode(frame)
        for packet in packets:
            self._container.mux(packet)
        return True, current_time_s

    def _encode_audio_frame(self, track: MediaPublishTrack, frame: av.AudioFrame) -> None:
        if track._stream is None or self._container is None:
            raise RuntimeError("MediaPublish audio encoder is not initialized")
        config = track.config
        assert isinstance(config, AudioOutputConfig)
        sample_rate = track._audio_sample_rate
        layout = track._audio_layout
        if sample_rate is None or layout is None:
            raise RuntimeError("MediaPublish audio stream targets are not initialized")

        frame_layout = frame.layout.name if frame.layout is not None else None
        frame_format = frame.format.name if frame.format is not None else None
        needs_resample = (
            frame.sample_rate != sample_rate
            or frame_layout != layout
            or frame_format != config.format
        )
        if needs_resample:
            if track._audio_resampler is None:
                track._audio_resampler = av.AudioResampler(
                    format=config.format,
                    layout=layout,
                    rate=sample_rate,
                )
            for converted in track._audio_resampler.resample(frame):
                self._encode_audio_frame_converted(track, converted)
            return

        self._encode_audio_frame_converted(track, frame)

    def _encode_audio_frame_converted(self, track: MediaPublishTrack, frame: av.AudioFrame) -> None:
        if track._stream is None or self._container is None:
            raise RuntimeError("MediaPublish audio encoder is not initialized")

        _, out_pts = self._compute_audio_pts(track, frame)
        if track._last_out_pts is not None and out_pts <= track._last_out_pts:
            track._stats["frames_dropped_non_monotonic_pts"] += 1
            return
        track._last_out_pts = out_pts
        frame.pts = out_pts
        frame.time_base = _OUT_TIME_BASE

        packets = track._stream.encode(frame)
        for packet in packets:
            self._container.mux(packet)

    def _flush_encoder(self) -> None:
        if self._container is None:
            return
        for track in self._tracks:
            if track._stream is None:
                continue
            if track.kind == "audio" and track._audio_resampler is not None:
                for converted in track._audio_resampler.resample(None):
                    self._encode_audio_frame_converted(track, converted)
            packets = track._stream.encode(None)
            for packet in packets:
                self._container.mux(packet)

    def _compute_pts(
        self,
        track: MediaPublishTrack,
        pts: Optional[int],
        time_base: Optional[Fraction],
    ) -> tuple[float, int]:
        if pts is not None and time_base is not None:
            tb = _fraction_from_time_base(time_base)
            current_time_s = float(Fraction(pts) * tb)
            out_pts = _rescale_pts(pts, tb, _OUT_TIME_BASE)
            return current_time_s, out_pts

        now = time.time()
        if track._wallclock_start is None:
            track._wallclock_start = now
        current_time_s = now - track._wallclock_start
        return current_time_s, int(current_time_s * _OUT_TIME_BASE.denominator)

    def _compute_audio_pts(self, track: MediaPublishTrack, frame: av.AudioFrame) -> tuple[float, int]:
        if frame.pts is not None and frame.time_base is not None:
            tb = _fraction_from_time_base(frame.time_base)
            current_time_s = float(Fraction(frame.pts) * tb)
            out_pts = _rescale_pts(frame.pts, tb, _OUT_TIME_BASE)
            return current_time_s, out_pts

        now = time.time()
        if track._wallclock_start is None:
            track._wallclock_start = now
        current_time_s = now - track._wallclock_start
        if track._next_out_pts is None:
            track._next_out_pts = int(current_time_s * _OUT_TIME_BASE.denominator)
        out_pts = track._next_out_pts
        sample_rate = track._audio_sample_rate
        if sample_rate is None:
            raise RuntimeError("MediaPublish audio sample rate is not initialized")
        sample_rate_for_step = max(1, int(getattr(frame, "sample_rate", 0) or sample_rate))
        samples = max(0, int(getattr(frame, "samples", 0) or 0))
        step = int(round(samples * (_OUT_TIME_BASE.denominator / sample_rate_for_step)))
        track._next_out_pts = out_pts + max(1, step)
        return current_time_s, out_pts

    def _schedule_pipe_reader(self, read_file: BinaryIO) -> None:
        def _start() -> None:
            task = self._loop.create_task(self._stream_pipe_to_trickle(read_file))
            self._segment_tasks.add(task)
            task.add_done_callback(self._segment_tasks.discard)

        self._loop.call_soon_threadsafe(_start)

    async def _close_active_segment(self, *, mark_completed: bool) -> None:
        async with self._segment_stream_lock:
            await self._close_active_segment_locked(mark_completed=mark_completed)

    async def _close_active_segment_locked(self, *, mark_completed: bool) -> None:
        segment = self._active_segment
        self._active_segment = None
        self._active_segment_started_at = None
        if segment is None:
            return
        await segment.close()
        if mark_completed:
            self._stats["segments_completed"] += 1

    async def _stream_pipe_to_trickle(self, read_file: BinaryIO) -> None:
        segment_seq: Optional[int] = None
        try:
            async with self._segment_stream_lock:
                if self._active_segment is None:
                    self._active_segment = await self._publisher.next()
                    self._active_segment_started_at = _MONOTONIC()
                    self._stats["segments_started"] += 1
                segment = self._active_segment
                segment_seq = segment.seq()
                while True:
                    chunk = await asyncio.to_thread(read_file.read, _READ_CHUNK)
                    if not chunk:
                        break
                    self._stats["bytes_streamed_to_trickle"] += len(chunk)
                    # NB: the segment writer has its own chunk timeout, so
                    # lean on that instead of doing that here
                    await segment.write(chunk)
                should_close_segment = self._closed or self._min_segment_wallclock_s <= 0
                if (
                    not should_close_segment
                    and self._active_segment_started_at is not None
                ):
                    elapsed_s = max(0.0, _MONOTONIC() - self._active_segment_started_at)
                    should_close_segment = elapsed_s >= self._min_segment_wallclock_s
                if should_close_segment:
                    await self._close_active_segment_locked(mark_completed=True)
        except TricklePublisherTerminalError as e:
            # At this point, publisher.next() has exhausted its retries and the
            # publisher cannot be used for future segments.
            if self._error is None:
                self._error = e
            self._stats["segments_failed"] += 1
            self._stats["terminal_failures"] += 1
            await self._close_active_segment(mark_completed=False)
            _LOG.error("MediaPublish terminal failure while streaming", exc_info=True)
        except TrickleSegmentWriteError:
            self._stats["segments_failed"] += 1
            await self._close_active_segment(mark_completed=False)
            _LOG.warning("MediaPublish dropped segment seq=%s", segment_seq, exc_info=True)
        except Exception:
            self._stats["segments_failed"] += 1
            await self._close_active_segment(mark_completed=False)
            _LOG.exception(
                "MediaPublish unexpected failure while streaming segment seq=%s",
                segment_seq,
            )
        finally:
            # This block is critical for clean-up; always drain and close file
            # Because not all exceptions will be caught (eg, CancelledError)
            try:
                await self._drain_pipe(read_file)
                read_file.close()
            except Exception:
                pass

    async def _drain_pipe(self, read_file: BinaryIO) -> None:
        async def _drain() -> None:
            while True:
                chunk = await asyncio.to_thread(read_file.read, _READ_CHUNK)
                if not chunk:
                    break

        # Best-effort cleanup: absorb TimeoutError and CancelledError
        # so drain never blocks shutdown
        try:
            await asyncio.wait_for(_drain(), timeout=_DRAIN_TIMEOUT_S)
        except BaseException:
            pass

    def get_stats(self) -> MediaPublishStats:
        publisher_stats = self._publisher.get_stats()
        per_track = tuple(
            TrackQueueStats(
                label=track._label,
                frames_in=track._stats["frames_in"],
                frames_dropped_overflow=track._stats["frames_dropped_overflow"],
                frames_dropped_debt=track._stats["frames_dropped_debt"],
                frames_dropped_non_monotonic_pts=track._stats["frames_dropped_non_monotonic_pts"],
                time_debt_s=track._queue.time_debt_s,
                queue_depth=track._queue.qsize,
                queue_media_time_s=track._queue.queue_media_time_s,
                total_media_time_processed_s=track._queue.total_media_time_processed_s,
            )
            for track in self._tracks
        )
        return MediaPublishStats(
            elapsed_s=max(0.0, time.time() - self._started_at),
            segments_started=self._stats["segments_started"],
            segments_completed=self._stats["segments_completed"],
            segments_failed=self._stats["segments_failed"],
            bytes_streamed_to_trickle=self._stats["bytes_streamed_to_trickle"],
            segment_writer_put_timeouts=max(
                self._stats["segment_writer_put_timeouts"],
                publisher_stats.segment_writer_put_timeouts,
            ),
            terminal_failures=max(
                self._stats["terminal_failures"],
                publisher_stats.terminal_failures,
            ),
            encoder_errors=self._stats["encoder_errors"],
            publisher=publisher_stats,
            track_queue_stats=per_track,
        )


class _FrameQueue:
    """Queue helper for overflow handling and optional debt-based frame selection.

    Frames can arrive in bursts even when their timestamps are evenly spaced.
    The queue absorbs those bursts.  When ``debt_skip`` is enabled the queue
    also keeps output on playback cadence by skipping intermediate frames when
    the encoder is behind, picking the next frame that jumps far enough ahead
    in time to catch up.  When not behind (or when ``debt_skip`` is disabled),
    frames are delivered in FIFO order.

    "Media-time debt" (only tracked when ``debt_skip=True``) is the running gap
    between wall-clock time spent encoding frames and media-time progress
    achieved by the frames that were encoded.

    After each successful encode:
    - media_advance_s = encoded_media_time_s - previous_encoded_media_time_s
    - debt = max(0, debt + encode_duration_s - media_advance_s)
    - example: if encode_duration=0.080s and media_advance=0.033s,
      debt increases by 0.047s for that step.

    Intuition:
    - If encoding is slower than media progress, debt grows (we are behind).
    - If media progress catches up relative to encode cost, debt shrinks.
    """

    def __init__(self, *, maxsize: int, stats: dict[str, int], debt_skip: bool = False) -> None:
        self._queue: queue.Queue[object] = queue.Queue(maxsize=maxsize)
        self._stats = stats
        self._debt_skip = debt_skip
        self._time_debt_s = 0.0
        self._last_encoded_media_time_s: Optional[float] = None
        self._stop_after_current = False
        self._stop_enqueued = False
        # Source-PTS media-time watermarks for queue-span telemetry. These are
        # best-effort snapshots (single-writer-ish updates across threads) and
        # are not locked per op. They are only meaningful for frames that
        # expose pts + time_base; STOP sentinels and pts-less frames are
        # skipped.
        self._last_put_media_time_s: Optional[float] = None
        self._first_get_media_time_s: Optional[float] = None
        self._last_get_media_time_s: Optional[float] = None

    def resize(self, maxsize: int) -> None:
        if not isinstance(maxsize, int) or isinstance(maxsize, bool):
            raise TypeError(f"queue size must be int, got {type(maxsize).__name__}")
        if maxsize <= 0:
            raise ValueError("queue size must be > 0")
        with self._queue.mutex:
            depth = self._queue._qsize()
            if maxsize < depth:
                raise ValueError(
                    f"queue size {maxsize} is smaller than current depth {depth}"
                )
            self._queue.maxsize = maxsize
            self._queue.not_full.notify_all()

    def put(self, item: object) -> None:
        if self._stop_enqueued:
            return
        max_retries = 10
        for _ in range(max_retries):
            try:
                self._queue.put_nowait(item)
                if item is _STOP:
                    self._stop_enqueued = True
                else:
                    self._track_put_media_time(item)
                return
            except queue.Full:
                try:
                    dropped = self._queue.get_nowait()
                except queue.Empty:
                    continue
                self._stats["frames_dropped_overflow"] += 1
                # Overflow drops come off the head, so their media time should
                # advance the "consumed" watermark just like a normal get.
                self._track_get_media_time(dropped)
        _LOG.error("MediaPublish frame queue put exceeded retry limit (%d); dropping item", max_retries)
        if item is not _STOP:
            self._stats["frames_dropped_overflow"] += 1

    def get_nowait(self) -> Optional[object]:
        return self.get(timeout=0.0)

    def get(self, timeout: Optional[float] = None) -> Optional[object]:
        if self._stop_after_current:
            self._stop_after_current = False
            return _STOP

        try:
            if timeout is None:
                item = self._queue.get()
            else:
                item = self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
        if item is _STOP:
            return _STOP

        if not self._debt_skip:
            self._track_get_media_time(item)
            return item

        # Candidate selection is intentionally one-at-a-time:
        # - pop one frame candidate
        # - accept it if it advances enough media time to repay current debt
        # - otherwise skip it and inspect only the next immediately available frame
        # This avoids waiting for future frames and preserves low-latency behavior.
        candidate = item
        while True:
            if self._accept_candidate(candidate):
                self._track_get_media_time(candidate)
                return candidate
            try:
                next_item = self._queue.get_nowait()
            except queue.Empty:
                # No immediate replacement candidate; keep pipeline moving.
                self._track_get_media_time(candidate)
                return candidate
            if next_item is _STOP:
                # Treat STOP as a normal FIFO sentinel, but do not drop the
                # current candidate mid-selection. Request shutdown right after.
                self._stop_after_current = True
                self._track_get_media_time(candidate)
                return candidate
            self._stats["frames_dropped_debt"] += 1
            # Debt-skipped frames still leave the queue; treat them like a get
            # for watermark tracking so queue-span math stays accurate.
            self._track_get_media_time(candidate)
            candidate = next_item

    def update_after_encode(self, *, encoded_media_time_s: float, encode_duration_s: float) -> None:
        if not self._debt_skip:
            return
        if self._last_encoded_media_time_s is None:
            self._last_encoded_media_time_s = encoded_media_time_s
            self._time_debt_s = 0.0
            return

        # Debt tracks wall-clock encode cost relative to media-time progress.
        media_advance_s = max(0.0, encoded_media_time_s - self._last_encoded_media_time_s)
        self._time_debt_s = max(0.0, self._time_debt_s + encode_duration_s - media_advance_s)
        self._last_encoded_media_time_s = encoded_media_time_s

    @property
    def time_debt_s(self) -> float:
        # Metric for "how far behind are we", in seconds.
        return self._time_debt_s

    @property
    def qsize(self) -> int:
        return self._queue.qsize()

    @property
    def maxsize(self) -> int:
        with self._queue.mutex:
            return self._queue.maxsize

    @property
    def queue_media_time_s(self) -> float:
        last_put = self._last_put_media_time_s
        last_get = self._last_get_media_time_s
        if last_put is None or last_get is None:
            return 0.0
        return max(0.0, last_put - last_get)

    @property
    def total_media_time_processed_s(self) -> float:
        first_get = self._first_get_media_time_s
        last_get = self._last_get_media_time_s
        if first_get is None or last_get is None:
            return 0.0
        return max(0.0, last_get - first_get)

    def _track_put_media_time(self, item: object) -> None:
        media_time_s = self._frame_media_time_s(item)
        if media_time_s is None:
            return
        self._last_put_media_time_s = media_time_s

    def _track_get_media_time(self, item: object) -> None:
        media_time_s = self._frame_media_time_s(item)
        if media_time_s is None:
            return
        if self._first_get_media_time_s is None:
            self._first_get_media_time_s = media_time_s
        self._last_get_media_time_s = media_time_s

    def _accept_candidate(self, candidate: object) -> bool:
        candidate_media_time_s = self._frame_media_time_s(candidate)
        if candidate_media_time_s is None or self._last_encoded_media_time_s is None:
            return True
        media_advance_s = candidate_media_time_s - self._last_encoded_media_time_s
        # Candidate must advance enough media time to cover current debt.
        return media_advance_s >= self._time_debt_s

    @staticmethod
    def _frame_media_time_s(frame: object) -> Optional[float]:
        pts = getattr(frame, "pts", None)
        time_base = getattr(frame, "time_base", None)
        if pts is None or time_base is None:
            return None
        tb = _fraction_from_time_base(time_base)
        return float(Fraction(pts) * tb)
