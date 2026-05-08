from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from ..trickle_publisher import TricklePublisher
from ..trickle_subscriber import TrickleSubscriber
from .pipeline import PipelineState

if TYPE_CHECKING:
    # Gated to keep PyAV out of the import path for batch Pipeline users.
    from .frames import AudioFrame, VideoFrame


_LOG = logging.getLogger(__name__)

# 10s under the gateway's 30s events-gap watchdog (byoc/trickle.go: maxEventGap).
_HEARTBEAT_INTERVAL_S = 10

# cancel-then-wait bound; tasks beyond this are left to finish in the background
_STOP_TIMEOUT_S = 5.0


class StreamStartRequest(BaseModel):
    """Body of ``POST /stream/start`` — sent by the orchestrator.

    `subscribe_url`, `publish_url`, `data_url` are absent when the orchestrator
    has the corresponding `EnableVideoIngress` / `EnableVideoEgress` /
    `EnableDataOutput` flag disabled — the runner must tolerate any subset.
    """

    model_config = ConfigDict(extra="allow")

    gateway_request_id: str
    control_url: str
    events_url: str
    subscribe_url: str | None = None
    publish_url: str | None = None
    data_url: str | None = None
    params: dict[str, Any] | None = None  # null when caller sent no params


class StreamParamsRequest(BaseModel):
    """Body of ``POST /stream/params`` — passthrough JSON params from the caller."""

    model_config = ConfigDict(extra="allow")


class _LiveSession:
    """Per-session state held in `LivePipeline._session`.

    Constructed on `/stream/start`, cleared on `/stream/stop`.
    """

    def __init__(
        self,
        *,
        gateway_request_id: str,
        events_url: str,
        subscribe_url: str | None,
        publish_url: str | None,
        data_url: str | None,
        params: dict[str, Any] | None,
    ) -> None:
        self.gateway_request_id = gateway_request_id
        self.events_url = events_url
        self.subscribe_url = subscribe_url
        self.publish_url = publish_url
        self.data_url = data_url
        self.params: dict[str, Any] = params or {}
        self.events_publisher: TricklePublisher | None = None
        self.data_publisher: TricklePublisher | None = None
        self.task: asyncio.Task[None] | None = None
        self.heartbeat_task: asyncio.Task[None] | None = None

    async def cancel_tasks(self) -> None:
        """Cancel the heartbeat + frame-loop tasks; leave publishers open."""
        if self.heartbeat_task and not self.heartbeat_task.done():
            self.heartbeat_task.cancel()
            try:
                await asyncio.wait_for(self.heartbeat_task, timeout=_STOP_TIMEOUT_S)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            except Exception:
                _LOG.exception("LivePipeline heartbeat task ended with error")
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await asyncio.wait_for(self.task, timeout=_STOP_TIMEOUT_S)
            except asyncio.TimeoutError:
                _LOG.warning(
                    "LivePipeline session task did not terminate within %.1fs",
                    _STOP_TIMEOUT_S,
                )
            except asyncio.CancelledError:
                pass
            except Exception:
                _LOG.exception("LivePipeline session task ended with error")

    async def close_publishers(self) -> None:
        """Close events + data publishers. Call after user cleanup runs."""
        for label, pub in (
            ("events", self.events_publisher),
            ("data", self.data_publisher),
        ):
            if pub is None:
                continue
            try:
                await pub.close()
            except Exception:
                _LOG.warning(
                    "LivePipeline %s_publisher close failed", label, exc_info=True
                )

    async def close(self) -> None:
        """Tear down everything. Idempotent and tolerant of partial setup."""
        await self.cancel_tasks()
        await self.close_publishers()


class LivePipeline:
    """Base class for real-time A/V pipelines on the BYOC trickle protocol.

    Subclasses override any of the lifecycle / processing hooks below.
    A subclass that overrides nothing is a valid passthrough relay.
    """

    _state: PipelineState = PipelineState.LOADING
    # Single-session for now; multi-session is post-C8 (capacity demand-driven).
    _session: _LiveSession | None = None

    def setup(self) -> None:
        """Hook called once before serve() accepts requests.

        Sync, container-init time. Override to load model weights, warm up GPUs.
        """

    async def on_stream_start(self, params: dict[str, Any]) -> None:
        """Called when a new stream session begins, before the first frame.

        `params` is the initial pipeline params from the caller.
        """

    async def process_video(self, frame: VideoFrame) -> VideoFrame | None:
        """Transform one decoded video frame. Default: passthrough.

        The video output track is allocated when this method is overridden,
        or when neither `process_*` hook is overridden (a stock LivePipeline
        is a full passthrough relay). Override only `process_audio` and the
        video track is not allocated; this method is not called.

        Return `None` to drop the frame.
        """
        return frame

    async def process_audio(self, frame: AudioFrame) -> AudioFrame | None:
        """Transform one decoded audio frame. Default: passthrough.

        The audio output track is allocated when this method is overridden,
        or when neither `process_*` hook is overridden (a stock LivePipeline
        is a full passthrough relay). Override only `process_video` and the
        audio track is not allocated; this method is not called.

        Return `None` to drop the frame — useful for pipelines that consume
        audio without re-emitting (e.g. transcription, where results go out
        via `emit_data` on the data channel instead).
        """
        return frame

    async def on_params_update(self, params: dict[str, Any]) -> None:
        """Called when the caller posts new params mid-stream."""

    async def on_stream_stop(self) -> None:
        """Called when the stream session ends — for per-session cleanup."""

    async def emit_event(self, payload: dict[str, Any]) -> None:
        """Publish a JSON event on the events trickle channel.

        Use for telemetry, lifecycle signals, or runner-side observability:

            await self.emit_event({"type": "model_loaded", "version": "v1.2"})

        No-op outside an active session.
        """
        session = self._session
        if session is None or session.events_publisher is None:
            return
        await _publish_json(session.events_publisher, payload, "emit_event")

    async def emit_data(self, payload: dict[str, Any]) -> None:
        """Publish a JSON record on the data trickle channel.

        Use for structured pipeline output the caller subscribes to (e.g.
        transcripts, detections, classification results):

            await self.emit_data({"text": segment.text, "start_ms": ...})

        No-op outside an active session.
        """
        session = self._session
        if session is None or session.data_publisher is None:
            return
        await _publish_json(session.data_publisher, payload, "emit_data")


async def _publish_json(
    publisher: TricklePublisher, payload: dict[str, Any], context: str
) -> None:
    """Write a JSON object as one trickle segment. Errors are logged, not raised."""
    try:
        async with await publisher.next() as writer:
            await writer.write(json.dumps(payload).encode())
    except Exception:
        _LOG.warning("%s publish failed", context, exc_info=True)


async def _run_heartbeat_loop(pipeline: LivePipeline) -> None:
    """Publish a heartbeat to events_url every `_HEARTBEAT_INTERVAL_S`.

    Without this, the gateway's 30s events-gap watchdog kills the session.
    Per-publish errors are logged but don't kill the loop.
    """
    session = pipeline._session
    if session is None or session.events_publisher is None:
        return
    pub = session.events_publisher
    try:
        while True:
            await _publish_json(
                pub,
                {"type": "heartbeat", "timestamp": int(time.time() * 1000)},
                "heartbeat",
            )
            await asyncio.sleep(_HEARTBEAT_INTERVAL_S)
    except asyncio.CancelledError:
        return


def _has_user_processing(pipeline: LivePipeline) -> bool:
    """True if the pipeline overrides ``process_video`` or ``process_audio``.

    Used by ``/stream/start`` to pick the cheap bytes path when nothing's
    being transformed; otherwise the full decode → user → encode loop runs.
    """
    cls = type(pipeline)
    return (
        cls.process_video is not LivePipeline.process_video
        or cls.process_audio is not LivePipeline.process_audio
    )


async def _run_passthrough(subscribe_url: str, publish_url: str) -> None:
    """Forward bytes from a subscribe URL to a publish URL, unmodified.

    Each inbound trickle segment becomes one outbound segment (1:1) — never
    merged or split, so downstream consumers see the same segment count and
    ordering as the upstream sender. Returns when the subscribe channel ends
    (orchestrator deletes it → 404 from the trickle server), or when the
    task is cancelled / either side raises.
    """
    sub = TrickleSubscriber(subscribe_url)
    try:
        # MIME must match go-livepeer's publish channel (stream_orchestrator.go).
        async with TricklePublisher(publish_url, mime_type="video/MP2T") as pub:
            while True:
                segment = await sub.next()
                if segment is None:  # EOS — channel ended
                    return
                try:
                    reader = segment.make_reader()
                    async with await pub.next() as writer:
                        while True:
                            chunk = await reader.read()
                            if not chunk:
                                break
                            await writer.write(chunk)
                finally:
                    await segment.close()
    finally:
        await sub.close()


def _emit_flags(pipeline: LivePipeline) -> tuple[bool, bool]:
    """Decide which output tracks the pipeline emits on, by introspection.

    A track is emitted iff its `process_*` hook is overridden — except a
    stock LivePipeline (no overrides) emits both as a passthrough relay.
    """
    cls = type(pipeline)
    audio_overridden = cls.process_audio is not LivePipeline.process_audio
    video_overridden = cls.process_video is not LivePipeline.process_video
    if not (audio_overridden or video_overridden):
        return True, True
    return video_overridden, audio_overridden


def _build_media_publish_config(emit_video: bool, emit_audio: bool):
    """Allocate only the tracks the pipeline will emit on.

    Tracks where the pipeline's `process_*` returns `None` for every frame
    are still allocated briefly, then closed by `track_wait_timeout_s`.
    """
    from ..media_publish import AudioOutputConfig, MediaPublishConfig, VideoOutputConfig

    tracks: list = []
    if emit_video:
        tracks.append(VideoOutputConfig())
    if emit_audio:
        tracks.append(AudioOutputConfig())
    return MediaPublishConfig(tracks=tracks, track_wait_timeout_s=0.5)


async def _run_frame_loop(pipeline: LivePipeline) -> None:
    """Decode → user transform → encode loop.

    Per-frame errors drop the frame and continue; subscribe/publish errors
    end the session.
    """
    # Lazy import: defers PyAV until a pipeline actually needs the frame loop.
    from ..media_decode import VideoDecodedMediaFrame
    from ..media_output import MediaOutput
    from ..media_publish import MediaPublish

    session = pipeline._session  # set by serve.py before scheduling.
    emit_video, emit_audio = _emit_flags(pipeline)  # Determine required tracks.

    async with MediaOutput(session.subscribe_url) as media_output:
        media_publish = MediaPublish(
            session.publish_url,
            config=_build_media_publish_config(emit_video, emit_audio),
        )
        try:
            async for decoded in media_output.frames():
                is_video = isinstance(decoded, VideoDecodedMediaFrame)
                # Skip kinds we don't emit — input may carry them, but
                # writing to a non-allocated MediaPublish track raises.
                if not (emit_video if is_video else emit_audio):
                    continue
                try:
                    if is_video:
                        result = await pipeline.process_video(decoded)
                    else:
                        result = await pipeline.process_audio(decoded)
                except Exception:
                    _LOG.exception("LivePipeline %s failed", "process_video" if is_video else "process_audio")
                    continue
                if result is not None:
                    await media_publish.write_frame(result.frame)
        finally:
            await media_publish.close()
