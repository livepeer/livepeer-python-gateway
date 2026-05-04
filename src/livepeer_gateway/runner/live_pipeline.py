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

    async def process_video(self, frame: VideoFrame) -> VideoFrame:
        """Transform one decoded video frame. Default: passthrough."""
        return frame

    async def process_audio(self, frame: AudioFrame) -> AudioFrame:
        """Transform one decoded audio frame. Default: passthrough."""
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

    try:
        await pipeline.on_stream_start(session.params)
    except Exception:
        _LOG.exception("LivePipeline on_stream_start failed")
        return

    async with MediaOutput(session.subscribe_url) as media_output:
        media_publish = MediaPublish(session.publish_url)
        try:
            async for decoded in media_output.frames():
                is_video = isinstance(decoded, VideoDecodedMediaFrame)
                try:
                    if is_video:
                        result = await pipeline.process_video(decoded)
                    else:
                        result = await pipeline.process_audio(decoded)
                except Exception:
                    method = "process_video" if is_video else "process_audio"
                    _LOG.exception("LivePipeline %s failed", method)
                    continue
                if result is not None:
                    await media_publish.write_frame(result.frame)
        finally:
            await media_publish.close()
