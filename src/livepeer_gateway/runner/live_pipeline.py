from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from .pipeline import PipelineState

if TYPE_CHECKING:
    # Gated to keep PyAV out of the import path for batch Pipeline users.
    from ..media_decode import AudioDecodedMediaFrame, VideoDecodedMediaFrame


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
    params: dict[str, Any] = Field(default_factory=dict)


class StreamParamsRequest(BaseModel):
    """Body of ``POST /stream/params`` — passthrough JSON params from the caller."""

    model_config = ConfigDict(extra="allow")


class LivePipeline:
    """Base class for real-time A/V pipelines on the BYOC trickle protocol.

    Subclasses override any of the lifecycle / processing hooks below.
    A subclass that overrides nothing is a valid passthrough relay.
    """

    _state: PipelineState = PipelineState.LOADING

    def setup(self) -> None:
        """Hook called once before serve() accepts requests.

        Sync, container-init time. Override to load model weights, warm up GPUs.
        """

    async def on_stream_start(self, params: dict[str, Any]) -> None:
        """Called when a new stream session begins, before the first frame.

        `params` is the initial pipeline params from the caller.
        """

    async def process_video(
        self, frame: VideoDecodedMediaFrame
    ) -> VideoDecodedMediaFrame:
        """Transform one decoded video frame. Default: passthrough."""
        return frame

    async def process_audio(
        self, frame: AudioDecodedMediaFrame
    ) -> AudioDecodedMediaFrame:
        """Transform one decoded audio frame. Default: passthrough."""
        return frame

    async def on_params_update(self, params: dict[str, Any]) -> None:
        """Called when the caller posts new params mid-stream."""

    async def on_stream_stop(self) -> None:
        """Called when the stream session ends — for per-session cleanup."""

    async def emit_event(self, payload: dict[str, Any]) -> None:
        """Publish a JSON event on the events trickle channel.

        Bound at session start; calling outside an active session is a no-op.
        """
        # TODO: passthrough for now; wire up in next phase.
        return None

    async def emit_data(self, payload: dict[str, Any]) -> None:
        """Publish a JSON record on the data trickle channel (when enabled).

        Bound at session start; calling outside an active session is a no-op.
        """
        # TODO: passthrough for now; wire up in next phase.
        return None
