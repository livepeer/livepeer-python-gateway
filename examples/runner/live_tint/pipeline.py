"""Real-time chroma tint — defaults to grayscale; ``u``/``v`` params shift tint live."""

from __future__ import annotations

import logging
from typing import Any

from livepeer_gateway.runner import LivePipeline, serve
from livepeer_gateway.runner.frames import VideoFrame

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


# 128/128 = neutral 8-bit YUV chroma → no color, i.e. grayscale.
# Shifting U cools/warms blue↔yellow; shifting V red↔green.
_NEUTRAL = 128


def _clamp_byte(value: Any, default: int = _NEUTRAL) -> int:
    try:
        return max(0, min(255, int(value)))
    except (TypeError, ValueError):
        return default


class TintFilter(LivePipeline):
    """Overwrite U/V chroma planes with caller-supplied constants.

    Default ``(u=128, v=128)`` zeroes color → grayscale. Caller can POST
    new ``u``/``v`` mid-stream (``/process/stream/{id}/update``) to shift
    the tint without restarting the session.
    """

    async def on_stream_start(self, params: dict[str, Any]) -> None:
        # Initial params arrive on /stream/start; live updates follow via
        # on_params_update. Reads from process_video are atomic int loads,
        # so no lock is needed for the single-session case.
        self._u = _clamp_byte(params.get("u"))
        self._v = _clamp_byte(params.get("v"))

    async def on_params_update(self, params: dict[str, Any]) -> None:
        if "u" in params:
            self._u = _clamp_byte(params["u"], default=self._u)
        if "v" in params:
            self._v = _clamp_byte(params["v"], default=self._v)

    async def process_video(self, frame: VideoFrame) -> VideoFrame:
        av_frame = frame.frame
        if "yuv" not in av_frame.format.name.lower():
            return frame
        # In planar YUV the U and V planes are indices 1 and 2.
        for plane_idx, value in ((1, self._u), (2, self._v)):
            plane = av_frame.planes[plane_idx]
            plane.update(bytes([value]) * (plane.line_size * plane.height))
        return frame


if __name__ == "__main__":
    serve(TintFilter())
