"""Real-time monocular depth via DepthAnything V2 on GPU — bright = close, dark = far."""

from __future__ import annotations

import logging

import cv2
import numpy as np
import torch
from transformers import pipeline as hf_pipeline

from livepeer_gateway.runner import LivePipeline, serve
from livepeer_gateway.runner.frames import VideoFrame

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
_LOG = logging.getLogger(__name__)

_MODEL_ID = "depth-anything/Depth-Anything-V2-Base-hf"

# 128 = neutral 8-bit YUV chroma; keeps output grayscale so depth-as-luma reads clean.
_NEUTRAL_CHROMA = 128


class LiveDepth(LivePipeline):
    """Per-frame monocular depth estimation; bright = close, dark = far."""

    def setup(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        _LOG.info("Loading %s on %s (dtype=%s)...", _MODEL_ID, device, dtype)
        self._pipe = hf_pipeline(
            task="depth-estimation",
            model=_MODEL_ID,
            device=device,
            torch_dtype=dtype,
        )

        # Warm pass so first real frame doesn't include compile / cache cost.
        from PIL import Image
        self._pipe(Image.fromarray(np.zeros((384, 384, 3), dtype=np.uint8)))
        _LOG.info("Model loaded.")

    async def process_video(self, frame: VideoFrame) -> VideoFrame:
        av_frame = frame.frame
        if "yuv" not in av_frame.format.name.lower():
            return frame  # passthrough for unexpected pixel formats

        # HF depth-estimation returns {"depth": PIL.Image uint8 auto-normalized
        # for display, "predicted_depth": raw tensor}; we want the visualization.
        depth_pil = self._pipe(av_frame.to_image())["depth"]
        depth = np.asarray(depth_pil, dtype=np.uint8)

        # Defensive: HF normally returns the depth at the input size already.
        if depth.shape != (av_frame.height, av_frame.width):
            depth = cv2.resize(depth, (av_frame.width, av_frame.height))

        # Write depth into the Y plane and zero chroma so the output reads as grayscale.
        y_plane = av_frame.planes[0]
        if y_plane.line_size == av_frame.width:
            y_plane.update(depth.tobytes())
        else:
            buf = np.zeros((y_plane.height, y_plane.line_size), dtype=np.uint8)
            buf[: depth.shape[0], : depth.shape[1]] = depth
            y_plane.update(buf.tobytes())
        for plane_idx in (1, 2):
            plane = av_frame.planes[plane_idx]
            plane.update(bytes([_NEUTRAL_CHROMA]) * (plane.line_size * plane.height))

        return frame


if __name__ == "__main__":
    serve(LiveDepth())
