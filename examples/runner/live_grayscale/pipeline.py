"""Real-time grayscale filter — minimal LivePipeline E2E example.

Zeroes the U/V chroma planes of each video frame. Audio passes through
unchanged. Run via ``docker compose up`` — see README.md.
"""

from livepeer_gateway.runner import LivePipeline, serve
from livepeer_gateway.runner.frames import VideoFrame


# 128 = neutral chroma in 8-bit YUV (Y=luma, U/V=chroma offsets from 128).
_NEUTRAL_CHROMA = 128


class GrayscaleFilter(LivePipeline):
    """Strip chroma; output looks black-and-white."""

    async def process_video(self, frame: VideoFrame) -> VideoFrame:
        av_frame = frame.frame
        if "yuv" in av_frame.format.name.lower():
            # In planar YUV the U and V planes are indices 1 and 2.
            for plane_idx in (1, 2):
                plane = av_frame.planes[plane_idx]
                plane.update(bytes([_NEUTRAL_CHROMA]) * (plane.line_size * plane.height))
        return frame


if __name__ == "__main__":
    serve(GrayscaleFilter())
