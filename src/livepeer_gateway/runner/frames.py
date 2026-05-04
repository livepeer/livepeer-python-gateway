"""User-facing frame types for ``LivePipeline``.

Importing this submodule pulls in PyAV (via ``media_decode``). Pipeline
authors who override ``process_video`` / ``process_audio`` opt in to that
cost by importing here; ``from livepeer_gateway.runner import LivePipeline``
alone stays PyAV-free for batch ``Pipeline`` users.
"""

from ..media_decode import AudioDecodedMediaFrame as AudioFrame
from ..media_decode import VideoDecodedMediaFrame as VideoFrame

__all__ = ["AudioFrame", "VideoFrame"]
