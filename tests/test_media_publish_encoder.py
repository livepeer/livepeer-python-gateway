"""Unit tests for the codec-aware encoder options + VideoOutputConfig fields.

Pure logic — no ffmpeg/trickle needed:
- x264/x265 keep the original low-latency option set (backward compatible).
- other codecs start from empty base options so they open cleanly (the old
  hardcoded x264-only options broke non-x264 encoders at avcodec_open2).
- the new bit_rate / profile / encoder_options fields default to "no change".
"""
from __future__ import annotations

from livepeer_gateway.media_publish import (
    VideoOutputConfig,
    _encoder_base_options,
    _X264_LOWLATENCY,
)


def test_x264_family_keeps_lowlatency_defaults():
    assert _encoder_base_options("libx264") == _X264_LOWLATENCY
    assert _encoder_base_options("libx265") == _X264_LOWLATENCY
    # returns a copy, not the shared dict (callers mutate it)
    opts = _encoder_base_options("libx264")
    opts["preset"] = "veryfast"
    assert _X264_LOWLATENCY["preset"] == "superfast"


def test_non_x264_codecs_get_empty_base_options():
    for codec in ("libsvtav1", "av1_nvenc", "libvpx-vp9", "h264_nvenc", "hevc_nvenc"):
        assert _encoder_base_options(codec) == {}


def test_video_output_config_defaults_are_backward_compatible():
    cfg = VideoOutputConfig()
    assert cfg.bit_rate is None
    assert cfg.profile is None
    assert cfg.encoder_options is None
    assert cfg.codec == "libx264"


def test_video_output_config_accepts_new_fields():
    cfg = VideoOutputConfig(
        codec="av1_nvenc",
        bit_rate=3_000_000,
        profile="high",
        encoder_options={"preset": "p4", "tune": "ll"},
    )
    assert cfg.bit_rate == 3_000_000
    assert cfg.profile == "high"
    assert cfg.encoder_options == {"preset": "p4", "tune": "ll"}
