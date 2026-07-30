from __future__ import annotations

import math
from array import array

from livepeer_gateway import multi_track_verify as verify_mod


class TestMultiTrackVerifyHelper:
    def _tone(
        self,
        *,
        frequency_hz: float,
        sample_rate: int,
        duration_s: float,
        amplitude_fn,
    ) -> array:
        out = array("f")
        total_samples = int(sample_rate * duration_s)
        for index in range(total_samples):
            t_s = index / float(sample_rate)
            amplitude = float(amplitude_fn(t_s))
            out.append(amplitude * math.sin(2.0 * math.pi * frequency_hz * t_s))
        return out

    def test_goertzel_prefers_target_frequency(self) -> None:
        sample_rate = 48_000
        samples = self._tone(
            frequency_hz=440.0,
            sample_rate=sample_rate,
            duration_s=1.0,
            amplitude_fn=lambda _t: 0.75,
        )
        power_440 = verify_mod._goertzel_power(samples, sample_rate, 440.0)
        power_880 = verify_mod._goertzel_power(samples, sample_rate, 880.0)
        assert power_440 > power_880 * 20.0

    def test_verify_audio_track_accepts_expected_beep_pattern(self) -> None:
        spec = verify_mod.default_audio_specs(sample_rate=48_000)[0]
        observed = verify_mod.ObservedAudioTrack(
            stream_index=7, sample_rate=spec.sample_rate
        )
        observed.samples = self._tone(
            frequency_hz=spec.frequency_hz,
            sample_rate=spec.sample_rate,
            duration_s=2.0,
            amplitude_fn=lambda t: (
                spec.base_amplitude if int(t / spec.gate_period_s) % 2 == 0 else 0.03
            ),
        )
        observed.frame_count = 10
        result = verify_mod._verify_audio_track(spec, observed)
        assert result.ok, result.message
        assert (result.target_power or 0.0) > (
            result.strongest_other_power or 0.0
        ) * 2.5

    def test_match_video_tracks_uses_average_color_signature(self) -> None:
        red_spec, green_spec = verify_mod.default_video_specs()
        red_track = verify_mod.ObservedVideoTrack(
            stream_index=9,
            frames=[
                verify_mod.VideoFrameObservation(
                    pts_time=0.0,
                    mean_rgb=(170.0, 36.0, 38.0),
                    marker_centroids={},
                )
            ],
        )
        green_track = verify_mod.ObservedVideoTrack(
            stream_index=4,
            frames=[
                verify_mod.VideoFrameObservation(
                    pts_time=0.0,
                    mean_rgb=(40.0, 150.0, 52.0),
                    marker_centroids={},
                )
            ],
        )
        matched = verify_mod._match_video_tracks(
            {9: red_track, 4: green_track},
            [green_spec, red_spec],
        )
        assert matched[red_spec.name].stream_index == 9
        assert matched[green_spec.name].stream_index == 4
