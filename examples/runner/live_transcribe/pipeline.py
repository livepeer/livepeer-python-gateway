"""Live speech-to-text via faster-whisper tiny.en — transcripts emitted on the data channel."""

from __future__ import annotations

import logging
from typing import Any

import av
import numpy as np
from faster_whisper import WhisperModel

from livepeer_gateway.runner import LivePipeline, serve
from livepeer_gateway.runner.frames import AudioFrame

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
_LOG = logging.getLogger(__name__)

# Whisper expects 16 kHz mono float32 PCM in [-1.0, 1.0].
_TARGET_SAMPLE_RATE = 16_000

# Buffer ~3s of audio per transcribe pass — tiny.en runs faster than real-time
# on CPU, so 3s windows yield transcripts every ~3s with sub-second inference lag.
_WINDOW_SECONDS = 3.0


class LiveTranscribe(LivePipeline):
    """Streaming speech-to-text via faster-whisper tiny.en."""

    def setup(self) -> None:
        # Loaded once at container start; ~150 MB int8-quantized.
        _LOG.info("Loading whisper tiny.en (CPU, int8)...")
        self._model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        _LOG.info("Whisper model loaded.")

    async def on_stream_start(self, params: dict[str, Any]) -> None:
        # Per-session state — fresh on every stream/start.
        self._buffer = np.empty(0, dtype=np.float32)
        self._segment_index = 0

        # Normalise caller audio (any rate/layout/format) to whisper's input.
        self._resampler = av.AudioResampler(
            format="flt", layout="mono", rate=_TARGET_SAMPLE_RATE,
        )
        
        # Warm-up pass — primes caches before the first real audio window.
        self._model.transcribe(np.zeros(_TARGET_SAMPLE_RATE, dtype=np.float32))
        await self.emit_event({"type": "ready", "model": "whisper-tiny.en"})
        _LOG.info("Stream ready.")

    async def process_audio(self, frame: AudioFrame) -> AudioFrame:
        # Resampler buffers internally — may yield 0+ frames per input.
        for resampled in self._resampler.resample(frame.frame):
            samples = resampled.to_ndarray().flatten().astype(np.float32)
            self._buffer = np.concatenate([self._buffer, samples])

        window = int(_TARGET_SAMPLE_RATE * _WINDOW_SECONDS)
        if self._buffer.size >= window:
            await self._flush(self._buffer[:window])
            self._buffer = self._buffer[window:]

        return frame  # passthrough — caller still gets the audio track

    async def on_stream_stop(self) -> None:
        # Flush any partial window so the last few words make it out.
        if self._buffer.size > 0:
            await self._flush(self._buffer)
            self._buffer = np.empty(0, dtype=np.float32)
        await self.emit_event({"type": "stopped", "segments": self._segment_index})

    async def _flush(self, audio: np.ndarray) -> None:
        # tiny.en on CPU: ~500 ms for a 3 s window. beam_size=1 keeps it fast.
        # vad_filter=True runs Silero VAD first to skip silence inside the window.
        segments, _ = self._model.transcribe(audio, beam_size=1, vad_filter=True)
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            await self.emit_data({
                "type": "transcript",
                "index": self._segment_index,
                "text": text,
                "start_ms": int(seg.start * 1000),
                "end_ms": int(seg.end * 1000),
            })
            _LOG.info("transcript[%d]: %s", self._segment_index, text)
            self._segment_index += 1


if __name__ == "__main__":
    serve(LiveTranscribe())
