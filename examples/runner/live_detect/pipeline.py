"""Live object detection (YOLOv8n) + speech-to-text (faster-whisper) — multi-modal LivePipeline demo."""

from __future__ import annotations

import logging
from typing import Any

import av
import numpy as np
from faster_whisper import WhisperModel
from ultralytics import YOLO

from livepeer_gateway.runner import LivePipeline, serve
from livepeer_gateway.runner.frames import AudioFrame, VideoFrame

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
_LOG = logging.getLogger(__name__)

_TARGET_SAMPLE_RATE = 16_000  # Whisper expects 16 kHz mono float32 PCM in [-1.0, 1.0].
_WINDOW_SECONDS = 3.0  # Transcribe window; same trade-off as live_transcribe.
_DETECT_EVERY_N_FRAMES = 15  # YOLO cadence: at 15 fps → 1 inference/sec (CPU budget trade-off).
_DETECT_CONFIDENCE = 0.4  # Drop low-confidence detections before emitting.


class LiveDetect(LivePipeline):
    """Real-time object detection + transcription on the same stream."""

    def setup(self) -> None:
        # Both models loaded once at container start; baked into the image.
        _LOG.info("Loading YOLOv8n (~6 MB)...")
        self._yolo = YOLO("yolov8n.pt")
        _LOG.info("Loading whisper tiny.en (CPU, int8, ~150 MB)...")
        self._whisper = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        _LOG.info("Models loaded.")

    async def on_stream_start(self, params: dict[str, Any]) -> None:
        self._audio_buffer = np.empty(0, dtype=np.float32)
        self._transcript_index = 0
        self._frame_count = 0
        self._detection_index = 0

        # Normalise caller audio (any rate/layout/format) to whisper's input.
        self._resampler = av.AudioResampler(
            format="flt", layout="mono", rate=_TARGET_SAMPLE_RATE,
        )

        # Warm up both models — first inference is always slower.
        self._whisper.transcribe(np.zeros(_TARGET_SAMPLE_RATE, dtype=np.float32))
        self._yolo(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
        await self.emit_event({
            "type": "ready",
            "models": {"detection": "yolov8n", "transcription": "whisper-tiny.en"},
        })
        _LOG.info("Stream ready.")

    async def process_video(self, frame: VideoFrame) -> VideoFrame:
        self._frame_count += 1
        if self._frame_count % _DETECT_EVERY_N_FRAMES == 0:
            # YOLO wants RGB uint8 ndarray.
            img = frame.frame.to_ndarray(format="rgb24")
            results = self._yolo(img, verbose=False)
            objects = []
            for box in results[0].boxes:
                conf = float(box.conf.item())
                if conf < _DETECT_CONFIDENCE:
                    continue
                cls_id = int(box.cls.item())
                label = self._yolo.names[cls_id]
                bbox = [round(float(x), 1) for x in box.xyxy[0].tolist()]
                objects.append({"label": label, "conf": round(conf, 3), "bbox": bbox})
            if objects:
                await self.emit_data({
                    "type": "detection",
                    "index": self._detection_index,
                    "frame": self._frame_count,
                    "objects": objects,
                })
                _LOG.info(
                    "detection[%d] frame=%d: %s",
                    self._detection_index, self._frame_count,
                    ", ".join(f"{o['label']}({o['conf']:.2f})" for o in objects),
                )
                self._detection_index += 1
        return frame  # passthrough video — caller still gets the original frames

    async def process_audio(self, frame: AudioFrame) -> AudioFrame:
        # Resampler buffers internally — may yield 0+ frames per input.
        for resampled in self._resampler.resample(frame.frame):
            samples = resampled.to_ndarray().flatten().astype(np.float32)
            self._audio_buffer = np.concatenate([self._audio_buffer, samples])

        window = int(_TARGET_SAMPLE_RATE * _WINDOW_SECONDS)
        if self._audio_buffer.size >= window:
            await self._flush(self._audio_buffer[:window])
            self._audio_buffer = self._audio_buffer[window:]
        return frame  # passthrough audio

    async def on_stream_stop(self) -> None:
        # Flush any partial audio window. NB: under the curl-through-gateway
        # path this final emit is dropped; see README.
        if self._audio_buffer.size > 0:
            await self._flush(self._audio_buffer)
            self._audio_buffer = np.empty(0, dtype=np.float32)
        await self.emit_event({
            "type": "stopped",
            "transcripts": self._transcript_index,
            "detections": self._detection_index,
        })

    async def _flush(self, audio: np.ndarray) -> None:
        # tiny.en on CPU: ~500 ms for a 3 s window. beam_size=1 keeps it fast.
        # vad_filter=True runs Silero VAD first to skip silence inside the window.
        segments, _ = self._whisper.transcribe(audio, beam_size=1, vad_filter=True)
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            await self.emit_data({
                "type": "transcript",
                "index": self._transcript_index,
                "text": text,
                "start_ms": int(seg.start * 1000),
                "end_ms": int(seg.end * 1000),
            })
            _LOG.info("transcript[%d]: %s", self._transcript_index, text)
            self._transcript_index += 1


if __name__ == "__main__":
    serve(LiveDetect())
