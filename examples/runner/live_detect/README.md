# Live detect (BYOC, real-time, multi-modal)

> [!NOTE]
> **TODO** — `test.sh`, `demo.sh`, `_format_records.py`, and the `gateway:` compose service collapse into a single Python script using the client SDK once [livepeer/livepeer-python-gateway#6](https://github.com/livepeer/livepeer-python-gateway/pull/6) merges.

Real-time **object detection + speech-to-text** on the same stream.
Detection via [YOLOv8n](https://github.com/ultralytics/ultralytics) (~6 MB,
CPU-fast); transcription via [faster-whisper tiny.en](https://github.com/SYSTRAN/faster-whisper)
(~150 MB, int8-quantized). Both pipelines emit on the same `data` trickle
channel as separate record types — `{"type": "detection", ...}` and
`{"type": "transcript", ...}`.

The canonical example for **multi-modal LivePipeline** — exercises both
`process_video` and `process_audio` overrides on the same pipeline class:

| Hook | What this example does |
| --- | --- |
| `setup()` | Load YOLOv8n + whisper tiny.en at container start (both baked into the image) |
| `on_stream_start(params)` | Reset per-session state, build resampler, warm both models with dummy input, emit `ready` event |
| `process_video(frame)` | Run YOLO every 15th frame (~1 inference/sec at 15 fps); emit `detection` records via `emit_data` |
| `process_audio(frame)` | Resample to 16 kHz mono → buffer 3 s windows → whisper transcribe → emit `transcript` records via `emit_data` |
| `emit_data({...})` | Two record types interleave on the same `data_url` stream |
| `on_stream_stop()` | Flush partial audio buffer + emit final stats event |

Both `process_*` are overridden, so the SDK's track-allocation-by-introspection
allocates **both** output tracks. Both return the input frame as passthrough —
the caller sees the original video + audio plus the structured detection /
transcript records on the data channel.

## Run

> [!WARNING]
> Only one example can run at a time — all share container names
> (`gateway`, `orchestrator`, …) and ports (`1935`, `9935`, `5000`). If
> `./test.sh` fails at the capability-registration step, run `docker
> compose down` in the other example's directory first.

```bash
docker compose up -d --wait --build      # both models bake into the image (~250 MB total)

./test.sh                                 # CI: composited dog/cat photo + JFK audio
./demo.sh                                 # interactive: webcam + mic, see records live

docker compose down
```

The pipeline container starts in `LOADING` state until `setup()` finishes
loading both models. Healthcheck reflects this — first `up` takes ~20s while
the YOLOv8n + whisper-tiny.en weights stream from the bake layer.

### `test.sh` — automated assertion

Auto-downloads two public-domain fixtures into `assets/` on first run:

- **Video** — OpenCV's canonical [`vtest.avi`](https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi) (~8 MB, 80s plaza-pedestrian clip, looped)
- **Audio** — JFK's "Ask not..." inaugural (`jfk.flac`, also used by `live_transcribe`)

Composites both into a single FLV push. Asserts:

- At least one `detection` record contains `person` (vtest reliably yields 5-9 per frame)
- At least one `transcript` record contains `country` (the JFK speech says it three times)

Override either fixture with environment variables:

```bash
VIDEO_URL=https://example.com/your-video.mp4 ./test.sh
AUDIO_URL=https://example.com/your-audio.flac ./test.sh
```

### `demo.sh` — interactive webcam + mic

Pushes audio + video from your webcam and microphone for `DURATION` seconds
(default 60) and prints both detections and transcripts to the terminal as
they arrive. Show things to the camera, speak — `Ctrl-C` to stop early.

```bash
DURATION=30 ./demo.sh                          # shorter session
PULSE_SOURCE=alsa_input.usb-... ./demo.sh      # specific Linux mic
VIDEO_DEVICE=/dev/video1 ./demo.sh             # specific Linux webcam
MAC_VIDEO_DEVICE=1 MAC_AUDIO_DEVICE=0 ./demo.sh   # macOS — pick devices
```

Sample output while looking at a coffee mug and speaking:

```text
[T  0] Hello, this is a test.
[D  0] frame=15 person(0.91), cup(0.78)
[D  1] frame=30 person(0.93), cup(0.79), laptop(0.62)
[T  1] I'm just talking and showing things to the camera.
[D  2] frame=45 person(0.92), cup(0.81)
```

Each line is one record from the `data_url` SSE stream — `[T#]` for transcripts,
`[D#]` for detections. The same payload a real caller would receive.

## Throughput

| Component | CPU cost (per call) | Cadence | Total CPU budget |
| --- | --- | --- | --- |
| YOLOv8n (640×640, CPU) | ~30 ms | every 15th frame (~1 Hz) | ~30 ms/sec |
| whisper tiny.en (3 s window, CPU int8) | ~500 ms | every 3 s | ~170 ms/sec |
| Combined | — | — | ~200 ms/sec real-time-friendly |

Both run faster than real-time at 15 fps video + 16 kHz audio on a modest
CPU. To trade latency for accuracy, raise the YOLO cadence (run every frame),
swap to `yolov8s` / `yolov8m` (larger models), or use whisper `base.en`.

## Going further

- **Better video understanding** — for natural-language scene descriptions,
  swap YOLOv8n for a small vision-language model like
  [Moondream2](https://huggingface.co/vikhyatk/moondream2) (1.6 GB, ~300 ms
  per inference on GPU). Same `LivePipeline` lifecycle, different model in
  `process_video`. Out of scope for this example — separate `live_describe`
  example tracks.
- **Multi-class filtering** — restrict emissions to a class subset (e.g.
  only `person` + `cat` + `dog`) by filtering on `label` before `emit_data`.
- **Bounding box overlays** — modify `process_video` to draw boxes on the
  outgoing frame using OpenCV / PyAV before returning, so subscribers see
  annotated video.
- **Production-grade transcription** — see `live_transcribe`'s README
  "Going further" section for `whisper_streaming` / WhisperLive / native
  streaming ASR pointers. Same model swap applies here.
