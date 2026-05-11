# Live transcribe (BYOC, real-time)

> [!NOTE]
> **TODO** — `test.sh`, `demo.sh`, `_format_records.py`, and the `gateway:` compose service collapse into a single Python script using the client SDK once [livepeer/livepeer-python-gateway#6](https://github.com/livepeer/livepeer-python-gateway/pull/6) merges. The gateway-path on_stream_stop drop disappears with the migration.

Streaming speech-to-text via [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
running [tiny.en](https://huggingface.co/Systran/faster-whisper-tiny.en).
Push audio (or video-with-audio) into the BYOC pipeline; transcripts arrive
on the data trickle channel as structured JSON. Companion to
[`live_tint`](../live_tint/), which exercises only `process_video`
— this example exercises **the full LivePipeline lifecycle**:

| Hook | What this example does |
| --- | --- |
| `setup()` | Load the whisper model once at container start |
| `on_stream_start(params)` | Build a per-session resampler, warm the model with silence, emit `ready` event |
| `process_audio(frame)` | Resample to 16 kHz mono → buffer → transcribe each ~3 s window |
| `emit_data({...})` | Publish each transcript as JSON on `data_url` |
| `emit_event({...})` | Publish lifecycle signals (`ready`, `stopped`) on `events_url` |
| `on_stream_stop()` | Flush partial buffer so the last words are still emitted |

No `process_video` override — video passes through unchanged.

## Run

> [!WARNING]
> Only one example can run at a time — all share container names
> (`gateway`, `orchestrator`, …) and ports (`1935`, `9935`, `5000`). If
> `./test.sh` fails at the capability-registration step, run `docker
> compose down` in the other example's directory first.

```bash
docker compose up -d --wait --build      # whisper model bakes into the image

./test.sh                                 # CI: canned JFK clip, asserts transcripts
./demo.sh                                 # interactive: speak into your mic, see transcripts live

docker compose down
```

The pipeline container starts in `LOADING` state until `setup()` completes
and `on_stream_start()` warms the model. Healthcheck reflects this — give
it some time on the first `up` while the model loads.

### `test.sh` — automated assertion

Pushes a known English speech clip (JFK's "Ask not…" inaugural, ~11 s,
auto-downloaded to `assets/jfk.flac` on first run) through the BYOC stack,
subscribes to the gateway's data SSE proxy, and asserts that JSON
transcript records arrive containing recognisable words from the speech
("country" appears three times in the source). Catches both runtime
regressions and accidental no-op changes to `process_audio`.

### `demo.sh` — interactive microphone

Pushes audio from your microphone for `DURATION` seconds (default 60) and
prints transcripts to the terminal as whisper finishes each 3 s window.
Speak naturally — Ctrl-C to stop early.

```bash
DURATION=30 ./demo.sh                     # shorter session
PULSE_SOURCE=alsa_input.usb-...  ./demo.sh   # specific Linux mic (pactl list short sources)
MAC_AUDIO_DEVICE=1 ./demo.sh                  # macOS — pick a non-default device
```

Sample output while speaking:

```text
[  0] Hello, this is a test of the transcription pipeline.
[  1] I'm just speaking some words to verify it works.
[  2] And so it goes.
```

Each line is one record from the BYOC `data_url` SSE stream — the same
payload a real caller would receive.

**Troubleshooting `demo.sh`:**

- Records but no transcripts → microphone not captured. On Linux,
  `pactl list short sources` to find a usable source name, then
  `PULSE_SOURCE=<name> ./demo.sh`. On macOS, list devices with
  `ffmpeg -f avfoundation -list_devices true -i ""` and pick the index
  via `MAC_AUDIO_DEVICE=N ./demo.sh`.
- "FAIL: register_capability hasn't logged success" → the stack isn't
  fully up yet. Wait for `docker compose up -d --wait --build` to finish.

## Throughput

`tiny.en` on CPU runs faster than real-time for English speech but is the
**slowest part of the pipeline**. The `_WINDOW_SECONDS = 3.0` constant in
`pipeline.py` controls the trade-off:

| Window | First transcript | CPU usage | Trade-off |
| --- | --- | --- | --- |
| 1.5 s | ~1.5 s | high | Snappier; risk falling behind on slow CPUs |
| 3 s (default) | ~3 s | moderate | Balanced |
| 5 s | ~5 s | low | Smoother on weak hardware |

For non-English speech, swap `tiny.en` → `tiny` in `setup()` (multi-lingual,
slightly larger). For higher quality on a faster CPU, use `base.en` or
`small.en`.

## Going further (production-grade live transcription)

This example is **not tuned for real-time latency** — it's the canonical
`LivePipeline` lifecycle demo. The 3 s fixed-window chunking is intentionally
visible code so readers can see "what does live A/V processing look like
inside `process_audio`." For a production transcription service you'd reach
for one of:

- **[`whisper_streaming`](https://github.com/ufal/whisper_streaming)** — a
  streaming wrapper around faster-whisper using LocalAgreement-2 to emit
  tokens once consecutive overlapping windows agree. Drops first-transcript
  latency to ~1 s and avoids mid-word splits at window edges.
- **[WhisperLive](https://github.com/collabora/WhisperLive)** — server-style
  streaming with WebSocket / WebRTC and built-in VAD.
- **Native streaming ASR** — e.g. NVIDIA's
  [Canary](https://huggingface.co/nvidia/canary-1b) or
  [Conformer-CTC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_conformer_ctc_large)
  models, which are streaming by architecture (no chunking workaround).

Whisper itself isn't natively streaming — its encoder applies full self-
attention over the entire input, so true incremental decoding requires
wrappers like the above. The same `LivePipeline` lifecycle hooks apply to
all of them; only the `process_audio` body changes.
