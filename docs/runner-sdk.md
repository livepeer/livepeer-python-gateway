# Runner SDK

> Status: **draft**. Interface not yet locked. Examples still landing —
> see [Coverage](#coverage) and [Not yet implemented](#not-yet-implemented).
>
> Source: [`src/livepeer_gateway/runner/`](../src/livepeer_gateway/runner/).
> Tracking PR: [livepeer/livepeer-python-gateway#7][pr7].

## Where this fits

This is the **deploy half** of the Livepeer Python SDK — "I run
pipelines on the network." The other half is the **client / gateway
side** — "I send requests to the network" — tracked in epic
[#9][issue9]. Both will ship from one monorepo using PEP 420 namespace
packages: shared `livepeer` import root, separate distributions
(`livepeer-runner`, `livepeer-client`, `livepeer-trickle`) installed
independently. Today the code lives under `livepeer_gateway.runner`
pending the restructure (tracked as C12 in [#8][issue8]).
Full architecture: [pipeline-sdk spec][spec].

## What the SDK provides

A runner author writes one Python class, calls `serve(pipeline)`, gets a
FastAPI app the orchestrator can hit. Two base classes cover the two
transport modes:

| Base class                                                        | Transport                                                          | Hook surface                                                                                                                          |
| ----------------------------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| [`Pipeline`](../src/livepeer_gateway/runner/pipeline.py)          | `POST /predict` — request/response, or SSE when `predict()` yields | `setup()`, `predict(**kwargs)`                                                                                                        |
| [`LivePipeline`](../src/livepeer_gateway/runner/live_pipeline.py) | trickle (BYOC live) over `POST /stream/{start,stop,params}`        | `setup()`, `on_stream_start`, `process_video`, `process_audio`, `on_params_update`, `on_stream_stop`, plus `emit_event` / `emit_data` |

`make_app(pipeline)` dispatches on the base class. Health is `GET /health`
on both.

## `Pipeline` — batch

```python
class Sentiment(Pipeline):
    def setup(self) -> None:
        self.model = load_model()

    def predict(self, text: str) -> dict:
        return {"label": self.model(text)}
```

- `predict` signature is introspected. A single `BaseModel` parameter is
  used directly; bare params get wrapped in a generated model.
- Generator `predict` is auto-detected → response framed as SSE.
- Return type annotation, if `BaseModel`, becomes the OpenAPI response.

Examples: [`hello_world`](../examples/runner/hello_world/),
[`sentiment`](../examples/runner/sentiment/),
[`image_upscale`](../examples/runner/image_upscale/),
[`llm`](../examples/runner/llm/) (SSE).

## `LivePipeline` — trickle

Minimal — video transform only:

```python
class Grayscale(LivePipeline):
    async def process_video(self, frame: VideoFrame) -> VideoFrame | None:
        return frame.to_grayscale()
```

Audio in, structured data out — uses per-session state and the side
channels:

```python
class LiveTranscribe(LivePipeline):
    def setup(self) -> None:
        self._model = WhisperModel("tiny.en")  # loaded once at container start

    async def on_stream_start(self, params: dict) -> None:
        self._buffer = np.empty(0, dtype=np.float32)  # fresh per session
        await self.emit_event({"type": "ready", "model": "whisper-tiny.en"})

    async def process_audio(self, frame: AudioFrame) -> AudioFrame:
        self._buffer = np.concatenate([self._buffer, _to_pcm(frame)])
        if self._buffer.size >= WINDOW_SAMPLES:
            for seg in self._model.transcribe(self._buffer[:WINDOW_SAMPLES]):
                await self.emit_data({"type": "transcript", "text": seg.text})
            self._buffer = self._buffer[WINDOW_SAMPLES:]
        return frame  # passthrough — caller still gets the audio track

    async def on_stream_stop(self) -> None:
        await self.emit_event({"type": "stopped"})
```

Lifecycle, in call order per session:

1. `setup()` — once, before the server binds (sync).
2. `on_stream_start(params)` — per-session init, before the first frame.
3. `process_video(frame)` / `process_audio(frame)` — per-frame transform.
   Return `None` to drop.
4. `on_params_update(params)` — caller posted new params mid-stream.
5. `on_stream_stop()` — per-session cleanup. `emit_data` / `emit_event`
   still work here.

Side channels, callable from any hook during a session:

- `emit_event({...})` — JSON on the events trickle channel (lifecycle,
  telemetry).
- `emit_data({...})` — JSON on the data trickle channel (transcripts,
  detections, classifications). No-op if the orchestrator didn't
  enable a data channel.

### Heartbeat

The SDK auto-publishes a `{"type": "heartbeat", "timestamp": ...}` event
on the events channel every 10 seconds while a session is active. This
exists to satisfy the gateway's 30s events-gap watchdog
(`byoc/trickle.go: maxEventGap`) — without it, the gateway tears down
sessions whose pipelines aren't emitting their own events. User
`emit_event` calls also reset the watchdog, so a pipeline that emits
its own telemetry doesn't strictly need the heartbeat — but it's always
on. The task is started before the frame loop (so `events_url` shows
activity during cold start) and cancelled on `/stream/stop`.

The payload is intentionally minimal today; FPS / throughput fields are
in [Not yet implemented](#not-yet-implemented).

### Track allocation by introspection

A track is allocated **iff** its `process_*` hook is overridden — except
a stock `LivePipeline` (no overrides) which allocates both as a
passthrough relay. See
[`_emit_flags`](../src/livepeer_gateway/runner/live_pipeline.py).

Concretely:

| Override              | Video out | Audio out |
| --------------------- | --------- | --------- |
| neither (passthrough) | yes       | yes       |
| `process_video` only  | yes       | no        |
| `process_audio` only  | no        | yes       |
| both                  | yes       | yes       |

Examples: [`live_grayscale`](../examples/runner/live_grayscale/) (video
only), [`live_depth`](../examples/runner/live_depth/) (video only, GPU
model), [`live_transcribe`](../examples/runner/live_transcribe/) (audio
in → data out).

## Coverage

| Mode                  | Example                               | Status |
| --------------------- | ------------------------------------- | ------ |
| Batch sync            | hello_world, sentiment, image_upscale | ✅     |
| Batch streaming (SSE) | llm                                   | ✅     |
| Live video → video    | live_grayscale, live_depth            | ✅     |
| Live audio → data     | live_transcribe                       | ⏳     |
| Live audio → audio    | _planned: pitch-shift_                | ⏳     |
| Live video + audio    | _planned: combined_                   | ⏳     |

> [!NOTE]
> `live_transcribe` is feature-complete on the runner side but is marked
> ⏳ pending two known transport bugs that drop records before they reach
> the SSE caller — one SDK-side ([#12][issue12]: first segment of every
> trickle channel duplicate-writes seg 0, one-line fix), one upstream
> ([go-livepeer#3924][go3924]: gateway tears down the data SSE proxy
> before the runner's final `emit_data` is relayed). Combined effect:
> 5 transcripts emitted → 3 delivered. Promote to ✅ once both land.

## Open questions

Design points still open before the surface is declared stable.

1. **`emit_data` / `emit_event` outside an active session.** Today:
   silent no-op. Alternative: raise. Silent matches "the orchestrator
   didn't enable this channel" but masks programming errors.
2. **Single-session constraint.** `LivePipeline._session` is one slot;
   `/stream/start` returns 409 if busy. Multi-session is deferred to
   post-C8 (demand-driven). Documenting the constraint as a contract
   means callers won't lean on it being temporary.

## Not yet implemented

Items that are decided in principle but not in the SDK yet. Tracked in
[the SDK epic][issue8].

### `livepeer.yaml` + `livepeer push`

The biggest pending piece. Today every example ships a hand-written
Dockerfile, `register_capability.py`, and `docker-compose.yml`. A
manifest + CLI collapses that to a one-file declaration:

Before — `examples/runner/sentiment/`:

```text
Dockerfile              ← hand-written
docker-compose.yml      ← hand-written
register_capability.py  ← boilerplate
requirements.txt
prepare_models.py
pipeline.py
README.md
test.sh
```

After:

```text
livepeer.yaml           ← single source of truth
prepare_models.py
pipeline.py
README.md
```

Sample manifest (shape illustrative, not final):

```yaml
pipeline_id: sentiment
pipeline: pipeline:Sentiment # module:class

python:
  version: "3.11"
  packages:
    - transformers==4.40.0
    - torch==2.3.0

system:
  packages:
    - ffmpeg

build:
  prepare: prepare_models.py # runs at image build, caches weights

gpu: false # or "T4", "L4", etc.
```

`livepeer push` parses the manifest, generates the Dockerfile, builds the
image, embeds the schema as an image label, and registers the capability
on the network. Builders never write Dockerfiles or call the
registration RPC.

The manifest is the recommended path, not the only one. Builders who want
full control over the container — custom base images, multi-stage builds,
non-trivial system setup — can still write their own Dockerfile, call
`serve(pipeline)` from any entrypoint, and register the capability
manually. `livepeer push` is sugar; the underlying contract (HTTP
endpoints + image labels) is what the orchestrator cares about.

### Other pending items

- **Schema generation (`/openapi.json`) + discovery doc (`GET /`).**
  Phases C3 / C4 / C5 in the staircase.
- **Health state machine** on `/health` (`LOADING → OK | ERROR | IDLE`).
  Currently `setup()` blocks the bind, examples use a docker
  healthcheck workaround.
- **Structured error events.** `emit_event` is freeform JSON today.
  We want a typed `error` event shape so the orchestrator and caller
  can react uniformly to runner-side failures (model OOM, GPU lost,
  per-frame error budget exceeded).
- **FPS / throughput in heartbeats.** Today's heartbeat is
  `{type, timestamp}` only. Adding observed input fps + processed-frame
  fps lets the orchestrator detect a slow pipeline before the
  events-gap watchdog fires.
- **Inbound JSON streaming.** `/stream/params` is one-shot today. Pipelines
  that need a continuous JSON stream from the caller (live prompts,
  streaming text, high-frequency control beyond a single param dict)
  have nothing — likely an `on_control(payload)` hook exposing the
  `control_url` stream directly. Tracked: [#8][issue8].
- **Lock down `_session` access.** The user-facing `emit_event` /
  `emit_data` methods reach into `pipeline._session` directly, leaking
  the private attribute as an accidental subclass contract. Replace
  with a private accessor or move the methods to operate on session
  state passed in. Tracked: [#8][issue8].
- **Multi-session per pipeline.** Capacity demand-driven; post-C8.

Tracking PR: [livepeer/livepeer-python-gateway#7][pr7]. Upstream
go-livepeer work and the broader SDK roadmap are tracked in the
[SDK epic][issue8].

[pr7]: https://github.com/livepeer/livepeer-python-gateway/pull/7
[issue8]: https://github.com/livepeer/livepeer-python-gateway/issues/8
[issue9]: https://github.com/livepeer/livepeer-python-gateway/issues/9
[issue12]: https://github.com/livepeer/livepeer-python-gateway/issues/12
[go3924]: https://github.com/livepeer/go-livepeer/issues/3924
[spec]: https://github.com/rickstaa/livepeer-specs/blob/main/docs/developer-journey/pipeline-sdk.md
