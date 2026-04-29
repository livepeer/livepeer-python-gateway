# TODO

Repo-level follow-ups that don't belong in code comments or spec docs.
Pending operational decisions and triggers for future work.

## Examples directory

- [ ] **Extract `examples/runner/` to a dedicated repo
  (`livepeer/livepeer-pipeline-examples`)** when any of these triggers hit:
  - 3+ runner examples in this repo
  - First multi-GB-model example (image gen, LLM, real-time video) — CI
    cost becomes painful
  - First community contribution
  - Cog precedent: `replicate/cog-examples` lives separately from `cog`.

  Keep `hello_world` in this repo as the CI smoke test even after extraction.
  Each example in the new repo pins to a specific `livepeer-gateway` version.

## BYOC offchain support

- [ ] **Drop `-network arbitrum-one-mainnet`, `-ethUrl`, `-ethPassword`
  from example compose files** once
  [livepeer/go-livepeer#3906](https://github.com/livepeer/go-livepeer/pull/3906)
  ships in `:master`. After that, examples can run with bare
  `-network offchain`. Tracked: TODO comment already in each compose.

## SDK round-trip

- [ ] **Replace `test.sh`'s curl + base64 Livepeer header with a Python SDK
  batch caller** once a batch caller lands (built on
  [livepeer/livepeer-python-gateway#6](https://github.com/livepeer/livepeer-python-gateway/pull/6)'s
  signing primitives). At that point the example compose can drop the
  `gateway` service entirely — caller talks direct to the orchestrator via
  the remote signer (per
  [livepeer/go-livepeer#3869](https://github.com/livepeer/go-livepeer/pull/3869)).
  Tracked: TODO comment in each example's `test.sh`.

## SDK feature gaps

- [ ] **Health state machine** (`LOADING / READY / ERROR / IDLE`) on
  `/health` body. Currently `setup()` blocks the server bind, and the
  example uses a docker compose healthcheck as a workaround. When the
  state machine lands, drop the healthcheck from `sentiment/docker-compose.yml`.

- [ ] **`Input()` / `Output()` typed descriptors** (C3 in the planned
  staircase). Required before schema generation.

- [ ] **Schema generation + `GET /openapi.json`** (C4). Inspects
  `predict()` / `on_frame()` signature, emits OpenAPI JSON.

- [ ] **`GET /` discovery doc** (C5). Points at schema URL, capability id,
  version, supported transports. Cog parallel.

- [ ] **`StreamPipeline` for trickle transport.** With `on_frame()` /
  `on_video_frame()` / `on_audio_frame()`. Reuse existing trickle primitives
  in `livepeer_gateway.transport`.

- [ ] **SSE auto-detection** for generators. When `predict()` yields,
  emit `text/event-stream`.

- [ ] **`livepeer push` CLI** + `livepeer.yaml` manifest. Parses the
  manifest, generates the Dockerfile (no more hand-written examples),
  builds, registers. Drops the example Dockerfiles.

- [ ] **Schema as Docker image label** (`org.livepeer.pipeline.schema`).
  Lands with `livepeer push`. Removes the runtime `/openapi.json` as the
  primary schema delivery; keeps it as a fallback.

## Tracking related upstream work

- go-livepeer offchain BYOC: [#3905](https://github.com/livepeer/go-livepeer/issues/3905) /
  [#3906](https://github.com/livepeer/go-livepeer/pull/3906)
- BYOC remote signer: [#3869](https://github.com/livepeer/go-livepeer/pull/3869)
- Caller-side BYOC SDK: [#6](https://github.com/livepeer/livepeer-python-gateway/pull/6)
- This SDK's draft PR: [#7](https://github.com/livepeer/livepeer-python-gateway/pull/7)
