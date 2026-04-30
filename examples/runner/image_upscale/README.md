# Image upscale (BYOC)

A ~2x image super-resolution BYOC capability — proves the SDK handles binary
I/O cleanly via Pydantic's `Base64Bytes`. Built on
[Swin2SR](https://huggingface.co/caidas/swin2SR-classical-sr-x2-64), small
enough to run on CPU.

A `Pipeline` subclass loads the model once in `setup()`, then takes a
base64-encoded image on each `POST /predict` and returns the upscaled PNG.
The processor pads inputs to its window size before upscaling, so output
dimensions are at least 2x input but may be slightly larger. Registered as
a BYOC capability, called through the gateway, response flows back
end-to-end.

## Run

```bash
docker compose up -d --wait
./test.sh
docker compose down
```

`test.sh` prints `PASS` on success.

`prepare_models.py` bakes the model into the image at build time so
`setup()` loads from local cache in milliseconds.

## Browse the API

- Swagger UI: <http://localhost:5000/docs>
- ReDoc: <http://localhost:5000/redoc>
- OpenAPI JSON: <http://localhost:5000/openapi.json>

## What's running

```mermaid
sequenceDiagram
    autonumber
    participant curl
    participant gateway
    participant orchestrator
    participant image_upscale as image_upscale<br/>(SDK container)

    curl->>gateway: POST /process/request/predict
    gateway->>orchestrator: forward (Livepeer-signed)
    orchestrator->>image_upscale: POST /predict {"image":"<base64>"}
    image_upscale-->>orchestrator: {"image":"<base64>","width":W,"height":H}
    orchestrator-->>gateway: response
    gateway-->>curl: response
```

Four compose services:

| Service                   | What it is                                                                                                                                         |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `gateway`, `orchestrator` | `livepeer/go-livepeer:master` from Docker Hub                                                                                                      |
| `image_upscale`           | The pipeline container — a [BYOC](https://github.com/livepeer/go-livepeer/blob/main/doc/byoc.md) capability built with `livepeer_gateway.runner`.  |
| `register_capability`     | One-shot helper that POSTs to `orchestrator:8935/capability/register` once `image_upscale` is healthy                                              |

The pipeline service has a healthcheck that probes `GET /health` until the
model finishes loading. `register_capability` waits on `service_healthy`, so
the orchestrator never sees a "registered but not loaded" container.

## Binary I/O contract

Both `image` fields use Pydantic's `Base64Bytes`:

- **Input** — `image` is a base64-encoded string in the JSON body. Pydantic
  decodes to `bytes` before `predict()` runs.
- **Output** — `image` is `bytes` in the pipeline; Pydantic encodes back to
  base64 in the JSON response.

`width` and `height` are returned alongside for convenience. The pipeline
always emits PNG; document the format in the field description if you need
to surface it to callers.

## Try with your own image

```bash
TEST_IMAGE=/path/to/your.png \
INPUT_WIDTH=$W INPUT_HEIGHT=$H \
./test.sh
```

The test asserts output is at least 2x input dimensions.

Or manually:

```bash
INPUT_B64=$(base64 -w0 < your.png)

LIVEPEER_HDR=$(printf '%s' \
  '{"request":"{}","parameters":"{}","capability":"image-upscale","timeout_seconds":60}' \
  | base64 -w0)

curl -X POST http://localhost:9935/process/request/predict \
    -H "Livepeer: ${LIVEPEER_HDR}" \
    -H "Content-Type: application/json" \
    -d "{\"image\":\"${INPUT_B64}\"}" \
    | jq -r '.image' | base64 -d > upscaled.png
```
