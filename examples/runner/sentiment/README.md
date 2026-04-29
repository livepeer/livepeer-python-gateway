# Sentiment analysis (BYOC)

A Hugging Face sentiment classifier shipped as a BYOC capability. Demonstrates
the `setup()` lifecycle hook for one-time model loading. Built on
[distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) — small enough to run on CPU.

A `Pipeline` subclass loads the model once in `setup()`, then classifies text
on each `POST /predict`. Registered as a BYOC capability, called through the
gateway, response flows back end-to-end.

## Run

```bash
docker compose up -d --wait
./test.sh
docker compose down
```

`test.sh` prints `PASS` on success.

> **First build is ~5 minutes** — pulls torch CPU (~200 MB), transformers, and
> bakes the ~250 MB model into the image. Cached after that; rebuilds are fast.

## What's running

```mermaid
sequenceDiagram
    autonumber
    participant curl
    participant gateway
    participant orchestrator
    participant sentiment as sentiment<br/>(SDK container)

    curl->>gateway: POST /process/request/predict
    gateway->>orchestrator: forward (Livepeer-signed)
    orchestrator->>sentiment: POST /predict {"text":"..."}
    sentiment-->>orchestrator: {"label":"POSITIVE","score":0.99,...}
    orchestrator-->>gateway: response
    gateway-->>curl: response
```

Four compose services:

| Service                   | What it is                                                                                                                                                                         |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `gateway`, `orchestrator` | `livepeer/go-livepeer:master` from Docker Hub                                                                                                                                      |
| `sentiment`               | The pipeline container — a [BYOC](https://github.com/livepeer/go-livepeer/blob/main/doc/byoc.md) capability built with `livepeer_gateway.runner`. Loads the HF model in `setup()`. |
| `register_capability`     | One-shot helper that POSTs to `orchestrator:8935/capability/register` once `sentiment` is healthy                                                                                  |

The sentiment service has a healthcheck that probes `GET /health` until the
model finishes loading. `register_capability` waits on `service_healthy`, so
the orchestrator never sees a "registered but not loaded" container.

## Try variations

```bash
TEXT="this is awful" EXPECTED_LABEL=NEGATIVE ./test.sh
```

Or manually:

```bash
LIVEPEER_HDR=$(printf '%s' '{"request":"{}","parameters":"{}","capability":"sentiment","timeout_seconds":30}' | base64 -w0)

curl -X POST http://localhost:9935/process/request/predict \
    -H "Livepeer: ${LIVEPEER_HDR}" \
    -H "Content-Type: application/json" \
    -d '{"text":"distributed inference is the future"}'
```
