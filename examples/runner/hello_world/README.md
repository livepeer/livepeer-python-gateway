# Hello world (BYOC)

> [!NOTE]
> **TODO** — `test.sh` and the `gateway:` compose service collapse into a single Python script using the client SDK once [livepeer/livepeer-python-gateway#6](https://github.com/livepeer/livepeer-python-gateway/pull/6) merges.


Smallest end-to-end test of the Pipeline SDK against a real
[go-livepeer](https://github.com/livepeer/go-livepeer) BYOC stack. A `Pipeline`
subclass returns `{"message": "hello, X"}` over HTTP. Registered as a BYOC
capability, called through the gateway, response flows back end-to-end.

## Run

> [!WARNING]
> Only one example can run at a time — all share container names
> (`gateway`, `orchestrator`, …) and ports (`1935`, `9935`, `5000`). If
> `./test.sh` fails at the capability-registration step, run `docker
> compose down` in the other example's directory first.

```bash
docker compose up -d
./test.sh
docker compose down
```

`test.sh` prints `PASS` on success.

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
    participant hello_world as hello_world<br/>(SDK container)

    curl->>gateway: POST /process/request/predict
    gateway->>orchestrator: forward (Livepeer-signed)
    orchestrator->>hello_world: POST /predict {"name":"..."}
    hello_world-->>orchestrator: {"message":"hello, ..."}
    orchestrator-->>gateway: response
    gateway-->>curl: response
```

Four compose services:

| Service | What it is |
| --- | --- |
| `gateway`, `orchestrator` | `livepeer/go-livepeer:master` from Docker Hub |
| `hello_world` | The pipeline container — a [BYOC](https://github.com/livepeer/go-livepeer/blob/main/doc/byoc.md) capability built with `livepeer_gateway.runner`. Attached via HTTP register, not the `-aiWorker` mechanism. |
| `register_capability` | One-shot helper that POSTs to `orchestrator:8935/capability/register` |

First `docker compose up` pulls `livepeer/go-livepeer:master` (~few hundred MB,
cached after) and builds the `hello_world` image locally.
