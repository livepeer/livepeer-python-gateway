#  Ping/Pong Websocket Runner Demo

This example demonstrates runtime registration for a single-shot websocket
runner. The runner exposes a websocket endpoint:

- `/ws` receives `{"ping": <timestamp>}` and responds with
  `{"pong": <timestamp>, "delta_ms": <receiver delta>}`.

When the websocket closes, the single-shot workload is over and the runner
handler releases its per-connection state.

Start go-livepeer:

```sh
./livepeer -orchestrator -useLiveRunners -serviceAddr localhost:8935 -v 99 -orchSecret abcdef
```

Start the runner:

```sh
uv run runner.py --orchestrator http://localhost:8935 --orchSecret abcdef
```

Run the client:

```sh
uv run client.py
```

The client discovers the `livepeer-sample/ping-pong` runner, connects
to its proxied websocket URL, sends one ping every second, and prints both the
receiver-side delta and the client round-trip time.

To send a fixed number of pings:

```sh
uv run client.py --count 10
```
