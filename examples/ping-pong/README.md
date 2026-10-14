#  Ping/Pong Websocket Runner Demo

This example demonstrates runtime registration for a single-shot websocket
runner. The runner exposes a websocket endpoint:

- `/ws` receives `{"ping": <timestamp>}` and responds with
  `{"pong": <timestamp>, "delta_ms": <receiver delta>}`.

When the websocket closes, the single-shot workload is over and the runner
handler releases its per-connection state.

Install [Livepeer Node](https://github.com/livepeer/node), put its executables on
your `PATH`, and start its local orchestrator:

```sh
LIVEPEER_ORCHESTRATOR_BOOTSTRAP_SECRET=abcdef livepeer orchestrator \
  --runner-grants 127.0.0.1
```

Start the runner:

```sh
uv run examples/ping-pong/runner.py --orchestrator http://localhost:8935 --orchSecret abcdef
```

Run the client:

```sh
uv run examples/ping-pong/client.py
```

The client discovers the `livepeer-sample/ping-pong` runner, connects
to its proxied websocket URL, sends one ping every second, and prints both the
receiver-side delta and the client round-trip time.

To send a fixed number of pings:

```sh
uv run examples/ping-pong/client.py --count 10
```
