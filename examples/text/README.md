# Text Stream Demo

Single-shot text streaming on Livepeer Node with static configuration.

This demo exposes a tiny aiohttp runner with two streaming endpoints:

- `/text` streams a story as `text/plain`, one character at a time.
- `/sse` streams a story as Server-Sent Events, one line per event.

The app also exposes `/healthz` for Livepeer Node runner health checks.

Install [Livepeer Node](https://github.com/livepeer/node) and put its executables
on your `PATH`. From the repository root, start its orchestrator:

```sh
livepeer orchestrator --config="" --runner-config examples/text/runners.toml \
  --runner-grants 127.0.0.1:8990 --health-grants 127.0.0.1:8990
```

Start the runner app in another terminal:

```sh
uv run examples/text/runner.py
```

Allow up to five seconds for the first health check, then verify registration:

```sh
curl http://localhost:8935/discovery | jq
```

Call the endpoints through Livepeer Node:

```sh
# Plain text stream. `-N` disables curl output buffering
curl -N http://localhost:8935/apps/story-runner/app/text

# SSE stream. Each story line is emitted as one `data:` event
curl -N http://localhost:8935/apps/story-runner/app/sse
```
