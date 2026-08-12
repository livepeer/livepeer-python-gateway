# Text Stream Demo

Single-shot text streaming on Livpeeer with static configuration.

This demo exposes a tiny aiohttp runner with two streaming endpoints:

- `/text` streams a story as `text/plain`, one character at a time.
- `/sse` streams a story as Server-Sent Events, one line per event.

The app also exposes `/healthz` for go-livepeer runner health checks.

Start go-livepeer with a static runner config:

```sh
# assumes `livepeer` is somewhere in your PATH
livepeer -config go-livepeer.conf
```

Start the runner app:

```sh
uv run runner.py
```

Call the endpoints through go-livepeer:

```sh
# Plain text stream. `-N` disables curl output buffering
curl -N http://localhost:8935/apps/story-runner/app/text

# SSE stream. Each story line is emitted as one `data:` event
curl -N http://localhost:8935/apps/story-runner/app/sse
```

Verify the runner registration with go-livepeer:

```
curl http://localhost:8935/discovery | jq
```
