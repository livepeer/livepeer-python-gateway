# Echo Live Runner Demo

Registers `livepeer-sample/echo` and shows input, loopback, and output in a
2×2 composite with FPS and latency overlays.

Install [Livepeer Node](https://github.com/livepeer/node), put its executables on
your `PATH`, then run from the repository root:

```sh
uv sync --extra examples
LIVEPEER_ORCHESTRATOR_BOOTSTRAP_SECRET=abcdef livepeer orchestrator \
  --config="" --listen 127.0.0.1:8935 --runner-grants 127.0.0.1:8989
```

Start the runner in another terminal:

```sh
uv run examples/echo/runner.py --orchestrator http://localhost:8935 --orchSecret abcdef
```

Run the client with the macOS camera, or supply a video file on any platform:

```sh
uv run examples/echo/client.py --blur
uv run examples/echo/client.py --blur "<input-video>"
```

`--blur` sweeps the runner's blur radius; without it, frames return unchanged.
Press `q`, close the window, or press Ctrl-C to stop. Files play at their source
rate and the client exits after the output drains.

The latency overlays show input minus loopback (RTT), input minus output (Total Delta),
and loopback minus output (Processing). These are differences between the
latest frame timestamps, not measurements of individual frame processing.

Use `--output` to save raw MPEG-TS or pipe it to a player instead of opening
the composite. Diagnostics go to stderr:

```sh
uv run examples/echo/client.py --blur --output echo-out.ts "<input-video>"
uv run examples/echo/client.py --blur --output - "<input-video>" | ffplay -fflags nobuffer -flags low_delay -probesize 32 -i -
```

See `uv run examples/echo/client.py --help` for camera and runner selection options.
