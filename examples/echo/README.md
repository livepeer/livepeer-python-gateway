# Echo Live Runner Demo

This example demonstrates:

* Runner registration
* Video input - taken from a local file
* Video output - echoed to output with blur applied
* Parameter updates - adjust the amount of blur

Start go-livepeer:

```sh
./livepeer -orchestrator -useLiveRunners -serviceAddr localhost:8935 -v 99 -orchSecret abcdef
```

Start the runner:

```sh
uv run examples/echo/runner.py --orchestrator https://localhost:8935 --orchSecret abcdef
```

Run the client with a local sample input (`~/samples/bbb_720p.mp4`):

```sh
uv run examples/echo/client.py --blur ~/samples/bbb_720p.mp4
```

The resulting file is stored at echo-out.ts. To use a different file
or redirect to stdout for live playback:

```sh
uv run client.py --blur --output - ~/samples/bbb_720p.mp4 | ffplay -
```

The client discovers the `livepeer-sample/echo` runner automatically. To use a
different orchestrator or discovery endpoint:

```sh
uv run examples/echo/client.py --discovery http://localhost:8935/discovery --blur ~/samples/bbb_720p.mp4
```
