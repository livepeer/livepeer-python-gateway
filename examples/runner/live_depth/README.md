# Live depth (BYOC, real-time, GPU)

> [!NOTE]
> **TODO** — `test.sh`, `demo.sh`, and the `gateway:` compose service collapse into a single Python script using the client SDK once [livepeer/livepeer-python-gateway#6](https://github.com/livepeer/livepeer-python-gateway/pull/6) merges.


Real-time monocular depth estimation via [DepthAnything V2](https://github.com/DepthAnything/Depth-Anything-V2).
Each video frame goes through the model; the depth map replaces the luma
plane and the chroma planes are zeroed, so the egress reads as a
grayscale "bright = close, dark = far" visualization.

## Requirements

- **NVIDIA GPU** with CUDA 12.x driver
- [`nvidia-container-toolkit`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
  installed and registered with Docker
- Several GB of free disk (image bundles PyTorch + CUDA runtime + model)

Verify with:

```bash
docker info | grep -i nvidia                  # should show "Runtimes: ... nvidia ..."
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

Without GPU access the container will still start but PyTorch falls back
to CPU — depth inference is too slow to sustain real-time and the publisher
tears down within seconds.

## Run

> [!WARNING]
> Only one example can run at a time — all share container names
> (`gateway`, `orchestrator`, …) and ports (`1935`, `9935`, `5000`). If
> `./test.sh` fails at the capability-registration step, run `docker
> compose down` in the other example's directory first.

```bash
docker compose up -d --wait --build      # first build downloads model + CUDA layers

./test.sh                                 # CI: real depth-rich clip, asserts non-trivial depth output
./demo.sh                                 # interactive: webcam in, live depth ffplay window

docker compose down
```

The pipeline container starts in `LOADING` state until `setup()` finishes
loading the model on GPU and runs the warm pass.

### `test.sh` — automated assertion

Pushes a depth-rich basketball clip (auto-downloaded to `assets/sample.mp4`
on first run) through the full BYOC chain, captures the egress, and asserts
both that chroma is zeroed *and* the luma plane has spatial variance — so a
no-op pipeline can't pass by accidentally producing uniform output.

### `demo.sh` — live webcam viewer

Pushes your webcam through the depth pipeline and opens an ffplay window
showing the depth output in real time. Bright = close, dark = far. Wave
your hand toward the camera to see the gradient shift.

Defaults to 640×480 @ 15fps. Bump or override via env vars:

```bash
WEBCAM_FPS=30 ./demo.sh                # smoother, may stall on slower CPUs
WEBCAM_RES=1280x720 ./demo.sh
WEBCAM_DEVICE=/dev/video1 ./demo.sh    # Linux: pick a different camera
```

If the live viewer disconnects mid-stream, you've hit the throughput
ceiling — drop FPS or resolution.

## Throughput

DepthAnything V2 Base with `torch.float16` on a 30/40-series GPU runs
~25 ms per frame — comfortably inside the 33 ms (30 fps) or 66 ms (15 fps)
budget. Switch variant by editing `_MODEL_ID` in `pipeline.py` (and the
matching bake line in the `Dockerfile`).

| Variant | Params | VRAM (fp16) | Latency on RTX 3090 |
| --- | --- | --- | --- |
| `Small-hf` | 24 M | ~1 GB | ~15 ms |
| `Base-hf` (default) | 97 M | ~2 GB | ~25 ms |
| `Large-hf` | 335 M | ~3 GB | ~80 ms |
