#!/usr/bin/env python3
"""Minimal repro: does MediaPublish deliver unevenly when a second (audio) track is added?

Removes the orchestrator, the trickle network and any example app. A local sink
implements the three endpoints TricklePublisher needs and just drains bodies, so the
only thing under test is MediaPublish itself.

    python scripts/repro_two_track_stall.py            # video only
    python scripts/repro_two_track_stall.py --audio    # video + audio
"""

from __future__ import annotations

import argparse
import asyncio
import time

import av
import numpy as np
from aiohttp import web

from livepeer_gateway.media_publish import (
    AudioOutputConfig,
    MediaPublish,
    MediaPublishConfig,
    VideoOutputConfig,
)

W, H, FPS = 1280, 720, 30
SR, ASAMPLES = 48000, 1024

recv: list[float] = []


async def _sink(port: int) -> web.AppRunner:
    async def create(_req):
        return web.Response()

    async def nxt(_req):
        return web.Response(headers={"Lp-Trickle-Latest": "0"})

    async def seg(req):
        while True:
            chunk = await req.content.readany()
            if not chunk:
                break
            recv.append(time.monotonic())
        return web.Response()

    app = web.Application(client_max_size=0)
    app.router.add_post("/t", create)
    app.router.add_get("/t/next", nxt)
    app.router.add_post("/t/{seq}", seg)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    return runner


def _video(i: int) -> av.VideoFrame:
    img = np.full((H, W, 3), (i * 7) % 256, dtype=np.uint8)
    f = av.VideoFrame.from_ndarray(img, format="bgr24")
    f.pts, f.time_base = i, __import__("fractions").Fraction(1, FPS)
    return f


def _audio(i: int) -> av.AudioFrame:
    t = (np.arange(ASAMPLES, dtype=np.float32) + i * ASAMPLES) / SR
    pcm = (np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16).reshape(1, -1)
    f = av.AudioFrame.from_ndarray(pcm, format="s16", layout="mono")
    f.sample_rate = SR
    f.pts, f.time_base = i * ASAMPLES, __import__("fractions").Fraction(1, SR)
    return f


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", action="store_true")
    ap.add_argument("--seconds", type=float, default=14.0)
    ap.add_argument("--port", type=int, default=871)
    args = ap.parse_args()

    sink = await _sink(args.port)
    tracks = [VideoOutputConfig()]
    if args.audio:
        tracks.append(AudioOutputConfig())
    pub = MediaPublish(f"http://127.0.0.1:{args.port}/t", config=MediaPublishConfig(tracks=tracks))

    emit: list[float] = []
    blocked = 0.0
    t0 = time.monotonic()
    vi = ai = 0
    try:
        while time.monotonic() - t0 < args.seconds:
            now = time.monotonic()
            target_v = int((now - t0) * FPS)
            while vi <= target_v:
                b = time.monotonic()
                await pub.write_frame(_video(vi))
                blocked = max(blocked, time.monotonic() - b)
                emit.append(b)
                vi += 1
            if args.audio:
                target_a = int((now - t0) * SR / ASAMPLES)
                while ai <= target_a:
                    await pub.write_frame(_audio(ai))
                    ai += 1
            await asyncio.sleep(0.002)
    finally:
        st = pub.get_stats()
        await pub.close()
        await sink.cleanup()

    def gaps(ts):
        return [round(ts[i] - ts[i - 1], 3) for i in range(1, len(ts))]

    eg, rg = gaps(emit), gaps(recv)
    mode = "video+audio" if args.audio else "video only"
    print(f"\n=== {mode} ===")
    print(f"emit   n={len(emit):4d} max_gap={max(eg or [0]):.3f}s  write_frame max block={blocked:.3f}s")
    print(f"sink   n={len(recv):4d} max_gap={max(rg or [0]):.3f}s  gaps>0.3s={sum(1 for g in rg if g > 0.3)}")
    for t in st.track_queue_stats:
        print(f"  {t.label:6} in={t.frames_in:5d} ovf={t.frames_dropped_overflow:4d} "
              f"debt={t.frames_dropped_debt:4d} debt_s={t.time_debt_s:.3f}")
    print(f"  segments started={st.segments_started} completed={st.segments_completed} "
          f"failed={st.segments_failed} enc_err={st.encoder_errors}")


if __name__ == "__main__":
    asyncio.run(main())
