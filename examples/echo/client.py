#!/usr/bin/env python3
"""Show camera/file input, loopback, and echo output, or stream raw output."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import queue
import sys
import threading
import time
from collections import deque
from contextlib import nullcontext, suppress
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, BinaryIO

import av

from livepeer_gateway.errors import LivepeerGatewayError
from livepeer_gateway.http import post_json
from livepeer_gateway.live_runner import stop_runner_session
from livepeer_gateway.media_output import MediaOutput
from livepeer_gateway.media_publish import (
    MediaPublish,
    MediaPublishConfig,
    VideoOutputConfig,
)
from livepeer_gateway.selection import reserve_session
from livepeer_gateway.token import parse_token

DEFAULT_DISCOVERY = "http://localhost:8935/discovery"
ECHO_APP_ID = "livepeer-sample/echo"
BLUR_UPDATE_INTERVAL_S = 0.05
BLUR_RADIUS_STEP = 5
MAX_BLUR_RADIUS = 100
_TIME_BASE = Fraction(1, 90_000)
_STOP = object()


def _log(*args: object) -> None:
    print(*args, file=sys.stderr)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", nargs="?", help="Local video file; otherwise capture the macOS camera."
    )
    parser.add_argument(
        "--orchestrator", help="Orchestrator host:port or URL to discover runners on."
    )
    parser.add_argument(
        "--discovery", help=f"Discovery endpoint (default: {DEFAULT_DISCOVERY})."
    )
    parser.add_argument(
        "--signer", help="Remote signer URL; omitted for offchain mode."
    )
    parser.add_argument(
        "--token", help="Base64 gateway token; non-empty fields override CLI settings."
    )
    parser.add_argument(
        "--output",
        help="Write raw output to a file or '-' / 'stdout' instead of opening the composite.",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="avfoundation camera index (default: 0); ignored for files.",
    )
    parser.add_argument(
        "--fps", type=float, default=30.0, help="Camera FPS (default: 30)."
    )
    parser.add_argument(
        "--video-size", default="640x480", help="Camera size (default: 640x480)."
    )
    parser.add_argument(
        "--pixel-format", default="nv12", help="Camera pixel format (default: nv12)."
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many video frames (0 = unlimited).",
    )
    parser.add_argument(
        "--blur", action="store_true", help="Sweep blur radius while publishing."
    )
    return parser.parse_args(argv)


def _channel_url(echo_response: dict[str, object], name: str) -> str:
    url = echo_response.get(name)
    if not isinstance(url, str) or not url:
        raise LivepeerGatewayError(f"echo response missing {name!r} url")
    return url


def _enqueue(item: object, frames: queue.Queue[object], stop: threading.Event) -> bool:
    while not stop.is_set():
        try:
            frames.put(item, timeout=0.1)
            return True
        except queue.Full:
            pass
    return False


def _capture_frames(
    args: argparse.Namespace, frames: queue.Queue[object], stop: threading.Event
) -> None:
    """Own the input container entirely on this thread, including its close."""
    try:
        if args.input:
            input_ = av.open(str(Path(args.input).expanduser()))
        else:
            input_ = av.open(
                args.device,
                format="avfoundation",
                container_options={
                    "framerate": str(args.fps),
                    "video_size": args.video_size,
                    "pixel_format": args.pixel_format,
                },
            )
        with input_:
            if not input_.streams.video:
                raise LivepeerGatewayError(
                    f"No video stream found in input: {args.input or args.device}"
                )
            stream = input_.streams.video[0]
            rate = stream.average_rate or stream.guessed_rate
            fps = float(rate) if args.input and rate else args.fps
            if not _enqueue(VideoOutputConfig(fps=fps), frames, stop):
                return
            start_time: float | None = None
            prev_pts_time: float | None = None
            prev_wall: float | None = None
            count = 0
            while not stop.is_set():
                try:
                    for frame in input_.decode(video=0):
                        if stop.is_set():
                            return
                        now = time.monotonic()
                        if args.input:
                            pts_time = (
                                float(frame.pts * frame.time_base)
                                if frame.pts is not None and frame.time_base is not None
                                else None
                            )
                            if (
                                pts_time is not None
                                and prev_pts_time is not None
                                and prev_wall is not None
                            ):
                                if stop.wait(
                                    max(
                                        0.0,
                                        pts_time - prev_pts_time - (now - prev_wall),
                                    )
                                ):
                                    return
                            prev_pts_time = pts_time
                            prev_wall = (
                                time.monotonic() if pts_time is not None else None
                            )
                        else:
                            if start_time is None:
                                start_time = now
                            frame.pts = round((now - start_time) / _TIME_BASE)
                            frame.time_base = _TIME_BASE
                        if not _enqueue(frame, frames, stop):
                            return
                        count += 1
                        if args.max_frames > 0 and count >= args.max_frames:
                            return
                    return
                except av.BlockingIOError:
                    if args.input:
                        raise
                    stop.wait(0.01)
    except Exception as exc:
        _enqueue(exc, frames, stop)
    finally:
        _enqueue(_STOP, frames, stop)


@dataclass
class _Tile:
    image: Any = None
    pts_time: float | None = None
    frame_times: deque[float] = field(default_factory=deque)


class _CompositeViewer:
    def __init__(self) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise LivepeerGatewayError(
                "The composite requires OpenCV and numpy; run: uv sync --extra examples"
            ) from exc
        self.cv2 = cv2
        self.np = np
        self.tiles = {name: _Tile() for name in ("input", "loopback", "output")}
        self.samples: dict[str, deque[tuple[float, float]]] = {
            name: deque() for name in ("RTT", "Total Delta", "Processing")
        }
        self.title = "input | loopback | output | empty (press q to quit)"

    def update(self, name: str, frame: av.VideoFrame, pts_time: float | None) -> None:
        tile = self.tiles[name]
        tile.image = frame.to_ndarray(format="bgr24")
        tile.pts_time = pts_time
        tile.frame_times.append(time.monotonic())

    def _latency_label(self, name: str, delta: float | None, now: float) -> str:
        samples = self.samples[name]
        while samples and samples[0][0] < now - 3.5:
            samples.popleft()
        if delta is None:
            return f"{name}: n/a"
        samples.append((now, delta))
        average = sum(value for _, value in samples) / len(samples)
        return f"{name}: {delta:+.3f}s (avg {average:+.3f}s)"

    def _letterbox(self, image: Any, size: int) -> Any:
        tile = self.np.zeros((size, size, 3), dtype=self.np.uint8)
        if image is not None:
            height, width = image.shape[:2]
            scale = min(size / width, size / height)
            new_width, new_height = (
                max(1, round(width * scale)),
                max(1, round(height * scale)),
            )
            resized = self.cv2.resize(image, (new_width, new_height))
            x, y = (size - new_width) // 2, (size - new_height) // 2
            tile[y : y + new_height, x : x + new_width] = resized
        return tile

    def _draw_labels(self, image: Any, lines: list[str]) -> None:
        for index, line in enumerate(lines, start=1):
            width = self.cv2.getTextSize(line, self.cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[
                0
            ][0]
            scale = min(0.7, 0.7 * max(1, image.shape[1] - 24) / max(1, width))
            (text_width, text_height), baseline = self.cv2.getTextSize(
                line, self.cv2.FONT_HERSHEY_SIMPLEX, scale, 2
            )
            y = 28 * index
            self.cv2.rectangle(
                image,
                (8, y - text_height - 4),
                (16 + text_width, y + baseline + 3),
                (0, 0, 0),
                self.cv2.FILLED,
            )
            self.cv2.putText(
                image,
                line,
                (12, y),
                fontFace=self.cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=scale,
                color=(0, 255, 0),
                thickness=2,
                lineType=self.cv2.LINE_AA,
            )

    def render(self) -> Any:
        now = time.monotonic()
        source, loop, output = (
            self.tiles[name] for name in ("input", "loopback", "output")
        )
        base = next(
            (tile.image for tile in self.tiles.values() if tile.image is not None), None
        )
        size = min(base.shape[:2]) if base is not None else 480
        rtt = (
            source.pts_time - loop.pts_time
            if source.pts_time is not None and loop.pts_time is not None
            else None
        )
        total = (
            source.pts_time - output.pts_time
            if source.pts_time is not None and output.pts_time is not None
            else None
        )
        processing = (
            loop.pts_time - output.pts_time
            if loop.pts_time is not None and output.pts_time is not None
            else None
        )
        rtt_label = self._latency_label("RTT", rtt, now)
        total_label = self._latency_label("Total Delta", total, now)
        processing_label = self._latency_label("Processing", processing, now)
        images = []
        for name, tile in self.tiles.items():
            while tile.frame_times and tile.frame_times[0] < now - 1.0:
                tile.frame_times.popleft()
            title = (
                name.upper() if tile.image is not None else f"{name.upper()} PENDING"
            )
            labels = [title, f"FPS: {len(tile.frame_times):.1f}"]
            if name == "loopback":
                labels.append(rtt_label)
            elif name == "output":
                labels.extend([processing_label, total_label])
            image = self._letterbox(tile.image, size)
            self._draw_labels(image, labels)
            images.append(image)
        empty = self._letterbox(None, size)
        self._draw_labels(empty, ["EMPTY"])
        return self.np.vstack(
            [self.np.hstack(images[:2]), self.np.hstack([images[2], empty])]
        )

    async def run(self) -> None:
        try:
            self.cv2.namedWindow(self.title, self.cv2.WINDOW_AUTOSIZE)
            while True:
                self.cv2.imshow(self.title, self.render())
                if self.cv2.waitKey(1) & 0xFF == ord("q"):
                    return
                try:
                    visible = self.cv2.getWindowProperty(
                        self.title, self.cv2.WND_PROP_VISIBLE
                    )
                except self.cv2.error:
                    # Some backends remove the window as soon as it is closed.
                    return
                if visible < 1:
                    return
                await asyncio.sleep(0.005)
        except self.cv2.error as exc:
            raise LivepeerGatewayError(
                "Cannot display the composite; use GUI-capable OpenCV and a desktop, or --output."
            ) from exc
        finally:
            with suppress(self.cv2.error):
                self.cv2.destroyAllWindows()


async def _sweep_blur(app_url: str) -> None:
    update_url = f"{app_url.rstrip('/')}/update"
    radius, step = 0, BLUR_RADIUS_STEP
    while True:
        started = time.monotonic()
        await post_json(update_url, {"mode": "blur", "radius": radius})
        if radius == MAX_BLUR_RADIUS:
            step = -BLUR_RADIUS_STEP
        elif radius == 0:
            step = BLUR_RADIUS_STEP
        radius += step
        # Slow requests reduce the update rate without accumulating catch-up work.
        await asyncio.sleep(
            max(0.0, BLUR_UPDATE_INTERVAL_S - (time.monotonic() - started))
        )


async def _publish_video(
    args: argparse.Namespace,
    publish_url: str,
    *,
    viewer: _CompositeViewer | None = None,
) -> None:
    frames: queue.Queue[object] = queue.Queue(maxsize=8)
    stop = threading.Event()
    capture = threading.Thread(
        target=_capture_frames,
        args=(args, frames, stop),
        name="EchoCapture",
        daemon=True,
    )
    publisher = None
    capture.start()
    try:
        while True:
            try:
                item = await asyncio.to_thread(frames.get, True, 0.1)
            except queue.Empty:
                continue
            if item is _STOP:
                break
            if isinstance(item, Exception):
                raise item
            if isinstance(item, VideoOutputConfig):
                publisher = MediaPublish(
                    publish_url, config=MediaPublishConfig(tracks=[item])
                )
                continue
            frame = item
            if viewer is not None:
                pts_time = (
                    float(frame.pts * frame.time_base)
                    if frame.pts is not None and frame.time_base is not None
                    else None
                )
                viewer.update("input", frame, pts_time)
            await publisher.write_frame(frame)
    finally:
        stop.set()
        await asyncio.to_thread(capture.join, 2.0)
        if capture.is_alive():
            _log(
                "Camera decode is still stopping; its capture thread will close the input."
            )
        if publisher is not None:
            await publisher.close()
            _log("Media publish stats:", publisher.get_stats())


async def _run_media(
    args: argparse.Namespace,
    in_url: str,
    out_url: str,
    app_url: str,
    viewer: _CompositeViewer | None,
    output_file: BinaryIO | None,
    output_stdout: bool,
) -> None:
    streams: list[MediaOutput] = []
    publish = display = blur = None
    drain = False
    try:
        if viewer is not None:

            def _on_output(decoded) -> None:
                if decoded.kind == "video":
                    viewer.update("output", decoded.frame, decoded.pts_time)

            def _on_loopback(decoded) -> None:
                if decoded.kind == "video":
                    viewer.update("loopback", decoded.frame, decoded.pts_time)

            streams.append(MediaOutput(out_url, on_frame=_on_output))
            streams.append(MediaOutput(in_url, on_frame=_on_loopback))
        else:

            def _write_chunk(chunk: bytes) -> None:
                output_file.write(chunk)
                if output_stdout:
                    output_file.flush()

            streams.append(MediaOutput(out_url, on_bytes=_write_chunk))
        display = asyncio.create_task(viewer.run()) if viewer is not None else None
        publish = asyncio.create_task(_publish_video(args, in_url, viewer=viewer))
        blur = asyncio.create_task(_sweep_blur(app_url)) if args.blur else None
        waiters = {
            publish,
            *(task for stream in streams for task in stream.callback_tasks()),
        }
        if display is not None:
            waiters.add(display)
        if blur is not None:
            waiters.add(blur)
        done, _ = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
        drain = publish in done
        if drain:
            _log("publish complete; waiting for output to drain...")
    finally:
        close_error = None
        tasks = [task for task in (publish, blur) if task is not None]
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for stream in streams:
            try:
                await stream.close(wait_callbacks=drain)
            except Exception as exc:
                if close_error is None:
                    close_error = exc
                _log("Error closing media stream:", exc)
            _log(f"Media output stats ({stream.subscribe_url}):", stream.get_stats())
        if display is not None:
            display.cancel()
            await asyncio.gather(display, return_exceptions=True)
    # Reached only if the media loop had no error or cancellation to preserve.
    if close_error is not None:
        raise close_error


async def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO, stream=sys.stderr
    )
    av.logging.set_level(av.logging.ERROR)
    session = None
    try:
        if args.input:
            if not Path(args.input).expanduser().is_file():
                raise LivepeerGatewayError(f"input file does not exist: {args.input}")
        elif sys.platform != "darwin":
            raise LivepeerGatewayError(
                "Camera capture uses macOS avfoundation; supply a file input on other platforms."
            )
        viewer = _CompositeViewer() if args.output is None else None
        output_stdout = args.output is not None and args.output.strip().lower() in {
            "-",
            "stdout",
        }
        output_context = nullcontext(sys.stdout.buffer if output_stdout else None)
        if args.output is not None and not output_stdout:
            output_context = Path(args.output).expanduser().open("wb")
        with output_context as output_file:
            token = parse_token(args.token) if args.token else {}
            orchestrators = token.get("orchestrators") or (
                [args.orchestrator] if args.orchestrator else None
            )
            signer_url = token.get("signer") or args.signer
            discovery_url = token.get("discovery") or args.discovery
            if not (orchestrators or signer_url or discovery_url):
                discovery_url = DEFAULT_DISCOVERY
            session = await reserve_session(
                app=ECHO_APP_ID,
                orchestrators=orchestrators,
                signer_url=signer_url,
                signer_headers=token.get("signer_headers"),
                discovery_url=discovery_url,
                discovery_headers=token.get("discovery_headers"),
            )
            _log(
                "runner_url:",
                session.runner.url
                if session.runner is not None
                else session.runner_url,
            )
            _log("session_id:", session.session_id)
            _log("app_url:", session.app_url)
            echo = await post_json(f"{session.app_url.rstrip('/')}/echo", {})
            in_url, out_url = _channel_url(echo, "in"), _channel_url(echo, "out")
            _log("in:", in_url)
            _log("out:", out_url)
            await _run_media(
                args,
                in_url,
                out_url,
                session.app_url,
                viewer,
                output_file,
                output_stdout,
            )
            if output_file is not None:
                output_file.flush()
    except BrokenPipeError:
        _log("Playback pipe closed.")
        raise
    except (LivepeerGatewayError, av.FFmpegError, OSError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    finally:
        if session is not None:
            try:
                await stop_runner_session(session)
            except Exception as exc:
                _log("Error releasing runner session:", exc)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except BrokenPipeError:
        # Prevent the interpreter's final stdout flush from raising again.
        with open(os.devnull, "wb") as sink, suppress(OSError, ValueError):
            os.dup2(sink.fileno(), sys.stdout.fileno())
    except KeyboardInterrupt:
        _log("Stopped by user.")
