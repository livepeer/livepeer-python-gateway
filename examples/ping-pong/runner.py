#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from contextlib import suppress

from aiohttp import web

from livepeer_gateway.live_runner import LiveRunnerRegistration, register_runner

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8991
APP_ID = "livepeer-sample/ping-pong"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live Runner websocket ping/pong demo.")
    parser.add_argument("--orchestrator", default="http://localhost:8935")
    parser.add_argument("--orchSecret", default="abcdef")
    parser.add_argument("--runner-url", default=f"http://{DEFAULT_HOST}:{DEFAULT_PORT}")
    return parser.parse_args()


def _pong_response(payload: str, *, now: float | None = None) -> dict[str, float]:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("message must be JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("message must be a JSON object")

    ping = data.get("ping")
    if isinstance(ping, bool) or not isinstance(ping, (int, float)):
        raise ValueError("message must include numeric ping")

    received_at = time.time() if now is None else now
    return {
        "pong": float(ping),
        "delta_ms": max(0.0, (received_at - float(ping)) * 1000.0),
    }


async def _handle_ws(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    print("websocket session opened")

    try:
        async for msg in ws:
            if msg.type != web.WSMsgType.TEXT:
                continue
            try:
                response = _pong_response(msg.data)
            except ValueError as exc:
                await ws.send_json({"error": str(exc)})
                continue
            await ws.send_json(response)
    finally:
        print("websocket session closed")

    return ws


async def _on_startup(app: web.Application) -> None:
    args = _parse_args()
    registration = await register_runner(
        args.orchestrator,
        secret=args.orchSecret,
        runner_url=args.runner_url,
        app=APP_ID,
        mode="single-shot",
    )
    app["registration"] = registration
    print(
        f"runner_id={registration.runner_id} orchestrator={registration.orchestrator_url}"
    )


async def _on_cleanup(app: web.Application) -> None:
    registration = app.get("registration")
    if isinstance(registration, LiveRunnerRegistration):
        with suppress(Exception):
            await registration.close()


def main() -> None:
    app = web.Application()
    app.router.add_get("/ws", _handle_ws)
    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)
    web.run_app(app, host=DEFAULT_HOST, port=DEFAULT_PORT)


if __name__ == "__main__":
    main()
