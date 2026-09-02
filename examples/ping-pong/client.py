#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

import aiohttp

from livepeer_gateway.errors import LivepeerGatewayError
from livepeer_gateway.live_runner import aiohttp_connector
from livepeer_gateway.selection import runner_selector

DEFAULT_DISCOVERY = "http://localhost:8935/discovery"
APP_ID = "livepeer-sample/ping-pong"


def _log(*args: object) -> None:
    print(*args, file=sys.stderr)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the websocket ping/pong Live Runner demo.")
    parser.add_argument("--discovery", default=DEFAULT_DISCOVERY)
    parser.add_argument("--count", type=int, default=10, help="Stop after this many pings (0 = until closed).")
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Skip TLS certificate verification (self-signed localhost).",
    )
    return parser.parse_args()

async def _select_runner(discovery_url: str) -> str:
    cursor = await runner_selector(discovery_url=discovery_url, app=APP_ID)
    for candidate in cursor.candidates:
        return candidate.url
    raise LivepeerGatewayError(f"no websocket runner discovered for app {APP_ID!r}")


async def _run_client(url: str, *, count: int, insecure: bool) -> None:
    async with aiohttp.ClientSession(
        connector=aiohttp_connector(insecure=insecure),
    ) as session:
        async with session.ws_connect(url) as ws:
            _log("connected:", url)
            sent = 0
            while count <= 0 or sent < count:
                ping = time.time()
                await ws.send_json({"ping": ping})
                sent += 1

                msg = json.loads((await ws.receive()).data)
                received_at = time.time()
                receiver_delta_ms = float(msg.get("delta_ms", -1))
                round_trip_ms = (received_at - ping) * 1000.0
                print(
                    f"ping-pong receiver_delta_ms={receiver_delta_ms:.2f} round_trip_ms={round_trip_ms:.2f}"
                )

                elapsed = time.time() - ping
                await asyncio.sleep(max(0.0, 1.0 - elapsed))


async def main() -> None:
    args = _parse_args()
    app_url = await _select_runner(args.discovery)
    await _run_client(app_url + "/ws", count=max(0, args.count), insecure=args.insecure)


if __name__ == "__main__":
    asyncio.run(main())
