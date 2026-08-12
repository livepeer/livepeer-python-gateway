#!/usr/bin/env python3
from __future__ import annotations

import asyncio

from aiohttp import web


async def _handle_sse(request: web.Request) -> web.StreamResponse:
    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
    await response.prepare(request)

    with open("story.txt", encoding="utf-8", errors="replace") as lines:
        for line in lines:
            await response.write(f"data: {line.rstrip('\n')}\n\n".encode())
            await asyncio.sleep(0.5)

    await response.write_eof()
    return response


async def _handle_text(request: web.Request) -> web.StreamResponse:
    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/plain; charset=utf-8",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
    await response.prepare(request)

    with open("story.txt", encoding="utf-8", errors="replace") as story:
        while char := story.read(1):
            await response.write(char.encode("utf-8"))
            await asyncio.sleep(0.02)

    await response.write_eof()
    return response


async def _handle_health(_: web.Request) -> web.Response:
    return web.json_response({"ok": True})


def main() -> None:
    app = web.Application()
    app.router.add_get("/sse", _handle_sse)
    app.router.add_get("/text", _handle_text)
    app.router.add_get("/healthz", _handle_health)
    web.run_app(app, host="127.0.0.1", port=8990)


if __name__ == "__main__":
    main()
