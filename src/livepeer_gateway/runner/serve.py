from __future__ import annotations

import logging
from typing import Any

from aiohttp import web

from .pipeline import Pipeline

_LOG = logging.getLogger(__name__)


async def handle_predict(request: web.Request) -> web.Response:
    """Run one inference.

    Body is JSON; passed as kwargs to ``pipeline.predict()``. Returns the
    result as JSON. ``TypeError`` from ``predict()`` becomes HTTP 400;
    other exceptions become 500.
    """
    pipeline: Pipeline = request.app["pipeline"]

    try:
        body = await request.json()
    except Exception as exc:
        return web.json_response(
            {"error": f"invalid JSON body: {exc}"},
            status=400,
        )
    if not isinstance(body, dict):
        return web.json_response(
            {"error": "request body must be a JSON object"},
            status=400,
        )

    try:
        result: Any = pipeline.predict(**body)
    except TypeError as exc:
        return web.json_response(
            {"error": f"input mismatch: {exc}"},
            status=400,
        )
    except Exception:
        _LOG.exception("predict() failed")
        return web.json_response({"error": "internal error"}, status=500)

    return web.json_response(result)


async def handle_health(_: web.Request) -> web.Response:
    """Health probe. Returns ``{"status": "ready"}``."""
    return web.json_response({"status": "ready"})


def make_app(pipeline: Pipeline) -> web.Application:
    """Build an aiohttp application exposing ``pipeline`` over HTTP.

    Calls ``pipeline.setup()`` synchronously before binding routes, so the
    server only starts accepting requests once the pipeline is initialised.
    """
    pipeline.setup()
    app = web.Application()
    app["pipeline"] = pipeline
    app.router.add_post("/predict", handle_predict)
    app.router.add_get("/health", handle_health)
    return app


def serve(pipeline: Pipeline, *, host: str = "0.0.0.0", port: int = 5000) -> None:
    """Run the pipeline as an HTTP server on host:port."""
    web.run_app(make_app(pipeline), host=host, port=port)
