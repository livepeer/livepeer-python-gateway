from __future__ import annotations

import logging
from typing import Any

import uvicorn
from fastapi import Body, FastAPI, HTTPException

from .pipeline import Pipeline

_LOG = logging.getLogger(__name__)


def make_app(pipeline: Pipeline) -> FastAPI:
    """Build a FastAPI app exposing ``pipeline`` over HTTP."""
    pipeline.setup()

    app = FastAPI(title=type(pipeline).__name__)
    app.state.pipeline = pipeline

    @app.post("/predict", summary="Run one inference")
    def handle_predict(body: dict = Body(...)) -> Any:
        try:
            return pipeline.predict(**body)
        except TypeError as exc:
            raise HTTPException(status_code=400, detail=f"input mismatch: {exc}")
        except HTTPException:
            raise
        except Exception:
            _LOG.exception("predict() failed")
            raise HTTPException(status_code=500, detail="internal error")

    @app.get("/health", summary="Liveness probe")
    def handle_health() -> dict:
        return {"status": "ready"}

    return app


def serve(pipeline: Pipeline, *, host: str = "0.0.0.0", port: int = 5000) -> None:
    """Run the pipeline as an HTTP server on host:port."""
    uvicorn.run(make_app(pipeline), host=host, port=port)
