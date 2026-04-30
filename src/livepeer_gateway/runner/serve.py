import inspect
import json
import logging
from typing import Any, Iterator

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, create_model

from .live_pipeline import LivePipeline, StreamParamsRequest, StreamStartRequest
from .pipeline import Pipeline, PipelineState

logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    """Response shape for ``GET /health`` — matches go-livepeer's HealthCheck."""

    status: PipelineState


def _is_basemodel(t: Any) -> bool:
    return isinstance(t, type) and issubclass(t, BaseModel)


def _build_input_model(predict_fn: Any, owner_name: str) -> tuple[type[BaseModel], bool]:
    """Inspect predict()'s signature; return (InputModel, is_explicit_basemodel).

    If predict() takes a single ``BaseModel`` parameter, use it directly.
    Otherwise build a model from the bare parameters via ``create_model``.
    """
    sig = inspect.signature(predict_fn)
    params = [param for param in sig.parameters.values() if param.name != "self"]

    if len(params) == 1 and _is_basemodel(params[0].annotation):
        return params[0].annotation, True

    fields: dict[str, tuple[Any, Any]] = {}
    for param in params:
        annotation = param.annotation if param.annotation is not inspect.Parameter.empty else Any
        default = param.default if param.default is not inspect.Parameter.empty else ...
        fields[param.name] = (annotation, default)

    return create_model(f"{owner_name}Input", **fields), False


def _format_sse(generator: Iterator[Any]) -> Iterator[bytes]:
    """Frame yielded values as SSE events with [DONE] terminator.

    Required by go-livepeer and the Python caller-side gateway.
    """
    try:
        for chunk in generator:
            payload = chunk.model_dump_json() if isinstance(chunk, BaseModel) else json.dumps(chunk)
            yield f"data: {payload}\n\n".encode()
    except Exception:
        logger.exception("predict() generator failed")
        yield b'data: {"error": "internal error"}\n\n'
    yield b"data: [DONE]\n\n"


def _build_predict_handler(
    pipeline: Pipeline,
    InputModel: type[BaseModel],
    OutputModel: type[BaseModel] | None,
    explicit_basemodel: bool,
    is_generator: bool,
):
    def handler(body: InputModel):
        try:
            if explicit_basemodel:
                result = pipeline.predict(body)
            else:
                result = pipeline.predict(**body.model_dump())
        except HTTPException:
            raise
        except Exception:
            logger.exception("predict() failed")
            raise HTTPException(status_code=500, detail="internal error")

        if is_generator:
            return StreamingResponse(_format_sse(result), media_type="text/event-stream")
        return result

    if OutputModel is not None and not is_generator:
        handler.__annotations__["return"] = OutputModel
    return handler


def _add_health_route(app: FastAPI, pipeline: Pipeline | LivePipeline) -> None:
    @app.get("/health", summary="Liveness probe", response_model=HealthResponse)
    def handle_health() -> HealthResponse:
        return HealthResponse(status=pipeline._state)


def _run_setup(pipeline: Pipeline | LivePipeline) -> None:
    pipeline._state = PipelineState.LOADING
    try:
        pipeline.setup()
        pipeline._state = PipelineState.OK
    except Exception:
        pipeline._state = PipelineState.ERROR
        logger.exception("setup() failed")
        raise


def _make_live_pipeline_app(pipeline: LivePipeline) -> FastAPI:
    """Build a FastAPI app for a real-time ``LivePipeline``."""
    _run_setup(pipeline)

    app = FastAPI(title=type(pipeline).__name__)
    app.state.pipeline = pipeline

    @app.post("/stream/start", summary="Start a stream session")
    async def handle_stream_start(body: StreamStartRequest) -> dict[str, Any]:
        logger.info("stream/start request_id=%s", body.gateway_request_id)
        return {"status": "started", "gateway_request_id": body.gateway_request_id}

    @app.post("/stream/stop", summary="Stop the active stream session")
    async def handle_stream_stop() -> dict[str, str]:
        logger.info("stream/stop")
        return {"status": "stopped"}

    @app.post("/stream/params", summary="Update params on the active stream")
    async def handle_stream_params(body: StreamParamsRequest) -> dict[str, str]:
        logger.info("stream/params keys=%s", list(body.model_dump().keys()))
        return {"status": "ok"}

    _add_health_route(app, pipeline)
    return app


def _make_pipeline_app(pipeline: Pipeline) -> FastAPI:
    """Build a FastAPI app for a request/response ``Pipeline`` (HTTP `/predict`)."""
    _run_setup(pipeline)

    is_generator = inspect.isgeneratorfunction(pipeline.predict)

    InputModel, explicit_basemodel = _build_input_model(
        pipeline.predict, type(pipeline).__name__
    )
    return_annotation = inspect.signature(pipeline.predict).return_annotation
    OutputModel = return_annotation if _is_basemodel(return_annotation) else None

    handler = _build_predict_handler(
        pipeline, InputModel, OutputModel, explicit_basemodel, is_generator
    )

    app = FastAPI(title=type(pipeline).__name__)
    app.state.pipeline = pipeline

    app.add_api_route(
        "/predict",
        handler,
        methods=["POST"],
        summary="Run one inference",
    )

    _add_health_route(app, pipeline)
    return app


def make_app(pipeline: Pipeline | LivePipeline) -> FastAPI:
    """Build a FastAPI app exposing ``pipeline`` over HTTP.

    Dispatches on `LivePipeline` vs `Pipeline` to register `/stream/*`
    or `/predict` respectively.
    """
    if isinstance(pipeline, LivePipeline):
        return _make_live_pipeline_app(pipeline)
    return _make_pipeline_app(pipeline)


def serve(
    pipeline: Pipeline | LivePipeline, *, host: str = "0.0.0.0", port: int = 5000
) -> None:
    """Run the pipeline as an HTTP server on host:port."""
    uvicorn.run(make_app(pipeline), host=host, port=port)
