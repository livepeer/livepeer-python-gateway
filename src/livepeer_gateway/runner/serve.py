import inspect
import logging
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, create_model

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


def _build_predict_handler(
    pipeline: Pipeline,
    InputModel: type[BaseModel],
    OutputModel: type[BaseModel] | None,
    explicit_basemodel: bool,
):
    def handler(body: InputModel):
        try:
            if explicit_basemodel:
                return pipeline.predict(body)
            return pipeline.predict(**body.model_dump())
        except HTTPException:
            raise
        except Exception:
            logger.exception("predict() failed")
            raise HTTPException(status_code=500, detail="internal error")

    if OutputModel is not None:
        handler.__annotations__["return"] = OutputModel
    return handler


def make_app(pipeline: Pipeline) -> FastAPI:
    """Build a FastAPI app exposing ``pipeline`` over HTTP."""
    pipeline._state = PipelineState.LOADING
    try:
        pipeline.setup()
        pipeline._state = PipelineState.OK
    except Exception:
        pipeline._state = PipelineState.ERROR
        logger.exception("setup() failed")
        raise

    InputModel, explicit_basemodel = _build_input_model(
        pipeline.predict, type(pipeline).__name__
    )
    return_annotation = inspect.signature(pipeline.predict).return_annotation
    OutputModel = return_annotation if _is_basemodel(return_annotation) else None

    handler = _build_predict_handler(pipeline, InputModel, OutputModel, explicit_basemodel)

    app = FastAPI(title=type(pipeline).__name__)
    app.state.pipeline = pipeline

    app.add_api_route(
        "/predict",
        handler,
        methods=["POST"],
        summary="Run one inference",
    )

    @app.get("/health", summary="Liveness probe", response_model=HealthResponse)
    def handle_health() -> HealthResponse:
        return HealthResponse(status=pipeline._state)

    return app


def serve(pipeline: Pipeline, *, host: str = "0.0.0.0", port: int = 5000) -> None:
    """Run the pipeline as an HTTP server on host:port."""
    uvicorn.run(make_app(pipeline), host=host, port=port)
