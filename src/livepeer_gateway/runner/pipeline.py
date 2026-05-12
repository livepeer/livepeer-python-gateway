from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any


class PipelineState(str, Enum):
    """Wire-level health state — matches go-livepeer's HealthCheckStatus."""

    LOADING = "LOADING"
    OK = "OK"
    ERROR = "ERROR"
    IDLE = "IDLE"


class Pipeline(ABC):
    """Base class for batch pipelines.

    Override ``run()`` with the work the pipeline does — inference, generation,
    transformation, agent invocation, anything that fits a request/response (or
    streaming-response) shape. The method name is intentionally generic; the
    class name and docstring describe the specific workload.
    """

    _state: PipelineState = PipelineState.LOADING

    def setup(self) -> None:
        """Hook called once before serve() accepts requests.

        Override to load model weights, warm up GPUs, allocate buffers.
        Default: no-op for stateless pipelines.
        """

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """Execute the pipeline once; kwargs are the JSON request body fields.

        Return any JSON-serialisable value, or raise to signal an error.
        Yield to stream results as Server-Sent Events.
        """
        ...
