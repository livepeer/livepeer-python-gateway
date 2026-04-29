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
    """Base class for batch inference pipelines."""

    _state: PipelineState = PipelineState.LOADING

    def setup(self) -> None:
        """Hook called once before serve() accepts requests.

        Override to load model weights, warm up GPUs, allocate buffers.
        Default: no-op for stateless pipelines.
        """

    @abstractmethod
    def predict(self, **kwargs: Any) -> Any:
        """Run one inference; kwargs are the JSON request body fields.

        Return any JSON-serialisable value, or raise to signal an error.
        """
        ...
