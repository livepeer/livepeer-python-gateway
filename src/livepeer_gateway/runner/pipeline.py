from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Pipeline(ABC):
    """Base class for batch inference pipelines."""

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
