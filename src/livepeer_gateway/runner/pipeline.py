from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Pipeline(ABC):
    """Base class for batch inference pipelines."""

    @abstractmethod
    def predict(self, **kwargs: Any) -> Any:
        """Run one inference; kwargs are the JSON request body fields.

        Return any JSON-serialisable value, or raise to signal an error.
        """
        ...
