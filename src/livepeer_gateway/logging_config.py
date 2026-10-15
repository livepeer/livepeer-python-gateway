from __future__ import annotations

import logging
import os


_PACKAGE_LOGGER = "livepeer_gateway"
_DEFAULT_LEVEL = logging.INFO
_LOG_FORMAT = "%(levelname)s %(name)s: %(message)s"


def _resolve_level(level: str | int | None = None) -> int:
    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO")
    if isinstance(level, int):
        return level
    return getattr(logging, str(level).upper(), _DEFAULT_LEVEL)


def apply_package_log_level(level: str | int | None = None) -> None:
    """Set the livepeer_gateway logger level from arg or LOG_LEVEL env (default INFO)."""
    logging.getLogger(_PACKAGE_LOGGER).setLevel(_resolve_level(level))


def configure_logging(level: str | int | None = None) -> None:
    """Configure livepeer_gateway log level from arg or LOG_LEVEL env (default INFO).

    Ensures a basic root handler exists so package logs are visible.
    """
    resolved = _resolve_level(level)
    apply_package_log_level(resolved)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=resolved, format=_LOG_FORMAT)
