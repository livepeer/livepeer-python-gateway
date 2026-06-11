"""API key authentication for the web API.

Uses FastAPI dependency injection so auth integrates with OpenAPI docs.
When no API keys are configured (dev mode), auth is a no-op.
"""

from __future__ import annotations

from typing import Optional

from fastapi import Header, HTTPException, Query

from .config import Config


def get_api_key_dependency(config: Config):
    """Return a FastAPI dependency that validates API keys.

    When ``config.auth_enabled`` is False the dependency always returns None
    (open access / dev mode).  When enabled it checks the ``X-API-Key`` header.
    """

    async def verify_api_key(
        x_api_key: Optional[str] = Header(None),
    ) -> Optional[str]:
        if not config.auth_enabled:
            return None
        if not x_api_key or x_api_key not in config.parsed_api_keys:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        return x_api_key

    return verify_api_key


def get_ws_api_key_dependency(config: Config):
    """Return a FastAPI dependency for WebSocket API key validation.

    WebSocket clients often cannot set custom headers, so we accept the
    key as a query parameter ``api_key`` instead.
    """

    async def verify_ws_api_key(
        api_key: Optional[str] = Query(None),
    ) -> Optional[str]:
        if not config.auth_enabled:
            return None
        if not api_key or api_key not in config.parsed_api_keys:
            return "__REJECT__"
        return api_key

    return verify_ws_api_key
