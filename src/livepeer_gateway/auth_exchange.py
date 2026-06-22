from __future__ import annotations

import logging
from typing import Any, Optional

from .errors import LivepeerGatewayError
from .orchestrator import post_json

_LOG = logging.getLogger(__name__)
DEFAULT_SCOPE = "sign:job"


def _signer_access_token(payload: dict[str, Any]) -> str:
    token_obj = payload.get("token")
    if isinstance(token_obj, dict):
        for key in ("accessToken", "access_token"):
            value = token_obj.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for key in ("accessToken", "access_token"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise LivepeerGatewayError("API key exchange response missing signer access token")


def _signer_url(payload: dict[str, Any]) -> Optional[str]:
    for key in ("signerUrl", "signer_url"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def exchange_api_key_for_signer(
    billing_url: str,
    api_key: str,
    *,
    client_id: Optional[str] = None,
    scope: Optional[str] = DEFAULT_SCOPE,
    timeout: float = 15.0,
) -> tuple[Optional[str], dict[str, str]]:
    """
    Exchange a PymtHouse API key (``pmth_*``) for a signer JWT via the Dashboard BFF.

    Returns ``(signer_url, {"Authorization": "Bearer <jwt>"})``.
    """
    key = api_key.strip()
    if not key:
        raise LivepeerGatewayError("API key exchange requires a non-empty API key")
    url = f"{billing_url.rstrip('/')}/api/pymthouse/keys/exchange"
    body: dict[str, Any] = {"apiKey": key}
    if client_id:
        body["clientId"] = client_id
    if scope:
        body["scope"] = scope
    _LOG.info("Exchanging API key for signer JWT at %s", url)
    data = post_json(url, body, timeout=timeout)
    return _signer_url(data), {"Authorization": f"Bearer {_signer_access_token(data)}"}
