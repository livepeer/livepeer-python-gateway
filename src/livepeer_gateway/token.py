from __future__ import annotations

import base64
import binascii
import json
from typing import Any, Optional

from .errors import LivepeerGatewayError


def _is_str_dict(v: object) -> bool:
    return isinstance(v, dict) and all(isinstance(k, str) and isinstance(val, str) for k, val in v.items())


def parse_token(token: str) -> dict[str, Any]:
    try:
        decoded = base64.b64decode(token, validate=True)
    except (binascii.Error, ValueError) as e:
        raise LivepeerGatewayError("Invalid token: expected base64-encoded JSON") from e

    try:
        payload = json.loads(decoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise LivepeerGatewayError("Invalid token: expected UTF-8 JSON payload") from e

    if not isinstance(payload, dict):
        raise LivepeerGatewayError("Invalid token: payload must be a JSON object")

    signer = payload.get("signer")
    discovery = payload.get("discovery")
    if signer is not None and not isinstance(signer, str):
        raise LivepeerGatewayError("Invalid token: signer must be a string")
    if discovery is not None and not isinstance(discovery, str):
        raise LivepeerGatewayError("Invalid token: discovery must be a string")

    signer_headers = payload.get("signer_headers")
    discovery_headers = payload.get("discovery_headers")
    orchestrators = payload.get("orchestrators")
    if signer_headers is not None and not _is_str_dict(signer_headers):
        raise LivepeerGatewayError("Invalid token: signer_headers must be a {string: string} object")
    if discovery_headers is not None and not _is_str_dict(discovery_headers):
        raise LivepeerGatewayError("Invalid token: discovery_headers must be a {string: string} object")
    if orchestrators is not None and not isinstance(orchestrators, list):
        raise LivepeerGatewayError("Invalid token: orchestrators must be an array of strings")

    normalized_orchestrators: Optional[list[str]] = None
    if isinstance(orchestrators, list):
        normalized_orchestrators = []
        for item in orchestrators:
            if not isinstance(item, str) or not item.strip():
                raise LivepeerGatewayError(
                    "Invalid token: orchestrators must contain only non-empty strings"
                )
            normalized_orchestrators.append(item.strip())

    return {
        "orchestrators": normalized_orchestrators,
        "signer": signer,
        "discovery": discovery,
        "signer_headers": signer_headers,
        "discovery_headers": discovery_headers,
    }
