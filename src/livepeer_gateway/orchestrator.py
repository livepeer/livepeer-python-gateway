from __future__ import annotations

from .discovery import _append_caps, discover_orchestrators
from .errors import (
    LivepeerGatewayError,
    SignerRefreshRequired,
    SkipPaymentCycle,
)
from .http import (
    _extract_error_message,
    _extract_error_message_from_body,
    _http_error_body,
    _http_origin,
    _json_request_parts,
    _parse_http_url,
    _raise_http_json_error,
    _truncate,
    get_json_sync,
    post_json_sync,
    request_json_sync,
)

# Compatibility aliases for the original synchronous helpers.
request_json = request_json_sync
post_json = post_json_sync
get_json = get_json_sync

__all__ = [
    "LivepeerGatewayError",
    "SignerRefreshRequired",
    "SkipPaymentCycle",
    "_append_caps",
    "_extract_error_message",
    "_extract_error_message_from_body",
    "_http_error_body",
    "_http_origin",
    "_json_request_parts",
    "_parse_http_url",
    "_raise_http_json_error",
    "_truncate",
    "discover_orchestrators",
    "get_json",
    "get_json_sync",
    "post_json",
    "post_json_sync",
    "request_json",
    "request_json_sync",
]
