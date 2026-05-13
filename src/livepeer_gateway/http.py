from __future__ import annotations

import asyncio
import json
import ssl
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import ParseResult, urlparse
from urllib.request import Request, urlopen

import aiohttp

from .errors import (
    LivepeerGatewayError,
    SignerRefreshRequired,
    SkipPaymentCycle,
)


def _truncate(s: str, max_len: int = 2000) -> str:
    if len(s) <= max_len:
        return s
    return s[:max_len] + f"...(+{len(s) - max_len} chars)"


def _http_error_body(e: HTTPError) -> str:
    """
    Best-effort read of an HTTPError response body for debugging.
    """
    try:
        b = e.read()
        if not b:
            return ""
        if isinstance(b, bytes):
            return b.decode("utf-8", errors="replace")
        return str(b)
    except Exception:
        return ""


def _extract_error_message_from_body(body: str) -> str:
    """
    Best-effort extraction of a useful error message from an HTTP error body.

    If the body is JSON and matches {"error": {"message": "..."}}, return that message.
    Otherwise return the full body.

    Always truncates the returned value for readability.
    """
    s = body.strip()
    if not s:
        return ""

    try:
        data = json.loads(s)
    except Exception:
        return _truncate(body)

    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict):
            msg = err.get("message")
            if isinstance(msg, str) and msg:
                return _truncate(msg)

    return _truncate(body)


def _extract_error_message(e: HTTPError) -> str:
    """
    Best-effort extraction of a useful error message from an HTTPError body.
    """
    return _extract_error_message_from_body(_http_error_body(e))


def _json_request_parts(
    url: str,
    *,
    method: Optional[str] = None,
    payload: Optional[dict[str, Any]] = None,
    headers: Optional[dict[str, str]] = None,
) -> tuple[str, dict[str, str], Optional[bytes]]:
    req_headers: dict[str, str] = {
        "Accept": "application/json",
        "User-Agent": "livepeer-python-gateway/0.1",
    }
    body: Optional[bytes] = None
    if payload is not None:
        req_headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode("utf-8")
    if headers:
        req_headers.update(headers)

    resolved_method = method.upper() if method else ("POST" if payload is not None else "GET")
    return resolved_method, req_headers, body


def _raise_http_json_error(status: int, url: str, body: str = "") -> None:
    message = _extract_error_message_from_body(body)
    body_part = f"; body={message!r}" if message else ""
    if status == 480:
        raise SignerRefreshRequired(
            f"Signer returned HTTP 480 (refresh session required) (url={url}){body_part}"
        )
    if status == 482:
        raise SkipPaymentCycle(
            f"Signer returned HTTP 482 (skip payment cycle) (url={url}){body_part}"
        )
    raise LivepeerGatewayError(
        f"HTTP JSON error: HTTP {status} from endpoint (url={url}){body_part}"
    )


def _ensure_json_object(data: Any, *, url: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise LivepeerGatewayError(
            f"HTTP JSON error: expected JSON object, got {type(data).__name__} (url={url})"
        )
    return data


def request_json_sync(
    url: str,
    *,
    method: Optional[str] = None,
    payload: Optional[dict[str, Any]] = None,
    headers: Optional[dict[str, str]] = None,
    timeout: float = 5.0,
) -> Any:
    """
    Make a JSON HTTP request and parse the JSON response.

    If method is None, defaults to POST when payload is provided, otherwise GET.

    Raises LivepeerGatewayError on HTTP/network/JSON parsing errors.
    """
    resolved_method, req_headers, body = _json_request_parts(
        url,
        method=method,
        payload=payload,
        headers=headers,
    )
    req = Request(url, data=body, headers=req_headers, method=resolved_method)

    # Always ignore HTTPS certificate validation (matches our gRPC behavior).
    ssl_ctx = ssl._create_unverified_context()

    try:
        with urlopen(req, timeout=timeout, context=ssl_ctx) as resp:
            raw = resp.read().decode("utf-8")
        data: Any = json.loads(raw)
    except HTTPError as e:
        body_text = _extract_error_message(e)
        body_part = f"; body={body_text!r}" if body_text else ""
        if e.code == 480:
            raise SignerRefreshRequired(
                f"Signer returned HTTP 480 (refresh session required) (url={url}){body_part}"
            ) from e
        if e.code == 482:
            raise SkipPaymentCycle(
                f"Signer returned HTTP 482 (skip payment cycle) (url={url}){body_part}"
            ) from e
        raise LivepeerGatewayError(
            f"HTTP JSON error: HTTP {e.code} from endpoint (url={url}){body_part}"
        ) from e
    except ConnectionRefusedError as e:
        raise LivepeerGatewayError(
            f"HTTP JSON error: connection refused (is the server running? is the host/port correct?) (url={url})"
        ) from e
    except URLError as e:
        raise LivepeerGatewayError(
            f"HTTP JSON error: failed to reach endpoint: {getattr(e, 'reason', e)} (url={url})"
        ) from e
    except json.JSONDecodeError as e:
        raise LivepeerGatewayError(f"HTTP JSON error: endpoint did not return valid JSON: {e} (url={url})") from e
    except Exception as e:
        raise LivepeerGatewayError(
            f"HTTP JSON error: unexpected error: {e.__class__.__name__}: {e} (url={url})"
        ) from e

    return data


def post_json_sync(
    url: str,
    payload: dict[str, Any],
    *,
    headers: Optional[dict[str, str]] = None,
    timeout: float = 5.0,
) -> dict[str, Any]:
    """
    POST JSON to `url` and parse a JSON object response.
    """
    data = request_json_sync(
        url,
        payload=payload,
        headers=headers,
        timeout=timeout,
    )
    return _ensure_json_object(data, url=url)


def get_json_sync(
    url: str,
    *,
    headers: Optional[dict[str, str]] = None,
    timeout: float = 5.0,
) -> Any:
    """
    GET JSON from `url` and parse the response.
    """
    return request_json_sync(url, headers=headers, timeout=timeout)


async def request_json(
    url: str,
    *,
    method: Optional[str] = None,
    payload: Optional[dict[str, Any]] = None,
    headers: Optional[dict[str, str]] = None,
    timeout: float = 5.0,
) -> Any:
    """
    Make an async JSON HTTP request and parse the JSON response.

    If method is None, defaults to POST when payload is provided, otherwise GET.

    Raises LivepeerGatewayError on HTTP/network/JSON parsing errors.
    """
    resolved_method, req_headers, body = _json_request_parts(
        url,
        method=method,
        payload=payload,
        headers=headers,
    )

    try:
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(timeout=client_timeout, connector=connector) as session:
            async with session.request(resolved_method, url, data=body, headers=req_headers) as resp:
                raw = await resp.text()
                if resp.status >= 400:
                    _raise_http_json_error(resp.status, url, raw)
        data: Any = json.loads(raw)
    except (SignerRefreshRequired, SkipPaymentCycle, LivepeerGatewayError):
        raise
    except json.JSONDecodeError as e:
        raise LivepeerGatewayError(f"HTTP JSON error: endpoint did not return valid JSON: {e} (url={url})") from e
    except ConnectionRefusedError as e:
        raise LivepeerGatewayError(
            f"HTTP JSON error: connection refused (is the server running? is the host/port correct?) (url={url})"
        ) from e
    except getattr(aiohttp, "ClientConnectorError", ()) as e:
        os_error = getattr(e, "os_error", None)
        if isinstance(os_error, ConnectionRefusedError):
            raise LivepeerGatewayError(
                f"HTTP JSON error: connection refused (is the server running? is the host/port correct?) (url={url})"
            ) from e
        raise LivepeerGatewayError(
            f"HTTP JSON error: failed to reach endpoint: {getattr(e, 'message', e)} (url={url})"
        ) from e
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        raise LivepeerGatewayError(
            f"HTTP JSON error: failed to reach endpoint: {getattr(e, 'message', e)} (url={url})"
        ) from e
    except Exception as e:
        raise LivepeerGatewayError(
            f"HTTP JSON error: unexpected error: {e.__class__.__name__}: {e} (url={url})"
        ) from e

    return data


async def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    headers: Optional[dict[str, str]] = None,
    timeout: float = 5.0,
) -> dict[str, Any]:
    """
    POST JSON to `url` and parse a JSON object response.
    """
    data = await request_json(
        url,
        payload=payload,
        headers=headers,
        timeout=timeout,
    )
    return _ensure_json_object(data, url=url)


async def get_json(
    url: str,
    *,
    headers: Optional[dict[str, str]] = None,
    timeout: float = 5.0,
) -> Any:
    """
    GET JSON from `url` and parse the response.
    """
    return await request_json(url, headers=headers, timeout=timeout)


def _parse_http_url(url: str, *, context: str = "URL") -> ParseResult:
    """
    Normalize a URL for HTTP(S) endpoints.

    Accepts:
    - "host:port" (implicitly https://host:port)
    - "http://host:port[/...]"
    - "https://host:port[/...]"
    """
    url = url.strip()
    normalized = url if "://" in url else f"https://{url}"
    parsed = urlparse(normalized)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Only http:// or https:// {context}s are supported (got {parsed.scheme!r})")
    if not parsed.netloc:
        raise ValueError(f"Invalid {context}: {url!r}")
    return parsed


def _http_origin(url: str) -> str:
    """
    Normalize a URL (possibly with a path) into a scheme:// origin (scheme + host:port).

    Accepts:
    - "host:port" (implicitly https://host:port)
    - "http://host:port[/...]" (path/query/fragment are ignored)
    - "https://host:port[/...]" (path/query/fragment are ignored)
    """
    parsed = _parse_http_url(url)
    return f"{parsed.scheme}://{parsed.netloc}"
