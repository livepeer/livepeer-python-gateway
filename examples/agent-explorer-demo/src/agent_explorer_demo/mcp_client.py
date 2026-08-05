from __future__ import annotations

import base64
import json
import os
from typing import Any, Optional

from livepeer_gateway.token import parse_token
from mcp import ClientSession
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

from .state import DemoState, McpSessionState, StateStore


class McpClientError(RuntimeError):
    """Hosted MCP client failure."""


def mcp_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/api/v1/mcp"


def _parse_tool_payload(result: Any) -> Any:
    """Extract JSON (or raw text) from an MCP CallToolResult."""
    if result is None:
        return None
    if getattr(result, "isError", False):
        raise McpClientError(f"MCP tool error: {_result_text(result)}")
    text = _result_text(result)
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _result_text(result: Any) -> str:
    content = getattr(result, "content", None) or []
    parts: list[str] = []
    for item in content:
        text = getattr(item, "text", None)
        if isinstance(text, str):
            parts.append(text)
        elif isinstance(item, dict) and isinstance(item.get("text"), str):
            parts.append(item["text"])
    return "\n".join(parts).strip()


async def _call_tool(
    session: ClientSession,
    name: str,
    arguments: Optional[dict[str, Any]] = None,
) -> Any:
    result = await session.call_tool(name, arguments or {})
    return _parse_tool_payload(result)


async def run_mcp_session(
    store: StateStore,
    *,
    base_url: str,
    api_key: str,
    model_id: str = "noop",
) -> McpSessionState:
    """Connect to hosted MCP: info, capabilities smoke, create_signer_session."""
    url = mcp_url(base_url)
    headers = {"Authorization": f"Bearer {api_key}"}

    async with create_mcp_http_client(headers) as http:
        async with streamable_http_client(url, http_client=http) as streams:
            read_stream, write_stream = streams
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                info = await _call_tool(session, "livepeer_mcp_info")
                capabilities = await _call_tool(session, "list_capabilities")

                orchestrators: Any = None
                try:
                    orchestrators = await _call_tool(
                        session,
                        "query_orchestrators",
                        {
                            "capabilities": [model_id],
                            "service_types": ["live-video-to-video"],
                            "top_n": 5,
                        },
                    )
                except McpClientError as exc:
                    orchestrators = {"error": str(exc)}

                signer_session = await _call_tool(session, "create_signer_session")
                if not isinstance(signer_session, dict):
                    raise McpClientError(
                        f"create_signer_session returned unexpected payload: {signer_session!r}"
                    )

    mcp_state = apply_mcp_signer_overrides(
        McpSessionState(
            access_token=(
                signer_session.get("access_token")
                if isinstance(signer_session.get("access_token"), str)
                else None
            ),
            signer_url=(
                signer_session.get("signer_url")
                if isinstance(signer_session.get("signer_url"), str)
                else None
            ),
            discovery_url=(
                signer_session.get("discovery_url")
                if isinstance(signer_session.get("discovery_url"), str)
                else None
            ),
            sdk_token=(
                signer_session.get("sdk_token")
                if isinstance(signer_session.get("sdk_token"), str)
                else None
            ),
            client_id=(
                signer_session.get("client_id")
                if isinstance(signer_session.get("client_id"), str)
                else None
            ),
            balance_usd_micros=_stringify_optional(
                signer_session.get("balanceUsdMicros")
                if "balanceUsdMicros" in signer_session
                else signer_session.get("balance_usd_micros")
            ),
            info=info if isinstance(info, dict) else {"raw": info},
            capabilities=(
                capabilities if isinstance(capabilities, dict) else {"raw": capabilities}
            ),
            orchestrators=(
                orchestrators if isinstance(orchestrators, dict) else {"raw": orchestrators}
            ),
        )
    )

    state = store.load()
    state.mcp = mcp_state
    if mcp_state.sdk_token and not state.register.sdk_token:
        state.register.sdk_token = mcp_state.sdk_token
    store.save(state)
    return mcp_state


def resolve_sdk_token(state: DemoState) -> Optional[str]:
    """Prefer MCP sdk_token; fall back to register-time sdkToken.

    Applies SIGNER_URL / DISCOVERY_URL overrides when set so the job can use
    a production remote signer while register/MCP stay on PYMTHOUSE_BASE_URL.
    """
    raw: Optional[str] = None
    if state.mcp.sdk_token:
        raw = state.mcp.sdk_token
    elif state.register.sdk_token:
        raw = state.register.sdk_token
    if not raw:
        return None
    return apply_signer_url_overrides(raw)


def env_signer_url() -> Optional[str]:
    value = os.environ.get("SIGNER_URL", "").strip()
    return value.rstrip("/") or None


def env_discovery_url() -> Optional[str]:
    value = os.environ.get("DISCOVERY_URL", "").strip()
    return value or None


def apply_signer_url_overrides(sdk_token: str) -> str:
    """Rewrite token signer/discovery while preserving Authorization headers."""
    signer = env_signer_url()
    discovery = env_discovery_url()
    if not signer and not discovery:
        return sdk_token

    data = parse_token(sdk_token)
    payload: dict[str, Any] = {}
    if data.get("orchestrators") is not None:
        payload["orchestrators"] = data["orchestrators"]
    payload["signer"] = signer or data.get("signer")
    if discovery:
        payload["discovery"] = discovery
    elif signer:
        payload["discovery"] = f"{signer}/discover-orchestrators"
    elif data.get("discovery"):
        payload["discovery"] = data["discovery"]
    if data.get("signer_headers") is not None:
        payload["signer_headers"] = data["signer_headers"]
    if data.get("discovery_headers") is not None:
        payload["discovery_headers"] = data["discovery_headers"]
    return base64.b64encode(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")


def apply_mcp_signer_overrides(mcp: McpSessionState) -> McpSessionState:
    """Update stored MCP session URLs/token when SIGNER_URL overrides are set."""
    signer = env_signer_url()
    discovery = env_discovery_url()
    if not signer and not discovery:
        return mcp
    if signer:
        mcp.signer_url = signer
        mcp.discovery_url = discovery or f"{signer}/discover-orchestrators"
    elif discovery:
        mcp.discovery_url = discovery
    if mcp.sdk_token:
        mcp.sdk_token = apply_signer_url_overrides(mcp.sdk_token)
    return mcp


def _stringify_optional(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)
