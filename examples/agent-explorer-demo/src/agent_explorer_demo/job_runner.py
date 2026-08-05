from __future__ import annotations

import asyncio
import inspect
from fractions import Fraction
from typing import Any, Optional, Sequence

import av
from livepeer_gateway.errors import LivepeerGatewayError, NoOrchestratorAvailableError
from livepeer_gateway.lv2v import StartJobRequest, start_lv2v
from livepeer_gateway.media_publish import MediaPublishConfig, VideoOutputConfig
from livepeer_gateway.token import parse_token

from .mcp_client import resolve_sdk_token
from .state import DemoState, JobState, StateStore


class JobRunnerError(RuntimeError):
    """LV2V job submission failure."""


def _format_job_error(exc: BaseException) -> str:
    message = str(exc)
    if isinstance(exc, NoOrchestratorAvailableError):
        rejections = getattr(exc, "rejections", None) or []
        reasons: list[str] = []
        for item in rejections[:3]:
            reason = getattr(item, "reason", None) or str(item)
            if reason:
                reasons.append(reason)
        if reasons:
            message = f"{message}: {'; '.join(reasons)}"
        if any(
            "not a valid access token for this issuer" in r
            or "oidc verification failed" in r
            for r in reasons
        ):
            message = (
                f"{message}. Signer rejected the session token (issuer mismatch). "
                "SIGNER_URL must share an OIDC issuer with PYMTHOUSE_BASE_URL — "
                "production signer needs tokens from https://pymthouse.com (register/MCP "
                "not deployed there yet); use local signer-dmz with local PymtHouse, or "
                "clear SIGNER_URL so MCP’s signer_url is used."
            )
    if "Connection refused" in message or "failed to reach" in message:
        message = (
            f"Discovery/signer unreachable (is SIGNER_INTERNAL_URL / SIGNER_URL up?): "
            f"{message}"
        )
    return message


def _solid_rgb_frame(width: int, height: int, rgb: tuple[int, int, int]) -> av.VideoFrame:
    frame = av.VideoFrame(width, height, "rgb24")
    r, g, b = rgb
    frame.planes[0].update(bytes([r, g, b]) * (width * height))
    return frame


def _extract_orchestrators_from_mcp(payload: Any) -> list[str]:
    """Best-effort parse of query_orchestrators tool JSON for host:port strings."""
    found: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, str):
            text = node.strip()
            if text and (":" in text or text.startswith("http")):
                # Prefer bare host:port over full URLs when it looks like one.
                if "://" not in text and text.count(":") == 1:
                    found.append(text)
            return
        if isinstance(node, dict):
            for key in (
                "orchestrator",
                "address",
                "serviceUrl",
                "service_url",
                "transcoder",
                "url",
            ):
                value = node.get(key)
                if isinstance(value, str) and value.strip():
                    walk(value.strip())
            for value in node.values():
                walk(value)
            return
        if isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    # Deduplicate preserving order
    seen: set[str] = set()
    out: list[str] = []
    for item in found:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def resolve_orchestrator(
    state: DemoState,
    *,
    explicit: Optional[str] = None,
) -> Optional[str | Sequence[str]]:
    if explicit:
        return explicit
    if state.mcp.orchestrators:
        candidates = _extract_orchestrators_from_mcp(state.mcp.orchestrators)
        if candidates:
            return candidates[0]
    return None


async def run_lv2v_job(
    store: StateStore,
    *,
    model_id: str = "noop",
    orchestrator: Optional[str] = None,
    width: int = 320,
    height: int = 180,
    fps: float = 30.0,
    frame_count: int = 30,
) -> JobState:
    """Submit LV2V noop/write_frames-style job using sdk_token (--token)."""
    state = store.load()
    sdk_token = resolve_sdk_token(state)
    if not sdk_token:
        raise JobRunnerError(
            "No sdk_token available. Run mcp-session (or register) first so a "
            "livepeer_gateway --token payload is stored."
        )

    # Validate token shape early (also surfaces signer_headers presence).
    token_data = parse_token(sdk_token)
    if not token_data.get("signer_headers"):
        raise JobRunnerError(
            "sdk_token is missing signer_headers; remote signer will not receive Authorization"
        )

    orch = resolve_orchestrator(state, explicit=orchestrator)
    job_state = JobState(
        model_id=model_id,
        orchestrator=orch if isinstance(orch, str) else (orch[0] if orch else None),
    )

    job = None
    try:
        job = start_lv2v(
            orch,
            StartJobRequest(model_id=model_id),
            token=sdk_token,
        )
        job_state.publish_url = getattr(job, "publish_url", None)
        job_state.subscribe_url = getattr(job, "subscribe_url", None)
        job_state.control_url = getattr(job, "control_url", None)
        job_state.events_url = getattr(job, "events_url", None)
        job_state.manifest_id = getattr(job, "manifest_id", None)

        media = job.start_media(
            MediaPublishConfig(
                tracks=[VideoOutputConfig(fps=fps)],
            )
        )
        frame_interval = 1.0 / max(1e-6, fps)
        time_base = Fraction(1, int(round(fps)))
        for i in range(max(0, frame_count)):
            color = (i * 5) % 255
            frame = _solid_rgb_frame(width, height, (color, 0, 255 - color))
            frame.pts = i
            frame.time_base = time_base
            await media.write_frame(frame)
            await asyncio.sleep(frame_interval)
    except LivepeerGatewayError as exc:
        message = _format_job_error(exc)
        job_state.error = message
        raise JobRunnerError(message) from exc
    finally:
        if job is not None:
            result = job.close()
            if inspect.isawaitable(result):
                await result
        state = store.load()
        state.job = job_state
        store.save(state)

    return job_state
