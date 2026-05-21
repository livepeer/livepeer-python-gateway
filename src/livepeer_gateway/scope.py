from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

from .errors import LivepeerGatewayError, NoRunnerAvailableError
from .http import post_json
from .lv2v import LiveVideoToVideo, StartJobRequest
from .selection import runner_selector
from .token import parse_token

_SCOPE_RUNNER_APP = "live-video-to-video/scope"
_LOG = logging.getLogger(__name__)


async def start_scope(
    orch_url: Optional[Sequence[str] | str],
    req: StartJobRequest,
    *,
    token: Optional[str] = None,
    signer_url: Optional[str] = None,
    signer_headers: Optional[dict[str, str]] = None,
    discovery_url: Optional[str] = None,
    discovery_headers: Optional[dict[str, str]] = None,
    timeout: float = 5.0,
) -> LiveVideoToVideo:
    """
    Start a Scope job through a live runner.

    Scope is treated as a single-shot live runner app. The request body is sent
    to a discovered ``live-video-to-video/scope`` runner and any paid runner
    challenge is handled by the live-runner payment flow.

    Optional ``token`` can be provided as a base64-encoded JSON object.
    Token values take precedence over explicit keyword arguments.
    Explicit keyword arguments are used only for fields missing in the token.

    Runner discovery precedence (highest -> lowest):
    1) token ``orchestrators`` value, converted by appending ``/discovery``
    2) explicit ``orch_url`` value, converted by appending ``/discovery``
    3) token ``discovery`` value
    4) explicit ``discovery_url`` argument
    5) remote signer discovery endpoint derived from the resolved signer URL

    """
    token_data: Optional[dict[str, Any]] = None
    if token is not None:
        token_data = parse_token(token)

    resolved_orch_url = token_data.get("orchestrators") if token_data else None
    if resolved_orch_url is None:
        resolved_orch_url = orch_url

    resolved_signer_url = token_data.get("signer") if token_data else None
    if resolved_signer_url is None:
        resolved_signer_url = signer_url

    resolved_signer_headers = token_data.get("signer_headers") if token_data else None
    if resolved_signer_headers is None:
        resolved_signer_headers = signer_headers

    resolved_discovery_url = token_data.get("discovery") if token_data else None
    if resolved_discovery_url is None:
        resolved_discovery_url = discovery_url

    resolved_discovery_headers = token_data.get("discovery_headers") if token_data else None
    if resolved_discovery_headers is None:
        resolved_discovery_headers = discovery_headers

    body = req.to_json()
    result = await _select_scope_runner(
        body=body,
        signer_url=resolved_signer_url,
        signer_headers=resolved_signer_headers,
        discovery_url=resolved_discovery_url,
        discovery_headers=resolved_discovery_headers,
        orch_url=resolved_orch_url,
        timeout=timeout,
    )

    data = result.data
    if not _is_serverless_runner(result.runner):
        app_url = data.get("app_url")
        if not isinstance(app_url, str) or not app_url.strip():
            raise LivepeerGatewayError("Scope runner response missing app_url")
        data = await post_json(f"{app_url.strip().rstrip('/')}/scope", body, timeout=timeout)

    job = LiveVideoToVideo.from_json(
        data,
        signer_url=resolved_signer_url,
        payment_session=result.payment_session,
    )
    if not job.manifest_id:
        raise LivepeerGatewayError("Scope response missing manifest_id")
    return job


async def _select_scope_runner(
    *,
    body: dict[str, Any],
    signer_url: Optional[str],
    signer_headers: Optional[dict[str, str]],
    discovery_url: Optional[str],
    discovery_headers: Optional[dict[str, str]],
    orch_url: Optional[Sequence[str] | str],
    timeout: float,
):
    cursor = await runner_selector(
        body=body,
        signer_url=signer_url,
        signer_headers=signer_headers,
        orchestrators=orch_url,
        discovery_url=discovery_url,
        discovery_headers=discovery_headers,
        app=_SCOPE_RUNNER_APP,
        timeout=timeout,
    )
    try:
        return await cursor.next()
    except NoRunnerAvailableError as e:
        for rejection in e.rejections:
            _LOG.info("scope runner rejected: %s: %s", rejection.url, rejection.reason)
        raise


def _is_serverless_runner(runner: object) -> bool:
    raw = getattr(runner, "raw", None)
    version = raw.get("version") if isinstance(raw, dict) else None
    return isinstance(version, str) and version.startswith("serverless")
