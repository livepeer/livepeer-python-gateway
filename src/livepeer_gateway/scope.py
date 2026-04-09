from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

from .capabilities import CapabilityId, build_capabilities
from .control import ControlConfig, ControlMode
from .errors import LivepeerGatewayError, NoOrchestratorAvailableError, OrchestratorRejection
from .lv2v import LiveVideoToVideo, StartJobRequest
from .orchestrator import _http_origin, post_json
from .remote_signer import PaymentSession
from .selection import orchestrator_selector
from .token import parse_token

_LOG = logging.getLogger(__name__)


def start_scope(
    orch_url: Optional[Sequence[str] | str],
    req: StartJobRequest,
    *,
    start_payments: bool = True,
    token: Optional[str] = None,
    signer_url: Optional[str] = None,
    signer_headers: Optional[dict[str, str]] = None,
    discovery_url: Optional[str] = None,
    discovery_headers: Optional[dict[str, str]] = None,
    control_config: Optional[ControlConfig] = None,
    use_tofu: bool = True,
    timeout: float = 5.0,
) -> LiveVideoToVideo:
    """
    Start a scope job.

    Selects an orchestrator with Scope capability and calls
    POST {info.transcoder}/scope with JSON body.

    If ``start_payments`` is true and the call happens within a running
    asyncio event loop, a background task is automatically started to
    send per-segment payments. Otherwise a warning is logged and
    payments can be started later via ``job.start_payment_sender()``.

    Optional ``token`` can be provided as a base64-encoded JSON object.
    Token values take precedence over explicit keyword arguments.
    Explicit keyword arguments are used only for fields missing in the token.

    Orchestrator selection/discovery precedence (highest -> lowest):
    1) token ``orchestrators`` value
    2) explicit ``orch_url`` list
    3) token ``discovery`` value
    4) explicit ``discovery_url`` argument
    5) remote signer discovery endpoint derived from the resolved signer URL

    ``timeout`` controls only the initial HTTP POST to
    ``/scope`` after an orchestrator has been selected.
    Discovery and ``GetOrchestrator`` calls use their own timeouts.

    ``use_tofu`` controls TLS mode for ``GetOrchestrator``:
    - True: trust-on-first-use certificate pinning
    - False: default gRPC/system CA roots

    ``control_config`` controls control-channel behavior. Use
    ``ControlConfig(mode=ControlMode.DISABLED)`` to disable keepalives.

    ``model_id`` is ignored for now; internally this is hard-coded to "scope".
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

    capabilities = build_capabilities(CapabilityId.LIVE_VIDEO_TO_VIDEO, "scope")
    # Orchestrator discovery precedence after token-first field resolution:
    # token orchestrators -> explicit orch_url -> token discovery ->
    # explicit discovery_url -> signer_url
    cursor = orchestrator_selector(
        resolved_orch_url,
        signer_url=resolved_signer_url,
        signer_headers=resolved_signer_headers,
        discovery_url=resolved_discovery_url,
        discovery_headers=resolved_discovery_headers,
        capabilities=capabilities,
        use_tofu=use_tofu,
    )

    start_rejections: list[OrchestratorRejection] = []
    while True:
        try:
            selected_url, info = cursor.next()
        except NoOrchestratorAvailableError as e:
            all_rejections = list(e.rejections) + start_rejections
            if all_rejections:
                raise NoOrchestratorAvailableError(
                    f"All orchestrators failed ({len(all_rejections)} tried)",
                    rejections=all_rejections,
                ) from None
            raise

        try:
            session = PaymentSession(
                resolved_signer_url,
                info,
                signer_headers=resolved_signer_headers,
                type="lv2v",
                capabilities=capabilities,
                use_tofu=use_tofu,
            )
            p = session.get_payment()
            headers: dict[str, str] = {
                "Livepeer-Payment": p.payment,
                "Livepeer-Segment": p.seg_creds,
            }

            base = _http_origin(info.transcoder)
            url = f"{base}/scope"
            payload = req.to_json()
            payload.setdefault("model_id", "scope")
            data = post_json(url, payload, headers=headers, timeout=timeout)
            job = LiveVideoToVideo.from_json(
                data,
                signer_url=resolved_signer_url,
                orchestrator_info=info,
                payment_session=session,
            )
            if not job.manifest_id:
                raise LivepeerGatewayError("LiveVideoToVideo response missing manifest_id")
            session.set_manifest_id(job.manifest_id)
            return job
        except LivepeerGatewayError as e:
            _LOG.debug(
                "start_scope candidate failed, trying fallback if available: %s (%s)",
                selected_url,
                str(e),
            )
            start_rejections.append(OrchestratorRejection(url=selected_url, reason=str(e)))
