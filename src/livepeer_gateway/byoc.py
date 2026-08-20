"""
BYOC (Bring Your Own Capability) job submission for the Livepeer network.

Provides a simple synchronous API to submit inference requests (image generation,
video generation, music, etc.) to a Livepeer BYOC orchestrator.

On-chain usage (with signer for payment tickets):
    from livepeer_gateway.byoc import submit_byoc_job, ByocJobRequest

    result = submit_byoc_job(
        req=ByocJobRequest(capability="recraft-v4", payload={"prompt": "a dragon"}),
        orch_url="https://byoc-orch.daydream.monster:8935",
        signer_url="https://signer.daydream.live",
        signer_headers={"Authorization": "Bearer sk_..."},
    )
    print(result.image_url)

Offchain usage (no payment, for testing):
    result = submit_byoc_job(
        req=ByocJobRequest(capability="nano-banana", payload={"prompt": "a cat"}),
        orch_url="https://localhost:8935",
    )

    # With discovery:
    result = submit_byoc_job(
        discovery_url="https://discovery.example.com",
        req=ByocJobRequest(capability="recraft-v4", payload={"prompt": "sunset"}),
    )
"""

from __future__ import annotations

import base64
import http.client
import json
import logging
import ssl
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .orchestrator import _http_origin, discover_orchestrators
from .errors import LivepeerGatewayError, NoOrchestratorAvailableError, OrchestratorRejection

_LOG = logging.getLogger(__name__)

# Reusable SSL context (skip verification for self-signed certs)
_ssl_ctx = ssl.create_default_context()
_ssl_ctx.check_hostname = False
_ssl_ctx.verify_mode = ssl.CERT_NONE


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ByocJobRequest:
    """A BYOC job request to submit to the network."""

    capability: str
    """Capability name (e.g. 'nano-banana', 'recraft-v4', 'ltx-t2v-23')."""

    payload: dict[str, Any] = field(default_factory=dict)
    """Request body payload (sent as JSON)."""

    timeout_seconds: int = 300
    """Maximum time the orchestrator should wait for the worker response."""

    job_id: Optional[str] = None
    """Optional job ID. Auto-generated if not provided."""

    parameters: Optional[dict[str, Any]] = None
    """Optional job parameters (orchestrator filtering, video ingress/egress)."""


@dataclass
class ByocJobResponse:
    """Response from a BYOC job submission."""

    data: Any
    """Parsed JSON response body from the orchestrator/worker."""

    status_code: int = 200
    """HTTP status code."""

    headers: dict[str, str] = field(default_factory=dict)
    """Response headers (includes Livepeer-Balance, etc.)."""

    orchestrator_url: Optional[str] = None
    """The orchestrator URL that processed this request."""

    raw_body: bytes = b""
    """Raw response body bytes."""

    @property
    def balance(self) -> Optional[str]:
        return self.headers.get("Livepeer-Balance") or self.headers.get("livepeer-balance")

    @property
    def images(self) -> list[dict]:
        """Extract images from response (convenience)."""
        if isinstance(self.data, dict):
            return self.data.get("images", [])
        return []

    @property
    def image_url(self) -> Optional[str]:
        """Extract first image URL from response."""
        for img in self.images:
            if "url" in img:
                return img["url"]
        if isinstance(self.data, dict):
            return self.data.get("image_url") or self.data.get("url")
        return None

    @property
    def video_url(self) -> Optional[str]:
        """Extract video URL from response."""
        if not isinstance(self.data, dict):
            return None
        if "video" in self.data:
            vid = self.data["video"]
            return vid.get("url") if isinstance(vid, dict) else vid
        return self.data.get("video_url") or self.data.get("url")

    @property
    def audio_url(self) -> Optional[str]:
        """Extract audio URL from response."""
        if not isinstance(self.data, dict):
            return None
        if "audio" in self.data:
            aud = self.data["audio"]
            return aud.get("url") if isinstance(aud, dict) else aud
        if "audio_file" in self.data:
            af = self.data["audio_file"]
            return af.get("url") if isinstance(af, dict) else af
        return self.data.get("url")


# ---------------------------------------------------------------------------
# Header building
# ---------------------------------------------------------------------------

def _create_byoc_payment(
    *,
    orch_origin: str,
    capability: str,
    livepeer_hdr: str,
    signer_url: str,
    signer_headers: Optional[dict[str, str]] = None,
    timeout: float = 30.0,
) -> dict[str, str]:
    """
    Create on-chain payment tickets for a BYOC job.

    Flow:
      1. Get OrchestratorInfo via gRPC (same as LV2V) — contains ticket params + price
      2. Generate payment via signer (/generate-live-payment)
      3. Return headers to include in the job request

    Returns dict with Livepeer-Payment and Livepeer-Segment headers.
    """
    from .orch_info import get_orch_info

    # Step 1: Get OrchestratorInfo via gRPC (port 8935)
    # The BYOC orch_origin is on :8936 (HTTP), but gRPC is on :8935.
    # Derive the gRPC URL from the HTTP origin.
    parsed = urlparse(orch_origin)
    grpc_url = f"https://{parsed.hostname}:8935"

    info = get_orch_info(
        grpc_url,
        signer_url=signer_url,
        signer_headers=signer_headers,
    )

    # Check if orch has a price set — if price is 0, skip payment
    if info.HasField("ticket_params"):
        tp = info.ticket_params
        if not tp.face_value or tp.face_value == b'\x00':
            _LOG.info("BYOC orch ticket face_value=0, skipping payment")
            return {}
    else:
        _LOG.info("BYOC orch has no ticket_params, skipping payment")
        return {}

    # Step 2: Generate payment via signer
    orch_info_b64 = base64.b64encode(info.SerializeToString()).decode("ascii")

    signer_origin = _http_origin(signer_url)
    payment_url = f"{signer_origin}/generate-live-payment"
    payment_body = json.dumps({
        "orchestrator": orch_info_b64,
        "type": "lv2v",
        "capability": capability,
    }).encode("utf-8")
    payment_headers = {
        "Content-Type": "application/json",
        "Livepeer-Capability": capability,
    }
    if signer_headers:
        payment_headers.update(signer_headers)

    payment_req = Request(payment_url, data=payment_body, headers=payment_headers, method="POST")
    try:
        with urlopen(payment_req, timeout=timeout) as resp:
            payment_data = json.loads(resp.read())
    except http.client.IncompleteRead as e:
        # The signer advertises a Content-Length but closes the connection
        # early, so urllib raises IncompleteRead and discards the partial
        # body — which usually contains the real signer error
        # (e.g. {"error":{"message":"..."}}). Surface the partial bytes so
        # the underlying signer failure is legible instead of an opaque
        # "IncompleteRead(85 bytes read, 108 more expected)".
        partial = e.partial.decode("utf-8", errors="replace")
        expected = len(e.partial) + e.expected
        raise LivepeerGatewayError(
            f"BYOC payment: signer truncated response "
            f"({len(e.partial)} of {expected} bytes); "
            f"partial body: {partial!r}"
        ) from e
    except HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="replace")[:200]
        except http.client.IncompleteRead as ie:
            # Error responses can be truncated too — keep whatever bytes the
            # signer managed to send rather than losing the message entirely.
            body = ie.partial.decode("utf-8", errors="replace")
        raise LivepeerGatewayError(f"BYOC payment generation failed: HTTP {e.code}: {body}") from e

    result = {}
    if payment_data.get("payment"):
        result["Livepeer-Payment"] = payment_data["payment"]
    if payment_data.get("segCreds"):
        result["Livepeer-Segment"] = payment_data["segCreds"]

    # Distinguish "signer returned empty payment" (bug) from "orch
    # face_value=0" (noop, returned at line 186-190 above as `{}`).
    # If we reached this point, the orch wanted a payment but the signer
    # gave us nothing — raise rather than silently return `{}` so the
    # caller sees a real error.
    if not result:
        raise LivepeerGatewayError(
            "BYOC payment generation: signer returned 200 but empty "
            "payment/segCreds. This is a signer bug or misconfiguration."
        )

    _LOG.info("BYOC payment tickets generated for %s", orch_origin)
    return result


def _sign_byoc_job(
    signer_url: str,
    signer_headers: Optional[dict[str, str]],
    job_id: str,
    capability: str,
    request_json: str,
    parameters_json: str,
    timeout_seconds: int,
) -> dict:
    """Call signer /sign-byoc-job to get sender + signature for the BYOC header."""
    from .orchestrator import _http_origin

    url = f"{_http_origin(signer_url)}/sign-byoc-job"
    payload = {
        "id": job_id,
        "capability": capability,
        "request": request_json,
        "parameters": parameters_json,
        "timeout_seconds": timeout_seconds,
    }
    headers = {"Content-Type": "application/json"}
    if signer_headers:
        headers.update(signer_headers)

    req = Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    try:
        with urlopen(req, timeout=30.0) as resp:
            return json.loads(resp.read())
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:200]
        raise LivepeerGatewayError(f"sign-byoc-job failed: HTTP {e.code}: {body}") from e


def _build_livepeer_header(
    req: ByocJobRequest,
    job_id: str,
    sender: str = "",
    sig: str = "",
) -> str:
    """Build the base64-encoded Livepeer job request header."""
    request_json = json.dumps(req.payload)
    parameters_json = json.dumps(req.parameters) if req.parameters else ""
    job_request = {
        "id": job_id,
        "request": request_json,
        "capability": req.capability,
        "timeout_seconds": req.timeout_seconds,
    }
    if parameters_json:
        job_request["parameters"] = parameters_json
    if sender:
        job_request["sender"] = sender
    if sig:
        job_request["sig"] = sig
    return base64.b64encode(json.dumps(job_request).encode()).decode()


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def submit_byoc_job(
    req: ByocJobRequest,
    *,
    orch_url: Optional[Sequence[str] | str] = None,
    discovery_url: Optional[str] = None,
    signer_url: Optional[str] = None,
    signer_headers: Optional[dict[str, str]] = None,
    discovery_headers: Optional[dict[str, str]] = None,
    timeout: Optional[float] = None,
) -> ByocJobResponse:
    """
    Submit a BYOC job request to the Livepeer network.

    Discovers an orchestrator (or uses explicit orch_url), builds the Livepeer
    header, and POSTs the request to /process/request/{capability}.

    Args:
        req: The job request (capability, payload, timeout).
        orch_url: Direct orchestrator URL(s). Highest priority.
        discovery_url: Discovery endpoint to find orchestrators.
        signer_url: Remote signer URL (also used for discovery fallback).
        signer_headers: Headers for signer requests.
        discovery_headers: Headers for discovery requests.
        timeout: HTTP request timeout in seconds. Defaults to req.timeout_seconds.

    Returns:
        ByocJobResponse with parsed result data.

    Raises:
        NoOrchestratorAvailableError: No orchestrator could process the request.
        LivepeerGatewayError: Network or protocol error.
    """
    job_id = req.job_id or str(uuid.uuid4())
    http_timeout = timeout or req.timeout_seconds

    # Discover orchestrators
    orch_list = _resolve_orchestrators(
        orch_url=orch_url,
        discovery_url=discovery_url,
        signer_url=signer_url,
        signer_headers=signer_headers,
        discovery_headers=discovery_headers,
    )

    _LOG.info("BYOC job %s: capability=%s, orchestrators=%s", job_id, req.capability, orch_list)

    # Sign the job request if signer is available (on-chain)
    sender = ""
    sig = ""
    if signer_url:
        try:
            request_json = json.dumps(req.payload)
            parameters_json = json.dumps(req.parameters) if req.parameters else ""
            sign_resp = _sign_byoc_job(
                signer_url=signer_url,
                signer_headers=signer_headers,
                job_id=job_id,
                capability=req.capability,
                request_json=request_json,
                parameters_json=parameters_json,
                timeout_seconds=req.timeout_seconds,
            )
            sender = sign_resp.get("sender", "")
            sig = sign_resp.get("signature", "")
            _LOG.info("BYOC job %s: signed by sender=%s", job_id, sender[:12] + "..." if sender else "none")
        except Exception as e:
            _LOG.warning("BYOC job %s: signing failed: %s", job_id, e)

    # Build headers
    livepeer_hdr = _build_livepeer_header(req, job_id, sender=sender, sig=sig)
    body = json.dumps(req.payload).encode("utf-8")

    # Try each orchestrator
    rejections: list[OrchestratorRejection] = []

    for orch in orch_list:
        orch_origin = _http_origin(orch)
        url = f"{orch_origin}/process/request/{req.capability}"

        headers = {
            "Content-Type": "application/json",
            "Livepeer": livepeer_hdr,
            "Livepeer-Capability": req.capability,
        }

        # On-chain payment: get token from orch, create payment via signer
        if signer_url:
            try:
                payment_headers = _create_byoc_payment(
                    orch_origin=orch_origin,
                    capability=req.capability,
                    livepeer_hdr=livepeer_hdr,
                    signer_url=signer_url,
                    signer_headers=signer_headers,
                    timeout=http_timeout,
                )
                headers.update(payment_headers)
                _LOG.info("BYOC job %s: payment tickets created for %s", job_id, orch_origin)
            except Exception as e:
                _LOG.warning("BYOC job %s: payment creation failed for %s: %s", job_id, orch_origin, e)
                rejections.append(OrchestratorRejection(url=orch_origin, reason=f"payment failed: {e}"))
                continue

        http_req = Request(url, data=body, headers=headers, method="POST")

        _LOG.info("BYOC job %s: trying orchestrator %s", job_id, orch_origin)

        try:
            with urlopen(http_req, timeout=http_timeout, context=_ssl_ctx) as resp:
                raw_body = resp.read()
                resp_headers = {k: v for k, v in resp.headers.items()}

                try:
                    data = json.loads(raw_body.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    data = raw_body

                return ByocJobResponse(
                    data=data,
                    status_code=resp.status,
                    headers=resp_headers,
                    orchestrator_url=orch_origin,
                    raw_body=raw_body,
                )

        except HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                pass
            reason = f"HTTP {e.code}: {err_body}"
            _LOG.warning("BYOC job %s: orchestrator %s rejected: %s", job_id, orch_origin, reason)

            # Non-retryable (4xx except 408/429)
            if 400 <= e.code < 500 and e.code not in (408, 429):
                raise LivepeerGatewayError(
                    f"BYOC job rejected by orchestrator {orch_origin}: {reason}"
                ) from e

            rejections.append(OrchestratorRejection(url=orch_origin, reason=reason))

        except (URLError, ConnectionRefusedError, TimeoutError, OSError) as e:
            reason = f"{type(e).__name__}: {e}"
            _LOG.warning("BYOC job %s: orchestrator %s unreachable: %s", job_id, orch_origin, reason)
            rejections.append(OrchestratorRejection(url=orch_origin, reason=reason))

    reasons = "; ".join(r.reason for r in rejections) if rejections else "no orchestrators configured"
    raise NoOrchestratorAvailableError(
        f"No orchestrator available for capability '{req.capability}': {reasons}",
        rejections=rejections,
    )


def list_capabilities(
    adapter_url: str,
    *,
    timeout: float = 10.0,
) -> list[dict[str, Any]]:
    """
    List capabilities registered on an adapter.

    Args:
        adapter_url: Base URL of the inference adapter (e.g. http://34.134.195.88:9090).
        timeout: HTTP timeout.

    Returns:
        List of capability dicts with 'name', 'model_id', 'capacity' keys.
    """
    url = f"{adapter_url.rstrip('/')}/capabilities"
    http_req = Request(url, headers={"Accept": "application/json"})

    try:
        with urlopen(http_req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("capabilities", [])
    except Exception as e:
        _LOG.warning("Failed to list capabilities from %s: %s", adapter_url, e)
        raise LivepeerGatewayError(f"Failed to list capabilities: {e}") from e


# ---------------------------------------------------------------------------
# Training API
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ByocTrainingRequest:
    """A BYOC training job request."""

    capability: str
    """Capability name for the training job."""

    model_id: str
    """fal.ai model ID for training (e.g. 'fal-ai/flux-lora-fast-training')."""

    params: dict[str, Any] = field(default_factory=dict)
    """Training parameters (images_data_url, trigger_word, steps, etc.)."""

    timeout_seconds: int = 300
    """Timeout for the initial submit request (not the training itself)."""

    callback_url: Optional[str] = None
    """Optional webhook URL for completion notification."""


@dataclass
class ByocTrainingResponse:
    """Response from a BYOC training job submission."""

    job_id: str
    """Unique job ID for status polling."""

    status: str = "submitted"
    """Current status: submitted, running, completed, failed, cancelled."""

    orchestrator_url: Optional[str] = None
    """The orchestrator handling this job."""

    status_url: Optional[str] = None
    """Full URL to poll for status."""

    data: Optional[dict] = None
    """Raw response data."""

    @property
    def is_done(self) -> bool:
        return self.status in ("completed", "failed", "cancelled")


@dataclass
class ByocTrainingStatus:
    """Status of a training job."""

    job_id: str
    status: str
    progress: int = 0
    result: Optional[dict] = None
    error: Optional[str] = None
    model_id: Optional[str] = None
    created_at: Optional[float] = None
    updated_at: Optional[float] = None
    cost: Optional[str] = None
    """Total cost charged so far (wei)."""
    balance: Optional[str] = None
    """Remaining sender balance (wei)."""

    @property
    def is_done(self) -> bool:
        return self.status in ("completed", "failed", "cancelled")

    @property
    def lora_url(self) -> Optional[str]:
        """Extract LoRA weights URL from completed result."""
        if not self.result:
            return None
        # fal.ai returns diffusers_lora_file.url
        lora_file = self.result.get("diffusers_lora_file")
        if isinstance(lora_file, dict):
            return lora_file.get("url")
        return self.result.get("lora_url")

    @property
    def config_url(self) -> Optional[str]:
        """Extract config file URL from completed result."""
        if not self.result:
            return None
        config_file = self.result.get("config_file")
        if isinstance(config_file, dict):
            return config_file.get("url")
        return None


def submit_training_job(
    req: ByocTrainingRequest,
    *,
    orch_url: Optional[Sequence[str] | str] = None,
    discovery_url: Optional[str] = None,
    signer_url: Optional[str] = None,
    signer_headers: Optional[dict[str, str]] = None,
    discovery_headers: Optional[dict[str, str]] = None,
    timeout: Optional[float] = None,
) -> ByocTrainingResponse:
    """
    Submit an async training job to the Livepeer BYOC network.

    Returns immediately with a job_id that can be polled for status.

    Args:
        req: Training request (capability, model_id, params).
        orch_url: Direct orchestrator URL(s).
        discovery_url: Discovery endpoint.
        timeout: HTTP timeout for the submit request.

    Returns:
        ByocTrainingResponse with job_id and status_url.
    """
    job_id = str(uuid.uuid4())
    http_timeout = timeout or req.timeout_seconds

    orch_list = _resolve_orchestrators(
        orch_url=orch_url,
        discovery_url=discovery_url,
        signer_url=signer_url,
        signer_headers=signer_headers,
        discovery_headers=discovery_headers,
    )

    # Build the Livepeer header (reuse existing infrastructure)
    byoc_req = ByocJobRequest(
        capability=req.capability,
        payload={"model_id": req.model_id, **req.params},
        timeout_seconds=req.timeout_seconds,
        job_id=job_id,
    )

    # Sign the job request if signer is available — mirror submit_byoc_job.
    # The orch's training handler runs setupOrchJob → verifyJobCreds (same
    # code path as inference) and rejects with HTTP 400 "Could not verify
    # job creds" if Livepeer-Job-Request / -Token are missing. The original
    # unsigned path worked only against an older orch that lacked the
    # /process/train/ route; the v2-with-training merge brings that route
    # online and demands signed creds.
    sender = ""
    sig = ""
    if signer_url:
        try:
            request_json = json.dumps(byoc_req.payload)
            parameters_json = json.dumps(byoc_req.parameters) if byoc_req.parameters else ""
            sign_resp = _sign_byoc_job(
                signer_url=signer_url,
                signer_headers=signer_headers,
                job_id=job_id,
                capability=req.capability,
                request_json=request_json,
                parameters_json=parameters_json,
                timeout_seconds=req.timeout_seconds,
            )
            sender = sign_resp.get("sender", "")
            sig = sign_resp.get("signature", "")
            _LOG.info("Training job %s: signed by sender=%s", job_id,
                      sender[:12] + "..." if sender else "none")
        except Exception as e:
            _LOG.warning("Training job %s: signing failed: %s", job_id, e)

    livepeer_hdr = _build_livepeer_header(byoc_req, job_id, sender=sender, sig=sig)

    # Build training body
    body = json.dumps({
        "model_id": req.model_id,
        "params": req.params,
        **({"callback_url": req.callback_url} if req.callback_url else {}),
    }).encode("utf-8")

    rejections: list[OrchestratorRejection] = []

    for orch in orch_list:
        orch_origin = _http_origin(orch)
        url = f"{orch_origin}/process/train/{req.capability}"

        headers = {
            "Content-Type": "application/json",
            "Livepeer": livepeer_hdr,
            "Livepeer-Capability": req.capability,
        }

        # On-chain payment ticket — same flow as inference. Required for
        # the orch's verifyJobCreds + per-second metering to succeed.
        # On staging the capability price is 0 so deduction is a no-op,
        # but the orch still validates the ticket structure.
        if signer_url:
            try:
                payment_headers = _create_byoc_payment(
                    orch_origin=orch_origin,
                    capability=req.capability,
                    livepeer_hdr=livepeer_hdr,
                    signer_url=signer_url,
                    signer_headers=signer_headers,
                    timeout=http_timeout,
                )
                headers.update(payment_headers)
                _LOG.info("Training job %s: payment tickets created for %s",
                          job_id, orch_origin)
            except Exception as e:
                _LOG.warning("Training job %s: payment creation failed for %s: %s",
                             job_id, orch_origin, e)
                rejections.append(OrchestratorRejection(
                    url=orch_origin, reason=f"payment failed: {e}",
                ))
                continue

        http_req = Request(url, data=body, headers=headers, method="POST")
        _LOG.info("Training job %s: trying orchestrator %s", job_id, orch_origin)

        try:
            with urlopen(http_req, timeout=http_timeout, context=_ssl_ctx) as resp:
                raw_body = resp.read()
                data = json.loads(raw_body.decode("utf-8"))

                return ByocTrainingResponse(
                    job_id=data.get("job_id", job_id),
                    status=data.get("status", "submitted"),
                    orchestrator_url=orch_origin,
                    status_url=data.get("status_url"),
                    data=data,
                )

        except HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                pass
            reason = f"HTTP {e.code}: {err_body}"
            _LOG.warning("Training job %s: orchestrator %s rejected: %s", job_id, orch_origin, reason)

            if 400 <= e.code < 500 and e.code not in (408, 429):
                raise LivepeerGatewayError(
                    f"Training job rejected by orchestrator {orch_origin}: {reason}"
                ) from e

            rejections.append(OrchestratorRejection(url=orch_origin, reason=reason))

        except (URLError, ConnectionRefusedError, TimeoutError, OSError) as e:
            reason = f"{type(e).__name__}: {e}"
            _LOG.warning("Training job %s: orchestrator %s unreachable: %s", job_id, orch_origin, reason)
            rejections.append(OrchestratorRejection(url=orch_origin, reason=reason))

    reasons = "; ".join(r.reason for r in rejections) if rejections else "no orchestrators configured"
    raise NoOrchestratorAvailableError(
        f"No orchestrator available for capability '{req.capability}': {reasons}",
        rejections=rejections,
    )


def refresh_training_payment(
    job_id: str,
    orch_url: str,
    capability: str,
    *,
    signer_url: str,
    signer_headers: Optional[dict[str, str]] = None,
    timeout: float = 30.0,
    max_attempts: int = 3,
) -> dict[str, str]:
    """
    Top up the orch's deposit ledger for an in-flight async training job.

    Called by the SDK's status-poll loop when the orch-reported balance
    approaches zero (refresh-on-watermark per design §3.A). Generates a
    fresh ticket batch from the same wallet that signed the submit, then
    POSTs it to the orch at /process/job/{job_id}/refresh-payment.

    Invariants per §10.1:
    - I5 (no double-charge on retry): orch idempotency key is
      (job_id, ticket_nonce). The signer's /generate-live-payment
      includes nonce in the payment payload; orch deduplicates.
    - I6 (sender attribution): the refresh ticket is signed by the SAME
      wallet as the submit (signer_url resolves bearer → wallet
      deterministically; bearer is the same for the same SDK session).

    Args:
        job_id: training job_id assigned by orch on submit.
        orch_url: orchestrator URL accepting the job.
        capability: capability name (needed by signer to pick correct
            ticket_params).
        signer_url: remote signer with /generate-live-payment.
        signer_headers: pass-through headers (notably Authorization
            Bearer of the user).
        timeout: per-request timeout in seconds.
        max_attempts: retry count for signer/orch transient errors.
            Default 3 with simple linear backoff (no exponential — keeps
            refresh latency bounded under SDK's watermark budget).

    Returns:
        dict with at least {"credited_wei", "new_balance_wei"} fields
        returned by the orch. Subset is reported to the SDK caller.

    Raises:
        LivepeerGatewayError on permanent failure after max_attempts.
    """
    # Mint the ticket ONCE, outside the retry loop. Re-minting on each
    # retry would burn N distinct nonces for a single refresh attempt,
    # and if the orch already credited the first ticket but the response
    # was lost on the network, the second mint would double-credit.
    # Reviewer note (I1): per-job idempotency at the PM layer keys on
    # ticket nonce; identical headers credit once, distinct nonces credit
    # separately. The fix is to never produce distinct nonces for a
    # single logical refresh.
    try:
        payment_headers = _create_byoc_payment(
            orch_origin=_http_origin(orch_url),
            capability=capability,
            livepeer_hdr="",  # not used by refresh path
            signer_url=signer_url,
            signer_headers=signer_headers,
            timeout=timeout,
        )
    except LivepeerGatewayError as e:
        # _create_byoc_payment raises on signer bug; surface as a refresh
        # error rather than retrying (the signer state is what's wrong,
        # not a network blip).
        raise LivepeerGatewayError(
            f"Training refresh {job_id}: payment generation failed: {e}"
        ) from e

    if not payment_headers.get("Livepeer-Payment"):
        # The orch's `ticket_params.face_value` is zero — the only way
        # _create_byoc_payment returns an empty result without raising
        # (it raises on signer-empty per the C1 fix). This means refresh
        # is a no-op for this cap on this orch.
        _LOG.info("Training refresh %s: orch face_value=0, noop", job_id)
        return {"credited_wei": "0", "new_balance_wei": "n/a", "noop": "true"}

    # Retry only the orch POST. The payment headers above are pinned to
    # one nonce; identical headers on every attempt → idempotent credit.
    url = f"{_http_origin(orch_url)}/process/job/{job_id}/refresh-payment"
    headers = {
        "Content-Type": "application/json",
        **payment_headers,  # Livepeer-Payment + Livepeer-Segment
    }

    last_err: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        # Empty JSON body — refresh carries everything in headers
        http_req = Request(url, data=b"{}", headers=headers, method="POST")
        try:
            with urlopen(http_req, timeout=timeout, context=_ssl_ctx) as resp:
                body = resp.read().decode("utf-8")
                if resp.status not in (200, 202):
                    raise LivepeerGatewayError(
                        f"Refresh rejected: HTTP {resp.status}: {body[:200]}"
                    )
                try:
                    return json.loads(body) or {"credited_wei": "unknown"}
                except json.JSONDecodeError:
                    # Older orch may return empty body; treat as success.
                    return {"credited_wei": "unknown", "raw": body[:200]}

        except (HTTPError, URLError, OSError) as e:
            last_err = e
            # HTTP 4xx (other than 408/429) are not transient — fail fast
            if isinstance(e, HTTPError) and e.code not in (408, 429, 502, 503, 504):
                err_body = ""
                try:
                    err_body = e.read().decode("utf-8", errors="replace")[:200]
                except Exception:
                    pass
                raise LivepeerGatewayError(
                    f"Training refresh permanent failure for {job_id}: "
                    f"HTTP {e.code}: {err_body}"
                ) from e

            _LOG.warning(
                "Training refresh %s attempt %d/%d failed (%s); retrying",
                job_id, attempt, max_attempts, type(e).__name__,
            )
            if attempt < max_attempts:
                import time
                time.sleep(0.5 * attempt)  # linear backoff: 0.5s, 1.0s

    raise LivepeerGatewayError(
        f"Training refresh exhausted {max_attempts} attempts for {job_id}: {last_err}"
    )


def get_training_status(
    job_id: str,
    orch_url: str,
    *,
    timeout: float = 10.0,
) -> ByocTrainingStatus:
    """
    Poll training job status from the orchestrator.

    Args:
        job_id: The training job ID returned by submit_training_job.
        orch_url: The orchestrator URL that accepted the job.
        timeout: HTTP request timeout.

    Returns:
        ByocTrainingStatus with current status, progress, and result.
    """
    orch_origin = _http_origin(orch_url)
    url = f"{orch_origin}/process/job/{job_id}"
    http_req = Request(url, headers={"Accept": "application/json"})

    try:
        with urlopen(http_req, timeout=timeout, context=_ssl_ctx) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return ByocTrainingStatus(
                job_id=data.get("job_id", job_id),
                status=data.get("status", "unknown"),
                progress=data.get("progress", 0),
                result=data.get("result"),
                error=data.get("error"),
                model_id=data.get("model_id"),
                created_at=data.get("created_at"),
                updated_at=data.get("updated_at"),
                cost=data.get("cost"),
                balance=data.get("balance"),
            )
    except HTTPError as e:
        if e.code == 404:
            raise LivepeerGatewayError(f"Training job {job_id} not found") from e
        raise LivepeerGatewayError(f"Status check failed: HTTP {e.code}") from e
    except Exception as e:
        raise LivepeerGatewayError(f"Status check failed: {e}") from e


def wait_for_training(
    job_id: str,
    orch_url: str,
    *,
    poll_interval: float = 5.0,
    timeout: float = 28800.0,
) -> ByocTrainingStatus:
    """
    Poll until a training job completes.

    Args:
        job_id: The training job ID.
        orch_url: The orchestrator URL.
        poll_interval: Seconds between polls.
        timeout: Maximum wait time in seconds.

    Returns:
        Final ByocTrainingStatus.
    """
    import time

    elapsed = 0.0
    while elapsed < timeout:
        status = get_training_status(job_id, orch_url)
        if status.is_done:
            return status
        _LOG.info("Training job %s: status=%s progress=%d%% elapsed=%.0fs",
                  job_id, status.status, status.progress, elapsed)
        time.sleep(poll_interval)
        elapsed += poll_interval

    return get_training_status(job_id, orch_url)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_orchestrators(
    *,
    orch_url: Optional[Sequence[str] | str] = None,
    discovery_url: Optional[str] = None,
    signer_url: Optional[str] = None,
    signer_headers: Optional[dict[str, str]] = None,
    discovery_headers: Optional[dict[str, str]] = None,
) -> list[str]:
    """Resolve orchestrator list from various sources."""
    # Direct orchestrator URL(s)
    if orch_url is not None:
        if isinstance(orch_url, str):
            urls = [u.strip() for u in orch_url.split(",") if u.strip()]
        else:
            urls = [u.strip() for u in orch_url if isinstance(u, str) and u.strip()]
        if urls:
            return urls

    # Use discovery
    if discovery_url or signer_url:
        return discover_orchestrators(
            discovery_url=discovery_url,
            signer_url=signer_url,
            signer_headers=signer_headers,
            discovery_headers=discovery_headers,
        )

    raise LivepeerGatewayError(
        "submit_byoc_job requires orch_url, discovery_url, or signer_url"
    )
