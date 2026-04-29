from __future__ import annotations

import asyncio
import base64
import json
import logging
import numbers
import ssl
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin, urlparse, urlunparse
from urllib.request import Request, urlopen

from .byoc_payments import BYOCPaymentSession
from .capabilities import CapabilityId, build_capabilities
from .control import Control
from .errors import (
    LivepeerGatewayError,
    NoOrchestratorAvailableError,
    OrchestratorRejection,
    PaymentError,
    PaymentRequiredError,
    SkipPaymentCycle,
)
from .events import Events
from .media_output import LagPolicy, MediaOutput
from .media_publish import MediaPublish, MediaPublishConfig
from .orch_info import get_orch_info as _get_orch_info
from .orchestrator import _extract_error_message, resolve_transcoder_http_url
from .selection import orchestrator_selector
from .sse import SSEClient

_LOG = logging.getLogger(__name__)


def _header_get(headers: dict[str, str], key: str) -> Optional[str]:
    key_lower = key.lower()
    for name, value in headers.items():
        if name.lower() == key_lower:
            return value
    return None


def _field_value(obj: Any, snake: str, camel: str) -> Any:
    if isinstance(obj, dict):
        if snake in obj:
            return obj[snake]
        if camel in obj:
            return obj[camel]
        return None
    v = getattr(obj, snake, None)
    if v is not None:
        return v
    return getattr(obj, camel, None)


def _nonzero_real_scalar(val: Any) -> bool:
    if val is None or isinstance(val, bool):
        return False
    if isinstance(val, bytes):
        if not val:
            return False
        return int.from_bytes(val, "big") > 0
    if isinstance(val, numbers.Real):
        return val != 0
    return False


def _orch_info_ticket_params_usable(info: Any) -> bool:
    """True when ticket_params has non-zero face_value and win_prob."""
    params = getattr(info, "ticket_params", None)
    if params is None:
        return False
    face = _field_value(params, "face_value", "faceValue")
    win = _field_value(params, "win_prob", "winProb")
    return _nonzero_real_scalar(face) and _nonzero_real_scalar(win)


def _price_info_matches_byoc(price_info: Any, capability_name: str) -> bool:
    if price_info is None:
        return False
    capability = _field_value(price_info, "capability", "capability")
    constraint = _field_value(price_info, "constraint", "constraint")
    price_per_unit = _field_value(price_info, "price_per_unit", "pricePerUnit")
    pixels_per_unit = _field_value(price_info, "pixels_per_unit", "pixelsPerUnit")
    return (
        capability == int(CapabilityId.BYOC)
        and constraint == capability_name
        and _nonzero_real_scalar(price_per_unit)
        and _nonzero_real_scalar(pixels_per_unit)
    )


def _orch_info_has_byoc_price(info: Any, capability_name: str) -> bool:
    top_price = _field_value(info, "price_info", "priceInfo")
    if _price_info_matches_byoc(top_price, capability_name):
        return True

    caps_prices = _field_value(info, "capabilities_prices", "capabilitiesPrices")
    if caps_prices is None:
        return False
    for price_info in caps_prices:
        if _price_info_matches_byoc(price_info, capability_name):
            return True
    return False


def _orch_info_supports_byoc_payment(info: Any, capability_name: str) -> bool:
    return _orch_info_ticket_params_usable(info) and _orch_info_has_byoc_price(info, capability_name)


def _get_payment_orch_info(
    orch_url: str,
    *,
    signer_url: Optional[str],
    signer_headers: Optional[dict[str, str]],
    capabilities: Any,
    capability_name: str,
    use_tofu: bool = True,
) -> tuple[Any, Any]:
    """
    Fetch OrchestratorInfo for BYOC payment preflight. If the capability-scoped
    request comes back without usable BYOC pricing, retry without capability
    filtering (some legacy orchestrators only advertise BYOC pricing on the
    unfiltered path).
    """
    payment_info = _get_orch_info(
        orch_url,
        signer_url=signer_url,
        signer_headers=signer_headers,
        capabilities=capabilities,
        use_tofu=use_tofu,
    )
    if _orch_info_supports_byoc_payment(payment_info, capability_name):
        return payment_info, capabilities

    legacy_info = _get_orch_info(
        orch_url,
        signer_url=signer_url,
        signer_headers=signer_headers,
        use_tofu=use_tofu,
    )
    if _orch_info_supports_byoc_payment(legacy_info, capability_name):
        _LOG.debug(
            "BYOC payment preflight: using legacy orch info response for %s",
            capability_name,
        )
        return legacy_info, None

    return payment_info, capabilities


@dataclass(frozen=True)
class BYOCJobRequest:
    capability: str
    request_id: Optional[str] = None
    stream_id: Optional[str] = None
    request: Optional[dict[str, Any]] = None
    parameters: Optional[dict[str, Any]] = None
    body: Optional[dict[str, Any]] = None
    timeout_seconds: int = 30
    enable_video_ingress: bool = True
    enable_video_egress: bool = True
    enable_data_output: bool = False
    # Orchestrator default in go-livepeer is POST {transcoder}/ai/stream/start;
    # the gateway uses /process/stream/start. Either a path on the selected
    # transcoder origin or a full http(s) URL.
    stream_start_endpoint: str = "/ai/stream/start"
    stream_payment_endpoint: str = "/ai/stream/payment"

    def _job_id(self) -> str:
        if self.request_id and self.request_id.strip():
            return self.request_id.strip()
        if self.stream_id and self.stream_id.strip():
            return self.stream_id.strip()
        return uuid.uuid4().hex

    def _request_json(self, job_id: str) -> str:
        payload: dict[str, Any] = {}
        if self.request:
            payload.update(self.request)
        payload.setdefault("stream_id", self.stream_id or job_id)
        return json.dumps(payload, separators=(",", ":"))

    def _parameters_json(self) -> str:
        payload: dict[str, Any] = {}
        if self.parameters:
            payload.update(self.parameters)
        payload.setdefault("enable_video_ingress", self.enable_video_ingress)
        payload.setdefault("enable_video_egress", self.enable_video_egress)
        payload.setdefault("enable_data_output", self.enable_data_output)
        return json.dumps(payload, separators=(",", ":"))

    def _body(self, job_id: str) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.body:
            payload.update(self.body)
        payload.setdefault("stream_id", self.stream_id or job_id)
        return payload


@dataclass(frozen=True)
class BYOCProcessRequest:
    capability: str
    route: str = "predict"
    request_id: Optional[str] = None
    request: Optional[dict[str, Any]] = None
    parameters: Optional[dict[str, Any]] = None
    body: Optional[dict[str, Any]] = None
    timeout_seconds: int = 30
    request_endpoint: str = "/process/request"
    stream_payment_endpoint: str = "/ai/stream/payment"

    def _job_id(self) -> str:
        if self.request_id and self.request_id.strip():
            return self.request_id.strip()
        return uuid.uuid4().hex

    def _request_json(self) -> str:
        payload: dict[str, Any] = {}
        if self.request:
            payload.update(self.request)
        return json.dumps(payload, separators=(",", ":"))

    def _parameters_json(self) -> str:
        payload: dict[str, Any] = {}
        if self.parameters:
            payload.update(self.parameters)
        return json.dumps(payload, separators=(",", ":"))

    def _body(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.body:
            payload.update(self.body)
        return payload


@dataclass(frozen=True)
class BYOCProcessResponse:
    status_code: int
    headers: dict[str, str]
    body: Any
    job_id: str
    capability: str
    orchestrator_url: str


@dataclass(frozen=True)
class BYOCProcessStream:
    status_code: int
    headers: dict[str, str]
    events: SSEClient
    job_id: str
    capability: str
    orchestrator_url: str


@dataclass(frozen=True)
class BYOCJob:
    raw: dict[str, Any]
    job_id: str
    capability: str
    publish_url: Optional[str] = None
    subscribe_url: Optional[str] = None
    control_url: Optional[str] = None
    events_url: Optional[str] = None
    data_url: Optional[str] = None
    control: Optional[Control] = None
    events: Optional[Events] = None
    _media: Optional[MediaPublish] = field(default=None, repr=False, compare=False)
    _payment_session: Optional[BYOCPaymentSession] = field(default=None, repr=False, compare=False)
    _signed_job_header: Optional[str] = field(default=None, repr=False, compare=False)
    _payment_task: Optional[asyncio.Task] = field(default=None, repr=False, compare=False)
    _stream_stop_url: Optional[str] = field(default=None, repr=False, compare=False)
    _stop_timeout_s: float = field(default=30.0, repr=False, compare=False)

    @staticmethod
    def from_start_response(
        data: dict[str, Any],
        *,
        job_id: str,
        capability: str,
        payment_session: Optional[BYOCPaymentSession] = None,
        signed_job_header: Optional[str] = None,
        stream_stop_url: Optional[str] = None,
        stop_timeout_s: float = 30.0,
    ) -> "BYOCJob":
        headers = data.get("headers")
        if not isinstance(headers, dict):
            headers = {}

        control_url = _header_get(headers, "X-Control-Url")
        events_url = _header_get(headers, "X-Events-Url")
        publish_url = _header_get(headers, "X-Publish-Url")
        subscribe_url = _header_get(headers, "X-Subscribe-Url")
        data_url = _header_get(headers, "X-Data-Url")

        return BYOCJob(
            raw=data,
            job_id=job_id,
            capability=capability,
            publish_url=publish_url,
            subscribe_url=subscribe_url,
            control_url=control_url,
            events_url=events_url,
            data_url=data_url,
            control=Control(control_url) if control_url else None,
            events=Events(events_url) if events_url else None,
            _payment_session=payment_session,
            _signed_job_header=signed_job_header,
            _stream_stop_url=stream_stop_url,
            _stop_timeout_s=stop_timeout_s,
        )

    def start_media(self, config: MediaPublishConfig) -> MediaPublish:
        if not self.publish_url:
            raise LivepeerGatewayError("No publish_url present on this BYOC job")
        if self._media is None:
            media = MediaPublish(self.publish_url, config=config)
            object.__setattr__(self, "_media", media)
        return self._media

    def media_output(
        self,
        *,
        start_seq: int = -2,
        max_retries: int = 5,
        max_segment_bytes: Optional[int] = None,
        connection_close: bool = False,
        chunk_size: int = 64 * 1024,
        max_segments: int = 5,
        on_lag: LagPolicy = LagPolicy.LATEST,
    ) -> MediaOutput:
        if not self.subscribe_url:
            raise LivepeerGatewayError("No subscribe_url present on this BYOC job")
        return MediaOutput(
            self.subscribe_url,
            start_seq=start_seq,
            max_retries=max_retries,
            max_segment_bytes=max_segment_bytes,
            connection_close=connection_close,
            chunk_size=chunk_size,
            max_segments=max_segments,
            on_lag=on_lag,
        )

    @property
    def payment_session(self) -> Optional[BYOCPaymentSession]:
        return self._payment_session

    def start_payment_sender(self, *, interval_s: float = 5.0) -> Optional[asyncio.Task]:
        if getattr(self, "_payment_task", None) is not None:
            return self._payment_task
        if not self._payment_session or not self._signed_job_header:
            return None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            _LOG.warning(
                "No running event loop; BYOC payment sender not started. "
                "Call job.start_payment_sender() from async code to enable."
            )
            return None

        task = loop.create_task(
            _byoc_payment_sender(
                self._signed_job_header,
                self._payment_session,
                interval_s=interval_s,
            )
        )
        object.__setattr__(self, "_payment_task", task)
        return task

    async def stop(self) -> dict[str, Any]:
        if not self._stream_stop_url:
            raise LivepeerGatewayError("No stream stop URL present on this BYOC job")
        if not self._signed_job_header:
            raise LivepeerGatewayError("No signed job header present on this BYOC job")
        return await asyncio.to_thread(
            _post_byoc_stop,
            self._stream_stop_url,
            payload={"stream_id": self.job_id},
            headers={"Livepeer": self._signed_job_header},
            timeout=self._stop_timeout_s,
        )

    async def close(self) -> None:
        tasks = []
        payment_task = getattr(self, "_payment_task", None)
        if payment_task is not None and not payment_task.done():
            payment_task.cancel()
            tasks.append(payment_task)
        if self.control is not None:
            tasks.append(self.control.close())
        if self._media is not None:
            tasks.append(self._media.close())
        if not tasks:
            return
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                raise result


async def _byoc_payment_sender(
    signed_job_header: str,
    session: BYOCPaymentSession,
    *,
    interval_s: float,
) -> None:
    try:
        await asyncio.to_thread(session.send_stream_payment, signed_job_header)
    except SkipPaymentCycle as e:
        _LOG.debug("BYOC payment sender: first payment skipped (%s)", e)
    except Exception:
        _LOG.exception("BYOC payment sender: first immediate payment failed")

    while True:
        await asyncio.sleep(interval_s)
        try:
            await asyncio.to_thread(session.send_stream_payment, signed_job_header)
        except SkipPaymentCycle as e:
            _LOG.debug("BYOC payment sender: skipping payment cycle (%s)", e)
        except Exception:
            _LOG.exception("BYOC payment sender failed")


def _get_start_payment_headers(
    session: BYOCPaymentSession,
    *,
    payment_info: Any,
    capability_name: str,
    allow_skip: bool,
) -> tuple[Optional[str], str]:
    payment_header: Optional[str] = None
    segment_header = ""

    try:
        payment = session.get_payment()
        payment_header = payment.payment
        segment_header = payment.seg_creds or ""
    except SkipPaymentCycle as pay_skip:
        if not allow_skip:
            raise LivepeerGatewayError(
                "BYOC start endpoint returned HTTP 402 payment required, "
                "but the signer skipped payment generation; cannot retry start "
                "without a payment ticket."
            ) from pay_skip
        if not _orch_info_ticket_params_usable(payment_info):
            raise LivepeerGatewayError(
                "BYOC signer returned skip-payment, but OrchestratorInfo ticket_params "
                "are missing or zero (face_value and win_prob are required). "
                "Ticket expected value (EV) cannot be computed; refusing to start."
            ) from pay_skip
        _LOG.debug("BYOC signer returned skip-payment response on start (%s)", pay_skip)
    except (PaymentError, LivepeerGatewayError) as pay_err:
        msg = str(pay_err)
        if "priceinfo" in msg.lower() or "price" in msg.lower():
            raise LivepeerGatewayError(
                f"BYOC signer pricing error: the remote signer rejected payment generation "
                f"(likely missing or zero priceInfo in OrchestratorInfo for capability "
                f"'{capability_name}'). Ensure the orchestrator advertises "
                f"capability-specific pricing for BYOC/{capability_name}. "
                f"Signer detail: {msg}"
            ) from pay_err
        raise

    if payment_header is not None and not payment_header:
        raise LivepeerGatewayError(
            f"BYOC signer returned empty payment ticket for capability "
            f"'{capability_name}'. Check signer configuration and "
            f"orchestrator pricing for BYOC/{capability_name}."
        )

    return payment_header, segment_header


@dataclass(frozen=True)
class _SignedBYOCRequest:
    job_id: str
    capability: str
    timeout_seconds: int
    signed_job_header: str
    headers: dict[str, str]
    payment_info: Any
    payment_session: BYOCPaymentSession

    def payment_retry_headers(self) -> dict[str, str]:
        payment_header, segment_header = _get_start_payment_headers(
            self.payment_session,
            payment_info=self.payment_info,
            capability_name=self.capability,
            allow_skip=False,
        )
        return {
            "Livepeer": self.signed_job_header,
            "Livepeer-Payment": payment_header or "",
            "Livepeer-Segment": segment_header,
        }


def _signed_byoc_request(
    *,
    selected_url: str,
    req: BYOCJobRequest | BYOCProcessRequest,
    signer_url: Optional[str],
    signer_headers: Optional[dict[str, str]],
    capabilities: Any,
    use_tofu: bool,
) -> _SignedBYOCRequest:
    capability_name = req.capability.strip()
    payment_info, payment_capabilities = _get_payment_orch_info(
        selected_url,
        signer_url=signer_url,
        signer_headers=signer_headers,
        capabilities=capabilities,
        capability_name=capability_name,
        use_tofu=use_tofu,
    )

    session = BYOCPaymentSession(
        signer_url,
        payment_info,
        capability_name=capability_name,
        signer_headers=signer_headers,
        capabilities=payment_capabilities,
        stream_payment_endpoint=req.stream_payment_endpoint,
        use_tofu=use_tofu,
    )

    job_id = req._job_id()
    request_json = req._request_json(job_id) if isinstance(req, BYOCJobRequest) else req._request_json()
    parameters_json = req._parameters_json()
    timeout_seconds = max(1, int(req.timeout_seconds))
    signed = session.sign_byoc_job(
        job_id=job_id,
        capability=capability_name,
        request=request_json,
        parameters=parameters_json,
        timeout_seconds=timeout_seconds,
    )

    signed_payload = {
        "id": job_id,
        "request": request_json,
        "parameters": parameters_json,
        "capability": capability_name,
        "sender": signed.sender,
        "sig": signed.signature,
        "timeout_seconds": timeout_seconds,
    }
    signed_job_header = base64.b64encode(
        json.dumps(signed_payload, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")

    payment_header, segment_header = _get_start_payment_headers(
        session,
        payment_info=payment_info,
        capability_name=capability_name,
        allow_skip=True,
    )
    headers = {"Livepeer": signed_job_header}
    if payment_header:
        headers["Livepeer-Payment"] = payment_header
        headers["Livepeer-Segment"] = segment_header

    return _SignedBYOCRequest(
        job_id=job_id,
        capability=capability_name,
        timeout_seconds=timeout_seconds,
        signed_job_header=signed_job_header,
        headers=headers,
        payment_info=payment_info,
        payment_session=session,
    )


def _post_byoc_json(
    url: str,
    *,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
    op: str,
) -> dict[str, Any]:
    """
    POST a JSON payload to a BYOC endpoint and return a structured response.

    On HTTP 402, raises ``PaymentRequiredError``. Other HTTP errors raise
    ``LivepeerGatewayError``. The success response always includes
    ``{"status_code", "headers", "body"}``; ``body`` is decoded JSON when
    possible, raw text otherwise.
    """
    body_bytes = json.dumps(payload).encode("utf-8")
    req_headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "livepeer-python-gateway/0.1",
    }
    req_headers.update(headers)

    req = Request(url, data=body_bytes, headers=req_headers, method="POST")
    ssl_ctx = ssl._create_unverified_context()
    try:
        with urlopen(req, timeout=timeout, context=ssl_ctx) as resp:
            raw_body = resp.read().decode("utf-8", errors="replace")
            response_headers = {k: v for k, v in resp.headers.items()}
            status = resp.status
    except HTTPError as e:
        body = _extract_error_message(e)
        body_part = f"; body={body!r}" if body else ""
        if e.code == 402:
            raise PaymentRequiredError(
                f"HTTP BYOC {op} error: HTTP 402 from endpoint (url={url}){body_part}"
            ) from e
        raise LivepeerGatewayError(
            f"HTTP BYOC {op} error: HTTP {e.code} from endpoint (url={url}){body_part}"
        ) from e
    except ConnectionRefusedError as e:
        raise LivepeerGatewayError(
            f"HTTP BYOC {op} error: connection refused (is the server running? is the host/port correct?) (url={url})"
        ) from e
    except URLError as e:
        raise LivepeerGatewayError(
            f"HTTP BYOC {op} error: failed to reach endpoint: {getattr(e, 'reason', e)} (url={url})"
        ) from e
    except LivepeerGatewayError:
        raise
    except Exception as e:
        raise LivepeerGatewayError(
            f"HTTP BYOC {op} error: unexpected error: {e.__class__.__name__}: {e} (url={url})"
        ) from e

    parsed_body: Any = None
    if raw_body.strip():
        try:
            parsed_body = json.loads(raw_body)
        except Exception:
            parsed_body = raw_body
    return {"status_code": status, "headers": response_headers, "body": parsed_body}


def _post_byoc_process(
    url: str,
    *,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    return _post_byoc_json(url, payload=payload, headers=headers, timeout=timeout, op="process request")


def _post_byoc_start(
    url: str,
    *,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    return _post_byoc_json(url, payload=payload, headers=headers, timeout=timeout, op="start")


def _post_byoc_stop(
    url: str,
    *,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    return _post_byoc_json(url, payload=payload, headers=headers, timeout=timeout, op="stop")


def _derive_stream_stop_url(start_url: str, job_id: str) -> str:
    parsed = urlparse(start_url)
    path = parsed.path or ""
    quoted_job_id = quote(job_id, safe="")
    if path.endswith("/process/stream/start"):
        path = path[: -len("/process/stream/start")] + f"/process/stream/{quoted_job_id}/stop"
    elif path.endswith("/start"):
        path = path[: -len("/start")] + "/stop"
    else:
        raise ValueError(f"Cannot derive stream stop URL from start URL: {start_url!r}")
    return urlunparse(parsed._replace(path=path))


def _process_request_url(base_url: str, req: BYOCProcessRequest) -> str:
    endpoint = req.request_endpoint.strip()
    if not endpoint:
        raise LivepeerGatewayError("BYOCProcessRequest.request_endpoint must be non-empty")
    route = req.route.strip().lstrip("/")
    parsed = urlparse(endpoint)
    if parsed.scheme and parsed.netloc:
        root = endpoint.rstrip("/") + "/"
    else:
        root = resolve_transcoder_http_url(base_url, endpoint.rstrip("/") + "/")
    return urljoin(root, route)


def _resolve_byoc_token(
    token: Optional[str],
    *,
    signer_url: Optional[str],
    signer_headers: Optional[dict[str, str]],
    discovery_url: Optional[str],
    discovery_headers: Optional[dict[str, str]],
) -> tuple[Optional[str], Optional[dict[str, str]], Optional[str], Optional[dict[str, str]]]:
    resolved_signer_url = signer_url
    resolved_signer_headers = signer_headers
    resolved_discovery_url = discovery_url
    resolved_discovery_headers = discovery_headers
    if token is not None:
        from .token import parse_token

        token_data = parse_token(token)
        if resolved_signer_url is None:
            resolved_signer_url = token_data.get("signer")
        if resolved_signer_headers is None:
            resolved_signer_headers = token_data.get("signer_headers")
        if resolved_discovery_url is None:
            resolved_discovery_url = token_data.get("discovery")
        if resolved_discovery_headers is None:
            resolved_discovery_headers = token_data.get("discovery_headers")
    return resolved_signer_url, resolved_signer_headers, resolved_discovery_url, resolved_discovery_headers


def start_byoc_job(
    orch_url: Optional[Sequence[str] | str],
    req: BYOCJobRequest,
    *,
    token: Optional[str] = None,
    signer_url: Optional[str] = None,
    signer_headers: Optional[dict[str, str]] = None,
    discovery_url: Optional[str] = None,
    discovery_headers: Optional[dict[str, str]] = None,
    use_tofu: bool = True,
) -> BYOCJob:
    """
    Start a BYOC job through the Python gateway.

    Mirrors the LV2V ``start_lv2v`` shape: an orchestrator is selected (token
    orchestrators -> explicit orch_url -> token discovery -> explicit discovery
    -> signer-derived discovery), payment is preflighted, the job credential is
    signed via ``POST /sign-byoc-job``, and the job is started against
    ``stream_start_endpoint`` (default ``/ai/stream/start``).
    """
    if not isinstance(req.capability, str) or not req.capability.strip():
        raise LivepeerGatewayError("start_byoc_job requires a non-empty capability")
    if not isinstance(req.stream_start_endpoint, str) or not req.stream_start_endpoint.strip():
        raise LivepeerGatewayError("BYOCJobRequest.stream_start_endpoint must be non-empty")
    if not isinstance(req.stream_payment_endpoint, str) or not req.stream_payment_endpoint.strip():
        raise LivepeerGatewayError("BYOCJobRequest.stream_payment_endpoint must be non-empty")

    (
        resolved_signer_url,
        resolved_signer_headers,
        resolved_discovery_url,
        resolved_discovery_headers,
    ) = _resolve_byoc_token(
        token,
        signer_url=signer_url,
        signer_headers=signer_headers,
        discovery_url=discovery_url,
        discovery_headers=discovery_headers,
    )

    capabilities = build_capabilities(CapabilityId.BYOC, req.capability.strip())
    cursor = orchestrator_selector(
        orch_url,
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
            signed_req = _signed_byoc_request(
                selected_url=selected_url,
                req=req,
                signer_url=resolved_signer_url,
                signer_headers=resolved_signer_headers,
                capabilities=capabilities,
                use_tofu=use_tofu,
            )

            start_url = resolve_transcoder_http_url(info.transcoder, req.stream_start_endpoint)
            try:
                stop_url = _derive_stream_stop_url(start_url, signed_req.job_id)
            except ValueError as e:
                raise LivepeerGatewayError(str(e)) from e
            start_payload = req._body(signed_req.job_id)
            start_timeout = float(signed_req.timeout_seconds)
            try:
                data = _post_byoc_start(
                    start_url,
                    payload=start_payload,
                    headers=signed_req.headers,
                    timeout=start_timeout,
                )
            except PaymentRequiredError:
                _LOG.debug("BYOC start returned HTTP 402; retrying with a fresh payment ticket")
                data = _post_byoc_start(
                    start_url,
                    payload=start_payload,
                    headers=signed_req.payment_retry_headers(),
                    timeout=start_timeout,
                )
            job = BYOCJob.from_start_response(
                data,
                job_id=signed_req.job_id,
                capability=signed_req.capability,
                payment_session=signed_req.payment_session,
                signed_job_header=signed_req.signed_job_header,
                stream_stop_url=stop_url,
                stop_timeout_s=float(signed_req.timeout_seconds),
            )
            job.start_payment_sender()
            return job
        except LivepeerGatewayError as e:
            _LOG.debug(
                "start_byoc_job candidate failed, trying fallback if available: %s (%s)",
                selected_url,
                str(e),
            )
            start_rejections.append(OrchestratorRejection(url=selected_url, reason=str(e)))


def process_byoc_request(
    orch_url: Optional[Sequence[str] | str],
    req: BYOCProcessRequest,
    *,
    token: Optional[str] = None,
    signer_url: Optional[str] = None,
    signer_headers: Optional[dict[str, str]] = None,
    discovery_url: Optional[str] = None,
    discovery_headers: Optional[dict[str, str]] = None,
    use_tofu: bool = True,
) -> BYOCProcessResponse:
    if not isinstance(req.capability, str) or not req.capability.strip():
        raise LivepeerGatewayError("process_byoc_request requires a non-empty capability")
    if not isinstance(req.route, str):
        raise LivepeerGatewayError("BYOCProcessRequest.route must be a string")
    if not isinstance(req.stream_payment_endpoint, str) or not req.stream_payment_endpoint.strip():
        raise LivepeerGatewayError("BYOCProcessRequest.stream_payment_endpoint must be non-empty")

    (
        resolved_signer_url,
        resolved_signer_headers,
        resolved_discovery_url,
        resolved_discovery_headers,
    ) = _resolve_byoc_token(
        token,
        signer_url=signer_url,
        signer_headers=signer_headers,
        discovery_url=discovery_url,
        discovery_headers=discovery_headers,
    )
    capabilities = build_capabilities(CapabilityId.BYOC, req.capability.strip())
    cursor = orchestrator_selector(
        orch_url,
        signer_url=resolved_signer_url,
        signer_headers=resolved_signer_headers,
        discovery_url=resolved_discovery_url,
        discovery_headers=resolved_discovery_headers,
        capabilities=capabilities,
        use_tofu=use_tofu,
    )

    process_rejections: list[OrchestratorRejection] = []
    while True:
        try:
            selected_url, info = cursor.next()
        except NoOrchestratorAvailableError as e:
            all_rejections = list(e.rejections) + process_rejections
            if all_rejections:
                raise NoOrchestratorAvailableError(
                    f"All orchestrators failed ({len(all_rejections)} tried)",
                    rejections=all_rejections,
                ) from None
            raise

        try:
            signed_req = _signed_byoc_request(
                selected_url=selected_url,
                req=req,
                signer_url=resolved_signer_url,
                signer_headers=resolved_signer_headers,
                capabilities=capabilities,
                use_tofu=use_tofu,
            )
            process_url = _process_request_url(info.transcoder, req)
            payload = req._body()
            timeout = float(signed_req.timeout_seconds)
            try:
                data = _post_byoc_process(
                    process_url,
                    payload=payload,
                    headers=signed_req.headers,
                    timeout=timeout,
                )
            except PaymentRequiredError:
                _LOG.debug("BYOC process returned HTTP 402; retrying with a fresh payment ticket")
                data = _post_byoc_process(
                    process_url,
                    payload=payload,
                    headers=signed_req.payment_retry_headers(),
                    timeout=timeout,
                )
            return BYOCProcessResponse(
                status_code=int(data["status_code"]),
                headers=data["headers"],
                body=data["body"],
                job_id=signed_req.job_id,
                capability=signed_req.capability,
                orchestrator_url=selected_url,
            )
        except LivepeerGatewayError as e:
            _LOG.debug(
                "process_byoc_request candidate failed, trying fallback if available: %s (%s)",
                selected_url,
                str(e),
            )
            process_rejections.append(OrchestratorRejection(url=selected_url, reason=str(e)))


def stream_byoc_request(
    orch_url: Optional[Sequence[str] | str],
    req: BYOCProcessRequest,
    *,
    token: Optional[str] = None,
    signer_url: Optional[str] = None,
    signer_headers: Optional[dict[str, str]] = None,
    discovery_url: Optional[str] = None,
    discovery_headers: Optional[dict[str, str]] = None,
    use_tofu: bool = True,
) -> BYOCProcessStream:
    if not isinstance(req.capability, str) or not req.capability.strip():
        raise LivepeerGatewayError("stream_byoc_request requires a non-empty capability")

    (
        resolved_signer_url,
        resolved_signer_headers,
        resolved_discovery_url,
        resolved_discovery_headers,
    ) = _resolve_byoc_token(
        token,
        signer_url=signer_url,
        signer_headers=signer_headers,
        discovery_url=discovery_url,
        discovery_headers=discovery_headers,
    )
    capabilities = build_capabilities(CapabilityId.BYOC, req.capability.strip())
    cursor = orchestrator_selector(
        orch_url,
        signer_url=resolved_signer_url,
        signer_headers=resolved_signer_headers,
        discovery_url=resolved_discovery_url,
        discovery_headers=resolved_discovery_headers,
        capabilities=capabilities,
        use_tofu=use_tofu,
    )

    stream_rejections: list[OrchestratorRejection] = []
    while True:
        try:
            selected_url, info = cursor.next()
        except NoOrchestratorAvailableError as e:
            all_rejections = list(e.rejections) + stream_rejections
            if all_rejections:
                raise NoOrchestratorAvailableError(
                    f"All orchestrators failed ({len(all_rejections)} tried)",
                    rejections=all_rejections,
                ) from None
            raise

        try:
            signed_req = _signed_byoc_request(
                selected_url=selected_url,
                req=req,
                signer_url=resolved_signer_url,
                signer_headers=resolved_signer_headers,
                capabilities=capabilities,
                use_tofu=use_tofu,
            )
            process_url = _process_request_url(info.transcoder, req)
            events = SSEClient.post_json(
                process_url,
                payload=req._body(),
                headers=signed_req.headers,
                timeout=float(signed_req.timeout_seconds),
                retry_headers=signed_req.payment_retry_headers,
            )
            return BYOCProcessStream(
                status_code=0,
                headers={},
                events=events,
                job_id=signed_req.job_id,
                capability=signed_req.capability,
                orchestrator_url=selected_url,
            )
        except LivepeerGatewayError as e:
            _LOG.debug(
                "stream_byoc_request candidate failed, trying fallback if available: %s (%s)",
                selected_url,
                str(e),
            )
            stream_rejections.append(OrchestratorRejection(url=selected_url, reason=str(e)))
