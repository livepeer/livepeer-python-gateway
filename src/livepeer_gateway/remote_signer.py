from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import ssl
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from . import lp_rpc_pb2
from .async_cache import async_lru_cache
from .errors import (
    LivepeerGatewayError,
    LivepeerHTTPError,
    PaymentError,
    SignerRefreshRequired,
    SkipPaymentCycle,
)
_LOG = logging.getLogger(__name__)

# Client payment cadence. The orchestrator debits metered sessions every
# -livePaymentInterval (5s by default) and drops the session on the first tick
# it cannot cover, so pay comfortably ahead of it.
PAYMENT_INTERVAL_S = 3.0

@dataclass(frozen=True)
class GetPaymentResponse:
    payment: str
    seg_creds: Optional[str] = None


@dataclass(frozen=True)
class SignerMaterial:
    """
    Material returned by the remote signer.
    address: opaque broadcaster address string.
    sig: opaque signature string.
    """
    address: str | None
    sig: str | None


@dataclass
class RemoteSignerError(LivepeerGatewayError):
    signer_url: str
    message: str
    cause: Optional[BaseException] = None

    def __str__(self) -> str:
        return f"Remote signer error: {self.message} (url={self.signer_url})"


_HEX_RE = re.compile(r"^(0x)?[0-9a-fA-F]*$")


def _freeze_headers(
    headers: Optional[dict[str, str]],
) -> Optional[frozenset[tuple[str, str]]]:
    """Convert a headers dict to a frozenset for use with @lru_cache."""
    if headers is None:
        return None
    return frozenset(headers.items())

def _hex_to_bytes(s: str, *, expected_len: Optional[int] = None) -> bytes:
    s = s.strip()
    if not _HEX_RE.match(s):
        raise ValueError(f"Not a hex string: {s!r}")
    if s.startswith(("0x", "0X")):
        s = s[2:]
    if len(s) % 2 == 1:
        # allow odd-length hex (pad left)
        s = "0" + s
    b = bytes.fromhex(s)
    if expected_len is not None and len(b) != expected_len:
        raise ValueError(f"Expected {expected_len} bytes, got {len(b)} bytes")
    return b


def _signer_material_from_json(
    data: dict[str, Any],
    signer_url: str,
) -> SignerMaterial:
    if "address" not in data or "signature" not in data:
        raise RemoteSignerError(
            signer_url,
            f"Remote signer JSON must contain 'address' and 'signature': {data!r}",
            cause=None,
        ) from None

    address = data["address"]
    sig = data["signature"]
    if not isinstance(address, str) or not address:
        raise RemoteSignerError(
            signer_url,
            f"Remote signer 'address' must be a non-empty string: {address!r}",
            cause=None,
        ) from None
    if not isinstance(sig, str) or not sig:
        raise RemoteSignerError(
            signer_url,
            f"Remote signer 'signature' must be a non-empty string: {sig!r}",
            cause=None,
        ) from None

    return SignerMaterial(address=address, sig=sig)


@lru_cache(maxsize=None)
def get_orch_info_sig(
    signer_url: str,
    # frozenset instead of dict because @lru_cache requires hashable arguments.
    _signer_headers: Optional[frozenset[tuple[str, str]]] = None,
) -> SignerMaterial:
    """
    Fetch signer material exactly once per (signer_url, headers) combination
    for the lifetime of the process. Subsequent calls return cached data.
    """
    from .http import _extract_error_message, _http_origin, post_json_sync as post_json

    # check for offchain mode
    if not signer_url:
        return SignerMaterial(address=None, sig=None)

    # Accept either a base URL or a full URL that includes /sign-orchestrator-info.
    # Normalize to an https:// origin and append the expected path.
    signer_url = f"{_http_origin(signer_url)}/sign-orchestrator-info"
    headers = dict(_signer_headers) if _signer_headers else None

    try:
        # Some signers accept/expect POST with an empty JSON object.
        data = post_json(signer_url, {}, headers=headers, timeout=5.0)

        signer = _signer_material_from_json(data, signer_url)

    except LivepeerGatewayError as e:
        if isinstance(e, RemoteSignerError):
            raise

        # post_json wraps the underlying exception as __cause__; convert back into
        # a signer-specific error message.
        cause = e.__cause__ or e

        if isinstance(cause, HTTPError):
            body = _extract_error_message(cause)
            body_part = f"; body={body!r}" if body else ""
            raise RemoteSignerError(
                signer_url,
                f"HTTP {cause.code} from signer{body_part}",
                cause=cause,
            ) from None

        if isinstance(cause, ConnectionRefusedError):
            raise RemoteSignerError(
                signer_url,
                "connection refused (is the signer running? is the host/port correct?)",
                cause=cause,
            ) from None

        if isinstance(cause, URLError):
            raise RemoteSignerError(
                signer_url,
                f"failed to reach signer: {getattr(cause, 'reason', cause)}",
                cause=cause,
            ) from None

        if isinstance(cause, json.JSONDecodeError):
            raise RemoteSignerError(
                signer_url,
                f"signer did not return valid JSON: {cause}",
                cause=cause,
            ) from None

        raise RemoteSignerError(
            signer_url,
            f"unexpected error: {cause.__class__.__name__}: {cause}",
            cause=cause if isinstance(cause, BaseException) else e,
        ) from None

    return signer


@async_lru_cache(maxsize=128)
async def get_signer_info(
    signer_url: str,
    # frozenset instead of dict because cache keys require hashable arguments.
    _signer_headers: frozenset[tuple[str, str]] | None = None,
) -> SignerMaterial:
    """
    Async-native version of get_orch_info_sig for callers that should not block
    the event loop or use gRPC.
    """
    from .http import _http_origin, post_json

    if not signer_url:
        return SignerMaterial(address=None, sig=None)

    url = f"{_http_origin(signer_url)}/sign-orchestrator-info"
    headers = dict(_signer_headers) if _signer_headers else None
    data = await post_json(url, {}, headers=headers, timeout=5.0)
    return _signer_material_from_json(data, url)


class LivePaymentSession:
    def __init__(
        self,
        signer_url: str | None,
        *,
        signer_headers: dict[str, str] | None = None,
        type: str,
        payment_params: str,
        manifest_id: str,
        orchestrator_url: str | None = None,
        max_refresh_retries: int = 3,
    ) -> None:
        self._signer_url = signer_url
        self._signer_headers = _freeze_headers(signer_headers)
        self._type = type
        self._payment_params = payment_params
        self._manifest_id = manifest_id
        self._max_refresh_retries = max(0, int(max_refresh_retries))
        self._state: dict[str, Any] | None = None
        self._orchestrator_url = orchestrator_url

    async def get_payment(self) -> GetPaymentResponse:
        if not self._signer_url:
            return GetPaymentResponse(payment="", seg_creds=None)

        attempts = 0
        while True:
            try:
                return await self._payment_request()
            except SignerRefreshRequired as e:
                if attempts >= self._max_refresh_retries:
                    raise PaymentError(
                        f"Signer refresh required after {attempts} retries: {e}"
                    ) from e
                if self._state is None:
                    raise
                orchestrator_url = e.orchestrator_url
                if not orchestrator_url:
                    raise PaymentError(
                        "Signer refresh response missing Livepeer-Orchestrator-URL header"
                    ) from e
                await self._refresh_payment_params(orchestrator_url)
                attempts += 1

    async def send_payment(
        self,
        orchestrator_url: Optional[str] = None,
        *,
        payment_url: Optional[str] = None,
    ) -> None:
        """Generate a payment and POST it to the orchestrator.

        ``payment_url`` targets a specific endpoint, such as the session-scoped
        one which 404s once the session is released. Without it the payment
        goes to the orchestrator's generic ``/payment`` endpoint, which credits
        the payer balance blindly and cannot report a dead session.

        Raises LivepeerHTTPError on error responses so callers can branch on
        the status code, and SkipPaymentCycle when the signer gates the cycle.
        """
        if not self._signer_url:
            return

        from .http import _http_origin, post_empty

        if payment_url:
            url = payment_url
        else:
            target = orchestrator_url or self._orchestrator_url
            if not target:
                raise PaymentError("orchestrator_url is required before sending payment")
            url = f"{_http_origin(target)}/payment"

        payment = await self.get_payment()
        headers = {
            "Livepeer-Payment": payment.payment,
            "Livepeer-Segment": payment.seg_creds or "",
        }
        await post_empty(url, headers=headers, timeout=5.0)

    async def run_payments(
        self,
        *,
        payment_url: Optional[str] = None,
        interval_s: float = PAYMENT_INTERVAL_S,
    ) -> bool:
        """Keep a metered session funded until cancelled or the session ends.

        Returns True when the orchestrator reports the session gone, so the
        owner can surface it as released; returns False for the other terminal
        rejections. Cancel the task to stop funding.

        The caller pays upfront before starting this loop, so the first
        follow-up waits one interval. A payment covers the time since the
        previous one, so transient failures are retried rather than fatal:
        the next payment settles the arrears.
        """
        while True:
            await asyncio.sleep(interval_s)
            try:
                await self.send_payment(payment_url=payment_url)
            except asyncio.CancelledError:
                raise
            except SkipPaymentCycle as e:
                _LOG.debug("Payment loop skipped cycle: %s", e)
            except LivepeerHTTPError as e:
                # 404 session released, 409 fixed price, 403 session/payment
                # mismatch: all terminal, and paying on would mint tickets the
                # orchestrator will never honor.
                if 400 <= e.status_code < 500 and e.status_code not in (408, 429):
                    _LOG.info("Payment loop stopping (HTTP %d): %s", e.status_code, e)
                    return e.status_code == 404
                _LOG.warning("Payment failed; retrying next cycle: %s", e)
            except Exception as e:
                _LOG.warning("Payment failed; retrying next cycle: %s", e)

    async def _payment_request(self) -> GetPaymentResponse:
        from .http import _http_origin, post_json

        url = f"{_http_origin(self._signer_url)}/generate-live-payment"
        payload: dict[str, Any] = {
            "orchestrator": self._payment_params,
            "type": self._type,
            "ManifestID": self._manifest_id,
        }
        if self._state is not None:
            payload["state"] = self._state

        headers = dict(self._signer_headers) if self._signer_headers else None
        data = await post_json(url, payload, headers=headers)
        payment = data.get("payment")
        if not isinstance(payment, str) or not payment:
            raise PaymentError(
                f"GetPayment error: missing/invalid 'payment' in response (url={url})"
            )

        seg_creds = data.get("segCreds")
        if seg_creds is not None and not isinstance(seg_creds, str):
            raise PaymentError(
                f"GetPayment error: invalid 'segCreds' in response (url={url})"
            )

        state = data.get("state")
        if not isinstance(state, dict):
            raise PaymentError(
                f"Remote signer response missing 'state' object (url={url})"
            )

        self._state = state
        return GetPaymentResponse(payment=payment, seg_creds=seg_creds)

    async def _refresh_payment_params(self, orchestrator_url: str) -> None:
        from .http import _http_origin, post_json

        signer = await get_signer_info(self._signer_url or "", self._signer_headers)
        if not signer.address:
            raise PaymentError("Cannot refresh payment without signer address")

        url = f"{_http_origin(orchestrator_url)}/refresh-payment"
        data = await post_json(
            url,
            {
                "sender": signer.address,
                "manifest_id": self._manifest_id,
            },
        )
        payment_params = data.get("payment_params")
        if not isinstance(payment_params, str) or not payment_params:
            raise PaymentError(
                f"RefreshPayment error: missing/invalid 'payment_params' in response (url={url})"
            )
        self._payment_params = payment_params
        refreshed_orchestrator_url = data.get("orchestrator")
        self._orchestrator_url = (
            refreshed_orchestrator_url
            if isinstance(refreshed_orchestrator_url, str) and refreshed_orchestrator_url.strip()
            else orchestrator_url
        )


class PaymentSession:
    def __init__(
        self,
        signer_url: Optional[str],
        info: lp_rpc_pb2.OrchestratorInfo,
        *,
        signer_headers: Optional[dict[str, str]] = None,
        type: str,
        capabilities: Optional[lp_rpc_pb2.Capabilities] = None,
        use_tofu: bool = True,
        max_refresh_retries: int = 3,
    ) -> None:
        self._signer_url = signer_url
        self._signer_headers = signer_headers
        self._info = info
        self._type = type
        self._manifest_id: Optional[str] = None
        self._capabilities = capabilities
        self._use_tofu = use_tofu
        self._max_refresh_retries = max(0, int(max_refresh_retries))
        self._state: Optional[dict[str, str]] = None

    def set_manifest_id(self, manifest_id: str) -> None:
        if not isinstance(manifest_id, str) or not manifest_id.strip():
            raise PaymentError("manifest_id must be a non-empty string")
        self._manifest_id = manifest_id.strip()

    def get_payment(self) -> GetPaymentResponse:
        """
        Generate a payment via the remote signer.

        Handles signer state round-tripping internally.
        On HTTP 480, refreshes OrchestratorInfo and retries
        (up to max_refresh_retries).
        Returns payment + seg_creds for use as HTTP headers.
        """

        # Offchain mode: still send the expected headers, but with empty content.
        if not self._signer_url:
            seg = lp_rpc_pb2.SegData()
            if not self._info.HasField("auth_token"):
                raise PaymentError(
                    "Orchestrator did not provide an auth token."
                )
            seg.auth_token.CopyFrom(self._info.auth_token)
            seg = base64.b64encode(seg.SerializeToString()).decode("ascii")
            return GetPaymentResponse(seg_creds=seg, payment="")

        def _payment_request() -> GetPaymentResponse:
            from .http import _http_origin, post_json_sync as post_json

            base = _http_origin(self._signer_url)
            url = f"{base}/generate-live-payment"

            pb = self._info.SerializeToString()
            orch_b64 = base64.b64encode(pb).decode("ascii")
            payload: dict[str, Any] = {
                "orchestrator": orch_b64,
                "type": self._type,
            }
            if self._capabilities is not None:
                payload["capabilities"] = base64.b64encode(
                    self._capabilities.SerializeToString()
                ).decode("ascii")
            if self._manifest_id is not None:
                payload["ManifestID"] = self._manifest_id
            if self._state is not None:
                payload["state"] = self._state

            data = post_json(url, payload, headers=self._signer_headers)
            payment = data.get("payment")
            if not isinstance(payment, str) or not payment:
                raise PaymentError(
                    f"GetPayment error: missing/invalid 'payment' in response (url={url})"
                )

            seg_creds = data.get("segCreds")
            if seg_creds is not None and not isinstance(seg_creds, str):
                raise PaymentError(
                    f"GetPayment error: invalid 'segCreds' in response (url={url})"
                )

            state = data.get("state")
            if not isinstance(state, dict):
                raise PaymentError(
                    f"Remote signer response missing 'state' object (url={url})"
                )

            self._state = state
            return GetPaymentResponse(payment=payment, seg_creds=seg_creds)

        attempts = 0
        while True:
            try:
                return _payment_request()
            except SignerRefreshRequired as e:
                if attempts >= self._max_refresh_retries:
                    raise PaymentError(
                        f"Signer refresh required after {attempts} retries: {e}"
                    ) from e
                if not self._info.transcoder:
                    raise PaymentError(
                        "OrchestratorInfo missing transcoder URL for refresh"
                    )
                from .orch_info import get_orch_info

                self._info = get_orch_info(
                    self._info.transcoder,
                    signer_url=self._signer_url,
                    signer_headers=self._signer_headers,
                    capabilities=self._capabilities,
                    use_tofu=self._use_tofu,
                )
                attempts += 1

    def send_payment(self) -> None:
        """
        Generate a payment (via get_payment) and forward it
        to the orchestrator via POST {orch}/payment.
        """
        from .http import _extract_error_message, _http_origin

        p = self.get_payment()
        if not self._info.transcoder:
            raise PaymentError("OrchestratorInfo missing transcoder URL for payment")
        base = _http_origin(self._info.transcoder)
        url = f"{base}/payment"
        headers = {
            "Livepeer-Payment": p.payment,
            "Livepeer-Segment": p.seg_creds or "",
        }
        req = Request(url, data=b"", headers=headers, method="POST")
        ssl_ctx = ssl._create_unverified_context()
        try:
            with urlopen(req, timeout=5.0, context=ssl_ctx) as resp:
                resp.read()
        except HTTPError as e:
            body = _extract_error_message(e)
            body_part = f"; body={body!r}" if body else ""
            raise PaymentError(
                f"HTTP payment error: HTTP {e.code} from endpoint (url={url}){body_part}"
            ) from e
        except ConnectionRefusedError as e:
            raise PaymentError(
                f"HTTP payment error: connection refused (is the server running? is the host/port correct?) (url={url})"
            ) from e
        except URLError as e:
            raise PaymentError(
                f"HTTP payment error: failed to reach endpoint: {getattr(e, 'reason', e)} (url={url})"
            ) from e
        except Exception as e:
            raise PaymentError(
                f"HTTP payment error: unexpected error: {e.__class__.__name__}: {e} (url={url})"
            ) from e
