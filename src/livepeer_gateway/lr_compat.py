"""Live Runner payment compatibility layer (ADDITIVE port).

This module exists so Live Runner support can be added to a gateway build whose
`remote_signer` predates `LivePaymentSession`, WITHOUT modifying any shared file.
`remote_signer.py`, `errors.py` and `byoc.py` drive the BYOC payment and
orchestrator-selection path in production, so they are deliberately left
untouched: swapping them wholesale was measured to change BYOC orchestrator
selection (it picked an unreachable orch from DISCOVERY_URL) and break inference.

Everything here is either new (`LivepeerHTTPError`) or lifted verbatim from the
newer gateway lineage (`LivePaymentSession`, `get_signer_info`). When the two
lineages converge upstream, delete this file and re-point the imports.
"""
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

import aiohttp
from aiohttp import ClientSession, ClientTimeout

from . import lp_rpc_pb2
from .async_cache import async_lru_cache
from .errors import LivepeerGatewayError, SignerRefreshRequired, SkipPaymentCycle
from .remote_signer import (
    GetPaymentResponse,
    PaymentError,
    RemoteSignerError,
    SignerMaterial,
    _freeze_headers,
)

_LOG = logging.getLogger(__name__)


class LivepeerHTTPError(LivepeerGatewayError):
    """Raised when an HTTP endpoint returns a non-success status.

    Copied verbatim from the newer lineage's errors.py; the vendored errors.py
    predates it. Keep the signature identical — callers pass (status, url, body,
    message) positionally.
    """

    def __init__(self, status_code: int, url: str, body: str = "", message: str | None = None) -> None:
        self.status_code = int(status_code)
        self.url = url
        self.body = body
        super().__init__(message or f"HTTP {status_code} from endpoint (url={url})")


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


@async_lru_cache(maxsize=128)
async def get_signer_info(
    signer_url: str,
    # frozenset instead of dict because cache keys require hashable arguments.
    _signer_headers: Optional[frozenset[tuple[str, str]]] = None,
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
        signer_url: Optional[str],
        *,
        signer_headers: Optional[dict[str, str]] = None,
        type: str,
        payment_params: str,
        manifest_id: str,
        orchestrator_url: Optional[str] = None,
        capabilities: Optional[lp_rpc_pb2.Capabilities] = None,
        in_pixels: Optional[int] = None,
        max_refresh_retries: int = 3,
    ) -> None:
        self._signer_url = signer_url
        self._signer_headers = _freeze_headers(signer_headers)
        self._type = type
        self._payment_params = payment_params
        self._manifest_id = manifest_id
        self._capabilities = capabilities
        # Explicit unit count for the signer's `inPixels`. When set (e.g. 1 for a
        # fixed-price single-shot live-runner generation) it is forwarded to
        # /generate-live-payment and takes precedence over the signer's automatic
        # continuous 720p30 estimate on the lv2v path. Left None for continuous
        # live-video runners so their payload stays byte-identical.
        self._in_pixels = in_pixels
        self._max_refresh_retries = max(0, int(max_refresh_retries))
        self._state: Optional[dict[str, Any]] = None
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

    async def send_payment(self, orchestrator_url: Optional[str] = None) -> None:
        if not self._signer_url:
            return

        target = orchestrator_url or self._orchestrator_url
        if not target:
            raise PaymentError("orchestrator_url is required before sending payment")

        from .http import _extract_error_message_from_body, _http_origin

        payment = await self.get_payment()
        url = f"{_http_origin(target)}/payment"
        headers = {
            "Livepeer-Payment": payment.payment,
            "Livepeer-Segment": payment.seg_creds,
        }
        try:
            timeout = aiohttp.ClientTimeout(total=5.0)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, data=b"", headers=headers) as resp:
                    if resp.status >= 400:
                        body = await resp.text()
                        message = _extract_error_message_from_body(body)
                        body_part = f"; body={message!r}" if message else ""
                        raise PaymentError(
                            f"HTTP payment error: HTTP {resp.status} from endpoint (url={url}){body_part}"
                        )
                    await resp.read()
        except PaymentError:
            raise
        except getattr(aiohttp, "ClientConnectorError", ()) as e:
            raise PaymentError(
                f"HTTP payment error: failed to reach endpoint: {getattr(e, 'message', e)} (url={url})"
            ) from e
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            raise PaymentError(
                f"HTTP payment error: failed to reach endpoint: {getattr(e, 'message', e)} (url={url})"
            ) from e

    async def _payment_request(self) -> GetPaymentResponse:
        from .http import _http_origin, post_json

        url = f"{_http_origin(self._signer_url)}/generate-live-payment"
        payload: dict[str, Any] = {
            "orchestrator": self._payment_params,
            "type": self._type,
            "ManifestID": self._manifest_id,
        }
        if self._in_pixels is not None:
            payload["inPixels"] = self._in_pixels
        if self._capabilities is not None:
            payload["capabilities"] = base64.b64encode(
                self._capabilities.SerializeToString()
            ).decode("ascii")
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
