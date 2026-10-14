from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, replace
from typing import Any, Optional

from .async_cache import async_lru_cache
from .errors import (
    LivepeerGatewayError,
    LivepeerHTTPError,
    PaymentError,
    SignerRefreshRequired,
    SkipPaymentCycle,
)
_LOG = logging.getLogger(__name__)

# Must stay under the signer's opening payment of 10 seconds.
PAYMENT_INTERVAL_S = 3.0

@dataclass(frozen=True)
class GetPaymentResponse:
    payment: str
    seg_creds: Optional[str] = None


@dataclass(frozen=True)
class LivePaymentChallenge:
    """The complete payment contract returned by a live-runner 402."""

    payment_params: str
    manifest_id: str
    payment_url: str


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


def _freeze_headers(
    headers: Optional[dict[str, str]],
) -> Optional[frozenset[tuple[str, str]]]:
    """Convert a headers dict to a hashable cache key."""
    if headers is None:
        return None
    return frozenset(headers.items())


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
    _signer_headers: frozenset[tuple[str, str]] | None = None,
) -> SignerMaterial:
    """Fetch and cache the signer identity used for Live Runner payments."""
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
        challenge: LivePaymentChallenge,
        app: str | None = None,
        max_price: dict[str, Any] | None = None,
        max_refresh_retries: int = 3,
    ) -> None:
        self._signer_url = signer_url
        self._signer_headers = _freeze_headers(signer_headers)
        self._type = type
        self._challenge = challenge
        self._app = app
        self._max_price = dict(max_price) if max_price is not None else None
        self._max_refresh_retries = max(0, int(max_refresh_retries))
        self._state: dict[str, Any] | None = None

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
                await self._refresh_payment_params()
                attempts += 1

    async def send_payment(self) -> None:
        """Generate a payment and POST it to the challenge's endpoint.

        Raises LivepeerHTTPError on error responses so callers can branch on
        the status code, and SkipPaymentCycle when the signer gates the cycle.
        """
        if not self._signer_url:
            return

        from .http import _post_empty

        payment = await self.get_payment()
        if not payment.seg_creds:
            # An empty segment header fails the orchestrator's sig check and
            # comes back 403, which reads as a dead session, not a bad signer.
            raise PaymentError("Signer returned a payment with no segCreds")
        headers = {
            "Livepeer-Payment": payment.payment,
            "Livepeer-Segment": payment.seg_creds,
        }
        await _post_empty(self._challenge.payment_url, headers=headers, timeout=5.0)

    async def run_payments(self) -> bool:
        """Keep a metered session funded until cancelled or the session ends.

        Cancel the task to stop; the first payment waits one interval, since
        the caller pays upfront. Returns True if the orchestrator reports that
        the challenge's session-scoped endpoint is gone.
        """
        while True:
            await asyncio.sleep(PAYMENT_INTERVAL_S)
            try:
                await self.send_payment()
            except SkipPaymentCycle as e:
                _LOG.debug("Payment loop skipped cycle: %s", e)
            except LivepeerHTTPError as e:
                # A 4xx will not change on a retry (404 gone, 409 fixed price,
                # 403 mismatch), so stop rather than mint tickets nobody will
                # honour. 408 and 429 are the two that do ask to be retried.
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
            "orchestrator": self._challenge.payment_params,
            "type": self._type,
            "ManifestID": self._challenge.manifest_id,
        }
        if self._app:
            payload["app"] = self._app
        if self._max_price is not None:
            payload["maxPrice"] = dict(self._max_price)
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

    async def _refresh_payment_params(self) -> None:
        from .http import _http_origin, post_json

        signer = await get_signer_info(self._signer_url or "", self._signer_headers)
        if not signer.address:
            raise PaymentError("Cannot refresh payment without signer address")

        url = f"{_http_origin(self._challenge.payment_url)}/refresh-payment"
        data = await post_json(
            url,
            {
                "sender": signer.address,
                "manifest_id": self._challenge.manifest_id,
            },
        )
        payment_params = data.get("payment_params")
        if not isinstance(payment_params, str) or not payment_params:
            raise PaymentError(
                f"RefreshPayment error: missing/invalid 'payment_params' in response (url={url})"
            )
        # Refresh rotates the embedded payment material. The initial scoped
        # endpoint remains authoritative for the lifetime of this session.
        self._challenge = replace(
            self._challenge,
            payment_params=payment_params,
        )
