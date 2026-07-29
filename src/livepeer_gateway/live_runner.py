from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Literal,
    Mapping,
    NotRequired,
    Optional,
    Protocol,
    TypedDict,
    cast,
    overload,
)
from urllib.parse import quote, urlparse, urlunparse

import aiohttp

from .channel_reader import ChannelReader
from .errors import LivepeerGatewayError, LivepeerHTTPError, SignerRefreshRequired
from .http import open_stream, post_empty, post_json, request_json
from .remote_signer import (
    GetPaymentResponse,
    LivePaymentSession,
    _freeze_headers,
    get_signer_info,
)

_LOG = logging.getLogger(__name__)

_DEFAULT_HEARTBEAT_INTERVAL_S = 5.0
_LIVE_RUNNER_PAYER_ADDRESS_HEADER = "Livepeer-Payer-Address"
_LIVE_RUNNER_MODES = frozenset({"persistent", "single-shot"})
_RUNNER_PAYMENT_TYPES_BY_UNIT = {
    "hour": "live",
    "seconds": "live",
    "720p": "lv2v",
    "720p-pixel-seconds": "lv2v",
    "fixed": "fixed",
}

# golang format duration, eg "10s"
_DURATION_RE = re.compile(r"^\s*(?P<value>[0-9]+(?:\.[0-9]+)?)(?P<unit>ns|us|\u00b5s|ms|s|m|h)\s*$")


class LiveRunnerTrickleChannelRequest(TypedDict):
    name: str
    mime_type: str


class LiveRunnerTrickleChannel(TypedDict):
    name: str
    channel_name: str
    # Public/external trickle URL.
    url: str
    # Optional private-network URL for runner-to-orchestrator traffic. When the
    # runner and orchestrator share a network, such as Docker, this can bypass
    # public TLS/routing, but it is only present when the orchestrator is
    # configured to return one.
    internal_url: NotRequired[str]
    mime_type: str


class LiveRunnerSessionHeaders(Protocol):
    def get(self, key: str, default: str = "") -> str: ...


class LiveRunnerSessionRequest(Protocol):
    headers: LiveRunnerSessionHeaders


@dataclass(frozen=True)
class LiveRunnerSessionEvent:
    session_id: str
    event: Literal["reserved", "released"]
    timestamp: Optional[str]
    raw: dict[str, Any]


LiveRunnerSessionCallback = Callable[[LiveRunnerSessionEvent], None | Awaitable[None]]


@dataclass(frozen=True)
class LiveRunnerInstance:
    """A normalized live runner discovered from an orchestrator entry."""

    url: str
    app: str
    runner_id: str
    mode: str
    orchestrator_url: str
    raw: dict[str, Any]
    price_info: Optional[LiveRunnerPriceInfo] = None


@dataclass(frozen=True)
class LiveRunnerSession:
    session_id: str
    app_url: str
    runner_url: str
    runner: Optional[LiveRunnerInstance] = None


@dataclass(frozen=True)
class LiveRunnerProxy:
    proxy_id: str
    url: str


@dataclass(frozen=True)
class LiveRunnerCallResult:
    data: dict[str, Any]
    runner_url: str
    runner: Optional[LiveRunnerInstance] = None
    session_id: str = ""
    payment_session: Optional[LivePaymentSession] = field(
        default=None,
        repr=False,
        compare=False,
    )
    # Orchestrator debit cadence in seconds, from the payment challenge's
    # payment_interval_ms. None when the orchestrator does not report it.
    server_payment_interval: Optional[float] = None


@dataclass
class LiveRunnerCallStream:
    """A streaming live-runner response (SSE / chunked).

    Returned by ``call_runner(..., stream=True)``. It owns the underlying HTTP session
    and response, so use it as an async context manager (or call ``aclose()``) to
    release the connection.
    """

    status: int
    headers: Mapping[str, str]
    runner_url: str
    runner: Optional[LiveRunnerInstance]
    payment_session: Optional[LivePaymentSession]
    # Session backing this single-shot request, from the payment challenge's
    # manifest_id. Empty offchain: a stream body cannot supply the JSON-path
    # session_id fallback.
    session_id: str = ""
    # Orchestrator debit cadence in seconds, from the payment challenge's
    # payment_interval_ms. None when the orchestrator does not report it.
    server_payment_interval: Optional[float] = None
    _session: aiohttp.ClientSession = field(repr=False, compare=False, kw_only=True)
    _response: aiohttp.ClientResponse = field(repr=False, compare=False, kw_only=True)

    @property
    def content_type(self) -> str:
        return self._response.content_type

    async def aiter_bytes(self) -> AsyncIterator[bytes]:
        async for chunk in self._response.content.iter_any():
            yield chunk

    async def aiter_lines(self) -> AsyncIterator[str]:
        async for line in self._response.content:
            yield line.decode(errors="replace").rstrip("\n")

    async def aclose(self) -> None:
        self._response.release()
        await self._session.close()

    async def __aenter__(self) -> LiveRunnerCallStream:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


@dataclass(frozen=True)
class LiveRunnerGPU:
    id: str = ""
    name: str = ""
    vram_mb: int = 0

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.id:
            data["id"] = self.id
        if self.name:
            data["name"] = self.name
        if self.vram_mb > 0:
            data["vram_mb"] = self.vram_mb
        return data


@dataclass(frozen=True)
class LiveRunnerPriceInfo:
    price: int | float | str
    currency: str = "usd"
    unit: str = "hour"

    def to_json(self) -> dict[str, Any]:
        return {
            "price": self.price,
            "currency": str(self.currency or "usd").strip().lower(),
            "unit": str(self.unit or "hour").strip().lower(),
        }


class LiveRunnerRegistration:
    def __init__(
        self,
        *,
        orchestrator_url: str,
        secret: str,
        runner_url: str,
        app: str,
        price_info: LiveRunnerPriceInfo,
        runner_id: str = "",
        mode: str = "persistent",
        proxy: bool = False,
        label: str = "",
        version: str = "",
        metadata: str = "",
        status: str = "ready",
        capacity: int = 1,
        gpu: Optional[LiveRunnerGPU] = None,
        timeout: float = 5.0,
        heartbeat_interval_s: Optional[float] = None,
        unregister_on_close: bool = True,
        on_session_reserve: Optional[LiveRunnerSessionCallback] = None,
        on_session_release: Optional[LiveRunnerSessionCallback] = None,
    ) -> None:
        self.orchestrator_url = _normalize_http_base(orchestrator_url)
        self.runner_id = runner_id
        self.heartbeat_interval_s = heartbeat_interval_s or _DEFAULT_HEARTBEAT_INTERVAL_S
        self.heartbeat_ttl_s: Optional[float] = None

        self._bootstrap_secret = secret
        self._heartbeat_secret: Optional[str] = None
        self._runner_url = runner_url
        self._app = app
        self._mode = _normalize_runner_mode(mode)
        self._proxy = proxy
        self._price_info = price_info
        self._label = label
        self._version = version
        self._metadata = metadata
        self._status = status
        self._capacity = capacity
        self._gpu = gpu
        self._timeout = timeout
        self._heartbeat_interval_override = heartbeat_interval_s
        self._unregister_on_close = unregister_on_close
        self._on_session_reserve = on_session_reserve
        self._on_session_release = on_session_release
        self._active_session_ids: list[str] = []
        self.o2r_channel: Optional[LiveRunnerTrickleChannel] = None
        self._o2r_reader: Optional[ChannelReader] = None
        self._closed = False
        self._task: Optional[asyncio.Task[None]] = None
        self._o2r_task: Optional[asyncio.Task[None]] = None

    async def start(self) -> LiveRunnerRegistration:
        await self._send_heartbeat()
        self._task = asyncio.create_task(self._heartbeat_loop())
        return self

    @property
    def active_session_ids(self) -> tuple[str, ...]:
        # Return an immutable snapshot; internal storage stays list-backed to preserve reservation order.
        return tuple(self._active_session_ids)

    async def close(self) -> None:
        self._closed = True
        o2r_reader = self._o2r_reader
        self._o2r_reader = None
        self._o2r_task = None
        if o2r_reader is not None:
            try:
                await o2r_reader.close(wait_callback=False)
            except asyncio.CancelledError:
                raise
            except Exception:
                _LOG.exception("Live runner O2R reader failed during shutdown")

        task = self._task
        self._task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                _LOG.exception("Live runner heartbeat task failed during shutdown")

        if self._unregister_on_close and self.runner_id:
            secret = self._heartbeat_secret
            if not secret:
                _LOG.warning("Skipping live runner unregister without heartbeat secret")
                return
            try:
                await post_empty(
                    _join_endpoint(self.orchestrator_url, f"/runners/{quote(self.runner_id, safe='')}/unregister"),
                    headers={"Authorization": secret},
                    timeout=self._timeout,
                )
            except Exception:
                _LOG.debug("Live runner unregister failed", exc_info=True)

    async def __aenter__(self) -> LiveRunnerRegistration:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.close()

    async def create_trickle_channels(
        self,
        session: str | LiveRunnerSessionRequest,
        channels: list[LiveRunnerTrickleChannelRequest],
        *,
        session_token: str = "",
    ) -> list[LiveRunnerTrickleChannel]:
        """Create channels for a live runner app session.

        This is intended for apps running behind the orchestrator's live-runner
        proxy, not end-user clients. Apps should normally pass the incoming
        request so the orchestrator-provided session headers are used.
        """
        return await create_trickle_channels(
            session,
            channels,
            orchestrator_url=self.orchestrator_url,
            runner_id=self.runner_id,
            session_token=session_token,
            timeout=self._timeout,
        )

    async def remove_trickle_channels(
        self,
        session: str | LiveRunnerSessionRequest,
        channels: list[str],
        *,
        session_token: str = "",
    ) -> list[str]:
        """Remove channels for a live runner app session.

        This is intended for apps running behind the orchestrator's live-runner
        proxy, not end-user clients. Apps should normally pass the incoming
        request so the orchestrator-provided session headers are used.
        """
        return await remove_trickle_channels(
            session,
            channels,
            orchestrator_url=self.orchestrator_url,
            runner_id=self.runner_id,
            session_token=session_token,
            timeout=self._timeout,
        )

    async def create_proxy(
        self,
        session: str | LiveRunnerSessionRequest,
        target_url: str | None = None,
        *,
        session_token: str = "",
        timeout: Optional[float] = None,
    ) -> LiveRunnerProxy:
        """Create a public proxy URL for a live runner app session or target."""
        return await create_proxy(
            session,
            target_url,
            orchestrator_url=self.orchestrator_url,
            runner_id=self.runner_id,
            session_token=session_token,
            timeout=self._timeout if timeout is None else timeout,
        )

    def _payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "runner_url": self._runner_url,
            "app": self._app,
            "mode": self._mode,
            "capacity": self._capacity,
            "price_info": self._price_info.to_json(),
            "session_ids": list(self.active_session_ids),
        }
        if self.runner_id:
            payload["runner_id"] = self.runner_id
        if self._proxy:
            payload["proxy"] = True
        if self._label:
            payload["label"] = self._label
        if self._version:
            payload["version"] = self._version
        if self._metadata:
            payload["metadata"] = self._metadata
        if self._status:
            payload["status"] = self._status
        if self._gpu is not None:
            gpu = self._gpu.to_json()
            if gpu:
                payload["gpu"] = gpu
        return payload

    async def _heartbeat_loop(self) -> None:
        while not self._closed:
            await asyncio.sleep(self.heartbeat_interval_s)
            if self._closed:
                return
            try:
                await self._send_heartbeat()
            except LivepeerGatewayError as exc:
                _LOG.warning("Live runner heartbeat failed; retrying on next interval: %s", exc)
            except Exception:
                _LOG.warning("Live runner heartbeat failed; retrying on next interval", exc_info=True)

    async def _send_heartbeat(self) -> None:
        is_initial_heartbeat = self._heartbeat_secret is None
        auth = self._heartbeat_secret or self._bootstrap_secret
        request_orchestrator_url = self.orchestrator_url
        if is_initial_heartbeat:
            _LOG.info(
                "Registering live runner with orchestrator %s",
                request_orchestrator_url,
            )
        try:
            data = await self._post_heartbeat(auth)
        except LivepeerGatewayError as exc:
            if is_initial_heartbeat or not _is_invalid_authorization_error(exc):
                raise
            _LOG.info("Live runner heartbeat authorization expired; resetting heartbeat auth")
            self._heartbeat_secret = None
            is_initial_heartbeat = True
            data = await self._post_heartbeat(self._bootstrap_secret)

        runner_id = data.get("runner_id")
        if not isinstance(runner_id, str) or not runner_id.strip():
            raise LivepeerGatewayError("Live runner heartbeat response missing runner_id")
        self.runner_id = runner_id.strip()

        orchestrator = data.get("orchestrator")
        if isinstance(orchestrator, str) and orchestrator.strip():
            self.orchestrator_url = _normalize_http_base(orchestrator)
        if is_initial_heartbeat:
            if self.orchestrator_url != request_orchestrator_url:
                _LOG.info(
                    "Live runner registration using orchestrator %s returned by %s",
                    self.orchestrator_url,
                    request_orchestrator_url,
                )
            else:
                _LOG.info(
                    "Live runner registration using orchestrator %s",
                    self.orchestrator_url,
                )

        if self._heartbeat_interval_override is None:
            self.heartbeat_interval_s = _parse_go_duration_s(
                data.get("heartbeat_interval"),
                default=_DEFAULT_HEARTBEAT_INTERVAL_S,
            )
        self.heartbeat_ttl_s = _parse_go_duration_s(data.get("heartbeat_ttl"), default=None)

        heartbeat_secret = data.get("heartbeat_secret")
        if isinstance(heartbeat_secret, str) and heartbeat_secret.strip():
            self._heartbeat_secret = heartbeat_secret.strip()
        elif is_initial_heartbeat:
            raise LivepeerGatewayError("Live runner heartbeat response missing heartbeat_secret")

        if is_initial_heartbeat:
            self._start_o2r(data.get("o2r"))

    async def _post_heartbeat(self, auth: str) -> dict[str, Any]:
        return await post_json(
            _join_endpoint(self.orchestrator_url, "/runners/heartbeat"),
            self._payload(),
            headers={"Authorization": auth},
            timeout=self._timeout,
        )

    def _start_o2r(self, value: object) -> None:
        if self._closed or self._o2r_reader is not None:
            return
        if not _is_trickle_channel_response(value):
            if value is not None:
                _LOG.warning("Ignoring malformed live runner O2R channel: %r", value)
            return
        channel = cast(LiveRunnerTrickleChannel, value)
        url = channel.get("url", "").strip()
        if not url:
            return
        self.o2r_channel = channel
        reader = ChannelReader(url, start_seq=0, on_event=self._handle_o2r_message)
        self._o2r_reader = reader
        self._o2r_task = reader.callback_task()

    async def _handle_o2r_message(self, message: dict[str, Any]) -> None:
        if message.get("keep") == "alive":
            return

        event = message.get("event")
        session_id = message.get("session")
        if event not in ("reserved", "released") or not isinstance(session_id, str) or not session_id.strip():
            _LOG.warning("Ignoring unknown live runner O2R message: %r", message)
            return

        session_id = session_id.strip()
        typed_event = cast(Literal["reserved", "released"], event)
        if typed_event == "reserved":
            self._reserve_session_id(session_id)
        else:
            self._release_session_id(session_id)

        timestamp = message.get("timestamp")
        callback = self._on_session_reserve if typed_event == "reserved" else self._on_session_release
        if callback is None:
            return

        session_event = LiveRunnerSessionEvent(
            session_id=session_id,
            event=typed_event,
            timestamp=timestamp if isinstance(timestamp, str) else None,
            raw=message,
        )
        try:
            result = callback(session_event)
            if inspect.isawaitable(result):
                await result
        except Exception:
            _LOG.exception("Live runner %s callback failed for session %s", typed_event, session_id)

    def _reserve_session_id(self, session_id: str) -> None:
        if session_id not in self._active_session_ids:
            self._active_session_ids.append(session_id)

    def _release_session_id(self, session_id: str) -> None:
        try:
            self._active_session_ids.remove(session_id)
        except ValueError:
            pass


async def register_runner(
    orchestrator_url: str,
    *,
    secret: str,
    runner_url: str,
    app: str,
    price: int | float | str = 0,
    currency: str = "usd",
    unit: str = "hour",
    runner_id: str = "",
    mode: str = "persistent",
    proxy: bool = False,
    label: str = "",
    version: str = "",
    metadata: str = "",
    status: str = "ready",
    capacity: int = 1,
    gpu: Optional[LiveRunnerGPU] = None,
    auto_detect_gpu: bool = True,
    timeout: float = 5.0,
    heartbeat_interval_s: Optional[float] = None,
    unregister_on_close: bool = True,
    on_session_reserve: Optional[LiveRunnerSessionCallback] = None,
    on_session_release: Optional[LiveRunnerSessionCallback] = None,
) -> LiveRunnerRegistration:
    if gpu is None and auto_detect_gpu:
        gpu = detect_process_gpu()

    registration = LiveRunnerRegistration(
        orchestrator_url=orchestrator_url,
        secret=secret,
        runner_url=runner_url,
        app=app,
        price_info=LiveRunnerPriceInfo(price, currency, unit),
        runner_id=runner_id,
        mode=mode,
        proxy=proxy,
        label=label,
        version=version,
        metadata=metadata,
        status=status,
        capacity=capacity,
        gpu=gpu,
        timeout=timeout,
        heartbeat_interval_s=heartbeat_interval_s,
        unregister_on_close=unregister_on_close,
        on_session_reserve=on_session_reserve,
        on_session_release=on_session_release,
    )
    return await registration.start()


async def create_trickle_channels(
    session: str | LiveRunnerSessionRequest,
    channels: list[LiveRunnerTrickleChannelRequest],
    *,
    orchestrator_url: str = "",
    runner_id: str = "",
    session_token: str = "",
    timeout: float = 5.0,
) -> list[LiveRunnerTrickleChannel]:
    """Create trickle channels for a live runner app session."""
    runner, session_id, token, control_url = _resolve_session_credentials(
        session,
        runner_id=runner_id,
        session_token=session_token,
    )
    _validate_trickle_channel_requests(channels)
    data = await post_json(
        _trickle_channels_endpoint(orchestrator_url, runner, session_id, control_url),
        {"channels": channels},
        headers={"Livepeer-Session-Token": token},
        timeout=timeout,
    )
    response_channels = data.get("channels")
    if not isinstance(response_channels, list) or not all(
        _is_trickle_channel_response(channel) for channel in response_channels
    ):
        raise LivepeerGatewayError("Live runner trickle channel create response missing channels")
    return cast(list[LiveRunnerTrickleChannel], response_channels)


async def remove_trickle_channels(
    session: str | LiveRunnerSessionRequest,
    channels: list[str],
    *,
    orchestrator_url: str = "",
    runner_id: str = "",
    session_token: str = "",
    timeout: float = 5.0,
) -> list[str]:
    """Remove trickle channels for a live runner app session."""
    runner, session_id, token, control_url = _resolve_session_credentials(
        session,
        runner_id=runner_id,
        session_token=session_token,
    )
    data = await request_json(
        _trickle_channels_endpoint(orchestrator_url, runner, session_id, control_url),
        method="DELETE",
        payload={"channels": channels},
        headers={"Livepeer-Session-Token": token},
        timeout=timeout,
    )
    if not isinstance(data, dict):
        raise LivepeerGatewayError(
            f"Live runner trickle channel remove expected JSON object, got {type(data).__name__}"
        )
    deleted = data.get("deleted")
    if not isinstance(deleted, list) or not all(isinstance(channel, str) for channel in deleted):
        raise LivepeerGatewayError("Live runner trickle channel remove response missing deleted")
    return deleted


async def create_proxy(
    session: str | LiveRunnerSessionRequest,
    target_url: str | None = None,
    *,
    orchestrator_url: str = "",
    runner_id: str = "",
    session_token: str = "",
    timeout: float = 5.0,
) -> LiveRunnerProxy:
    """Create a public proxy URL for a live runner app session or target."""
    runner, session_id, token, control_url = _resolve_session_credentials(
        session,
        runner_id=runner_id,
        session_token=session_token,
        request_name="proxy",
    )
    payload: dict[str, str] = {}
    if target_url is not None:
        if not isinstance(target_url, str):
            raise LivepeerGatewayError("Live runner proxy request target_url must be a string")
        if target_url.strip():
            payload["target_url"] = target_url.strip()
    data = await post_json(
        _session_proxy_endpoint(orchestrator_url, runner, session_id, control_url),
        payload,
        headers={"Livepeer-Session-Token": token},
        timeout=timeout,
    )
    if not isinstance(data, dict):
        raise LivepeerGatewayError(
            f"Live runner proxy create expected JSON object, got {type(data).__name__}"
        )
    proxy_id = data.get("proxy_id")
    proxy_url = data.get("url")
    if not isinstance(proxy_id, str) or not proxy_id.strip() or not isinstance(proxy_url, str) or not proxy_url.strip():
        raise LivepeerGatewayError("Live runner proxy create response missing proxy_id or url")
    return LiveRunnerProxy(proxy_id=proxy_id.strip(), url=proxy_url.strip())


@overload
async def call_runner(
    runner_url: str = ...,
    *,
    runner: Optional[LiveRunnerInstance] = ...,
    payload: Optional[dict[str, Any]] = ...,
    method: str = ...,
    signer_url: Optional[str] = ...,
    signer_headers: Optional[dict[str, str]] = ...,
    payment_unit: Optional[str] = ...,
    timeout: float = ...,
    max_payment_challenge_retries: int = ...,
    stream: Literal[False] = False,
) -> LiveRunnerCallResult: ...


@overload
async def call_runner(
    runner_url: str = ...,
    *,
    runner: Optional[LiveRunnerInstance] = ...,
    payload: Optional[dict[str, Any]] = ...,
    method: str = ...,
    signer_url: Optional[str] = ...,
    signer_headers: Optional[dict[str, str]] = ...,
    payment_unit: Optional[str] = ...,
    timeout: float = ...,
    max_payment_challenge_retries: int = ...,
    stream: Literal[True],
) -> LiveRunnerCallStream: ...


async def call_runner(
    runner_url: str = "",
    *,
    runner: Optional[LiveRunnerInstance] = None,
    payload: Optional[dict[str, Any]] = None,
    method: str = "POST",
    signer_url: Optional[str] = None,
    signer_headers: Optional[dict[str, str]] = None,
    payment_unit: Optional[str] = None,
    timeout: float = 5.0,
    max_payment_challenge_retries: int = 3,
    stream: bool = False,
) -> LiveRunnerCallResult | LiveRunnerCallStream:
    """Call a runner once and return its result (or a live stream if ``stream=True``).

    With ``signer_url`` set, payment is automatic and **per call**: a 402 challenge is
    paid via the signer and retried (up to ``max_payment_challenge_retries``), one job,
    one upfront payment. Raises ``LivepeerHTTPError`` on non-402 errors.
    """
    runner_url = runner_url.strip() or (runner.url.strip() if runner is not None else "")
    if not runner_url:
        raise LivepeerGatewayError("Live runner call requires runner_url")
    request_payload = payload or {}
    payer_address = ""
    if signer_url:
        signer = await get_signer_info(signer_url, _freeze_headers(signer_headers))
        payer_address = cast(str, signer.address)
    challenge: Optional[_RunnerPaymentChallenge] = None
    attempts = (max(0, int(max_payment_challenge_retries)) + 1) * 2
    for attempt in range(attempts):
        payment_session: Optional[LivePaymentSession] = None
        payment_type = ""
        session_id = ""
        request_headers: dict[str, str] = {}
        if signer_url:
            request_headers[_LIVE_RUNNER_PAYER_ADDRESS_HEADER] = payer_address
        # Pending challenge means payment is needed.
        if challenge is not None:
            try:
                payment_type = _runner_payment_type(runner, payment_unit)
                payment_session, payment = await _get_runner_payment(
                    challenge,
                    payment_type=payment_type,
                    signer_url=signer_url or "",
                    signer_headers=signer_headers,
                )
            except SignerRefreshRequired as e:
                if attempt + 1 >= attempts:
                    raise
                # Could happen if embedded payment params expire; just retry in this case.
                _LOG.info(
                    "Live runner reservation payment challenge needs refresh; retrying with a fresh challenge: %s",
                    e,
                )
                challenge = None
                continue
            request_headers["Livepeer-Payment"] = payment.payment
            request_headers["Livepeer-Segment"] = payment.seg_creds or ""
            session_id = challenge.manifest_id

        try:
            request_kwargs: dict[str, Any] = {"timeout": timeout}
            if request_headers:
                request_kwargs["headers"] = request_headers

            if stream:
                # Hand back the live response unbuffered. open_stream raises on a 402
                # before any body, so the payment retry below still catches it.
                session, resp = await open_stream(
                    runner_url,
                    method=method,
                    payload=request_payload,
                    headers=request_headers or None,
                )
                return LiveRunnerCallStream(
                    status=resp.status,
                    headers=resp.headers,
                    runner_url=runner_url,
                    runner=runner,
                    payment_session=None if payment_type == "fixed" else payment_session,
                    session_id=session_id,
                    server_payment_interval=(
                        challenge.payment_interval_s if challenge is not None else None
                    ),
                    _session=session,
                    _response=resp,
                )

            data = await request_json(
                runner_url,
                method=method,
                payload=request_payload,
                **request_kwargs,
            )
            if not isinstance(data, dict):
                raise LivepeerGatewayError(
                    f"Live runner call expected JSON object, got {type(data).__name__}"
                )
            return LiveRunnerCallResult(
                data,
                runner_url=runner_url,
                runner=runner,
                session_id=(
                    session_id
                    or (data["session_id"].strip() if isinstance(data.get("session_id"), str) else "")
                ),
                payment_session=None if payment_type == "fixed" else payment_session,
                server_payment_interval=(
                    challenge.payment_interval_s if challenge is not None else None
                ),
            )
        except LivepeerHTTPError as e:
            if e.status_code != 402:
                raise
            if not signer_url:
                raise LivepeerGatewayError("Live runner paid call requires signer_url") from e
            challenge = _parse_runner_payment_challenge(e)
            continue

    raise LivepeerGatewayError("Live runner call exhausted payment challenge retries")


@dataclass(frozen=True)
class _RunnerPaymentChallenge:
    payment_params: str
    orchestrator_url: str
    manifest_id: str
    # Orchestrator debit cadence in seconds (payment_interval_ms); None when
    # the orchestrator does not report it.
    payment_interval_s: Optional[float] = None


def _parse_runner_payment_challenge(error: LivepeerHTTPError) -> _RunnerPaymentChallenge:
    try:
        data = json.loads(error.body)
    except json.JSONDecodeError as e:
        raise LivepeerGatewayError("Live runner payment challenge response was not valid JSON") from e
    if not isinstance(data, dict):
        raise LivepeerGatewayError("Live runner payment challenge response must be a JSON object")

    payment_params = data.get("payment_params")
    orchestrator_url = data.get("orchestrator")
    manifest_id = data.get("manifest_id")
    if not isinstance(payment_params, str) or not payment_params:
        raise LivepeerGatewayError("Live runner payment challenge missing payment_params")
    if not isinstance(orchestrator_url, str) or not orchestrator_url:
        raise LivepeerGatewayError("Live runner payment challenge missing orchestrator")
    if not isinstance(manifest_id, str) or not manifest_id:
        raise LivepeerGatewayError("Live runner payment challenge missing manifest_id")

    interval_ms = data.get("payment_interval_ms")
    payment_interval_s: Optional[float] = None
    if isinstance(interval_ms, (int, float)) and not isinstance(interval_ms, bool) and interval_ms > 0:
        payment_interval_s = float(interval_ms) / 1000.0

    return _RunnerPaymentChallenge(
        payment_params=payment_params,
        orchestrator_url=orchestrator_url,
        manifest_id=manifest_id,
        payment_interval_s=payment_interval_s,
    )


async def _get_runner_payment(
    challenge: _RunnerPaymentChallenge,
    *,
    payment_type: str,
    signer_url: str,
    signer_headers: Optional[dict[str, str]],
) -> tuple[LivePaymentSession, GetPaymentResponse]:
    session = LivePaymentSession(
        signer_url=signer_url,
        signer_headers=signer_headers,
        type=payment_type,
        payment_params=challenge.payment_params,
        manifest_id=challenge.manifest_id,
        orchestrator_url=challenge.orchestrator_url,
    )
    payment = await session.get_payment()
    if not payment.payment:
        raise LivepeerGatewayError("Live runner payment response missing payment")
    if not payment.seg_creds:
        raise LivepeerGatewayError("Live runner payment response missing segCreds")
    return session, payment


def _runner_payment_type(
    runner: Optional[LiveRunnerInstance],
    payment_unit: Optional[str] = None,
) -> str:
    # Discovery supplies price_info.unit; direct calls may supply payment_unit instead.
    # If both are present, they must agree.
    explicit_unit = str(payment_unit or "").strip().lower()
    discovered_unit = (
        str(runner.price_info.unit or "").strip().lower()
        if runner is not None and runner.price_info is not None
        else ""
    )
    if explicit_unit and discovered_unit and explicit_unit != discovered_unit:
        raise LivepeerGatewayError(
            "payment_unit conflicts with runner price metadata"
        )
    unit = discovered_unit or explicit_unit
    if unit:
        payment_type = _RUNNER_PAYMENT_TYPES_BY_UNIT.get(unit)
        if payment_type is None:
            supported = ", ".join(sorted(_RUNNER_PAYMENT_TYPES_BY_UNIT))
            raise LivepeerGatewayError(
                f"Unsupported live runner payment unit {unit!r}; expected one of {supported}"
            )
        return payment_type

    # Compatibility for callers constructing instances without discovery
    # price metadata. Discovery price_info.unit is authoritative when present.
    if runner is not None and runner.app == "live-video-to-video/scope":
        return "lv2v"
    return "live"


def _live_runner_price_info_from_json(value: object) -> Optional[LiveRunnerPriceInfo]:
    if not isinstance(value, dict):
        return None
    price = value.get("price")
    if isinstance(price, bool) or not isinstance(price, (int, float, str)):
        return None
    currency = value.get("currency")
    unit = value.get("unit")
    return LiveRunnerPriceInfo(
        price=price,
        currency=currency.strip().lower() if isinstance(currency, str) else "",
        unit=unit.strip().lower() if isinstance(unit, str) else "",
    )


def _live_runner_session_from_json(
    data: dict[str, Any],
    *,
    runner_url: str,
    runner: Optional[LiveRunnerInstance],
) -> LiveRunnerSession:
    session_id = data.get("session_id")
    app_url = data.get("app_url")
    if not isinstance(session_id, str) or not session_id.strip():
        raise LivepeerGatewayError("Live runner session reserve response missing session_id")
    if not isinstance(app_url, str) or not app_url.strip():
        raise LivepeerGatewayError("Live runner session reserve response missing app_url")
    return LiveRunnerSession(
        session_id=session_id.strip(),
        app_url=app_url.strip(),
        runner_url=runner_url,
        runner=runner,
    )

async def stop_runner_session(
    session: LiveRunnerSession | LiveRunnerSessionRequest,
    *,
    timeout: float = 5.0,
) -> None:
    request_headers: dict[str, str] = {}
    if isinstance(session, LiveRunnerSession):
        runner_url = session.runner_url.strip()
        session_id = session.session_id.strip()
        if not runner_url:
            raise LivepeerGatewayError("Live runner session stop requires runner_url")
        if not session_id:
            raise LivepeerGatewayError("Live runner session stop requires session_id")
        url = _join_endpoint(runner_url, f"/{quote(session_id, safe='')}/stop")
    else:
        headers = getattr(session, "headers", None)
        get = getattr(headers, "get", None)
        control_url = get("Livepeer-Session-Control", "") if callable(get) else ""
        token = get("Livepeer-Session-Token", "") if callable(get) else ""
        if not isinstance(control_url, str) or not control_url.strip():
            raise LivepeerGatewayError("Live runner session stop requires session_control")
        url = _join_endpoint(control_url, "stop")
        if isinstance(token, str) and token.strip():
            request_headers = {"Livepeer-Session-Token": token}
    await post_empty(
        url,
        headers=request_headers,
        timeout=timeout,
    )


def detect_process_gpu() -> Optional[LiveRunnerGPU]:
    for detector in (_detect_gpu_pynvml, _detect_gpu_torch, _detect_gpu_nvidia_smi):
        try:
            gpu = detector()
        except Exception:
            _LOG.debug("GPU auto-discovery detector failed: %s", detector.__name__, exc_info=True)
            continue
        if gpu is not None:
            return gpu
    return None


def _normalize_http_base(url: str) -> str:
    url = url.strip()
    normalized = url if "://" in url else f"https://{url}"
    parsed = urlparse(normalized)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise LivepeerGatewayError(f"Invalid orchestrator URL: {url!r}")
    path = parsed.path.rstrip("/")
    return urlunparse((parsed.scheme, parsed.netloc, path, "", parsed.query, ""))


def _join_endpoint(base_url: str, suffix: str) -> str:
    parsed = urlparse(_normalize_http_base(base_url))
    suffix_path = suffix if suffix.startswith("/") else f"/{suffix}"
    path = f"{parsed.path.rstrip('/')}{suffix_path}"
    return urlunparse((parsed.scheme, parsed.netloc, path, "", parsed.query, ""))


def _trickle_channels_endpoint(
    orchestrator_url: str,
    runner_id: str,
    session_id: str,
    control_url: str = "",
) -> str:
    if control_url:
        return _join_endpoint(control_url, "channels")
    if not orchestrator_url:
        raise LivepeerGatewayError("Live runner trickle channel request requires session_control")
    if not runner_id:
        raise LivepeerGatewayError("Live runner trickle channel request requires runner_id")
    return _join_endpoint(
        orchestrator_url,
        (
            f"/runner/{quote(runner_id, safe='')}"
            f"/session/{quote(session_id, safe='')}"
            "/channels"
        ),
    )


def _session_proxy_endpoint(
    orchestrator_url: str,
    runner_id: str,
    session_id: str,
    control_url: str = "",
) -> str:
    if control_url:
        return _join_endpoint(control_url, "proxy")
    if not orchestrator_url:
        raise LivepeerGatewayError("Live runner proxy request requires session_control")
    if not runner_id:
        raise LivepeerGatewayError("Live runner proxy request requires runner_id")
    return _join_endpoint(
        orchestrator_url,
        (
            f"/runner/{quote(runner_id, safe='')}"
            f"/session/{quote(session_id, safe='')}"
            "/proxy"
        ),
    )


def _parse_go_duration_s(value: object, *, default: Optional[float]) -> Optional[float]:
    if not isinstance(value, str) or not value.strip():
        return default
    match = _DURATION_RE.match(value)
    if not match:
        return default
    number = float(match.group("value"))
    unit = match.group("unit")
    scale = {
        "ns": 1e-9,
        "us": 1e-6,
        "\u00b5s": 1e-6,
        "ms": 1e-3,
        "s": 1.0,
        "m": 60.0,
        "h": 3600.0,
    }[unit]
    return number * scale


def _normalize_runner_mode(mode: str) -> str:
    normalized = mode.strip()
    if normalized not in _LIVE_RUNNER_MODES:
        raise ValueError(f"live runner mode must be one of {sorted(_LIVE_RUNNER_MODES)}")
    return normalized


def _is_invalid_authorization_error(exc: LivepeerGatewayError) -> bool:
    return isinstance(exc, LivepeerHTTPError) and exc.status_code == 401


def _resolve_session_credentials(
    session: str | LiveRunnerSessionRequest,
    *,
    runner_id: str = "",
    session_token: str = "",
    request_name: str = "trickle channel",
) -> tuple[str, str, str, str]:
    runner = runner_id.strip()
    session_id = ""
    token = session_token.strip()
    control_url = ""

    if isinstance(session, str):
        session_id = session.strip()
    else:
        headers = getattr(session, "headers", None)
        if headers is not None:
            get = getattr(headers, "get", None)
            if callable(get):
                runner_value = get("Livepeer-Runner-Route", "")
                session_id_value = get("Livepeer-Session-Id", "")
                token_value = get("Livepeer-Session-Token", "")
                control_value = get("Livepeer-Session-Control", "")
                if not runner and isinstance(runner_value, str):
                    runner = runner_value.strip()
                if isinstance(session_id_value, str):
                    session_id = session_id_value.strip()
                if not token and isinstance(token_value, str):
                    token = token_value.strip()
                if isinstance(control_value, str):
                    control_url = control_value.strip()

    if not session_id:
        raise LivepeerGatewayError(f"Live runner {request_name} request requires session_id")
    if not token:
        raise LivepeerGatewayError(f"Live runner {request_name} request requires session_token")
    return runner, session_id, token, control_url


def _validate_trickle_channel_requests(channels: list[LiveRunnerTrickleChannelRequest]) -> None:
    for channel in channels:
        if not isinstance(channel, dict):
            raise TypeError(f"trickle channel must be dict, got {type(channel).__name__}")
        if not isinstance(channel.get("name"), str):
            raise TypeError("trickle channel name must be str")
        if not isinstance(channel.get("mime_type"), str):
            raise TypeError("trickle channel mime_type must be str")


def _is_trickle_channel_response(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    return all(
        isinstance(value.get(key), str)
        for key in ("name", "channel_name", "url", "mime_type")
    ) and ("internal_url" not in value or isinstance(value.get("internal_url"), str))


def _detect_gpu_pynvml() -> Optional[LiveRunnerGPU]:
    try:
        import pynvml  # type: ignore[import-not-found]
    except Exception:
        return None

    pynvml.nvmlInit()
    try:
        index = _pynvml_process_device_index(pynvml)
        if index is None:
            index = _first_visible_cuda_index()
        if index is None:
            return None
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        uuid = _decode_maybe_bytes(pynvml.nvmlDeviceGetUUID(handle))
        name = _decode_maybe_bytes(pynvml.nvmlDeviceGetName(handle))
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return LiveRunnerGPU(id=uuid, name=name, vram_mb=int(getattr(mem, "total", 0)) // (1024 * 1024))
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _pynvml_process_device_index(pynvml: Any) -> Optional[int]:
    pid = os.getpid()
    count = int(pynvml.nvmlDeviceGetCount())
    for index in range(count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        processes: list[Any] = []
        for name in ("nvmlDeviceGetComputeRunningProcesses_v2", "nvmlDeviceGetComputeRunningProcesses"):
            fn = getattr(pynvml, name, None)
            if fn is None:
                continue
            try:
                processes = list(fn(handle))
                break
            except Exception:
                continue
        if any(int(getattr(proc, "pid", -1)) == pid for proc in processes):
            return index
    return None


def _detect_gpu_torch() -> Optional[LiveRunnerGPU]:
    try:
        import torch  # type: ignore[import-not-found]
    except Exception:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        index = int(torch.cuda.current_device())
        props = torch.cuda.get_device_properties(index)
        name = str(getattr(props, "name", "") or torch.cuda.get_device_name(index))
        total = int(getattr(props, "total_memory", 0) or 0)
        return LiveRunnerGPU(id=str(index), name=name, vram_mb=total // (1024 * 1024))
    except Exception:
        _LOG.debug("torch.cuda GPU discovery failed", exc_info=True)
        return None


def _detect_gpu_nvidia_smi() -> Optional[LiveRunnerGPU]:
    if shutil.which("nvidia-smi") is None:
        return None
    uuid = _nvidia_smi_process_gpu_uuid()
    rows = _nvidia_smi_gpu_rows()
    if not rows:
        return None
    if uuid:
        for row in rows:
            if row.get("uuid") == uuid:
                return _gpu_from_nvidia_smi_row(row)
    index = _first_visible_cuda_index()
    if index is not None:
        for row in rows:
            if row.get("index") == str(index):
                return _gpu_from_nvidia_smi_row(row)
    return _gpu_from_nvidia_smi_row(rows[0])


def _nvidia_smi_process_gpu_uuid() -> str:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        )
    except Exception:
        return ""
    pid = str(os.getpid())
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2 and parts[0] == pid:
            return parts[1]
    return ""


def _nvidia_smi_gpu_rows() -> list[dict[str, str]]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        )
    except Exception:
        return []
    rows = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",", maxsplit=3)]
        if len(parts) != 4:
            continue
        rows.append({"index": parts[0], "uuid": parts[1], "name": parts[2], "vram_mb": parts[3]})
    return rows


def _gpu_from_nvidia_smi_row(row: dict[str, str]) -> LiveRunnerGPU:
    try:
        vram_mb = int(float(row.get("vram_mb", "0")))
    except ValueError:
        vram_mb = 0
    return LiveRunnerGPU(id=row.get("uuid", ""), name=row.get("name", ""), vram_mb=vram_mb)


def _first_visible_cuda_index() -> Optional[int]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return 0
    first = visible.split(",")[0].strip()
    if not first or first == "-1":
        return None
    if first.isdigit():
        return int(first)
    return 0


def _decode_maybe_bytes(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value or "")
