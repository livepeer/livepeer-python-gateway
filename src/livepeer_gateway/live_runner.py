"""Live runner discovery, registration, sessions, trickle channels, and calls."""

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
from typing import Any, Awaitable, Callable, Literal, NotRequired, Optional, Protocol, TypedDict, cast
from urllib.parse import quote, urlparse, urlunparse

import aiohttp

from .channel_reader import ChannelReader
from .errors import LivepeerGatewayError, LivepeerHTTPError, SignerRefreshRequired
from .http import post_json, request_json
from .remote_signer import (
    GetPaymentResponse,
    LivePaymentSession,
    _freeze_headers,
    _payment_type_for_signer,
    get_signer_info,
)

_LOG = logging.getLogger(__name__)

_DEFAULT_HEARTBEAT_INTERVAL_S = 5.0
_LIVE_RUNNER_PAYER_ADDRESS_HEADER = "Livepeer-Payer-Address"
_LIVE_RUNNER_MODES = frozenset({"persistent", "single-shot"})

# golang format duration, eg "10s"
_DURATION_RE = re.compile(r"^\s*(?P<value>[0-9]+(?:\.[0-9]+)?)(?P<unit>ns|us|\u00b5s|ms|s|m|h)\s*$")


class LiveRunnerTrickleChannelRequest(TypedDict):
    """A trickle channel to create.

    Attributes:
        name: Caller-chosen channel name, unique within the session.
        mime_type: MIME type of the media carried on the channel.
    """

    name: str
    mime_type: str


class LiveRunnerTrickleChannel(TypedDict):
    """A trickle channel created on a live runner session.

    Attributes:
        name: The channel name requested by the caller.
        channel_name: The orchestrator-assigned channel identifier.
        url: Public/external trickle URL.
        internal_url: Private-network runner-to-orchestrator URL. When the runner
            and orchestrator share a network (e.g. Docker), this bypasses public
            TLS/routing; present only when the orchestrator is configured to
            return one.
        mime_type: MIME type of the media carried on the channel.
    """

    name: str
    channel_name: str
    url: str
    internal_url: NotRequired[str]
    mime_type: str


class LiveRunnerSessionHeaders(Protocol):
    """Mapping-like session headers exposing a ``get(key, default)`` accessor."""

    def get(self, key: str, default: str = "") -> str:
        """Return the value for ``key``, or ``default`` if absent."""
        ...


class LiveRunnerSessionRequest(Protocol):
    """An incoming request carrying the orchestrator's live-runner session headers.

    Any object exposing a ``headers`` mapping (e.g. a web framework request)
    satisfies this protocol.

    Attributes:
        headers: Session headers set by the orchestrator proxy, such as
            ``Livepeer-Session-Id``, ``Livepeer-Session-Token``,
            ``Livepeer-Runner-Route``, and ``Livepeer-Session-Control``.
    """

    headers: LiveRunnerSessionHeaders


@dataclass(frozen=True)
class LiveRunnerSessionEvent:
    """A session reservation lifecycle event delivered over the O2R channel.

    Attributes:
        session_id: The session the event concerns.
        event: Whether the session was ``"reserved"`` or ``"released"``.
        timestamp: Orchestrator-supplied event timestamp, if any.
        raw: The raw event message as received.
    """

    session_id: str
    event: Literal["reserved", "released"]
    timestamp: Optional[str]
    raw: dict[str, Any]


LiveRunnerSessionCallback = Callable[[LiveRunnerSessionEvent], None | Awaitable[None]]


@dataclass(frozen=True)
class LiveRunnerInstance:
    """A normalized live runner discovered from an orchestrator entry.

    Attributes:
        url: Runner endpoint used to call the app.
        app: App the runner serves.
        runner_id: Orchestrator-assigned runner identifier.
        mode: Runner mode, e.g. ``"persistent"`` or ``"single-shot"``.
        orchestrator_url: URL of the orchestrator advertising the runner.
        raw: The raw runner entry as received from discovery.
    """

    url: str
    app: str
    runner_id: str
    mode: str
    orchestrator_url: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class LiveRunnerSession:
    """A reserved session on a live runner.

    Attributes:
        session_id: Identifier of the reserved session.
        app_url: URL where the app serves this session. Clients send their session
            requests here (e.g. ``{app_url}/echo``).
        runner_url: Base URL of the runner hosting the session, used to manage the
            session lifecycle such as stopping it.
        runner: The runner instance backing the session, when known.
    """

    session_id: str
    app_url: str
    runner_url: str
    runner: Optional[LiveRunnerInstance] = None


@dataclass(frozen=True)
class LiveRunnerProxy:
    """A public proxy exposing a runner-served URL to the public internet.

    The orchestrator maps a public URL onto a target URL the runner serves, forwarding
    inbound traffic to it.

    Attributes:
        proxy_id: Identifier of the created proxy.
        url: Public URL that forwards to the runner-served target.
    """

    proxy_id: str
    url: str


@dataclass(frozen=True)
class LiveRunnerCallResult:
    """The result of a call to a live runner.

    Attributes:
        data: The runner's JSON response body.
        runner_url: URL of the runner that produced the response.
        runner: The runner instance that was called, when known.
        session_id: Session identifier associated with the call, if any.
        payment_session: Payment session used to settle a paid call. ``None`` for
            unpaid calls. Excluded from ``repr`` and equality.
    """

    data: dict[str, Any]
    runner_url: str
    runner: Optional[LiveRunnerInstance] = None
    session_id: str = ""
    payment_session: Optional[LivePaymentSession] = field(
        default=None,
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class LiveRunnerGPU:
    """GPU details advertised when registering a runner.

    Attributes:
        id: GPU identifier (e.g. device index or UUID).
        name: Human-readable GPU model name.
        vram_mb: Video memory in megabytes.
    """

    id: str = ""
    name: str = ""
    vram_mb: int = 0

    def to_json(self) -> dict[str, Any]:
        """Return the JSON payload, omitting empty/zero fields."""
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
    """Pricing advertised for a runner's work.

    Attributes:
        price: Amount charged per ``unit``, in ``currency`` (a number or numeric
            string).
        currency: Currency of ``price`` (e.g. ``"usd"``).
        unit: Billing unit that ``price`` applies to (e.g. ``"hour"``).
    """

    price: int | float | str
    currency: str = "usd"
    unit: str = "hour"

    def to_json(self) -> dict[str, Any]:
        """Return the pricing as a JSON payload."""
        return {
            "price": self.price,
            "currency": str(self.currency or "usd").strip().lower(),
            "unit": str(self.unit or "hour").strip().lower(),
        }


class LiveRunnerRegistration:
    """A live runner's registration with an orchestrator.

    Keeps the runner registered by sending periodic heartbeats and tracks the
    sessions the orchestrator has reserved on it. Prefer ``register_runner``
    to construct and start one, and use it as an async context manager to
    unregister on exit. Also exposes session-scoped helpers
    (``create_trickle_channels``, ``remove_trickle_channels``, ``create_proxy``)
    pre-bound to this registration's orchestrator and runner id.

    Attributes:
        orchestrator_url: Orchestrator the runner is registered with. May be updated to
            the orchestrator returned by the initial heartbeat.
        runner_id: Orchestrator-assigned runner id, populated after the first heartbeat.
        heartbeat_interval_s: How often the runner sends heartbeats, in seconds.
        heartbeat_ttl_s: How long the orchestrator keeps the runner registered without
            a heartbeat, in seconds, if advertised.
        o2r_channel: Orchestrator-to-runner control channel, when established.
    """

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
        label: str = "",
        version: str = "",
        status: str = "ready",
        capacity: int = 1,
        gpu: Optional[LiveRunnerGPU] = None,
        timeout: float = 5.0,
        heartbeat_interval_s: Optional[float] = None,
        unregister_on_close: bool = True,
        on_session_reserve: Optional[LiveRunnerSessionCallback] = None,
        on_session_release: Optional[LiveRunnerSessionCallback] = None,
    ) -> None:
        """Construct a registration without starting it.

        Prefer ``register_runner``, which assembles ``price_info``,
        optionally auto-detects the GPU, and starts heartbeating. When
        constructing directly, call ``start`` (or use the instance as an
        async context manager) before the runner is registered.

        Args:
            orchestrator_url: Orchestrator to register with.
            secret: Bootstrap secret authorizing the initial registration.
            runner_url: URL at which this runner serves the app.
            app: App this runner serves.
            price_info: Pricing advertised for the runner's work.
            runner_id: Preferred runner id. The orchestrator may assign one otherwise.
            mode: Runner mode, ``"persistent"`` or ``"single-shot"``.
            label: Optional human-readable label for the runner.
            version: Optional runner/app version string.
            status: Initial advertised status, e.g. ``"ready"``.
            capacity: Number of concurrent sessions the runner can serve.
            gpu: GPU details to advertise, if any.
            timeout: Per-request timeout, in seconds.
            heartbeat_interval_s: Override the heartbeat interval. Defaults to the
                orchestrator-advertised value.
            unregister_on_close: Unregister the runner when the registration closes.
            on_session_reserve: Callback invoked when a session is reserved.
            on_session_release: Callback invoked when a session is released.

        Raises:
            LivepeerGatewayError: If ``orchestrator_url`` is invalid.
            ValueError: If ``mode`` is not ``"persistent"`` or ``"single-shot"``.
        """
        self.orchestrator_url = _normalize_http_base(orchestrator_url)
        self.runner_id = runner_id
        self.heartbeat_interval_s = heartbeat_interval_s or _DEFAULT_HEARTBEAT_INTERVAL_S
        self.heartbeat_ttl_s: Optional[float] = None

        self._bootstrap_secret = secret
        self._heartbeat_secret: Optional[str] = None
        self._runner_url = runner_url
        self._app = app
        self._mode = _normalize_runner_mode(mode)
        self._price_info = price_info
        self._label = label
        self._version = version
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

    async def start(self) -> "LiveRunnerRegistration":
        """Send the initial heartbeat and start the background heartbeat loop.

        Returns:
            This registration, for chaining.

        Raises:
            LivepeerGatewayError: If the initial registration heartbeat fails.
        """
        await self._send_heartbeat()
        self._task = asyncio.create_task(self._heartbeat_loop())
        return self

    @property
    def active_session_ids(self) -> tuple[str, ...]:
        """Immutable snapshot of currently reserved session ids, in reservation order."""
        # Return an immutable snapshot; internal storage stays list-backed to preserve reservation order.
        return tuple(self._active_session_ids)

    async def close(self) -> None:
        """Stop heartbeating and, unless disabled, unregister the runner.

        Idempotent and safe to call from ``__aexit__``. Shutdown errors are logged
        rather than raised.
        """
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
                await _post_empty(
                    _join_endpoint(self.orchestrator_url, f"/runners/{quote(self.runner_id, safe='')}/unregister"),
                    {"Authorization": secret},
                    self._timeout,
                )
            except Exception:
                _LOG.debug("Live runner unregister failed", exc_info=True)

    async def __aenter__(self) -> "LiveRunnerRegistration":
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

        Args:
            session: The session to target: a ``session_id`` string, or a
                ``LiveRunnerSessionRequest`` from which the session id and token are 
                read.
            channels: Channels to create, each a dict with ``name`` and ``mime_type``.
            session_token: Session auth token. Overrides the value carried by 
                ``session``. Required when ``session`` is a plain id string.

        Returns:
            The created channels as returned by the runner.

        Raises:
            LivepeerGatewayError: If credentials are missing or the response omits the
                created channels.
            TypeError: If a channel entry is malformed.
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

        Args:
            session: The session to target: a ``session_id`` string, or a
                ``LiveRunnerSessionRequest`` from which the session id and token are 
                read.
            channels: Names of the channels to remove.
            session_token: Session auth token. Overrides the value carried by
                ``session``. Required when ``session`` is a plain id string.

        Returns:
            Names of the channels that were removed.

        Raises:
            LivepeerGatewayError: If credentials are missing, the response is not a 
                JSON object, or it omits the deleted channel list.
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
        target_url: str,
        *,
        session_token: str = "",
        timeout: Optional[float] = None,
    ) -> LiveRunnerProxy:
        """Create a public proxy URL for a live runner app target.

        This is intended for apps running behind the orchestrator's live-runner
        proxy, not end-user clients. Apps should normally pass the incoming
        request so the orchestrator-provided session headers are used.

        Args:
            session: The session to target: a ``session_id`` string, or a
                ``LiveRunnerSessionRequest`` from which the session id and token are
                read.
            target_url: The runner-served URL to expose through the proxy.
            session_token: Session auth token. Overrides the value carried by
                ``session``. Required when ``session`` is a plain id string.
            timeout: Per-request timeout, in seconds. Defaults to the registration's
                timeout.

        Returns:
            The created proxy, carrying its ``proxy_id`` and public ``url``.

        Raises:
            LivepeerGatewayError: If credentials or ``target_url`` are missing,
                the response is not a JSON object, or it omits ``proxy_id`` or
                ``url``.
        """
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
        if self._label:
            payload["label"] = self._label
        if self._version:
            payload["version"] = self._version
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
    label: str = "",
    version: str = "",
    status: str = "ready",
    capacity: int = 1,
    gpu: Optional[LiveRunnerGPU] = None,
    auto_detect_gpu: bool = True,
    timeout: float = 5.0,
    heartbeat_interval_s: Optional[float] = None,
    unregister_on_close: bool = True,
    on_session_reserve: Optional[LiveRunnerSessionCallback] = None,
    on_session_release: Optional[LiveRunnerSessionCallback] = None,
) -> "LiveRunnerRegistration":
    """Register a live runner with an orchestrator and start heartbeating.

    Args:
        orchestrator_url: Orchestrator to register with.
        secret: Bootstrap secret authorizing the initial registration.
        runner_url: URL at which this runner serves the app.
        app: App this runner serves.
        price: Amount charged per ``unit``, in ``currency`` (a number or numeric
            string).
        currency: Currency of ``price`` (e.g. ``"usd"``).
        unit: Billing unit that ``price`` applies to (e.g. ``"hour"``).
        runner_id: Preferred runner id. The orchestrator may assign one otherwise.
        mode: Runner mode, ``"persistent"`` or ``"single-shot"``.
        label: Optional human-readable label for the runner.
        version: Optional runner/app version string.
        status: Initial advertised status, e.g. ``"ready"``.
        capacity: Number of concurrent sessions the runner can serve.
        gpu: GPU details to advertise. Auto-detected when omitted and
            ``auto_detect_gpu`` is set.
        auto_detect_gpu: Detect local GPU details when ``gpu`` is not provided.
        timeout: Per-request timeout, in seconds.
        heartbeat_interval_s: Override the heartbeat interval. Defaults to the
            orchestrator-advertised value.
        unregister_on_close: Unregister the runner when the registration closes.
        on_session_reserve: Callback invoked when a session is reserved.
        on_session_release: Callback invoked when a session is released.

    Returns:
        A started ``LiveRunnerRegistration`` whose heartbeat loop is running.

    Raises:
        LivepeerGatewayError: If ``orchestrator_url`` is invalid or the initial
            registration heartbeat fails.
        ValueError: If ``mode`` is not ``"persistent"`` or ``"single-shot"``.
    """
    if gpu is None and auto_detect_gpu:
        gpu = _detect_process_gpu()

    registration = LiveRunnerRegistration(
        orchestrator_url=orchestrator_url,
        secret=secret,
        runner_url=runner_url,
        app=app,
        price_info=LiveRunnerPriceInfo(price, currency, unit),
        runner_id=runner_id,
        mode=mode,
        label=label,
        version=version,
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
    """Create trickle channels for a live runner app session.

    Args:
        session: The session to target: a ``session_id`` string, or a
            ``LiveRunnerSessionRequest`` from which the session id, token, runner route,
            and control URL are read.
        channels: Channels to create, each a dict with ``name`` and ``mime_type``.
        orchestrator_url: Orchestrator base URL, e.g. ``https://orch.example`` (the
            request path is appended, so pass no path). Falls back to the session's
            control URL when empty.
        runner_id: Runner route for the request. Overrides the value carried by
            ``session``. Required when ``session`` is a plain id string.
        session_token: Session auth token. Overrides the value carried by
            ``session``. Required when ``session`` is a plain id string.
        timeout: Per-request timeout, in seconds.

    Returns:
        The created channels as returned by the runner.

    Raises:
        LivepeerGatewayError: If credentials are missing or the response omits the
            created channels.
        TypeError: If a channel entry is malformed.
    """
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
    """Remove trickle channels for a live runner app session.

    Args:
        session: The session to target: a ``session_id`` string, or a
            ``LiveRunnerSessionRequest`` from which the session id, token, runner route,
            and control URL are read.
        channels: Names of the channels to remove.
        orchestrator_url: Orchestrator base URL, e.g. ``https://orch.example`` (the
            request path is appended, so pass no path). Falls back to the session's
            control URL when empty.
        runner_id: Runner route for the request. Overrides the value carried by
            ``session``. Required when ``session`` is a plain id string.
        session_token: Session auth token. Overrides the value carried by ``session``.
            Required when ``session`` is a plain id string.
        timeout: Per-request timeout, in seconds.

    Returns:
        Names of the channels that were removed.

    Raises:
        LivepeerGatewayError: If credentials are missing, the response is not a JSON
            object, or it omits the deleted channel list.
    """
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
    target_url: str,
    *,
    orchestrator_url: str = "",
    runner_id: str = "",
    session_token: str = "",
    timeout: float = 5.0,
) -> LiveRunnerProxy:
    """Create a public proxy URL for a target served by a live runner app session.

    Args:
        session: The session to target: a ``session_id`` string, or a
            ``LiveRunnerSessionRequest`` from which the session id, token, runner route,
            and control URL are read.
        target_url: The runner-served URL to expose through the proxy.
        orchestrator_url: Orchestrator base URL, e.g. ``https://orch.example`` (the
            request path is appended, so pass no path). Falls back to the session's
            control URL when empty.
        runner_id: Runner route for the request. Overrides the value carried by
            ``session``. Required when ``session`` is a plain id string.
        session_token: Session auth token. Overrides the value carried by
            ``session``. Required when ``session`` is a plain id string.
        timeout: Per-request timeout, in seconds.

    Returns:
        The created proxy, carrying its ``proxy_id`` and public ``url``.

    Raises:
        LivepeerGatewayError: If credentials or ``target_url`` are missing, the response
            is not a JSON object, or it omits ``proxy_id`` or ``url``.
    """
    runner, session_id, token, control_url = _resolve_session_credentials(
        session,
        runner_id=runner_id,
        session_token=session_token,
        request_name="proxy",
    )
    if not isinstance(target_url, str) or not target_url.strip():
        raise LivepeerGatewayError("Live runner proxy request requires target_url")
    data = await post_json(
        _session_proxy_endpoint(orchestrator_url, runner, session_id, control_url),
        {"target_url": target_url.strip()},
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


async def call_runner(
    runner_url: str = "",
    *,
    runner: Optional[LiveRunnerInstance] = None,
    payload: Optional[dict[str, Any]] = None,
    method: str = "POST",
    signer_url: Optional[str] = None,
    signer_headers: Optional[dict[str, str]] = None,
    timeout: float = 5.0,
    max_payment_challenge_retries: int = 3,
) -> LiveRunnerCallResult:
    """Send a request to a live runner, handling payment challenges transparently.

    If ``signer_url`` is set, a 402 payment challenge is answered and the request
    retried. Paid calls without a signer fail.

    Args:
        runner_url: Runner endpoint to call. Optional if ``runner`` is given.
        runner: Discovered runner instance to call. Its URL is used when ``runner_url``
            is empty.
        payload: JSON request body sent to the runner. Defaults to an empty body.
        method: HTTP method for the request.
        signer_url: Signer service used to answer 402 payment challenges. Required for
            paid runners.
        signer_headers: Extra HTTP headers sent to the signer service.
        timeout: Per-request timeout, in seconds.
        max_payment_challenge_retries: Maximum number of payment challenges to answer
            before giving up.

    Returns:
        A ``LiveRunnerCallResult`` wrapping the runner's JSON response.

    Raises:
        LivepeerGatewayError: If neither target is given, a paid call is made without
            ``signer_url``, the response is not a JSON object, or the payment-challenge
            retries are exhausted.
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
        session_id = ""
        request_headers: dict[str, str] = {}
        if signer_url:
            request_headers[_LIVE_RUNNER_PAYER_ADDRESS_HEADER] = payer_address
        # Pending challenge means payment is needed.
        if challenge is not None:
            try:
                payment_session, payment = await _get_runner_payment(
                    challenge,
                    runner=runner,
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
                payment_session=payment_session,
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

    return _RunnerPaymentChallenge(
        payment_params=payment_params,
        orchestrator_url=orchestrator_url,
        manifest_id=manifest_id,
    )


async def _get_runner_payment(
    challenge: _RunnerPaymentChallenge,
    *,
    runner: Optional[LiveRunnerInstance],
    signer_url: str,
    signer_headers: Optional[dict[str, str]],
) -> tuple[LivePaymentSession, GetPaymentResponse]:
    # Scope/LV2V is a live-video job → always "lv2v". Every other runner (single-shot
    # fal/tool caps) pays via the per-signer dual-path: legacy Daydream → "lv2v",
    # modern signers → "byoc". The old hardcoded "live" was rejected by the Daydream
    # signer ("invalid job type"), breaking LR payment on the default signer.
    payment_type = (
        "lv2v"
        if runner is not None and runner.app == "live-video-to-video/scope"
        else _payment_type_for_signer(signer_url)
    )
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
    """Stop a live runner session, releasing its reservation.

    Args:
        session: The session to stop: a ``LiveRunnerSession`` (stopped via its runner
            URL) or a ``LiveRunnerSessionRequest`` carrying the orchestrator's
            session-control headers.
        timeout: Per-request timeout, in seconds.

    Raises:
        LivepeerGatewayError: If the required runner URL, session id, or session control
            URL is missing, or the stop request fails.
    """
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
    await _post_empty(
        url,
        request_headers,
        timeout,
    )


def _detect_process_gpu() -> Optional[LiveRunnerGPU]:
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


async def _post_empty(url: str, headers: dict[str, str], timeout: float) -> None:
    try:
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(timeout=client_timeout, connector=connector) as session:
            async with session.post(url, data=b"", headers=headers) as resp:
                body = await resp.text()
                if resp.status >= 400:
                    raise LivepeerGatewayError(
                        f"HTTP empty POST error: HTTP {resp.status}; body={body!r}"
                    )
    except LivepeerGatewayError:
        raise
    except getattr(aiohttp, "ClientConnectorError", ()) as e:
        raise LivepeerGatewayError(f"HTTP empty POST error: {getattr(e, 'message', e)}") from e
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        raise LivepeerGatewayError(f"HTTP empty POST error: {getattr(e, 'message', e)}") from e


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
