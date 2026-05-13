from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Optional, Protocol, TypedDict, cast
from urllib.parse import quote, urlparse, urlunparse

import aiohttp

from .errors import LivepeerGatewayError
from .http import post_json, request_json

_LOG = logging.getLogger(__name__)

_DEFAULT_HEARTBEAT_INTERVAL_S = 5.0

# golang format duration, eg "10s"
_DURATION_RE = re.compile(r"^\s*(?P<value>[0-9]+(?:\.[0-9]+)?)(?P<unit>ns|us|\u00b5s|ms|s|m|h)\s*$")


class LiveRunnerTrickleChannelRequest(TypedDict):
    name: str
    mime_type: str


class LiveRunnerTrickleChannel(TypedDict):
    name: str
    channel_name: str
    url: str
    mime_type: str


class LiveRunnerSessionHeaders(Protocol):
    def get(self, key: str, default: str = "") -> str: ...


class LiveRunnerSessionRequest(Protocol):
    headers: LiveRunnerSessionHeaders


@dataclass(frozen=True)
class LiveRunnerInstance:
    """A normalized live runner discovered from an orchestrator entry."""

    url: str
    app: str
    runner_id: str
    mode: str
    orchestrator_url: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class LiveRunnerSession:
    session_id: str
    app_url: str
    session_url: str
    runner: Optional[LiveRunnerInstance] = None


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
    price_per_unit: int
    pixels_per_unit: int
    unit: str = "USD"

    def to_json(self) -> dict[str, Any]:
        return {
            "price_per_unit": self.price_per_unit,
            "pixels_per_unit": self.pixels_per_unit,
            "unit": self.unit,
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
        label: str = "",
        version: str = "",
        status: str = "ready",
        capacity: int = 1,
        gpu: Optional[LiveRunnerGPU] = None,
        timeout: float = 5.0,
        heartbeat_interval_s: Optional[float] = None,
        unregister_on_close: bool = True,
    ) -> None:
        self.orchestrator_url = _normalize_http_base(orchestrator_url)
        self.runner_id = runner_id
        self.heartbeat_interval_s = heartbeat_interval_s or _DEFAULT_HEARTBEAT_INTERVAL_S
        self.heartbeat_ttl_s: Optional[float] = None

        self._bootstrap_secret = secret
        self._heartbeat_secret: Optional[str] = None
        self._runner_url = runner_url
        self._app = app
        self._price_info = price_info
        self._label = label
        self._version = version
        self._status = status
        self._capacity = capacity
        self._gpu = gpu
        self._timeout = timeout
        self._heartbeat_interval_override = heartbeat_interval_s
        self._unregister_on_close = unregister_on_close
        self._closed = False
        self._task: Optional[asyncio.Task[None]] = None

    async def start(self) -> "LiveRunnerRegistration":
        await self._send_heartbeat()
        self._task = asyncio.create_task(self._heartbeat_loop())
        return self

    async def close(self) -> None:
        self._closed = True
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

    def _payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "runner_url": self._runner_url,
            "app": self._app,
            "capacity": self._capacity,
            "price_info": self._price_info.to_json(),
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

    async def _post_heartbeat(self, auth: str) -> dict[str, Any]:
        return await post_json(
            _join_endpoint(self.orchestrator_url, "/runners/heartbeat"),
            self._payload(),
            headers={"Authorization": auth},
            timeout=self._timeout,
        )


async def register_runner(
    orchestrator_url: str,
    *,
    secret: str,
    runner_url: str,
    app: str,
    price_per_unit: int = 0,
    pixels_per_unit: int = 1,
    price_unit: str = "USD",
    runner_id: str = "",
    label: str = "",
    version: str = "",
    status: str = "ready",
    capacity: int = 1,
    gpu: Optional[LiveRunnerGPU] = None,
    auto_detect_gpu: bool = True,
    timeout: float = 5.0,
    heartbeat_interval_s: Optional[float] = None,
    unregister_on_close: bool = True,
) -> LiveRunnerRegistration:
    if gpu is None and auto_detect_gpu:
        gpu = detect_process_gpu()

    registration = LiveRunnerRegistration(
        orchestrator_url=orchestrator_url,
        secret=secret,
        runner_url=runner_url,
        app=app,
        price_info=LiveRunnerPriceInfo(price_per_unit, pixels_per_unit, price_unit),
        runner_id=runner_id,
        label=label,
        version=version,
        status=status,
        capacity=capacity,
        gpu=gpu,
        timeout=timeout,
        heartbeat_interval_s=heartbeat_interval_s,
        unregister_on_close=unregister_on_close,
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


async def reserve_runner_session(
    session_url: str = "",
    *,
    runner: Optional[LiveRunnerInstance] = None,
    timeout: float = 5.0,
) -> LiveRunnerSession:
    session_url = session_url.strip() or (runner.url.strip() if runner is not None else "")
    if not session_url:
        raise LivepeerGatewayError("Live runner session reserve requires session_url")
    data = await post_json(
        session_url,
        {},
        timeout=timeout,
    )
    session_id = data.get("session_id")
    app_url = data.get("app_url")
    if not isinstance(session_id, str) or not session_id.strip():
        raise LivepeerGatewayError("Live runner session reserve response missing session_id")
    if not isinstance(app_url, str) or not app_url.strip():
        raise LivepeerGatewayError("Live runner session reserve response missing app_url")
    return LiveRunnerSession(
        session_id=session_id.strip(),
        app_url=app_url.strip(),
        session_url=session_url,
        runner=runner,
    )


async def stop_runner_session(
    session: LiveRunnerSession,
    *,
    timeout: float = 5.0,
) -> None:
    session_url = session.session_url.strip()
    session_id = session.session_id.strip()
    if not session_url:
        raise LivepeerGatewayError("Live runner session stop requires session_url")
    if not session_id:
        raise LivepeerGatewayError("Live runner session stop requires session_id")
    await _post_empty(
        _join_endpoint(session_url, f"/{quote(session_id, safe='')}/stop"),
        {},
        timeout,
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


def _is_invalid_authorization_error(exc: LivepeerGatewayError) -> bool:
    message = str(exc).lower()
    return "http 401" in message and "invalid authorization" in message


def _resolve_session_credentials(
    session: str | LiveRunnerSessionRequest,
    *,
    runner_id: str = "",
    session_token: str = "",
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
        raise LivepeerGatewayError("Live runner trickle channel request requires session_id")
    if not token:
        raise LivepeerGatewayError("Live runner trickle channel request requires session_token")
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
    )


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
