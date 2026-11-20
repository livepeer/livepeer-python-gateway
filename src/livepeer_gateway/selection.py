from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Sequence
from typing import Any
from urllib.parse import SplitResult, quote, unquote, urlsplit

from .discovery import (
    FilterValue,
    discover_orchestrator_runners,
    discover_runners,
)
from .errors import (
    LivepeerGatewayError,
    NoRunnerAvailableError,
    RunnerRejection,
)
from .live_runner import (
    LiveRunnerCallResult,
    LiveRunnerInstance,
    LiveRunnerSession,
    _join_endpoint,
    _live_runner_price_info_from_json,
    call_runner,
    stop_runner_session,
)
from .runner_payload import RunnerPayload, _ReplayUnavailableError, _StreamState, _stream_state

_LOG = logging.getLogger(__name__)


class RunnerSelectionCursor:
    """
    Stateful selector that advances through live runners sequentially.

    Runner attempts are intentionally not parallelized: selecting a persistent
    runner reserves capacity, and selecting a single-shot runner may perform
    the caller's actual app operation.
    """

    def __init__(
        self,
        candidates: Sequence[LiveRunnerInstance],
        *,
        body: RunnerPayload = None,
        content_type: str | None = None,
        method: str = "POST",
        signer_url: str | None = None,
        signer_headers: dict[str, str] | None = None,
        timeout: float = 5.0,
    ) -> None:
        self._candidates = list(candidates)
        self._body = dict(body) if isinstance(body, dict) else body
        self._content_type = content_type
        self._payload_state: _StreamState | None = None
        self._closed = False
        self._next_running = False
        self._method = method
        self._signer_url = signer_url
        self._signer_headers = signer_headers
        self._timeout = timeout
        self._next_index = 0
        self.rejections: list[RunnerRejection] = []

    @property
    def candidates(self) -> tuple[LiveRunnerInstance, ...]:
        return tuple(self._candidates)

    async def next(self) -> LiveRunnerCallResult:
        if self._closed:
            raise LivepeerGatewayError("Runner selection cursor is closed")
        if self._next_running:
            raise LivepeerGatewayError("Runner selection cursor.next() is already running")
        self._next_running = True
        try:
            if self._payload_state is None:
                self._payload_state = _stream_state(self._body)
            while self._next_index < len(self._candidates):
                if self._payload_state is not None:
                    self._payload_state.require_replayable()
                runner = self._candidates[self._next_index]
                self._next_index += 1
                try:
                    kwargs: dict[str, Any] = {
                        "runner": runner,
                        "payload": self._payload_state if self._payload_state is not None else self._body,
                        "method": self._method,
                        "timeout": self._timeout,
                    }
                    if self._content_type is not None:
                        kwargs["content_type"] = self._content_type
                    if self._signer_url is not None:
                        kwargs["signer_url"] = self._signer_url
                    if self._signer_headers is not None:
                        kwargs["signer_headers"] = self._signer_headers
                    result = await call_runner(**kwargs)
                except _ReplayUnavailableError:
                    raise
                except Exception as e:
                    reason = str(e)
                    _LOG.debug(
                        "select_runner candidate failed: %s (%s)",
                        runner.url,
                        reason,
                    )
                    self.rejections.append(RunnerRejection(url=runner.url, reason=reason))
                    if self._payload_state is not None:
                        self._payload_state.require_replayable(e)
                    continue

                _LOG.debug("select_runner selected: %s", runner.url)
                return result

            _LOG.debug(
                "select_runner failed: all %d runners rejected",
                len(self._candidates),
            )
            raise NoRunnerAvailableError(
                f"All runners failed ({len(self.rejections)} tried)",
                rejections=list(self.rejections),
            )
        except BaseException:
            try:
                await self.aclose()
            except Exception:
                _LOG.warning("Runner cursor cleanup failed", exc_info=True)
            raise
        finally:
            self._next_running = False

    async def aclose(self) -> None:
        """Release the replay cache without closing the caller's upload source."""
        self._closed = True
        if self._payload_state is not None:
            await self._payload_state.aclose()

    async def __aenter__(self) -> RunnerSelectionCursor:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


async def runner_selector(
    *,
    body: RunnerPayload = None,
    content_type: str | None = None,
    method: str = "POST",
    orchestrators: Sequence[str] | str | None = None,
    signer_url: str | None = None,
    signer_headers: dict[str, str] | None = None,
    discovery_url: str | None = None,
    discovery_headers: dict[str, str] | None = None,
    app: FilterValue | None = None,
    gpu: FilterValue | None = None,
    timeout: float = 5.0,
) -> RunnerSelectionCursor:
    if orchestrators is not None:
        entries = await discover_orchestrator_runners(
            orchestrators,
            app=app,
            gpu=gpu,
        )
    else:
        entries = await discover_runners(
            signer_url=signer_url,
            signer_headers=signer_headers,
            discovery_url=discovery_url,
            discovery_headers=discovery_headers,
            app=app,
            gpu=gpu,
        )

    candidates = _runner_candidates_from_discovery(entries)

    if not candidates:
        _LOG.debug("select_runner failed: empty runner list")
        raise NoRunnerAvailableError("No runners available to select")

    return RunnerSelectionCursor(
        candidates,
        body=body,
        content_type=content_type,
        method=method,
        signer_url=signer_url,
        signer_headers=signer_headers,
        timeout=timeout,
    )


async def reserve_session(
    *,
    signer_url: str | None = None,
    signer_headers: dict[str, str] | None = None,
    discovery_url: str | None = None,
    discovery_headers: dict[str, str] | None = None,
    orchestrators: Sequence[str] | str | None = None,
    app: FilterValue | None = None,
    gpu: FilterValue | None = None,
    timeout: float = 5.0,
) -> LiveRunnerSession:
    cursor = await runner_selector(
        orchestrators=orchestrators,
        signer_url=signer_url,
        signer_headers=signer_headers,
        discovery_url=discovery_url,
        discovery_headers=discovery_headers,
        app=app,
        gpu=gpu,
        timeout=timeout,
    )
    result = await cursor.next()
    data = result.data if isinstance(result.data, dict) else {}
    session_id = _string_value(data.get("session_id"))
    app_url = _string_value(data.get("app_url"))
    control_url = _string_value(data.get("control_url"))
    known_id = _string_value(result.session_id)
    session = None
    try:
        if not session_id:
            raise LivepeerGatewayError("runner session response missing session_id")
        if known_id and session_id != known_id:
            raise LivepeerGatewayError("runner session response session_id does not match payment manifest")
        if not app_url:
            raise LivepeerGatewayError("runner session response missing app_url")
        if not control_url:
            raise LivepeerGatewayError("runner session response missing control_url")
        try:
            session = LiveRunnerSession(session_id, app_url, result.runner_url, control_url, result.runner)
        except ValueError as error:
            raise LivepeerGatewayError("runner session response has invalid control_url") from error
        if result.payment_session is not None:
            session._start_payments(result.payment_session)
        return session
    except BaseException as error:
        cleanup = session or _invalid_reservation_handle(result, known_id or session_id, control_url, app_url)
        if cleanup is not None:
            await _release_invalid_reservation(cleanup, error)
        elif known_id or session_id or control_url:
            error.add_note("Reservation cleanup unresolved: no verified stop route")
        raise


def _invalid_reservation_handle(
    result: LiveRunnerCallResult, session_id: str, control_url: str, app_url: str = "",
) -> LiveRunnerSession | None:
    runner_url = result.runner_url
    try:
        base = urlsplit(runner_url)
    except ValueError:
        return None
    path = base.path.rstrip("/")
    canonical = re.search(r"/apps/[^/]+/session$", path) is not None
    payment = result.payment_session
    body_id = _string_value(result.data.get("session_id")) if isinstance(result.data, dict) else ""
    # On offchain calls result.session_id just repeats the body field.
    manifest = getattr(payment, "manifest_id", "") or (result.session_id if result.session_id != body_id else "")

    def verified(url: str) -> SplitResult | None:
        try:
            candidate = urlsplit(url)
            if _origin(candidate) == _origin(base):
                return candidate
        except ValueError:
            pass
        return None

    def handle(sid: str, url: str) -> LiveRunnerSession:
        return LiveRunnerSession(sid, "", runner_url, url, result.runner)

    try:
        control = verified(control_url)
        control_path = control.path.rstrip("/") if control is not None else ""
        if not canonical and payment is not None and manifest and session_id == manifest:
            payment_url = getattr(getattr(payment, "_challenge", None), "payment_url", "")
            payment_route = verified(payment_url)
            suffix = "/" + quote(manifest, safe="") + "/payment"
            if payment_route and payment_route.path.endswith(suffix) and re.search(
                r"/apps/[^/]+/session$", payment_route.path.removesuffix(suffix),
            ):
                path = payment_route.path.removesuffix("/payment")
                stop_url = control_url if control_path == path else payment_route._replace(path=path).geturl()
                return handle(manifest, stop_url)
        match = re.search(r"(/apps/[^/]+/session)/([A-Za-z0-9_-]+)$", unquote(control_path))
        if match and (not canonical or control_path.rsplit("/", 1)[0] == path):
            control_id = match[2]
            app = verified(app_url)
            if app is not None and app.path.rstrip("/") == control_path + "/app" and (
                not manifest or manifest == control_id
            ):
                return handle(control_id, control_url)
            if session_id == control_id or (canonical and not session_id):
                return handle(control_id, control_url)
            if session_id and not manifest:
                # The two offchain fields disagree without independent corroboration.
                return None
        if canonical and session_id:
            return handle(session_id, _join_endpoint(runner_url, quote(session_id, safe="")))
    except (ValueError, LivepeerGatewayError):
        pass  # No usable stop identity in the response or request URL.
    return None


def _origin(url: SplitResult) -> tuple[str, str | None, int]:
    default_port = 443 if url.scheme == "https" else 80
    return url.scheme, url.hostname, url.port if url.port is not None else default_port


async def _release_invalid_reservation(session: LiveRunnerSession, original: BaseException) -> None:
    async def release() -> None:
        for attempt in range(3):
            try:
                await stop_runner_session(session, timeout=5.0)
                return
            except Exception as error:
                status = getattr(error, "status_code", None)
                if status in (404, 410):
                    return
                if attempt == 2 or status is not None and status not in (408, 429) and not 500 <= status < 600:
                    raise
                await asyncio.sleep(0.25 * 2**attempt)

    task = asyncio.create_task(release())
    cancellation = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as error:
            cancellation = error
        except Exception:
            break
    try:
        task.result()
    except Exception as error:
        note = f"Reservation cleanup failed for session {session.session_id}: {error}"
        original.add_note(note)
        original.cleanup_session = session
        if cancellation is not None:
            cancellation.add_note(note)
            cancellation.cleanup_session = session
        _LOG.error("Invalid reservation cleanup failed for session %s", session.session_id, exc_info=True)
    if cancellation is not None:
        raise cancellation from original


def _runner_candidates_from_discovery(entries: Sequence[dict[str, Any]]) -> list[LiveRunnerInstance]:
    candidates: list[LiveRunnerInstance] = []
    for entry in entries:
        orchestrator_url = _string_value(entry.get("address"))
        runners = entry.get("runners")
        if not isinstance(runners, list):
            continue

        for runner in runners:
            if not isinstance(runner, dict):
                continue
            url = _string_value(runner.get("url"))
            app = _string_value(runner.get("app"))
            if not url or not app:
                continue
            candidates.append(
                LiveRunnerInstance(
                    url=url,
                    app=app,
                    runner_id=_string_value(runner.get("runner_id")),
                    mode=_string_value(runner.get("mode")),
                    orchestrator_url=orchestrator_url,
                    raw=dict(runner),
                    price_info=_live_runner_price_info_from_json(runner.get("price_info")),
                )
            )
    return candidates


def _string_value(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""
