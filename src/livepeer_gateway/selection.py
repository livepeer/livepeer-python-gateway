from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

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
    _is_session_endpoint,
    _live_runner_price_info_from_json,
    _string_value,
    call_runner,
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
    _matches: Callable[[LiveRunnerInstance], bool] = bool,
) -> RunnerSelectionCursor:
    if orchestrators is not None:
        entries = await discover_orchestrator_runners(
            orchestrators,
            app=app,
            gpu=gpu,
            _has_runners=lambda entries: any(_matches(runner) for runner in _runner_candidates_from_discovery(entries)),
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

    candidates = [runner for runner in _runner_candidates_from_discovery(entries) if _matches(runner)]

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
        _matches=lambda runner: _is_session_endpoint(runner.url, runner),
    )
    result = await cursor.next()
    session = LiveRunnerSession(result.session_id, result.data["app_url"].strip(), result.runner_url, result.data["control_url"], result.runner)
    if result.payment_session is not None:
        session._start_payments(result.payment_session)
    return session


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
