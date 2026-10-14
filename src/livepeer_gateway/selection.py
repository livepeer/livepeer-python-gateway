from __future__ import annotations

import logging
from collections.abc import Sequence
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
    _live_runner_price_info_from_json,
    call_runner,
)

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
        body: dict[str, Any] | None = None,
        method: str = "POST",
        signer_url: str | None = None,
        signer_headers: dict[str, str] | None = None,
        timeout: float = 5.0,
    ) -> None:
        self._candidates = list(candidates)
        self._body = dict(body or {})
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
        while self._next_index < len(self._candidates):
            runner = self._candidates[self._next_index]
            self._next_index += 1
            try:
                kwargs: dict[str, Any] = {
                    "runner": runner,
                    "payload": self._body,
                    "method": self._method,
                    "timeout": self._timeout,
                }
                if self._signer_url is not None:
                    kwargs["signer_url"] = self._signer_url
                if self._signer_headers is not None:
                    kwargs["signer_headers"] = self._signer_headers
                result = await call_runner(**kwargs)
            except Exception as e:
                reason = str(e)
                _LOG.debug(
                    "select_runner candidate failed: %s (%s)",
                    runner.url,
                    reason,
                )
                self.rejections.append(RunnerRejection(url=runner.url, reason=reason))
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


async def runner_selector(
    *,
    body: dict[str, Any] | None = None,
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
    session_id = result.data.get("session_id")
    app_url = result.data.get("app_url")
    control_url = _string_value(result.data.get("control_url"))
    if not isinstance(session_id, str) or not session_id.strip():
        raise LivepeerGatewayError("runner session response missing session_id")
    if not isinstance(app_url, str) or not app_url.strip():
        raise LivepeerGatewayError("runner session response missing app_url")
    if not control_url:
        raise LivepeerGatewayError("runner session response missing control_url")
    session = LiveRunnerSession(
        session_id=session_id.strip(),
        app_url=app_url.strip(),
        runner_url=result.runner_url,
        control_url=control_url,
        runner=result.runner,
    )

    # No payment session means fixed price or offchain: nothing to fund.
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


def _string_value(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""
