from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional, Sequence, Tuple

from . import lp_rpc_pb2
from .discovery import FilterValue, discover_orchestrators, discover_runners
from .errors import (
    LivepeerGatewayError,
    NoOrchestratorAvailableError,
    NoRunnerAvailableError,
    OrchestratorRejection,
    RunnerRejection,
)
from .live_runner import LiveRunnerCallResult, LiveRunnerInstance, LiveRunnerSession, call_runner
from .orch_info import get_orch_info

_LOG = logging.getLogger(__name__)

_BATCH_SIZE = 5


class SelectionCursor:
    """
    Stateful selector that advances through orchestrators in batches.

    For each batch, all ``get_orch_info`` calls run in parallel and successful
    responses are cached in arrival order. Consumers pop cached successes one at
    a time and only trigger the next batch once the current cache is exhausted.
    """

    def __init__(
        self,
        orch_list: Sequence[str],
        *,
        signer_url: Optional[str] = None,
        signer_headers: Optional[dict[str, str]] = None,
        capabilities: Optional[lp_rpc_pb2.Capabilities] = None,
        use_tofu: bool = True,
    ) -> None:
        self._orch_list = list(orch_list)
        self._signer_url = signer_url
        self._signer_headers = signer_headers
        self._capabilities = capabilities
        self._use_tofu = use_tofu
        self._batch_start = 0
        self._pending_successes: list[Tuple[str, lp_rpc_pb2.OrchestratorInfo]] = []
        self.rejections: list[OrchestratorRejection] = []

    def next(self) -> Tuple[str, lp_rpc_pb2.OrchestratorInfo]:
        while True:
            if self._pending_successes:
                selected = self._pending_successes.pop(0)
                _LOG.debug("select_orchestrator selected: %s", selected[0])
                return selected

            if self._batch_start >= len(self._orch_list):
                _LOG.debug(
                    "select_orchestrator failed: all %d orchestrators rejected",
                    len(self._orch_list),
                )
                raise NoOrchestratorAvailableError(
                    f"All orchestrators failed ({len(self.rejections)} tried)",
                    rejections=list(self.rejections),
                )

            self._populate_next_batch_successes()

    def _populate_next_batch_successes(self) -> None:
        batch = self._orch_list[self._batch_start : self._batch_start + _BATCH_SIZE]
        batch_start = self._batch_start
        self._batch_start += _BATCH_SIZE
        _LOG.debug(
            "select_orchestrator trying batch %d-%d: %s",
            batch_start,
            batch_start + len(batch) - 1,
            batch,
        )

        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = {
                executor.submit(
                    get_orch_info,
                    url,
                    signer_url=self._signer_url,
                    signer_headers=self._signer_headers,
                    capabilities=self._capabilities,
                    use_tofu=self._use_tofu,
                ): url
                for url in batch
            }

            batch_successes: list[Tuple[str, lp_rpc_pb2.OrchestratorInfo]] = []
            for future in as_completed(futures):
                url = futures[future]
                try:
                    info = future.result()
                except Exception as e:
                    reason = str(e)
                    _LOG.debug(
                        "select_orchestrator candidate failed: %s (%s)",
                        url,
                        reason,
                    )
                    self.rejections.append(OrchestratorRejection(url=url, reason=reason))
                    continue
                batch_successes.append((url, info))

        self._pending_successes.extend(batch_successes)
        if not batch_successes:
            _LOG.debug("select_orchestrator batch exhausted, trying next batch")


def orchestrator_selector(
    orchestrators: Optional[Sequence[str] | str] = None,
    *,
    signer_url: Optional[str] = None,
    signer_headers: Optional[dict[str, str]] = None,
    discovery_url: Optional[str] = None,
    discovery_headers: Optional[dict[str, str]] = None,
    capabilities: Optional[lp_rpc_pb2.Capabilities] = None,
    use_tofu: bool = True,
) -> SelectionCursor:
    orch_list = discover_orchestrators(
        orchestrators,
        signer_url=signer_url,
        signer_headers=signer_headers,
        discovery_url=discovery_url,
        discovery_headers=discovery_headers,
        capabilities=capabilities,
    )

    if not orch_list:
        _LOG.debug("select_orchestrator failed: empty orchestrator list")
        raise NoOrchestratorAvailableError("No orchestrators available to select")

    return SelectionCursor(
        orch_list,
        signer_url=signer_url,
        signer_headers=signer_headers,
        capabilities=capabilities,
        use_tofu=use_tofu,
    )


class RunnerSelectionCursor:
    """
    Stateful selector that advances through live runners sequentially.

    Runner attempts are intentionally not parallelized: selecting a session
    runner reserves capacity, and selecting a single-shot runner may perform
    the caller's actual app operation.
    """

    def __init__(
        self,
        candidates: Sequence[LiveRunnerInstance],
        *,
        signer_url: Optional[str] = None,
        signer_headers: Optional[dict[str, str]] = None,
        timeout: float = 5.0,
    ) -> None:
        self._candidates = list(candidates)
        self._signer_url = signer_url
        self._signer_headers = signer_headers
        self._timeout = timeout
        self._next_index = 0
        self.rejections: list[RunnerRejection] = []

    async def next(self) -> LiveRunnerCallResult:
        while self._next_index < len(self._candidates):
            runner = self._candidates[self._next_index]
            self._next_index += 1
            try:
                kwargs: dict[str, Any] = {"runner": runner, "timeout": self._timeout}
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


def runner_selector(
    *,
    signer_url: Optional[str] = None,
    signer_headers: Optional[dict[str, str]] = None,
    discovery_url: Optional[str] = None,
    discovery_headers: Optional[dict[str, str]] = None,
    app: Optional[FilterValue] = None,
    gpu: Optional[FilterValue] = None,
    timeout: float = 5.0,
) -> RunnerSelectionCursor:
    entries = discover_runners(
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
        signer_url=signer_url,
        signer_headers=signer_headers,
        timeout=timeout,
    )


async def reserve_session(
    *,
    signer_url: Optional[str] = None,
    signer_headers: Optional[dict[str, str]] = None,
    discovery_url: Optional[str] = None,
    discovery_headers: Optional[dict[str, str]] = None,
    app: Optional[FilterValue] = None,
    gpu: Optional[FilterValue] = None,
    timeout: float = 5.0,
) -> LiveRunnerSession:
    result = await runner_selector(
        signer_url=signer_url,
        signer_headers=signer_headers,
        discovery_url=discovery_url,
        discovery_headers=discovery_headers,
        app=app,
        gpu=gpu,
        timeout=timeout,
    ).next()
    session_id = result.data.get("session_id")
    app_url = result.data.get("app_url")
    if not isinstance(session_id, str) or not session_id.strip():
        raise LivepeerGatewayError("runner session response missing session_id")
    if not isinstance(app_url, str) or not app_url.strip():
        raise LivepeerGatewayError("runner session response missing app_url")
    return LiveRunnerSession(
        session_id=session_id.strip(),
        app_url=app_url.strip(),
        runner_url=result.runner_url,
        runner=result.runner,
    )


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
                )
            )
    return candidates


def _string_value(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""
