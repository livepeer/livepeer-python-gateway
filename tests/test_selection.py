from __future__ import annotations

import asyncio
from unittest import mock

import pytest

from livepeer_gateway import selection
from livepeer_gateway.errors import LivepeerGatewayError, NoRunnerAvailableError
from livepeer_gateway.live_runner import LiveRunnerCallResult, LiveRunnerInstance


class TestRunnerSelection:
    async def test_runner_selector_flattens_discovery_entries_in_order(self) -> None:
        calls: list[LiveRunnerInstance] = []

        async def _call_runner(
            *,
            runner: LiveRunnerInstance,
            payload: dict[str, object],
            method: str,
            timeout: float,
        ) -> LiveRunnerCallResult:
            del payload, method, timeout
            calls.append(runner)
            if len(calls) < 3:
                raise LivepeerGatewayError("not this one")
            return LiveRunnerCallResult(
                {"ok": True}, runner_url=runner.url, runner=runner
            )

        with (
            mock.patch.object(
                selection, "discover_runners", return_value=_discovery_entries()
            ),
            mock.patch.object(selection, "call_runner", side_effect=_call_runner),
        ):
            cursor = await selection.runner_selector(
                discovery_url="https://example.com/discovery"
            )
            result = await cursor.next()

        candidate = result.runner
        assert candidate is not None
        assert [candidate.url for candidate in calls] == [
            "https://orch-a/apps/a/session",
            "https://orch-a/apps/b/app",
            "https://orch-b/apps/c/session",
        ]
        assert candidate.url == "https://orch-b/apps/c/session"
        assert candidate.app == "app-c"
        assert candidate.runner_id == "runner-c"
        assert candidate.mode == "persistent"
        assert candidate.orchestrator_url == "https://orch-b"
        assert candidate.raw["metadata"] == "runner-c-metadata"
        assert candidate.price_info is not None
        assert candidate.price_info.price == 25
        assert candidate.price_info.currency == "wei"
        assert candidate.price_info.unit == "fixed"
        assert isinstance(cursor.candidates, tuple)
        assert [candidate.url for candidate in cursor.candidates] == [
            "https://orch-a/apps/a/session",
            "https://orch-a/apps/b/app",
            "https://orch-b/apps/c/session",
        ]
        assert [rejection.url for rejection in cursor.rejections] == [
            "https://orch-a/apps/a/session",
            "https://orch-a/apps/b/app",
        ]

    @pytest.mark.parametrize(
        ("payload", "method", "timeout"),
        [({}, "POST", 9.0), ({"prompt": "hi"}, "PUT", 5.0)],
        ids=["defaults", "explicit"],
    )
    async def test_runner_selector_forwards_default_and_explicit_call_arguments(
        self, payload: dict[str, object], method: str, timeout: float
    ) -> None:
        calls: list[tuple[str, dict[str, object], str, float]] = []

        async def _call_runner(
            *,
            runner: LiveRunnerInstance,
            payload: dict[str, object],
            method: str,
            timeout: float,
        ) -> LiveRunnerCallResult:
            calls.append((runner.url, payload, method, timeout))
            return LiveRunnerCallResult(
                {
                    "session_id": "session-1",
                    "app_url": "https://orch-a/apps/a/app",
                    "control_url": "https://orch-a/apps/a/session/session-1",
                },
                runner_url=runner.url,
                runner=runner,
                session_id="session-1",
            )

        with (
            mock.patch.object(
                selection,
                "discover_runners",
                return_value=[
                    _entry(
                        [
                            {
                                "url": "https://orch-a/apps/a/session",
                                "app": "app-a",
                            }
                        ]
                    )
                ],
            ),
            mock.patch.object(
                selection,
                "call_runner",
                side_effect=_call_runner,
            ),
        ):
            cursor = await selection.runner_selector(
                discovery_url="https://example.com/discovery",
                body=payload,
                method=method,
                timeout=timeout,
            )
            result = await cursor.next()

        candidate = result.runner
        assert candidate is not None
        assert result.session_id == "session-1"
        assert result.data["app_url"] == "https://orch-a/apps/a/app"
        assert result.runner is candidate
        assert calls == [("https://orch-a/apps/a/session", payload, method, timeout)]

    async def test_runner_selector_records_failed_calls_and_tries_next(self) -> None:
        async def _call_runner(
            *,
            runner: LiveRunnerInstance,
            payload: dict[str, object],
            method: str,
            timeout: float,
        ) -> LiveRunnerCallResult:
            del payload, method, timeout
            if runner.url.endswith("/a/session"):
                raise LivepeerGatewayError("capacity exhausted")
            return LiveRunnerCallResult(
                {"session_id": "session-2", "app_url": "https://orch-a/apps/b/app"},
                runner_url=runner.url,
                runner=runner,
                session_id="session-2",
            )

        with (
            mock.patch.object(
                selection,
                "discover_runners",
                return_value=[
                    _entry(
                        [
                            {"url": "https://orch-a/apps/a/session", "app": "app-a"},
                            {"url": "https://orch-a/apps/b/session", "app": "app-b"},
                        ]
                    )
                ],
            ),
            mock.patch.object(selection, "call_runner", side_effect=_call_runner),
        ):
            cursor = await selection.runner_selector(
                discovery_url="https://example.com/discovery"
            )
            result = await cursor.next()

        candidate = result.runner
        assert candidate is not None
        assert candidate.url == "https://orch-a/apps/b/session"
        assert result.session_id == "session-2"
        assert result.runner is candidate
        assert len(cursor.rejections) == 1
        assert cursor.rejections[0].url == "https://orch-a/apps/a/session"
        assert cursor.rejections[0].reason == "capacity exhausted"

    async def test_runner_selector_empty_discovery_raises_no_runner_available(
        self,
    ) -> None:
        with mock.patch.object(selection, "discover_runners", return_value=[]):
            with pytest.raises(NoRunnerAvailableError) as raised:
                await selection.runner_selector(
                    discovery_url="https://example.com/discovery"
                )

        assert raised.value.rejections == []

    async def test_runner_selector_accepts_orchestrator_string_and_preserves_path(
        self,
    ) -> None:
        async def _call_runner(
            *,
            runner: LiveRunnerInstance,
            payload: dict[str, object],
            method: str,
            timeout: float,
        ) -> LiveRunnerCallResult:
            del payload, method, timeout
            return LiveRunnerCallResult(
                {"ok": True}, runner_url=runner.url, runner=runner
            )

        with (
            mock.patch.object(
                selection,
                "discover_orchestrator_runners",
                return_value=[
                    _entry(
                        [
                            {
                                "url": "https://orch-b.example.com/apps/a/session",
                                "app": "app-a",
                            }
                        ]
                    )
                ],
            ) as discover_orchestrator_runners_mock,
            mock.patch.object(selection, "call_runner", side_effect=_call_runner),
        ):
            cursor = await selection.runner_selector(
                orchestrators="https://orch-a.example.com/base/, https://orch-b.example.com",
                app="app-a",
            )
            result = await cursor.next()

        discover_orchestrator_runners_mock.assert_awaited_once()
        assert (
            discover_orchestrator_runners_mock.call_args.args[0]
            == "https://orch-a.example.com/base/, https://orch-b.example.com"
        )
        assert result.runner_url == "https://orch-b.example.com/apps/a/session"

    async def test_runner_selector_orchestrators_take_precedence_over_discovery_url(
        self,
    ) -> None:
        async def _call_runner(
            *,
            runner: LiveRunnerInstance,
            payload: dict[str, object],
            method: str,
            timeout: float,
        ) -> LiveRunnerCallResult:
            del payload, method, timeout
            return LiveRunnerCallResult(
                {"ok": True}, runner_url=runner.url, runner=runner
            )

        with (
            mock.patch.object(
                selection,
                "discover_orchestrator_runners",
                return_value=[
                    _entry(
                        [
                            {
                                "url": "https://orch.example.com/base/apps/a/session",
                                "app": "app-a",
                            }
                        ]
                    )
                ],
            ) as discover_orchestrator_runners_mock,
            mock.patch.object(selection, "call_runner", side_effect=_call_runner),
        ):
            cursor = await selection.runner_selector(
                orchestrators=["https://orch.example.com/base"],
                discovery_url="https://explicit.example.com/discovery",
            )
            await cursor.next()

        discover_orchestrator_runners_mock.assert_awaited_once()
        assert discover_orchestrator_runners_mock.call_args.args[0] == [
            "https://orch.example.com/base"
        ]

    async def test_runner_selector_accepts_orchestrator_list_and_skips_empty_discoveries(
        self,
    ) -> None:
        async def _call_runner(
            *,
            runner: LiveRunnerInstance,
            payload: dict[str, object],
            method: str,
            timeout: float,
        ) -> LiveRunnerCallResult:
            del payload, method, timeout
            return LiveRunnerCallResult(
                {"ok": True}, runner_url=runner.url, runner=runner
            )

        with (
            mock.patch.object(
                selection,
                "discover_orchestrator_runners",
                return_value=[
                    _entry(
                        [
                            {
                                "url": "https://orch-b.example.com/apps/a/session",
                                "app": "app-a",
                            }
                        ]
                    )
                ],
            ) as discover_orchestrator_runners_mock,
            mock.patch.object(selection, "call_runner", side_effect=_call_runner),
        ):
            cursor = await selection.runner_selector(
                orchestrators=[
                    "https://orch-a.example.com",
                    "https://orch-b.example.com",
                ],
                app="app-a",
            )
            result = await cursor.next()

        discover_orchestrator_runners_mock.assert_awaited_once()
        assert discover_orchestrator_runners_mock.call_args.args[0] == [
            "https://orch-a.example.com",
            "https://orch-b.example.com",
        ]
        assert result.runner_url == "https://orch-b.example.com/apps/a/session"

    async def test_runner_selector_rejects_invalid_orchestrator_url(self) -> None:
        with pytest.raises(LivepeerGatewayError):
            await selection.runner_selector(orchestrators="ftp://orch.example.com")

    async def test_reserve_session_returns_session_from_call_result(self) -> None:
        async def _call_runner(
            *,
            runner: LiveRunnerInstance,
            payload: dict[str, object],
            method: str,
            timeout: float,
        ) -> LiveRunnerCallResult:
            del payload, method, timeout
            return LiveRunnerCallResult(
                {
                    "session_id": "session-1",
                    "app_url": "https://orch-a/apps/a/app",
                    "control_url": "https://orch-a/apps/a/session/session-1",
                },
                runner_url=runner.url,
                runner=runner,
                session_id="session-1",
            )

        with (
            mock.patch.object(
                selection,
                "discover_runners",
                return_value=[
                    _entry([{"url": "https://orch-a/apps/a/session", "app": "app-a"}])
                ],
            ),
            mock.patch.object(selection, "call_runner", side_effect=_call_runner),
        ):
            session = await selection.reserve_session(
                discovery_url="https://example.com/discovery", app="app-a"
            )

        assert session.session_id == "session-1"
        assert session.app_url == "https://orch-a/apps/a/app"
        assert session.runner_url == "https://orch-a/apps/a/session"
        assert session.runner is not None
        assert session.runner.app == "app-a"

    async def test_reserve_session_rejects_non_session_json(self) -> None:
        async def _call_runner(
            *,
            runner: LiveRunnerInstance,
            payload: dict[str, object],
            method: str,
            timeout: float,
        ) -> LiveRunnerCallResult:
            del runner, payload, method, timeout
            return LiveRunnerCallResult(
                {"ok": True}, runner_url="https://orch-a/apps/a/app"
            )

        with (
            mock.patch.object(
                selection,
                "discover_runners",
                return_value=[
                    _entry([{"url": "https://orch-a/apps/a/app", "app": "app-a"}])
                ],
            ),
            mock.patch.object(selection, "call_runner", side_effect=_call_runner),
        ):
            with pytest.raises(LivepeerGatewayError):
                await selection.reserve_session(
                    discovery_url="https://example.com/discovery", app="app-a"
                )

    async def test_runner_selector_supports_single_shot_url(self) -> None:
        async def _call_runner(
            *,
            runner: LiveRunnerInstance,
            payload: dict[str, object],
            method: str,
            timeout: float,
        ) -> LiveRunnerCallResult:
            del payload, method, timeout
            return LiveRunnerCallResult(
                {"text": "story"}, runner_url=runner.url, runner=runner
            )

        with (
            mock.patch.object(
                selection,
                "discover_runners",
                return_value=[
                    _entry(
                        [
                            {
                                "url": "https://orch-a/apps/story-runner/app",
                                "app": "livepeer/read-story",
                                "mode": "single-shot",
                            }
                        ]
                    )
                ],
            ),
            mock.patch.object(selection, "call_runner", side_effect=_call_runner),
        ):
            cursor = await selection.runner_selector(
                discovery_url="https://example.com/discovery",
                app="livepeer/read-story",
            )
            result = await cursor.next()

        candidate = result.runner
        assert candidate is not None
        assert candidate.mode == "single-shot"
        assert result.data == {"text": "story"}

    async def test_runner_selector_call_failures_raise_aggregate_error(self) -> None:
        async def _call_runner(
            *,
            runner: LiveRunnerInstance,
            payload: dict[str, object],
            method: str,
            timeout: float,
        ) -> LiveRunnerCallResult:
            del payload, method, timeout
            raise LivepeerGatewayError(f"{runner.app} failed")

        with (
            mock.patch.object(
                selection, "discover_runners", return_value=_discovery_entries()
            ),
            mock.patch.object(selection, "call_runner", side_effect=_call_runner),
        ):
            cursor = await selection.runner_selector(
                discovery_url="https://example.com/discovery"
            )
            with pytest.raises(NoRunnerAvailableError) as raised:
                await cursor.next()

        assert len(raised.value.rejections) == 3
        assert raised.value.rejections[0].url == "https://orch-a/apps/a/session"
        assert raised.value.rejections[2].reason == "app-c failed"
        assert "https://orch-a/apps/a/session: app-a failed" in str(raised.value)
        assert "https://orch-b/apps/c/session: app-c failed" in str(raised.value)


def _entry(
    runners: list[dict[str, object]], address: str = "https://orch-a"
) -> dict[str, object]:
    return {"address": address, "runners": runners}


def _discovery_entries() -> list[dict[str, object]]:
    return [
        _entry(
            [
                {
                    "url": "https://orch-a/apps/a/session",
                    "app": "app-a",
                    "runner_id": "runner-a",
                },
                {
                    "url": "https://orch-a/apps/b/app",
                    "app": "app-b",
                    "mode": "single-shot",
                },
            ]
        ),
        _entry(
            [
                {
                    "url": "https://orch-b/apps/c/session",
                    "app": "app-c",
                    "runner_id": "runner-c",
                    "mode": "persistent",
                    "metadata": "runner-c-metadata",
                    "price_info": {
                        "price": 25,
                        "currency": "wei",
                        "unit": "fixed",
                    },
                }
            ],
            address="https://orch-b",
        ),
    ]


@pytest.mark.parametrize("body", [123, mock.Mock(read=mock.AsyncMock())])
async def test_invalid_body_fails_before_attempting_candidates(body):
    runner = LiveRunnerInstance("http://unused", "test", "runner", "single-shot", "http://unused", {})
    cursor = selection.RunnerSelectionCursor([runner, runner], body=body)
    with mock.patch.object(selection, "call_runner") as call:
        with pytest.raises(TypeError):
            await cursor.next()
    call.assert_not_called()
    assert cursor.rejections == []


async def test_cursor_cleanup_failure_preserves_original_error():
    cursor = selection.RunnerSelectionCursor([])
    with mock.patch.object(cursor, "aclose", side_effect=RuntimeError("cleanup failed")):
        with pytest.raises(NoRunnerAvailableError):
            await cursor.next()


async def test_overlapping_next_does_not_interrupt_or_advance_cursor():
    runner = LiveRunnerInstance("http://unused", "test", "runner", "single-shot", "http://unused", {})
    result = LiveRunnerCallResult({"ok": True}, runner_url=runner.url, runner=runner)
    started, finish = asyncio.Event(), asyncio.Event()

    async def call(**kwargs):
        started.set()
        await finish.wait()
        return result

    async with selection.RunnerSelectionCursor([runner, runner]) as cursor:
        with mock.patch.object(selection, "call_runner", side_effect=call) as request:
            active = asyncio.create_task(cursor.next())
            try:
                await asyncio.wait_for(started.wait(), 1)
                with pytest.raises(LivepeerGatewayError, match="already running"):
                    await asyncio.wait_for(cursor.next(), 1)
                assert not active.done()
            finally:
                finish.set()
                assert await active is result
            assert await cursor.next() is result
    assert request.await_count == 2


async def test_cursor_cancellation_closes_upload_cache():
    async def source():
        yield b"data"

    started = asyncio.Event()
    cursor = selection.RunnerSelectionCursor([LiveRunnerInstance("http://unused", "test", "runner", "single-shot", "http://unused", {})], body=source())

    async def call(**kwargs):
        assert await anext(kwargs["payload"].chunks()) == b"data"
        started.set()
        await asyncio.Future()

    with mock.patch.object(selection, "call_runner", side_effect=call):
        attempt = asyncio.create_task(cursor.next())
        await asyncio.wait_for(started.wait(), 1)
        cache = cursor._payload_state.cache
        attempt.cancel()
        with pytest.raises(asyncio.CancelledError):
            await attempt
    assert cache.closed and cursor._payload_state.closed
