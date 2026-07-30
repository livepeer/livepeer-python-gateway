from __future__ import annotations

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
        assert candidate.raw["label"] == "runner-c-label"
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
                {"session_id": "session-1", "app_url": "https://orch-a/apps/a/app"},
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
                {"session_id": "session-1", "app_url": "https://orch-a/apps/a/app"},
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
                    "label": "runner-c-label",
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
