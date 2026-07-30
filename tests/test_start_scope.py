from __future__ import annotations

import base64
import json
from types import SimpleNamespace
from unittest import mock

import pytest

from livepeer_gateway import lv2v as lv2v_mod
from livepeer_gateway import scope as scope_mod
from livepeer_gateway.errors import (
    LivepeerHTTPError,
    NoRunnerAvailableError,
    RunnerRejection,
)
from livepeer_gateway.live_runner import LiveRunnerCallResult, LiveRunnerInstance


class TestStartScopeRunner:
    async def _run_start_scope(
        self,
        *,
        req: lv2v_mod.StartJobRequest | None = None,
        job_control: object | None = None,
        payment_session: object | None = None,
        result_data: dict[str, object] | None = None,
        job_manifest_id: str | None = "manifest",
        orch_url: object = None,
        discovery_url: str | None = "https://discovery.example.com",
        token: str | None = None,
        runner_version: object = "serverless-1.0.0",
    ) -> tuple[object, mock.Mock, mock.Mock]:
        raw: dict[str, object] = {}
        if runner_version is not None:
            raw["version"] = runner_version
        runner = LiveRunnerInstance(
            url="https://runner.example.com/app",
            app="live-video-to-video/scope",
            runner_id="runner-1",
            mode="",
            orchestrator_url="https://orch.example.com",
            raw=raw,
        )

        class _Cursor:
            async def next(self) -> LiveRunnerCallResult:
                return LiveRunnerCallResult(
                    result_data or {"manifest_id": "manifest"},
                    runner_url="https://runner.example.com/app",
                    runner=runner,
                    payment_session=payment_session,
                )

        runner_selector_mock = mock.AsyncMock(return_value=_Cursor())
        job = SimpleNamespace(
            manifest_id=job_manifest_id,
            control=job_control,
            start_payment_sender=mock.Mock(),
        )
        from_json_mock = mock.Mock(return_value=job)

        with mock.patch.object(scope_mod, "runner_selector", runner_selector_mock):
            with mock.patch.object(
                lv2v_mod.LiveVideoToVideo, "from_json", from_json_mock
            ):
                start_req = req or lv2v_mod.StartJobRequest(model_id="noop")
                result = await scope_mod.start_scope(
                    orch_url,
                    start_req,
                    discovery_url=discovery_url,
                    token=token,
                    signer_url="https://signer.example.com",
                    signer_headers={"Authorization": "token"},
                    timeout=9.0,
                )

        return result, runner_selector_mock, from_json_mock

    async def test_start_scope_selects_scope_runner_and_parses_result(self) -> None:
        payment_session = object()
        control = mock.Mock()
        result, runner_selector_mock, from_json_mock = await self._run_start_scope(
            job_control=control,
            payment_session=payment_session,
        )

        assert result is not None
        runner_selector_mock.assert_called_once()
        assert (
            runner_selector_mock.call_args.kwargs["app"] == "live-video-to-video/scope"
        )
        assert runner_selector_mock.call_args.kwargs["body"] == {"model_id": "noop"}
        assert (
            runner_selector_mock.call_args.kwargs["discovery_url"]
            == "https://discovery.example.com"
        )
        from_json_mock.assert_called_once_with(
            {"manifest_id": "manifest"},
            signer_url="https://signer.example.com",
            payment_session=payment_session,
        )
        result.start_payment_sender.assert_not_called()
        control.start_keepalive.assert_not_called()

    async def test_start_scope_non_serverless_posts_to_app_scope_with_helper(
        self,
    ) -> None:
        payment_session = object()
        post_json_mock = mock.AsyncMock(
            return_value={"manifest_id": "manifest-from-scope"}
        )
        with mock.patch.object(scope_mod, "post_json", post_json_mock):
            (
                _result,
                _runner_selector_mock,
                from_json_mock,
            ) = await self._run_start_scope(
                result_data={
                    "session_id": "session-1",
                    "app_url": "https://orch.example.com/apps/runner-1/session/session-1/app/",
                },
                payment_session=payment_session,
                runner_version="1.2.3",
            )

        post_json_mock.assert_awaited_once_with(
            "https://orch.example.com/apps/runner-1/session/session-1/app/scope",
            {"model_id": "noop"},
            timeout=9.0,
        )
        from_json_mock.assert_called_once_with(
            {"manifest_id": "manifest-from-scope"},
            signer_url="https://signer.example.com",
            payment_session=payment_session,
        )

    @pytest.mark.parametrize("runner_version", [None, 123])
    async def test_start_scope_missing_or_non_string_runner_version_uses_app_scope(
        self, runner_version: object
    ) -> None:
        post_json_mock = mock.AsyncMock(
            return_value={"manifest_id": "manifest-from-scope"}
        )
        with mock.patch.object(scope_mod, "post_json", post_json_mock):
            await self._run_start_scope(
                result_data={
                    "session_id": "session-1",
                    "app_url": "https://orch.example.com/app",
                },
                runner_version=runner_version,
            )

        post_json_mock.assert_awaited_once_with(
            "https://orch.example.com/app/scope",
            {"model_id": "noop"},
            timeout=9.0,
        )

    async def test_start_scope_non_serverless_requires_app_url(self) -> None:
        runner = LiveRunnerInstance(
            url="https://runner.example.com/session",
            app="live-video-to-video/scope",
            runner_id="runner-1",
            mode="",
            orchestrator_url="https://orch.example.com",
            raw={"version": "1.2.3"},
        )

        class _Cursor:
            def __init__(self) -> None:
                self.rejections: list[RunnerRejection] = []
                self.calls = 0

            async def next(self) -> LiveRunnerCallResult:
                self.calls += 1
                if self.calls == 1:
                    return LiveRunnerCallResult(
                        {"session_id": "session-1"},
                        runner_url=runner.url,
                        runner=runner,
                    )
                raise NoRunnerAvailableError(
                    "All runners failed (1 tried)",
                    rejections=list(self.rejections),
                )

        cursor = _Cursor()
        with mock.patch.object(
            scope_mod, "runner_selector", mock.AsyncMock(return_value=cursor)
        ):
            with pytest.raises(NoRunnerAvailableError) as raised:
                await scope_mod.start_scope(
                    None,
                    lv2v_mod.StartJobRequest(),
                    discovery_url="https://discovery.example.com",
                )

        assert len(raised.value.rejections) == 1
        assert raised.value.rejections[0].url == runner.url
        assert "missing app_url" in raised.value.rejections[0].reason

    async def test_start_scope_retries_next_runner_when_app_scope_post_fails(
        self,
    ) -> None:
        runners = [
            LiveRunnerInstance(
                url="https://runner-a.example.com/session",
                app="live-video-to-video/scope",
                runner_id="runner-a",
                mode="",
                orchestrator_url="https://orch-a.example.com",
                raw={"version": "1.2.3"},
            ),
            LiveRunnerInstance(
                url="https://runner-b.example.com/session",
                app="live-video-to-video/scope",
                runner_id="runner-b",
                mode="",
                orchestrator_url="https://orch-b.example.com",
                raw={"version": "1.2.3"},
            ),
        ]

        class _Cursor:
            def __init__(self) -> None:
                self.rejections: list[RunnerRejection] = []
                self.calls = 0

            async def next(self) -> LiveRunnerCallResult:
                if self.calls >= len(runners):
                    raise NoRunnerAvailableError(
                        f"All runners failed ({len(self.rejections)} tried)",
                        rejections=list(self.rejections),
                    )
                runner = runners[self.calls]
                self.calls += 1
                return LiveRunnerCallResult(
                    {
                        "session_id": f"session-{self.calls}",
                        "app_url": f"https://orch-{self.calls}.example.com/app",
                    },
                    runner_url=runner.url,
                    runner=runner,
                    payment_session=f"payment-{self.calls}",
                )

        cursor = _Cursor()
        post_json_mock = mock.AsyncMock(
            side_effect=[
                LivepeerHTTPError(502, "https://orch-1.example.com/app/scope", "bad"),
                {"manifest_id": "manifest-from-second-runner"},
            ]
        )
        job = SimpleNamespace(
            manifest_id="manifest-from-second-runner",
            control=None,
            start_payment_sender=mock.Mock(),
        )
        from_json_mock = mock.Mock(return_value=job)

        with (
            mock.patch.object(
                scope_mod, "runner_selector", mock.AsyncMock(return_value=cursor)
            ),
            mock.patch.object(scope_mod, "post_json", post_json_mock),
            mock.patch.object(lv2v_mod.LiveVideoToVideo, "from_json", from_json_mock),
        ):
            result = await scope_mod.start_scope(
                None,
                lv2v_mod.StartJobRequest(model_id="noop"),
                discovery_url="https://discovery.example.com",
                signer_url="https://signer.example.com",
            )

        assert result is job
        assert cursor.calls == 2
        assert len(cursor.rejections) == 1
        assert cursor.rejections[0].url == "https://runner-a.example.com/session"
        assert "HTTP 502" in cursor.rejections[0].reason
        assert [call.args[0] for call in post_json_mock.await_args_list] == [
            "https://orch-1.example.com/app/scope",
            "https://orch-2.example.com/app/scope",
        ]
        from_json_mock.assert_called_once_with(
            {"manifest_id": "manifest-from-second-runner"},
            signer_url="https://signer.example.com",
            payment_session="payment-2",
        )

    async def test_start_scope_aggregates_startup_failures(self) -> None:
        runners = [
            LiveRunnerInstance(
                url="https://runner-a.example.com/session",
                app="live-video-to-video/scope",
                runner_id="runner-a",
                mode="",
                orchestrator_url="https://orch-a.example.com",
                raw={"version": "1.2.3"},
            ),
            LiveRunnerInstance(
                url="https://runner-b.example.com/session",
                app="live-video-to-video/scope",
                runner_id="runner-b",
                mode="",
                orchestrator_url="https://orch-b.example.com",
                raw={"version": "serverless-1.0.0"},
            ),
        ]

        class _Cursor:
            def __init__(self) -> None:
                self.rejections: list[RunnerRejection] = []
                self.calls = 0

            async def next(self) -> LiveRunnerCallResult:
                if self.calls >= len(runners):
                    raise NoRunnerAvailableError(
                        f"All runners failed ({len(self.rejections)} tried)",
                        rejections=list(self.rejections),
                    )
                runner = runners[self.calls]
                self.calls += 1
                return LiveRunnerCallResult(
                    {"session_id": f"session-{self.calls}"},
                    runner_url=runner.url,
                    runner=runner,
                )

        cursor = _Cursor()
        missing_manifest_job = SimpleNamespace(
            manifest_id=None,
            control=None,
            start_payment_sender=mock.Mock(),
        )

        with (
            mock.patch.object(
                scope_mod, "runner_selector", mock.AsyncMock(return_value=cursor)
            ),
            mock.patch.object(
                lv2v_mod.LiveVideoToVideo,
                "from_json",
                mock.Mock(return_value=missing_manifest_job),
            ),
            mock.patch.object(scope_mod._LOG, "info") as log_mock,
        ):
            with pytest.raises(NoRunnerAvailableError) as raised:
                await scope_mod.start_scope(
                    None,
                    lv2v_mod.StartJobRequest(),
                    discovery_url="https://discovery.example.com",
                )

        assert [rejection.url for rejection in raised.value.rejections] == [
            "https://runner-a.example.com/session",
            "https://runner-b.example.com/session",
        ]
        assert "missing app_url" in raised.value.rejections[0].reason
        assert "missing manifest_id" in raised.value.rejections[1].reason
        assert log_mock.call_count == 2

    @pytest.mark.parametrize(
        ("model_id", "expected_body"),
        [
            (None, {}),
            ("custom-scope", {"model_id": "custom-scope"}),
        ],
        ids=["missing", "explicit"],
    )
    async def test_start_scope_handles_missing_and_explicit_model_id(
        self, model_id: str | None, expected_body: dict[str, str]
    ) -> None:
        (
            _result,
            runner_selector_mock,
            _from_json_mock,
        ) = await self._run_start_scope(
            req=lv2v_mod.StartJobRequest(model_id=model_id),
            job_control=mock.Mock(),
        )

        assert runner_selector_mock.call_args.kwargs["body"] == expected_body

    async def test_start_scope_derives_discovery_from_orch_url(self) -> None:
        _result, runner_selector_mock, _from_json_mock = await self._run_start_scope(
            orch_url="http://orch.example.com:8935/base",
            discovery_url=None,
        )

        assert (
            runner_selector_mock.call_args.kwargs["orchestrators"]
            == "http://orch.example.com:8935/base"
        )
        assert runner_selector_mock.call_args.kwargs["discovery_url"] is None

    async def test_start_scope_passes_multiple_orch_urls_to_runner_selector(
        self,
    ) -> None:
        class _Cursor:
            async def next(self) -> LiveRunnerCallResult:
                runner = LiveRunnerInstance(
                    url="https://runner-b.example.com/app",
                    app="live-video-to-video/scope",
                    runner_id="runner-b",
                    mode="",
                    orchestrator_url="https://orch-b.example.com",
                    raw={"version": "serverless-1.0.0"},
                )
                return LiveRunnerCallResult(
                    {"manifest_id": "manifest"},
                    runner_url="https://runner-b.example.com/app",
                    runner=runner,
                )

        runner_selector_mock = mock.AsyncMock(return_value=_Cursor())
        job = SimpleNamespace(
            manifest_id="manifest",
            control=None,
            start_payment_sender=mock.Mock(),
        )

        with mock.patch.object(scope_mod, "runner_selector", runner_selector_mock):
            with mock.patch.object(
                lv2v_mod.LiveVideoToVideo, "from_json", mock.Mock(return_value=job)
            ):
                await scope_mod.start_scope(
                    ["https://orch-a.example.com", "https://orch-b.example.com"],
                    lv2v_mod.StartJobRequest(),
                    discovery_url=None,
                )

        runner_selector_mock.assert_called_once()
        assert runner_selector_mock.call_args.kwargs["orchestrators"] == [
            "https://orch-a.example.com",
            "https://orch-b.example.com",
        ]
        assert runner_selector_mock.call_args.kwargs["discovery_url"] is None

    async def test_start_scope_uses_orch_url_before_token_discovery(self) -> None:
        token = base64.b64encode(
            json.dumps({"discovery": "https://token.example.com/discovery"}).encode(
                "utf-8"
            )
        ).decode("utf-8")

        _result, runner_selector_mock, _from_json_mock = await self._run_start_scope(
            orch_url="https://orch.example.com",
            discovery_url="https://explicit.example.com/discovery",
            token=token,
        )

        assert (
            runner_selector_mock.call_args.kwargs["orchestrators"]
            == "https://orch.example.com"
        )
        assert (
            runner_selector_mock.call_args.kwargs["discovery_url"]
            == "https://token.example.com/discovery"
        )

    async def test_start_scope_propagates_runner_rejections(self) -> None:
        class _Cursor:
            async def next(self) -> LiveRunnerCallResult:
                raise NoRunnerAvailableError(
                    "All runners failed (1 tried)",
                    rejections=[
                        RunnerRejection(
                            url="https://runner.example.com/app", reason="capacity"
                        )
                    ],
                )

        with mock.patch.object(
            scope_mod, "runner_selector", mock.AsyncMock(return_value=_Cursor())
        ):
            with mock.patch.object(scope_mod._LOG, "info") as log_mock:
                with pytest.raises(NoRunnerAvailableError) as raised:
                    await scope_mod.start_scope(
                        None,
                        lv2v_mod.StartJobRequest(),
                        discovery_url="https://discovery.example.com",
                    )

        assert raised.value.rejections[0].reason == "capacity"
        log_mock.assert_called_once_with(
            "scope runner rejected: %s: %s",
            "https://runner.example.com/app",
            "capacity",
        )

    async def test_start_scope_rejects_missing_manifest_id(self) -> None:
        runner = LiveRunnerInstance(
            url="https://runner.example.com/app",
            app="live-video-to-video/scope",
            runner_id="runner-1",
            mode="",
            orchestrator_url="https://orch.example.com",
            raw={"version": "serverless-1.0.0"},
        )

        class _Cursor:
            def __init__(self) -> None:
                self.rejections: list[RunnerRejection] = []
                self.calls = 0

            async def next(self) -> LiveRunnerCallResult:
                self.calls += 1
                if self.calls == 1:
                    return LiveRunnerCallResult(
                        {"publish_url": "https://example.com/in"},
                        runner_url=runner.url,
                        runner=runner,
                    )
                raise NoRunnerAvailableError(
                    "All runners failed (1 tried)",
                    rejections=list(self.rejections),
                )

        cursor = _Cursor()
        job = SimpleNamespace(
            manifest_id=None,
            control=None,
            start_payment_sender=mock.Mock(),
        )
        with (
            mock.patch.object(
                scope_mod, "runner_selector", mock.AsyncMock(return_value=cursor)
            ),
            mock.patch.object(
                lv2v_mod.LiveVideoToVideo, "from_json", mock.Mock(return_value=job)
            ),
        ):
            with pytest.raises(NoRunnerAvailableError) as raised:
                await scope_mod.start_scope(
                    None,
                    lv2v_mod.StartJobRequest(),
                    discovery_url="https://discovery.example.com",
                )

        assert len(raised.value.rejections) == 1
        assert "missing manifest_id" in raised.value.rejections[0].reason
