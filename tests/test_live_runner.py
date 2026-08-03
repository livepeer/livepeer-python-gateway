from __future__ import annotations

import asyncio
import json
import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from livepeer_gateway import live_runner
from livepeer_gateway.errors import (
    LivepeerGatewayError,
    LivepeerHTTPError,
    SignerRefreshRequired,
)
from livepeer_gateway.live_runner import (
    call_runner,
    LiveRunnerGPU,
    LiveRunnerInstance,
    LiveRunnerPriceInfo,
    LiveRunnerRegistration,
    LiveRunnerSessionEvent,
    register_runner,
    stop_runner_session,
    create_proxy,
)
from livepeer_gateway.remote_signer import LivePaymentChallenge


class TestLiveRunnerHelpers:
    def test_join_endpoint_preserves_base_path(self) -> None:
        assert (
            live_runner._join_endpoint(
                "http://orch.example.com/base/path", "/runners/heartbeat"
            )
            == "http://orch.example.com/base/path/runners/heartbeat"
        )
        assert (
            live_runner._join_endpoint(
                "orch.example.com:8935/base", "/runners/heartbeat"
            )
            == "https://orch.example.com:8935/base/runners/heartbeat"
        )

    def test_payment_challenge_uses_server_supplied_url(self) -> None:
        body = json.dumps(
            {
                "payment_params": "opaque-payment-params",
                "manifest_id": "manifest-1",
                "payment_url": _payment_url("manifest-1"),
            }
        )

        challenge = live_runner._parse_runner_payment_challenge(
            LivepeerHTTPError(402, "https://runner.example.com", body)
        )

        assert challenge == _payment_challenge("manifest-1")

    @pytest.mark.parametrize("payment_url", [None, ""])
    def test_payment_challenge_requires_payment_url(
        self, payment_url: str | None
    ) -> None:
        body = json.dumps(
            {
                "payment_params": "opaque-payment-params",
                "manifest_id": "manifest-1",
                "payment_url": payment_url,
            }
        )

        with pytest.raises(LivepeerGatewayError, match="missing payment_url"):
            live_runner._parse_runner_payment_challenge(
                LivepeerHTTPError(402, "https://runner.example.com", body)
            )

    def test_parse_go_duration(self) -> None:
        assert live_runner._parse_go_duration_s("500ms", default=5.0) == 0.5
        assert live_runner._parse_go_duration_s("5s", default=1.0) == 5.0
        assert live_runner._parse_go_duration_s("1m", default=1.0) == 60.0
        assert live_runner._parse_go_duration_s("nope", default=7.0) == 7.0
        assert live_runner._parse_go_duration_s("", default=None) is None

    @pytest.mark.parametrize(
        ("unit", "expected"),
        [
            ("hour", "live"),
            ("seconds", "live"),
            ("720p", "lv2v"),
            ("720p-pixel-seconds", "lv2v"),
            ("fixed", "fixed"),
        ],
    )
    def test_runner_payment_type_uses_explicit_and_discovered_units(
        self, unit: str, expected: str
    ) -> None:
        assert live_runner._runner_payment_type(None, f" {unit.upper()} ") == expected

        runner = LiveRunnerInstance(
            url="https://service.example.com/apps/runner/session",
            app="livepeer/app",
            runner_id="runner",
            mode="single-shot",
            orchestrator_url="https://service.example.com",
            raw={},
            price_info=LiveRunnerPriceInfo(10, "usd", "fixed"),
        )
        assert live_runner._runner_payment_type(runner, " FIXED ") == "fixed"
        with pytest.raises(
            LivepeerGatewayError,
            match="payment_unit conflicts with runner price metadata",
        ):
            live_runner._runner_payment_type(runner, "hour")


class TestLiveRunnerSession:
    async def test_call_runner_returns_json_and_metadata(self) -> None:
        calls: list[
            tuple[
                str, str | None, dict[str, object] | None, dict[str, str] | None, float
            ]
        ] = []

        def _request_body(
            url: str,
            *,
            method: str | None = None,
            payload: dict[str, object] | None = None,
            headers: dict[str, str] | None = None,
            timeout: float,
        ) -> tuple[bytes, str]:
            calls.append((url, method, payload, headers, timeout))
            return _json_data({"session_id": "session-1", "ok": "true"})

        with mock.patch.object(live_runner, "_request_body", side_effect=_request_body):
            result = await call_runner(
                "https://service.example.com/apps/runner-1/app",
                payload={"hello": "world"},
                method="PUT",
                timeout=9.0,
            )

        assert calls == [
            (
                "https://service.example.com/apps/runner-1/app",
                "PUT",
                {"hello": "world"},
                {"Accept": "*/*"},
                9.0,
            )
        ]
        assert result.data == {"session_id": "session-1", "ok": "true"}
        assert result.runner_url == "https://service.example.com/apps/runner-1/app"
        assert result.session_id == "session-1"

    async def test_stop_runner_session_uses_discovered_runner_url(self) -> None:
        stopped: list[tuple[str, dict[str, str], float]] = []

        def _post_empty(url: str, headers: dict[str, str], timeout: float) -> None:
            stopped.append((url, headers, timeout))

        session = live_runner.LiveRunnerSession(
            session_id="session-1",
            app_url="https://service.example.com/app",
            runner_url="https://service.example.com/apps/runner-1/session",
            control_url="https://service.example.com/apps/runner-1/session/session-1",
        )

        with mock.patch.object(live_runner, "_post_empty", side_effect=_post_empty):
            await stop_runner_session(session)

        assert stopped == [
            (
                "https://service.example.com/apps/runner-1/session/session-1/stop",
                {},
                5.0,
            )
        ]

    async def test_stop_runner_session_accepts_request_control_header(self) -> None:
        stopped: list[tuple[str, dict[str, str], float]] = []

        def _post_empty(url: str, headers: dict[str, str], timeout: float) -> None:
            stopped.append((url, headers, timeout))

        request = SimpleNamespace(
            headers={
                "Livepeer-Session-Control": "https://service.example.com/api/runner/runner-1/session/session-1",
                "Livepeer-Session-Token": "session-token",
            }
        )

        with mock.patch.object(live_runner, "_post_empty", side_effect=_post_empty):
            await stop_runner_session(request, timeout=12.0)

        assert stopped == [
            (
                "https://service.example.com/api/runner/runner-1/session/session-1/stop",
                {"Livepeer-Session-Token": "session-token"},
                12.0,
            )
        ]

    async def test_stop_runner_session_request_requires_control_header(self) -> None:
        request = SimpleNamespace(
            headers={
                "Livepeer-Session-Id": "session-1",
                "Livepeer-Session-Token": "session-token",
            }
        )

        with pytest.raises(LivepeerGatewayError):
            await stop_runner_session(request)

    async def test_call_runner_can_attach_runner_instance(self) -> None:
        runner = LiveRunnerInstance(
            url="https://service.example.com/apps/runner-1/session",
            app="livepeer-sample/echo",
            runner_id="runner-1",
            mode="persistent",
            orchestrator_url="https://service.example.com",
            raw={"label": "echo"},
        )

        def _request_body(
            url: str,
            *,
            method: str | None = None,
            payload: dict[str, object] | None = None,
            headers: dict[str, str] | None = None,
            timeout: float,
        ) -> tuple[bytes, str]:
            del method, payload, timeout
            assert headers == {"Accept": "*/*"}
            return _json_data(
                {
                    "session_id": "session-1",
                    "app_url": "https://service.example.com/app",
                }
            )

        with mock.patch.object(live_runner, "_request_body", side_effect=_request_body):
            result = await call_runner(runner=runner)

        assert result.runner is runner

    async def test_paid_call_retries_with_payment_headers(self) -> None:
        calls: list[
            tuple[
                str, str | None, dict[str, object] | None, dict[str, str] | None, float
            ]
        ] = []
        sessions: list[dict[str, object]] = []
        payment_sessions: list[object] = []
        runner_url = "https://service.example.com/apps/runner-1/session"

        class _PaymentSession:
            def __init__(self, signer_url: str, **kwargs: object) -> None:
                payment_sessions.append(self)
                sessions.append({"signer_url": signer_url, **kwargs})

            async def get_payment(self) -> object:
                return SimpleNamespace(payment="payment-b64", seg_creds="seg-b64")

        def _request_body(
            url: str,
            *,
            method: str | None = None,
            payload: dict[str, object] | None = None,
            headers: dict[str, str] | None = None,
            timeout: float,
        ) -> tuple[bytes, str]:
            calls.append((url, method, payload, headers, timeout))
            if len([call for call in calls if call[0] == runner_url]) == 1:
                body = _payment_challenge_body("manifest-1")
                raise LivepeerHTTPError(402, url, body, "payment required")
            return _json_data(
                {
                    "session_id": "session-1",
                    "app_url": "https://service.example.com/app",
                }
            )

        with (
            mock.patch.object(live_runner, "_request_body", side_effect=_request_body),
            mock.patch.object(live_runner, "LivePaymentSession", _PaymentSession),
            mock.patch.object(
                live_runner,
                "get_signer_info",
                new_callable=mock.AsyncMock,
                return_value=SimpleNamespace(
                    address="opaque-payer", sig="opaque-signature"
                ),
            ) as sig_mock,
        ):
            result = await call_runner(
                runner_url,
                payload={"prompt": "hi"},
                method="PATCH",
                signer_url="https://signer.example.com",
                signer_headers={"Authorization": "token"},
            )

        assert result.data == {
            "session_id": "session-1",
            "app_url": "https://service.example.com/app",
        }
        assert result.session_id == "manifest-1"
        assert len(calls) == 2
        assert calls[0][1] == "PATCH"
        assert calls[1][1] == "PATCH"
        assert calls[0][2] == {"prompt": "hi"}
        assert calls[1][2] == {"prompt": "hi"}
        assert calls[0][3] == {
            "Accept": "*/*",
            "Livepeer-Payer-Address": "opaque-payer",
        }
        assert sessions == [
            {
                "signer_url": "https://signer.example.com",
                "signer_headers": {"Authorization": "token"},
                "type": "live",
                "app": None,
                "challenge": _payment_challenge("manifest-1"),
            }
        ]
        assert result.payment_session is payment_sessions[0]
        assert calls[1][3] == {
            "Accept": "*/*",
            "Livepeer-Payer-Address": "opaque-payer",
            "Livepeer-Payment": "payment-b64",
            "Livepeer-Segment": "seg-b64",
        }
        assert sig_mock.call_count == 1

    async def test_paid_scope_runner_uses_lv2v_payment_type(self) -> None:
        sessions: list[dict[str, object]] = []
        runner_url = "https://service.example.com/apps/scope/session"
        runner = LiveRunnerInstance(
            url=runner_url,
            app="live-video-to-video/scope",
            runner_id="scope-runner",
            mode="single-shot",
            orchestrator_url="https://service.example.com",
            raw={},
        )

        class _PaymentSession:
            def __init__(self, signer_url: str, **kwargs: object) -> None:
                sessions.append({"signer_url": signer_url, **kwargs})

            async def get_payment(self) -> object:
                return SimpleNamespace(payment="payment-b64", seg_creds="seg-b64")

        def _request_body(
            url: str,
            *,
            method: str | None = None,
            payload: dict[str, object] | None = None,
            headers: dict[str, str] | None = None,
            timeout: float,
        ) -> tuple[bytes, str]:
            del method, payload, timeout
            if headers and "Livepeer-Payment" in headers:
                return _json_data({"session_id": "session-1"})
            raise LivepeerHTTPError(
                402, url, _payment_challenge_body("manifest-scope"), "payment required"
            )

        with (
            mock.patch.object(live_runner, "_request_body", side_effect=_request_body),
            mock.patch.object(live_runner, "LivePaymentSession", _PaymentSession),
            mock.patch.object(
                live_runner,
                "get_signer_info",
                new_callable=mock.AsyncMock,
                return_value=SimpleNamespace(
                    address="opaque-payer", sig="opaque-signature"
                ),
            ),
        ):
            result = await call_runner(
                runner=runner,
                signer_url="https://signer.example.com",
            )

        assert result.session_id == "manifest-scope"
        assert sessions == [
            {
                "signer_url": "https://signer.example.com",
                "signer_headers": None,
                "type": "lv2v",
                "app": "live-video-to-video/scope",
                "challenge": _payment_challenge("manifest-scope"),
            }
        ]

    async def test_paid_fixed_runner_retries_with_fresh_payment_and_no_renewal_session(
        self,
    ) -> None:
        sessions: list[dict[str, object]] = []
        runner_url = "https://service.example.com/apps/fixed/session"
        runner = LiveRunnerInstance(
            url=runner_url,
            app="livepeer/fixed-app",
            runner_id="fixed-runner",
            mode="persistent",
            orchestrator_url="https://service.example.com",
            raw={},
            price_info=LiveRunnerPriceInfo(10, "wei", "fixed"),
        )
        runner_calls = 0

        class _PaymentSession:
            def __init__(self, signer_url: str, **kwargs: object) -> None:
                sessions.append({"signer_url": signer_url, **kwargs})

            async def get_payment(self) -> object:
                payment_number = len(sessions)
                return SimpleNamespace(
                    payment=f"fixed-payment-{payment_number}",
                    seg_creds=f"fixed-segment-{payment_number}",
                )

        def _request_body(
            url: str,
            *,
            method: str | None = None,
            payload: dict[str, object] | None = None,
            headers: dict[str, str] | None = None,
            timeout: float,
        ) -> tuple[bytes, str]:
            nonlocal runner_calls
            del method, payload, timeout
            runner_calls += 1
            if runner_calls < 3:
                raise LivepeerHTTPError(
                    402,
                    url,
                    _payment_challenge_body("fixed-manifest"),
                    "payment required",
                )
            assert headers["Livepeer-Payment"] == "fixed-payment-2"
            return _json_data(
                {
                    "session_id": "fixed-manifest",
                    "app_url": "https://service.example.com/app",
                }
            )

        with (
            mock.patch.object(live_runner, "_request_body", side_effect=_request_body),
            mock.patch.object(live_runner, "LivePaymentSession", _PaymentSession),
            mock.patch.object(
                live_runner,
                "get_signer_info",
                new_callable=mock.AsyncMock,
                return_value=SimpleNamespace(
                    address="opaque-payer", sig="opaque-signature"
                ),
            ),
        ):
            result = await call_runner(
                runner=runner, signer_url="https://signer.example.com"
            )

        assert runner_calls == 3
        assert len(sessions) == 2
        assert sessions[0]["type"] == "fixed"
        assert sessions[1]["type"] == "fixed"
        assert sessions[0]["challenge"] == _payment_challenge("fixed-manifest")
        assert sessions[1]["challenge"] == _payment_challenge("fixed-manifest")
        assert result.payment_session is None

    async def test_paid_call_restarts_challenge_when_signer_requests_refresh(
        self,
    ) -> None:
        calls: list[
            tuple[str, str | None, dict[str, object] | None, dict[str, str] | None]
        ] = []
        sessions: list[dict[str, object]] = []
        payment_sessions: list[object] = []
        payment_attempts = 0
        unpaid_count = 0
        runner_url = "https://service.example.com/apps/runner-1/session"

        class _PaymentSession:
            def __init__(self, signer_url: str, **kwargs: object) -> None:
                payment_sessions.append(self)
                sessions.append({"signer_url": signer_url, **kwargs})

            async def get_payment(self) -> object:
                nonlocal payment_attempts
                payment_attempts += 1
                if payment_attempts == 1:
                    raise SignerRefreshRequired("refresh")
                return SimpleNamespace(payment="payment-2", seg_creds="seg-2")

        def _request_body(
            url: str,
            *,
            method: str | None = None,
            payload: dict[str, object] | None = None,
            headers: dict[str, str] | None = None,
            timeout: float,
        ) -> tuple[bytes, str]:
            nonlocal unpaid_count
            del timeout
            calls.append((url, method, payload, headers))
            if headers and "Livepeer-Payment" in headers:
                return _json_data(
                    {
                        "session_id": "session-2",
                        "app_url": "https://service.example.com/app",
                    }
                )
            unpaid_count += 1
            raise LivepeerHTTPError(
                402,
                url,
                _payment_challenge_body(f"manifest-{unpaid_count}"),
                "payment required",
            )

        with (
            mock.patch.object(live_runner, "_request_body", side_effect=_request_body),
            mock.patch.object(live_runner, "LivePaymentSession", _PaymentSession),
            mock.patch.object(
                live_runner,
                "get_signer_info",
                new_callable=mock.AsyncMock,
                return_value=SimpleNamespace(
                    address="opaque-payer", sig="opaque-signature"
                ),
            ) as sig_mock,
        ):
            result = await call_runner(
                runner_url,
                signer_url="https://signer.example.com",
            )

        assert result.data["session_id"] == "session-2"
        assert result.session_id == "manifest-2"
        assert result.payment_session is payment_sessions[1]
        challenge_headers = {
            "Accept": "*/*",
            "Livepeer-Payer-Address": "opaque-payer",
        }
        assert [headers for url, _, _, headers in calls if url == runner_url] == [
            challenge_headers,
            challenge_headers,
            {
                "Accept": "*/*",
                "Livepeer-Payer-Address": "opaque-payer",
                "Livepeer-Payment": "payment-2",
                "Livepeer-Segment": "seg-2",
            },
        ]
        assert payment_attempts == 2
        assert sessions == [
            {
                "signer_url": "https://signer.example.com",
                "signer_headers": None,
                "type": "live",
                "app": None,
                "challenge": _payment_challenge("manifest-1"),
            },
            {
                "signer_url": "https://signer.example.com",
                "signer_headers": None,
                "type": "live",
                "app": None,
                "challenge": _payment_challenge("manifest-2"),
            },
        ]
        assert sig_mock.call_count == 1

    async def test_paid_call_repeated_refresh_requests_are_bounded(self) -> None:
        calls: list[dict[str, str] | None] = []
        payment_attempts = 0
        unpaid_count = 0
        runner_url = "https://service.example.com/apps/runner-1/session"

        class _PaymentSession:
            def __init__(self, signer_url: str, **kwargs: object) -> None:
                del signer_url, kwargs

            async def get_payment(self) -> object:
                nonlocal payment_attempts
                payment_attempts += 1
                raise SignerRefreshRequired("fixed price not found for session")

        def _request_body(
            url: str,
            *,
            method: str | None = None,
            payload: dict[str, object] | None = None,
            headers: dict[str, str] | None = None,
            timeout: float,
        ) -> tuple[bytes, str]:
            nonlocal unpaid_count
            del method, payload, timeout
            calls.append(headers)
            unpaid_count += 1
            raise LivepeerHTTPError(
                402,
                url,
                _payment_challenge_body(f"manifest-{unpaid_count}"),
                "payment required",
            )

        with (
            mock.patch.object(live_runner, "_request_body", side_effect=_request_body),
            mock.patch.object(live_runner, "LivePaymentSession", _PaymentSession),
            mock.patch.object(
                live_runner,
                "get_signer_info",
                new_callable=mock.AsyncMock,
                return_value=SimpleNamespace(
                    address="opaque-payer", sig="opaque-signature"
                ),
            ),
        ):
            with pytest.raises(SignerRefreshRequired, match="fixed price not found"):
                await call_runner(
                    runner_url,
                    signer_url="https://signer.example.com",
                    max_payment_challenge_retries=1,
                )

        assert calls == [
            {"Accept": "*/*", "Livepeer-Payer-Address": "opaque-payer"},
            {"Accept": "*/*", "Livepeer-Payer-Address": "opaque-payer"},
        ]
        assert unpaid_count == 2
        assert payment_attempts == 2


def _payment_challenge_body(manifest_id: str) -> str:
    return json.dumps(
        {
            "payment_params": "opaque-payment-params",
            "orchestrator": "https://orchestrator.example.com",
            "manifest_id": manifest_id,
            "payment_url": _payment_url(manifest_id),
        }
    )


def _payment_challenge(manifest_id: str) -> LivePaymentChallenge:
    return LivePaymentChallenge(
        payment_params="opaque-payment-params",
        manifest_id=manifest_id,
        payment_url=_payment_url(manifest_id),
    )


def _payment_url(manifest_id: str) -> str:
    return (
        "https://orchestrator.example.com/apps/runner-1/session/"
        f"{manifest_id}/payment"
    )


def _json_data(data: dict[str, object]) -> tuple[bytes, str]:
    return json.dumps(data).encode("utf-8"), "application/json"


class _FakeO2RReader:
    instances: list[_FakeO2RReader] = []

    def __init__(
        self,
        events_url: str,
        *,
        start_seq: int,
        on_event: object,
        **_kwargs: object,
    ) -> None:
        self.events_url = events_url
        self.start_seq = start_seq
        self.on_event = on_event
        self.closed = False
        self._task = asyncio.create_task(asyncio.Event().wait())
        type(self).instances.append(self)

    def callback_task(self) -> asyncio.Task[None]:
        return self._task

    async def close(self, **_kwargs: object) -> None:
        self.closed = True
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)


class TestLiveRunnerRegistration:
    async def test_initial_heartbeat_starts_o2r_channel_reader_callback_once(
        self,
    ) -> None:
        _FakeO2RReader.instances = []
        calls: list[dict[str, object]] = []
        second_heartbeat = asyncio.Event()

        def _post_json(
            url: str,
            payload: dict[str, object],
            *,
            headers: dict[str, str],
            timeout: float,
        ) -> dict[str, object]:
            del url, headers, timeout
            calls.append(payload)
            if len(calls) >= 2:
                second_heartbeat.set()
            response: dict[str, object] = {
                "runner_id": "runner-1",
                "heartbeat_interval": "1h",
                "heartbeat_secret": "heartbeat-token",
            }
            if len(calls) == 1:
                response["o2r"] = {
                    "name": "o2r",
                    "channel_name": "runner-1-secret-o2r",
                    "url": "https://service.example.com/trickle/runner-1-secret-o2r",
                    "mime_type": "application/octet-stream",
                }
            return response

        with (
            mock.patch.object(live_runner, "post_json", side_effect=_post_json),
            mock.patch.object(live_runner, "ChannelReader", _FakeO2RReader),
        ):
            reg = await register_runner(
                "http://orch.example.com",
                secret="secret-token",
                runner_url="https://runner.example.com",
                app="live-video-to-video/scope",
                auto_detect_gpu=False,
                heartbeat_interval_s=0.01,
                unregister_on_close=False,
            )
            await asyncio.wait_for(second_heartbeat.wait(), timeout=1.0)
            await reg.close()

        assert len(_FakeO2RReader.instances) == 1
        reader = _FakeO2RReader.instances[0]
        assert (
            reader.events_url
            == "https://service.example.com/trickle/runner-1-secret-o2r"
        )
        assert reader.start_seq == 0
        assert callable(reader.on_event)
        assert reg.o2r_channel["name"] == "o2r"
        assert reader.closed
        assert calls[0]["session_ids"] == []
        assert calls[1]["session_ids"] == []

    async def test_o2r_session_messages_track_active_ids_by_age_and_invoke_callbacks(
        self,
    ) -> None:
        _FakeO2RReader.instances = []
        sync_events: list[LiveRunnerSessionEvent] = []
        async_events: list[LiveRunnerSessionEvent] = []
        heartbeat_payloads: list[dict[str, object]] = []
        sessions_advertised = asyncio.Event()

        def _on_reserve(event: LiveRunnerSessionEvent) -> None:
            sync_events.append(event)

        async def _on_release(event: LiveRunnerSessionEvent) -> None:
            async_events.append(event)

        def _post_json(
            url: str,
            payload: dict[str, object],
            *,
            headers: dict[str, str],
            timeout: float,
        ) -> dict[str, object]:
            del url, headers, timeout
            heartbeat_payloads.append(payload)
            if payload.get("session_ids") == ["session-a"]:
                sessions_advertised.set()
            response: dict[str, object] = {
                "runner_id": "runner-1",
                "heartbeat_interval": "1h",
                "heartbeat_secret": "heartbeat-token",
            }
            if len(heartbeat_payloads) == 1:
                response["o2r"] = {
                    "name": "o2r",
                    "channel_name": "runner-1-secret-o2r",
                    "url": "https://service.example.com/trickle/runner-1-secret-o2r",
                    "mime_type": "application/octet-stream",
                }
            return response

        with (
            mock.patch.object(live_runner, "post_json", side_effect=_post_json),
            mock.patch.object(live_runner, "ChannelReader", _FakeO2RReader),
        ):
            reg = await register_runner(
                "http://localhost:8935",
                secret="secret-token",
                runner_url="http://localhost:9000",
                app="live-video-to-video/scope",
                price=10,
                auto_detect_gpu=False,
                heartbeat_interval_s=0.01,
                unregister_on_close=False,
                on_session_reserve=_on_reserve,
                on_session_release=_on_release,
            )
            reader = _FakeO2RReader.instances[0]
            on_event = reader.on_event
            assert callable(on_event)

            await on_event({"keep": "alive"})
            await on_event(
                {
                    "event": "reserved",
                    "session": "session-b",
                    "timestamp": "2026-05-20T17:00:00Z",
                }
            )
            await on_event({"event": "reserved", "session": "session-a"})
            await on_event({"event": "reserved", "session": "session-b"})
            assert reg.active_session_ids == ("session-b", "session-a")

            await on_event(
                {
                    "event": "released",
                    "session": "session-b",
                    "timestamp": "2026-05-20T17:01:00Z",
                }
            )
            assert reg.active_session_ids == ("session-a",)

            await on_event({"event": "released", "session": "missing"})
            assert reg.active_session_ids == ("session-a",)
            await asyncio.wait_for(sessions_advertised.wait(), timeout=1.0)
            await reg.close()

        assert reg.active_session_ids == ("session-a",)
        assert heartbeat_payloads[0]["session_ids"] == []
        assert ["session-a"] in [
            payload["session_ids"] for payload in heartbeat_payloads
        ]
        assert [event.session_id for event in sync_events] == [
            "session-b",
            "session-a",
            "session-b",
        ]
        assert sync_events[0].event == "reserved"
        assert sync_events[0].timestamp == "2026-05-20T17:00:00Z"
        assert [event.session_id for event in async_events] == ["session-b", "missing"]
        assert async_events[0].event == "released"

    async def test_o2r_unknown_messages_are_ignored(self, caplog) -> None:
        _FakeO2RReader.instances = []
        events: list[LiveRunnerSessionEvent] = []
        response = {
            "runner_id": "runner-1",
            "heartbeat_interval": "1h",
            "heartbeat_secret": "heartbeat-token",
            "o2r": {
                "name": "o2r",
                "channel_name": "runner-1-secret-o2r",
                "url": "https://service.example.com/trickle/runner-1-secret-o2r",
                "mime_type": "application/octet-stream",
            },
        }
        with (
            mock.patch.object(live_runner, "post_json", return_value=response),
            mock.patch.object(live_runner, "ChannelReader", _FakeO2RReader),
        ):
            reg = await register_runner(
                "http://localhost:8935",
                secret="secret-token",
                runner_url="http://localhost:9000",
                app="live-video-to-video/scope",
                price=10,
                auto_detect_gpu=False,
                unregister_on_close=False,
                on_session_reserve=events.append,
                on_session_release=events.append,
            )
            on_event = _FakeO2RReader.instances[0].on_event
            with caplog.at_level("WARNING", logger=live_runner._LOG.name):
                await on_event({"event": "reserved"})
                await on_event({"event": "updated", "session": "session-1"})
            await reg.close()

        assert reg.active_session_ids == ()
        assert events == []
        assert len(caplog.records) == 2

    async def test_register_sends_payload_and_reuses_runner_id_with_returned_orchestrator(
        self, caplog
    ) -> None:
        calls: list[tuple[str, dict[str, object], dict[str, str]]] = []
        second_heartbeat = asyncio.Event()

        def _post_json(
            url: str,
            payload: dict[str, object],
            *,
            headers: dict[str, str],
            timeout: float,
        ) -> dict[str, str]:
            calls.append((url, payload, headers))
            if len(calls) == 1:
                return {
                    "runner_id": "runner-1",
                    "orchestrator": "https://service.example.com/orch-base",
                    "heartbeat_interval": "10s",
                    "heartbeat_ttl": "20s",
                    "heartbeat_secret": "heartbeat-token",
                }
            second_heartbeat.set()
            return {
                "runner_id": "runner-1",
                "orchestrator": "https://service.example.com/orch-base",
                "heartbeat_interval": "10s",
                "heartbeat_ttl": "20s",
            }

        with (
            mock.patch.object(live_runner, "post_json", side_effect=_post_json),
            mock.patch.object(
                live_runner,
                "detect_process_gpu",
            ) as detect_gpu_mock,
        ):
            with caplog.at_level("INFO", logger=live_runner._LOG.name):
                reg = await register_runner(
                    "https://initial.example.com/api",
                    secret="secret-token",
                    runner_url="https://runner.example.com",
                    app="live-video-to-video/scope",
                    price=10,
                    currency=" WEI ",
                    unit=" CUSTOM ",
                    proxy=True,
                    metadata='{"region":"us-west","tier":"warm"}',
                    gpu=LiveRunnerGPU(id="gpu-1", name="NVIDIA L40S", vram_mb=46068),
                    heartbeat_interval_s=0.01,
                    unregister_on_close=False,
                )
                await asyncio.wait_for(second_heartbeat.wait(), timeout=1.0)
                await reg.close()

        assert len(calls) >= 2
        assert calls[0][0] == "https://initial.example.com/api/runners/heartbeat"
        assert calls[1][0] == "https://service.example.com/orch-base/runners/heartbeat"
        assert (
            "Registering live runner with orchestrator https://initial.example.com/api"
            in caplog.messages[0]
        )
        assert (
            "Live runner registration using orchestrator https://service.example.com/orch-base "
            "returned by https://initial.example.com/api" in caplog.messages[1]
        )
        assert calls[0][2] == {"Authorization": "secret-token"}
        assert calls[1][2] == {"Authorization": "heartbeat-token"}
        assert calls[0][1]["mode"] == "persistent"
        assert calls[0][1]["proxy"] is True
        assert calls[1][1]["proxy"] is True
        assert calls[0][1]["price_info"] == {
            "price": 10,
            "currency": "wei",
            "unit": "custom",
        }
        assert calls[0][1]["gpu"] == {
            "id": "gpu-1",
            "name": "NVIDIA L40S",
            "vram_mb": 46068,
        }
        assert calls[0][1]["metadata"] == '{"region":"us-west","tier":"warm"}'
        assert calls[0][1]["session_ids"] == []
        assert "runner_id" not in calls[0][1]
        assert calls[1][1]["runner_id"] == "runner-1"
        assert calls[1][1]["metadata"] == '{"region":"us-west","tier":"warm"}'
        assert reg.runner_id == "runner-1"
        assert reg.orchestrator_url == "https://service.example.com/orch-base"
        assert reg.heartbeat_ttl_s == 20.0
        detect_gpu_mock.assert_not_called()

    async def test_register_uses_server_interval_when_no_override(self) -> None:
        with mock.patch.object(
            live_runner,
            "post_json",
            return_value={
                "runner_id": "runner-1",
                "heartbeat_interval": "500ms",
                "heartbeat_ttl": "1m",
                "heartbeat_secret": "heartbeat-token",
            },
        ):
            reg = await register_runner(
                "http://orch.example.com",
                secret="secret-token",
                runner_url="https://runner.example.com",
                app="live-video-to-video/scope",
                price=10,
                auto_detect_gpu=False,
            )
            await reg.close()

        assert reg.heartbeat_interval_s == 0.5
        assert reg.heartbeat_ttl_s == 60.0

    async def test_register_can_advertise_single_shot_mode(self) -> None:
        calls: list[tuple[str, dict[str, object], dict[str, str]]] = []

        def _post_json(
            url: str,
            payload: dict[str, object],
            *,
            headers: dict[str, str],
            timeout: float,
        ) -> dict[str, str]:
            calls.append((url, payload, headers))
            return {
                "runner_id": "runner-1",
                "heartbeat_interval": "1h",
                "heartbeat_secret": "heartbeat-token",
            }

        with mock.patch.object(live_runner, "post_json", side_effect=_post_json):
            reg = await register_runner(
                "http://orch.example.com",
                secret="secret-token",
                runner_url="https://runner.example.com",
                app="livepeer-sample/websocket-pingpong",
                mode="single-shot",
                auto_detect_gpu=False,
            )
            await reg.close()

        assert calls[0][1]["mode"] == "single-shot"

    async def test_register_rejects_invalid_mode_before_heartbeat(self) -> None:
        with mock.patch.object(live_runner, "post_json") as post_json_mock:
            with pytest.raises(ValueError):
                await register_runner(
                    "http://orch.example.com",
                    secret="secret-token",
                    runner_url="https://runner.example.com",
                    app="livepeer-sample/websocket-pingpong",
                    mode="stream",
                    auto_detect_gpu=False,
                )

        post_json_mock.assert_not_called()

    async def test_close_unregisters_with_canonical_orchestrator_path(self) -> None:
        unregistered: list[tuple[str, dict[str, str], float]] = []

        def _post_empty(url: str, headers: dict[str, str], timeout: float) -> None:
            unregistered.append((url, headers, timeout))

        with mock.patch.object(
            live_runner,
            "post_json",
            return_value={
                "runner_id": "runner-1",
                "orchestrator": "https://service.example.com/api",
                "heartbeat_interval": "1h",
                "heartbeat_secret": "heartbeat-token",
            },
        ):
            with mock.patch.object(live_runner, "_post_empty", side_effect=_post_empty):
                reg = await register_runner(
                    "http://orch.example.com",
                    secret="secret-token",
                    runner_url="https://runner.example.com",
                    app="live-video-to-video/scope",
                    price=10,
                    auto_detect_gpu=False,
                )
                await reg.close()

        assert unregistered == [
            (
                "https://service.example.com/api/runners/runner-1/unregister",
                {"Authorization": "heartbeat-token"},
                5.0,
            )
        ]

    async def test_initial_heartbeat_requires_heartbeat_secret(self) -> None:
        with mock.patch.object(
            live_runner,
            "post_json",
            return_value={"runner_id": "runner-1", "heartbeat_interval": "1h"},
        ):
            with pytest.raises(LivepeerGatewayError):
                await register_runner(
                    "http://orch.example.com",
                    secret="secret-token",
                    runner_url="https://runner.example.com",
                    app="live-video-to-video/scope",
                    price=10,
                    auto_detect_gpu=False,
                )

    async def test_heartbeat_resets_auth_after_invalid_authorization(self) -> None:
        calls: list[tuple[str, dict[str, object], dict[str, str]]] = []
        auth_refreshed = asyncio.Event()

        def _post_json(
            url: str,
            payload: dict[str, object],
            *,
            headers: dict[str, str],
            timeout: float,
        ) -> dict[str, str]:
            calls.append((url, payload, headers))
            if headers["Authorization"] == "heartbeat-token":
                raise LivepeerHTTPError(401, url, "any body")
            if len(calls) >= 3:
                auth_refreshed.set()
            return {
                "runner_id": "runner-1",
                "heartbeat_interval": "1h",
                "heartbeat_secret": (
                    "heartbeat-token" if len(calls) == 1 else "fresh-heartbeat-token"
                ),
            }

        with mock.patch.object(live_runner, "post_json", side_effect=_post_json):
            reg = await register_runner(
                "http://localhost:8935",
                secret="secret-token",
                runner_url="http://localhost:9000",
                app="live-video-to-video/scope",
                price=10,
                auto_detect_gpu=False,
                heartbeat_interval_s=0.01,
                unregister_on_close=False,
            )
            await asyncio.wait_for(auth_refreshed.wait(), timeout=1.0)
            await reg.close()

        assert [headers for _, _, headers in calls[:3]] == [
            {"Authorization": "secret-token"},
            {"Authorization": "heartbeat-token"},
            {"Authorization": "secret-token"},
        ]
        assert calls[2][1]["runner_id"] == "runner-1"

    async def test_heartbeat_non_401_logs_without_reset_or_traceback(
        self, caplog
    ) -> None:
        calls: list[tuple[str, dict[str, object], dict[str, str]]] = []
        heartbeat_failed = asyncio.Event()

        def _post_json(
            url: str,
            payload: dict[str, object],
            *,
            headers: dict[str, str],
            timeout: float,
        ) -> dict[str, str]:
            calls.append((url, payload, headers))
            if len(calls) == 1:
                return {
                    "runner_id": "runner-1",
                    "heartbeat_interval": "1h",
                    "heartbeat_secret": "heartbeat-token",
                }
            heartbeat_failed.set()
            raise LivepeerHTTPError(403, url, "forbidden")

        with mock.patch.object(live_runner, "post_json", side_effect=_post_json):
            with caplog.at_level("WARNING", logger=live_runner._LOG.name):
                reg = await register_runner(
                    "http://localhost:8935",
                    secret="secret-token",
                    runner_url="http://localhost:9000",
                    app="live-video-to-video/scope",
                    price=10,
                    auto_detect_gpu=False,
                    heartbeat_interval_s=0.01,
                    unregister_on_close=False,
                )
                await asyncio.wait_for(heartbeat_failed.wait(), timeout=1.0)
                await asyncio.sleep(0)
                await reg.close()

        assert len(calls) >= 2
        assert calls[0][2] == {"Authorization": "secret-token"}
        assert all(
            headers == {"Authorization": "heartbeat-token"}
            for _, _, headers in calls[1:]
        )
        assert len(caplog.records) == 1
        assert caplog.records[0].exc_info is None
        assert "http 403" in caplog.messages[0].lower()

    async def test_create_trickle_channels_sends_authenticated_payload_and_returns_channels(
        self,
    ) -> None:
        calls: list[tuple[str, dict[str, object], dict[str, str], float]] = []

        def _post_json(
            url: str,
            payload: dict[str, object],
            *,
            headers: dict[str, str],
            timeout: float,
        ) -> dict[str, object]:
            calls.append((url, payload, headers, timeout))
            return {
                "channels": [
                    {
                        "name": "foo",
                        "channel_name": "session-1-foo",
                        "url": "https://service.example.com/api/ai/trickle/session-1-foo",
                        "internal_url": "http://orchestrator:8935/api/ai/trickle/session-1-foo",
                        "mime_type": "video/MP2T",
                    }
                ]
            }

        reg = LiveRunnerRegistration(
            orchestrator_url="https://initial.example.com",
            secret="secret-token",
            runner_url="https://runner.example.com",
            app="live-video-to-video/scope",
            price_info=LiveRunnerPriceInfo(10),
            runner_id="runner/1",
            timeout=12.0,
        )
        reg.orchestrator_url = "https://service.example.com/api"

        with mock.patch.object(live_runner, "post_json", side_effect=_post_json):
            channels = await reg.create_trickle_channels(
                "session/1",
                [{"name": "foo", "mime_type": "video/MP2T"}],
                session_token="session-token",
            )

        assert len(channels) == 1
        assert channels[0]["channel_name"] == "session-1-foo"
        assert (
            channels[0]["url"]
            == "https://service.example.com/api/ai/trickle/session-1-foo"
        )
        assert (
            channels[0]["internal_url"]
            == "http://orchestrator:8935/api/ai/trickle/session-1-foo"
        )
        assert calls == [
            (
                "https://service.example.com/api/runner/runner%2F1/session/session%2F1/channels",
                {"channels": [{"name": "foo", "mime_type": "video/MP2T"}]},
                {"Livepeer-Session-Token": "session-token"},
                12.0,
            )
        ]

    async def test_create_trickle_channels_accepts_request_headers(self) -> None:
        calls: list[tuple[str, dict[str, str]]] = []

        def _post_json(
            url: str,
            payload: dict[str, object],
            *,
            headers: dict[str, str],
            timeout: float,
        ) -> dict[str, object]:
            calls.append((url, headers))
            return {
                "channels": [
                    {
                        "name": "foo",
                        "channel_name": "session-1-foo",
                        "url": "https://service.example.com/api/ai/trickle/session-1-foo",
                        "mime_type": "video/MP2T",
                    }
                ]
            }

        reg = LiveRunnerRegistration(
            orchestrator_url="https://service.example.com/api",
            secret="secret-token",
            runner_url="https://runner.example.com",
            app="live-video-to-video/scope",
            price_info=LiveRunnerPriceInfo(10),
            runner_id="runner-1",
        )
        request = SimpleNamespace(
            headers={
                "Livepeer-Session-Id": "session-1",
                "Livepeer-Session-Token": "session-token",
            }
        )

        with mock.patch.object(live_runner, "post_json", side_effect=_post_json):
            await reg.create_trickle_channels(
                request, [{"name": "foo", "mime_type": "video/MP2T"}]
            )

        assert calls == [
            (
                "https://service.example.com/api/runner/runner-1/session/session-1/channels",
                {"Livepeer-Session-Token": "session-token"},
            )
        ]

    async def test_create_trickle_channels_standalone_uses_session_control_header(
        self,
    ) -> None:
        calls: list[tuple[str, dict[str, object], dict[str, str], float]] = []

        def _post_json(
            url: str,
            payload: dict[str, object],
            *,
            headers: dict[str, str],
            timeout: float,
        ) -> dict[str, object]:
            calls.append((url, payload, headers, timeout))
            return {
                "channels": [
                    {
                        "name": "foo",
                        "channel_name": "session-1-foo",
                        "url": "https://service.example.com/api/ai/trickle/session-1-foo",
                        "mime_type": "video/MP2T",
                    }
                ]
            }

        request = SimpleNamespace(
            headers={
                "Livepeer-Session-Id": "session/1",
                "Livepeer-Session-Token": "session-token",
                "Livepeer-Session-Control": "https://service.example.com/api/runner/runner%2F1/session/session%2F1",
            }
        )

        with mock.patch.object(live_runner, "post_json", side_effect=_post_json):
            channels = await live_runner.create_trickle_channels(
                request,
                [{"name": "foo", "mime_type": "video/MP2T"}],
                timeout=12.0,
            )

        assert channels[0]["channel_name"] == "session-1-foo"
        assert calls == [
            (
                "https://service.example.com/api/runner/runner%2F1/session/session%2F1/channels",
                {"channels": [{"name": "foo", "mime_type": "video/MP2T"}]},
                {"Livepeer-Session-Token": "session-token"},
                12.0,
            )
        ]

    async def test_remove_trickle_channels_sends_authenticated_delete_payload(
        self,
    ) -> None:
        calls: list[tuple[str, str, dict[str, object], dict[str, str], float]] = []

        def _request_json(
            url: str,
            *,
            method: str,
            payload: dict[str, object],
            headers: dict[str, str],
            timeout: float,
        ) -> dict[str, object]:
            calls.append((url, method, payload, headers, timeout))
            return {"deleted": ["foo", "bar"]}

        reg = LiveRunnerRegistration(
            orchestrator_url="https://service.example.com/api",
            secret="secret-token",
            runner_url="https://runner.example.com",
            app="live-video-to-video/scope",
            price_info=LiveRunnerPriceInfo(10),
            runner_id="runner-1",
            timeout=12.0,
        )

        with mock.patch.object(live_runner, "request_json", side_effect=_request_json):
            deleted = await reg.remove_trickle_channels(
                "session-1", ["foo", "bar"], session_token="session-token"
            )

        assert deleted == ["foo", "bar"]
        assert calls == [
            (
                "https://service.example.com/api/runner/runner-1/session/session-1/channels",
                "DELETE",
                {"channels": ["foo", "bar"]},
                {"Livepeer-Session-Token": "session-token"},
                12.0,
            )
        ]

    async def test_remove_trickle_channels_accepts_request_headers(self) -> None:
        calls: list[tuple[str, dict[str, str]]] = []

        def _request_json(
            url: str,
            *,
            method: str,
            payload: dict[str, object],
            headers: dict[str, str],
            timeout: float,
        ) -> dict[str, object]:
            calls.append((url, headers))
            return {"deleted": ["foo"]}

        reg = LiveRunnerRegistration(
            orchestrator_url="https://service.example.com/api",
            secret="secret-token",
            runner_url="https://runner.example.com",
            app="live-video-to-video/scope",
            price_info=LiveRunnerPriceInfo(10),
            runner_id="runner-1",
        )
        request = SimpleNamespace(
            headers={
                "Livepeer-Session-Id": "session-1",
                "Livepeer-Session-Token": "session-token",
            }
        )

        with mock.patch.object(live_runner, "request_json", side_effect=_request_json):
            deleted = await reg.remove_trickle_channels(request, ["foo"])

        assert deleted == ["foo"]
        assert calls == [
            (
                "https://service.example.com/api/runner/runner-1/session/session-1/channels",
                {"Livepeer-Session-Token": "session-token"},
            )
        ]

    async def test_remove_trickle_channels_standalone_uses_session_control_header(
        self,
    ) -> None:
        calls: list[tuple[str, str, dict[str, object], dict[str, str], float]] = []

        def _request_json(
            url: str,
            *,
            method: str,
            payload: dict[str, object],
            headers: dict[str, str],
            timeout: float,
        ) -> dict[str, object]:
            calls.append((url, method, payload, headers, timeout))
            return {"deleted": ["foo"]}

        request = SimpleNamespace(
            headers={
                "Livepeer-Session-Id": "session-1",
                "Livepeer-Session-Token": "session-token",
                "Livepeer-Session-Control": "https://service.example.com/api/runner/runner-1/session/session-1",
            }
        )

        with mock.patch.object(live_runner, "request_json", side_effect=_request_json):
            deleted = await live_runner.remove_trickle_channels(
                request,
                ["foo"],
                timeout=12.0,
            )

        assert deleted == ["foo"]
        assert calls == [
            (
                "https://service.example.com/api/runner/runner-1/session/session-1/channels",
                "DELETE",
                {"channels": ["foo"]},
                {"Livepeer-Session-Token": "session-token"},
                12.0,
            )
        ]

    @pytest.mark.parametrize(
        ("runner_id", "session_token"),
        [("", "token"), ("runner-1", "")],
        ids=["missing-runner-id", "missing-session-token"],
    )
    async def test_trickle_channel_methods_require_runner_id_and_session_token(
        self, runner_id: str, session_token: str
    ) -> None:
        reg = LiveRunnerRegistration(
            orchestrator_url="https://service.example.com",
            secret="secret-token",
            runner_url="https://runner.example.com",
            app="live-video-to-video/scope",
            price_info=LiveRunnerPriceInfo(10),
            runner_id=runner_id,
        )

        with pytest.raises(LivepeerGatewayError):
            await reg.create_trickle_channels(
                "session-1",
                [{"name": "foo", "mime_type": "video/MP2T"}],
                session_token=session_token,
            )
        with pytest.raises(LivepeerGatewayError):
            await reg.remove_trickle_channels(
                "session-1",
                ["foo"],
                session_token=session_token,
            )

    async def test_create_trickle_channels_rejects_invalid_channel_request_shape(
        self,
    ) -> None:
        reg = LiveRunnerRegistration(
            orchestrator_url="https://service.example.com",
            secret="secret-token",
            runner_url="https://runner.example.com",
            app="live-video-to-video/scope",
            price_info=LiveRunnerPriceInfo(10),
            runner_id="runner-1",
        )

        with pytest.raises(TypeError):
            await reg.create_trickle_channels(
                "session-1", [{"name": "foo"}], session_token="token"
            )  # type: ignore[typeddict-item]
        with pytest.raises(TypeError):
            await reg.create_trickle_channels(
                "session-1", [{"name": "foo", "mime_type": 123}], session_token="token"
            )  # type: ignore[typeddict-item]

    async def test_create_trickle_channels_rejects_malformed_response(self) -> None:
        reg = LiveRunnerRegistration(
            orchestrator_url="https://service.example.com",
            secret="secret-token",
            runner_url="https://runner.example.com",
            app="live-video-to-video/scope",
            price_info=LiveRunnerPriceInfo(10),
            runner_id="runner-1",
        )

        with mock.patch.object(
            live_runner, "post_json", return_value={"channels": "not-a-list"}
        ):
            with pytest.raises(LivepeerGatewayError):
                await reg.create_trickle_channels(
                    "session-1",
                    [{"name": "foo", "mime_type": "video/MP2T"}],
                    session_token="token",
                )

        with mock.patch.object(
            live_runner,
            "post_json",
            return_value={
                "channels": [
                    {
                        "name": "foo",
                        "channel_name": "session-1-foo",
                        "url": "https://service.example.com/api/ai/trickle/session-1-foo",
                        "internal_url": 123,
                        "mime_type": "video/MP2T",
                    }
                ]
            },
        ):
            with pytest.raises(LivepeerGatewayError):
                await reg.create_trickle_channels(
                    "session-1",
                    [{"name": "foo", "mime_type": "video/MP2T"}],
                    session_token="token",
                )

    async def test_remove_trickle_channels_rejects_malformed_response(self) -> None:
        reg = LiveRunnerRegistration(
            orchestrator_url="https://service.example.com",
            secret="secret-token",
            runner_url="https://runner.example.com",
            app="live-video-to-video/scope",
            price_info=LiveRunnerPriceInfo(10),
            runner_id="runner-1",
        )

        with mock.patch.object(
            live_runner, "request_json", return_value={"deleted": "not-a-list"}
        ):
            with pytest.raises(LivepeerGatewayError):
                await reg.remove_trickle_channels(
                    "session-1", ["foo"], session_token="token"
                )

    async def test_create_proxy_sends_authenticated_payload_and_returns_proxy(
        self,
    ) -> None:
        calls: list[tuple[str, dict[str, object], dict[str, str], float]] = []

        def _post_json(
            url: str,
            payload: dict[str, object],
            *,
            headers: dict[str, str],
            timeout: float,
        ) -> dict[str, object]:
            calls.append((url, payload, headers, timeout))
            return {
                "proxy_id": "proxy-1",
                "url": "https://proxy.example.com/app",
            }

        reg = LiveRunnerRegistration(
            orchestrator_url="https://service.example.com/api",
            secret="secret-token",
            runner_url="https://runner.example.com",
            app="live-video-to-video/scope",
            price_info=LiveRunnerPriceInfo(10),
            runner_id="runner/1",
            timeout=12.0,
        )

        with mock.patch.object(live_runner, "post_json", side_effect=_post_json):
            proxy = await reg.create_proxy(
                "session/1",
                "http://runner.example.com:7860/app",
                session_token="session-token",
            )

        assert proxy.proxy_id == "proxy-1"
        assert proxy.url == "https://proxy.example.com/app"
        assert calls == [
            (
                "https://service.example.com/api/runner/runner%2F1/session/session%2F1/proxy",
                {"target_url": "http://runner.example.com:7860/app"},
                {"Livepeer-Session-Token": "session-token"},
                12.0,
            )
        ]

    async def test_create_proxy_standalone_uses_session_control_header(self) -> None:
        calls: list[tuple[str, dict[str, object], dict[str, str], float]] = []

        def _post_json(
            url: str,
            payload: dict[str, object],
            *,
            headers: dict[str, str],
            timeout: float,
        ) -> dict[str, object]:
            calls.append((url, payload, headers, timeout))
            return {
                "proxy_id": "proxy-2",
                "url": "https://proxy.example.com/2",
            }

        request = SimpleNamespace(
            headers={
                "Livepeer-Session-Id": "session/1",
                "Livepeer-Session-Token": "session-token",
                "Livepeer-Session-Control": "https://service.example.com/api/runner/runner%2F1/session/session%2F1",
            }
        )

        with mock.patch.object(live_runner, "post_json", side_effect=_post_json):
            proxy = await create_proxy(
                request,
                "http://runner.example.com:7860/app",
                timeout=12.0,
            )

        assert proxy.proxy_id == "proxy-2"
        assert calls == [
            (
                "https://service.example.com/api/runner/runner%2F1/session/session%2F1/proxy",
                {"target_url": "http://runner.example.com:7860/app"},
                {"Livepeer-Session-Token": "session-token"},
                12.0,
            )
        ]

    @pytest.mark.parametrize(
        "target_url",
        [None, " \t "],
        ids=["missing", "blank"],
    )
    async def test_create_proxy_omits_missing_or_blank_target_url(
        self, target_url: str | None
    ) -> None:
        calls: list[tuple[str, dict[str, object], dict[str, str], float]] = []

        def _post_json(
            url: str,
            payload: dict[str, object],
            *,
            headers: dict[str, str],
            timeout: float,
        ) -> dict[str, object]:
            calls.append((url, payload, headers, timeout))
            return {
                "proxy_id": "proxy-default",
                "url": "https://proxy.example.com/default",
            }

        reg = LiveRunnerRegistration(
            orchestrator_url="https://service.example.com/api",
            secret="secret-token",
            runner_url="https://runner.example.com",
            app="live-video-to-video/scope",
            price_info=LiveRunnerPriceInfo(10),
            runner_id="runner-1",
        )

        with mock.patch.object(live_runner, "post_json", side_effect=_post_json):
            proxy = await reg.create_proxy(
                "session-1",
                target_url,
                session_token="session-token",
            )
            assert proxy.proxy_id == "proxy-default"
            assert proxy.url == "https://proxy.example.com/default"

        assert calls == [
            (
                "https://service.example.com/api/runner/runner-1/session/session-1/proxy",
                {},
                {"Livepeer-Session-Token": "session-token"},
                5.0,
            ),
        ]


class TestLiveRunnerGPU:
    def test_pynvml_detects_process_gpu(self) -> None:
        fake = SimpleNamespace()
        handle0 = object()
        handle1 = object()
        proc = SimpleNamespace(pid=os.getpid())
        mem = SimpleNamespace(total=16 * 1024 * 1024)
        fake.nvmlInit = mock.Mock()
        fake.nvmlShutdown = mock.Mock()
        fake.nvmlDeviceGetCount = mock.Mock(return_value=2)
        fake.nvmlDeviceGetHandleByIndex = mock.Mock(
            side_effect=[handle0, handle1, handle1]
        )
        fake.nvmlDeviceGetComputeRunningProcesses_v2 = mock.Mock(
            side_effect=[[], [proc]]
        )
        fake.nvmlDeviceGetUUID = mock.Mock(return_value=b"GPU-uuid")
        fake.nvmlDeviceGetName = mock.Mock(return_value=b"NVIDIA Test")
        fake.nvmlDeviceGetMemoryInfo = mock.Mock(return_value=mem)

        with mock.patch.dict(sys.modules, {"pynvml": fake}):
            gpu = live_runner._detect_gpu_pynvml()

        assert gpu == LiveRunnerGPU(id="GPU-uuid", name="NVIDIA Test", vram_mb=16)

    def test_torch_detects_current_device(self) -> None:
        cuda = SimpleNamespace(
            is_available=mock.Mock(return_value=True),
            current_device=mock.Mock(return_value=1),
            get_device_properties=mock.Mock(
                return_value=SimpleNamespace(
                    name="Torch GPU", total_memory=32 * 1024 * 1024
                )
            ),
            get_device_name=mock.Mock(return_value="unused"),
        )
        torch = SimpleNamespace(cuda=cuda)

        with mock.patch.dict(sys.modules, {"torch": torch}):
            gpu = live_runner._detect_gpu_torch()

        assert gpu == LiveRunnerGPU(id="1", name="Torch GPU", vram_mb=32)
