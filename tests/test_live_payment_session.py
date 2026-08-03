from __future__ import annotations

import types
from unittest import mock

import pytest

from livepeer_gateway import lp_rpc_pb2
from livepeer_gateway.errors import (
    PaymentError,
    SignerRefreshRequired,
)
from livepeer_gateway.remote_signer import (
    LivePaymentSession,
    PaymentSession,
    get_signer_info,
)


class TestPaymentSession:
    def test_get_payment_round_trips_state_without_cross_session_leak(self) -> None:
        calls: list[tuple[str, dict[str, object], dict[str, str] | None]] = []
        request_counts: dict[str, int] = {}

        def _post_json(
            url: str,
            payload: dict[str, object],
            *,
            headers: dict[str, str] | None = None,
            timeout: float = 5.0,
        ) -> dict[str, object]:
            del timeout
            calls.append((url, dict(payload), headers))
            manifest_id = payload["ManifestID"]
            assert isinstance(manifest_id, str)
            request_counts[manifest_id] = request_counts.get(manifest_id, 0) + 1
            sequence = request_counts[manifest_id]
            return {
                "payment": f"payment-{manifest_id}-{sequence}",
                "segCreds": f"segment-{manifest_id}-{sequence}",
                "state": {"session": manifest_id, "sequence": str(sequence)},
            }

        info = lp_rpc_pb2.OrchestratorInfo(transcoder="https://orch.example.com")
        first_session = PaymentSession(
            "https://signer.example.com",
            info,
            signer_headers={"Authorization": "token"},
            type="lv2v",
            app="live-video-to-video/scope",
        )
        first_session.set_manifest_id("first")
        second_session = PaymentSession(
            "https://signer.example.com",
            info,
            signer_headers={"Authorization": "token"},
            type="lv2v",
        )
        second_session.set_manifest_id("second")

        with mock.patch(
            "livepeer_gateway.http.post_json_sync", side_effect=_post_json
        ):
            first_payment = first_session.get_payment()
            second_payment = second_session.get_payment()
            first_session.get_payment()
            second_session.get_payment()

        assert first_payment.payment == "payment-first-1"
        assert second_payment.seg_creds == "segment-second-1"
        assert [call[0] for call in calls] == [
            "https://signer.example.com/generate-live-payment"
        ] * 4
        assert all(call[2] == {"Authorization": "token"} for call in calls)
        assert calls[0][1]["app"] == "live-video-to-video/scope"
        assert "app" not in calls[1][1]
        assert "state" not in calls[0][1]
        assert "state" not in calls[1][1]
        assert calls[2][1]["app"] == "live-video-to-video/scope"
        assert calls[2][1]["state"] == {"session": "first", "sequence": "1"}
        assert calls[3][1]["state"] == {"session": "second", "sequence": "1"}


class TestLivePaymentSession:
    @pytest.fixture(autouse=True)
    def clear_signer_info_cache(self):
        yield
        get_signer_info.cache_clear()

    async def test_none_signer_exits_early(self) -> None:
        session = LivePaymentSession(
            None,
            type="lv2v",
            payment_params="opaque",
            manifest_id="manifest-1",
        )

        payment = await session.get_payment()
        await session.send_payment("https://orchestrator.example.com")

        assert payment.payment == ""
        assert payment.seg_creds is None

    async def test_send_payment_uses_default_tls_verification(self) -> None:
        class _Response:
            status = 204
            headers: dict[str, str] = {}

            async def __aenter__(self) -> _Response:
                return self

            async def __aexit__(self, *args: object) -> None:
                return None

            async def read(self) -> bytes:
                return b""

            async def text(self) -> str:
                raise AssertionError(
                    "successful payment responses must not be decoded as text"
                )

        class _Session:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

            async def __aenter__(self) -> _Session:
                return self

            async def __aexit__(self, *args: object) -> None:
                return None

            def post(self, *args: object, **kwargs: object) -> _Response:
                return _Response()

        session = LivePaymentSession(
            "https://signer.example.com",
            type="lv2v",
            payment_params="opaque",
            manifest_id="manifest-1",
        )

        with (
            mock.patch.object(
                session,
                "get_payment",
                new=mock.AsyncMock(
                    return_value=types.SimpleNamespace(payment="p", seg_creds="s")
                ),
            ),
            mock.patch(
                "livepeer_gateway.remote_signer.aiohttp.TCPConnector"
            ) as connector_mock,
            mock.patch(
                "livepeer_gateway.remote_signer.aiohttp.ClientSession",
                side_effect=_Session,
            ) as client_session_mock,
        ):
            await session.send_payment("https://orchestrator.example.com")

        connector_mock.assert_not_called()
        assert "connector" not in client_session_mock.call_args.kwargs

    async def test_send_payment_uses_constructor_orchestrator_url(self) -> None:
        posts: list[tuple[object, dict[str, object]]] = []

        class _Response:
            status = 204
            headers: dict[str, str] = {}

            async def __aenter__(self) -> _Response:
                return self

            async def __aexit__(self, *args: object) -> None:
                return None

            async def read(self) -> bytes:
                return b""

        class _Session:
            def __init__(self, **kwargs: object) -> None:
                del kwargs

            async def __aenter__(self) -> _Session:
                return self

            async def __aexit__(self, *args: object) -> None:
                return None

            def post(self, url: object, **kwargs: object) -> _Response:
                posts.append((url, kwargs))
                return _Response()

        session = LivePaymentSession(
            "https://signer.example.com",
            type="lv2v",
            payment_params="opaque",
            manifest_id="manifest-1",
            orchestrator_url="https://orchestrator.example.com/base",
        )

        with (
            mock.patch.object(
                session,
                "get_payment",
                new=mock.AsyncMock(
                    return_value=types.SimpleNamespace(payment="p", seg_creds="s")
                ),
            ),
            mock.patch(
                "livepeer_gateway.remote_signer.aiohttp.ClientSession",
                side_effect=_Session,
            ),
        ):
            await session.send_payment()

        assert posts[0][0] == "https://orchestrator.example.com/payment"
        assert posts[0][1]["headers"] == {
            "Livepeer-Payment": "p",
            "Livepeer-Segment": "s",
        }

    async def test_send_payment_accepts_binary_payment_result_response(self) -> None:
        class _Response:
            status = 200
            headers: dict[str, str] = {}

            async def __aenter__(self) -> _Response:
                return self

            async def __aexit__(self, *args: object) -> None:
                return None

            async def read(self) -> bytes:
                return b"\x82\x01protobuf-payment-result"

            async def text(self) -> str:
                raise AssertionError(
                    "successful binary payment response decoded as text"
                )

        class _Session:
            def __init__(self, **kwargs: object) -> None:
                del kwargs

            async def __aenter__(self) -> _Session:
                return self

            async def __aexit__(self, *args: object) -> None:
                return None

            def post(self, *args: object, **kwargs: object) -> _Response:
                del args, kwargs
                return _Response()

        session = LivePaymentSession(
            "https://signer.example.com",
            type="lv2v",
            payment_params="opaque",
            manifest_id="manifest-1",
        )

        with (
            mock.patch.object(
                session,
                "get_payment",
                new=mock.AsyncMock(
                    return_value=types.SimpleNamespace(payment="p", seg_creds="s")
                ),
            ),
            mock.patch(
                "livepeer_gateway.remote_signer.aiohttp.ClientSession",
                side_effect=_Session,
            ),
        ):
            await session.send_payment("https://orchestrator.example.com")

    async def test_send_payment_error_decodes_body_for_message(self) -> None:
        class _Response:
            status = 400
            headers: dict[str, str] = {}

            async def __aenter__(self) -> _Response:
                return self

            async def __aexit__(self, *args: object) -> None:
                return None

            async def read(self) -> bytes:
                raise AssertionError("error payment responses should use text decoding")

            async def text(self) -> str:
                return '{"error":{"message":"payment rejected"}}'

        class _Session:
            def __init__(self, **kwargs: object) -> None:
                del kwargs

            async def __aenter__(self) -> _Session:
                return self

            async def __aexit__(self, *args: object) -> None:
                return None

            def post(self, *args: object, **kwargs: object) -> _Response:
                del args, kwargs
                return _Response()

        session = LivePaymentSession(
            "https://signer.example.com",
            type="lv2v",
            payment_params="opaque",
            manifest_id="manifest-1",
        )

        with (
            mock.patch.object(
                session,
                "get_payment",
                new=mock.AsyncMock(
                    return_value=types.SimpleNamespace(payment="p", seg_creds="s")
                ),
            ),
            mock.patch(
                "livepeer_gateway.remote_signer.aiohttp.ClientSession",
                side_effect=_Session,
            ),
        ):
            with pytest.raises(PaymentError) as raised:
                await session.send_payment("https://orchestrator.example.com")

        assert "payment rejected" in str(raised.value)

    async def test_get_payment_sends_opaque_payment_params_and_state(self) -> None:
        calls: list[tuple[str, dict[str, object], dict[str, str] | None]] = []

        async def _post_json(
            url: str,
            payload: dict[str, object],
            *,
            headers: dict[str, str] | None = None,
            timeout: float = 5.0,
        ) -> dict[str, object]:
            del timeout
            calls.append((url, payload, headers))
            return {
                "payment": "payment",
                "segCreds": "segment",
                "state": {"state": "one"},
            }

        with mock.patch("livepeer_gateway.http.post_json", side_effect=_post_json):
            session = LivePaymentSession(
                "https://signer.example.com",
                signer_headers={"Authorization": "token"},
                type="lv2v",
                payment_params="opaque-payment-params",
                manifest_id="manifest-1",
                app="live-video-to-video/scope",
            )
            first = await session.get_payment()
            second = await session.get_payment()

        assert first.payment == "payment"
        assert second.seg_creds == "segment"
        assert calls[0][0] == "https://signer.example.com/generate-live-payment"
        assert calls[0][1] == {
            "orchestrator": "opaque-payment-params",
            "type": "lv2v",
            "ManifestID": "manifest-1",
            "app": "live-video-to-video/scope",
        }
        assert calls[0][2] == {"Authorization": "token"}
        assert calls[1][1]["app"] == "live-video-to-video/scope"
        assert calls[1][1]["state"] == {"state": "one"}

    async def test_initial_480_restarts_challenge_without_refresh(self) -> None:
        calls: list[tuple[str, dict[str, object]]] = []

        async def _post_json(
            url: str,
            payload: dict[str, object],
            *,
            headers: dict[str, str] | None = None,
            timeout: float = 5.0,
        ) -> dict[str, object]:
            del headers, timeout
            calls.append((url, payload))
            raise SignerRefreshRequired(
                "refresh",
                orchestrator_url="https://orch.example.com",
            )

        with mock.patch("livepeer_gateway.http.post_json", side_effect=_post_json):
            session = LivePaymentSession(
                "https://signer.example.com",
                type="lv2v",
                payment_params="old-payment-params",
                manifest_id="manifest-1",
            )
            with pytest.raises(SignerRefreshRequired):
                await session.get_payment()

        assert calls == [
            (
                "https://signer.example.com/generate-live-payment",
                {
                    "orchestrator": "old-payment-params",
                    "type": "lv2v",
                    "ManifestID": "manifest-1",
                },
            )
        ]

    async def test_stateful_480_refreshes_payment_params_from_orchestrator_header(
        self,
    ) -> None:
        calls: list[tuple[str, dict[str, object]]] = []
        payment_requests = 0

        async def _post_json(
            url: str,
            payload: dict[str, object],
            *,
            headers: dict[str, str] | None = None,
            timeout: float = 5.0,
        ) -> dict[str, object]:
            nonlocal payment_requests
            del headers, timeout
            calls.append((url, payload))
            if url == "https://signer.example.com/generate-live-payment":
                payment_requests += 1
                if payment_requests == 1:
                    return {
                        "payment": "payment-1",
                        "segCreds": "segment-1",
                        "state": {"state": "one"},
                    }
                if payment_requests == 2:
                    raise SignerRefreshRequired(
                        "refresh",
                        orchestrator_url="https://orch.example.com",
                    )
                return {
                    "payment": "payment-2",
                    "segCreds": "segment-2",
                    "state": {"state": "two"},
                }
            if url == "https://signer.example.com/sign-orchestrator-info":
                return {"address": "opaque-sender", "signature": "opaque-signature"}
            if url == "https://orch.example.com/refresh-payment":
                return {
                    "payment_params": "new-payment-params",
                    "orchestrator": "https://orch.example.com",
                }
            raise AssertionError(f"unexpected POST {url}")

        with mock.patch("livepeer_gateway.http.post_json", side_effect=_post_json):
            session = LivePaymentSession(
                "https://signer.example.com",
                type="lv2v",
                payment_params="old-payment-params",
                manifest_id="manifest-1",
            )
            first_payment = await session.get_payment()
            payment = await session.get_payment()

        assert first_payment.payment == "payment-1"
        assert payment.payment == "payment-2"
        assert calls[1][1]["state"] == {"state": "one"}
        assert calls[3] == (
            "https://orch.example.com/refresh-payment",
            {"sender": "opaque-sender", "manifest_id": "manifest-1"},
        )
        assert calls[4][1]["orchestrator"] == "new-payment-params"

    async def test_480_without_orchestrator_header_fails(self) -> None:
        payment_requests = 0

        async def _post_json(
            url: str,
            payload: dict[str, object],
            *,
            headers: dict[str, str] | None = None,
            timeout: float = 5.0,
        ) -> dict[str, object]:
            nonlocal payment_requests
            del url, payload, headers, timeout
            payment_requests += 1
            if payment_requests == 1:
                return {
                    "payment": "payment-1",
                    "segCreds": "segment-1",
                    "state": {"state": "one"},
                }
            raise SignerRefreshRequired("refresh")

        with mock.patch("livepeer_gateway.http.post_json", side_effect=_post_json):
            session = LivePaymentSession(
                "https://signer.example.com",
                type="lv2v",
                payment_params="old-payment-params",
                manifest_id="manifest-1",
            )
            await session.get_payment()
            with pytest.raises(PaymentError, match="missing Livepeer-Orchestrator-URL"):
                await session.get_payment()

    async def test_get_signer_info_caches_result(self) -> None:
        calls: list[tuple[str, dict[str, object]]] = []

        async def _post_json(
            url: str,
            payload: dict[str, object],
            *,
            headers: dict[str, str] | None = None,
            timeout: float = 5.0,
        ) -> dict[str, object]:
            del headers, timeout
            calls.append((url, payload))
            return {"address": "opaque-sender", "signature": "opaque-signature"}

        with mock.patch("livepeer_gateway.http.post_json", side_effect=_post_json):
            first = await get_signer_info("https://signer.example.com")
            second = await get_signer_info("https://signer.example.com")

        assert first is second
        assert first.address == "opaque-sender"
        assert first.sig == "opaque-signature"
        assert len(calls) == 1
