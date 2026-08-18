from __future__ import annotations

import json
import types
from unittest import mock

import pytest

from livepeer_gateway import lp_rpc_pb2
from livepeer_gateway.errors import (
    LivepeerHTTPError,
    SignerRefreshRequired,
)
from livepeer_gateway.remote_signer import (
    LivePaymentChallenge,
    LivePaymentSession,
    PaymentSession,
    get_signer_info,
)


_PAYMENT_URL = "https://orch.example.com/apps/runner/session/manifest-1/payment"


def _challenge(*, payment_params: str = "opaque") -> LivePaymentChallenge:
    return LivePaymentChallenge(
        payment_params=payment_params,
        manifest_id="manifest-1",
        payment_url=_PAYMENT_URL,
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
            challenge=_challenge(),
        )

        payment = await session.get_payment()
        await session.send_payment()

        assert payment.payment == ""
        assert payment.seg_creds is None

    async def test_send_payment_rotates_payment_params_for_next_payment(self) -> None:
        signer_calls: list[dict[str, object]] = []
        payment_posts: list[tuple[str, str, dict[str, str]]] = []
        refreshed_params = iter(("fresh-payment-params-1", "fresh-payment-params-2"))

        async def _post_json(
            url: str,
            payload: dict[str, object],
            *,
            headers: dict[str, str] | None = None,
            timeout: float = 5.0,
        ) -> dict[str, object]:
            del url, headers, timeout
            signer_calls.append(dict(payload))
            sequence = len(signer_calls)
            return {
                "payment": f"payment-{sequence}",
                "segCreds": f"segment-{sequence}",
                "state": {"sequence": sequence},
            }

        async def _request_body(
            url: str,
            *,
            method: str,
            payload: dict[str, object] | None,
            headers: dict[str, str],
            timeout: float,
        ) -> tuple[bytes, str]:
            del payload, timeout
            payment_posts.append((url, method, dict(headers)))
            payment_params = next(refreshed_params)
            return (
                json.dumps(
                    {
                        "payment_params": payment_params,
                        "orchestrator": "https://orch.example.com",
                        "manifest_id": "manifest-1",
                        "payment_url": _PAYMENT_URL,
                    }
                ).encode(),
                "application/json",
            )

        session = LivePaymentSession(
            "https://signer.example.com",
            type="lv2v",
            challenge=_challenge(payment_params="initial-payment-params"),
            app="live-video-to-video/scope",
            max_price={"price": 10.12, "currency": "wei", "unit": "pixels"},
        )

        with (
            mock.patch("livepeer_gateway.http.post_json", side_effect=_post_json),
            mock.patch("livepeer_gateway.http._request_body", side_effect=_request_body),
        ):
            await session.send_payment()
            await session.send_payment()

        assert [call["orchestrator"] for call in signer_calls] == [
            "initial-payment-params",
            "fresh-payment-params-1",
        ]
        assert signer_calls[1]["state"] == {"sequence": 1}
        assert [call["app"] for call in signer_calls] == [
            "live-video-to-video/scope",
            "live-video-to-video/scope",
        ]
        assert [call["maxPrice"] for call in signer_calls] == [
            {"price": 10.12, "currency": "wei", "unit": "pixels"},
            {"price": 10.12, "currency": "wei", "unit": "pixels"},
        ]
        assert payment_posts == [
            (
                _PAYMENT_URL,
                "POST",
                {
                    "Livepeer-Payment": "payment-1",
                    "Livepeer-Segment": "segment-1",
                },
            ),
            (
                _PAYMENT_URL,
                "POST",
                {
                    "Livepeer-Payment": "payment-2",
                    "Livepeer-Segment": "segment-2",
                },
            ),
        ]
        assert session._challenge == _challenge(
            payment_params="fresh-payment-params-2"
        )

    async def test_send_payment_accepts_legacy_non_json_response(self) -> None:
        session = LivePaymentSession(
            "https://signer.example.com",
            type="lv2v",
            challenge=_challenge(payment_params="legacy-payment-params"),
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
                "livepeer_gateway.http._request_body",
                new=mock.AsyncMock(
                    return_value=(b"legacy-protobuf", "application/octet-stream")
                ),
            ) as request_body,
        ):
            await session.send_payment()

        request_body.assert_awaited_once_with(
            _PAYMENT_URL,
            method="POST",
            payload=None,
            headers={"Livepeer-Payment": "p", "Livepeer-Segment": "s"},
            timeout=5.0,
        )
        assert session._challenge == _challenge(
            payment_params="legacy-payment-params"
        )

    @pytest.mark.parametrize(
        "body",
        [
            b"not-json",
            b"[]",
            b"{}",
            b'{"payment_params":""}',
        ],
    )
    async def test_send_payment_ignores_invalid_json_response_without_mutation(
        self,
        body: bytes,
    ) -> None:
        initial_challenge = _challenge(payment_params="initial-payment-params")
        session = LivePaymentSession(
            "https://signer.example.com",
            type="lv2v",
            challenge=initial_challenge,
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
                "livepeer_gateway.http._request_body",
                new=mock.AsyncMock(return_value=(body, "application/json")),
            ),
        ):
            await session.send_payment()

        assert session._challenge is initial_challenge

    async def test_send_payment_preserves_typed_http_error(self) -> None:
        session = LivePaymentSession(
            "https://signer.example.com",
            type="lv2v",
            challenge=_challenge(),
        )

        error = LivepeerHTTPError(
            400,
            "https://orchestrator.example.com/payment",
            body='{"error":{"message":"payment rejected"}}',
            message="payment rejected",
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
                "livepeer_gateway.http._request_body",
                new=mock.AsyncMock(side_effect=error),
            ),
        ):
            with pytest.raises(LivepeerHTTPError) as raised:
                await session.send_payment()

        assert raised.value is error

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
                challenge=_challenge(payment_params="opaque-payment-params"),
                app="live-video-to-video/scope",
                max_price={
                    "price": 10.12,
                    "currency": "wei",
                    "unit": "720p-pixel-seconds",
                },
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
            "maxPrice": {
                "price": 10.12,
                "currency": "wei",
                "unit": "720p-pixel-seconds",
            },
        }
        assert calls[0][2] == {"Authorization": "token"}
        assert calls[1][1]["app"] == "live-video-to-video/scope"
        assert calls[1][1]["maxPrice"] == calls[0][1]["maxPrice"]
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
                challenge=_challenge(payment_params="old-payment-params"),
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

    async def test_stateful_480_refreshes_params_from_payment_url_origin(
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
                    raise SignerRefreshRequired("refresh")
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
                    "manifest_id": "manifest-1",
                    "payment_url": "https://orch.example.com/payment",
                }
            raise AssertionError(f"unexpected POST {url}")

        with mock.patch("livepeer_gateway.http.post_json", side_effect=_post_json):
            session = LivePaymentSession(
                "https://signer.example.com",
                type="lv2v",
                challenge=_challenge(payment_params="old-payment-params"),
                max_price={
                    "price": 10.12,
                    "currency": "wei",
                    "unit": "720p-pixel-seconds",
                },
            )
            first_payment = await session.get_payment()
            payment = await session.get_payment()

        assert first_payment.payment == "payment-1"
        assert payment.payment == "payment-2"
        payment_calls = [
            payload
            for url, payload in calls
            if url == "https://signer.example.com/generate-live-payment"
        ]
        assert [payload["maxPrice"] for payload in payment_calls] == [
            {
                "price": 10.12,
                "currency": "wei",
                "unit": "720p-pixel-seconds",
            }
        ] * 3
        assert calls[1][1]["state"] == {"state": "one"}
        assert calls[3] == (
            "https://orch.example.com/refresh-payment",
            {"sender": "opaque-sender", "manifest_id": "manifest-1"},
        )
        assert calls[4][1]["orchestrator"] == "new-payment-params"

        assert session._challenge.payment_url == _PAYMENT_URL

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
