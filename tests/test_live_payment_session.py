from __future__ import annotations

import types
from unittest import mock

import pytest

from livepeer_gateway.errors import (
    LivepeerHTTPError,
    SignerRefreshRequired,
)
from livepeer_gateway.remote_signer import (
    LivePaymentChallenge,
    LivePaymentSession,
    get_signer_info,
)


_PAYMENT_URL = "https://orch.example.com/apps/runner/session/manifest-1/payment"


def _challenge(*, payment_params: str = "opaque") -> LivePaymentChallenge:
    return LivePaymentChallenge(
        payment_params=payment_params,
        manifest_id="manifest-1",
        payment_url=_PAYMENT_URL,
    )


class TestLivePaymentSession:
    @pytest.fixture(autouse=True)
    def clear_signer_info_cache(self):
        yield
        get_signer_info.cache_clear()

    async def test_none_signer_exits_early(self) -> None:
        session = LivePaymentSession(
            None,
            type="live",
            challenge=_challenge(),
        )

        payment = await session.get_payment()
        await session.send_payment()

        assert payment.payment == ""
        assert payment.seg_creds is None

    async def test_send_payment_reuses_empty_post_helper(self) -> None:
        session = LivePaymentSession(
            "https://signer.example.com",
            type="live",
            challenge=_challenge(),
        )

        post_empty = mock.AsyncMock()
        with (
            mock.patch.object(
                session,
                "get_payment",
                new=mock.AsyncMock(
                    return_value=types.SimpleNamespace(payment="p", seg_creds="s")
                ),
            ),
            mock.patch("livepeer_gateway.http._post_empty", post_empty),
        ):
            await session.send_payment()

        post_empty.assert_awaited_once_with(
            _PAYMENT_URL,
            headers={"Livepeer-Payment": "p", "Livepeer-Segment": "s"},
            timeout=5.0,
        )

    async def test_send_payment_preserves_typed_http_error(self) -> None:
        session = LivePaymentSession(
            "https://signer.example.com",
            type="live",
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
                "livepeer_gateway.http._post_empty",
                new=mock.AsyncMock(side_effect=error),
            ),
        ):
            with pytest.raises(LivepeerHTTPError) as raised:
                await session.send_payment()

        assert raised.value is error

    async def test_get_payment_sends_opaque_payment_params_and_state(self) -> None:
        calls: list[tuple[str, dict[str, object], dict[str, str] | None]] = []
        reported: list[str] = []

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
                type="live",
                challenge=_challenge(payment_params="opaque-payment-params"),
                app="livepeer/app",
                max_price={
                    "price": 10.12,
                    "currency": "wei",
                    "unit": "seconds",
                },
                _on_payment=lambda: reported.append("signed"),
            )
            first = await session.get_payment()
            second = await session.get_payment()

        assert first.payment == "payment"
        assert second.seg_creds == "segment"
        assert reported == ["signed"]
        assert calls[0][0] == "https://signer.example.com/generate-live-payment"
        assert calls[0][1] == {
            "orchestrator": "opaque-payment-params",
            "type": "live",
            "ManifestID": "manifest-1",
            "app": "livepeer/app",
            "maxPrice": {
                "price": 10.12,
                "currency": "wei",
                "unit": "seconds",
            },
        }
        assert calls[0][2] == {"Authorization": "token"}
        assert calls[1][1]["app"] == "livepeer/app"
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
            raise SignerRefreshRequired("refresh")

        with mock.patch("livepeer_gateway.http.post_json", side_effect=_post_json):
            session = LivePaymentSession(
                "https://signer.example.com",
                type="live",
                challenge=_challenge(payment_params="old-payment-params"),
            )
            with pytest.raises(SignerRefreshRequired):
                await session.get_payment()

        assert calls == [
            (
                "https://signer.example.com/generate-live-payment",
                {
                    "orchestrator": "old-payment-params",
                    "type": "live",
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
                type="live",
                challenge=_challenge(payment_params="old-payment-params"),
                max_price={
                    "price": 10.12,
                    "currency": "wei",
                    "unit": "seconds",
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
                "unit": "seconds",
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

    async def test_manifest_id_comes_from_the_challenge_and_survives_refresh(
        self,
    ) -> None:
        payment_requests = 0

        async def _post_json(
            url: str,
            payload: dict[str, object],
            *,
            headers: dict[str, str] | None = None,
            timeout: float = 5.0,
        ) -> dict[str, object]:
            nonlocal payment_requests
            del payload, headers, timeout
            if url == "https://signer.example.com/generate-live-payment":
                payment_requests += 1
                if payment_requests == 2:
                    raise SignerRefreshRequired("refresh")
                return {
                    "payment": "payment",
                    "segCreds": "segment",
                    "state": {"state": "one"},
                }
            if url == "https://signer.example.com/sign-orchestrator-info":
                return {"address": "opaque-sender", "signature": "opaque-signature"}
            if url == "https://orch.example.com/refresh-payment":
                return {"payment_params": "new-payment-params"}
            raise AssertionError(f"unexpected POST {url}")

        session = LivePaymentSession(
            "https://signer.example.com",
            type="fixed",
            challenge=_challenge(),
        )
        assert session.manifest_id == "manifest-1"
        with mock.patch("livepeer_gateway.http.post_json", side_effect=_post_json):
            await session.get_payment()
            await session.get_payment()

        assert session.manifest_id == "manifest-1"
        assert session._challenge.payment_params == "new-payment-params"
