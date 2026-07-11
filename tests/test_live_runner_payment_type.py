"""Live-runner reservation payments use byoc type with BYOC caps for attribution."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from livepeer_gateway import lp_rpc_pb2
from livepeer_gateway.capabilities import CapabilityId
from livepeer_gateway.live_runner import LiveRunnerInstance, _get_runner_payment
from livepeer_gateway.remote_signer import GetPaymentResponse


def test_get_runner_payment_uses_byoc_type_and_caps() -> None:
    captured: dict = {}

    class _Session:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        async def get_payment(self) -> GetPaymentResponse:
            return GetPaymentResponse(payment="pay", seg_creds="creds")

    runner = LiveRunnerInstance(
        url="http://orch/apps/runner/session",
        app="transcode/ffmpeg",
        runner_id="runner",
        mode="persistent",
        orchestrator_url="http://orch",
        raw={},
    )

    challenge = type(
        "_Challenge",
        (),
        {
            "payment_params": "params",
            "orchestrator_url": "http://orch",
            "manifest_id": "manifest-1",
        },
    )()

    with patch("livepeer_gateway.live_runner.LivePaymentSession", _Session):
        asyncio.run(
            _get_runner_payment(
                challenge,
                signer_url="https://signer.example.com",
                signer_headers={"Authorization": "Bearer token"},
                runner=runner,
            )
        )

    assert captured["type"] == "byoc"
    caps = captured["capabilities"]
    assert isinstance(caps, lp_rpc_pb2.Capabilities)
    assert caps.capacities[int(CapabilityId.BYOC)] == 1
    assert "transcode/ffmpeg" in caps.constraints.PerCapability[int(CapabilityId.BYOC)].models
