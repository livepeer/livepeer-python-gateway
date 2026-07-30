from __future__ import annotations

import base64
import json

import pytest

from livepeer_gateway.errors import LivepeerGatewayError
from livepeer_gateway.token import parse_token


def _encode_token(payload: object) -> str:
    return base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")


class TestParseToken:
    def test_parse_token_round_trips_expected_fields(self) -> None:
        token = _encode_token(
            {
                "orchestrators": [
                    "  https://orch-1.example.com:8935  ",
                    "https://orch-2.example.com:8935",
                ],
                "signer": "https://signer.example.com",
                "signer_headers": {"Authorization": "Bearer signer-token"},
                "discovery": "https://discovery.example.com/orchestrators",
                "discovery_headers": {"Authorization": "Bearer discovery-token"},
            }
        )

        result = parse_token(token)

        assert result == {
            "orchestrators": [
                "https://orch-1.example.com:8935",
                "https://orch-2.example.com:8935",
            ],
            "signer": "https://signer.example.com",
            "signer_headers": {"Authorization": "Bearer signer-token"},
            "discovery": "https://discovery.example.com/orchestrators",
            "discovery_headers": {"Authorization": "Bearer discovery-token"},
        }

    def test_parse_token_rejects_non_base64_payload(self) -> None:
        with pytest.raises(LivepeerGatewayError, match="base64-encoded JSON"):
            parse_token("not-base64")

    def test_parse_token_rejects_invalid_headers_shape(self) -> None:
        token = _encode_token({"signer_headers": {"Authorization": 123}})

        with pytest.raises(LivepeerGatewayError, match="signer_headers must be a"):
            parse_token(token)
