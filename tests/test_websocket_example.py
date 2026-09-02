from __future__ import annotations

import importlib.util
import json
import os

import pytest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNNER_PATH = os.path.join(ROOT, "examples", "ping-pong", "runner.py")
CLIENT_PATH = os.path.join(ROOT, "examples", "ping-pong", "client.py")

spec = importlib.util.spec_from_file_location("websocket_runner_example", RUNNER_PATH)
assert spec is not None
runner = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(runner)

client_spec = importlib.util.spec_from_file_location("websocket_client_example", CLIENT_PATH)
assert client_spec is not None
client = importlib.util.module_from_spec(client_spec)
assert client_spec.loader is not None
client_spec.loader.exec_module(client)


class TestWebsocketRunnerExample:
    def test_pong_response_echoes_timestamp_and_computes_delta(self) -> None:
        response = runner._pong_response(json.dumps({"ping": 10.0}), now=10.25)

        assert response["pong"] == 10.0
        assert response["delta_ms"] == 250.0

    def test_pong_response_rejects_invalid_payload(self) -> None:
        with pytest.raises(ValueError):
            runner._pong_response(json.dumps({"ping": "10.0"}), now=10.25)


class TestWebsocketClientExample:
    def test_insecure_defaults_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.argv", ["client.py"])
        args = client._parse_args()
        assert args.insecure is False

    def test_insecure_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.argv", ["client.py", "--insecure"])
        args = client._parse_args()
        assert args.insecure is True
