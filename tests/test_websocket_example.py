from __future__ import annotations

import importlib.util
import json
import os

import pytest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNNER_PATH = os.path.join(ROOT, "examples", "ping-pong", "runner.py")

spec = importlib.util.spec_from_file_location("websocket_runner_example", RUNNER_PATH)
assert spec is not None
runner = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(runner)


class TestWebsocketRunnerExample:
    def test_pong_response_echoes_timestamp_and_computes_delta(self) -> None:
        response = runner._pong_response(json.dumps({"ping": 10.0}), now=10.25)

        assert response["pong"] == 10.0
        assert response["delta_ms"] == 250.0

    def test_pong_response_rejects_invalid_payload(self) -> None:
        with pytest.raises(ValueError):
            runner._pong_response(json.dumps({"ping": "10.0"}), now=10.25)
