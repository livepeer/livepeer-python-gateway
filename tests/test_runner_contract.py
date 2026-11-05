from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest import mock

import pytest
from pydantic import ValidationError

from livepeer_gateway import live_runner
from livepeer_gateway.errors import LivepeerGatewayError
from livepeer_gateway.live_runner import register_runner
from livepeer_gateway.runner_contract import (
    RunnerContractCapacityGroup,
    RunnerContractChannels,
    RunnerContractEndpoint,
    RunnerContractEnvelope,
    RunnerContractModel,
    RunnerContractRuntime,
    deployment_revision,
    openapi_document,
    write_openapi,
)

_DIGEST = "sha256:" + ("ab" * 32)
_REVISION = "sha256:" + ("cd" * 32)


def _chat_envelope() -> RunnerContractEnvelope:
    return RunnerContractEnvelope(
        deployment_revision=_DIGEST,
        model=RunnerContractModel(
            name="qwen3.8-27b",
            source="hf:Qwen/Qwen3.8-27B-AWQ",
            revision=_REVISION,
        ),
        endpoints={
            "chat_completions": RunnerContractEndpoint(profile="openai.chat.v1"),
            "messages": RunnerContractEndpoint(profile="anthropic.messages.v1"),
        },
        limits={"context_tokens": 131072},
    )


def test_chat_envelope_is_compact_contract_json() -> None:
    text = _chat_envelope().to_metadata()
    assert " " not in text
    assert json.loads(text) == {
        "runner_contract": 1,
        "deployment_revision": _DIGEST,
        "model": {
            "name": "qwen3.8-27b",
            "source": "hf:Qwen/Qwen3.8-27B-AWQ",
            "revision": _REVISION,
        },
        "endpoints": {
            "chat_completions": {"profile": "openai.chat.v1"},
            "messages": {"profile": "anthropic.messages.v1"},
        },
        "limits": {"context_tokens": 131072},
    }


def test_rejects_bad_profile_and_missing_endpoint() -> None:
    with pytest.raises(ValidationError):
        RunnerContractEndpoint(profile="not-a-profile")
    with pytest.raises(ValidationError):
        RunnerContractEnvelope(deployment_revision=_DIGEST, endpoints={})


def test_deployment_revision_matches_spec_recipe() -> None:
    images = ["sha256:" + ("b" * 64), "sha256:" + ("a" * 64)]
    profiles = ["openai.chat.v1", "anthropic.messages.v1"]
    settings = {"context_tokens": 131072}
    model = {"source": "hf:Qwen/Qwen3.8-27B-AWQ", "revision": "rev-1"}
    recipe = {
        "images": sorted(images),
        "model": {"revision": model.get("revision"), "source": model["source"]},
        "profiles": sorted(profiles),
        "settings": settings,
    }
    text = json.dumps(recipe, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    expected = "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert deployment_revision(images, profiles, settings, model) == expected


def test_metadata_over_1024_bytes_fails() -> None:
    envelope = RunnerContractEnvelope(
        deployment_revision=_DIGEST,
        endpoints={"chat_completions": RunnerContractEndpoint(profile="openai.chat.v1")},
        attributes={"note": "a" * 1100},
    )
    with pytest.raises(LivepeerGatewayError, match="1024"):
        envelope.to_metadata()


def test_persistent_endpoint_uses_channel_alias() -> None:
    envelope = RunnerContractEnvelope(
        deployment_revision=_DIGEST,
        endpoints={
            "transcribe": RunnerContractEndpoint(
                profile="livepeer.transcription.v1",
                path="/transcribe",
                method="GET",
                stop="/stop",
                channels=RunnerContractChannels(in_=["audio"], out=["text"]),
                idle_timeout_s=15,
                drain_timeout_s=0,
            )
        },
        runtime=RunnerContractRuntime(name="comfystream", image=_DIGEST),
        capacity_group=RunnerContractCapacityGroup(id="gpu-a", slots=1),
        admission="candidate",
    )
    body = json.loads(envelope.to_metadata())
    endpoint = body["endpoints"]["transcribe"]
    assert endpoint["channels"] == {"in": ["audio"], "out": ["text"]}
    assert endpoint["stop"] == "/stop"
    assert body["capacity_group"] == {"id": "gpu-a", "slots": 1}


def test_committed_openapi_matches_exporter() -> None:
    committed = json.loads((Path(__file__).resolve().parents[1] / "openapi.json").read_text(encoding="utf-8"))
    assert committed == json.loads(json.dumps(openapi_document(), sort_keys=True))


def test_openapi_document_describes_the_envelope(tmp_path: Path) -> None:
    document = openapi_document()
    assert document["openapi"] == "3.1.0"
    assert "paths" not in document
    schemas = document["components"]["schemas"]
    assert "runner_contract" in schemas["RunnerContractEnvelope"]["properties"]
    assert "RunnerContractEndpoint" in schemas

    destination = tmp_path / "openapi.json"
    write_openapi(destination)
    written = json.loads(destination.read_text(encoding="utf-8"))
    assert written == json.loads(json.dumps(document, sort_keys=True))


async def test_register_runner_sends_envelope_metadata() -> None:
    envelope = _chat_envelope()
    calls: list[dict[str, object]] = []

    def _post_json(
        url: str,
        payload: dict[str, object],
        *,
        headers: dict[str, str],
        timeout: float,
    ) -> dict[str, str]:
        calls.append(payload)
        return {
            "runner_id": "runner-1",
            "orchestrator": "https://orch.example/base",
            "heartbeat_interval": "10s",
            "heartbeat_ttl": "20s",
            "heartbeat_secret": "tok",
        }

    with mock.patch.object(live_runner, "post_json", side_effect=_post_json):
        registration = await register_runner(
            "https://orch.example",
            secret="secret",
            runner_url="https://runner.example",
            app="chat",
            metadata=envelope,
            auto_detect_gpu=False,
            unregister_on_close=False,
        )
        await registration.close()

    assert calls[0]["metadata"] == envelope.to_metadata()
