"""Unit tests for BYOC capability protobuf helpers."""

from __future__ import annotations

from livepeer_gateway.capabilities import (
    CapabilityId,
    byoc_capabilities_from_app,
)


def test_byoc_capabilities_from_app_builds_constraint():
    caps = byoc_capabilities_from_app("flux-schnell")
    assert caps is not None
    assert caps.capacities[int(CapabilityId.BYOC)] == 1
    assert "flux-schnell" in caps.constraints.PerCapability[int(CapabilityId.BYOC)].models


def test_byoc_capabilities_from_app_empty_returns_none():
    assert byoc_capabilities_from_app("") is None
    assert byoc_capabilities_from_app("   ") is None
