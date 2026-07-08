from livepeer_gateway.capabilities import (
    CapabilityId,
    byoc_capabilities_from_app,
)


def test_byoc_capabilities_from_app() -> None:
    caps = byoc_capabilities_from_app("transcode/ffmpeg")
    assert caps is not None
    assert caps.capacities[int(CapabilityId.BYOC)] == 1
    assert "transcode/ffmpeg" in caps.constraints.PerCapability[int(CapabilityId.BYOC)].models


def test_byoc_capabilities_from_app_empty() -> None:
    assert byoc_capabilities_from_app("") is None
    assert byoc_capabilities_from_app("   ") is None
