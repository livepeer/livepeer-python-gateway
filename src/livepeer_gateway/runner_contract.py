"""Runner contract v1 envelope for live-runner registration metadata.

The shape matches the live-runner-standard envelope locked on 5 November 2026.
``to_metadata`` emits the compact JSON string the orchestrator stores.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .errors import LivepeerGatewayError

METADATA_LIMIT_BYTES = 1024

_DIGEST = r"^sha256:[0-9a-f]{64}$"
_PROFILE_ID = r"^[a-z][a-z0-9]*(\.[a-z0-9]+(-[a-z0-9]+)*)+\.v[1-9][0-9]*$"
_ENDPOINT_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_SOURCE = r"^[a-z][a-z0-9-]*:.+$"
_LIMIT_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_PATH = r"^/"

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)


class RunnerContractChannels(BaseModel):
    """Persistent-runner channel names. The inbound registration key is in."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    in_: list[Annotated[str, Field(min_length=1)]] | None = Field(default=None, alias="in")
    out: list[Annotated[str, Field(min_length=1)]] | None = None


class RunnerContractEndpoint(BaseModel):
    """One operation on the runner, named by a registry wire profile."""

    model_config = _MODEL_CONFIG

    profile: Annotated[str, Field(pattern=_PROFILE_ID)]
    path: Annotated[str, Field(pattern=_PATH)] | None = None
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"] | None = None
    stop: Annotated[str, Field(pattern=_PATH)] | None = None
    channels: RunnerContractChannels | None = None
    idle_timeout_s: Annotated[int, Field(ge=1)] | None = None
    drain_timeout_s: Annotated[int, Field(ge=0)] | None = None


class RunnerContractModel(BaseModel):
    """The model a runner serves. source is prefix:locator."""

    model_config = _MODEL_CONFIG

    name: Annotated[str, Field(min_length=1)]
    source: Annotated[str, Field(pattern=_SOURCE)]
    revision: Annotated[str, Field(min_length=1)] | None = None


class RunnerContractRuntime(BaseModel):
    """True values about the runtime, omitted when unknown."""

    model_config = _MODEL_CONFIG

    name: str | None = None
    image: Annotated[str, Field(pattern=_DIGEST)] | None = None
    versions: dict[str, str] | None = None
    encoder: str | None = None


class RunnerContractCapacityGroup(BaseModel):
    """Registrations that share one GPU."""

    model_config = _MODEL_CONFIG

    id: Annotated[str, Field(min_length=1)]
    slots: Annotated[int, Field(ge=1)]


def _number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


class RunnerContractEnvelope(BaseModel):
    """Runner contract v1 registration metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    runner_contract: Literal[1] = 1
    deployment_revision: Annotated[str, Field(pattern=_DIGEST)]
    model: RunnerContractModel | None = None
    endpoints: dict[str, RunnerContractEndpoint]
    runner_id: Annotated[str, Field(min_length=1)] | None = None
    runtime: RunnerContractRuntime | None = None
    limits: dict[str, int | float] | None = None
    admission: str | None = None
    capacity_group: RunnerContractCapacityGroup | None = None
    attributes: dict[str, str | int | float | bool | list[str | int | float | bool]] | None = None

    @field_validator("endpoints")
    @classmethod
    def _endpoint_names(cls, value: dict[str, RunnerContractEndpoint]) -> dict[str, RunnerContractEndpoint]:
        if not value:
            raise ValueError("at least one endpoint is required")
        for name in value:
            if _ENDPOINT_NAME.fullmatch(name) is None:
                raise ValueError(f"invalid endpoint name {name!r}")
        return value

    @field_validator("limits")
    @classmethod
    def _limit_values(cls, value: dict[str, int | float] | None) -> dict[str, int | float] | None:
        if value is None:
            return None
        for name, number in value.items():
            if _LIMIT_NAME.fullmatch(name) is None:
                raise ValueError(f"invalid limit name {name!r}")
            if not _number(number) or number < 0:
                raise ValueError(f"limit {name!r} must be a non-negative number")
        return value

    @field_validator("attributes")
    @classmethod
    def _attribute_values(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        for name, item in value.items():
            if not _attribute_value(item):
                raise ValueError(f"attribute {name!r} must be a string, number, boolean, or list of those")
        return value

    def to_metadata(self) -> str:
        """Compact JSON for the registration ``metadata`` string."""
        text = self.model_dump_json(exclude_none=True, by_alias=True)
        size = len(text.encode("utf-8"))
        if size > METADATA_LIMIT_BYTES:
            raise LivepeerGatewayError(
                f"registration metadata is {size} bytes; limit is {METADATA_LIMIT_BYTES}"
            )
        return text


def _attribute_value(value: object) -> bool:
    if _number(value) or isinstance(value, (str, bool)):
        return True
    if isinstance(value, list):
        return all(_number(item) or isinstance(item, (str, bool)) for item in value)
    return False


def deployment_revision(
    images: list[str],
    profiles: list[str],
    settings: dict[str, Any],
    model: dict[str, Any] | None = None,
) -> str:
    """Canonical ``sha256:`` identity from the runner-contract recipe."""
    obj: dict[str, Any] = {
        "images": sorted(images),
        "profiles": sorted(profiles),
        "settings": settings,
    }
    if model is not None:
        obj["model"] = {"source": model["source"], "revision": model.get("revision")}
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def openapi_document() -> dict[str, Any]:
    """OpenAPI 3.1 document whose schemas are the envelope and its nested models."""
    schema = RunnerContractEnvelope.model_json_schema(ref_template="#/components/schemas/{model}")
    definitions = schema.pop("$defs", {})
    schemas = {"RunnerContractEnvelope": schema, **definitions}
    return {
        "openapi": "3.1.0",
        "info": {"title": "Live runner contract", "version": "1"},
        "components": {"schemas": schemas},
    }


def write_openapi(path: str | Path = "openapi.json") -> None:
    """Write ``openapi_document`` with sorted keys so the file stays stable."""
    text = json.dumps(openapi_document(), indent=2, sort_keys=True) + "\n"
    Path(path).write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    write_openapi(args[0] if args else "openapi.json")


if __name__ == "__main__":
    main()
