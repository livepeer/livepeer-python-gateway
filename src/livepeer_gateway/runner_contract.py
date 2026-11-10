"""Runner contract v1 envelope for live-runner registration metadata.

The shape matches the live-runner-standard envelope locked on 5 November 2026.
``to_metadata`` emits the compact JSON string the orchestrator stores.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
)

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


# Strict types: in lax mode pydantic turns True into 1 for an int, and NaN or infinity serialize
# as JSON null. A limit or attribute must reach the orchestrator as the number the caller meant.
_Number = StrictInt | Annotated[StrictFloat, Field(allow_inf_nan=False)]
_Limit = Annotated[StrictInt, Field(ge=0)] | Annotated[StrictFloat, Field(ge=0, allow_inf_nan=False)]
_Scalar = StrictStr | StrictBool | _Number
_Attribute = _Scalar | list[_Scalar]


class RunnerContractEnvelope(BaseModel):
    """Runner contract v1 registration metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    runner_contract: Literal[1] = 1
    deployment_revision: Annotated[str, Field(pattern=_DIGEST)]
    model: RunnerContractModel | None = None
    endpoints: dict[str, RunnerContractEndpoint]
    runner_id: Annotated[str, Field(min_length=1)] | None = None
    runtime: RunnerContractRuntime | None = None
    limits: dict[str, _Limit] | None = None
    admission: str | None = None
    capacity_group: RunnerContractCapacityGroup | None = None
    attributes: dict[str, _Attribute] | None = None

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
    def _limit_names(cls, value: dict[str, int | float] | None) -> dict[str, int | float] | None:
        for name in value or {}:
            if _LIMIT_NAME.fullmatch(name) is None:
                raise ValueError(f"invalid limit name {name!r}")
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


def deployment_revision(
    images: list[str],
    profiles: list[str],
    settings: dict[str, Any],
    model: RunnerContractModel | Mapping[str, Any] | None = None,
) -> str:
    """Canonical ``sha256:`` identity from the runner-contract recipe."""
    obj: dict[str, Any] = {
        "images": sorted(images),
        "profiles": sorted(profiles),
        "settings": settings,
    }
    if model is not None:
        obj["model"] = _model_identity(model)
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _model_identity(model: RunnerContractModel | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(model, RunnerContractModel):
        source, revision = model.source, model.revision
    else:
        source, revision = model.get("source"), model.get("revision")
    if not isinstance(source, str) or not source:
        raise ValueError("deployment_revision needs the model's source")
    # The recipe hashes an absent revision as null, not as an omitted key. comfystream's runner
    # computes it the same way, and both must agree on one digest for one deployment.
    return {"source": source, "revision": revision}


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
