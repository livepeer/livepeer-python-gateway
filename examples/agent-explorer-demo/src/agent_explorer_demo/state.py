from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


DEFAULT_STATE_DIR = ".agent-demo"
STATE_FILENAME = "state.json"
PRIVATE_KEY_FILENAME = "ed25519_private.hex"


@dataclass
class AlchemyState:
    instance_name: str = "agent-explorer-demo"
    network: Optional[str] = None
    session_address: Optional[str] = None
    evm_address: Optional[str] = None
    solana_address: Optional[str] = None
    balance: Optional[str] = None
    balance_symbol: Optional[str] = None
    status: Optional[dict[str, Any]] = None


@dataclass
class RegisterState:
    public_key_hex: Optional[str] = None
    client_id: Optional[str] = None
    external_user_id: Optional[str] = None
    api_key: Optional[str] = None
    sdk_token: Optional[str] = None
    key_id: Optional[str] = None
    label: Optional[str] = None


@dataclass
class McpSessionState:
    access_token: Optional[str] = None
    signer_url: Optional[str] = None
    discovery_url: Optional[str] = None
    sdk_token: Optional[str] = None
    client_id: Optional[str] = None
    balance_usd_micros: Optional[str] = None
    info: Optional[dict[str, Any]] = None
    capabilities: Optional[dict[str, Any]] = None
    orchestrators: Optional[dict[str, Any]] = None


@dataclass
class JobState:
    publish_url: Optional[str] = None
    subscribe_url: Optional[str] = None
    control_url: Optional[str] = None
    events_url: Optional[str] = None
    manifest_id: Optional[str] = None
    model_id: Optional[str] = None
    orchestrator: Optional[str] = None
    error: Optional[str] = None


@dataclass
class DemoState:
    alchemy: AlchemyState = field(default_factory=AlchemyState)
    register: RegisterState = field(default_factory=RegisterState)
    mcp: McpSessionState = field(default_factory=McpSessionState)
    job: JobState = field(default_factory=JobState)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DemoState":
        return cls(
            alchemy=AlchemyState(**(data.get("alchemy") or {})),
            register=RegisterState(**(data.get("register") or {})),
            mcp=McpSessionState(**(data.get("mcp") or {})),
            job=JobState(**(data.get("job") or {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StateStore:
    """Persist demo artifacts under AGENT_STATE_DIR (default `.agent-demo/`)."""

    def __init__(
        self,
        state_dir: Path | str,
    ) -> None:
        self.state_dir = Path(state_dir).expanduser().resolve()
        self.state_path = self.state_dir / STATE_FILENAME
        self.private_key_path = self.state_dir / PRIVATE_KEY_FILENAME

    def ensure_dir(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.chmod(0o700)

    def load(self) -> DemoState:
        if not self.state_path.is_file():
            return DemoState()
        raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid state file (expected object): {self.state_path}")
        return DemoState.from_dict(raw)

    def save(self, state: DemoState) -> None:
        self.ensure_dir()
        payload = json.dumps(state.to_dict(), indent=2, sort_keys=True) + "\n"
        self.state_path.write_text(payload, encoding="utf-8")
        self.state_path.chmod(0o600)

    def write_private_key_hex(self, private_key_hex: str) -> None:
        self.ensure_dir()
        self.private_key_path.write_text(private_key_hex.strip() + "\n", encoding="utf-8")
        self.private_key_path.chmod(0o600)

    def read_private_key_hex(self) -> Optional[str]:
        if not self.private_key_path.is_file():
            return None
        value = self.private_key_path.read_text(encoding="utf-8").strip()
        return value or None


def resolve_state_dir(explicit: Optional[str] = None) -> Path:
    raw = explicit or os.environ.get("AGENT_STATE_DIR") or DEFAULT_STATE_DIR
    return Path(raw).expanduser()
