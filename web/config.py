from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Config:
    """Environment-based configuration for the web wrapper."""

    # Orchestrator URL(s), comma-separated. If empty, discovery is used.
    orchestrator_url: Optional[str] = field(
        default_factory=lambda: os.environ.get("ORCHESTRATOR_URL") or None
    )

    # Remote signer URL. If empty, runs in offchain mode.
    signer_url: Optional[str] = field(
        default_factory=lambda: os.environ.get("SIGNER_URL") or None
    )

    # Base64-encoded JSON token for authentication.
    livepeer_token: Optional[str] = field(
        default_factory=lambda: os.environ.get("LIVEPEER_TOKEN") or None
    )

    # Default model ID for jobs.
    default_model_id: str = field(
        default_factory=lambda: os.environ.get("DEFAULT_MODEL_ID", "noop")
    )

    # FPS for media publishing.
    fps: float = field(
        default_factory=lambda: float(os.environ.get("FPS", "24"))
    )

    # JPEG quality for output frames sent to browser (0-100).
    jpeg_quality: int = field(
        default_factory=lambda: int(os.environ.get("JPEG_QUALITY", "80"))
    )

    # Host and port for uvicorn.
    host: str = field(
        default_factory=lambda: os.environ.get("HOST", "0.0.0.0")
    )
    port: int = field(
        default_factory=lambda: int(os.environ.get("PORT", "8000"))
    )

    # --- Authentication ---

    # Comma-separated API keys. When empty, auth is disabled (dev mode).
    api_keys: str = field(
        default_factory=lambda: os.environ.get("API_KEYS", "")
    )

    # Maximum concurrent jobs per API key.
    max_jobs_per_key: int = field(
        default_factory=lambda: int(os.environ.get("MAX_JOBS_PER_KEY", "10"))
    )

    # --- Daydream ---

    # Daydream signer URL. When set, used as the signer URL for all jobs.
    daydream_url: Optional[str] = field(
        default_factory=lambda: os.environ.get("DAYDREAM_URL") or None
    )

    @property
    def parsed_api_keys(self) -> set[str]:
        """Return the set of valid API keys (stripped, non-empty)."""
        if not self.api_keys:
            return set()
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    @property
    def auth_enabled(self) -> bool:
        """True when at least one API key is configured."""
        return bool(self.parsed_api_keys)

    @property
    def effective_signer_url(self) -> Optional[str]:
        """DAYDREAM_URL takes precedence over SIGNER_URL."""
        return self.daydream_url or self.signer_url
