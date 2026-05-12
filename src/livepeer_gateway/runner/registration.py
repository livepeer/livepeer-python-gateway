"""Fire-and-forget auto-registration with a Livepeer orchestrator.

Registers on container startup, deregisters on graceful shutdown.
"""

# TODO: replace with heartbeat-based registration once upstream go-livepeer design lands.

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os

import aiohttp

_LOG = logging.getLogger(__name__)

# Tunable when a real ask arrives; module constants for now.
_REGISTER_TIMEOUT_S = 5.0
_MAX_ATTEMPTS = 30
_RETRY_DELAY_S = 2.0
_DEREGISTER_TIMEOUT_S = 2.0


@dataclasses.dataclass(frozen=True)
class RegistrationConfig:
    """Auto-registration config; built from ``LIVEPEER_*`` env via :meth:`from_env`."""

    orch_url: str
    orch_secret: str
    capability_name: str
    capability_url: str
    capacity: int = 1
    price_per_unit: int = 0

    @classmethod
    def from_env(cls) -> "RegistrationConfig | None":
        """Returns ``None`` if any required ``LIVEPEER_*`` env var is missing."""
        required = {
            "orch_url": os.environ.get("LIVEPEER_ORCH_URL"),
            "orch_secret": os.environ.get("LIVEPEER_ORCH_SECRET"),
            "capability_name": os.environ.get("LIVEPEER_CAPABILITY_NAME"),
            "capability_url": os.environ.get("LIVEPEER_CAPABILITY_URL"),
        }
        if any(v is None for v in required.values()):
            return None
        return cls(
            **required,
            capacity=int(os.environ.get("LIVEPEER_CAPACITY", "1")),
            price_per_unit=int(os.environ.get("LIVEPEER_PRICE_PER_UNIT", "0")),
        )


async def register(cfg: RegistrationConfig) -> None:
    """POST ``/capability/register``; retry transients, fail-fast on 4xx.

    Retry policy:
      * Connection refused / timeout / 5xx → transient → retry
      * 400 / 404 / 405                    → permanent → :exc:`RuntimeError`

    Raises :exc:`RuntimeError` after ``_MAX_ATTEMPTS`` exhausted.
    """
    payload = {
        "name": cfg.capability_name,
        "url": cfg.capability_url,
        "capacity": cfg.capacity,
        "price_per_unit": cfg.price_per_unit,
        "price_scaling": 1,
        "currency": "wei",
    }
    url = f"{cfg.orch_url.rstrip('/')}/capability/register"
    headers = {"Authorization": cfg.orch_secret}
    timeout = aiohttp.ClientTimeout(total=_REGISTER_TIMEOUT_S)

    async with aiohttp.ClientSession() as session:
        for _ in range(_MAX_ATTEMPTS):
            try:
                async with session.post(
                    url,
                    json=payload,
                    headers=headers,
                    ssl=False,  # Go-livepeer uses self-signed certs.
                    timeout=timeout,
                ) as r:
                    if r.status == 200:
                        _LOG.info(
                            "registered capability=%s url=%s",
                            cfg.capability_name,
                            cfg.capability_url,
                        )
                        return
                    if r.status in (400, 404, 405):
                        body = await r.text()
                        raise RuntimeError(f"orch rejected ({r.status}): {body}")
            except (aiohttp.ClientConnectionError, asyncio.TimeoutError):
                pass
            await asyncio.sleep(_RETRY_DELAY_S)

    raise RuntimeError(f"registration failed after {_MAX_ATTEMPTS} attempts")


async def deregister(cfg: RegistrationConfig) -> None:
    """Best-effort POST ``/capability/unregister``.

    Asymmetric wire format: register expects JSON, unregister expects the
    plain-text capability name as the body
    (go-livepeer ``byoc/job_orchestrator.go:90``).
    """
    url = f"{cfg.orch_url.rstrip('/')}/capability/unregister"
    timeout = aiohttp.ClientTimeout(total=_DEREGISTER_TIMEOUT_S)

    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            data=cfg.capability_name,  # plain text body, NOT JSON
            headers={"Authorization": cfg.orch_secret},
            ssl=False,  # Go-livepeer uses self-signed certs.
            timeout=timeout,
        ) as r:
            if r.status == 200:
                _LOG.info("deregistered capability=%s", cfg.capability_name)
            else:
                body = await r.text()
                _LOG.warning("deregister returned %d: %s", r.status, body)
