#!/usr/bin/env python3
"""Call any live-runner app using a gateway token or a Bearer API key.

``--token`` fills signer, headers, and discovery. ``--api-key`` sets
``Authorization: Bearer …`` on signer requests (wins over token headers).
``--discovery`` overrides token discovery while keeping signer credentials.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from contextlib import suppress
from typing import Any
from urllib.parse import urljoin

from livepeer_gateway.errors import LivepeerGatewayError
from livepeer_gateway.http import get_json
from livepeer_gateway.live_runner import (
    LiveRunnerInstance,
    LiveRunnerSession,
    call_runner,
)
from livepeer_gateway.selection import reserve_session, runner_selector
from livepeer_gateway.token import parse_token

log = logging.getLogger("call-live-runner")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover and call a live-runner capability/app.",
    )
    parser.add_argument(
        "--app",
        "--capability",
        dest="app",
        required=True,
        help="Live-runner app id (capability), e.g. livepeer-example/hello-world.",
    )
    parser.add_argument(
        "--token",
        default="",
        help="Base64 gateway token (signer, headers, discovery).",
    )
    parser.add_argument(
        "--discovery",
        default=None,
        help="Discovery URL. Overrides discovery from --token when set.",
    )
    parser.add_argument(
        "--signer",
        default="",
        help="Remote signer URL. Used when --token does not include signer.",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Bearer credential for the signer (Authorization header).",
    )
    parser.add_argument(
        "--path",
        default="",
        help="Path appended after the runner/app URL (e.g. /hello).",
    )
    parser.add_argument(
        "--json",
        dest="json_body",
        default="{}",
        help="JSON object payload for the call (default: {}).",
    )
    parser.add_argument(
        "--method",
        default="POST",
        help="HTTP method for the app call (default: POST).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds (default: 30).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def _join_path(base_url: str, path: str) -> str:
    base = base_url.rstrip("/") + "/"
    suffix = path.lstrip("/")
    if not suffix:
        return base_url.rstrip("/")
    return urljoin(base, suffix)


def _is_single_shot(runner: LiveRunnerInstance) -> bool:
    if runner.mode == "single-shot":
        return True
    return runner.url.rstrip("/").endswith("/app")


def _parse_payload(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--json must be a JSON object: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("--json must be a JSON object")
    return payload


def _resolve_auth(
    args: argparse.Namespace,
) -> tuple[str, str | None, dict[str, str] | None, dict[str, str] | None]:
    signer_url = args.signer.strip() or None
    signer_headers: dict[str, str] | None = None
    discovery_headers: dict[str, str] | None = None
    discovery_url = args.discovery.strip() if args.discovery else None

    raw = args.token.strip()
    if raw:
        token: dict[str, Any] = parse_token(raw)
        if token.get("signer") is not None:
            signer_url = token["signer"]
        if token.get("signer_headers") is not None:
            signer_headers = dict(token["signer_headers"])
        if token.get("discovery_headers") is not None:
            discovery_headers = dict(token["discovery_headers"])
        if not discovery_url and token.get("discovery") is not None:
            discovery_url = token["discovery"]
    if args.api_key.strip():
        signer_headers = {"Authorization": f"Bearer {args.api_key.strip()}"}

    if not discovery_url:
        raise SystemExit(
            "discovery URL required: pass --discovery or a token that includes discovery"
        )
    return discovery_url, signer_url, signer_headers, discovery_headers


def _raw_discovery_orchestrators(data: list[Any], *, app: str) -> list[str]:
    orchs: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        address = item.get("address")
        caps = item.get("capabilities")
        if not isinstance(address, str) or not address.strip():
            continue
        if not isinstance(caps, list) or app not in caps:
            continue
        orchs.append(address.strip())
    return orchs


async def _select_runner(
    *,
    app: str,
    discovery_url: str,
    signer_url: str | None,
    signer_headers: dict[str, str] | None,
    discovery_headers: dict[str, str] | None,
    timeout: float,
):
    data = await get_json(discovery_url, headers=discovery_headers)
    if isinstance(data, list) and data and isinstance(data[0], dict) and "capabilities" in data[0]:
        orchs = _raw_discovery_orchestrators(data, app=app)
        if not orchs:
            raise LivepeerGatewayError(
                f"no orchestrators advertise app {app!r} in {discovery_url}"
            )
        return await runner_selector(
            orchestrators=orchs,
            app=app,
            signer_url=signer_url,
            signer_headers=signer_headers,
            timeout=timeout,
        )

    return await runner_selector(
        discovery_url=discovery_url,
        discovery_headers=discovery_headers,
        app=app,
        signer_url=signer_url,
        signer_headers=signer_headers,
        timeout=timeout,
    )


async def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    payload = _parse_payload(args.json_body)
    discovery_url, signer_url, signer_headers, discovery_headers = _resolve_auth(args)
    session: LiveRunnerSession | None = None
    try:
        cursor = await _select_runner(
            app=args.app,
            discovery_url=discovery_url,
            signer_url=signer_url,
            signer_headers=signer_headers,
            discovery_headers=discovery_headers,
            timeout=args.timeout,
        )
        runner = cursor.candidates[0]
        log.info(
            "discovery=%s runner_url=%s app=%s mode=%s",
            discovery_url,
            runner.url,
            runner.app,
            runner.mode or "?",
        )

        if _is_single_shot(runner):
            result = await call_runner(
                runner=runner,
                runner_url=_join_path(runner.url, args.path),
                payload=payload,
                method=args.method,
                signer_url=signer_url,
                signer_headers=signer_headers,
                timeout=args.timeout,
            )
        else:
            session = await reserve_session(
                signer_url=signer_url,
                signer_headers=signer_headers,
                discovery_url=discovery_url,
                discovery_headers=discovery_headers,
                app=args.app,
                timeout=args.timeout,
            )
            log.info("session_id=%s app_url=%s", session.session_id, session.app_url)
            result = await call_runner(
                runner=session.runner or runner,
                runner_url=_join_path(session.app_url, args.path),
                payload=payload,
                method=args.method,
                signer_url=signer_url,
                signer_headers=signer_headers,
                timeout=args.timeout,
            )
        print(json.dumps(result.data, indent=2, sort_keys=True))
    except LivepeerGatewayError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    finally:
        if session is not None:
            with suppress(Exception):
                await session.aclose()


if __name__ == "__main__":
    asyncio.run(main())
