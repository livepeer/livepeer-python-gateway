#!/usr/bin/env python3
"""Discover and call api-proxy over live-runner capabilities."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from livepeer_gateway.errors import LivepeerGatewayError
from livepeer_gateway.live_runner import call_runner
from livepeer_gateway.selection import runner_selector

DEFAULT_DISCOVERY = "https://localhost:8935/discovery"
APP_ID = "livepeer-example/fal-flux-schnell"
EXECUTE_PATH = "/proxy"
DEFAULT_PROMPT = "A lighthouse in a winter storm, editorial photograph"

log = logging.getLogger("fal-flux-schnell")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover and call livepeer-example/fal-flux-schnell.",
    )
    parser.add_argument("--discovery", default=DEFAULT_DISCOVERY)
    parser.add_argument("--signer", default="", help="Remote signer base URL.")
    parser.add_argument(
        "--api-key",
        default="",
        help="Bearer credential for the signer (Authorization header).",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--input-json",
        type=Path,
        default=None,
        help="JSON object payload. Overrides --prompt when set.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=210.0,
        help="Request timeout in seconds (default: 210).",
    )
    return parser.parse_args()


def _payload(args: argparse.Namespace) -> dict[str, Any]:
    if args.input_json is None:
        return {
            "prompt": args.prompt,
            "image_size": "landscape_4_3",
            "num_inference_steps": 4,
            "num_images": 1,
            "output_format": "jpeg",
            "seed": 12345,
        }
    try:
        data = json.loads(args.input_json.expanduser().read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"--input-json must be a JSON object: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit("--input-json must be a JSON object")
    return data


def _resolve_auth(
    args: argparse.Namespace,
) -> tuple[str, str | None, dict[str, str] | None]:
    signer_url = args.signer.strip() or None
    signer_headers: dict[str, str] | None = None
    if args.api_key.strip():
        signer_headers = {"Authorization": f"Bearer {args.api_key.strip()}"}
    return args.discovery, signer_url, signer_headers


async def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    discovery_url, signer_url, signer_headers = _resolve_auth(args)
    try:
        cursor = await runner_selector(
            discovery_url=discovery_url,
            app=APP_ID,
            signer_url=signer_url,
            signer_headers=signer_headers,
        )
        runner = cursor.candidates[0]
        runner_url = urljoin(runner.url.rstrip("/") + "/", EXECUTE_PATH.lstrip("/"))
        log.info("runner_url=%s mode=%s", runner_url, runner.mode or "?")
        result = await call_runner(
            runner=runner,
            runner_url=runner_url,
            payload=_payload(args),
            signer_url=signer_url,
            signer_headers=signer_headers,
            timeout=args.timeout,
        )
    except LivepeerGatewayError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(result.data, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
