#!/usr/bin/env python3
"""Submit a BYOC /process/request job via discovery + signer token."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from livepeer_gateway.byoc import ByocJobRequest, submit_byoc_job
from livepeer_gateway.errors import LivepeerGatewayError
from livepeer_gateway.token import parse_token


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Submit a BYOC job.")
    p.add_argument(
        "--capability",
        required=True,
        help="BYOC capability / app id (e.g. flux-schnell, nano-banana).",
    )
    p.add_argument(
        "--token",
        default=None,
        help="Base64 gateway token (signer, discovery, headers).",
    )
    p.add_argument("--signer", default=None, help="Remote signer URL.")
    p.add_argument("--discovery", default=None, help="Discovery URL.")
    p.add_argument(
        "--api-key",
        default=None,
        help="Bearer credential when not using --token (Authorization header).",
    )
    p.add_argument(
        "--prompt",
        default="a small red cube on a white table",
        help="Text prompt for image-style BYOC payloads.",
    )
    p.add_argument(
        "--payload",
        default="",
        help="Optional JSON object payload (overrides --prompt).",
    )
    p.add_argument(
        "--timeout-seconds",
        type=int,
        default=120,
        help="Job wait / payment credit seconds (default: 120).",
    )
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    signer_url = args.signer
    discovery_url = args.discovery
    signer_headers = None
    discovery_headers = None

    if args.token:
        token = parse_token(args.token)
        signer_url = token.get("signer") or signer_url
        discovery_url = token.get("discovery") or discovery_url
        signer_headers = token.get("signer_headers")
        discovery_headers = token.get("discovery_headers")
    elif args.api_key and signer_url:
        signer_headers = {"Authorization": f"Bearer {args.api_key.strip()}"}

    if args.payload.strip():
        payload = json.loads(args.payload)
    else:
        payload = {"prompt": args.prompt}

    # Prefer capability-filtered discovery when using discovery-service raw.
    if discovery_url and "/v1/discovery/raw" in discovery_url and "caps=" not in discovery_url:
        sep = "&" if "?" in discovery_url else "?"
        discovery_url = f"{discovery_url}{sep}caps={args.capability}"

    try:
        result = submit_byoc_job(
            req=ByocJobRequest(
                capability=args.capability,
                payload=payload,
                timeout_seconds=args.timeout_seconds,
            ),
            discovery_url=discovery_url,
            signer_url=signer_url,
            signer_headers=signer_headers,
            discovery_headers=discovery_headers,
        )
    except LivepeerGatewayError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(f"capability={args.capability}")
    print(f"job_id={result.job_id}")
    print(f"image_url={getattr(result, 'image_url', None)}")
    if result.data is not None:
        print(json.dumps(result.data, indent=2, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
