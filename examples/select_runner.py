#!/usr/bin/env python3
"""Discover a runner candidate using Livepeer Node's app and GPU filters."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from livepeer_gateway.errors import LivepeerGatewayError
from livepeer_gateway.selection import runner_selector


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Examples:\n"
            "  uv run examples/select_runner.py http://localhost:8935\n"
            "  uv run examples/select_runner.py http://localhost:8935 \\\n"
            "    --app livepeer-sample/echo\n"
            "  uv run examples/select_runner.py --signer http://localhost:8937 \\\n"
            "    --app example/app --gpu H100 --gpu L40S\n"
            "\n"
            "This prints the first discovered candidate. Calling cursor.next()\n"
            "would execute a single-shot app or reserve a persistent session."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "orchestrators",
        nargs="*",
        help="Orchestrator base URLs or host:port values; comma-separated lists are accepted.",
    )
    parser.add_argument(
        "--discovery",
        help="Explicit discovery endpoint URL (overrides signer discovery).",
    )
    parser.add_argument(
        "--signer",
        help="Remote signer base URL; discovery uses /discover-orchestrators.",
    )
    parser.add_argument(
        "--app",
        action="append",
        help="Exact app ID; repeat to match any listed app. Omit to allow all apps.",
    )
    parser.add_argument(
        "--gpu",
        action="append",
        help="Exact GPU name (gpu.name); repeat to match any listed GPU. Both app and GPU must match.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable discovery diagnostics."
    )
    args = parser.parse_args(argv)
    args.orchestrators = [
        url.strip()
        for value in args.orchestrators
        for url in value.split(",")
        if url.strip()
    ]
    if not (args.orchestrators or args.discovery or args.signer):
        parser.error("provide orchestrators, --discovery, or --signer")
    return args


async def main(args: argparse.Namespace) -> int:
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    try:
        cursor = await runner_selector(
            orchestrators=args.orchestrators or None,
            signer_url=args.signer,
            discovery_url=args.discovery,
            app=args.app,
            gpu=args.gpu,
        )
        # Inspect discovery metadata without executing an app or reserving capacity.
        runner = cursor.candidates[0]
        print("=== Selected Runner Candidate ===")
        print("Orchestrator:", runner.orchestrator_url)
        print(json.dumps(runner.raw, indent=2))
    except LivepeerGatewayError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(_parse_args())))
