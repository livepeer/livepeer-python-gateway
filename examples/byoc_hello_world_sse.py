import argparse
import asyncio
import json
import logging
from typing import Optional

from livepeer_gateway import BYOCProcessRequest, LivepeerGatewayError, stream_byoc_request


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hello-world BYOC SSE request through the Python SDK.")
    parser.add_argument("--name", default="livepeer", help="Name to send to the worker.")
    parser.add_argument("--capability", default="hello-world", help="Registered BYOC capability name.")
    parser.add_argument("--route", default="predict-sse", help="Worker route after /process/request/.")
    parser.add_argument("--orchestrator", default=None, help="Optional orchestrator or gateway URL.")
    parser.add_argument("--signer", default=None, help="Remote signer base URL.")
    parser.add_argument("--discovery", default=None, help="Optional discovery endpoint URL.")
    parser.add_argument("--token", default=None, help="Optional gateway token.")
    parser.add_argument("--timeout-seconds", type=int, default=30, help="Signed BYOC request timeout.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def _parse_orchestrator_arg(orchestrator_arg: Optional[str]):
    if orchestrator_arg is None:
        return None
    parts = [part.strip() for part in orchestrator_arg.split(",") if part.strip()]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return parts


async def _amain() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        stream = stream_byoc_request(
            _parse_orchestrator_arg(args.orchestrator),
            BYOCProcessRequest(
                capability=args.capability,
                route=args.route,
                body={"name": args.name},
                timeout_seconds=args.timeout_seconds,
            ),
            token=args.token,
            signer_url=args.signer,
            discovery_url=args.discovery,
        )

        print("=== BYOC hello-world SSE ===")
        print(f"job_id:       {stream.job_id}")
        print(f"capability:   {stream.capability}")
        print(f"orchestrator: {stream.orchestrator_url}")
        print()

        async for event in stream.events:
            if event.data == "[DONE]":
                print("DONE")
                return
            try:
                payload = event.json()
            except json.JSONDecodeError:
                payload = event.data
            print(f"{event.event}: {json.dumps(payload, sort_keys=True) if isinstance(payload, dict) else payload}")
    except LivepeerGatewayError as err:
        print(f"ERROR: {err}")


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
