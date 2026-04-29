import argparse
import json
import logging
from typing import Optional

from livepeer_gateway import BYOCProcessRequest, LivepeerGatewayError, process_byoc_request


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hello-world BYOC request through the Python SDK.")
    parser.add_argument("--name", default="livepeer", help="Name to send to the worker.")
    parser.add_argument("--capability", default="hello-world", help="Registered BYOC capability name.")
    parser.add_argument("--route", default="predict", help="Worker route after /process/request/.")
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


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        response = process_byoc_request(
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
    except LivepeerGatewayError as err:
        print(f"ERROR: {err}")
        return

    print("=== BYOC hello-world ===")
    print(f"job_id:       {response.job_id}")
    print(f"capability:   {response.capability}")
    print(f"orchestrator: {response.orchestrator_url}")
    print(f"status:       {response.status_code}")
    print("body:")
    print(json.dumps(response.body, indent=2, sort_keys=True) if isinstance(response.body, dict) else response.body)


if __name__ == "__main__":
    main()
