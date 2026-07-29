"""SDK client for the text stream demo: a self-funding single-shot SSE call.

Offchain (no payments):

    uv run client.py http://localhost:8935/apps/story-runner/app/sse

On-chain: pass a remote signer. call_runner pays the 402 challenge and,
for metered pricing (hour/seconds or 720p), keeps funding the call on a
cadence while the stream runs — no payment code needed here.

    uv run client.py https://orch:8935/apps/story-runner/app/sse \
        --signer-url http://localhost:7936
"""
from __future__ import annotations

import argparse
import asyncio

from livepeer_gateway.live_runner import call_runner


async def main() -> None:
    p = argparse.ArgumentParser(description="Stream a story from a single-shot runner.")
    p.add_argument("runner_url", help="Runner app endpoint, e.g. .../apps/story-runner/app/sse")
    p.add_argument("--signer-url", default=None, help="Remote signer URL. Omit for offchain.")
    p.add_argument(
        "--payment-unit",
        default=None,
        help="hour|seconds|720p|720p-pixel-seconds|fixed. Default: metered (live).",
    )
    args = p.parse_args()

    stream = await call_runner(
        args.runner_url,
        method="GET",
        signer_url=args.signer_url,
        payment_unit=args.payment_unit,
        stream=True,
    )
    async with stream:
        async for line in stream.aiter_lines():
            print(line)
        if stream.released:
            print("(session released by orchestrator)")


if __name__ == "__main__":
    asyncio.run(main())
