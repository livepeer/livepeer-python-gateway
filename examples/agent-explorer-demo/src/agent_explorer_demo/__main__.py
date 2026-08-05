from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Optional

from dotenv import load_dotenv

from .alchemy import AlchemyError, bootstrap_alchemy, resolve_network
from .job_runner import JobRunnerError, run_lv2v_job
from .mcp_client import McpClientError, resolve_sdk_token, run_mcp_session
from .pymthouse_register import RegisterError, register_agent
from .state import StateStore, resolve_state_dir


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None:
        return default
    return float(raw)


def _store_from_args(args: argparse.Namespace) -> StateStore:
    return StateStore(resolve_state_dir(getattr(args, "state_dir", None)))


def _require_base_url() -> str:
    base = _env("PYMTHOUSE_BASE_URL")
    if not base:
        raise SystemExit(
            "PYMTHOUSE_BASE_URL is required. Copy .env.example to .env and set it."
        )
    return base.rstrip("/")


def cmd_bootstrap(args: argparse.Namespace) -> int:
    store = _store_from_args(args)
    instance = _env("ALCHEMY_INSTANCE_NAME", "agent-explorer-demo") or "agent-explorer-demo"
    if getattr(args, "instance_name", None):
        instance = args.instance_name
    network = resolve_network()
    if getattr(args, "network", None):
        network = args.network
    alchemy = bootstrap_alchemy(
        store,
        instance_name=instance,
        network=network,
    )
    print("=== Alchemy bootstrap (parallel onchain/x402 identity) ===")
    print(json.dumps(alchemy.__dict__, indent=2, default=str))
    print()
    print(
        "Note: Alchemy wallet is NOT used for PymtHouse Ed25519 register. "
        "Identities stay separate."
    )
    return 0


def cmd_register(args: argparse.Namespace) -> int:
    store = _store_from_args(args)
    base_url = _require_base_url()
    label = _env("AGENT_LABEL", "agent-explorer-demo")
    if getattr(args, "label", None):
        label = args.label
    register = register_agent(store, base_url=base_url, label=label)
    print("=== PymtHouse register ===")
    safe = {
        "public_key_hex": register.public_key_hex,
        "client_id": register.client_id,
        "external_user_id": register.external_user_id,
        "api_key_prefix": (register.api_key or "")[:20] + "…",
        "has_sdk_token": bool(register.sdk_token),
        "label": register.label,
    }
    print(json.dumps(safe, indent=2))
    return 0


async def _cmd_mcp_session_async(args: argparse.Namespace) -> int:
    store = _store_from_args(args)
    base_url = _require_base_url()
    state = store.load()
    api_key = state.register.api_key
    if not api_key:
        raise SystemExit("No apiKey in state. Run `register` first.")
    model_id = _env("MODEL_ID", "noop") or "noop"
    mcp = await run_mcp_session(
        store,
        base_url=base_url,
        api_key=api_key,
        model_id=model_id,
    )
    print("=== Hosted MCP session ===")
    print(
        json.dumps(
            {
                "mcp_url": f"{base_url}/api/v1/mcp",
                "has_access_token": bool(mcp.access_token),
                "signer_url": mcp.signer_url,
                "discovery_url": mcp.discovery_url,
                "has_sdk_token": bool(mcp.sdk_token),
                "client_id": mcp.client_id,
                "balance_usd_micros": mcp.balance_usd_micros,
                "info": mcp.info,
                "capabilities_keys": (
                    list(mcp.capabilities.keys()) if isinstance(mcp.capabilities, dict) else None
                ),
                "orchestrators_error": (
                    mcp.orchestrators.get("error")
                    if isinstance(mcp.orchestrators, dict)
                    else None
                ),
            },
            indent=2,
            default=str,
        )
    )
    return 0


def cmd_mcp_session(args: argparse.Namespace) -> int:
    return asyncio.run(_cmd_mcp_session_async(args))


async def _cmd_job_async(args: argparse.Namespace) -> int:
    store = _store_from_args(args)
    model_id = _env("MODEL_ID", "noop") or "noop"
    if getattr(args, "model", None):
        model_id = args.model
    orchestrator = _env("ORCHESTRATOR")
    if getattr(args, "orchestrator", None):
        orchestrator = args.orchestrator
    job = await run_lv2v_job(
        store,
        model_id=model_id,
        orchestrator=orchestrator,
        width=_env_int("JOB_WIDTH", 320),
        height=_env_int("JOB_HEIGHT", 180),
        fps=_env_float("JOB_FPS", 30.0),
        frame_count=_env_int("JOB_FRAME_COUNT", 30),
    )
    print("=== LV2V job ===")
    print(json.dumps(job.__dict__, indent=2, default=str))
    return 0 if not job.error else 1


def cmd_job(args: argparse.Namespace) -> int:
    return asyncio.run(_cmd_job_async(args))


def cmd_status(args: argparse.Namespace) -> int:
    store = _store_from_args(args)
    state = store.load()
    report = _status_report(state)
    print(json.dumps(report, indent=2, default=str))
    return 0


def _status_report(state: Any) -> dict[str, Any]:
    return {
        "alchemy": {
            "instance_name": state.alchemy.instance_name,
            "network": state.alchemy.network,
            "session_address": state.alchemy.session_address,
            "evm_address": state.alchemy.evm_address,
            "solana_address": state.alchemy.solana_address,
            "balance": state.alchemy.balance,
            "balance_symbol": state.alchemy.balance_symbol,
        },
        "pymthouse": {
            "external_user_id": state.register.external_user_id,
            "client_id": state.register.client_id,
            "public_key_hex": state.register.public_key_hex,
            "has_api_key": bool(state.register.api_key),
            "has_register_sdk_token": bool(state.register.sdk_token),
        },
        "mcp": {
            "tools_ok": bool(state.mcp.info),
            "has_sdk_token": bool(state.mcp.sdk_token),
            "signer_url": state.mcp.signer_url,
            "discovery_url": state.mcp.discovery_url,
            "balance_usd_micros": state.mcp.balance_usd_micros,
        },
        "job": {
            "publish_url": state.job.publish_url,
            "manifest_id": state.job.manifest_id,
            "model_id": state.job.model_id,
            "error": state.job.error,
        },
        "effective_sdk_token_present": bool(resolve_sdk_token(state)),
    }


async def _cmd_run_all_async(args: argparse.Namespace) -> int:
    store = _store_from_args(args)
    errors: list[str] = []

    # 1) Alchemy bootstrap (best-effort if CLI missing — report clearly)
    try:
        cmd_bootstrap(args)
    except AlchemyError as exc:
        errors.append(f"alchemy: {exc}")
        print(f"WARNING: Alchemy bootstrap skipped/failed: {exc}", file=sys.stderr)

    # 2) Register
    try:
        cmd_register(args)
    except RegisterError as exc:
        if exc.status == 409:
            state = store.load()
            if state.register.api_key:
                print(
                    "Register 409: public key already registered; reusing stored apiKey.",
                    file=sys.stderr,
                )
            else:
                errors.append(
                    "register 409 but no stored apiKey — delete .agent-demo keys or restore apiKey"
                )
                print(f"ERROR: {errors[-1]}", file=sys.stderr)
                return 1
        else:
            errors.append(f"register: {exc}")
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    # 3) MCP
    try:
        await _cmd_mcp_session_async(args)
    except (McpClientError, SystemExit) as exc:
        errors.append(f"mcp: {exc}")
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # 4) Job
    try:
        await _cmd_job_async(args)
    except JobRunnerError as exc:
        errors.append(f"job: {exc}")
        print(f"ERROR: job failed: {exc}", file=sys.stderr)

    print()
    print("=== run-all status report ===")
    print(json.dumps(_status_report(store.load()), indent=2, default=str))
    if errors:
        print()
        print("Errors:")
        for item in errors:
            print(f"  - {item}")
        return 1
    return 0


def cmd_run_all(args: argparse.Namespace) -> int:
    return asyncio.run(_cmd_run_all_async(args))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-explorer-demo",
        description=(
            "Demo agent: Alchemy wallet (parallel) + PymtHouse Ed25519 register + "
            "hosted MCP create_signer_session + livepeer_gateway LV2V job."
        ),
    )
    parser.add_argument(
        "--state-dir",
        default=None,
        help="Override AGENT_STATE_DIR (default: .agent-demo).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_boot = sub.add_parser("bootstrap", help="Alchemy auth check + session wallet connect")
    p_boot.add_argument("--instance-name", default=None)
    p_boot.add_argument(
        "--network",
        default=None,
        help="EVM network slug (default: ALCHEMY_NETWORK or arb-mainnet).",
    )
    p_boot.set_defaults(func=cmd_bootstrap)

    p_reg = sub.add_parser("register", help="Ed25519 challenge/register on PymtHouse")
    p_reg.add_argument("--label", default=None)
    p_reg.set_defaults(func=cmd_register)

    p_mcp = sub.add_parser(
        "mcp-session",
        help="Hosted MCP: info, capabilities/query smoke, create_signer_session",
    )
    p_mcp.set_defaults(func=cmd_mcp_session)

    p_job = sub.add_parser("job", help="Submit LV2V job via livepeer_gateway with sdk_token")
    p_job.add_argument("--model", default=None)
    p_job.add_argument("--orchestrator", default=None)
    p_job.set_defaults(func=cmd_job)

    p_all = sub.add_parser("run-all", help="bootstrap → register → mcp-session → job")
    p_all.add_argument("--instance-name", default=None)
    p_all.add_argument(
        "--network",
        default=None,
        help="EVM network slug (default: ALCHEMY_NETWORK or arb-mainnet).",
    )
    p_all.add_argument("--label", default=None)
    p_all.add_argument("--model", default=None)
    p_all.add_argument("--orchestrator", default=None)
    p_all.set_defaults(func=cmd_run_all)

    p_status = sub.add_parser("status", help="Print persisted demo state summary")
    p_status.set_defaults(func=cmd_status)

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        code = args.func(args)
    except (AlchemyError, RegisterError, McpClientError, JobRunnerError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    raise SystemExit(code)


if __name__ == "__main__":
    main()
