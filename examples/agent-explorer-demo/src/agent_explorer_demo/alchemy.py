from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Optional

from .state import AlchemyState, StateStore

# CLI network slugs (see `alchemy evm network list`). Agent Wallets support
# Arbitrum One + Sepolia; prefer mainnet to match PymtHouse Livepeer signing.
DEFAULT_ALCHEMY_NETWORK = "arb-mainnet"


class AlchemyError(RuntimeError):
    """Raised when Alchemy CLI is missing or a wallet command fails."""


@dataclass(frozen=True)
class AlchemyCommandResult:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    data: Any


def resolve_network() -> str:
    """Return preferred EVM network slug (env ALCHEMY_NETWORK, default arb-mainnet)."""
    raw = os.environ.get("ALCHEMY_NETWORK")
    if raw is None or not raw.strip():
        return DEFAULT_ALCHEMY_NETWORK
    return raw.strip()


def find_alchemy_bin() -> str:
    path = shutil.which("alchemy")
    if path:
        return path
    raise AlchemyError(
        "Alchemy CLI not found on PATH. Install with: "
        "npm i -g @alchemy/cli@latest"
    )


def _run_alchemy(
    *args: str,
    check: bool = True,
) -> AlchemyCommandResult:
    binary = find_alchemy_bin()
    argv = [binary, "--json", "--no-interactive", *args]
    proc = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        check=False,
    )
    data: Any = None
    stdout = (proc.stdout or "").strip()
    if stdout:
        try:
            data = json.loads(stdout)
        except json.JSONDecodeError:
            data = stdout
    result = AlchemyCommandResult(
        argv=argv,
        returncode=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
        data=data,
    )
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}"
        raise AlchemyError(f"alchemy {' '.join(args)} failed: {detail}")
    return result


def auth_status() -> AlchemyCommandResult:
    return _run_alchemy("auth", "status", check=False)


def require_auth() -> None:
    result = auth_status()
    authenticated = False
    if isinstance(result.data, dict):
        authenticated = bool(
            result.data.get("authenticated")
            or result.data.get("loggedIn")
            or result.data.get("email")
            or result.data.get("user")
        )
    if result.returncode != 0 or not authenticated:
        raise AlchemyError(
            "Alchemy CLI is not authenticated. Run once interactively:\n"
            "  alchemy auth login\n"
            "Or device-code (SSH/WSL without browser):\n"
            "  alchemy auth login --device-code\n"
            "  # open verificationUriComplete, approve in Alchemy dashboard\n"
            "Then:\n"
            "  alchemy wallet connect --mode session --instance-name agent-explorer-demo\n"
            "  alchemy wallet status --verify\n"
            "  # Ensure Arbitrum is on the app allowlist (Dashboard or CLI):\n"
            "  alchemy app configured-networks\n"
            "  alchemy app networks <appId> --networks ARB_MAINNET,ARB_SEPOLIA\n"
            "  agent-explorer-demo bootstrap"
        )


def wallet_connect(instance_name: str, *, force: bool = False) -> AlchemyCommandResult:
    args = [
        "wallet",
        "connect",
        "--mode",
        "session",
        "--instance-name",
        instance_name,
    ]
    if force:
        args.append("--force")
    return _run_alchemy(*args)


def wallet_status(*, verify: bool = True) -> AlchemyCommandResult:
    args = ["wallet", "status"]
    if verify:
        args.append("--verify")
    return _run_alchemy(*args)


def wallet_address() -> AlchemyCommandResult:
    return _run_alchemy("wallet", "address")


def evm_balance(address: str, *, network: str) -> AlchemyCommandResult:
    """Native balance for address on network (requires app network allowlist + API key)."""
    return _run_alchemy(
        "evm",
        "data",
        "balance",
        address,
        "-n",
        network,
    )


def configured_networks() -> AlchemyCommandResult:
    """RPC network slugs enabled on the selected Alchemy app."""
    return _run_alchemy("app", "configured-networks", check=False)


def _pick_address(data: Any, *keys: str) -> Optional[str]:
    if not isinstance(data, dict):
        return None
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        nested = data.get(key)
        if isinstance(nested, dict):
            for nested_key in ("address", "evm", "ethereum"):
                nested_value = nested.get(nested_key)
                if isinstance(nested_value, str) and nested_value.strip():
                    return nested_value.strip()
    return None


def _ensure_network_on_app(network: str) -> None:
    """Warn (do not fail) if preferred network is missing from the app allowlist."""
    result = configured_networks()
    if result.returncode != 0 or not isinstance(result.data, dict):
        return
    networks = result.data.get("networks")
    if not isinstance(networks, list):
        return
    slugs = {str(item) for item in networks}
    if network in slugs:
        return
    app_id = result.data.get("appId") or "<appId>"
    admin_id = network.upper().replace("-", "_")
    raise AlchemyError(
        f"Alchemy app is missing network '{network}'. Enable it, then re-run bootstrap:\n"
        f"  alchemy app networks {app_id} --networks {admin_id}\n"
        "Or Dashboard → Apps → Networks → enable Arbitrum Mainnet / Arbitrum Sepolia.\n"
        f"Currently configured: {', '.join(sorted(slugs)) or '(none)'}"
    )


def bootstrap_alchemy(
    store: StateStore,
    *,
    instance_name: str,
    network: Optional[str] = None,
) -> AlchemyState:
    """Ensure Alchemy auth + session wallet; verify balance on preferred network."""
    network = (network or resolve_network()).strip() or DEFAULT_ALCHEMY_NETWORK
    require_auth()
    _ensure_network_on_app(network)
    try:
        wallet_connect(instance_name)
    except AlchemyError as exc:
        # Reuse an existing session wallet instead of forcing a replace.
        detail = str(exc)
        if "wallet session already exists" not in detail.lower() and "INVALID_ARGS" not in detail:
            raise
    status = wallet_status(verify=True)
    address = wallet_address()

    session_address = _pick_address(
        address.data,
        "session",
        "sessionAddress",
        "address",
        "evm",
        "ethereum",
    )
    if session_address is None and isinstance(status.data, dict):
        session_address = _pick_address(
            status.data,
            "session",
            "sessionAddress",
            "address",
            "activeSigner",
            "walletAddress",
        )

    evm_address = _pick_address(address.data, "evm", "ethereum", "localEvm")
    solana_address = _pick_address(address.data, "solana", "localSolana")
    if isinstance(status.data, dict):
        if evm_address is None:
            by_chain = status.data.get("sessionsByChain")
            if isinstance(by_chain, dict):
                evm_address = _pick_address(by_chain.get("evm"), "walletAddress", "address")
                if solana_address is None:
                    solana_address = _pick_address(
                        by_chain.get("solana"),
                        "walletAddress",
                        "address",
                    )
        if session_address is None:
            session_address = status.data.get("walletAddress")
            if isinstance(session_address, str):
                session_address = session_address.strip() or None
        if evm_address is None and isinstance(session_address, str) and session_address.startswith("0x"):
            evm_address = session_address

    balance_value: Optional[str] = None
    balance_symbol: Optional[str] = None
    balance_address = evm_address or session_address
    if balance_address and balance_address.startswith("0x"):
        bal = evm_balance(balance_address, network=network)
        if isinstance(bal.data, dict):
            raw_balance = bal.data.get("balance")
            raw_symbol = bal.data.get("symbol")
            if isinstance(raw_balance, (str, int, float)):
                balance_value = str(raw_balance)
            if isinstance(raw_symbol, str) and raw_symbol.strip():
                balance_symbol = raw_symbol.strip()

    alchemy = AlchemyState(
        instance_name=instance_name,
        network=network,
        session_address=session_address,
        evm_address=evm_address,
        solana_address=solana_address,
        balance=balance_value,
        balance_symbol=balance_symbol,
        status=status.data if isinstance(status.data, dict) else {"raw": status.data},
    )

    state = store.load()
    state.alchemy = alchemy
    store.save(state)
    return alchemy
