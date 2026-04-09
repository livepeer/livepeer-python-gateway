import argparse
import json
import logging
from typing import Any

from livepeer_gateway.capabilities import (
    compute_available,
    format_capability,
    get_capacity_in_use,
    get_per_capability_map,
)
from livepeer_gateway import get_orch_info
from livepeer_gateway.orchestrator import LivepeerGatewayError, discover_orchestrators
from livepeer_gateway.token import parse_token

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch Livepeer orchestrator info.",
        epilog=(
            "Examples, in priority order of application (highest first):\n"
            "  # Orchestrator list\n"
            "  python examples/get_orchestrator_info.py localhost:8935 localhost:8936\n"
            "  python examples/get_orchestrator_info.py 'localhost:8935,localhost:8936'\n"
            "  python examples/get_orchestrator_info.py localhost:8935 --signer https://signer.example.com\n"
            "\n"
            "  # Discovery URL\n"
            "  python examples/get_orchestrator_info.py --discovery https://discover.example.com/orchestrators\n"
            "  python examples/get_orchestrator_info.py --discovery https://discover.example.com/orchestrators --signer https://signer.example.com\n"
            "\n"
            "  # Gateway token\n"
            "  python examples/get_orchestrator_info.py --token <base64-token>\n"
            "  python examples/get_orchestrator_info.py localhost:8935 --token <base64-token>\n"
            "\n"
            "  # Signer URL\n"
            "  python examples/get_orchestrator_info.py --signer https://signer.example.com\n"
            "\n"
            "  # JSON / JSONL output\n"
            "  python examples/get_orchestrator_info.py localhost:8935 --format json\n"
            "  python examples/get_orchestrator_info.py localhost:8935 --format jsonl\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "orchestrators",
        nargs="*",
        help="Optional list of orchestrators (host:port) or comma-delimited string.",
    )
    p.add_argument(
        "--discovery",
        default=None,
        help="Explicit discovery endpoint URL (overrides signer discovery).",
    )
    p.add_argument(
        "--signer",
        default=None,
        help="Remote signer base URL (no path). Can be combined with list/discovery.",
    )
    p.add_argument(
        "--token",
        default=None,
        help="Base64-encoded gateway token; token fields override explicit signer/discovery/orchestrator args.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging for discovery diagnostics.",
    )
    p.add_argument(
        "--format",
        choices=("text", "json", "jsonl"),
        default="text",
        help="Output format: text (default), json, or jsonl.",
    )
    return p.parse_args()


def _orch_error_summary(orch_url: str | None, err: Exception) -> dict[str, Any]:
    return {
        "orchestrator": orch_url,
        "ok": False,
        "error": str(err),
    }


def _orch_info_summary(orch_url: str, info: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "orchestrator": orch_url,
        "ok": True,
        "transcoder": info.transcoder,
        "eth_address": info.address.hex(),
        "version": None,
        "capabilities": None,
        "pricing": None,
        "hardware": None,
    }

    if not (info.HasField("capabilities") and info.capabilities.version):
        return result

    result["version"] = info.capabilities.version
    caps = info.capabilities
    per_capability = get_per_capability_map(caps)
    cap_ids = set(caps.capacities.keys())
    cap_ids.update(per_capability.keys())

    capabilities: list[dict[str, Any]] = []
    for cap_id in sorted(cap_ids):
        cap_entry: dict[str, Any] = {
            "capability_name": format_capability(cap_id),
        }
        has_capacity = cap_id in caps.capacities
        if has_capacity and caps.capacities[cap_id] > 1:
            cap_entry["total_capacity"] = int(caps.capacities[cap_id])

        cap_constraints = per_capability.get(cap_id)
        models = getattr(cap_constraints, "models", None) if cap_constraints else None
        if models:
            model_entries: dict[str, Any] = {}
            for model_name, model_constraint in models.items():
                warm = bool(getattr(model_constraint, "warm", False))
                runner = str(getattr(model_constraint, "runnerVersion", "")) or "-"
                capacity = int(getattr(model_constraint, "capacity", 0) or 0)
                in_use = get_capacity_in_use(model_constraint)
                available = compute_available(capacity, in_use)
                model_entries[model_name] = {
                    "warm": warm,
                    "runner": runner,
                    "capacity": capacity,
                    "in_use": in_use,
                    "available": available,
                }
            cap_entry["models"] = model_entries
        capabilities.append(cap_entry)
    result["capabilities"] = capabilities

    has_general_price = info.HasField("price_info") and info.price_info.pricePerUnit > 0
    has_cap_prices = bool(info.capabilities_prices)
    if has_general_price or has_cap_prices:
        general = None
        if has_general_price:
            general = {
                "price_per_unit_wei": int(info.price_info.pricePerUnit),
                "pixels_per_unit": int(info.price_info.pixelsPerUnit if info.price_info.pixelsPerUnit > 0 else 1),
            }

        cap_specific = []
        for cap_price in info.capabilities_prices:
            if cap_price.pricePerUnit <= 0:
                continue
            cap_id = cap_price.capability if cap_price.capability else "?"
            cap_name = format_capability(cap_id) if cap_id != "?" else "unknown"
            cap_specific.append(
                {
                    "capability_name": cap_name,
                    "constraint": cap_price.constraint or None,
                    "price_per_unit_wei": int(cap_price.pricePerUnit),
                    "pixels_per_unit": int(cap_price.pixelsPerUnit if cap_price.pixelsPerUnit > 0 else 1),
                }
            )

        result["pricing"] = {
            "general": general,
            "capability_specific": cap_specific,
        }

    if info.hardware:
        hardware_entries = []
        for hw in info.hardware:
            hw_entry: dict[str, Any] = {
                "pipeline": hw.pipeline or "-",
                "model_id": hw.model_id or "-",
                "gpu_info": None,
            }
            if hw.gpu_info:
                gpu_info: dict[str, Any] = {}
                for key, gpu in hw.gpu_info.items():
                    mem_free_bytes = int(gpu.memory_free)
                    mem_total_bytes = int(gpu.memory_total)
                    gpu_info[key] = {
                        "id": gpu.id or "-",
                        "name": gpu.name or "-",
                        "compute": f"{gpu.major}.{gpu.minor}",
                        "memory_free_bytes": mem_free_bytes,
                        "memory_total_bytes": mem_total_bytes,
                        "memory_free_human": format_bytes(mem_free_bytes),
                        "memory_total_human": format_bytes(mem_total_bytes),
                    }
                hw_entry["gpu_info"] = gpu_info
            hardware_entries.append(hw_entry)
        result["hardware"] = hardware_entries

    return result


def _print_text_info(orch_url: str, info: Any) -> None:
    print("=== OrchestratorInfo ===")
    print("Orchestrator:", orch_url)
    print("Transcoder URI:", info.transcoder)
    print("ETH Address:", info.address.hex())
    if info.HasField("capabilities") and info.capabilities.version:
        print("Version:", info.capabilities.version)
    else:
        print("Capabilities: not provided")
        print()
        return
    print()

    caps = info.capabilities
    per_capability = get_per_capability_map(caps)

    cap_ids = set(caps.capacities.keys())
    cap_ids.update(per_capability.keys())

    if not cap_ids:
        print("Capabilities: none advertised")
        print()
        return

    print("Capabilities:")
    for cap_id in sorted(cap_ids):
        print(f"- {format_capability(cap_id)}")

        has_capacity = cap_id in caps.capacities
        if has_capacity and caps.capacities[cap_id] > 1:
            print(f"  Total capacity: {caps.capacities[cap_id]}")

        cap_constraints = per_capability.get(cap_id)
        models = getattr(cap_constraints, "models", None) if cap_constraints else None
        if models:
            print("  Models:")
            for model_name, model_constraint in models.items():
                warm = bool(getattr(model_constraint, "warm", False))
                runner = str(getattr(model_constraint, "runnerVersion", "")) or "-"
                capacity = int(getattr(model_constraint, "capacity", 0) or 0)
                in_use = get_capacity_in_use(model_constraint)
                available = compute_available(capacity, in_use)
                print(
                    "    "
                    f"{model_name}: warm={warm} runner={runner} "
                    f"capacity={capacity} in_use={in_use} available={available}"
                )

        if not has_capacity and not models:
            print("  Capacity: not provided")

    print()

    has_general_price = info.HasField("price_info") and info.price_info.pricePerUnit > 0
    has_cap_prices = bool(info.capabilities_prices)

    if has_general_price or has_cap_prices:
        print("Pricing:")

        if has_general_price:
            price_per_unit = info.price_info.pricePerUnit
            pixels_per_unit = info.price_info.pixelsPerUnit if info.price_info.pixelsPerUnit > 0 else 1
            print(f"  General: {price_per_unit} wei per {pixels_per_unit} pixel(s)")

        if has_cap_prices:
            if has_general_price:
                print("  Capability-specific prices:")
            for cap_price in info.capabilities_prices:
                if cap_price.pricePerUnit > 0:
                    cap_id = cap_price.capability if cap_price.capability else "?"
                    cap_name = format_capability(cap_id) if cap_id != "?" else "unknown"
                    price_per_unit = cap_price.pricePerUnit
                    pixels_per_unit = cap_price.pixelsPerUnit if cap_price.pixelsPerUnit > 0 else 1
                    constraint = f" [{cap_price.constraint}]" if cap_price.constraint else ""
                    indent = "    " if has_general_price else "  "
                    print(f"{indent}{cap_name} {constraint}: {price_per_unit} wei per {pixels_per_unit} pixel(s)")

        print()
    else:
        print("Pricing: not provided")
        print()

    if info.hardware:
        print("Hardware / GPU:")
        for hw in info.hardware:
            pipeline = hw.pipeline or "-"
            model_id = hw.model_id or "-"
            print(f"- Pipeline: {pipeline} | Model: {model_id}")
            if not hw.gpu_info:
                print("  GPU info: not provided")
                continue
            for key, gpu in hw.gpu_info.items():
                gpu_id = gpu.id or "-"
                name = gpu.name or "-"
                compute = f"{gpu.major}.{gpu.minor}"
                mem_free = format_bytes(int(gpu.memory_free))
                mem_total = format_bytes(int(gpu.memory_total))
                print(
                    "  "
                    f"{key}: id={gpu_id} name={name} "
                    f"compute={compute} mem_free={mem_free} mem_total={mem_total}"
                )
        print()
    else:
        print("Hardware / GPU: not provided")
        print()


def _print_text_error(orch_url: str | None, err: Exception) -> None:
    if orch_url is None:
        print(f"ERROR: {err}")
        print()
        return
    print("=== OrchestratorInfo ===")
    print("Orchestrator:", orch_url)
    print(f"ERROR: {err}")
    print()


def _print_json_info(orch_url: str, info: Any, json_results: list[dict[str, Any]]) -> None:
    json_results.append(_orch_info_summary(orch_url, info))


def _print_json_error(orch_url: str | None, err: Exception, json_results: list[dict[str, Any]]) -> None:
    json_results.append(_orch_error_summary(orch_url, err))


def _print_jsonl_info(orch_url: str, info: Any) -> None:
    print(json.dumps(_orch_info_summary(orch_url, info), sort_keys=True))


def _print_jsonl_error(orch_url: str | None, err: Exception) -> None:
    print(json.dumps(_orch_error_summary(orch_url, err), sort_keys=True))


def _resolve_discovery_args(args: argparse.Namespace) -> tuple[Any, str | None, dict[str, str] | None, str | None, dict[str, str] | None]:
    token_data = parse_token(args.token) if args.token else None

    orchestrators = token_data.get("orchestrators") if token_data else None
    if orchestrators is None:
        orchestrators = args.orchestrators

    signer = token_data.get("signer") if token_data else None
    if signer is None:
        signer = args.signer

    signer_headers = token_data.get("signer_headers") if token_data else None

    discovery = token_data.get("discovery") if token_data else None
    if discovery is None:
        discovery = args.discovery

    discovery_headers = token_data.get("discovery_headers") if token_data else None

    return orchestrators, signer, signer_headers, discovery, discovery_headers


def main() -> None:
    args = _parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    json_results: list[dict[str, Any]] = []

    def _json_info(orch_url: str, info: Any) -> None:
        _print_json_info(orch_url, info, json_results)

    def _json_error(orch_url: str | None, err: Exception) -> None:
        _print_json_error(orch_url, err, json_results)

    info_printers = {
        "text": _print_text_info,
        "json": _json_info,
        "jsonl": _print_jsonl_info,
    }
    error_printers = {
        "text": _print_text_error,
        "json": _json_error,
        "jsonl": _print_jsonl_error,
    }
    print_info = info_printers[args.format]
    print_error = error_printers[args.format]

    try:
        orchestrators, signer, signer_headers, discovery, discovery_headers = _resolve_discovery_args(args)
        orch_list = discover_orchestrators(
            orchestrators,
            signer_url=signer,
            signer_headers=signer_headers,
            discovery_url=discovery,
            discovery_headers=discovery_headers,
        )

        for orch_url in orch_list:
            try:
                info = get_orch_info(
                    orch_url,
                    signer_url=signer,
                    signer_headers=signer_headers,
                )
            except LivepeerGatewayError as e:
                print_error(orch_url, e)
                continue

            print_info(orch_url, info)

        if args.format == "json":
            print(json.dumps(json_results, indent=2, sort_keys=True))

    except LivepeerGatewayError as e:
        json_results.clear()
        print_error(None, e)
        if args.format == "json":
            print(json.dumps(json_results, indent=2, sort_keys=True))

def format_bytes(num_bytes: int) -> str:
    if num_bytes < 0:
        return f"{num_bytes} B"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(num_bytes)
    unit_idx = 0
    while size >= 1024.0 and unit_idx < len(units) - 1:
        size /= 1024.0
        unit_idx += 1
    if unit_idx == 0:
        return f"{int(size)} {units[unit_idx]}"
    return f"{size:.2f} {units[unit_idx]} ({num_bytes} B)"

if __name__ == "__main__":
    main()
