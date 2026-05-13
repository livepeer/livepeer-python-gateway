from __future__ import annotations

import logging
from typing import Any, Optional, Sequence
from urllib.parse import parse_qsl, quote, urlencode, urlparse, urlunparse

from . import lp_rpc_pb2
from .capabilities import capabilities_to_query
from .errors import LivepeerGatewayError
from .remote_signer import RemoteSignerError
from .http import _http_origin, _parse_http_url, get_json_sync

_LOG = logging.getLogger(__name__)

FilterValue = str | Sequence[str]


def _normalize_filter_values(value: Optional[FilterValue]) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    else:
        values = list(value)
    return [item.strip() for item in values if isinstance(item, str) and item.strip()]


def _append_query_values(url: str, values: Sequence[tuple[str, str]]) -> str:
    if not values:
        return url

    parsed = urlparse(url)
    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    query_pairs.extend(values)
    query = urlencode(query_pairs, doseq=True, quote_via=quote, safe="/")
    return urlunparse(parsed._replace(query=query))


def _append_caps(url: str, capabilities: Optional[lp_rpc_pb2.Capabilities]) -> str:
    """
    Append repeated `caps` query parameters to a URL.

    Existing query params are preserved. Capability values keep `/` unescaped.
    """
    if capabilities is None:
        return url
    return _append_query_values(url, [("caps", cap) for cap in capabilities_to_query(capabilities)])


def _append_runner_filters(
    url: str,
    *,
    app: Optional[FilterValue] = None,
    gpu: Optional[FilterValue] = None,
) -> str:
    values: list[tuple[str, str]] = []
    values.extend(("app", item) for item in _normalize_filter_values(app))
    values.extend(("gpu", item) for item in _normalize_filter_values(gpu))
    return _append_query_values(url, values)


def discover_orchestrators(
    orchestrators: Optional[Sequence[str] | str] = None,
    *,
    signer_url: Optional[str] = None,
    signer_headers: Optional[dict[str, str]] = None,
    discovery_url: Optional[str] = None,
    discovery_headers: Optional[dict[str, str]] = None,
    capabilities: Optional[lp_rpc_pb2.Capabilities] = None,
) -> list[str]:
    """
    Discover orchestrators and return a list of addresses.

    This discovery can happen via the following parameters in priority order (highest first):
    - orchestrators: list or comma-delimited string
      (empty/whitespace-only input falls through)
    - discovery_url: use this discovery endpoint
    - signer_url: use signer-provided discovery service
    """
    if orchestrators is not None:
        if isinstance(orchestrators, str):
            orch_list = [orch.strip() for orch in orchestrators.split(",")]
        else:
            try:
                orch_list = list(orchestrators)
            except TypeError as e:
                raise LivepeerGatewayError(
                    "discover_orchestrators requires a list of orchestrator URLs or a comma-delimited string"
                ) from e
        orch_list = [orch.strip() for orch in orch_list if isinstance(orch, str) and orch.strip()]
        if orch_list:
            return orch_list

    if discovery_url:
        discovery_endpoint = _parse_http_url(discovery_url).geturl()
        request_headers = discovery_headers
    elif signer_url:
        discovery_endpoint = f"{_http_origin(signer_url)}/discover-orchestrators"
        request_headers = signer_headers
    else:
        _LOG.debug("discover_orchestrators failed: no discovery inputs")
        raise LivepeerGatewayError("discover_orchestrators requires discovery_url or signer_url")

    if capabilities is not None:
        discovery_endpoint = _append_caps(discovery_endpoint, capabilities)

    try:
        _LOG.debug("discover_orchestrators running discovery: %s", discovery_endpoint)
        data = get_json_sync(discovery_endpoint, headers=request_headers)
    except LivepeerGatewayError as e:
        _LOG.debug("discover_orchestrators discovery failed: %s", e)
        raise RemoteSignerError(
            discovery_endpoint,
            str(e),
            cause=e.__cause__ or e,
        ) from None

    if not isinstance(data, list):
        _LOG.debug(
            "discover_orchestrators discovery response not list: type=%s",
            type(data).__name__,
        )
        raise RemoteSignerError(
            discovery_endpoint,
            f"Discovery response must be a JSON list, got {type(data).__name__}",
            cause=None,
        ) from None

    _LOG.debug("discover_orchestrators discovery response: %s", data)

    orch_list = []
    for item in data:
        if not isinstance(item, dict):
            continue
        address = item.get("address")
        if isinstance(address, str) and address.strip():
            orch_list.append(address.strip())
    _LOG.debug("discover_orchestrators discovered %d orchestrators", len(orch_list))

    return orch_list


def discover_runners(
    *,
    signer_url: Optional[str] = None,
    signer_headers: Optional[dict[str, str]] = None,
    discovery_url: Optional[str] = None,
    discovery_headers: Optional[dict[str, str]] = None,
    app: Optional[FilterValue] = None,
    gpu: Optional[FilterValue] = None,
) -> list[dict[str, Any]]:
    """
    Discover live runners and return discovery entries.

    Filters are composed as OR within each field and AND across fields.
    For example, app=["a", "b"], gpu=["H100", "L40S"] matches
    (app=a OR app=b) AND (gpu=H100 OR gpu=L40S).
    """
    if discovery_url:
        discovery_endpoint = _parse_http_url(discovery_url).geturl()
        request_headers = discovery_headers
    elif signer_url:
        discovery_endpoint = f"{_http_origin(signer_url)}/discovery"
        request_headers = signer_headers
    else:
        _LOG.debug("discover_runners failed: no discovery inputs")
        raise LivepeerGatewayError("discover_runners requires discovery_url or signer_url")

    app_filters = _normalize_filter_values(app)
    gpu_filters = _normalize_filter_values(gpu)
    discovery_endpoint = _append_runner_filters(discovery_endpoint, app=app_filters, gpu=gpu_filters)

    try:
        _LOG.debug("discover_runners running discovery: %s", discovery_endpoint)
        data = get_json_sync(discovery_endpoint, headers=request_headers)
    except LivepeerGatewayError as e:
        _LOG.debug("discover_runners discovery failed: %s", e)
        raise RemoteSignerError(
            discovery_endpoint,
            str(e),
            cause=e.__cause__ or e,
        ) from None

    if not isinstance(data, list):
        _LOG.debug(
            "discover_runners discovery response not list: type=%s",
            type(data).__name__,
        )
        raise RemoteSignerError(
            discovery_endpoint,
            f"Discovery response must be a JSON list, got {type(data).__name__}",
            cause=None,
        ) from None

    entries = _filter_runner_discovery_entries(data, app_filters=app_filters, gpu_filters=gpu_filters)
    _LOG.debug("discover_runners discovered %d orchestrator entries", len(entries))
    return entries


def _filter_runner_discovery_entries(
    data: Sequence[Any],
    *,
    app_filters: Sequence[str],
    gpu_filters: Sequence[str],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        runners = item.get("runners")
        if not isinstance(runners, list):
            continue

        matched_runners = []
        for runner in runners:
            if not isinstance(runner, dict):
                continue
            if not _valid_runner(runner):
                continue
            if not _runner_matches_filters(runner, app_filters=app_filters, gpu_filters=gpu_filters):
                continue
            matched_runners.append(runner)

        if matched_runners:
            entry = dict(item)
            entry["runners"] = matched_runners
            entries.append(entry)
    return entries


def _valid_runner(runner: dict[str, Any]) -> bool:
    url = runner.get("url")
    app = runner.get("app")
    return isinstance(url, str) and bool(url.strip()) and isinstance(app, str) and bool(app.strip())


def _runner_matches_filters(
    runner: dict[str, Any],
    *,
    app_filters: Sequence[str],
    gpu_filters: Sequence[str],
) -> bool:
    app = runner.get("app")
    app_value = app.strip() if isinstance(app, str) else ""
    if app_filters and app_value not in app_filters:
        return False
    if gpu_filters and _runner_gpu_name(runner) not in gpu_filters:
        return False
    return True


def _runner_gpu_name(runner: dict[str, Any]) -> str:
    gpu = runner.get("gpu")
    if isinstance(gpu, dict):
        name = gpu.get("name")
        if isinstance(name, str):
            return name.strip()
    return ""
