from __future__ import annotations

from unittest import mock
from urllib.parse import parse_qs, urlparse

import pytest

from livepeer_gateway import discovery
from livepeer_gateway.remote_signer import RemoteSignerError


class TestRunnerDiscoveryQuery:
    def test_append_runner_filters_preserves_queries_and_ignores_empty_values(
        self,
    ) -> None:
        url = discovery._append_runner_filters(
            "https://example.com/discovery?x=1",
            app=["live-video-to-video/scope", "echo"],
            gpu=["H100", "NVIDIA L40S"],
        )

        parsed = urlparse(url)
        assert parsed.scheme == "https"
        assert parsed.netloc == "example.com"
        assert parsed.path == "/discovery"
        assert parse_qs(parsed.query) == {
            "x": ["1"],
            "app": ["live-video-to-video/scope", "echo"],
            "gpu": ["H100", "NVIDIA L40S"],
        }
        filtered_url = discovery._append_runner_filters(
            "https://example.com/discovery",
            app=["", "  ", "echo"],
            gpu="",
        )
        assert parse_qs(urlparse(filtered_url).query) == {"app": ["echo"]}


class TestRunnerDiscovery:
    def test_orchestrator_discovery_urls_preserves_paths(self) -> None:
        assert discovery.orchestrator_discovery_urls(
            "https://orch-a.example.com/base/, https://orch-b.example.com"
        ) == [
            "https://orch-a.example.com/base/discovery",
            "https://orch-b.example.com/discovery",
        ]

    async def test_discover_runners_uses_signer_origin_and_appends_filters(
        self,
    ) -> None:
        calls: list[tuple[str, dict[str, str] | None]] = []

        async def _get_json(
            url: str, *, headers: dict[str, str] | None = None
        ) -> list[dict[str, object]]:
            calls.append((url, headers))
            return [
                {
                    "address": "https://orch.example.com",
                    "runners": [
                        {
                            "url": "https://orch.example.com/apps/a/session",
                            "app": "live-video-to-video/scope",
                            "gpu": {"name": "H100"},
                        }
                    ],
                }
            ]

        with mock.patch.object(discovery, "get_json", side_effect=_get_json):
            result = await discovery.discover_runners(
                signer_url="https://signer.example.com/base",
                signer_headers={"Authorization": "token"},
                app="live-video-to-video/scope",
                gpu="H100",
            )

        assert len(result) == 1
        assert calls[0][1] == {"Authorization": "token"}
        parsed = urlparse(calls[0][0])
        assert (
            f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            == "https://signer.example.com/discover-orchestrators"
        )
        assert parse_qs(parsed.query) == {
            "app": ["live-video-to-video/scope"],
            "gpu": ["H100"],
        }

    async def test_discover_runners_response_must_be_list(self) -> None:
        with mock.patch.object(discovery, "get_json", return_value={"runners": []}):
            with pytest.raises(RemoteSignerError):
                await discovery.discover_runners(
                    discovery_url="https://example.com/discovery"
                )

    async def test_discover_runners_skips_malformed_entries_and_runners(self) -> None:
        with mock.patch.object(
            discovery,
            "get_json",
            return_value=[
                "bad",
                {"address": "https://orch-a.example.com", "runners": "bad"},
                {
                    "address": "https://orch-b.example.com",
                    "runners": [
                        "bad",
                        {"url": "", "app": "echo"},
                        {"url": "https://runner.example.com/session", "app": "echo"},
                    ],
                },
            ],
        ):
            result = await discovery.discover_runners(
                discovery_url="https://example.com/discovery"
            )

        assert result == [
            {
                "address": "https://orch-b.example.com",
                "runners": [
                    {"url": "https://runner.example.com/session", "app": "echo"}
                ],
            }
        ]

    @pytest.mark.parametrize(
        ("filters", "expected"),
        [
            (
                {"app": ["app-a", "app-b"]},
                [
                    ("app-a", "H100"),
                    ("app-b", "NVIDIA L40S"),
                    ("app-a", "A10"),
                ],
            ),
            (
                {"gpu": ["H100", "NVIDIA L40S"]},
                [
                    ("app-a", "H100"),
                    ("app-b", "NVIDIA L40S"),
                    ("app-c", "H100"),
                ],
            ),
            (
                {
                    "app": ["app-a", "app-b"],
                    "gpu": ["H100", "NVIDIA L40S"],
                },
                [("app-a", "H100"), ("app-b", "NVIDIA L40S")],
            ),
        ],
        ids=["app", "gpu", "app-and-gpu"],
    )
    async def test_discover_runners_filters_with_or_and_across_dimensions(
        self,
        filters: dict[str, list[str]],
        expected: list[tuple[str, str]],
    ) -> None:
        with mock.patch.object(
            discovery,
            "get_json",
            return_value=[_entry()],
        ):
            result = await discovery.discover_runners(
                discovery_url="https://example.com/discovery",
                **filters,
            )

        assert [
            (runner["app"], runner["gpu"]["name"]) for runner in result[0]["runners"]
        ] == expected


def _entry() -> dict[str, object]:
    return {
        "address": "https://orch.example.com",
        "runners": [
            {
                "url": "https://orch.example.com/apps/a/session",
                "app": "app-a",
                "gpu": {"name": "H100"},
            },
            {
                "url": "https://orch.example.com/apps/b/session",
                "app": "app-b",
                "gpu": {"name": "NVIDIA L40S"},
            },
            {
                "url": "https://orch.example.com/apps/c/session",
                "app": "app-c",
                "gpu": {"name": "H100"},
            },
            {
                "url": "https://orch.example.com/apps/d/session",
                "app": "app-a",
                "gpu": {"name": "A10"},
            },
        ],
    }
