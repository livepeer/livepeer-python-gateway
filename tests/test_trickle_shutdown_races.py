from __future__ import annotations

import asyncio
import importlib
from unittest import mock

import pytest

trickle_publisher_mod = importlib.import_module("livepeer_gateway.trickle_publisher")
trickle_subscriber_mod = importlib.import_module("livepeer_gateway.trickle_subscriber")

TricklePublisher = trickle_publisher_mod.TricklePublisher
TrickleSubscriber = trickle_subscriber_mod.TrickleSubscriber


class _FakeContent:
    async def read(self, _size: int) -> bytes:
        return b""


class _FakeResponse:
    def __init__(
        self, status: int = 200, headers: dict[str, str] | None = None
    ) -> None:
        self.status = status
        self.headers = headers or {}
        self.closed = False
        self.content = _FakeContent()

    async def text(self) -> str:
        return ""

    def release(self) -> None:
        self.closed = True

    def close(self) -> None:
        self.closed = True


class _PublisherSessionFactory:
    def __init__(self, post_gate: asyncio.Event) -> None:
        self.post_gate = post_gate
        self.created = 0
        self.sessions: list[object] = []

    def build(self, *args, **kwargs):
        self.created += 1
        gate = self.post_gate

        class _Session:
            def __init__(self) -> None:
                self.closed = False

            async def post(self, url: str, **_kwargs):
                if not url.endswith("/next"):
                    await gate.wait()
                return _FakeResponse(200, {"Lp-Trickle-Latest": "0"})

            async def get(self, _url: str):
                return _FakeResponse(200, {"Lp-Trickle-Latest": "0"})

            async def delete(self, _url: str):
                return _FakeResponse(200, {})

            async def close(self) -> None:
                self.closed = True

        session = _Session()
        self.sessions.append(session)
        return session


class _SubscriberSession:
    def __init__(self, get_gate: asyncio.Event) -> None:
        self.get_gate = get_gate
        self.prefetch_started = asyncio.Event()
        self.prefetch_cancelled = asyncio.Event()
        self.closed = False
        self.get_calls = 0
        self.late_response = _FakeResponse(200, {"Lp-Trickle-Seq": "1"})

    async def get(self, _url: str, **_kwargs):
        self.get_calls += 1
        if self.get_calls == 1:
            return _FakeResponse(200, {"Lp-Trickle-Seq": "0"})

        self.prefetch_started.set()
        try:
            await self.get_gate.wait()
        except asyncio.CancelledError:
            # Simulate an HTTP client that still hands back a response after the
            # caller begins shutdown.
            self.prefetch_cancelled.set()
            await self.get_gate.wait()
        return self.late_response

    async def close(self) -> None:
        self.closed = True


class TestTrickleShutdownRace:
    async def test_publisher_close_is_terminal_and_does_not_reopen_session(
        self,
    ) -> None:
        post_gate = asyncio.Event()
        session_factory = _PublisherSessionFactory(post_gate)
        with mock.patch.object(
            trickle_publisher_mod.aiohttp,
            "ClientSession",
            side_effect=session_factory.build,
        ):
            publisher = TricklePublisher(
                "http://example.test/trickle", "video/mp2t", start_seq=0
            )
            segment = await publisher.next()
            created_before_close = session_factory.created
            await publisher.close()
            post_gate.set()
            await asyncio.sleep(0)
            assert session_factory.created == created_before_close
            await segment.close()

    async def test_subscriber_close_prevents_pending_get_repopulation(self) -> None:
        get_gate = asyncio.Event()
        session = _SubscriberSession(get_gate)
        subscriber = TrickleSubscriber(
            "http://example.test/trickle", start_seq=0, max_retries=1
        )
        with mock.patch.object(
            trickle_subscriber_mod.aiohttp,
            "ClientSession",
            return_value=session,
        ):
            segment = await subscriber.next()
            assert segment is not None
            await segment.close()
            await asyncio.wait_for(session.prefetch_started.wait(), timeout=1.0)

            close_task = asyncio.create_task(subscriber.close())
            await asyncio.wait_for(session.prefetch_cancelled.wait(), timeout=1.0)
            get_gate.set()
            await asyncio.wait_for(close_task, timeout=1.0)

        assert session.late_response.closed
        assert session.closed
        assert await subscriber.next() is None

    async def test_publisher_close_does_not_set_terminal_error_state(self) -> None:
        publisher = TricklePublisher("http://example.test/trickle", "video/mp2t")
        await publisher.close()
        stats = publisher.get_stats()
        assert not stats.terminal_error
        assert stats.terminal_failures == 0
        with pytest.raises(RuntimeError, match="closed|closing"):
            await publisher.next()

    async def test_close_is_idempotent(self) -> None:
        publisher = TricklePublisher("http://example.test/trickle", "video/mp2t")
        await publisher.close()
        await publisher.close()

        subscriber = TrickleSubscriber("http://example.test/trickle")
        await subscriber.close()
        await subscriber.close()
