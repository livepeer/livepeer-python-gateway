from __future__ import annotations

import asyncio
import importlib
from unittest import mock

import pytest

channel_reader_mod = importlib.import_module("livepeer_gateway.channel_reader")

ChannelReader = channel_reader_mod.ChannelReader
JSONLReader = channel_reader_mod.JSONLReader


class _FakeReader:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)

    async def read(self, chunk_size: int = 32 * 1024):
        if not self._chunks:
            return b""
        chunk = self._chunks.pop(0)
        return chunk[:chunk_size]


class _FakeSegment:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.closed = False

    def make_reader(self) -> _FakeReader:
        return _FakeReader(self._chunks)

    async def close(self) -> None:
        self.closed = True


class _FakeSubscriber:
    instances: list[_FakeSubscriber] = []
    segments: list[_FakeSegment] = []
    init_kwargs: dict[str, object] = {}

    def __init__(self, url: str, **kwargs: object) -> None:
        self.url = url
        self.kwargs = kwargs
        self._segments = list(type(self).segments)
        self.closed = False
        type(self).init_kwargs = kwargs
        type(self).instances.append(self)

    async def __aenter__(self) -> _FakeSubscriber:
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        self.closed = True

    async def next(self):
        if not self._segments:
            return None
        return self._segments.pop(0)


class TestChannelReaderSync:
    @pytest.fixture(autouse=True)
    def reset_fake_subscriber(self) -> None:
        _FakeSubscriber.instances = []
        _FakeSubscriber.segments = []
        _FakeSubscriber.init_kwargs = {}

    def test_callbacks_can_start_later_from_async_context(self) -> None:
        seen: list[dict[str, object]] = []
        reader = ChannelReader("http://example.test/events", on_event=seen.append)

        assert reader.callback_task() is None

        async def _run() -> None:
            _FakeSubscriber.segments = [_FakeSegment([b'{"ok": true}'])]
            with mock.patch.object(
                channel_reader_mod, "TrickleSubscriber", _FakeSubscriber
            ):
                async with reader:
                    await reader.wait_callback(timeout=1.0)

        asyncio.run(_run())

        assert seen == [{"ok": True}]


class TestChannelReader:
    @pytest.fixture(autouse=True)
    def reset_fake_subscriber(self) -> None:
        _FakeSubscriber.instances = []
        _FakeSubscriber.segments = []
        _FakeSubscriber.init_kwargs = {}

    @pytest.mark.parametrize(
        ("reader_type", "segments"),
        [
            (
                ChannelReader,
                [
                    _FakeSegment([b'{"one": 1}']),
                    _FakeSegment([b'{"two": 2}']),
                ],
            ),
            (
                JSONLReader,
                [_FakeSegment([b'{"one": 1}\n{"two":', b" 2}\n"])],
            ),
        ],
        ids=["channel-reader", "jsonl-reader"],
    )
    async def test_readers_iterate_json_objects(
        self, reader_type, segments: list[_FakeSegment]
    ) -> None:
        _FakeSubscriber.segments = segments
        with mock.patch.object(
            channel_reader_mod,
            "TrickleSubscriber",
            _FakeSubscriber,
        ):
            reader = reader_type("http://example.test/events")
            events = [event async for event in reader()]

        assert events == [{"one": 1}, {"two": 2}]

    @pytest.mark.parametrize(
        ("reader_type", "segments", "expected"),
        [
            (
                ChannelReader,
                [_FakeSegment([b'{"ok": true}'])],
                [{"ok": True}],
            ),
            (
                JSONLReader,
                [_FakeSegment([b'{"one": 1}\n{"two": 2}\n'])],
                [{"one": 1}, {"two": 2}],
            ),
        ],
        ids=["channel-reader", "jsonl-reader"],
    )
    async def test_readers_start_background_callback_consumers(
        self,
        reader_type,
        segments: list[_FakeSegment],
        expected: list[dict[str, object]],
    ) -> None:
        _FakeSubscriber.segments = segments
        seen: list[dict[str, object]] = []
        with mock.patch.object(
            channel_reader_mod,
            "TrickleSubscriber",
            _FakeSubscriber,
        ):
            reader = reader_type(
                "http://example.test/events",
                on_event=seen.append,
            )
            await reader.wait_callback(timeout=1.0)

        assert seen == expected
        assert reader.callback_task() is not None

    async def test_async_event_callback_is_awaited(self) -> None:
        _FakeSubscriber.segments = [_FakeSegment([b'{"ok": true}'])]
        order: list[str] = []

        async def _on_event(_event: dict[str, object]) -> None:
            order.append("start")
            await asyncio.sleep(0)
            order.append("done")

        with mock.patch.object(
            channel_reader_mod, "TrickleSubscriber", _FakeSubscriber
        ):
            reader = ChannelReader("http://example.test/events", on_event=_on_event)
            await reader.wait_callback(timeout=1.0)

        assert order == ["start", "done"]

    async def test_callback_uses_constructor_read_options(self) -> None:
        _FakeSubscriber.segments = [_FakeSegment([b'{"ok": true}'])]
        seen: list[dict[str, object]] = []
        reader = ChannelReader(
            "http://example.test/events",
            start_seq=7,
            max_retries=2,
            max_event_bytes=123,
            on_event=None,
        )
        reader.on_event = seen.append

        with mock.patch.object(
            channel_reader_mod, "TrickleSubscriber", _FakeSubscriber
        ):
            reader.start_callback()
            await reader.wait_callback(timeout=1.0)

        assert seen == [{"ok": True}]
        assert _FakeSubscriber.init_kwargs == {
            "start_seq": 7,
            "max_retries": 2,
            "max_bytes": 123,
        }

    async def test_callback_exception_raises_from_wait_and_close(self) -> None:
        _FakeSubscriber.segments = [_FakeSegment([b'{"ok": true}'])]

        def _on_event(_event: dict[str, object]) -> None:
            raise RuntimeError("event callback boom")

        with mock.patch.object(
            channel_reader_mod, "TrickleSubscriber", _FakeSubscriber
        ):
            reader = JSONLReader("http://example.test/events", on_event=_on_event)
            with pytest.raises(RuntimeError, match="event callback boom"):
                await reader.wait_callback(timeout=1.0)
            with pytest.raises(RuntimeError, match="event callback boom"):
                await reader.close()

    async def test_wait_callback_returns_none_without_callback(self) -> None:
        reader = ChannelReader("http://example.test/events")
        assert reader.callback_task() is None
        assert await reader.wait_callback(timeout=1.0) is None
