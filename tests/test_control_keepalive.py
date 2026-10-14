from __future__ import annotations

import asyncio
from unittest import mock

from livepeer_gateway import control as control_mod
from livepeer_gateway.trickle_publisher import (
    TricklePublisherTerminalError,
    TrickleSegmentWriteError,
)


class _FakePublisher:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _ControlledSleep:
    def __init__(self) -> None:
        self.started: asyncio.Queue[float] = asyncio.Queue()
        self.permits: asyncio.Queue[None] = asyncio.Queue()

    async def __call__(self, delay: float) -> None:
        await self.started.put(delay)
        await self.permits.get()


class TestControlKeepalive:
    async def test_start_keepalive_sends_periodic_message(self) -> None:
        publisher = _FakePublisher()
        with mock.patch.object(
            control_mod,
            "TricklePublisher",
            return_value=publisher,
        ):
            control = control_mod.Control("http://example.test/control")

        write_called = asyncio.Event()

        async def _write(_msg: dict[str, str]) -> None:
            write_called.set()

        control.write = mock.AsyncMock(side_effect=_write)  # type: ignore[method-assign]
        sleep = _ControlledSleep()
        with mock.patch.object(control_mod.asyncio, "sleep", new=sleep):
            task = control.start_keepalive()
            assert task is not None
            assert await asyncio.wait_for(sleep.started.get(), 1.0) == 10.0
            sleep.permits.put_nowait(None)
            await asyncio.wait_for(write_called.wait(), 1.0)
            await control.close()

        assert control.write.await_count >= 1
        control.write.assert_any_await({"keep": "alive"})
        assert publisher.closed

    async def test_keepalive_retries_after_segment_write_error(self) -> None:
        publisher = _FakePublisher()
        with mock.patch.object(
            control_mod,
            "TricklePublisher",
            return_value=publisher,
        ):
            control = control_mod.Control("http://example.test/control")

        failure = TrickleSegmentWriteError("boom", seq=1, status=500)
        call_state = {"count": 0}
        write_succeeded = asyncio.Event()

        async def _write(_msg: dict[str, str]) -> None:
            call_state["count"] += 1
            if call_state["count"] == 1:
                raise failure
            write_succeeded.set()

        control.write = mock.AsyncMock(side_effect=_write)  # type: ignore[method-assign]
        sleep = _ControlledSleep()
        with mock.patch.object(control_mod.asyncio, "sleep", new=sleep):
            task = control.start_keepalive()
            assert task is not None
            await asyncio.wait_for(sleep.started.get(), 1.0)
            sleep.permits.put_nowait(None)
            await asyncio.wait_for(sleep.started.get(), 1.0)
            sleep.permits.put_nowait(None)
            await asyncio.wait_for(write_succeeded.wait(), 1.0)
            await control.close()

        assert control.write.await_count >= 2
        assert publisher.closed

    async def test_keepalive_stops_on_terminal_error(self) -> None:
        publisher = _FakePublisher()
        with mock.patch.object(
            control_mod,
            "TricklePublisher",
            return_value=publisher,
        ):
            control = control_mod.Control("http://example.test/control")

        terminal = TricklePublisherTerminalError("terminal", consecutive_failures=3)
        control.write = mock.AsyncMock(side_effect=terminal)  # type: ignore[method-assign]
        sleep = _ControlledSleep()
        with mock.patch.object(control_mod.asyncio, "sleep", new=sleep):
            task = control.start_keepalive()
            assert task is not None
            await asyncio.wait_for(sleep.started.get(), 1.0)
            sleep.permits.put_nowait(None)
            await asyncio.wait_for(task, 1.0)
            await control.close()

        assert task.done()
        assert publisher.closed
