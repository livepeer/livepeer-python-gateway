from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest import mock

import pytest

from livepeer_gateway import control as control_mod
from livepeer_gateway import lv2v as lv2v_mod
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


class _Cursor:
    def next(self):
        return ("http://orch", SimpleNamespace(transcoder="http://orch"))


class _Payment:
    payment = "payment-token"
    seg_creds = "segment-token"


class _PaymentSession:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.manifest_id: str | None = None

    def get_payment(self) -> _Payment:
        return _Payment()

    def set_manifest_id(self, manifest_id: str) -> None:
        self.manifest_id = manifest_id


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


class TestStartLv2VKeepaliveWiring:
    def _start_with_job(
        self,
        *,
        control: object,
        start_payments: bool = True,
    ) -> tuple[object, object]:
        job = SimpleNamespace(
            manifest_id="manifest",
            control=control,
            start_payment_sender=mock.Mock(),
        )
        with (
            mock.patch.object(
                lv2v_mod,
                "build_capabilities",
                return_value=object(),
            ),
            mock.patch.object(
                lv2v_mod,
                "orchestrator_selector",
                return_value=_Cursor(),
            ),
            mock.patch.object(lv2v_mod, "PaymentSession", _PaymentSession),
            mock.patch.object(
                lv2v_mod,
                "post_json_sync",
                return_value={"manifest_id": "manifest"},
            ),
            mock.patch.object(
                lv2v_mod.LiveVideoToVideo,
                "from_json",
                return_value=job,
            ),
        ):
            result = lv2v_mod.start_lv2v(
                "http://orch",
                lv2v_mod.StartJobRequest(model_id="noop"),
                start_payments=start_payments,
            )
        return result, job

    def test_start_lv2v_starts_control_keepalive_when_control_present(self) -> None:
        control = mock.Mock()
        result, job = self._start_with_job(control=control)

        assert result is job
        job.start_payment_sender.assert_called_once_with()
        control.start_keepalive.assert_called_once_with()

    def test_start_lv2v_can_skip_payment_loop_start(self) -> None:
        control = mock.Mock()
        result, job = self._start_with_job(
            control=control,
            start_payments=False,
        )

        assert result is job
        job.start_payment_sender.assert_not_called()
        control.start_keepalive.assert_called_once_with()

    def test_start_lv2v_skips_keepalive_when_control_missing(self) -> None:
        result, job = self._start_with_job(control=None)

        assert result is job
        job.start_payment_sender.assert_called_once_with()

    @pytest.mark.parametrize(
        "mode",
        [control_mod.ControlMode.DISABLED, control_mod.ControlMode.TIME],
    )
    def test_start_lv2v_skips_keepalive_for_disabled_and_time_modes(
        self, mode: control_mod.ControlMode
    ) -> None:
        with (
            mock.patch.object(
                control_mod.Control,
                "start_keepalive",
                autospec=True,
            ) as start_keepalive,
            mock.patch.object(
                lv2v_mod,
                "build_capabilities",
                return_value=object(),
            ),
            mock.patch.object(
                lv2v_mod,
                "orchestrator_selector",
                return_value=_Cursor(),
            ),
            mock.patch.object(
                lv2v_mod,
                "PaymentSession",
                _PaymentSession,
            ),
            mock.patch.object(
                lv2v_mod,
                "post_json_sync",
                return_value={
                    "manifest_id": "manifest",
                    "control_url": "http://orch/ai/trickle/manifest-control",
                },
            ),
        ):
            result = lv2v_mod.start_lv2v(
                "http://orch",
                lv2v_mod.StartJobRequest(model_id="noop"),
                control_config=control_mod.ControlConfig(mode=mode),
            )

        assert result.control is not None
        start_keepalive.assert_not_called()
