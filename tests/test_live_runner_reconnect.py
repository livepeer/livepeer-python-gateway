"""Free lifecycle regressions: an orchestrator restart must not strand reservations."""

from __future__ import annotations

import asyncio
from unittest import mock

import pytest

from livepeer_gateway import live_runner
from livepeer_gateway.errors import LivepeerHTTPError
from livepeer_gateway.live_runner import LiveRunnerPriceInfo, LiveRunnerRegistration


class Reader:
    instances = []
    close_started = None
    close_release = None

    def __init__(self, url, *, on_event, **kwargs):
        self.url = url
        self.on_event = on_event
        self.closed = False
        self.task = asyncio.create_task(asyncio.Event().wait())
        self.instances.append(self)

    def callback_task(self):
        return self.task

    async def close(self, **kwargs):
        if self.close_started is not None:
            self.close_started.set()
            await self.close_release.wait()
        self.closed = True
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)


def registration():
    return LiveRunnerRegistration(
        orchestrator_url="http://orchestrator.invalid",
        secret="bootstrap",
        runner_url="http://runner.invalid",
        app="test",
        price_info=LiveRunnerPriceInfo(0),
        unregister_on_close=False,
    )


def response(channel):
    return {
        "runner_id": "runner",
        "heartbeat_secret": "heartbeat",
        "o2r": {
            "name": "o2r",
            "channel_name": channel,
            "mime_type": "application/json",
            "url": "http://orchestrator.invalid/" + channel,
        },
    }


@pytest.fixture(autouse=True)
def readers():
    Reader.instances = []
    Reader.close_started = None
    Reader.close_release = None
    with mock.patch.object(live_runner, "ChannelReader", Reader):
        yield


async def test_401_registration_switches_reader_and_delivers_new_reservation():
    reg = registration()
    events = []
    reg._on_session_reserve = lambda event: events.append(event.session_id)
    with mock.patch.object(
        reg,
        "_post_heartbeat",
        side_effect=[
            response("old"),
            LivepeerHTTPError(401, "heartbeat", "expired"),
            response("new"),
        ],
    ) as post:
        await reg._send_heartbeat()
        old = Reader.instances[0]
        await reg._send_heartbeat()
        new = Reader.instances[1]
        assert old.closed and old.task.done()
        assert new.url.endswith("/new") and not new.task.done()
        assert [call.args[0] for call in post.call_args_list] == [
            "bootstrap",
            "heartbeat",
            "bootstrap",
        ]
        await new.on_event(
            {"event": "reserved", "session": "fresh", "session_token": "token"}
        )
        assert events == ["fresh"] and reg.active_session_ids == ("fresh",)
        await reg.close()
    assert new.closed


@pytest.mark.parametrize("termination", ["eof", "error", "cancel"])
async def test_ended_reader_reregisters_even_when_heartbeat_auth_would_still_work(
    termination,
):
    reg = registration()
    with mock.patch.object(
        reg, "_post_heartbeat", side_effect=[response("old"), response("new")]
    ) as post:
        await reg._send_heartbeat()
        old = Reader.instances[0]
        old.task.cancel()
        await asyncio.gather(old.task, return_exceptions=True)
        if termination != "cancel":

            async def ended():
                if termination == "error":
                    raise RuntimeError("channel failed")

            old.task = asyncio.create_task(ended())
            await asyncio.gather(old.task, return_exceptions=True)
            reg._o2r_task = old.task
        await reg._send_heartbeat()
        assert post.call_args.args == ("bootstrap",)
        assert old.closed and len(Reader.instances) == 2
        await reg.close()


async def test_same_live_channel_and_concurrent_heartbeats_keep_one_reader():
    reg = registration()
    inflight = maximum = 0
    first_entered = asyncio.Event()
    release = asyncio.Event()

    async def heartbeat(auth):
        nonlocal inflight, maximum
        inflight += 1
        maximum = max(maximum, inflight)
        first_entered.set()
        try:
            await release.wait()
            await asyncio.sleep(0)
            return response("same")
        finally:
            inflight -= 1

    with mock.patch.object(reg, "_post_heartbeat", side_effect=heartbeat):
        requests = [asyncio.create_task(reg._send_heartbeat()) for _ in range(8)]
        await asyncio.wait_for(first_entered.wait(), 1)
        await asyncio.sleep(0)
        release.set()
        await asyncio.gather(*requests)
        assert maximum == 1 and inflight == 0
        assert len(Reader.instances) == 1
        await reg.close()
        await reg._send_heartbeat()
        assert len(Reader.instances) == 1


async def test_close_during_replacement_never_starts_new_reader():
    reg = registration()
    with mock.patch.object(
        reg, "_post_heartbeat", side_effect=[response("old"), response("new")]
    ):
        await reg._send_heartbeat()
        Reader.close_started = asyncio.Event()
        Reader.close_release = asyncio.Event()
        update = asyncio.create_task(reg._send_heartbeat())
        await asyncio.wait_for(Reader.close_started.wait(), 1)
        closing = asyncio.create_task(reg.close())
        await asyncio.sleep(0)
        Reader.close_release.set()
        await asyncio.gather(update, closing)
        assert len(Reader.instances) == 1 and Reader.instances[0].closed
        assert reg._o2r_reader is None


async def test_cancelled_replacement_can_recover_on_next_heartbeat():
    reg = registration()
    with mock.patch.object(
        reg,
        "_post_heartbeat",
        side_effect=[response("old"), response("new"), response("new")],
    ) as post:
        await reg._send_heartbeat()
        Reader.close_started = asyncio.Event()
        Reader.close_release = asyncio.Event()
        update = asyncio.create_task(reg._send_heartbeat())
        await asyncio.wait_for(Reader.close_started.wait(), 1)
        update.cancel()
        with pytest.raises(asyncio.CancelledError):
            await update
        Reader.close_release.set()
        await reg._send_heartbeat()
        assert post.call_args.args == ("bootstrap",)
        assert Reader.instances[0].closed
        assert len(Reader.instances) == 2
        await reg.close()


async def test_real_channel_404_ends_reader_and_next_heartbeat_registers_again():
    from aiohttp import web
    from livepeer_gateway.channel_reader import ChannelReader

    requested = []

    async def missing(request):
        requested.append(request.path)
        return web.Response(status=404)

    app = web.Application()
    app.router.add_get("/old/{seq}", missing)
    server = web.AppRunner(app)
    await server.setup()
    site = web.TCPSite(server, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    reg = registration()
    old = response("old")
    old["o2r"]["url"] = f"http://127.0.0.1:{port}/old"
    new = response("new")
    new["o2r"]["url"] = f"http://127.0.0.1:{port}/new"
    try:
        with (
            mock.patch.object(live_runner, "ChannelReader", ChannelReader),
            mock.patch.object(reg, "_post_heartbeat", side_effect=[old, new]) as post,
        ):
            await reg._send_heartbeat()
            reader = reg._o2r_reader
            await asyncio.wait_for(asyncio.shield(reg._o2r_task), 1)
            assert requested == ["/old/0"]
            await reg._send_heartbeat()
            assert post.call_args.args == ("bootstrap",)
            assert reg._o2r_reader is not reader
            assert reg.o2r_channel["url"].endswith("/new")
            await reg.close()
    finally:
        await reg.close()
        await server.cleanup()


async def test_repeated_cancel_and_shutdown_share_transport_cleanup_task():
    import aiohttp
    from livepeer_gateway import channel_reader

    opened = asyncio.Event()
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()
    cleanup_done = asyncio.Event()

    class Subscriber:
        instance = None

        def __init__(self, *args, **kwargs):
            Subscriber.instance = self
            self.session = None
            self.close_calls = 0
            self.cancellations = 0

        async def __aenter__(self):
            self.session = aiohttp.ClientSession()
            opened.set()
            return self

        async def next(self):
            await asyncio.Event().wait()

        async def __aexit__(self, *exc):
            self.close_calls += 1
            cleanup_started.set()
            try:
                # Model a subscriber waiting for its cancelled prefetch task
                # before closing its actual HTTP session.
                await cleanup_release.wait()
                await self.session.close()
                cleanup_done.set()
            except asyncio.CancelledError:
                self.cancellations += 1
                raise

    reg = registration()
    try:
        with (
            mock.patch.object(
                live_runner, "ChannelReader", channel_reader.ChannelReader
            ),
            mock.patch.object(channel_reader, "TrickleSubscriber", Subscriber),
            mock.patch.object(
                reg, "_post_heartbeat", side_effect=[response("old"), response("new")]
            ),
        ):
            await reg._send_heartbeat()
            await asyncio.wait_for(opened.wait(), 1)
            subscriber = Subscriber.instance
            replacement = asyncio.create_task(reg._send_heartbeat())
            await asyncio.wait_for(cleanup_started.wait(), 1)
            replacement.cancel()
            with pytest.raises(asyncio.CancelledError):
                await replacement
            closing = asyncio.create_task(reg.close())
            await asyncio.sleep(0)
            closing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await closing
            retry_close = asyncio.create_task(reg.close())
            await asyncio.sleep(0)
            assert subscriber.close_calls == 1 and subscriber.cancellations == 0
            assert not subscriber.session.closed
            cleanup_release.set()
            await asyncio.wait_for(retry_close, 1)
            assert cleanup_done.is_set() and subscriber.session.closed
            assert subscriber.close_calls == 1 and subscriber.cancellations == 0
            assert reg._o2r_reader is None
    finally:
        cleanup_release.set()
        await reg.close()
        if Subscriber.instance is not None and Subscriber.instance.session is not None:
            await Subscriber.instance.session.close()


@pytest.mark.parametrize("ending", ["eof", "error"])
async def test_natural_subscriber_cleanup_finishes_before_replacement_attaches(ending):
    from livepeer_gateway import channel_reader
    from livepeer_gateway.trickle_subscriber import TrickleSubscriber

    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()
    subscribers = []

    class Subscriber(TrickleSubscriber):
        async def __aenter__(self):
            await self._ensure_runtime()
            self.owned_session = self._session
            real_close = self.owned_session.close

            async def slow_close():
                cleanup_started.set()
                await cleanup_release.wait()
                await real_close()

            self.owned_session.close = slow_close
            subscribers.append(self)
            return self

        async def next(self):
            if ending == "error":
                raise RuntimeError("subscription ended")
            return None  # Real ChannelReader enters natural EOF cleanup.

    reg = registration()
    try:
        with (
            mock.patch.object(
                live_runner, "ChannelReader", channel_reader.ChannelReader
            ),
            mock.patch.object(channel_reader, "TrickleSubscriber", Subscriber),
            mock.patch.object(
                reg, "_post_heartbeat", side_effect=[response("old"), response("new")]
            ),
        ):
            await reg._send_heartbeat()
            await asyncio.wait_for(cleanup_started.wait(), 1)
            old_reader = reg._o2r_reader
            replacement = asyncio.create_task(reg._send_heartbeat())
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not replacement.done()
            assert (
                reg._o2r_reader is old_reader
                and not subscribers[0].owned_session.closed
            )
            cleanup_release.set()
            await asyncio.wait_for(replacement, 1)
            assert subscribers[0].owned_session.closed
            assert reg._o2r_reader is not old_reader
            await reg.close()
    finally:
        cleanup_release.set()
        await reg.close()
        for subscriber in subscribers:
            await subscriber.owned_session.close()


async def test_subscriber_cancel_grace_is_bounded_and_cleanup_remains_owned():
    from livepeer_gateway import trickle_subscriber

    started = asyncio.Event()
    release = asyncio.Event()

    class Subscriber(trickle_subscriber.TrickleSubscriber):
        async def _close_transport(self):
            started.set()
            await release.wait()
            await super()._close_transport()

    subscriber = Subscriber("http://unused.invalid")
    await subscriber._ensure_runtime()
    session = subscriber._session
    try:
        with mock.patch.object(trickle_subscriber, "_CLOSE_CANCEL_GRACE_S", 0.01):
            closing = asyncio.create_task(subscriber.close())
            await asyncio.wait_for(started.wait(), 1)
            closing.cancel()
            await asyncio.sleep(0)
            closing.cancel()
            done, _ = await asyncio.wait([closing], timeout=1)
            assert closing in done
            with pytest.raises(asyncio.CancelledError):
                await closing
            assert not session.closed and not subscriber._close_task.done()
            release.set()
            await asyncio.wait_for(subscriber.close(), 1)
            assert session.closed
    finally:
        release.set()
        await subscriber.close()
        await session.close()


async def test_session_callback_can_close_registration_without_self_await():
    class CallbackReader(Reader):
        def __init__(self, url, **kwargs):
            super().__init__(url, **kwargs)
            self.task.cancel()
            self.task = asyncio.create_task(
                self.on_event({"event": "reserved", "session": "session"})
            )

    reg = registration()

    async def on_reserve(event):
        await reg.close()

    reg._on_session_reserve = on_reserve
    with (
        mock.patch.object(live_runner, "ChannelReader", CallbackReader),
        mock.patch.object(reg, "_post_heartbeat", return_value=response("channel")),
    ):
        await reg._send_heartbeat()
        reader = reg._o2r_reader
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(reader.task), 1)
        await asyncio.wait_for(reg.close(), 1)
        assert reg._closed and reg._o2r_reader is None and reader.closed
