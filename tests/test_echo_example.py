from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from examples.echo import client
from livepeer_gateway.media_output import MediaOutput


@pytest.fixture
async def media_run(monkeypatch):
    names = ("publish", "display", "blur", "in", "out")
    release = {name: asyncio.Event() for name in names}
    started = {name: asyncio.Event() for name in names}
    stopped = {name: asyncio.Event() for name in names}
    tasks = {}
    resources = []
    received = []
    run = SimpleNamespace(
        release=release,
        stopped=stopped,
        tasks=tasks,
        resources=resources,
        received=received,
        consumer_error=None,
    )

    async def wait(name):
        tasks[name] = asyncio.current_task()
        started[name].set()
        try:
            await release[name].wait()
        finally:
            stopped[name].set()

    def make_output(url, **kwargs):
        async def frames():
            await wait(url)
            yield SimpleNamespace(kind="video", frame=url, pts_time=1.0)

        stream = MediaOutput(url, **kwargs)
        stream.frames = frames
        segment = SimpleNamespace(close=AsyncMock())
        subscriber = SimpleNamespace(close=AsyncMock(), get_stats=lambda: None)
        stream._segments = [segment]
        stream._sub = subscriber
        resources.extend([segment, subscriber])
        return stream

    def update(name, frame, pts_time):
        if name == "output" and run.consumer_error is not None:
            raise run.consumer_error
        received.append(name)

    async def publish(*args, **kwargs):
        await wait("publish")

    async def blur(*args):
        await wait("blur")

    monkeypatch.setattr(client, "MediaOutput", make_output)
    monkeypatch.setattr(client, "_publish_video", publish)
    monkeypatch.setattr(client, "_sweep_blur", blur)
    viewer = SimpleNamespace(run=lambda: wait("display"), update=update)
    run.task = asyncio.create_task(
        client._run_media(
            SimpleNamespace(blur=True), "in", "out", "app", viewer, None, False
        )
    )
    try:
        async with asyncio.timeout(5):
            await asyncio.gather(*(event.wait() for event in started.values()))
        yield run
    finally:
        for event in release.values():
            event.set()
        run.task.cancel()
        await asyncio.gather(run.task, return_exceptions=True)
        # Clean up even if an assertion exposes a leak in the client.
        for task in tasks.values():
            task.cancel()
        await asyncio.gather(*tasks.values(), return_exceptions=True)


def assert_cleaned_up(run):
    assert all(task.done() for task in run.tasks.values())
    for resource in run.resources:
        resource.close.assert_awaited_once()


async def test_eof_drains_both_outputs_before_closing_display(media_run):
    run = media_run
    async with asyncio.timeout(5):
        run.release["publish"].set()
        await run.stopped["blur"].wait()
        assert not run.task.done()
        assert not run.stopped["display"].is_set()

        run.release["out"].set()
        await run.tasks["out"]
        assert not run.task.done()
        assert not run.stopped["display"].is_set()

        run.release["in"].set()
        await run.task

    assert run.received == ["output", "loopback"]
    assert_cleaned_up(run)


@pytest.mark.parametrize("after_eof", [False, True], ids=["publishing", "draining"])
async def test_consumer_failure_propagates_after_cleanup(media_run, after_eof):
    run = media_run
    error = OSError("output consumer failed")
    run.consumer_error = error
    async with asyncio.timeout(5):
        if after_eof:
            run.release["publish"].set()
            await run.stopped["blur"].wait()
            run.release["in"].set()
        run.release["out"].set()
        with pytest.raises(OSError) as raised:
            await run.task

    assert raised.value is error
    assert_cleaned_up(run)
    if not after_eof:
        assert run.tasks["publish"].cancelled()
        assert run.tasks["in"].cancelled()


async def test_cancellation_stops_publishing_consumers_and_display(media_run):
    run = media_run
    run.task.cancel()
    async with asyncio.timeout(5):
        with pytest.raises(asyncio.CancelledError):
            await run.task

    assert all(task.cancelled() for task in run.tasks.values())
    assert_cleaned_up(run)
