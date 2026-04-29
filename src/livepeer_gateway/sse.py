from __future__ import annotations

import json
import ssl
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Optional

import aiohttp

from .errors import LivepeerGatewayError, PaymentRequiredError


@dataclass(frozen=True)
class SSEEvent:
    data: str
    event: str = "message"
    id: Optional[str] = None
    retry: Optional[int] = None

    def json(self) -> Any:
        return json.loads(self.data)


def parse_sse_lines(lines: list[str]) -> Optional[SSEEvent]:
    event = "message"
    event_id: Optional[str] = None
    retry: Optional[int] = None
    data: list[str] = []

    for raw_line in lines:
        line = raw_line.rstrip("\r")
        if not line or line.startswith(":"):
            continue
        if ":" in line:
            field, value = line.split(":", 1)
            if value.startswith(" "):
                value = value[1:]
        else:
            field = line
            value = ""

        if field == "event":
            event = value
        elif field == "data":
            data.append(value)
        elif field == "id":
            event_id = value
        elif field == "retry":
            try:
                retry = int(value)
            except ValueError:
                continue

    if not data and event == "message" and event_id is None and retry is None:
        return None
    return SSEEvent(data="\n".join(data), event=event, id=event_id, retry=retry)


class SSEClient:
    def __init__(
        self,
        url: str,
        *,
        method: str = "GET",
        payload: Optional[dict[str, Any]] = None,
        headers: Optional[dict[str, str]] = None,
        timeout: Optional[float] = None,
        retry_headers: Optional[Callable[[], dict[str, str]]] = None,
    ) -> None:
        self.url = url
        self.method = method
        self.payload = payload
        self.headers = headers or {}
        self.timeout = timeout
        self.retry_headers = retry_headers

    @classmethod
    def post_json(
        cls,
        url: str,
        *,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: Optional[float] = None,
        retry_headers: Optional[Callable[[], dict[str, str]]] = None,
    ) -> "SSEClient":
        return cls(
            url,
            method="POST",
            payload=payload,
            headers=headers,
            timeout=timeout,
            retry_headers=retry_headers,
        )

    def __aiter__(self) -> AsyncIterator[SSEEvent]:
        return self.events()

    async def events(self) -> AsyncIterator[SSEEvent]:
        async for event in self._events_once(self.headers, allow_retry=True):
            yield event

    async def json_events(self) -> AsyncIterator[Any]:
        async for event in self.events():
            if event.data == "[DONE]":
                yield event.data
                return
            try:
                yield event.json()
            except json.JSONDecodeError as e:
                raise LivepeerGatewayError(f"SSE JSON decode failed: {e} (data={event.data[:256]!r})") from e

    async def _events_once(
        self,
        headers: dict[str, str],
        *,
        allow_retry: bool,
    ) -> AsyncIterator[SSEEvent]:
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        ssl_ctx = ssl._create_unverified_context()
        req_headers = {
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "User-Agent": "livepeer-python-gateway/0.1",
        }
        req_headers.update(headers)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.request(
                    self.method,
                    self.url,
                    json=self.payload,
                    headers=req_headers,
                    ssl=ssl_ctx,
                ) as resp:
                    if resp.status == 402 and self.retry_headers is not None and allow_retry:
                        raise PaymentRequiredError("SSE request returned HTTP 402 payment required")
                    if resp.status >= 400:
                        body = await resp.text()
                        raise LivepeerGatewayError(
                            f"SSE request failed: HTTP {resp.status} from endpoint "
                            f"(url={self.url}); body={body[:512]!r}"
                        )

                    content_type = resp.headers.get("Content-Type", "")
                    if "text/event-stream" not in content_type:
                        raise LivepeerGatewayError(
                            f"SSE request expected text/event-stream, got {content_type!r}"
                        )

                    pending: list[str] = []
                    buffered = ""
                    async for raw in resp.content:
                        buffered += raw.decode("utf-8", errors="replace")
                        lines = buffered.split("\n")
                        buffered = lines.pop()
                        for line in lines:
                            line = line.rstrip("\r")
                            if line == "":
                                event = parse_sse_lines(pending)
                                pending = []
                                if event is not None:
                                    yield event
                                    if event.data == "[DONE]":
                                        return
                            else:
                                pending.append(line)

                    if buffered:
                        pending.append(buffered.rstrip("\r"))
                    event = parse_sse_lines(pending)
                    if event is not None:
                        yield event
        except PaymentRequiredError:
            if self.retry_headers is None:
                raise
            retry_headers = self.retry_headers()
            async for event in self._events_once(retry_headers, allow_retry=False):
                yield event
        except LivepeerGatewayError:
            raise
        except Exception as e:
            raise LivepeerGatewayError(
                f"SSE request error: {e.__class__.__name__}: {e} (url={self.url})"
            ) from e
