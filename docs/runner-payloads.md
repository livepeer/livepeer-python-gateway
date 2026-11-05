# Calling runners

Use `call_runner()` for a known runner URL, or a [selector](#selectors) to discover
one. By default, calls return a complete `LiveRunnerCallResult`: JSON objects in
`result.data`, other formats as bytes in `result.content`, and the response media
type in `result.content_type`.

## `call_runner` parameters

All parameters except `runner_url` must be passed by keyword.

| Parameter | Default | Meaning |
|---|---|---|
| `runner_url` | `""` | Runner endpoint URL. Required unless supplied by `runner`. |
| `runner` | `None` | A `LiveRunnerInstance` with URL and app/pricing metadata. An explicit `runner_url` takes precedence. |
| `payload` | `None` | Request body; see [body types](#body-types). |
| `content_type` | `None` | Override the request's `Content-Type` header without changing its encoding. |
| `method` | `"POST"` | HTTP request method. |
| `signer_url` | `None` | Remote signer URL for automatic payment. Required for on-chain calls. |
| `signer_headers` | `None` | Dictionary of additional HTTP headers for signer requests. |
| `payment_unit` | `None` | `"hour"`, `"seconds"`, or `"fixed"`. Defaults to the runner's pricing unit; an explicit value must agree with it. With neither, payment uses ongoing funding. |
| `timeout` | `5.0` | Seconds per request attempt; see [timeouts](#timeouts) for streaming behavior. |
| `max_payment_challenge_retries` | `3` | Payment retry budget, with additional allowance to refresh expired payment parameters. |
| `stream` | `False` | Set to `True` for an incremental [response](#streaming-responses) as a `LiveRunnerCallStream`. |

### Body types

`dict` values default to `application/json`; other bodies default to
`application/octet-stream`. Set `content_type` for another media type, such as
`"audio/wav"`.

| `payload` | Sent as |
|---|---|
| `dict` | JSON. |
| `str` | UTF-8. |
| `bytes`, `bytearray`, or `memoryview` | Raw bytes. |
| `BinaryIO` or `AsyncIterable[bytes]` | Incremental upload using `StreamPayload` defaults. |
| `StreamPayload` | Incremental upload with explicit settings. |
| Omitted or `None` | Empty body. |

## Streaming uploads

`StreamPayload` is an immutable configuration for a file or async iterable
payload. Bare files and async iterables use the same defaults.

| Field | Default | Meaning |
|---|---|---|
| `source` | Required | A binary file with a synchronous `read()` method, or an async iterable yielding `bytes`. |
| `retry_buffer` | `True` | Retain input from non-seekable files and async iterables so it can be sent again on a retry. |
| `retry_buffer_memory_limit` | `25_000_000` | Bytes of retained input allowed in memory before using temporary disk storage (25 MB). |
| `retry_buffer_limit` | `250_000_000` | Total bytes of input retained for retries (250 MB). |

Limits use decimal bytes and require
`0 < retry_buffer_memory_limit <= retry_buffer_limit`. They bound the retained
copy, not upload size or memory used by your code. Exceeding the total limit or a
storage failure lets the current upload continue but prevents later retries.
Seekable files rewind for retries and do not use these limits.

### When input is read

Creating a `StreamPayload` or discovering runners does not read the input.

For streaming requests to on-chain single-shot runners, `call_runner(runner=discovered_runner, ...)`
or `cursor.next()` first sends an empty request to the app URL to receive a
`402 Payment Required` challenge. After obtaining payment from the signer,
the SDK sends a second request with payment headers and the streamed body.
Challenge acquisition and signing leave the body unread.

A failed or rejected upload may leave the source partly consumed. Retrying
requires either rewinding a seekable file or resending the consumed input
from the retry buffer; resending may not be appropriate for all content types
(eg, live video that should stay close to the edge). If resending is not
possible, the SDK raises `LivepeerGatewayError`.

You are responsible for closing files and managing generator resources. Keep
inputs usable until the call finishes or the response/selector context closes,
and do not read them elsewhere during the operation. The examples nest contexts
to ensure this.

## Retries

A retry sends the same body in a new request:

- After `402 Payment Required`, `call_runner()` obtains payment through the
  configured signer and submits the request again.
- If a runner call fails, a selector's `next()` tries another candidate.

For streamed uploads, `call_runner()` does not automatically resend after
connection failures or follow redirects. A selector can try another candidate.

Failover can repeat work if the original runner executed it but its response was
lost.

Dictionaries, strings, and byte values can be sent again directly. Streamed input
must still be usable:

- Seekable files rewind to their position at the start of the operation, even
  with `retry_buffer=False`.
- Non-seekable files and async iterables can resend the full body, including
  after a partial upload, if retry buffering remains available.
- With `retry_buffer=False`, non-seekable input can only be retried before reading
  starts.

A required retry raises `LivepeerGatewayError` if the body cannot be sent again.
This retry support lasts for one call or selector cursor. For a separate call,
supply a fresh input or rewind the file yourself.

## Streaming responses

`stream=True` controls the **response**, independently of the upload type. It
returns after payment challenges are handled and final response headers arrive;
the upload may still be running. Read bytes with `aiter_bytes()` or decoded text
lines with `aiter_lines()`. JSON is not parsed automatically.

For an app that accepts audio and returns text incrementally:

```python
from livepeer_gateway import call_runner


async def transcribe(runner, signer_url, path):
    with open(path, "rb") as audio:
        async with await call_runner(
            runner=runner,
            signer_url=signer_url,
            payload=audio,
            content_type="audio/wav",
            stream=True,
        ) as response:
            async for line in response.aiter_lines():
                print(line)
```

Use the response as an async context manager or call `await response.aclose()`.
Closing it closes the connection and stops any unfinished upload and ongoing
payment funding. Errors while reading a returned response are not retried.

### Timeouts

| Response mode | What `timeout` bounds per attempt |
|---|---|
| `stream=False` | The whole HTTP request, including upload and response reading. Increase it for longer uploads or jobs. |
| `stream=True` | Connection establishment only. Uploads and response reading have no deadline. |

Payment challenge requests use `timeout` for the complete request.

Retries can make the overall operation take longer. Use `asyncio.timeout()`
around the operation if you need an overall deadline.

## Selectors

`runner_selector()` discovers candidates and returns a cursor. Pass input as
`body` instead of `payload`, with the same optional `content_type`. Each
`cursor.next()` returns the next successful result; another call sends the same
body to subsequent candidates. Responses are always read in full.
Await each `cursor.next()` before calling it again.

Use an async context manager or `await cursor.aclose()` when finished with the
cursor. Closing it ends selection but leaves your file open.

```python
from livepeer_gateway import runner_selector


async def upload_to_available_runner(path, discovery_url):
    with open(path, "rb") as audio:
        async with await runner_selector(
            discovery_url=discovery_url,
            app="example/transcribe",
            body=audio,
            content_type="audio/wav",
            timeout=60,
        ) as cursor:
            return await cursor.next()
```
