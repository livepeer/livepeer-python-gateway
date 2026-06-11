"""
FastAPI web wrapper for livepeer-gateway SDK.

Bridges browser JPEG frames to the SDK's av.VideoFrame-based media pipeline
and streams AI-processed frames back over WebSocket.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Optional

import av
from fastapi import Depends, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from starlette.responses import StreamingResponse

from livepeer_gateway.lv2v import LiveVideoToVideo, StartJobRequest, start_lv2v
from livepeer_gateway.media_publish import MediaPublishConfig
from livepeer_gateway.scope import start_scope

# Models served by the live-runner discovery/reserve flow (vs the legacy
# AI-worker /live-video-to-video path). Scope runs as a live runner, so its
# capacity lives in the live-runner registry, not the AI-worker capacity check.
_LIVE_RUNNER_MODELS = {"scope"}

from .auth import get_api_key_dependency, get_ws_api_key_dependency
from .config import Config
from .models import (
    ControlMessageBody,
    HealthResponse,
    JobListItem,
    JobStatusResponse,
    StartJobRequestBody,
    StartJobResponse,
)

_LOG = logging.getLogger(__name__)
_TIME_BASE = 90_000
_VERSION = "1.0.0"

config = Config()

# Build auth dependencies from config.
verify_api_key = get_api_key_dependency(config)
verify_ws_api_key = get_ws_api_key_dependency(config)

# ---------------------------------------------------------------------------
# Job state
# ---------------------------------------------------------------------------


@dataclass
class JobState:
    job_id: str
    model_id: str
    job: LiveVideoToVideo
    created_at: float = field(default_factory=time.time)
    orchestrator_url: Optional[str] = None
    api_key: Optional[str] = None
    _media_started: bool = False


_jobs: dict[str, JobState] = {}

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Livepeer Gateway API", version=_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_BROWSER_DIR = Path(__file__).resolve().parent.parent / "browser"

if _BROWSER_DIR.is_dir():
    app.mount("/app", StaticFiles(directory=str(_BROWSER_DIR)), name="browser-static")


# ---------------------------------------------------------------------------
# Shutdown handler — close all active jobs on SIGTERM (Cloud Run)
# ---------------------------------------------------------------------------


@app.on_event("shutdown")
async def shutdown_event():
    _LOG.info("Shutting down — closing %d active job(s)", len(_jobs))
    tasks = []
    for state in list(_jobs.values()):
        tasks.append(_close_job(state))
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    _jobs.clear()


async def _close_job(state: JobState) -> None:
    try:
        await state.job.close()
    except Exception:
        _LOG.exception("Error closing job %s during shutdown", state.job_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _jobs_for_key(api_key: Optional[str]) -> int:
    """Count active jobs owned by a given API key."""
    if api_key is None:
        return 0
    return sum(1 for s in _jobs.values() if s.api_key == api_key)


# ---------------------------------------------------------------------------
# Public routes (no auth)
# ---------------------------------------------------------------------------


@app.get("/")
async def index():
    index_file = _BROWSER_DIR / "index.html"
    if index_file.is_file():
        return FileResponse(str(index_file), media_type="text/html")
    return JSONResponse({"error": "Browser app not found"}, status_code=404)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(active_jobs=len(_jobs), version=_VERSION)


# ---------------------------------------------------------------------------
# Authenticated routes
# ---------------------------------------------------------------------------


@app.get("/jobs", response_model=list[JobListItem])
async def list_jobs(api_key: Optional[str] = Depends(verify_api_key)):
    return [
        JobListItem(
            job_id=s.job_id,
            model_id=s.model_id,
            created_at=s.created_at,
            orchestrator_url=s.orchestrator_url,
            media_started=s._media_started,
        )
        for s in _jobs.values()
    ]


@app.post("/start-job", response_model=StartJobResponse)
async def start_job(
    body: StartJobRequestBody = StartJobRequestBody(),
    api_key: Optional[str] = Depends(verify_api_key),
):
    # Per-key job limit.
    if api_key and _jobs_for_key(api_key) >= config.max_jobs_per_key:
        return JSONResponse(
            {"error": f"Job limit reached ({config.max_jobs_per_key} per key)"},
            status_code=429,
        )

    model_id = body.model_id or config.default_model_id
    orch_url = body.orchestrator_url or config.orchestrator_url

    req = StartJobRequest(
        model_id=model_id,
        params=body.params,
        request_id=body.request_id,
        stream_id=body.stream_id,
    )

    try:
        if model_id in _LIVE_RUNNER_MODELS:
            # Live-runner flow: discover a runner via {orch}/discovery, reserve a
            # session, and stream to it (handles the runner payment challenge).
            job = await start_scope(
                orch_url,
                req,
                token=config.livepeer_token,
                signer_url=config.effective_signer_url,
            )
        else:
            # Legacy AI-worker flow (/live-video-to-video).
            job = await asyncio.to_thread(
                start_lv2v,
                orch_url,
                req,
                token=config.livepeer_token,
                signer_url=config.effective_signer_url,
            )
    except Exception as e:
        _LOG.exception("Failed to start job")
        detail: dict = {"error": str(e)}
        # Include orchestrator rejection details if available.
        if hasattr(e, "rejections") and e.rejections:
            detail["rejections"] = [
                {"url": r.url, "reason": str(r.reason)} for r in e.rejections
            ]
        return JSONResponse(detail, status_code=500)

    job_id = str(uuid.uuid4())
    _jobs[job_id] = JobState(
        job_id=job_id,
        model_id=model_id,
        job=job,
        orchestrator_url=orch_url,
        api_key=api_key,
    )

    _LOG.info("Started job %s (model=%s)", job_id, model_id)
    return StartJobResponse(
        job_id=job_id,
        model_id=model_id,
        publish_url=job.publish_url,
        subscribe_url=job.subscribe_url,
        control_url=job.control_url,
        events_url=job.events_url,
    )


@app.get("/job/{job_id}", response_model=JobStatusResponse)
async def get_job(
    job_id: str,
    api_key: Optional[str] = Depends(verify_api_key),
):
    state = _jobs.get(job_id)
    if not state:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    job = state.job
    return JobStatusResponse(
        job_id=state.job_id,
        model_id=state.model_id,
        created_at=state.created_at,
        orchestrator_url=state.orchestrator_url,
        publish_url=job.publish_url,
        subscribe_url=job.subscribe_url,
        control_url=job.control_url,
        events_url=job.events_url,
        has_payment_session=job._payment_session is not None,
        media_started=state._media_started,
    )


@app.delete("/stop-job/{job_id}")
async def stop_job(
    job_id: str,
    api_key: Optional[str] = Depends(verify_api_key),
):
    state = _jobs.pop(job_id, None)
    if not state:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    try:
        await state.job.close()
    except Exception:
        _LOG.exception("Error closing job %s", job_id)

    _LOG.info("Stopped job %s", job_id)
    return {"status": "stopped", "job_id": job_id}


@app.post("/job/{job_id}/control")
async def send_control(
    job_id: str,
    body: ControlMessageBody,
    api_key: Optional[str] = Depends(verify_api_key),
):
    state = _jobs.get(job_id)
    if not state:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    control = state.job.control
    if not control:
        return JSONResponse(
            {"error": "Job has no control channel"}, status_code=400
        )

    try:
        await control.write_control(body.message)
    except Exception as e:
        _LOG.exception("Failed to send control message for job %s", job_id)
        return JSONResponse({"error": str(e)}, status_code=500)

    return {"status": "sent", "job_id": job_id}


@app.get("/job/{job_id}/events")
async def stream_events(
    job_id: str,
    api_key: Optional[str] = Depends(verify_api_key),
):
    """SSE endpoint that streams job events."""
    state = _jobs.get(job_id)
    if not state:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    events = state.job.events
    if not events:
        return JSONResponse(
            {"error": "Job has no events channel"}, status_code=400
        )

    async def event_generator():
        try:
            async for event in events():
                data = json.dumps(event)
                yield f"data: {data}\n\n"
        except asyncio.CancelledError:
            return
        except Exception:
            _LOG.debug("Events stream ended for job %s", job_id, exc_info=True)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# WebSocket — bidirectional JPEG streaming
# ---------------------------------------------------------------------------


@app.websocket("/ws/stream")
async def ws_stream(
    ws: WebSocket,
    job_id: str,
    api_key: Optional[str] = Depends(verify_ws_api_key),
):
    # Reject if auth failed (WebSocket deps can't raise HTTPException).
    if api_key == "__REJECT__":
        await ws.close(code=4001, reason="Invalid or missing API key")
        return

    state = _jobs.get(job_id)
    if not state:
        await ws.close(code=4004, reason="Job not found")
        return

    await ws.accept()
    _LOG.info("WebSocket connected for job %s", job_id)

    job = state.job

    # Start media publisher if not yet started.
    if not state._media_started:
        job.start_media(MediaPublishConfig(fps=config.fps))
        state._media_started = True

    media = job._media
    output = job.media_output()

    # PTS tracking for input frames.
    last_pts = 0
    last_time: Optional[float] = None
    time_base = Fraction(1, _TIME_BASE)

    async def _send_output():
        """Read AI-processed frames from SDK and send as JPEG to browser."""
        try:
            async for decoded in output.frames():
                if decoded.kind != "video":
                    continue
                try:
                    pil_img = decoded.frame.to_image()
                    buf = io.BytesIO()
                    pil_img.save(buf, format="JPEG", quality=config.jpeg_quality)
                    await ws.send_bytes(buf.getvalue())
                except WebSocketDisconnect:
                    return
                except Exception:
                    _LOG.debug("Failed to send output frame", exc_info=True)
        except asyncio.CancelledError:
            return
        except Exception:
            _LOG.debug("Output stream ended", exc_info=True)

    output_task = asyncio.create_task(_send_output())

    try:
        while True:
            data = await ws.receive_bytes()

            # Decode JPEG from browser into av.VideoFrame.
            try:
                pil_img = Image.open(io.BytesIO(data))
                frame = av.VideoFrame.from_image(pil_img)
            except Exception:
                _LOG.debug("Failed to decode input JPEG frame", exc_info=True)
                continue

            # Compute PTS.
            now = time.time()
            if last_time is not None:
                last_pts += int((now - last_time) * _TIME_BASE)
            else:
                last_pts = 0
            last_time = now

            frame.pts = last_pts
            frame.time_base = time_base

            await media.write_frame(frame)

    except WebSocketDisconnect:
        _LOG.info("WebSocket disconnected for job %s", job_id)
    except Exception:
        _LOG.exception("WebSocket error for job %s", job_id)
    finally:
        output_task.cancel()
        with suppress(asyncio.CancelledError):
            await output_task
        await output.close()
