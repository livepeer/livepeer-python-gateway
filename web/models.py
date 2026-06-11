"""Pydantic request/response models for the web API."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class StartJobRequestBody(BaseModel):
    model_id: Optional[str] = None
    params: Optional[dict[str, Any]] = None
    request_id: Optional[str] = None
    stream_id: Optional[str] = None
    orchestrator_url: Optional[str] = None


class ControlMessageBody(BaseModel):
    message: dict[str, Any]


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class StartJobResponse(BaseModel):
    job_id: str
    model_id: str
    publish_url: Optional[str] = None
    subscribe_url: Optional[str] = None
    control_url: Optional[str] = None
    events_url: Optional[str] = None


class JobStatusResponse(BaseModel):
    job_id: str
    model_id: str
    created_at: float
    orchestrator_url: Optional[str] = None
    publish_url: Optional[str] = None
    subscribe_url: Optional[str] = None
    control_url: Optional[str] = None
    events_url: Optional[str] = None
    has_payment_session: bool = False
    media_started: bool = False


class JobListItem(BaseModel):
    job_id: str
    model_id: str
    created_at: float
    orchestrator_url: Optional[str] = None
    media_started: bool = False


class HealthResponse(BaseModel):
    status: str = "ok"
    active_jobs: int = 0
    version: str = "1.0.0"
