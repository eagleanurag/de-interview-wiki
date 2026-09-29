from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


JobStatus = Literal[
    "pending",
    "running",
    "completed",
    "failed",
]


class WorkerJob(BaseModel):
    """
    Portable description of one worker job.

    The job is independent of GitHub Actions, Azure, or any
    particular execution environment.
    """

    job_id: str

    post_directory: str
    output_path: str

    status: JobStatus = "pending"

    attempt: int = 0
    max_attempts: int = 3

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    started_at: datetime | None = None
    completed_at: datetime | None = None

    worker_id: str | None = None

    error: str | None = None