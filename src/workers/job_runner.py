from __future__ import annotations

import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

from src.workers.job import WorkerJob
from src.workers.worker import process_enrichment_job


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _worker_id() -> str:
    """
    Generate a simple worker identity.

    Cloud environments can later override this with WORKER_ID.
    """

    return os.getenv(
        "WORKER_ID",
        f"{platform.node()}-worker",
    )


def save_job(
    job: WorkerJob,
    path: str | Path,
) -> None:
    """Persist a job manifest atomically."""

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary_path.write_text(
        job.model_dump_json(indent=2),
        encoding="utf-8",
    )

    temporary_path.replace(path)


def run_job(
    job_path: str | Path,
) -> WorkerJob:
    """
    Execute one WorkerJob.

    The job manifest is updated as it moves through its lifecycle.
    """

    job_path = Path(job_path)

    if not job_path.exists():
        raise FileNotFoundError(
            f"Job file does not exist: {job_path}"
        )

    job = WorkerJob.model_validate_json(
        job_path.read_text(
            encoding="utf-8"
        )
    )

    if job.status == "completed":
        return job

    job.attempt += 1
    job.status = "running"
    job.started_at = _utc_now()
    job.worker_id = _worker_id()
    job.error = None

    save_job(job, job_path)

    try:
        process_enrichment_job(
            post_directory=job.post_directory,
            output_path=job.output_path,
        )

        job.status = "completed"
        job.completed_at = _utc_now()

        save_job(job, job_path)

        return job

    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)

        save_job(job, job_path)

        raise