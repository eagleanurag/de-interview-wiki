from pathlib import Path

from src.workers.job import WorkerJob
from src.workers.job_runner import run_job


def main() -> None:
    job_path = Path(
        "data/jobs/sample_001.json"
    )

    print("=== WORKER JOB TEST ===")

    job = run_job(job_path)

    print(f"Job ID: {job.job_id}")
    print(f"Status: {job.status}")
    print(f"Attempt: {job.attempt}")
    print(f"Worker: {job.worker_id}")
    print(f"Output: {job.output_path}")

    assert job.status == "completed"
    assert job.attempt == 1
    assert Path(job.output_path).exists()

    saved_job = WorkerJob.model_validate_json(
        job_path.read_text(
            encoding="utf-8"
        )
    )

    assert saved_job.status == "completed"

    print()
    print("Worker job lifecycle OK")


if __name__ == "__main__":
    main()