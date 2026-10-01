from __future__ import annotations

import argparse
import sys

from src.workers.job_runner import run_job


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one de-interview-wiki worker job."
    )

    parser.add_argument(
        "--job",
        required=True,
        help="Path to the WorkerJob JSON manifest.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-run a job that already completed. Without this a "
            "completed manifest is skipped, so a change to the "
            "enrichment code would otherwise have no effect."
        ),
    )

    args = parser.parse_args()

    try:
        job = run_job(args.job, force=args.force)

        print(f"JOB_ID={job.job_id}")
        print(f"STATUS={job.status}")
        print(f"ATTEMPT={job.attempt}")
        print(f"WORKER={job.worker_id}")
        print(f"OUTPUT={job.output_path}")

        return 0 if job.status == "completed" else 1

    except Exception as exc:
        print(
            f"WORKER_ERROR={exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
