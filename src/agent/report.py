#!/usr/bin/env python3
"""
Render the final task report and deliver it.

Called after the agent job. Posts one comment into the originating
issue when the run was issue-driven, and always writes a job summary
plus an artifact.

    python -m src.agent.report \\
      --trigger trigger.json \\
      --attempt attempt.json \\
      --out report.md \\
      --summary job-summary.md
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from src.agent.events import Trigger
from src.agent.reporting import (
    STATUS_FAILED,
    TASK_STATUSES,
    ReportContext,
    TaskOutcome,
    build_job_summary,
    build_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render and deliver the task report."
    )

    parser.add_argument("--trigger", required=True)
    parser.add_argument("--attempt", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument(
        "--post",
        action="store_true",
        help="Post the report into the originating issue.",
    )
    parser.add_argument(
        "--failure-evidence",
        default="",
        help="File containing bounded, redacted failure output.",
    )

    return parser


def _load_json(path: str) -> dict:
    if not path:
        return {}

    file = Path(path)

    if not file.is_file():
        return {}

    try:
        payload = json.loads(
            file.read_text(encoding="utf-8", errors="replace")
        )
    except json.JSONDecodeError:
        return {}

    return payload if isinstance(payload, dict) else {}


def trigger_from_dict(payload: dict) -> Trigger:
    return Trigger(
        kind=str(payload.get("kind", "unknown")),
        task=str(payload.get("task", "")),
        instruction=str(payload.get("instruction", "")),
        issue_number=payload.get("issue_number"),
        issue_title=str(payload.get("issue_title", "")),
        actor=str(payload.get("actor", "")),
        prior_context=str(payload.get("prior_context", "")),
    )


def outcome_from_dict(
    payload: dict,
    *,
    failure_evidence: str = "",
) -> TaskOutcome:
    """
    Build an outcome from an attempt file.

    A missing or unreadable file yields an honest FAILED outcome rather
    than an optimistic guess, so a missing file can never be reported
    as a success. The status is also rejected when it is not a status
    the control plane can actually produce, so a corrupted or
    hand-edited attempt file cannot claim SUCCESS either.
    """

    status = str(payload.get("status") or "FAILED")

    if status not in TASK_STATUSES:
        status = STATUS_FAILED

    return TaskOutcome(
        status=status,
        reason=str(payload.get("reason") or payload.get("error") or ""),
        commit_sha=str(payload.get("head_sha") or "")
        if payload.get("commit_created", True)
        else "",
        tests=str(payload.get("tests") or "not recorded"),
        validation_run=str(payload.get("validation_run") or ""),
        validation_url=str(payload.get("validation_url") or ""),
        ci_result=str(payload.get("ci_result") or "not run"),
        recovery_attempts=int(payload.get("attempt") or 1) - 1,
        max_attempts=int(payload.get("max_attempts") or 3),
        files_changed=str(payload.get("files_changed") or "none"),
        summary=str(payload.get("agent_text") or ""),
        human_action=str(payload.get("human_action") or "none"),
        failure_evidence=failure_evidence,
    )


def known_secrets() -> tuple[str, ...]:
    markers = ("token", "secret", "password", "api_key", "key")
    values = []

    for name, value in os.environ.items():
        if any(marker in name.lower() for marker in markers):
            if value and len(value) >= 8:
                values.append(value)

    return tuple(values)


def post_to_issue(
    issue_number: int,
    report: str,
    *,
    repository: str = "",
) -> bool:
    """Post the report as an issue comment."""

    command = [
        "gh",
        "issue",
        "comment",
        str(issue_number),
        "--body-file",
        "-",
    ]

    if repository:
        command += ["--repo", repository]

    completed = subprocess.run(
        command,
        input=report,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        shell=False,
    )

    if completed.returncode != 0:
        print(
            f"COMMENT_FAILED={completed.stderr.strip()[:1000]}",
            file=sys.stderr,
        )
        return False

    print(f"COMMENT_URL={completed.stdout.strip()}")

    return True


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    trigger_payload = _load_json(args.trigger)
    attempt_payload = _load_json(args.attempt)

    failure_evidence = ""
    if args.failure_evidence:
        file = Path(args.failure_evidence)
        if file.is_file():
            failure_evidence = file.read_text(
                encoding="utf-8", errors="replace"
            )

    trigger = trigger_from_dict(trigger_payload)
    outcome = outcome_from_dict(
        attempt_payload, failure_evidence=failure_evidence
    )

    context = ReportContext(
        trigger=trigger,
        outcome=outcome,
        secrets=known_secrets(),
        workflow_run_url=os.environ.get(
            "GITHUB_SERVER_URL", ""
        )
        + f"/{os.environ.get('GITHUB_REPOSITORY', '')}/actions/runs/"
        + os.environ.get("GITHUB_RUN_ID", ""),
    )

    report = build_report(context)
    summary = build_job_summary(context)

    Path(args.out).write_text(report, encoding="utf-8")
    Path(args.summary).write_text(summary, encoding="utf-8")

    print("=== JOB SUMMARY ===")
    print(summary)

    if args.post and trigger.issue_number:
        post_to_issue(
            int(trigger.issue_number),
            report,
            repository=os.environ.get("GITHUB_REPOSITORY", ""),
        )
    elif args.post:
        print(
            "No issue number; report available in the job summary "
            "and the artifact."
        )

    print(f"STATUS={outcome.status}")

    return 0 if outcome.is_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
