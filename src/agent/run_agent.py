#!/usr/bin/env python3
"""
Run one OpenCode attempt and record the outcome as JSON.

Called once per attempt by the agent job. Writes a result file that
later steps read, so nothing has to be carried between steps through
`$GITHUB_OUTPUT` (which is fragile for multi-line text).

    python -m src.agent.run_agent \\
      --trigger trigger.json \\
      --prompt prompt.md \\
      --out attempt.json \\
      --previous-attempt attempt-1.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from src.agent.ci import ValidationController, ValidationRun
from src.agent.opencode import (
    DEFAULT_AGENT,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    DEFAULT_VERSION,
    OpenCodeError,
    OpenCodeRunner,
)
from src.agent.prompt import MAX_REPAIR_ATTEMPTS, build_prompt
from src.agent.redaction import redact
from src.agent.reporting import (
    STATUS_BLOCKED,
    TaskOutcome,
)
from src.agent.verdict import (
    PUSH_NOT_PUSHED,
    PUSH_PUSHED,
    PUSH_UNKNOWN,
    Verdict,
    classify_task,
    dirty_files as changed_files,
    head_sha,
    last_commit_summary,
    push_state as resolve_push_state,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run OpenCode once and record the outcome."
    )

    parser.add_argument("--trigger", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--previous-attempt", default="")
    parser.add_argument("--workflow", default="run-python-worker.yml")
    parser.add_argument("--ref", default="main")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--agent", default=DEFAULT_AGENT)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--attempt",
        type=int,
        default=1,
        help="1-based attempt number.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=MAX_REPAIR_ATTEMPTS,
    )
    parser.add_argument(
        "--continue-session",
        action="store_true",
        help=(
            "Continue the previous OpenCode session so the agent "
            "keeps its context across a repair cycle."
        ),
    )
    parser.add_argument("--skip-install", action="store_true")

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


def test_result() -> tuple[str, bool]:
    """Run the project test suite and summarize it.

    Returns the summary and whether the suite passed, so a red suite
    can never be reported as a success.
    """

    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        shell=False,
    )

    tail = (completed.stdout or "").strip().splitlines()
    summary = tail[-1].strip() if tail else "no output"

    if completed.returncode != 0:
        failures = (completed.stdout or "")[-4000:]
        return f"FAILED ({completed.returncode})\n\n{failures}", False

    return f"PASSED ({completed.returncode})\n{summary}", True


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    trigger = _load_json(args.trigger)
    prompt_text = Path(args.prompt).read_text(
        encoding="utf-8", errors="replace"
    )

    controller = ValidationController(
        workflow=args.workflow, ref=args.ref
    )

    before = head_sha()

    runner = OpenCodeRunner(
        model=args.model,
        agent=args.agent,
        version=args.version,
        timeout_seconds=args.timeout,
    )

    try:
        result = runner.run(
            prompt_text,
            continue_session=args.continue_session,
        )
    except OpenCodeError as exc:
        print(f"OPENCODE_ERROR={exc}", file=sys.stderr)

        Path(args.out).write_text(
            json.dumps(
                {
                    "status": STATUS_BLOCKED,
                    "error": str(exc),
                    "reason": f"OpenCode could not be run: {exc}",
                    "head_sha": before,
                    "pushed": False,
                    "commit_created": False,
                    "push_state": PUSH_UNKNOWN,
                    "human_action": (
                        "OpenCode must be installable on the runner "
                        "before a task can run."
                    ),
                    "agent_text": "",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        return 1

    after = head_sha()
    files = changed_files()
    commits = last_commit_summary()

    tests_summary, tests_passed = test_result()

    # Whether the commit actually reached the remote branch, rather
    # than merely existing in the checkout. A local commit is not a
    # delivered one, and validation is dispatched on the remote.
    if after and after != before:
        push = resolve_push_state(after, ref=args.ref)
    else:
        push = PUSH_UNKNOWN

    pushed = push == PUSH_PUSHED

    verdict = classify_task(
        agent_succeeded=result.succeeded,
        start_sha=before,
        head_sha=after,
        dirty_files=files,
        push_state=push,
        tests_passed=tests_passed,
        agent_timed_out=result.timed_out,
    )

    # Only a confirmed push may trigger validation: dispatching for a
    # commit the remote has never seen would validate the previous
    # commit and report it as this task's result.
    if verdict.commit_created and pushed:
        try:
            controller.trigger()
            print(f"Triggered {args.workflow} for {after}")
        except Exception as exc:  # noqa: BLE001
            print(f"TRIGGER_FAILED={exc}", file=sys.stderr)

    run = (
        controller.find_run_for_commit(after)
        if verdict.commit_created and pushed
        else None
    )

    ci_result = _ci_result(verdict, push, after, run)

    outcome = TaskOutcome(
        status=verdict.status,
        reason=verdict.reason,
        commit_sha=after if verdict.commit_created else "",
        tests=tests_summary,
        validation_run=run.run_id if run else "",
        validation_url=run.url if run else "",
        ci_result=ci_result,
        recovery_attempts=args.attempt - 1,
        max_attempts=args.max_attempts,
        files_changed=(
            ", ".join(files) if files else "no uncommitted changes"
        ),
        summary=result.text or "OpenCode produced no final message.",
        human_action=verdict.human_action,
    )

    payload = {
        "status": outcome.status,
        "reason": verdict.reason,
        "head_sha": after,
        "pushed": pushed,
        "commit_created": verdict.commit_created,
        "push_state": push,
        "commits": commits,
        "agent_exit_code": result.exit_code,
        "agent_session_id": result.session_id,
        "agent_text": outcome.summary,
        "agent_timed_out": result.timed_out,
        "tests": outcome.tests,
        "tests_passed": tests_passed,
        "validation_run": outcome.validation_run,
        "validation_url": outcome.validation_url,
        "ci_result": outcome.ci_result,
        "files_changed": outcome.files_changed,
        "human_action": outcome.human_action,
        "recoverable": verdict.is_recoverable,
        "version": args.version,
        "model": args.model,
        "agent": args.agent,
        "attempt": args.attempt,
        "max_attempts": args.max_attempts,
    }

    Path(args.out).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    # Keep a redacted copy of the raw streams for the artifact.
    log_dir = Path("agent-logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    secrets = _known_secrets()

    (log_dir / f"stdout-{args.attempt}.log").write_text(
        redact(result.stdout, secrets=secrets),
        encoding="utf-8",
    )
    (log_dir / f"stderr-{args.attempt}.log").write_text(
        redact(result.stderr, secrets=secrets),
        encoding="utf-8",
    )
    (log_dir / "prompt.md").write_text(
        prompt_text, encoding="utf-8"
    )

    print(f"AGENT_STATUS={outcome.status}")
    print(f"AGENT_REASON={verdict.reason}")
    print(f"HEAD_SHA={after}")
    print(f"PUSHED={pushed}")
    print(f"VALIDATION_RUN={outcome.validation_run}")
    print(f"COMMITS={commits}")

    return 0 if verdict.is_success else 1


def _ci_result(
    verdict: Verdict,
    push: str,
    sha: str,
    run: ValidationRun | None,
) -> str:
    """Describe what happened to validation, honestly."""

    if run is not None:
        return (
            f"validation run {run.run_id} "
            f"({run.status}/{run.conclusion or 'pending'})"
        )

    if not verdict.commit_created:
        return (
            "no commit was created, so validation was not triggered; "
            "the repository is unchanged"
        )

    if push == PUSH_NOT_PUSHED:
        return (
            f"commit {sha} is not on the remote branch, so "
            "validation was not triggered"
        )

    if push == PUSH_UNKNOWN:
        return (
            f"the push of {sha} could not be verified, so validation "
            "was not triggered"
        )

    return (
        f"validation was dispatched for {sha} but no run was found "
        "for that commit"
    )


def _known_secrets() -> tuple[str, ...]:
    """
    Collect values that must never be echoed.

    Only names that look secret are read, and only from the process
    environment. Nothing is written to disk.
    """

    markers = ("token", "secret", "password", "api_key", "key")

    values = []

    for name, value in os.environ.items():
        lowered = name.lower()

        if not any(marker in lowered for marker in markers):
            continue

        if value and len(value) >= 8:
            values.append(value)

    return tuple(values)


if __name__ == "__main__":
    raise SystemExit(main())
