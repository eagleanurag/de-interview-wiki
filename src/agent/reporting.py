"""
Issue and job-summary reporting.

One report per task, written once at the end. Reports are structured,
bounded and redacted so they stay readable on a phone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.agent.events import Trigger
from src.agent.redaction import redact, truncate_for_comment
from src.agent.verdict import (
    STATUS_DIRTY_NO_COMMIT,
    STATUS_FAILED,
    STATUS_NO_CHANGES,
    STATUS_PUSH_FAILED,
    STATUS_PUSH_UNVERIFIED,
    STATUS_SUCCESS,
)


# The outcome statuses are owned by src.agent.verdict, so a report can
# never describe an outcome the classifier does not produce. These two
# are report-only, and stay here for their existing importers.
STATUS_BLOCKED = "BLOCKED"
STATUS_BLOCKED_BUDGET = "BLOCKED_AFTER_3_ATTEMPTS"

# Every status a task may end in, for validation and rendering.
TASK_STATUSES = frozenset(
    {
        STATUS_SUCCESS,
        STATUS_NO_CHANGES,
        STATUS_DIRTY_NO_COMMIT,
        STATUS_PUSH_FAILED,
        STATUS_PUSH_UNVERIFIED,
        STATUS_FAILED,
        STATUS_BLOCKED,
        STATUS_BLOCKED_BUDGET,
    }
)

MAX_COMMENT_CHARACTERS = 6000


@dataclass
class TaskOutcome:
    """Everything known about how a task ended."""

    status: str
    commit_sha: str = ""
    tests: str = "not recorded"
    validation_run: str = ""
    validation_url: str = ""
    ci_result: str = "not run"
    recovery_attempts: int = 0
    max_attempts: int = 3
    files_changed: str = "none recorded"
    summary: str = ""
    human_action: str = "none"
    reason: str = ""
    failure_evidence: str = ""

    @property
    def is_success(self) -> bool:
        return self.status in {
            STATUS_SUCCESS,
            STATUS_NO_CHANGES,
        }


@dataclass
class ReportContext:
    """Inputs used to render a report."""

    trigger: Trigger
    outcome: TaskOutcome
    secrets: tuple[str, ...] = ()
    workflow_run_url: str = ""
    extra_notes: list[str] = field(default_factory=list)


def build_report(context: ReportContext) -> str:
    """
    Render the structured task report.

    Every field is redacted before rendering, so a value that looks
    like a credential cannot reach the issue thread.
    """

    trigger = context.trigger
    outcome = context.outcome
    secrets = context.secrets

    def clean(value: object, empty: str = "not recorded") -> str:
        text = redact(str(value or "").strip(), secrets=secrets)

        return text or empty

    lines = [
        "## OpenCode Task Report",
        "",
        f"Status: {clean(outcome.status, STATUS_FAILED)}",
        "",
    ]

    if outcome.reason:
        lines += [
            "Why:",
            _block(clean(outcome.reason, "")),
            "",
        ]

    lines += [
        "Task:",
        _block(trigger.summary),
        "",
        "Commit:",
        f"`{clean(outcome.commit_sha, 'none')}`",
        "",
        "Tests:",
        _block(clean(outcome.tests, "not recorded")),
        "",
        "Validation workflow:",
        _validation_line(
            clean(outcome.validation_run, "not triggered"),
            redact(outcome.validation_url, secrets=secrets),
        ),
        "",
        "CI result:",
        _block(clean(outcome.ci_result, "not run")),
        "",
        "Recovery attempts:",
        f"{outcome.recovery_attempts}/{outcome.max_attempts}",
        "",
        "Files changed:",
        _block(clean(outcome.files_changed, "none recorded")),
        "",
        "Final result:",
        _block(clean(outcome.summary, "no summary produced")),
        "",
    ]

    if outcome.failure_evidence:
        lines += [
            "Failure evidence:",
            _code_block(
                truncate_for_comment(
                    redact(outcome.failure_evidence, secrets=secrets),
                    MAX_COMMENT_CHARACTERS // 2,
                )
            ),
            "",
        ]

    for note in context.extra_notes:
        cleaned = redact(str(note), secrets=secrets)

        if cleaned.strip():
            lines += [cleaned.strip(), ""]

    if context.workflow_run_url:
        lines += [
            f"Full logs and artifacts: "
            f"{redact(context.workflow_run_url, secrets=secrets)}",
            "",
        ]

    lines += [
        "Human action required:",
        _block(clean(outcome.human_action, "none")),
        "",
    ]

    report = "\n".join(lines)

    return truncate_for_comment(
        redact(report, secrets=secrets), MAX_COMMENT_CHARACTERS
    )


def build_job_summary(context: ReportContext) -> str:
    """Render the GitHub Actions job summary for this task."""

    trigger = context.trigger
    outcome = context.outcome
    secrets = context.secrets

    rows = [
        ("Status", redact(outcome.status, secrets=secrets)),
        ("Why", redact(outcome.reason or "not recorded", secrets=secrets)),
        ("Entry mode", trigger.kind),
        ("Issue", str(trigger.issue_number or "n/a")),
        ("Commit", redact(outcome.commit_sha or "none", secrets=secrets)),
        (
            "Validation run",
            redact(outcome.validation_run or "not triggered", secrets=secrets),
        ),
        ("CI result", redact(outcome.ci_result, secrets=secrets)),
        (
            "Recovery attempts",
            f"{outcome.recovery_attempts}/{outcome.max_attempts}",
        ),
        ("Tests", redact(outcome.tests, secrets=secrets)),
        (
            "Human action",
            redact(outcome.human_action or "none", secrets=secrets),
        ),
    ]

    lines = ["## OpenCode Task Report", ""]

    for label, value in rows:
        lines.append(f"- **{label}:** {value}")

    lines.append("")

    summary = redact(outcome.summary, secrets=secrets).strip()

    if summary:
        lines += ["### Result", "", summary, ""]

    files = redact(
        outcome.files_changed, secrets=secrets
    ).strip()

    if files:
        lines += ["### Files changed", "", files, ""]

    if context.workflow_run_url:
        lines += [
            f"[Full logs and artifacts]"
            f"({redact(context.workflow_run_url, secrets=secrets)})",
            "",
        ]

    return "\n".join(lines)


def render_result_comment(
    context: ReportContext,
) -> str:
    """Alias kept explicit for the workflow's readability."""

    return build_report(context)


def _block(value: str) -> str:
    cleaned = value.strip()

    if not cleaned:
        return "_(none)_"

    if "\n" in cleaned:
        return _code_block(cleaned)

    return cleaned


def _code_block(value: str) -> str:
    # A longer fence than any backtick run inside the value, so the
    # block cannot be terminated early by its own content.
    longest = 0
    current = 0

    for char in value:
        if char == "`":
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    fence = "`" * max(3, longest + 1)

    return f"{fence}text\n{value}\n{fence}"


def _validation_line(run: str, url: str) -> str:
    if url:
        return f"[`{run}`]({url})"

    return run
