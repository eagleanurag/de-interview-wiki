"""
Driving the repository's validation pipeline.

The existing worker pipeline is the authority: it discovers posts,
runs workers, aggregates, generates the wiki and deploys Pages. The
control plane never re-implements or bypasses it.

Two properties matter more than anything else here:

* Runs are always resolved by the exact commit SHA that the agent
  pushed, never by "the latest run", so a concurrent unrelated run
  cannot be mistaken for this task's validation.
* Failure output is bounded and redacted before it is handed back to
  the model or posted anywhere.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass


DEFAULT_VALIDATION_WORKFLOW = "run-python-worker.yml"

POLL_INTERVAL_SECONDS = 20

# Bounded so a failed validation cannot flood the model context or an
# issue comment.
MAX_LOG_CHARACTERS = 20000


@dataclass(frozen=True)
class ValidationRun:
    """One validation pipeline run tied to a specific commit."""

    run_id: str
    status: str
    conclusion: str
    url: str
    head_sha: str

    @property
    def is_complete(self) -> bool:
        return self.status == "completed"

    @property
    def succeeded(self) -> bool:
        return self.conclusion == "success"

    @property
    def failed(self) -> bool:
        return self.conclusion in {"failure", "cancelled", "timed_out"}


class CIError(RuntimeError):
    """Raised when the validation pipeline cannot be controlled."""


class GitHubCLI:
    """A very small, mockable wrapper over `gh`."""

    def __init__(self, repository: str = "") -> None:
        self.repository = repository

    def run(
        self,
        arguments: list[str],
        *,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        command = ["gh", *arguments]

        if self.repository:
            command += ["--repo", self.repository]

        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            shell=False,
        )

        if check and completed.returncode != 0:
            raise CIError(
                f"`{' '.join(command[:3])} ...` failed with exit "
                f"{completed.returncode}: "
                f"{completed.stderr.strip()[:2000]}"
            )

        return completed

    def json(
        self,
        arguments: list[str],
        *,
        default: object = None,
    ) -> object:
        completed = self.run(arguments, check=False)

        if completed.returncode != 0:
            if default is not None:
                return default

            raise CIError(
                f"gh {' '.join(arguments)} failed: "
                f"{completed.stderr.strip()[:2000]}"
            )

        text = completed.stdout.strip()

        if not text:
            return default if default is not None else {}

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise CIError(
                f"Could not parse gh output as JSON: {exc}"
            ) from exc


class ValidationController:
    """Trigger and observe the repository validation workflow."""

    def __init__(
        self,
        cli: GitHubCLI | None = None,
        *,
        workflow: str = DEFAULT_VALIDATION_WORKFLOW,
        ref: str = "main",
    ) -> None:
        self.cli = cli or GitHubCLI()
        self.workflow = workflow
        self.ref = ref

    def trigger(self) -> str:
        """
        Dispatch the validation workflow.

        Returns the dispatch confirmation. The run id is resolved
        later by commit, because a dispatched workflow is not
        immediately visible by id.
        """

        completed = self.cli.run(
            [
                "workflow",
                "run",
                self.workflow,
                "--ref",
                self.ref,
            ]
        )

        return completed.stdout.strip()

    def find_run_for_commit(
        self,
        commit_sha: str,
    ) -> ValidationRun | None:
        """
        Resolve the run that validates exactly this commit.

        `--commit` is the critical filter. Without it a run started by
        a concurrent dispatch could be reported as this task's result.
        """

        payload = self.cli.json(
            [
                "run",
                "list",
                "--workflow",
                self.workflow,
                "--commit",
                commit_sha,
                "--limit",
                "5",
                "--json",
                "databaseId,status,conclusion,url,headSha",
            ],
            default=[],
        )

        if not isinstance(payload, list) or not payload:
            return None

        for entry in payload:
            if not isinstance(entry, dict):
                continue

            if entry.get("headSha") != commit_sha:
                continue

            return ValidationRun(
                run_id=str(entry.get("databaseId", "")),
                status=str(entry.get("status", "")),
                conclusion=str(entry.get("conclusion") or ""),
                url=str(entry.get("url", "")),
                head_sha=commit_sha,
            )

        return None

    def view(self, run_id: str) -> ValidationRun:
        """Fetch the current state of one run."""

        payload = self.cli.json(
            [
                "run",
                "view",
                str(run_id),
                "--json",
                "databaseId,status,conclusion,url,headSha",
            ]
        )

        if not isinstance(payload, dict):
            raise CIError(
                f"Unexpected gh output for run {run_id}"
            )

        return ValidationRun(
            run_id=str(payload.get("databaseId", run_id)),
            status=str(payload.get("status", "")),
            conclusion=str(payload.get("conclusion") or ""),
            url=str(payload.get("url", "")),
            head_sha=str(payload.get("headSha", "")),
        )

    def failed_logs(self, run_id: str) -> str:
        """
        Fetch failing-job logs, bounded.

        An empty string means the run failed without producing
        readable failing-step logs, for example a cancelled run or a
        failure in job setup.
        """

        completed = self.cli.run(
            ["run", "view", str(run_id), "--log-failed"],
            check=False,
        )

        output = (completed.stdout or "") + (completed.stderr or "")

        return bound_log(output)


def bound_log(text: str | None) -> str:
    """Keep the head and tail of a log, dropping the middle."""

    cleaned = (text or "").strip()

    if len(cleaned) <= MAX_LOG_CHARACTERS:
        return cleaned

    head = MAX_LOG_CHARACTERS // 3
    tail = MAX_LOG_CHARACTERS - head - 60
    omitted = len(cleaned) - head - tail

    return (
        cleaned[:head]
        + f"\n\n... [{omitted} characters omitted] ...\n\n"
        + cleaned[-tail:]
    )


def summarize_failure(run: ValidationRun) -> str:
    """Classify a failed validation run for the agent."""

    if run.conclusion == "cancelled":
        return (
            "The validation run was cancelled. It may have been "
            "superseded or interrupted."
        )

    if run.conclusion == "timed_out":
        return (
            "The validation run exceeded the GitHub Actions time "
            "limit."
        )

    return (
        "The validation run failed. The logs below are from the "
        "failing steps."
    )
