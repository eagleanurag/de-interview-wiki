"""
Deciding what actually happened to a task.

An OpenCode process exiting zero is not evidence that a task was
implemented. The same zero exit is produced by several very different
situations, and only two of them are a success:

* the work was implemented, committed and pushed
* the task was genuinely read-only, so nothing changed

The remaining situations are failures that must never be reported as
a success:

* the agent process itself failed
* the working tree still holds uncommitted changes, so the work died
  with the runner
* a commit exists but never reached the remote branch
* the push could not be verified at all

Every status is therefore derived from observed repository state
rather than from the agent's exit code. The classification itself is
a pure function, so it is testable without a runner, a network or a
model.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass


# Implemented, committed and confirmed on the remote branch.
STATUS_SUCCESS = "SUCCESS"

# The agent exited cleanly and left the repository untouched. Only
# honest for a read-only task, and kept distinct from SUCCESS so a
# control-plane bug can never hide a task that did nothing.
STATUS_NO_CHANGES = "SUCCESS_NO_CHANGES"

# The agent exited cleanly but left changes uncommitted.
STATUS_DIRTY_NO_COMMIT = "DIRTY_NO_COMMIT"

# A commit was created and is not on the remote branch.
STATUS_PUSH_FAILED = "PUSH_FAILED"

# The push state could not be established, so nothing can be claimed.
STATUS_PUSH_UNVERIFIED = "PUSH_UNVERIFIED"

# The agent failed, or the test suite did not pass.
STATUS_FAILED = "FAILED"


SUCCESS_STATUSES = frozenset(
    {
        STATUS_SUCCESS,
        STATUS_NO_CHANGES,
    }
)


PUSH_PUSHED = "pushed"
PUSH_NOT_PUSHED = "not-pushed"
PUSH_UNKNOWN = "unknown"


@dataclass(frozen=True)
class Verdict:
    """The classified outcome of one attempt."""

    status: str
    reason: str
    human_action: str = "none"
    dirty_files: tuple[str, ...] = ()
    commit_created: bool = False
    push_state: str = PUSH_UNKNOWN

    @property
    def is_success(self) -> bool:
        return self.status in SUCCESS_STATUSES

    @property
    def is_recoverable(self) -> bool:
        """
        Whether a further attempt could plausibly do better.

        A read-only success and a real success are both final. Every
        other status describes unfinished work.
        """

        return not self.is_success


def classify_task(
    *,
    agent_succeeded: bool,
    start_sha: str,
    head_sha: str,
    dirty_files: list[str] | tuple[str, ...] = (),
    push_state: str = PUSH_UNKNOWN,
    tests_passed: bool = True,
    agent_timed_out: bool = False,
) -> Verdict:
    """
    Classify one attempt from observed repository state.

    `agent_succeeded` alone is never enough. A clean exit is accepted
    as success only when there is either a confirmed push of a new
    commit, or no repository change at all, and only while the test
    suite passes.
    """

    dirty = tuple(
        str(path) for path in dirty_files if str(path).strip()
    )
    commit_created = bool(head_sha) and head_sha != start_sha

    if not agent_succeeded:
        if agent_timed_out:
            reason = (
                "the agent run exceeded its time limit and was "
                "terminated, so the task is not assumed complete."
            )
        else:
            reason = (
                "the agent process did not exit cleanly, so the task "
                "is not assumed complete."
            )

        return Verdict(
            status=STATUS_FAILED,
            reason=reason,
            dirty_files=dirty,
            commit_created=commit_created,
            push_state=push_state,
        )

    if dirty:
        listed = ", ".join(dirty[:5])
        more = (
            f" and {len(dirty) - 5} more" if len(dirty) > 5 else ""
        )
        count = (
            f"{len(dirty)} file(s) are still uncommitted "
            f"({listed}{more})"
        )

        if commit_created:
            # A commit landed but the tree is not clean, so the
            # delivery is incomplete rather than absent.
            reason = (
                f"{count}. A commit was created, but the working "
                "tree is not clean, so the delivered state is not "
                "the state that was worked on."
            )
            action = (
                "Commit the remaining changes, or re-run the task so "
                "the agent commits them."
            )
        else:
            reason = (
                f"{count}. The work exists only on this runner, so "
                "nothing reached the repository and no validation "
                "could have run against it."
            )
            action = (
                "The implementation was never committed. Re-run the "
                "task so the agent commits and pushes it."
            )

        return Verdict(
            status=STATUS_DIRTY_NO_COMMIT,
            reason=reason,
            human_action=action,
            dirty_files=dirty,
            commit_created=commit_created,
            push_state=push_state,
        )

    if not commit_created:
        if not tests_passed:
            return Verdict(
                status=STATUS_FAILED,
                reason=(
                    "the agent made no repository change and the test "
                    "suite did not pass."
                ),
                commit_created=False,
                push_state=push_state,
            )

        return Verdict(
            status=STATUS_NO_CHANGES,
            reason=(
                "the agent exited cleanly without changing the "
                "repository, which is only a valid outcome for a "
                "read-only task."
            ),
            commit_created=False,
            push_state=PUSH_UNKNOWN,
        )

    if push_state == PUSH_NOT_PUSHED:
        return Verdict(
            status=STATUS_PUSH_FAILED,
            reason=(
                f"commit {head_sha} exists locally but is not on the "
                "remote branch, so the validation workflow would "
                "have validated the previous commit."
            ),
            human_action=(
                "Push the commit manually, or re-run the task so the "
                "agent pushes it."
            ),
            commit_created=True,
            push_state=push_state,
        )

    if push_state != PUSH_PUSHED:
        return Verdict(
            status=STATUS_PUSH_UNVERIFIED,
            reason=(
                f"commit {head_sha} was created but the push could not "
                "be verified, so success is not claimed."
            ),
            human_action=(
                "Confirm the commit reached the remote branch before "
                "trusting this task."
            ),
            commit_created=True,
            push_state=push_state,
        )

    if not tests_passed:
        return Verdict(
            status=STATUS_FAILED,
            reason=(
                f"commit {head_sha} was pushed but the test suite "
                "did not pass."
            ),
            commit_created=True,
            push_state=push_state,
        )

    return Verdict(
        status=STATUS_SUCCESS,
        reason=(
            f"commit {head_sha} was pushed and the test suite passed."
        ),
        commit_created=True,
        push_state=push_state,
    )


# ---------------------------------------------------------------------
# Repository inspection
# ---------------------------------------------------------------------


def _git(arguments: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *arguments],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        shell=False,
    )


def head_sha() -> str:
    """Current commit SHA, or an empty string outside a checkout."""

    return _git(["rev-parse", "HEAD"]).stdout.strip()


def dirty_files() -> list[str]:
    """Files the working tree has changed but not committed."""

    completed = _git(["status", "--porcelain"])

    files: list[str] = []

    for line in completed.stdout.splitlines():
        if len(line) < 4:
            continue

        path = line[3:].strip()

        if " -> " in path:
            path = path.split(" -> ", 1)[1]

        if path:
            files.append(path)

    return files


def last_commit_summary(limit: int = 1) -> str:
    completed = _git(
        [
            "log",
            f"-{max(1, limit)}",
            "--pretty=format:%h %s",
        ]
    )

    return completed.stdout.strip()


def is_ancestor(candidate: str, descendant: str) -> bool | None:
    """Whether `candidate` is reachable from `descendant`.

    Returns None when git cannot answer, so an unreadable repository
    is never mistaken for a definitive "no".
    """

    if not candidate or not descendant:
        return None

    completed = _git(
        ["merge-base", "--is-ancestor", candidate, descendant]
    )

    if completed.returncode == 0:
        return True

    if completed.returncode == 1:
        return False

    return None


def remote_head_sha(
    remote: str = "origin",
    ref: str = "main",
) -> str:
    """Ask the remote for the tip of a branch.

    Used only as a fallback when the local remote-tracking ref cannot
    answer, so a push that git did not record locally is still
    recognised.
    """

    completed = _git(
        ["ls-remote", remote, f"refs/heads/{ref}"]
    )

    if completed.returncode != 0:
        return ""

    first_line = completed.stdout.strip().splitlines()

    if not first_line:
        return ""

    return first_line[0].split("\t", 1)[0].strip()


def push_state(
    commit_sha: str,
    *,
    remote: str = "origin",
    ref: str = "main",
) -> str:
    """
    Establish whether the remote branch already contains a commit.

    The local remote-tracking ref is authoritative and needs no
    network, because a successful `git push` updates it. Only when
    that ref cannot answer does the remote itself get asked.
    """

    if not commit_sha:
        return PUSH_UNKNOWN

    tracking_ref = f"{remote}/{ref}"

    answer = is_ancestor(commit_sha, tracking_ref)

    if answer is not None:
        return PUSH_PUSHED if answer else PUSH_NOT_PUSHED

    remote_head = remote_head_sha(remote, ref)

    if not remote_head:
        return PUSH_UNKNOWN

    if remote_head == commit_sha:
        return PUSH_PUSHED

    answer = is_ancestor(commit_sha, remote_head)

    if answer is None:
        return PUSH_UNKNOWN

    return PUSH_PUSHED if answer else PUSH_NOT_PUSHED
