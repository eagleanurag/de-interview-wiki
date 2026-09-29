"""
Trigger detection, authorization and task extraction.

Everything that decides *whether* to run the agent, and *what* to ask
it, lives here so the rules can be tested directly rather than being
scattered through YAML expressions.

Security model
--------------
The agent can push code and trigger CI, so three gates must all pass:

1. The acting user must be the repository owner. Anyone else is
   ignored silently.
2. A new issue must carry the ``[OpenCode]`` title prefix.
3. A follow-up comment must start with a recognised command.

A comment body is treated as untrusted text. It is never interpolated
into a shell command, and it is passed to the model as prompt content
inside an explicit delimiter rather than as instructions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


ISSUE_TITLE_PREFIX = "[OpenCode]"

# Commands that may start or continue work on an existing task.
CONTINUE_COMMANDS = ("/continue", "/opencode")

# Bounds on how much untrusted text is handed to the model.
MAX_TASK_CHARACTERS = 8000
MAX_INSTRUCTION_CHARACTERS = 4000
MAX_CONTEXT_COMMENTS = 5
MAX_COMMENT_CHARACTERS = 1500

# The newest issues and comments are the most relevant, so anything
# older is dropped rather than truncated from the middle.
MAX_ISSUE_TITLE_CHARACTERS = 300

_COMMAND_PATTERN = re.compile(
    r"^\s*(?P<command>/continue|/opencode)"
    r"(?:\s+(?P<instruction>.*))?$",
    re.IGNORECASE | re.DOTALL,
)

TRIGGER_ISSUE = "issue"
TRIGGER_CONTINUE = "continue"
TRIGGER_DISPATCH = "dispatch"
TRIGGER_REJECTED = "rejected"


class TriggerRejected(Exception):
    """Raised when an event must not start an agent."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class Trigger:
    """A validated request to run the agent."""

    kind: str
    task: str
    instruction: str = ""
    issue_number: int | None = None
    issue_title: str = ""
    actor: str = ""
    prior_context: str = ""

    @property
    def is_issue_driven(self) -> bool:
        return self.issue_number is not None

    @property
    def summary(self) -> str:
        """A single line safe to print in logs and job summaries."""

        text = self.task or self.instruction

        condensed = " ".join(text.split())

        if len(condensed) > 200:
            condensed = condensed[:197] + "..."

        return condensed or "(no task text)"


def is_authorized_actor(
    actor: str | None,
    owner: str | None,
) -> bool:
    """
    Only the repository owner may start the agent.

    Comparison is case-insensitive because GitHub logins are
    case-insensitive, and a missing value on either side is a denial
    rather than an accidental match.
    """

    if not actor or not owner:
        return False

    return actor.strip().casefold() == owner.strip().casefold()


def has_agent_title_prefix(title: str | None) -> bool:
    """True when an issue title opts into agent execution."""

    if not title:
        return False

    return title.strip().startswith(ISSUE_TITLE_PREFIX)


def strip_title_prefix(title: str | None) -> str:
    """Remove the opt-in prefix, leaving the human task title."""

    if not title:
        return ""

    stripped = title.strip()

    if stripped.startswith(ISSUE_TITLE_PREFIX):
        stripped = stripped[len(ISSUE_TITLE_PREFIX) :]

    return stripped.strip()


def parse_continuation_command(
    body: str | None,
) -> tuple[str, str] | None:
    """
    Parse a follow-up comment.

    Returns ``(command, instruction)`` when the comment starts with a
    supported command, otherwise None. A command with no instruction
    is valid and means "continue the current task".
    """

    if not body:
        return None

    match = _COMMAND_PATTERN.match(body)

    if not match:
        return None

    command = match.group("command").lower()
    instruction = (match.group("instruction") or "").strip()

    return command, instruction


def extract_task_from_issue(
    title: str | None,
    body: str | None,
) -> str:
    """
    Build the initial task from an issue.

    The title supplies the headline and the body supplies the detail.
    When the body adds nothing beyond the title, the title alone is
    used, so the model is never handed a duplicated instruction.
    """

    headline = strip_title_prefix(title)
    details = (body or "").strip()

    if not details:
        return truncate(headline, MAX_TASK_CHARACTERS)

    if details.casefold() == headline.casefold():
        return truncate(headline, MAX_TASK_CHARACTERS)

    return truncate(
        f"{headline}\n\n{details}", MAX_TASK_CHARACTERS
    )


def authorize_issue_event(
    *,
    actor: str | None,
    owner: str | None,
    title: str | None,
    body: str | None,
    issue_number: int | None,
) -> Trigger:
    """
    Validate an ``issues: opened`` event.

    Raises TriggerRejected when the event must not run the agent.
    """

    if not is_authorized_actor(actor, owner):
        raise TriggerRejected(
            f"actor {actor!r} is not the repository owner"
        )

    if not has_agent_title_prefix(title):
        raise TriggerRejected(
            f"issue title must start with {ISSUE_TITLE_PREFIX}"
        )

    task = extract_task_from_issue(title, body)

    if not task.strip():
        raise TriggerRejected("issue contains no task text")

    return Trigger(
        kind=TRIGGER_ISSUE,
        task=task,
        issue_number=issue_number,
        issue_title=strip_title_prefix(title),
        actor=actor or "",
    )


def authorize_comment_event(
    *,
    actor: str | None,
    owner: str | None,
    body: str | None,
    issue_number: int | None,
    original_task: str = "",
    issue_title: str = "",
    prior_comments: list[str] | None = None,
) -> Trigger:
    """
    Validate an ``issue_comment: created`` event.

    The comment must come from the owner and start with a supported
    command. Unrelated comments raise TriggerRejected and the workflow
    job exits without running the agent.
    """

    if not is_authorized_actor(actor, owner):
        raise TriggerRejected(
            f"actor {actor!r} is not the repository owner"
        )

    parsed = parse_continuation_command(body)

    if parsed is None:
        raise TriggerRejected(
            "comment does not start with a supported command: "
            + ", ".join(CONTINUE_COMMANDS)
        )

    _command, instruction = parsed

    return Trigger(
        kind=TRIGGER_CONTINUE,
        task=truncate(original_task, MAX_TASK_CHARACTERS),
        instruction=truncate(
            instruction, MAX_INSTRUCTION_CHARACTERS
        ),
        issue_number=issue_number,
        issue_title=strip_title_prefix(issue_title),
        actor=actor or "",
        prior_context=build_prior_context(prior_comments),
    )


def authorize_dispatch_event(
    *,
    actor: str | None,
    owner: str | None,
    task: str | None,
    issue_number: int | None = None,
) -> Trigger:
    """Validate a ``workflow_dispatch`` event."""

    if not is_authorized_actor(actor, owner):
        raise TriggerRejected(
            f"actor {actor!r} is not the repository owner"
        )

    text = (task or "").strip()

    if not text:
        raise TriggerRejected("dispatch provided no task text")

    return Trigger(
        kind=TRIGGER_DISPATCH,
        task=truncate(text, MAX_TASK_CHARACTERS),
        issue_number=issue_number,
        actor=actor or "",
    )


def build_prior_context(
    comments: list[str] | None,
) -> str:
    """
    Select recent issue comments for continuation context.

    The full history is never passed in. Only the newest few comments
    are included, each bounded in length, because unbounded issue
    history quickly exceeds the model's context without adding signal.
    """

    if not comments:
        return ""

    recent = [comment for comment in comments if comment.strip()]

    if not recent:
        return ""

    selected = recent[-MAX_CONTEXT_COMMENTS:]

    blocks = []

    for index, comment in enumerate(selected, start=1):
        blocks.append(
            f"[comment {index}]\n"
            f"{truncate(comment.strip(), MAX_COMMENT_CHARACTERS)}"
        )

    return "\n\n".join(blocks)


def truncate(text: str | None, limit: int) -> str:
    """Bound a length, marking the cut so the model knows it happened."""

    cleaned = (text or "").strip()

    if len(cleaned) <= limit:
        return cleaned

    return cleaned[:limit] + "\n\n[truncated by the control plane]"
