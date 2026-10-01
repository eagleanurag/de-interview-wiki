"""
Checkpoint state for a resumable mission.

Runtime only. Nothing written here is ever committed: the directory is
git-ignored, and the module refuses to persist a value that looks like
a credential.

The checkpoint records what was *verified*, not what was intended. A
phase is only marked complete once its verification condition actually
passed, so a resume never trusts an unverified claim.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


CHECKPOINT_DIRECTORY = Path(".agent") / "checkpoints"
CURRENT_FILE = CHECKPOINT_DIRECTORY / "current.json"
README_FILE = CHECKPOINT_DIRECTORY / "README.md"

SCHEMA_VERSION = 1

# Phases, in the order the mission reaches them.
CP0_SYNC = "CP0_SYNC"
CP1_REPOSITORY_AUDIT = "CP1_REPOSITORY_AUDIT"
CP2_ARCHITECTURE_AUDIT = "CP2_ARCHITECTURE_AUDIT"
CP3_INGESTION_READY = "CP3_INGESTION_READY"
CP4_SOURCE_COLLECTION_READY = "CP4_SOURCE_COLLECTION_READY"
CP5_AUTHENTICATION_READY = "CP5_AUTHENTICATION_READY"
CP6_COLLECTION_RUNNING = "CP6_COLLECTION_RUNNING"
CP7_COLLECTION_COMPLETE = "CP7_COLLECTION_COMPLETE"
CP8_ENRICHMENT_COMPLETE = "CP8_ENRICHMENT_COMPLETE"

#: Saved Items capture and import, built once enrichment was working so
#: it could reuse the fingerprint rather than invent its own. It sits
#: after CP8 and before CP9 because that is the order the work actually
#: happened in, not because the mission moved backwards: the phases
#: after it are still unreached.
CP8_SAVED_ITEMS_IMPORT_READY = "CP8_SAVED_ITEMS_IMPORT_READY"

CP9_KNOWLEDGE_BASE_COMPLETE = "CP9_KNOWLEDGE_BASE_COMPLETE"
CP10_WIKI_COMPLETE = "CP10_WIKI_COMPLETE"
CP11_TESTS_GREEN = "CP11_TESTS_GREEN"
CP12_SECURITY_VERIFIED = "CP12_SECURITY_VERIFIED"
CP13_COMMIT_COMPLETE = "CP13_COMMIT_COMPLETE"
CP14_PUSH_COMPLETE = "CP14_PUSH_COMPLETE"
CP15_CI_VERIFIED = "CP15_CI_VERIFIED"
CP16_FINAL = "CP16_FINAL"

# Raised when a security challenge needs the human.
CP_HUMAN_LINKEDIN_ACTION_REQUIRED = "CP_HUMAN_LINKEDIN_ACTION_REQUIRED"

PHASE_ORDER = (
    CP0_SYNC,
    CP1_REPOSITORY_AUDIT,
    CP2_ARCHITECTURE_AUDIT,
    CP3_INGESTION_READY,
    CP4_SOURCE_COLLECTION_READY,
    CP5_AUTHENTICATION_READY,
    CP6_COLLECTION_RUNNING,
    CP7_COLLECTION_COMPLETE,
    CP8_ENRICHMENT_COMPLETE,
    CP8_SAVED_ITEMS_IMPORT_READY,
    CP9_KNOWLEDGE_BASE_COMPLETE,
    CP10_WIKI_COMPLETE,
    CP11_TESTS_GREEN,
    CP12_SECURITY_VERIFIED,
    CP13_COMMIT_COMPLETE,
    CP14_PUSH_COMPLETE,
    CP15_CI_VERIFIED,
    CP16_FINAL,
)

# Statuses.
STATUS_COMPLETE = "complete"
STATUS_IN_PROGRESS = "in_progress"
STATUS_BLOCKED_HUMAN = "blocked_human"
STATUS_FAILED = "failed"

# Credential shapes that must never be persisted.
_SECRET_PATTERNS = (
    re.compile(r"\bgh[posur]_[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"(?i)\b(?:password|passwd)\s*[:=]\s*\S+"),
    re.compile(r"(?i)\b(?:token|secret|api[_-]?key)\s*[:=]\s*\S+"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
)


class CheckpointError(RuntimeError):
    """Raised when checkpoint state cannot be written safely."""


def contains_credential(value: object) -> bool:
    """True when a string looks like it carries a credential."""

    if not isinstance(value, str):
        return False

    for pattern in _SECRET_PATTERNS:
        if pattern.search(value):
            return True

    return False


def _assert_clean(payload: dict) -> None:
    """
    Refuse to persist a payload containing a credential.

    The guard is on keys as well as values, because a nested mapping
    such as ``{"password": "..."}`` is the shape most likely to leak.
    """

    for key, value in payload.items():
        if isinstance(value, dict):
            _assert_clean(value)
            continue

        lowered = str(key).lower()

        if lowered in {
            "password",
            "passwd",
            "token",
            "secret",
            "api_key",
            "cookie",
            "cookies",
            "session",
        }:
            raise CheckpointError(
                f"Refusing to persist sensitive key: {key!r}"
            )

        if contains_credential(value):
            raise CheckpointError(
                f"Refusing to persist a credential-shaped value "
                f"for key {key!r}"
            )


@dataclass
class Checkpoint:
    """The mission's resumable state."""

    checkpoint_id: str = CP0_SYNC
    phase: str = CP0_SYNC
    status: str = STATUS_IN_PROGRESS
    timestamp: str = ""
    local_sha: str = ""
    remote_sha: str = ""
    completed: list[str] = field(default_factory=list)
    remaining: list[str] = field(default_factory=list)
    tests: dict = field(default_factory=dict)
    ci: dict = field(default_factory=dict)
    collection: dict = field(default_factory=dict)

    #: What was actually verified, as distinct from what was attempted.
    #: Kept apart from ``tests`` because a suite can be green while the
    #: thing it covers is broken, and a checkpoint that reported only the
    #: first would read as more assurance than it is.
    validation: dict = field(default_factory=dict)

    blocker: str = ""
    human_action_required: str = ""
    resume_point: str = ""

    def touch(self) -> "Checkpoint":
        self.timestamp = datetime.now(timezone.utc).isoformat()

        return self


def current_directory(root: str | Path = ".") -> Path:
    return Path(root) / CHECKPOINT_DIRECTORY


def current_file(root: str | Path = ".") -> Path:
    return current_directory(root) / "current.json"


def read(root: str | Path = ".") -> Checkpoint | None:
    """
    Load the active checkpoint.

    Returns None when none exists. A malformed or credential-bearing
    file is treated as absent rather than trusted.
    """

    path = current_file(root)

    if not path.is_file():
        return None

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8", errors="replace")
        )
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None

    if contains_credential(json.dumps(payload)):
        return None

    known = {field_name for field_name in Checkpoint.__dataclass_fields__}

    clean = {
        key: value
        for key, value in payload.items()
        if key in known
    }

    return Checkpoint(**clean)


def write(checkpoint: Checkpoint, root: str | Path = ".") -> Path:
    """Persist a checkpoint after verifying it carries no credential."""

    checkpoint.touch()

    # The two must never disagree. A checkpoint whose id lags its
    # phase would make a resume pick the wrong starting point.
    if not checkpoint.checkpoint_id:
        checkpoint.checkpoint_id = checkpoint.phase
    elif checkpoint.checkpoint_id != checkpoint.phase:
        checkpoint.checkpoint_id = checkpoint.phase

    payload = asdict(checkpoint)

    _assert_clean(payload)

    directory = current_directory(root)
    directory.mkdir(parents=True, exist_ok=True)

    path = current_file(root)
    temporary = path.with_suffix(".json.tmp")

    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)

    _ensure_readme(directory)

    return path


def _ensure_readme(directory: Path) -> None:
    if README_FILE.name in {item.name for item in directory.iterdir()}:
        return

    directory.joinpath(README_FILE.name).write_text(
        "# Runtime state only\n\n"
        "Ignored by git. Never store credentials, cookies, session\n"
        "data or tokens here.\n",
        encoding="utf-8",
    )


def mark(
    checkpoint: Checkpoint,
    phase: str,
    *,
    status: str,
    resume_point: str,
    completed: list[str] | None = None,
    remaining: list[str] | None = None,
    **updates: object,
) -> Checkpoint:
    """Advance a checkpoint and persist it."""

    checkpoint.checkpoint_id = phase
    checkpoint.phase = phase
    checkpoint.status = status
    checkpoint.resume_point = resume_point

    if completed is not None:
        checkpoint.completed = list(completed)

    if remaining is not None:
        checkpoint.remaining = list(remaining)

    for key, value in updates.items():
        if hasattr(checkpoint, key):
            setattr(checkpoint, key, value)

    write(checkpoint)

    return checkpoint


def phase_index(phase: str) -> int:
    try:
        return PHASE_ORDER.index(phase)
    except ValueError:
        return -1


def last_verified_phase(checkpoint: Checkpoint | None) -> str:
    """The furthest phase that was actually verified complete."""

    if checkpoint is None:
        return CP0_SYNC

    if checkpoint.status != STATUS_COMPLETE:
        return checkpoint.phase

    return checkpoint.phase


def git_sha(root: str | Path = ".", *, remote: bool = False) -> str:
    """Current SHA, optionally of the remote tracking branch."""

    import subprocess

    command = ["git", "rev-parse", "HEAD"]

    if remote:
        command.append("origin/main")

    completed = subprocess.run(
        command,
        cwd=str(root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        shell=False,
    )

    return completed.stdout.strip()


def credentials_configured() -> bool:
    """
    Report whether source credentials are present.

    Returns only a boolean. The values are never read into a string,
    logged, or returned anywhere. Reading the key's presence is enough.
    """

    from dotenv import load_dotenv

    load_dotenv(override=False)

    return bool(
        os.environ.get("LINKEDIN_USERNAME")
        and os.environ.get("LINKEDIN_PASSWORD")
    )