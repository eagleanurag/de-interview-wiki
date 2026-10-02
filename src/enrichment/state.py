"""
Durable state for one enrichment run.

The GitHub run lost its place twice: once when aggregation was skipped
because some batches failed, and once whenever a batch died mid-way with
no record of which posts it had already done. State kept only in memory
has the same problem locally -- a closed terminal ends the run and the
knowledge of what finished goes with it.

So the orchestrator's view of the batch is written to disk as it changes,
atomically, and read back on the next run. The file answers three
questions that matter: which posts are still owed an attempt, which
failed permanently and why, and what the last run cost.

Kept under ``build/`` with the rest of the pipeline's output, because it
describes a run rather than the repository. Nothing here is a source of
truth about the knowledge: a worker result on disk is that, and this
file is only a record of how the run went.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

#: Bumped when the shape changes, so an old file is rebuilt rather than
#: half-understood.
STATE_VERSION = 1


@dataclass
class PostRecord:
    """One post's journey through the run."""

    post_id: str
    status: str = "pending"
    attempts: int = 0
    retried: int = 0
    kind: str = ""
    error_type: str = ""
    error_message: str = ""
    seconds: float = 0.0
    finished_at: str = ""

    def as_dict(self) -> dict:
        return {
            "post_id": self.post_id,
            "status": self.status,
            "attempts": self.attempts,
            "retried": self.retried,
            "kind": self.kind,
            "error_type": self.error_type,
            "error_message": self.error_message[:600],
            "seconds": round(self.seconds, 2),
            "finished_at": self.finished_at,
        }


@dataclass
class RunState:
    """
    What the orchestrator knows about the whole batch.

    One instance owns the whole run, so there is a single place that
    knows the totals and a single writer of the state file. Workers
    return results; they do not touch this. That is what keeps concurrent
    workers from corrupting shared state -- they never write it.
    """

    posts: dict[str, PostRecord] = field(default_factory=dict)
    started_at: str = ""
    updated_at: str = ""
    runs: int = 0
    enricher_version: str = ""
    version: int = STATE_VERSION

    # -- loading and saving ------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "RunState":
        """
        Read the state, or start a new one.

        A missing, unreadable or wrong-version file starts fresh rather
        than failing the run: losing the record of a previous run costs a
        retry, and refusing to start costs the whole batch.
        """

        target = Path(path)

        if not target.is_file():
            return cls()

        try:
            payload = json.loads(target.read_text(encoding="utf-8"))

        except (OSError, json.JSONDecodeError):
            return cls()

        if not isinstance(payload, dict):
            return cls()

        if payload.get("version") != STATE_VERSION:
            return cls()

        state = cls(
            started_at=str(payload.get("started_at", "")),
            updated_at=str(payload.get("updated_at", "")),
            runs=int(payload.get("runs", 0)),
            enricher_version=str(payload.get("enricher_version", "")),
        )

        for entry in payload.get("posts", []):
            if not isinstance(entry, dict):
                continue

            post_id = str(entry.get("post_id", ""))

            if not post_id:
                continue

            state.posts[post_id] = PostRecord(
                post_id=post_id,
                status=str(entry.get("status", "pending")),
                attempts=int(entry.get("attempts", 0)),
                retried=int(entry.get("retried", 0)),
                kind=str(entry.get("kind", "")),
                error_type=str(entry.get("error_type", "")),
                error_message=str(entry.get("error_message", "")),
                seconds=float(entry.get("seconds", 0.0)),
                finished_at=str(entry.get("finished_at", "")),
            )

        return state

    def save(self, path: str | Path) -> Path:
        """
        Write the state atomically.

        Written to a temporary file in the same directory and moved into
        place, because a half-written state file is worse than none: it
        parses, and it is wrong.
        """

        target = Path(path)

        target.parent.mkdir(parents=True, exist_ok=True)

        self.updated_at = datetime.now(timezone.utc).isoformat()

        payload = {
            "version": STATE_VERSION,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "runs": self.runs,
            "enricher_version": self.enricher_version,
            "posts": [
                record.as_dict()
                for record in sorted(
                    self.posts.values(),
                    key=lambda item: item.post_id,
                )
            ],
        }

        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f"{target.name}.",
            suffix=".tmp",
            delete=False,
        )

        try:
            with handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)

            os.replace(handle.name, target)

        except BaseException:
            # A temporary file left behind is noise; a failed run that
            # cannot clean up after itself is worse.
            try:
                os.unlink(handle.name)

            except OSError:
                pass

            raise

        return target

    # -- bookkeeping --------------------------------------------------

    def begin_run(self, enricher_version: str) -> None:
        """Note that a run has started, and when this one did."""

        now = datetime.now(timezone.utc).isoformat()

        self.runs += 1
        self.enricher_version = enricher_version

        if not self.started_at:
            self.started_at = now

        for record in self.posts.values():
            if record.status == "running":
                # The process that owned this died without saying so.
                # Its post is owed an attempt, not marked as done.
                record.status = "pending"

    def register(self, post_ids: list[str]) -> None:
        """Make sure every post in the batch has a record."""

        for post_id in post_ids:
            self.posts.setdefault(
                post_id, PostRecord(post_id=post_id)
            )

    def record_outcome(self, outcome) -> None:
        """Fold one post's outcome into the state."""

        record = self.posts.setdefault(
            outcome.post_id, PostRecord(post_id=outcome.post_id)
        )

        record.status = outcome.status
        record.attempts = len(outcome.attempts)
        record.retried = outcome.retry_count
        record.seconds = outcome.seconds
        record.finished_at = datetime.now(timezone.utc).isoformat()

        if outcome.attempts and not outcome.succeeded:
            last = outcome.attempts[-1]
            record.kind = last.kind
            record.error_type = last.error_type
            record.error_message = last.error_message

        elif outcome.succeeded:
            record.kind = ""
            record.error_type = ""
            record.error_message = ""

    # -- questions the run needs answered ----------------------------

    def counts(self) -> dict[str, int]:
        """Every status, counted. Absent categories read as zero."""

        tally = {
            "pending": 0,
            "running": 0,
            "enriched": 0,
            "cached": 0,
            "failed": 0,
            "skipped": 0,
        }

        for record in self.posts.values():
            tally[record.status] = tally.get(record.status, 0) + 1

        return tally

    def failures(self) -> list[PostRecord]:
        """Posts that ended the run without a result."""

        return sorted(
            (
                record
                for record in self.posts.values()
                if record.status == "failed"
            ),
            key=lambda item: item.post_id,
        )

    def retried(self) -> list[PostRecord]:
        """Posts that needed more than one attempt, whatever ended."""

        return sorted(
            (
                record
                for record in self.posts.values()
                if record.retried > 0
            ),
            key=lambda item: item.post_id,
        )


__all__ = ["PostRecord", "RunState", "STATE_VERSION"]