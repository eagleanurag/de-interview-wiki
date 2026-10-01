"""
Collecting posts from a source into the repository.

The collector owns everything that is the same regardless of source:
bounds, deduplication, persistence, checkpointing, and stopping. A
source only has to discover and normalize, so adding a new source never
touches this file.

Deduplication is keyed on the source's own post identifier, and every
run records a stopping reason, so a partial collection is never
mistaken for a complete one.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from src.ingestion.checkpoints import (
    CP_HUMAN_LINKEDIN_ACTION_REQUIRED,
    STATUS_BLOCKED_HUMAN,
    Checkpoint,
    current_directory,
)
from src.ingestion.checkpoints import write as write_checkpoint
from src.ingestion.post_document import PostDocument
from src.ingestion.sources.base import (
    CollectedPost,
    CollectionState,
    CollectionStopped,
    SecurityChallenge,
    Source,
    StopReason,
)


COLLECTION_STATE_FILE = "collection.json"

# A source identifier may be a URN, a numeric id, or a slug.
_ID_CLEANER = re.compile(r"[^a-z0-9]+")

MAX_POST_ID_LENGTH = 80


class CollectionError(RuntimeError):
    """Raised when a collection cannot proceed."""


@dataclass
class CollectionLimits:
    """Bounds applied to one run."""

    max_posts: int | None = None
    since: str | None = None
    until: str | None = None
    scroll_limit: int | None = None
    dry_run: bool = False


@dataclass
class CollectionReport:
    """What a run did, for logging and checkpointing."""

    state: CollectionState = field(default_factory=CollectionState)
    imported: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    stopped_because: str = ""
    message: str = ""
    security_challenge: str = ""

    @property
    def succeeded(self) -> bool:
        return not self.failed and not self.security_challenge

    def summary(self) -> str:
        return (
            f"{self.stopped_because or 'stopped'}: "
            f"discovered={self.state.discovered} "
            f"persisted={self.state.persisted} "
            f"duplicates={self.state.duplicates} "
            f"failed={self.state.failed}"
        )


def state_path(root: str | Path = ".") -> Path:
    return current_directory(root) / COLLECTION_STATE_FILE


def read_state(root: str | Path = ".") -> CollectionState | None:
    """Load the recorded collection state, if any."""

    path = state_path(root)

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

    return CollectionState.from_dict(payload)


def write_state(
    state: CollectionState,
    root: str | Path = ".",
) -> Path:
    """Persist collection state so a later run can resume."""

    state.updated_at = _now()

    path = state_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(".json.tmp")

    temporary.write_text(
        json.dumps(state.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)

    return path


def post_id_for(source_post_id: str) -> str:
    """
    Map a source identifier to a repository post ID.

    Deterministic, so the same source post always resolves to the same
    directory and a repeated run can never create a second copy.
    """

    cleaned = (
        _ID_CLEANER.sub("-", source_post_id.strip().lower()).strip("-")
    )

    if not cleaned:
        raise CollectionError(
            "A source post identifier could not be normalized."
        )

    return cleaned[:MAX_POST_ID_LENGTH].rstrip("-")


class Collector:
    """
    Drives a source into ``data/posts/``.

    Safe to run repeatedly: an already-persisted post is skipped as a
    duplicate, and existing enrichment is never overwritten.
    """

    def __init__(
        self,
        source: Source,
        *,
        root: str | Path,
        limits: CollectionLimits | None = None,
        checkpoint: Checkpoint | None = None,
        progress: Callable[[str], None] | None = None,
        repository_root: str | Path = ".",
    ) -> None:
        self.source = source
        self.root = Path(root)
        self.limits = limits or CollectionLimits()
        self.checkpoint = checkpoint
        self.progress = progress or (lambda message: None)
        self.repository_root = Path(repository_root)

        self._state = CollectionState()

    # -----------------------------------------------------------------
    # What already exists
    # -----------------------------------------------------------------

    def existing_post_ids(self) -> set[str]:
        """Post IDs already present on disk."""

        if not self.root.exists():
            return set()

        return {
            path.name
            for path in self.root.iterdir()
            if path.is_dir() and (path / "post.json").is_file()
        }

    # -----------------------------------------------------------------
    # The run
    # -----------------------------------------------------------------

    def run(self, *, resume: bool = False) -> CollectionReport:
        """
        Collect from the source, bounded and resumable.

        Never raises for an ordinary stop: the reason is returned in the
        report. A security challenge is also reported rather than
        raised, so the state is always written before the process stops.
        """

        report = CollectionReport()

        previous = read_state(self.repository_root) if resume else None

        if previous is not None:
            self._state.started_at = previous.started_at

            if previous.last_post_id:
                try:
                    self.source.resume(
                        {post_id_for(previous.last_post_id)}
                    )
                except CollectionError:
                    pass

            self.progress(
                f"Resuming. {previous.persisted} post(s) previously "
                f"persisted."
            )

        known = self.existing_post_ids()

        # The bound is enforced here as well as in the source. A source
        # is third-party code; the collector must not depend on it
        # honouring a limit in order to stay bounded.
        budget = self.limits.max_posts

        reached_budget = False

        try:
            for collected in self.source.discover(
                max_posts=self.limits.max_posts,
                since=self.limits.since,
                until=self.limits.until,
                scroll_limit=self.limits.scroll_limit,
            ):
                if budget is not None and self._state.persisted >= budget:
                    reached_budget = True
                    break

                self._handle(collected, report, known)

        except SecurityChallenge as exc:
            report.stopped_because = StopReason.SECURITY_CHALLENGE.value
            report.message = str(exc)
            report.security_challenge = getattr(exc, "kind", "unknown")

        except CollectionStopped as exc:
            report.stopped_because = exc.reason.value
            report.message = str(exc)

        else:
            # A budget the collector enforced takes precedence over the
            # source's own end-of-feed signal, because the source was
            # stopped by the collector rather than running out.
            if reached_budget:
                report.stopped_because = StopReason.MAX_POSTS.value
                report.message = (
                    f"Reached the configured limit of {budget} posts."
                )
            else:
                report.stopped_because = StopReason.EXHAUSTED.value
                report.message = (
                    report.message
                    or "The source reported no more content."
                )

        self.progress(report.message)

        self._finish(report, previous)

        return report

    # -----------------------------------------------------------------
    # One post
    # -----------------------------------------------------------------

    def _handle(
        self,
        collected: CollectedPost,
        report: CollectionReport,
        known: set[str],
    ) -> None:
        """Deduplicate and persist one collected post."""

        self._state.discovered += 1

        try:
            post_id = post_id_for(collected.source_post_id)
        except CollectionError:
            self._state.failed += 1
            report.failed.append(collected.source_post_id)
            self.progress(
                f"Skipped a post with no usable identifier: "
                f"{collected.source_post_id!r}"
            )
            return

        if post_id in known:
            # Already collected. Refresh in place rather than creating a
            # second directory, so a repeated run is safe and still
            # picks up anything the source has since published.
            self._state.duplicates += 1
            report.duplicates.append(post_id)

            changed = self._refresh(
                post_id,
                collected,
                report,
            )

            self.progress(
                f"Duplicate {post_id}"
                + (" (refreshed)" if changed else " (unchanged)")
            )

            return

        if self.limits.dry_run:
            self._state.persisted += 1
            report.imported.append(post_id)
            self._state.last_post_id = post_id
            self._state.last_url = collected.url or ""
            self.progress(f"Would import: {post_id}")
            return

        try:
            # Normalization and the write share one failure boundary:
            # a source that yields something unnormalizable must cost
            # one post, not the whole run.
            document = collected.to_document(
                post_id=post_id,
                platform=self.source.platform,
                captured_at=_now(),
            )

            directory = self._persist(document)
        except Exception as exc:  # noqa: BLE001
            self._state.failed += 1
            report.failed.append(post_id)
            self.progress(f"Failed to import {post_id}: {exc}")
            return

        known.add(post_id)
        self._state.persisted += 1
        report.imported.append(post_id)
        self._state.last_post_id = post_id
        self._state.last_url = collected.url or ""

        self.progress(f"Imported {post_id} into {directory.name}")

        # Written after every post so a crash loses at most one.
        write_state(self._state, self.repository_root)

    def _refresh(
        self,
        post_id: str,
        collected: CollectedPost,
        report: CollectionReport,
    ) -> bool:
        """
        Update an already-collected post in place.

        Enrichment is preserved, because regenerating it would cost a
        model call and could change answers that were already
        reviewed. Only the captured text and provenance are refreshed.

        Returns whether anything changed.
        """

        directory = self.root / post_id
        target = directory / "post.json"

        if not target.is_file():
            return False

        existing = self._read_existing(target)

        if existing is None:
            return False

        text = collected.text.strip()

        if not text:
            return False

        changed = False

        if existing.original_text != text:
            existing.set_original_text(text)
            changed = True

        before = dict(existing.source)

        existing.merge_source(
            platform=self.source.platform,
            url=collected.url,
            author=collected.author,
            captured_at=collected.extra.get("collected_at"),
        )

        if existing.source != before:
            changed = True

        if not changed:
            return False

        try:
            existing.save(directory)
        except Exception as exc:  # noqa: BLE001
            self._state.failed += 1
            report.failed.append(post_id)
            self.progress(f"Failed to refresh {post_id}: {exc}")
            return False

        self._state.last_post_id = post_id
        self._state.last_url = collected.url or ""

        write_state(self._state, self.repository_root)

        return True

    def _persist(self, document: PostDocument) -> Path:
        """
        Save one post, preserving enrichment already in place.

        Only the post's own directory is written, so a source cannot
        place a file anywhere else in the repository.
        """

        directory = self.root / document.post_id
        directory.mkdir(parents=True, exist_ok=True)

        target = directory / "post.json"

        if target.is_file():
            existing = self._read_existing(target)

            if existing is not None and existing.has_enrichment():
                # A refresh must never discard enrichment that the
                # worker already produced, so the existing document is
                # kept and only provenance and text are refreshed.
                incoming = document.data.get("source", {})

                existing.merge_source(
                    platform=incoming.get("platform"),
                    url=incoming.get("url"),
                    author=incoming.get("author"),
                    captured_at=incoming.get("captured_at"),
                )

                if document.data.get("original_text"):
                    existing.set_original_text(
                        document.data["original_text"]
                    )

                document = existing

        document.save(directory)

        return directory

    @staticmethod
    def _read_existing(path: Path) -> PostDocument | None:
        """Read an existing post, tolerating a malformed file."""

        try:
            return PostDocument.load_file(path)
        except Exception:  # noqa: BLE001
            # A corrupt post.json is replaced rather than aborting the
            # run; validation reports it separately.
            return None

    # -----------------------------------------------------------------
    # Finishing
    # -----------------------------------------------------------------

    def _finish(
        self,
        report: CollectionReport,
        previous: CollectionState | None,
    ) -> None:
        """Fold in prior totals, persist, and update the checkpoint."""

        self._state.stopped_because = report.stopped_because
        self._state.updated_at = _now()

        if not self._state.started_at:
            self._state.started_at = self._state.updated_at

        if previous is not None:
            self._state = _merge(previous, self._state)

        report.state = self._state

        write_state(self._state, self.repository_root)

        self._update_checkpoint(report)

        self.progress(report.summary())

    def _update_checkpoint(self, report: CollectionReport) -> None:
        """Fold collection results into the mission checkpoint."""

        if self.checkpoint is None:
            return

        self.checkpoint.collection = report.state.to_dict()

        if report.security_challenge:
            self.checkpoint.checkpoint_id = (
                CP_HUMAN_LINKEDIN_ACTION_REQUIRED
            )
            self.checkpoint.phase = CP_HUMAN_LINKEDIN_ACTION_REQUIRED
            self.checkpoint.status = STATUS_BLOCKED_HUMAN
            self.checkpoint.blocker = report.message
            self.checkpoint.human_action_required = (
                f"LinkedIn presented a {report.security_challenge} "
                f"challenge. Complete it in the browser window the "
                f"collector leaves open, then resume."
            )
            self.checkpoint.resume_point = (
                "python -m src.ingestion.collect_cli run "
                "--source linkedin --resume"
            )

        write_checkpoint(self.checkpoint, self.repository_root)


def _merge(
    previous: CollectionState,
    current: CollectionState,
) -> CollectionState:
    """Combine a resumed run with the recorded totals."""

    return CollectionState(
        discovered=previous.discovered + current.discovered,
        persisted=previous.persisted + current.persisted,
        duplicates=previous.duplicates + current.duplicates,
        failed=previous.failed + current.failed,
        last_post_id=current.last_post_id or previous.last_post_id,
        last_url=current.last_url or previous.last_url,
        stopped_because=current.stopped_because,
        started_at=previous.started_at or current.started_at,
        updated_at=current.updated_at,
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()