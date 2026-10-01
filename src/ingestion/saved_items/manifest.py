"""
The Saved Items manifest.

The manifest is the answer to "what did the last run do, and what is
still outstanding". It is written after every change rather than at the
end, because a run that is interrupted half way has still learned
something, and throwing that away would mean redoing the work and
re-reading the user's files next time.

Every write is atomic. A manifest truncated by a crash would be
unreadable, and an unreadable manifest cannot be resumed from: the run
after it would re-import every item and create a second post for each
one. So the file is written to a temporary name and moved onto its
target, which either happens completely or not at all.

Nothing in the manifest is a credential. It holds URLs, dates, titles,
notes and the identifiers derived from them, which is exactly what the
user exported.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.ingestion.collect import replace_file
from src.ingestion.saved_items.model import SavedItem, SavedItemState


#: Where the manifest lives inside the capture drop zone.
MANIFEST_FILE = "saved-items-manifest.json"

#: Bumped when the stored shape changes, so an older manifest is read
#: as what it was rather than as something it is not.
MANIFEST_VERSION = 1

#: Distinguishes "leave this field alone" from "clear this field".
#: Passing None has to be able to mean clearing, because a failure
#: reason that stopped being true has to be removed rather than kept.
_UNCHANGED = object()


def manifest_path(root: str | Path) -> Path:
    """Where the manifest for a capture drop zone is written."""

    return Path(root) / MANIFEST_FILE


@dataclass
class SavedItemsReport:
    """
    What a Saved Items run did.

    The counts are answers a user can act on rather than internal
    bookkeeping. ``metadata_only`` is the one that matters most: it is
    how many saved links are known but not yet backed by content, which
    is the backlog, not a failure.
    """

    discovered: int = 0
    new: int = 0
    duplicates: int = 0

    #: Every item the manifest knows about, across every run. The
    #: counts below are about this run's reading of the list; this is
    #: about the backlog as a whole.
    known: int = 0

    with_content: int = 0
    metadata_only: int = 0

    imported: int = 0
    enriched: int = 0
    failed: int = 0
    pending: int = 0

    #: Items whose content changed since the last run and were re-read.
    changed: int = 0

    #: Items whose content is byte-for-byte what it was.
    unchanged: int = 0

    #: Rows and files that could not be read, with the reason.
    issues: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "discovered": self.discovered,
            "new": self.new,
            "duplicates": self.duplicates,
            "known": self.known,
            "with_content": self.with_content,
            "metadata_only": self.metadata_only,
            "imported": self.imported,
            "enriched": self.enriched,
            "failed": self.failed,
            "pending": self.pending,
            "changed": self.changed,
            "unchanged": self.unchanged,
            "issues": list(self.issues),
        }

    def render(self, *, title: str = "Saved Items Report") -> str:
        """The report as it is printed and written beside the manifest."""
        rows = (
            ("Discovered", self.discovered),
            ("New", self.new),
            ("Duplicates", self.duplicates),
            ("Known items", self.known),
            ("With content", self.with_content),
            ("Metadata only", self.metadata_only),
            ("Imported", self.imported),
            ("Enriched", self.enriched),
            ("Failed", self.failed),
            ("Pending", self.pending),
            ("Content changed", self.changed),
            ("Unchanged", self.unchanged),
        )

        width = max(len(label) for label, _ in rows)

        lines = [title, "-" * len(title)]

        for label, value in rows:
            lines.append(f"{label.ljust(width)} : {value}")

        if self.discovered and self.discovered != self.new + self.duplicates:
            # The first three rows are meant to add up. If they do not,
            # a row was read that produced no item, and the issues
            # below say which. Saying so beats leaving the reader to
            # wonder whether the numbers are wrong.
            lines.append("")
            lines.append(
                f"({self.discovered - self.new - self.duplicates} "
                "row(s) read but not usable; see issues)"
            )

        if self.issues:
            lines.append("")
            lines.append(f"Issues ({len(self.issues)})")
            lines.append("-" * 8)

            for issue in self.issues:
                lines.append(f"  {issue}")

        return "\n".join(lines)

    def line(self) -> str:
        """One line, for a log or a status line."""
        return (
            f"discovered={self.discovered} new={self.new} "
            f"duplicates={self.duplicates} with_content={self.with_content} "
            f"metadata_only={self.metadata_only} imported={self.imported} "
            f"enriched={self.enriched} failed={self.failed} "
            f"pending={self.pending}"
        )


class SavedItemsManifest:
    """
    The persistent record of every Saved Item seen.

    Keyed by source id, which is derived from the canonical URL, so the
    same saved post is one record however many times it is exported and
    however much its URL varies between exports.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        items: dict[str, SavedItem] | None = None,
    ) -> None:
        self.path = Path(path)
        self.items: dict[str, SavedItem] = items or {}
        self.updated_at: str = ""

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "SavedItemsManifest":
        """
        Load a manifest, or start an empty one.

        An unreadable manifest is reported rather than silently replaced
        with an empty one, because a manifest that cannot be read means
        every item in it would be imported a second time. The caller is
        told, and the existing file is left where it is.
        """
        target = Path(path)

        manifest = cls(target)

        if not target.is_file():
            return manifest

        try:
            payload = json.loads(
                target.read_text(encoding="utf-8", errors="replace")
            )

        except json.JSONDecodeError as exc:
            raise ManifestUnreadable(
                f"{target.name} is not valid JSON ({exc.msg} at line "
                f"{exc.lineno}); move it aside to start a new manifest"
            ) from exc

        except OSError as exc:
            raise ManifestUnreadable(
                f"{target.name} could not be read: {exc}"
            ) from exc

        if not isinstance(payload, dict):
            raise ManifestUnreadable(
                f"{target.name} does not contain a manifest object"
            )

        version = payload.get("manifest_version")

        if version != MANIFEST_VERSION:
            raise ManifestUnreadable(
                f"{target.name} was written by a different version "
                f"(found {version!r}, expected {MANIFEST_VERSION}); "
                "move it aside to start a new manifest"
            )

        stored = payload.get("items")

        if isinstance(stored, list):
            for record in stored:
                if isinstance(record, dict):
                    item = SavedItem.from_dict(record)

                    if item.source_id:
                        manifest.items[item.source_id] = item

        manifest.updated_at = str(payload.get("updated_at") or "")

        return manifest

    def save(self) -> Path:
        """Write the manifest, atomically."""
        self.updated_at = datetime.now(timezone.utc).isoformat()

        self.path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "manifest_version": MANIFEST_VERSION,
            "updated_at": self.updated_at,
            "items": [
                self.items[key].as_dict()
                for key in sorted(self.items)
            ],
        }

        temporary = self.path.with_suffix(".json.tmp")

        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        replace_file(temporary, self.path)

        return self.path

    # -----------------------------------------------------------------
    # Records
    # -----------------------------------------------------------------

    def upsert(self, item: SavedItem) -> str:
        """
        Record a freshly discovered item.

        Returns ``"new"`` when the item was not known, and
        ``"duplicate"`` when it was, which is what the report counts.
        A duplicate is not discarded: a later export may carry a saved
        date or a note the earlier one did not, and the content may
        have been captured since.
        """
        existing = self.items.get(item.source_id)

        if existing is None:
            item.touch()
            self.items[item.source_id] = item

            return "new"

        existing.merge(item)

        return "duplicate"

    def get(self, source_id: str) -> SavedItem | None:
        return self.items.get(source_id)

    def mark(self, source_id: str, **changes: object) -> SavedItem | None:
        """
        Update a stored item and write the manifest.

        Writing on every change is what makes a crash survivable: the
        record of what has already been imported is on disk before the
        next item is attempted, so a run that dies mid-way has already
        recorded the items before the failure.

        A field left out is untouched, and a field passed as None is
        cleared. Those are different intentions and a caller has to be
        able to express both: a failure reason that is no longer true
        must be removed, not preserved because nothing replaced it.
        """
        item = self.items.get(source_id)

        if item is None:
            return None

        for name, value in changes.items():
            if value is _UNCHANGED:
                continue

            if not hasattr(item, name):
                raise AttributeError(
                    f"SavedItem has no field {name!r}"
                )

            setattr(item, name, value)

        item.touch()

        self.save()

        return item

    def mark_imported(
        self,
        source_id: str,
        post_id: str,
    ) -> SavedItem | None:
        return self.mark(
            source_id,
            state=SavedItemState.IMPORTED,
            post_id=post_id,
            failure_reason=None,
        )

    def mark_failed(self, source_id: str, reason: str) -> SavedItem | None:
        return self.mark(
            source_id,
            state=SavedItemState.FAILED,
            failure_reason=reason,
        )

    def mark_enriched(
        self,
        source_id: str,
        post_id: str,
    ) -> SavedItem | None:
        return self.mark(
            source_id,
            state=SavedItemState.ENRICHED,
            post_id=post_id,
            failure_reason=None,
        )

    # -----------------------------------------------------------------
    # Reporting
    # -----------------------------------------------------------------

    def report(
        self,
        *,
        discovered: int | None = None,
        new: int | None = None,
        duplicates: int | None = None,
        issues: list[str] | None = None,
    ) -> SavedItemsReport:
        """
        Count the run.

        The content and outcome counts come from the stored items rather
        than from this run's activity, so the report describes the state
        of the saved list as a whole and not only what changed. ``enriched``
        is likewise derived from the stored state, because enrichment
        happens in a later stage and the manifest should not claim work
        it did not do.
        """
        stored = list(self.items.values())

        report = SavedItemsReport(
            discovered=len(stored) if discovered is None else discovered,
            new=sum(
                1 for item in stored if item.state is SavedItemState.PENDING
            )
            if new is None
            else new,
            duplicates=len(stored) if duplicates is None else duplicates,
            known=len(stored),
            with_content=sum(1 for item in stored if item.has_content),
            metadata_only=sum(1 for item in stored if item.is_metadata_only),
            imported=sum(
                1
                for item in stored
                if item.state.rank >= SavedItemState.IMPORTED.rank
            ),
            enriched=sum(
                1 for item in stored if item.state is SavedItemState.ENRICHED
            ),
            failed=sum(1 for item in stored if item.state is SavedItemState.FAILED),
            pending=sum(1 for item in stored if item.state is SavedItemState.PENDING),
            issues=list(issues or []),
        )

        return report

    def outstanding(self) -> list[SavedItem]:
        """Items a later run should still try."""
        return sorted(
            (
                item
                for item in self.items.values()
                if not item.state.is_terminal
            ),
            key=lambda item: item.source_id,
        )

    def __len__(self) -> int:
        return len(self.items)


class ManifestUnreadable(RuntimeError):
    """
    Raised when a manifest exists but cannot be trusted.

    A separate type from a validation issue because the difference
    matters: an unreadable manifest is a stop, not a warning. Continuing
    would create a second post for every item it already recorded.
    """
