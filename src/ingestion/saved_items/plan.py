"""
Read the Saved Items inbox and say what would happen.

One reading of the inbox, three answers from it. A dry run, a status
report and a validation pass all need the same facts — which items have
a capture, which captures nothing claims, which rows cannot be read —
and working them out three ways would let the preview and the import
disagree, which is the one thing a preview must never do.

So the inbox is read once into a :class:`Plan`, and everything else is a
view of it. Nothing here writes. A plan is an observation, and an
observation that edited the inbox would be a different thing entirely.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from src.ingestion.saved_items import capture as _capture
from src.ingestion.saved_items import diagnostics as diag
from src.ingestion.saved_items.bundles import (
    CAPTURE_QUALITIES,
    QUALITY_DOCUMENT_ONLY,
    QUALITY_IMAGE_ONLY,
    QUALITY_METADATA_ONLY,
    BundleError,
    BundleIndex,
    CapturedContent,
    content_digest,
    read_bundle,
)
from src.ingestion.saved_items.capture import find_capture_file
from src.ingestion.saved_items.manifest import (
    ManifestUnreadable,
    SavedItemsManifest,
)
from src.ingestion.saved_items.model import SavedItem, SavedItemState
from src.ingestion.saved_items.readers import (
    ManifestError,
    ManifestRead,
    read_manifest,
)
from src.ingestion.saved_items.urls import (
    SavedItemUrlError,
    normalize_linkedin_url,
)


#: What a run will do with one item.
#:
#: Named for the answer rather than the mechanism, because that is what
#: the user is deciding about.
NEW = "NEW"
UNCHANGED = "UNCHANGED"
CHANGED = "CHANGED"
DUPLICATE = "DUPLICATE"
MISSING_CAPTURE = "MISSING_CAPTURE"
INVALID = "INVALID"
FAILED = "FAILED"

#: Every outcome, in the order a report should show them.
OUTCOMES = (
    NEW,
    CHANGED,
    UNCHANGED,
    DUPLICATE,
    MISSING_CAPTURE,
    INVALID,
    FAILED,
)

#: Suffixes counted for the content-type breakdown.
_TEXT_SUFFIXES = frozenset({".md", ".txt"})
_HTML_SUFFIXES = frozenset({".html", ".htm"})
_DOCUMENT_SUFFIXES = frozenset({".pdf"})
_IMAGE_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
)

#: The file that names which item a capture belongs to. Counted
#: separately from the content, because "3 captures" and "3 of them are
#: Markdown" are different questions and mixing them answers neither.
_CAPTURE_FILES = frozenset(
    {name.lower() for name in _capture.CAPTURE_NAMES}
)


@dataclass
class ItemPlan:
    """What will happen to one saved item."""

    outcome: str
    item: SavedItem | None = None
    source_id: str = ""
    url: str = ""
    bundle: str = ""
    quality: str = QUALITY_METADATA_ONLY
    reason: str = ""
    changed_files: list[str] = field(default_factory=list)

    @property
    def is_importable(self) -> bool:
        return self.outcome in {NEW, CHANGED}

    @property
    def is_error(self) -> bool:
        return self.outcome in {INVALID, FAILED}


@dataclass
class Plan:
    """
    Everything a run needs to know, read without writing.

    Built from the manifest on disk plus whatever is in the inbox, so a
    plan against an unchanged inbox is the same plan twice.
    """

    root: Path
    manifest: SavedItemsManifest
    items: list[ItemPlan] = field(default_factory=list)
    problems: list[diag.Diagnostic] = field(default_factory=list)
    orphans: list[str] = field(default_factory=list)
    content_types: Counter = field(default_factory=Counter)

    #: The capture index, once the inbox has been read. Kept so a
    #: manifest-named bundle can be claimed the same way a matched one
    #: is, rather than through a second path that does not record the
    #: claim.
    _index: BundleIndex | None = None

    # -----------------------------------------------------------------
    # Counts
    # -----------------------------------------------------------------

    def counts(self) -> dict[str, int]:
        """How many items are in each outcome."""
        found = Counter(entry.outcome for entry in self.items)

        return {outcome: found.get(outcome, 0) for outcome in OUTCOMES}

    @property
    def errors(self) -> list[diag.Diagnostic]:
        return [entry for entry in self.problems if entry.is_error]

    @property
    def warnings(self) -> list[diag.Diagnostic]:
        return [entry for entry in self.problems if not entry.is_error]

    def by_quality(self) -> dict[str, int]:
        found = Counter(entry.quality for entry in self.items)

        return {
            quality: found.get(quality, 0) for quality in CAPTURE_QUALITIES
        }

    def manifest_items(self) -> int:
        return len(self.manifest)

    def with_capture(self) -> int:
        return sum(1 for entry in self.items if entry.item is not None
                   and entry.item.has_content)

    def metadata_only(self) -> int:
        return sum(1 for entry in self.items if not entry.is_importable
                   and entry.outcome == MISSING_CAPTURE)

    def already_imported(self) -> int:
        return sum(
            1
            for entry in self.items
            if entry.item is not None
            and entry.item.state
            in {SavedItemState.IMPORTED, SavedItemState.ENRICHED}
        )

    def importable(self) -> int:
        return sum(1 for entry in self.items if entry.is_importable)

    # -----------------------------------------------------------------
    # Reporting
    # -----------------------------------------------------------------

    #: How many entries of each kind a rendered plan prints.
    #:
    #: A saved list runs to hundreds of links and most of them never get
    #: a capture, so the honest list of what is outstanding can be a
    #: thousand lines. Printing a thousand lines is not something anyone
    #: reads, and a report nobody reads is a report nobody acts on. The
    #: counts above are always exact; only the listing is bounded, and
    #: ``--json`` returns all of it.
    MAX_LISTED = 25

    def render(self, *, title: str = "Saved Items Plan") -> str:
        """
        The plan, grouped by what will happen.

        Grouped rather than listed one per line: a thousand saved items
        is a backlog, not a document to read, and the reader wants the
        shape of it plus whatever needs a decision.
        """
        lines = [title, "-" * len(title), ""]

        for outcome in OUTCOMES:
            group = [entry for entry in self.items
                     if entry.outcome == outcome]

            if not group:
                continue

            lines.append(f"{outcome} ({len(group)})")
            lines.append("-" * (len(outcome) + len(str(len(group)))))

            lines.extend(_listing(_worst_first(group)))

            lines.append("")

        if self.orphans:
            lines.append(f"ORPHAN CAPTURE ({len(self.orphans)})")
            lines.append("-" * 26)

            lines.extend(
                f"  {folder}" for folder in self.orphans[: self.MAX_LISTED]
            )

            lines.extend(_omitted(len(self.orphans)))

            lines.append("")

        if self.problems:
            errors = self.errors
            warnings = self.warnings

            lines.append(
                f"Problems: {len(errors)} error(s), "
                f"{len(warnings)} warning(s)"
            )
            lines.append("-" * 20)

            ordered = _worst_problems(self.problems)

            for entry in ordered[: self.MAX_LISTED]:
                lines.append(f"  {entry.code}: {entry.location}")
                lines.append(f"    {entry.reason.splitlines()[0]}")
                lines.append(f"    fix: {entry.fix.splitlines()[0]}")

            lines.extend(
                f"  {entry.code}: {entry.location}"
                for entry in ordered[self.MAX_LISTED :]
            )

            lines.extend(_omitted(len(ordered), "problem"))

            lines.append("")

        if self.content_types:
            lines.append("Content types")
            lines.append("-" * 13)

            for label, count in sorted(self.content_types.items()):
                lines.append(f"{label:<12} {count}")

        return "\n".join(lines).rstrip()

    def as_dict(self) -> dict:
        """The plan as data, for --json."""
        return {
            "root": str(self.root),
            "manifest_items": self.manifest_items(),
            "outcomes": self.counts(),
            "quality": self.by_quality(),
            "content_types": dict(sorted(self.content_types.items())),
            "importable": self.importable(),
            "already_imported": self.already_imported(),
            "with_capture": self.with_capture(),
            "metadata_only": self.metadata_only(),
            "orphans": list(self.orphans),
            "problems": [entry.as_dict() for entry in self.problems],
            "items": [
                {
                    "outcome": entry.outcome,
                    "source_id": entry.source_id,
                    "url": entry.url,
                    "bundle": entry.bundle,
                    "quality": entry.quality,
                    "reason": entry.reason,
                }
                for entry in self.items
            ],
        }

    # -----------------------------------------------------------------
    # Reading
    # -----------------------------------------------------------------

    def read_list(self, path: Path) -> None:
        """
        Read one saved-list file into the plan.

        Into memory only. Nothing is written here, which is what makes a
        plan safe to build from a list the run has not seen before.
        """
        try:
            read = read_manifest(path)

        except ManifestError as exc:
            self.problems.append(
                diag.Diagnostic(
                    code=diag.UNREADABLE_MANIFEST,
                    severity=diag.SEVERITY_ERROR,
                    location=str(path),
                    reason=str(exc),
                    fix=(
                        "Fix the file and run again. A list that cannot "
                        "be read cannot be imported, and nothing was "
                        "changed."
                    ),
                )
            )
            return

        self.absorb(read, path)

    def absorb(self, read: ManifestRead, path: Path) -> None:
        """
        Record the items a list yielded, and what was wrong with the rest.

        A link repeated within one list is a duplicate rather than a
        second item, because that is what it is: the same saved post
        written down twice.
        """
        seen: dict[str, ItemPlan] = {}

        for entry in read.items:
            known = seen.get(entry.source_id)

            if known is not None:
                # The repeated row is the duplicate, and it is listed in
                # its own right. Folding it into the first row would
                # either hide the repeat or make the first row look like
                # the redundant one.
                self.items.append(
                    ItemPlan(
                        outcome=DUPLICATE,
                        item=known.item,
                        source_id=entry.source_id,
                        url=entry.canonical_url,
                        reason="the same link appears again in this list",
                    )
                )

                continue

            stored = self.manifest.get(entry.source_id)

            if stored is None:
                self.manifest.upsert(entry)

                # Read it back rather than using the copy that came from
                # the file: the manifest owns the record, and a plan that
                # matched against a copy would not see anything a previous
                # run already knew about the item.
                stored = self.manifest.get(entry.source_id)

            else:
                stored.merge(entry)

            plan = ItemPlan(
                outcome=NEW,
                item=stored,
                source_id=entry.source_id,
                url=entry.canonical_url,
            )

            seen[entry.source_id] = plan
            self.items.append(plan)

        for issue in read.issues:
            self.problems.append(
                _issue_to_diagnostic(issue, path)
            )

    def adopt_known_items(self) -> None:
        """
        Add every item the manifest knows about that is not already here.

        The stored list is the record of everything ever imported, so a
        run with no ``--input`` is about all of it. An item already in
        the plan keeps the outcome the list file gave it, so a duplicate
        row is still reported as one.
        """
        known = {entry.source_id for entry in self.items}

        for source_id in sorted(self.manifest.items):
            if source_id in known:
                continue

            stored = self.manifest.get(source_id)

            if stored is None:
                continue

            self.items.append(
                ItemPlan(
                    outcome=NEW,
                    item=stored,
                    source_id=source_id,
                    url=stored.canonical_url,
                )
            )

    def match_captures(self, *, include_orphans: bool = True) -> None:
        """
        Pair every item with its capture and decide the outcome.

        The content of each capture is read rather than only its
        fingerprint, because whether a capture is readable at all is what
        decides whether an item is a capture or a bare link, and the
        fingerprint of nothing is still a fingerprint.
        """
        try:
            index = BundleIndex(self.root)

        except BundleError as exc:
            self.problems.append(
                diag.diagnostic_from_exception(str(self.root), exc)
            )
            return

        for problem in index.problems:
            folder, _, message = problem.partition(": ")

            self.problems.append(
                diag.Diagnostic(
                    code=diag.UNREADABLE_FILE,
                    severity=diag.SEVERITY_WARNING,
                    location=folder,
                    reason=message or problem,
                    fix=(
                        "Check the capture and run again. Nothing was "
                        "deleted and nothing outside the drop zone was "
                        "read."
                    ),
                )
            )

        self._index = index

        for plan in self.items:
            self._match_one(plan)

        if include_orphans:
            self.orphans = list(index.unclaimed())

            for folder in self.orphans:
                self.problems.append(_orphan(index, folder))

    def _match_one(self, plan: ItemPlan) -> None:
        """Decide one item's outcome from the capture beside it."""
        item = plan.item

        if item is None:
            return

        if plan.outcome == DUPLICATE:
            # A repeated row is the same saved post written down twice.
            # It is not a second item with a capture of its own to find,
            # and treating it as one would either hide the repeat or
            # invent a capture for it.
            return

        if item.bundle:
            found = self._index.find_by_name(item.bundle)

            if found is not None:
                name, _reason = found

                self._index.claim(name)

                return self._absorb_bundle(plan, item, name)

        found = self._index.find(item)

        if found is None:
            plan.outcome = MISSING_CAPTURE
            plan.reason = "a saved link with no capture beside it"

            self.problems.append(diag.missing_capture_diagnostic(item))

            return

        name, _reason = found

        self._absorb_bundle(plan, item, name)

    def _absorb_bundle(
        self,
        plan: ItemPlan,
        item: SavedItem,
        name: str,
    ) -> None:
        """Read a capture and record what it means for the item."""
        plan.bundle = name

        content = _read(name, self)

        if content is None:
            plan.outcome = FAILED
            plan.reason = "the capture could not be read"

            return

        plan.quality = content.quality()

        _count_types(name, self)

        digest = content_digest(content)

        if not digest:
            plan.outcome = MISSING_CAPTURE
            plan.reason = "the capture held nothing readable"

            self.problems.append(diag.empty_capture_diagnostic(name))

            return

        for note in content.unreadable:
            self.problems.append(
                diag.Diagnostic(
                    code=diag.UNREADABLE_FILE,
                    severity=diag.SEVERITY_WARNING,
                    location=f"{name}: {note.split(':')[0]}",
                    reason=note,
                    fix=(
                        "Replace the file, or leave it. The rest of the "
                        "capture is imported without it, and the file "
                        "itself is not deleted."
                    ),
                )
            )

        item.capture_quality = plan.quality

        if not item.content_digest:
            plan.outcome = NEW

        elif digest == item.content_digest:
            plan.outcome = UNCHANGED
            plan.reason = "the capture is exactly what was imported"

        else:
            plan.outcome = CHANGED
            plan.reason = "the capture changed since it was last read"

            plan.changed_files = [path.name for path in content.media]


def _listing(group: list[ItemPlan]) -> list[str]:
    """
    The entries to print, bounded, with the rest counted rather than
    dropped.

    The count is always exact. A reader who is told "25 shown of 9,943"
    can decide whether to ask for the list; a reader shown a truncated
    list with no count is left guessing whether they saw everything.
    """
    shown = group[: Plan.MAX_LISTED]

    lines = [f"  {_describe(entry)}" for entry in shown]

    lines.extend(_omitted(len(group)))

    return lines


def _omitted(total: int, noun: str = "item") -> list[str]:
    """One line saying how much was not printed, when any was."""
    hidden = max(0, total - Plan.MAX_LISTED)

    if not hidden:
        return []

    return [
        f"  ... and {hidden} more {noun}(s), listed in full with --json"
    ]


def _describe(entry: ItemPlan) -> str:
    line = entry.url or entry.source_id or "(no url)"

    if entry.reason:
        line += f"  ({entry.reason})"

    return line


def _worst_first(group: list[ItemPlan]) -> list[ItemPlan]:
    """
    The entries a reader should see first.

    Failures before successes, and inside each, items with nothing behind
    them, because a plan is read to find what needs doing rather than to
    confirm what was done.
    """
    return sorted(
        group,
        key=lambda entry: (not entry.is_error, entry.url or entry.source_id),
    )


def _worst_problems(
    problems: list[diag.Diagnostic],
) -> list[diag.Diagnostic]:
    return sorted(
        problems,
        key=lambda entry: (not entry.is_error, entry.code, entry.location),
    )


# ---------------------------------------------------------------------
# Reading the inbox
# ---------------------------------------------------------------------


def build_plan(
    root: str | Path,
    *,
    manifest: SavedItemsManifest | None = None,
    inputs: list[str | Path] | None = None,
    include_orphans: bool = True,
) -> Plan:
    """
    Read a drop zone and say what a run would do.

    Takes a manifest rather than writing one, so calling this cannot
    change the state of the inbox. When ``inputs`` are given they are
    read into an in-memory copy, which is how a dry run can show what a
    new list would add without recording that it did.
    """
    base = Path(root)

    if manifest is not None:
        working = manifest
    else:
        try:
            working = SavedItemsManifest.load(base / "saved-items-manifest.json")

        except ManifestUnreadable as exc:
            plan = Plan(root=base, manifest=SavedItemsManifest(base / "x"))

            plan.problems.append(
                diag.Diagnostic(
                    code=diag.UNREADABLE_MANIFEST,
                    severity=diag.SEVERITY_ERROR,
                    location=str(base / "saved-items-manifest.json"),
                    reason=str(exc),
                    fix=(
                        "Move the file aside to start a new manifest. It "
                        "is not deleted or overwritten, because a manifest "
                        "that cannot be read is the only record of what "
                        "has already been imported."
                    ),
                )
            )

            return plan

    plan = Plan(root=base, manifest=working)

    for entry in inputs or []:
        plan.read_list(Path(entry))

    # Everything the manifest already knows about, whether or not a list
    # file was passed. A status or validation run with no --input is
    # asking about the stored list, and answering only about the rows in
    # the file it happens to be given would describe a fraction of it.
    plan.adopt_known_items()

    plan.match_captures(include_orphans=include_orphans)

    return plan


def _issue_to_diagnostic(issue, path: Path) -> diag.Diagnostic:
    """Turn a reader's issue into something the user can act on."""
    message = str(issue)

    if "no URL" in message:
        code = diag.MISSING_URL
        severity = diag.SEVERITY_ERROR
        fix = (
            "Add the address of the saved item, or remove the row. A row "
            "with no address does not describe a saved item."
        )

    elif "not an absolute URL" in message or "unsupported URL scheme" in message:
        code = diag.INVALID_URL
        severity = diag.SEVERITY_ERROR
        fix = (
            "Check the address. A saved post looks like:\n"
            "\n"
            "  https://www.linkedin.com/posts/...\n"
            "  https://www.linkedin.com/pulse/..."
        )

    elif "no URL column" in message:
        code = diag.NO_URL_COLUMN
        severity = diag.SEVERITY_ERROR
        fix = (
            "Name the column holding the address. Accepted spellings "
            "include url, link, LinkedIn URL and permalink."
        )

    else:
        code = diag.INVALID_ROW
        severity = diag.SEVERITY_ERROR
        fix = "Fix the row and run again."

    return diag.Diagnostic(
        code=code,
        severity=severity,
        location=f"{path.name}: {issue.location}",
        reason=message,
        fix=fix,
    )


def _read(name: str, plan: Plan) -> CapturedContent | None:
    """Read one capture, turning a failure into a diagnostic."""
    try:
        return read_bundle(name, root=plan.root)

    except BundleError as exc:
        plan.problems.append(
            diag.diagnostic_from_exception(name, exc)
        )

        return None

    except Exception as exc:  # noqa: BLE001
        plan.problems.append(
            diag.diagnostic_from_exception(name, exc)
        )

        return None


def _count_types(name: str, plan: Plan) -> None:
    """
    Count the file types a capture holds, for the summary.

    The name is relative to the drop zone, so it is joined back onto the
    root before anything touches the filesystem. A summary built from
    paths that do not resolve would report every capture as an unknown
    format, which looks like the pipeline cannot read its own inbox.
    """
    folder = Path(plan.root) / name

    if not folder.is_dir():
        label = folder.suffix.lower().lstrip(".").upper() or "OTHER"

        plan.content_types[label] += 1

        return

    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue

        if path.name.lower() in _CAPTURE_FILES:
            # The capture file names the item; it is not content, and
            # counting it would make "3 captures of Markdown" wrong.
            continue

        suffix = path.suffix.lower()

        if suffix in _TEXT_SUFFIXES:
            label = "Markdown" if suffix == ".md" else "Text"

        elif suffix in _HTML_SUFFIXES:
            label = "HTML"

        elif suffix in _DOCUMENT_SUFFIXES:
            label = "PDF"

        elif suffix in _IMAGE_SUFFIXES:
            label = "Images"

        else:
            label = "Other"

        plan.content_types[label] += 1


def _orphan(index: BundleIndex, folder: str) -> diag.Diagnostic:
    """Describe a capture that no saved item claims."""
    association = index.associations.get(folder)

    url = association.url if association else None
    source_id = association.source_id if association else None

    if not url:
        claim = find_capture_file(index.root / folder)

        if claim is not None:
            try:
                payload = json.loads(
                    claim.read_text(encoding="utf-8-sig", errors="replace")
                )

            except (OSError, json.JSONDecodeError):
                payload = None

            if isinstance(payload, dict):
                url, declared = diag.capture_claim_from_json(payload)

                source_id = source_id or declared

    return diag.orphan_diagnostic(
        folder,
        detected_url=url,
        detected_source_id=source_id,
    )


__all__ = [
    "CHANGED",
    "DUPLICATE",
    "FAILED",
    "INVALID",
    "MISSING_CAPTURE",
    "NEW",
    "OUTCOMES",
    "UNCHANGED",
    "ItemPlan",
    "Plan",
    "build_plan",
]
