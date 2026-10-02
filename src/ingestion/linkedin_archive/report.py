"""
What an archive import did, in the numbers a person needs.

The report is split the way the saved-items report already is, and for
the same reason: what was read is not what was stored, and a reader who
sees one number for both cannot tell which stage lost a post.

Nothing here is estimated. Every figure is counted off the archive as
read or off the posts as written, and a section with nothing in it says
zero rather than being left out, because an absent number reads as a
forgotten one.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from src.ingestion.linkedin_archive.archive import (
    Archive,
    ArchiveRecord,
)
from src.ingestion.linkedin_archive.prepare import PrepareResult


@dataclass
class ImportReport:
    """
    The result of reading an archive and preparing it for import.

    ``read`` describes the archive as it was found. ``written`` and
    ``failed`` describe the outcome. Keeping them apart is what makes a
    partial import readable: a hundred failures during a read of five
    hundred records is a different event from a hundred failures while
    writing them, and the recovery is different too.
    """

    archive_path: str = ""
    archive_file: str = ""
    drop_zone: str = ""

    # -- what was in the archive ------------------------------------
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    duplicate_records: int = 0
    unique_records: int = 0

    records_without_text: int = 0
    records_with_media: int = 0
    records_without_media: int = 0
    records_with_missing_media: int = 0
    records_with_all_media: int = 0
    records_multi_media: int = 0
    records_with_permalink: int = 0
    records_without_permalink: int = 0
    records_with_author: int = 0

    # -- what is in the media folder --------------------------------
    media_total: int = 0
    media_bytes: int = 0
    media_images: int = 0
    media_svg: int = 0
    media_pdf: int = 0
    media_other: int = 0
    media_unreadable: int = 0
    media_misnamed: int = 0
    media_shared: int = 0
    media_rejected: int = 0
    media_missing: int = 0

    # -- what happened ----------------------------------------------
    prepared: int = 0
    already_prepared: int = 0
    duplicates_skipped: int = 0
    failed: int = 0

    problems: list[dict] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "archive": {
                "path": self.archive_path,
                "file": self.archive_file,
                "drop_zone": self.drop_zone,
            },
            "records": {
                "total": self.total_records,
                "valid": self.valid_records,
                "invalid": self.invalid_records,
                "duplicates": self.duplicate_records,
                "unique": self.unique_records,
                "without_text": self.records_without_text,
                "with_permalink": self.records_with_permalink,
                "without_permalink": self.records_without_permalink,
                "with_author": self.records_with_author,
            },
            "media": {
                "referencing_records": self.records_with_media,
                "records_without_media": self.records_without_media,
                "records_multi_media": self.records_multi_media,
                "records_with_missing_media": (
                    self.records_with_missing_media
                ),
                "files": self.media_total,
                "bytes": self.media_bytes,
                "images": self.media_images,
                "svg": self.media_svg,
                "pdf": self.media_pdf,
                "other": self.media_other,
                "unreadable": self.media_unreadable,
                "misnamed": self.media_misnamed,
                "shared": self.media_shared,
                "rejected": self.media_rejected,
                "missing": self.media_missing,
            },
            "outcome": {
                "prepared": self.prepared,
                "already_prepared": self.already_prepared,
                "duplicates_skipped": self.duplicates_skipped,
                "failed": self.failed,
            },
            "problems": self.problems,
            "failures": self.failures,
        }

    def render(self) -> str:
        """The report as text, grouped by what a reader is deciding."""
        lines: list[str] = []

        lines.append("LinkedIn Archive")
        lines.append("----------------")
        lines.append(f"{'Archive':<28}{self.archive_file}")
        lines.append(
            f"{'Drop zone':<28}{_short_path(self.drop_zone) or '(not written)'}"
        )
        lines.append("")

        lines.append("Records")
        lines.append("-------")
        lines.extend(_row("Total records", self.total_records))
        lines.extend(_row("Valid records", self.valid_records))
        lines.extend(_row("Invalid records", self.invalid_records))
        lines.extend(_row("Duplicate records", self.duplicate_records))
        lines.extend(_row("Unique records", self.unique_records))
        lines.append("")
        lines.extend(_row("With a permalink", self.records_with_permalink))
        lines.extend(_row("With no permalink", self.records_without_permalink))
        lines.extend(_row("With an author name", self.records_with_author))
        lines.extend(_row("With no text", self.records_without_text))
        lines.append("")

        lines.append("Media")
        lines.append("-----")
        lines.extend(_row("Records with media", self.records_with_media))
        lines.extend(_row("Records without media", self.records_without_media))
        lines.extend(_row("Multi-media records", self.records_multi_media))
        lines.extend(
            _row(
                "Records missing a file",
                self.records_with_missing_media,
            )
        )
        lines.append("")
        lines.extend(_row("Files in the archive", self.media_total))
        lines.append(f"{'Total bytes':<28}{self.media_bytes:,}")
        lines.extend(_row("Images", self.media_images))
        lines.extend(_row("SVG", self.media_svg))
        lines.extend(_row("PDF", self.media_pdf))
        lines.extend(_row("Other", self.media_other))
        lines.extend(_row("Named for a different type", self.media_misnamed))
        lines.extend(_row("Unreadable", self.media_unreadable))
        lines.extend(_row("Identical to another file", self.media_shared))
        lines.extend(_row("Referenced but absent", self.media_missing))
        lines.extend(_row("Outside the media root", self.media_rejected))
        lines.append("")

        lines.append("Outcome")
        lines.append("-------")
        lines.extend(_row("Prepared", self.prepared))
        lines.extend(_row("Already prepared", self.already_prepared))
        lines.extend(
            _row("Duplicates left to their keeper", self.duplicates_skipped)
        )
        lines.extend(_row("Failed", self.failed))

        if self.problems:
            lines.append("")
            lines.append("Things worth knowing")
            lines.append("---------------------")

            kinds = Counter(
                _problem_kind(entry.get("note", ""))
                for entry in self.problems
            )

            for kind, count in kinds.most_common(12):
                lines.append(f"  {count:>5}  {kind}")

        if self.failures:
            lines.append("")
            lines.append("Failures")
            lines.append("--------")

            for entry in self.failures[:20]:
                lines.append(
                    f"  {entry.get('post_id', '?')}  "
                    f"{entry.get('error_type', '?')}: "
                    f"{str(entry.get('message', ''))[:100]}"
                )

            if len(self.failures) > 20:
                lines.append(
                    f"  ... and {len(self.failures) - 20} more"
                )

        return "\n".join(lines)

    def counts_for(self, key: str) -> int:
        """One number, for a caller that wants a single figure."""
        return int(self.as_dict().get(key, {}).get("total", 0))


def build_report(
    archive: Archive,
    *,
    prepared: PrepareResult | None = None,
) -> ImportReport:
    """
    Count the archive and the outcome.

    Every figure is derived from the archive object or the prepare
    result. Nothing is carried over from a previous run and nothing is
    assumed, so a report describes the archive that was actually read
    rather than the one this importer was written for.
    """
    result = PrepareResult(root=Path(archive.path)) if prepared is None else prepared

    report = ImportReport(
        archive_path=str(archive.path),
        archive_file="posts_archive.json",
        drop_zone=str(result.root) if prepared else "",
    )

    report.total_records = archive.total_seen
    report.invalid_records = len(archive.failures)
    report.valid_records = len(archive.records)
    report.duplicate_records = len(archive.duplicates)
    report.unique_records = len(archive.unique())

    _count_records(report, archive.records)
    _count_media(report, archive)
    _count_outcome(report, archive, result)

    report.failures = [failure.as_dict() for failure in archive.failures]

    return report


def _count_records(report: ImportReport, records: list[ArchiveRecord]) -> None:
    for record in records:
        if not record.has_text:
            report.records_without_text += 1

        if record.identity.has_permalink:
            report.records_with_permalink += 1
        else:
            report.records_without_permalink += 1

        if record.author and record.author.strip().lower() != "unknown":
            report.records_with_author += 1

        if record.media:
            report.records_with_media += 1
        else:
            report.records_without_media += 1

        if len(record.media) > 1:
            report.records_multi_media += 1

        if record.missing_media:
            report.records_with_missing_media += 1


def _count_media(report: ImportReport, archive: Archive) -> None:
    index = archive.media

    report.media_total = len(index)
    report.media_bytes = index.total_bytes
    report.media_unreadable = len(index.unreadable)
    report.media_rejected = len(index.rejected)

    for asset in index.assets.values():
        if asset.is_image:
            report.media_images += 1
        elif asset.is_svg:
            report.media_svg += 1
        elif asset.is_document:
            report.media_pdf += 1
        else:
            report.media_other += 1

        if asset.readable and not asset.extension_agrees:
            report.media_misnamed += 1

    for record in archive.records:
        report.media_missing += len(record.missing_media)

        for name in (
            asset.name for asset in record.media
        ):
            if index.shared_by_digest(name):
                report.media_shared += 1
                break


def _count_outcome(
    report: ImportReport,
    archive: Archive,
    result: PrepareResult,
) -> None:
    report.prepared = result.written
    report.already_prepared = result.unchanged
    report.duplicates_skipped = result.skipped_duplicates
    report.failed = len(archive.failures) + len(result.problems)

    problems: list[dict] = []

    for record in archive.unique():
        for note in record.notes:
            problems.append(
                {
                    "post_id": record.post_id,
                    "note": note,
                }
            )

    for entry in result.problems:
        problems.append({"post_id": "", "note": entry})

    report.problems = problems


def _row(label: str, value: object) -> list[str]:
    return [f"{label:<28}{value:>8}"]


def _short_path(path: str) -> str:
    """
    A path as short as it can honestly be.

    The rendered report gets pasted into issues and read by people who
    are not on this machine, so an absolute path is noise at best. A
    path inside the working directory is shown relative to it, which is
    the form a command can be pasted with; anything else is reduced to
    its last two parts, which is enough to tell two drop zones apart.

    The full path is still in the JSON report, which is a local file
    written for a machine rather than a person.
    """

    if not path:
        return ""

    candidate = Path(path)

    try:
        relative = candidate.resolve().relative_to(Path.cwd().resolve())

    except (ValueError, OSError):
        parts = candidate.parts

        return str(Path(*parts[-2:])) if len(parts) > 2 else str(candidate)

    return str(relative)


def _problem_kind(note: str) -> str:
    """
    A note reduced to its cause, so a hundred of them read as one line.

    Grouped rather than listed because a report nobody reads is a
    report nobody acts on, and the count is the part that tells a
    reader whether the notes are one recurring cause or a hundred
    different ones.
    """
    lowered = (note or "").lower()

    if "no usable author" in lowered:
        return "the archive recorded no author name"

    if "listed by the archive but not in the media folder" in lowered:
        return "a media file the archive names is not in the folder"

    if "not a format this project reads" in lowered:
        return "a file is not a format this project reads"

    if "named for a different type" in lowered or "but its content is" in lowered:
        return "a file's name does not describe its content"

    if "multi-slide but supplied" in lowered:
        return "the archive marked a carousel but supplied one file"

    if "no text for this post" in lowered:
        return "the archive recorded no text"

    if "listed more than once" in lowered:
        return "a record lists the same file twice"

    if "included a path" in lowered:
        return "a media reference included a path rather than a name"

    if "counted" in lowered and "usable" in lowered:
        return "the archive's own media count disagrees with its files"

    if "could not be read" in lowered:
        return "a file could not be read"

    if "could not be placed" in lowered:
        return "a file could not be written to the drop zone"

    return note[:80] if note else "(no detail)"


def write_report(report: ImportReport, path: str | Path) -> Path:
    """Write the report as JSON, for a caller that wants the numbers."""
    target = Path(path)

    target.parent.mkdir(parents=True, exist_ok=True)

    temporary = target.with_suffix(target.suffix + ".tmp")

    temporary.write_text(
        json.dumps(report.as_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    temporary.replace(target)

    return target


__all__ = ["ImportReport", "build_report", "write_report"]
