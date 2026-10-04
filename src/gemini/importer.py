"""
Bringing the package in, and proving what came in.

Import is two steps kept apart on purpose.

**Build** reads the package and the archive, cross-checks them, judges
the candidates and computes which slides still need CP12. It writes
nothing. Being able to look at what an import *would* do without
touching the repository is what makes it safe to run against a package
whose contents nobody has verified.

**Run** does the same work and then writes it, and skips entirely when
the package and the archive are both unchanged. Skipping is by digest,
not by timestamp, because a re-run must be distinguishable from a
re-import: if the bytes are the same there is nothing new to record, and
re-writing identical files under a new timestamp would make the
repository look busier than it is.

Three things are written that are not records, and all three exist so
that a claim can be compared with what was found:

* ``manifest.json`` -- the package's own headline figures beside the
  counts actually measured, and the digests that justify skipping.
* ``cross_check.json`` -- every disagreement with the archive.
* ``gaps.json`` -- the slides CP12 is still worth being run on.

Nothing here writes to the archive. The archive is opened, hashed and
listed, and that is the whole of its involvement.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from pydantic import BaseModel

from src.gemini.crosscheck import (
    ArchiveError,
    archive_digest,
    cross_check,
    group_verdicts,
    index_archive,
    load_archive_posts,
)
from src.gemini.models import (
    CrossCheckReport,
    GeminiCandidateQuestion,
    GeminiImageRecord,
    GeminiPostGroup,
    GeminiPostKnowledge,
    GeminiTopicIndex,
    ImportManifest,
    MatchVerdict,
    OcrQuality,
    OcrStatus,
)
from src.gemini.ocr_quality import needs_reprocessing
from src.gemini.package_reader import (
    PackageError,
    attach_transcriptions,
    groups_from_records,
    package_digest,
    package_info,
    parse_inventory,
    parse_knowledge,
    read_package,
)
from src.gemini.questions import build_candidates
from src.gemini.safety import assert_public
from src.gemini.taxonomy import (
    coverage,
    parse_technical,
    parse_topic_index,
    resolve_fragments,
    unmatched_groups,
)


#: Where an import lands, following the repository's one-directory-per-
#: subject convention under ``data/``.
DEFAULT_OUTPUT = Path("data/imported/gemini")


class GapReason(str, Enum):
    """
    Why a slide is still worth spending a vision call on.

    Four reasons, each observable, and no fifth. A score-based "maybe
    poor" bucket is the obvious thing to add and it is the thing that
    turns a bounded fallback into a full re-run of 2,774 images, so the
    reasons are stated as facts about the record instead.
    """

    #: The package said this image has no text.
    NO_TEXT = "no_text"

    #: The package marked the record unresolved or failed.
    UNRESOLVED = "unresolved"

    #: There is text, but it is symbol debris rather than words.
    DEBRIS = "debris"

    #: The package said nothing at all about an image that is on disk.
    ABSENT = "absent"

    #: The package transcribed an image the archive does not have. Not
    #: a reason to run a vision call -- there is no file to run it on --
    #: and reported separately so the two are not confused.
    MISSING_FROM_ARCHIVE = "missing_from_archive"


class Gap(BaseModel):
    """One slide CP12 could still read, and why."""

    filename: str
    group_id: str = ""
    activity_id: str = ""
    slide_number: int = 0
    reason: GapReason
    quality: OcrQuality = OcrQuality.EMPTY
    note: str = ""

    #: Whether a vision call could actually be made. False for a file
    #: the archive does not have.
    actionable: bool = True


@dataclass(frozen=True)
class ImportResult:
    """
    Everything one build produced.

    Frozen because a report that can be edited after the fact is not a
    report. Nothing here is a claim until it has been measured, which
    is why ``claimed_*`` figures are carried verbatim rather than
    replaced by the measured ones.
    """

    manifest: ImportManifest
    cross_check: CrossCheckReport
    records: list[GeminiImageRecord] = field(default_factory=list)
    groups: dict[str, GeminiPostGroup] = field(default_factory=dict)
    candidates: list[GeminiCandidateQuestion] = field(default_factory=list)
    topic_index: GeminiTopicIndex = field(default_factory=GeminiTopicIndex)
    knowledge: dict[str, GeminiPostKnowledge] = field(default_factory=dict)
    coverage: dict[str, int] = field(default_factory=dict)
    unresolved_topic_groups: list[str] = field(default_factory=list)
    gaps: list[Gap] = field(default_factory=list)
    wrote: bool = False

    def verdict_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}

        for candidate in self.candidates:
            key = candidate.verdict.value

            counts[key] = counts.get(key, 0) + 1

        return counts

    def describe(self) -> str:
        lines = [
            "Gemini package import",
            "====================",
            "",
            f"package          : {self.manifest.package.package_name}",
            f"package digest   : {self.manifest.package_digest[:16]}",
            f"archive digest   : {self.manifest.archive_digest[:16]}",
            "",
            f"image records    : {len(self.records)}",
            f"post groups      : {len(self.groups)}",
            f"candidate Qs     : {len(self.candidates)} "
            + " ".join(
                f"{name}={count}"
                for name, count in sorted(self.verdict_counts().items())
            ),
            f"topics           : {len(self.topic_index.topics)}",
            f"gap slides       : {len(self.gaps)} "
            f"({len([g for g in self.gaps if g.actionable])} actionable)",
            "",
            "cross-check against the archive",
            "----------------------------------",
        ]

        for name, value in self.cross_check.counts().items():
            lines.append(f"{name:<38}: {value}")

        lines.append("")

        lines.append("the package's own figures, against what was measured")
        lines.append("-----------------------------------------------")

        for label, claimed, measured in self._comparisons():
            marker = "agrees" if claimed == measured else "DIFFERS"

            lines.append(f"{label:<34}: claimed {claimed} / found {measured} ({marker})")

        return "\n".join(lines)

    def _comparisons(self) -> list[tuple[str, int | None, int]]:
        """
        Claimed against measured, side by side.

        Reported as pairs rather than reconciled. Where they differ the
        interesting number is the difference itself -- it is either a
        counting difference this project can explain, or a real gap in
        the package's coverage, and the two call for different
        responses.
        """

        available = len(
            [r for r in self.records if r.ocr_status is OcrStatus.AVAILABLE]
        )

        no_text = len(
            [r for r in self.records if r.ocr_status is OcrStatus.NO_TEXT]
        )

        duplicates = len([r for r in self.records if r.exact_duplicate_of])

        package = self.manifest.package

        return [
            ("total images", package.claimed_total_images, len(self.records)),
            ("activity groups", package.claimed_groups, len(self.groups)),
            ("ocr available", package.claimed_ocr_available, available),
            ("ocr no-text", package.claimed_ocr_no_text, no_text),
            ("exact duplicates", package.claimed_exact_duplicates, duplicates),
            (
                "unique after duplicates",
                package.claimed_unique_after_duplicates,
                len(
                    {
                        r.actual_sha256 or r.declared_sha256
                        for r in self.records
                    }
                ),
            ),
        ]


# ---------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------


def gaps_for(record: GeminiImageRecord) -> Gap | None:
    """
    Whether one slide is a candidate for the CP12 fallback.

    The gate is narrow by design. Importing this package exists to avoid
    thousands of vision calls, so a slide is offered to CP12 only when
    there is a factual reason it was not read -- no text, an unresolved
    status, debris, or no transcription at all. A fragmentary slide is
    not offered: it is real text from a real picture, it gets indexed,
    and it is labelled as fragmentary in the knowledge base.
    """

    if record.verdict is MatchVerdict.MISSING_FROM_ARCHIVE:
        return Gap(
            filename=record.filename,
            group_id=record.post_id,
            activity_id=record.activity_id,
            slide_number=record.slide_number,
            reason=GapReason.MISSING_FROM_ARCHIVE,
            quality=record.ocr_quality,
            note=(
                "the package transcribed a file the archive does not "
                "have, so there is nothing on disk for a vision call to "
                "read"
            ),
            actionable=False,
        )

    if record.ocr_status is OcrStatus.NO_TEXT:
        return Gap(
            filename=record.filename,
            group_id=record.post_id,
            activity_id=record.activity_id,
            slide_number=record.slide_number,
            reason=GapReason.NO_TEXT,
            quality=record.ocr_quality,
            note=(
                "the package reports no text in this image; a second "
                "reader may find some, since a claim of emptiness is a "
                "finding and not a certainty"
            ),
        )

    if record.ocr_status in {OcrStatus.UNRESOLVED, OcrStatus.FAILED, OcrStatus.PENDING}:
        return Gap(
            filename=record.filename,
            group_id=record.post_id,
            activity_id=record.activity_id,
            slide_number=record.slide_number,
            reason=GapReason.UNRESOLVED,
            quality=record.ocr_quality,
            note=record.verdict_note or record.ocr_note,
        )

    if record.ocr_status is OcrStatus.ABSENT:
        return Gap(
            filename=record.filename,
            group_id=record.post_id,
            activity_id=record.activity_id,
            slide_number=record.slide_number,
            reason=GapReason.ABSENT,
            quality=record.ocr_quality,
            note="the package said nothing about an image that is on disk",
        )

    if needs_reprocessing(
        record.ocr_quality,
        width=record.width,
        height=record.height,
    ):
        return Gap(
            filename=record.filename,
            group_id=record.post_id,
            activity_id=record.activity_id,
            slide_number=record.slide_number,
            reason=GapReason.DEBRIS,
            quality=record.ocr_quality,
            note=record.ocr_note,
        )

    return None


def archive_fingerprint(archive_root: str | Path) -> str:
    """
    A cheap fingerprint: names, sizes and modification times.

    Used only to decide whether a re-import can be skipped, because it
    needs a ``stat`` per file rather than a full read of 288 MB. It is
    deliberately weaker than :func:`archive_digest` and is not recorded
    as evidence of anything: a file edited in place while keeping its
    size and timestamp would be missed. That is the trade for a skip
    check that costs nothing, and the manifest's content digest still
    covers the case where the check is wrong.
    """

    media_root = Path(archive_root).expanduser() / "media"

    if not media_root.is_dir():
        raise ArchiveError(f"media directory is missing: {media_root}")

    digest = hashlib.sha256()

    for path in sorted(media_root.iterdir()):
        if not path.is_file():
            continue

        stat = path.stat()

        digest.update(path.name.encode("utf-8", "replace"))

        digest.update(b"\x00")

        digest.update(str(stat.st_size).encode("ascii"))

        digest.update(b"\x00")

        digest.update(str(stat.st_mtime_ns).encode("ascii"))

        digest.update(b"\x00")

    return digest.hexdigest()


def build(
    package_root: str | Path,
    archive_root: str | Path,
) -> ImportResult:
    """
    Read both sources, check one against the other, judge the rest.

    Writes nothing. Raises :class:`PackageError` or :class:`ArchiveError`
    rather than returning a partial import, because a partial import
    that looks complete is worse than one that does not happen.
    """

    contents = read_package(package_root)

    info = package_info(package_root, contents)

    records = parse_inventory(contents)

    transcriptions, declared_groups, _ = parse_knowledge(contents)

    records = attach_transcriptions(records, transcriptions)

    archive_posts = load_archive_posts(archive_root)

    archive_files = index_archive(archive_root)

    records, report = cross_check(records, archive_files, archive_posts)

    groups = groups_from_records(records)

    # The markdown's declared counts are kept for comparison; the
    # membership of each group comes from the rows that name it.
    declared_counts = {
        group_id: group.declared_slide_count
        for group_id, group in declared_groups.items()
        if group.declared_slide_count
    }

    verdicts = group_verdicts(groups, records, archive_posts)

    for group_id, group in verdicts.items():
        declared = declared_counts.get(group_id, 0)

        if declared:
            group = group.model_copy(
                update={"declared_slide_count": declared}
            )

        verdicts[group_id] = group

        if group.verdict_note:
            report.notes.append(f"{group_id}: {group.verdict_note}")

    topics = parse_topic_index(contents)

    technical = parse_technical(contents)

    knowledge, fragment_notes = resolve_fragments(technical, records)

    report.notes.extend(fragment_notes)

    candidates = build_candidates(contents, records)

    gaps = [gap for gap in (gaps_for(record) for record in records) if gap]

    manifest = ImportManifest(
        package=info,
        package_digest=package_digest(contents),
        archive_digest=archive_digest(archive_files),
        records=len(records),
        groups=len(verdicts),
        candidate_questions=len(candidates),
        ocr_available=len([r for r in records if r.ocr_status is OcrStatus.AVAILABLE]),
        ocr_no_text=len([r for r in records if r.ocr_status is OcrStatus.NO_TEXT]),
        ocr_unresolved=len(
            [
                r
                for r in records
                if r.ocr_status
                in {OcrStatus.UNRESOLVED, OcrStatus.FAILED, OcrStatus.PENDING}
            ]
        ),
        duplicates=len([r for r in records if r.exact_duplicate_of]),
        previews=len([r for r in records if r.likely_preview_of]),
        unreadable=len(
            [
                r
                for r in records
                if r.verdict is MatchVerdict.MISSING_FROM_ARCHIVE
            ]
        ),
    )

    return ImportResult(
        manifest=manifest,
        cross_check=report,
        records=records,
        groups=verdicts,
        candidates=candidates,
        topic_index=topics,
        knowledge=knowledge,
        coverage=coverage(contents),
        unresolved_topic_groups=unmatched_groups(topics, knowledge),
        gaps=gaps,
    )


# ---------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------


def public_record(record: GeminiImageRecord) -> dict:
    """
    One record as it may be committed and published.

    Drops :attr:`GeminiImageRecord.raw_ocr_text` and keeps
    :attr:`GeminiImageRecord.public_text` in its place, plus the count of
    what was replaced.

    The verbatim text is withheld rather than sanitised in place, so
    ``raw_ocr_text`` keeps meaning "exactly what the package wrote" and a
    reader comparing the two files is never misled about which is which.
    The write-time guard stays armed regardless: this function is the
    first line, not the only one.
    """

    payload = record.model_dump(mode="json")

    payload.pop("raw_ocr_text", None)

    return payload


def _write_json(path: Path, payload: object, *, where: str) -> None:
    """
    Write one JSON file atomically, after checking it.

    Checked before the write rather than after, so a payload containing
    a local path or a credential never reaches the disk even briefly.
    """

    serialisable = (
        payload.model_dump(mode="json")
        if isinstance(payload, BaseModel)
        else payload
    )

    assert_public(serialisable, where=where)

    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(path.suffix + ".tmp")

    temporary.write_text(
        json.dumps(serialisable, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    temporary.replace(path)


def read_manifest(output: str | Path = DEFAULT_OUTPUT) -> ImportManifest | None:
    """The manifest from a previous import, if there was one."""

    path = Path(output) / "manifest.json"

    if not path.is_file():
        return None

    try:
        return ImportManifest.model_validate_json(
            path.read_text(encoding="utf-8")
        )

    except (OSError, ValueError):
        # A manifest that cannot be read is treated as absent, which
        # causes a re-import rather than a silent skip.
        return None


def load_records(output: str | Path = DEFAULT_OUTPUT) -> list[GeminiImageRecord]:
    """
    The imported image records, from a previous import.

    Used by the search index and the wiki without re-reading the
    package, so neither needs the package to be present.
    """

    base = Path(output) / "images"

    if not base.is_dir():
        return []

    records: list[GeminiImageRecord] = []

    for path in sorted(base.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))

        except (OSError, json.JSONDecodeError):
            continue

        for item in payload.get("images", []) if isinstance(payload, dict) else []:
            try:
                records.append(GeminiImageRecord.model_validate(item))

            except ValueError:
                continue

    return records


def load_gaps(output: str | Path = DEFAULT_OUTPUT) -> list[Gap]:
    """The slides CP12 was left with, from a previous import."""

    path = Path(output) / "gaps.json"

    if not path.is_file():
        return []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))

    except (OSError, json.JSONDecodeError):
        return []

    gaps: list[Gap] = []

    for item in payload.get("gaps", []) if isinstance(payload, dict) else []:
        try:
            gaps.append(Gap.model_validate(item))

        except ValueError:
            continue

    return gaps


def load_knowledge(
    output: str | Path = DEFAULT_OUTPUT,
) -> dict[str, GeminiPostKnowledge]:
    """The per-post technology claims, from a previous import."""

    path = Path(output) / "knowledge.json"

    if not path.is_file():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))

    except (OSError, json.JSONDecodeError):
        return {}

    knowledge: dict[str, GeminiPostKnowledge] = {}

    for item in payload.get("posts", []) if isinstance(payload, dict) else []:
        try:
            entry = GeminiPostKnowledge.model_validate(item)

        except ValueError:
            continue

        knowledge[entry.group_id.upper()] = entry

    return knowledge


def load_topic_index(
    output: str | Path = DEFAULT_OUTPUT,
) -> GeminiTopicIndex:
    """The topic index, from a previous import."""

    path = Path(output) / "topics.json"

    if not path.is_file():
        return GeminiTopicIndex()

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))

    except (OSError, json.JSONDecodeError):
        return GeminiTopicIndex()

    topics = payload.get("topics", {}) if isinstance(payload, dict) else {}

    return GeminiTopicIndex(topics=topics if isinstance(topics, dict) else {})


def load_cross_check(
    output: str | Path = DEFAULT_OUTPUT,
) -> CrossCheckReport:
    """The disagreements found by the run that actually looked."""

    path = Path(output) / "cross_check.json"

    if not path.is_file():
        return CrossCheckReport()

    try:
        return CrossCheckReport.model_validate_json(
            path.read_text(encoding="utf-8")
        )

    except (OSError, ValueError):
        return CrossCheckReport()


def load_candidates(
    output: str | Path = DEFAULT_OUTPUT,
) -> list[GeminiCandidateQuestion]:
    """Every candidate question with its verdict, from a previous import."""

    path = Path(output) / "questions.json"

    if not path.is_file():
        return []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))

    except (OSError, json.JSONDecodeError):
        return []

    questions: list[GeminiCandidateQuestion] = []

    for item in payload.get("candidates", []) if isinstance(payload, dict) else []:
        try:
            questions.append(GeminiCandidateQuestion.model_validate(item))

        except ValueError:
            continue

    return questions


def run(
    package_root: str | Path,
    archive_root: str | Path,
    output: str | Path = DEFAULT_OUTPUT,
    *,
    force: bool = False,
) -> ImportResult:
    """
    Build, then write unless nothing changed.

    The skip decision is made before the expensive work: the archive's
    cheap fingerprint is compared against the one in the previous
    manifest, and if both it and the package digest are unchanged the
    import is skipped. The package digest still requires reading the
    five files, which is small and is the correct cost for knowing
    whether the package changed.
    """

    base = Path(output)

    previous = read_manifest(base)

    current_package = package_digest(read_package(package_root))

    current_archive = archive_fingerprint(archive_root)

    if (
        not force
        and previous is not None
        and previous.package_digest == current_package
        and previous.fingerprint == current_archive
    ):
        # Nothing changed, so nothing is re-read and nothing is
        # rewritten. The previous cross-check is returned rather than an
        # empty one so a caller reporting on the import still sees the
        # disagreements from the run that actually looked.
        return ImportResult(
            manifest=previous,
            cross_check=load_cross_check(base),
            records=load_records(base),
            candidates=load_candidates(base),
            gaps=load_gaps(base),
        )

    result = build(package_root, archive_root)

    result = _with_fingerprint(result, current_archive)

    records_by_group: dict[str, list[GeminiImageRecord]] = {}

    for record in result.records:
        records_by_group.setdefault(record.post_id, []).append(record)

    images_directory = base / "images"

    if images_directory.is_dir():
        # Remove stale per-group files so a group that vanished from the
        # package does not survive from a previous import.
        for path in images_directory.glob("*.json"):
            path.unlink()

    for group_id, members in records_by_group.items():
        activity = members[0].activity_id if members else ""

        _write_json(
            images_directory / f"{group_id}.json",
            {
                "group_id": group_id,
                "activity_id": activity,
                "images": [public_record(member) for member in members],
            },
            where=f"data/imported/gemini/images/{group_id}.json",
        )

    _write_json(
        base / "groups.json",
        {
            "groups": [
                group.model_dump(mode="json")
                for _, group in sorted(result.groups.items())
            ]
        },
        where="data/imported/gemini/groups.json",
    )

    _write_json(
        base / "questions.json",
        {
            "candidates": [
                candidate.model_dump(mode="json")
                for candidate in result.candidates
            ]
        },
        where="data/imported/gemini/questions.json",
    )

    _write_json(
        base / "topics.json",
        {
            "topics": result.topic_index.topics,
            "declared_coverage": result.coverage,
            "unresolved_groups": result.unresolved_topic_groups,
        },
        where="data/imported/gemini/topics.json",
    )

    _write_json(
        base / "knowledge.json",
        {
            "posts": [
                entry.model_dump(mode="json")
                for _, entry in sorted(result.knowledge.items())
            ]
        },
        where="data/imported/gemini/knowledge.json",
    )

    _write_json(
        base / "gaps.json",
        {
            "gaps": [gap.model_dump(mode="json") for gap in result.gaps],
        },
        where="data/imported/gemini/gaps.json",
    )

    _write_json(
        base / "cross_check.json",
        result.cross_check.model_dump(mode="json"),
        where="data/imported/gemini/cross_check.json",
    )

    _write_json(
        base / "manifest.json",
        result.manifest.model_dump(mode="json"),
        where="data/imported/gemini/manifest.json",
    )

    return ImportResult(
        manifest=result.manifest,
        cross_check=result.cross_check,
        records=result.records,
        groups=result.groups,
        candidates=result.candidates,
        topic_index=result.topic_index,
        knowledge=result.knowledge,
        coverage=result.coverage,
        unresolved_topic_groups=result.unresolved_topic_groups,
        gaps=result.gaps,
        wrote=True,
    )


def _with_fingerprint(result: ImportResult, fingerprint: str) -> ImportResult:
    """Attach the archive's cheap fingerprint to the manifest."""

    manifest = result.manifest

    updated = ImportManifest.model_validate(
        {
            **manifest.model_dump(mode="json"),
            "fingerprint": fingerprint,
        }
    )

    return ImportResult(
        manifest=updated,
        cross_check=result.cross_check,
        records=result.records,
        groups=result.groups,
        candidates=result.candidates,
        topic_index=result.topic_index,
        knowledge=result.knowledge,
        coverage=result.coverage,
        unresolved_topic_groups=result.unresolved_topic_groups,
        gaps=result.gaps,
        wrote=result.wrote,
    )


__all__ = [
    "DEFAULT_OUTPUT",
    "public_record",
    "ArchiveError",
    "Gap",
    "GapReason",
    "ImportResult",
    "PackageError",
    "archive_fingerprint",
    "build",
    "gaps_for",
    "load_candidates",
    "load_cross_check",
    "load_gaps",
    "load_knowledge",
    "load_records",
    "load_topic_index",
    "read_manifest",
    "run",
]