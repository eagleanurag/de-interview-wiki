"""
Reading the five package files, and nothing else.

Built from the files' actual structure rather than from a description of
them, because two of them turned out to carry less than their headings
suggest:

* ``LINKEDIN_IMAGE_INVENTORY_COMPLETE.csv`` has one row per image and
  no text column at all. The OCR lives in the knowledge archive, and the
  join between them is the filename.
* ``LINKEDIN_KNOWLEDGE_ARCHIVE_COMPLETE.md`` holds the transcription
  inside a ````text` fence under ``### Exact Transcription``, keyed by
  ``**Filename:**``.

So the inventory is the spine and the markdown is the text, and a
filename that appears in one and not the other is a finding rather than
something to paper over.

The parsers are deliberately tolerant of formatting and intolerant of
content. A heading that moved is not an error; a transcription inside a
fence that will not close is.

The boilerplate problem is handled here rather than in the models:
:data:`VISUAL_DESCRIPTION_BOILERPLATE` and its companion are recognised
and dropped, because they are a statement of method repeated for every
slide and storing them would put a sentence about policy into the
knowledge base once per image.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
from pathlib import Path

from src.gemini.models import (
    GeminiImageRecord,
    GeminiPackageInfo,
    GeminiPostGroup,
    OcrStatus,
)
from src.gemini.ocr_quality import assess, as_search_text
from src.redaction import redact
from src.redaction import summarise as redact_note


INVENTORY_CSV = "LINKEDIN_IMAGE_INVENTORY_COMPLETE.csv"
KNOWLEDGE_MD = "LINKEDIN_KNOWLEDGE_ARCHIVE_COMPLETE.md"
QUESTION_MD = "LINKEDIN_INTERVIEW_QUESTION_BANK.md"
TECHNICAL_MD = "LINKEDIN_TECHNICAL_KNOWLEDGE.md"
TOPIC_INDEX_MD = "LINKEDIN_TOPIC_INDEX.md"

PACKAGE_FILES = (
    INVENTORY_CSV,
    KNOWLEDGE_MD,
    QUESTION_MD,
    TECHNICAL_MD,
    TOPIC_INDEX_MD,
)

#: The two sections every post carries, identically. Recognised so they
#: can be dropped rather than stored.
VISUAL_DESCRIPTION_BOILERPLATE = (
    "Text was machine-transcribed. Non-text structure is not invented "
    "where it was not explicitly reviewed."
)

SOURCE_EXPLANATION_BOILERPLATE = (
    "Limited to the captured source material; external knowledge was not "
    "used to fill omissions."
)

#: The package's own statement of what it did not do. Kept verbatim so
#: this project cannot overstate the package's coverage, and carried
#: into the manifest and the wiki.
DECLARED_CAVEAT = (
    "The machine-readable OCR/transcription pass is complete, but a true "
    "human-style semantic visual review of every one of the unique images "
    "has not been performed individually."
)

_POST_HEADING = re.compile(r"^#\s+(POST-\d+)\s*$")
_FIELD = re.compile(r"^-\s+(?P<label>[^:]+):\s*(?P<value>.*)$")
_SLIDE_HEADING = re.compile(r"^##\s+Slide\s+(?P<number>\d+)\s+—\s+`slide_(?P<digit>\d+)`\s*$")
_FILENAME = re.compile(r"^\*\*Filename:\*\*\s*`(?P<name>[^`]+)`")
_OCR_STATUS = re.compile(r"^\*\*OCR status:\*\*\s*\*\*(?P<status>[A-Z_]+)\*\*")
_TRANSCRIPTION = re.compile(r"^###\s+Exact Transcription\s*$")
_SUBJECT = re.compile(r"^-\s+Subject / heading clue:\s*\*\*(?P<value>.*?)\*\*\s*$")
_CONFIDENCE = re.compile(r"^-\s+(?P<what>Grouping|Ordering) confidence:\s*\*\*(?P<value>.*?)\*\*\s*$")

_SUMMARY_NUMBER = re.compile(r"^-\s+(?P<label>[^:]+):\s*\*\*(?P<value>[\d,]+)\*\*\s*$")

#: How the package's headline labels map onto its own summary bullets.
_SUMMARY_LABELS = {
    "total uploaded images": "claimed_total_images",
    "activity groups/posts": "claimed_groups",
    "ocr available": "claimed_ocr_available",
    "ocr no-text": "claimed_ocr_no_text",
    "ocr failed/time-out": "claimed_ocr_failed",
    "ocr pending": "claimed_ocr_pending",
    "unique meaningful images after exact-byte duplicate removal": (
        "claimed_unique_after_duplicates"
    ),
    "exact duplicate copies": "claimed_exact_duplicates",
}


class PackageError(RuntimeError):
    """The package could not be read as a package."""


def read_package(root: str | Path) -> dict[str, str]:
    """
    Every package file's text.

    Keyed by filename, so a missing file is a key that is absent rather
    than an exception in the middle of parsing. Reported, not guessed
    around.
    """

    base = Path(root).expanduser()

    if not base.is_dir():
        raise PackageError(f"Gemini package not found: {base}")

    contents: dict[str, str] = {}

    missing: list[str] = []

    for name in PACKAGE_FILES:
        path = base / name

        if not path.is_file():
            missing.append(name)
            continue

        contents[name] = path.read_text(encoding="utf-8", errors="replace")

    if missing:
        raise PackageError(
            f"Gemini package at {base} is missing: {', '.join(missing)}"
        )

    return contents


def package_digest(contents: dict[str, str]) -> str:
    """
    One digest over the whole package.

    Over names as well as contents, so a file cannot be swapped for
    another of the same length unnoticed.
    """

    digest = hashlib.sha256()

    for name in sorted(contents):
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(contents[name].encode("utf-8", "replace"))

    return digest.hexdigest()


def package_info_from_text(
    contents: dict[str, str],
) -> GeminiPackageInfo:
    """
    The package's own claims about itself, read from its summary.

    Split from the path-taking wrapper because :func:`parse_knowledge`
    needs these figures and holds no package root, and because a logical
    name is all that may be recorded: these records are committed, so a
    machine's absolute path does not belong in them.
    """

    info = GeminiPackageInfo(declared_caveat=DECLARED_CAVEAT)

    body = contents.get(KNOWLEDGE_MD, "")

    in_summary = False

    for line in body.splitlines():
        stripped = line.strip()

        if stripped.startswith("## Archive Summary"):
            in_summary = True
            continue

        if in_summary and stripped.startswith("#"):
            break

        if not in_summary:
            continue

        match = _SUMMARY_NUMBER.match(stripped)

        if not match:
            continue

        label = match.group("label").strip().lower()

        attribute = _SUMMARY_LABELS.get(label)

        if attribute is None:
            continue

        value = match.group("value").replace(",", "").strip()

        if value.isdigit():
            setattr(info, attribute, int(value))

    return info


def package_info(
    root: str | Path,
    contents: dict[str, str],
) -> GeminiPackageInfo:
    """The package's own claims, with its directory's name attached."""

    base = Path(root).expanduser()

    info = package_info_from_text(contents)

    info.package_name = base.name or info.package_name

    return info


# ---------------------------------------------------------------------
# The inventory
# ---------------------------------------------------------------------


def _ocr_status(value: str) -> OcrStatus:
    """The package's own vocabulary, plus ABSENT for its silence."""

    token = (value or "").strip().upper().replace(" ", "_").replace("-", "_")

    for status in OcrStatus:
        if status.value == token:
            return status

    if not token:
        return OcrStatus.ABSENT

    # An unrecognised status is not silently treated as a known one.
    return OcrStatus.UNRESOLVED


def parse_inventory(contents: dict[str, str]) -> list[GeminiImageRecord]:
    """
    One record per image, from the CSV.

    The CSV has no text column, so ``raw_ocr_text`` is left empty here
    and filled from the knowledge archive by :func:`parse_knowledge`.
    Joining on the filename is the only link the package provides.
    """

    text = contents.get(INVENTORY_CSV, "")

    if not text:
        raise PackageError(f"{INVENTORY_CSV} is empty")

    reader = csv.DictReader(io.StringIO(text))

    expected = {
        "post_id",
        "activity_id",
        "slide_number",
        "filename",
        "ocr_status",
    }

    header = set(reader.fieldnames or [])

    absent = expected - header

    if absent:
        raise PackageError(
            f"{INVENTORY_CSV} lacks the expected columns: "
            f"{', '.join(sorted(absent))}"
        )

    records: list[GeminiImageRecord] = []

    for row in reader:
        filename = (row.get("filename") or "").strip()

        if not filename:
            continue

        slide_raw = (row.get("slide_number") or "").strip()

        try:
            slide = int(slide_raw)

        except ValueError:
            slide = -1

        activity = (row.get("activity_id") or "").strip()

        record = GeminiImageRecord(
            post_id=(row.get("post_id") or "").strip(),
            activity_id=activity,
            filename=filename,
            path=f"media/{filename}",
            slide_number=slide if slide >= 0 else 0,
            slide_number_source=(
                "package" if slide >= 0 else "unresolved"
            ),
            width=_to_int(row.get("width")),
            height=_to_int(row.get("height")),
            size_bytes=_to_int(row.get("size_bytes")) or 0,
            declared_sha256=(row.get("sha256") or "").strip().lower(),
            exact_duplicate_of=(
                (row.get("exact_duplicate_of") or "").strip()
            ),
            likely_preview_of=(
                (row.get("likely_preview_of") or "").strip()
            ),
            ocr_status=_ocr_status(row.get("ocr_status") or ""),
        )

        record = _with_fingerprint(record)

        records.append(record)

    return records


def _to_int(value: object) -> int | None:
    """An integer from a CSV cell, or None when it is not one."""

    text = str(value or "").strip().replace(",", "")

    if not text:
        return None

    try:
        return int(text)

    except ValueError:
        return None


def _with_fingerprint(record: GeminiImageRecord) -> GeminiImageRecord:
    """
    Attach a deterministic fingerprint.

    Over the fields that make the record what it is, so importing an
    unchanged package twice produces byte-identical output and a changed
    transcription is detectable. ``analysed_at`` is deliberately not
    part of it, because that would make every import look like a change.
    """

    digest = hashlib.sha256()

    for value in (
        record.post_id,
        record.activity_id,
        record.filename,
        str(record.slide_number),
        record.declared_sha256,
        record.exact_duplicate_of,
        record.likely_preview_of,
        record.ocr_status.value,
        record.readable_text,
    ):
        digest.update(value.encode("utf-8", "replace"))
        digest.update(b"\x1f")

    return record.model_copy(
        update={"fingerprint": digest.hexdigest()}
    )


# ---------------------------------------------------------------------
# The knowledge archive: where the OCR actually lives
# ---------------------------------------------------------------------


def parse_knowledge(
    contents: dict[str, str],
) -> tuple[dict[str, str], dict[str, GeminiPostGroup], GeminiPackageInfo]:
    """
    Transcriptions and post groups from the knowledge archive.

    Returns the transcription keyed by filename, the groups, and the
    package's own summary claims.

    The transcription is keyed by filename rather than by post and
    slide, because the filename is what the inventory and the archive
    agree on and what the cross-check can verify against disk. A
    post-and-slide key would be a second way to say the same thing and
    the two could disagree.
    """

    body = contents.get(KNOWLEDGE_MD, "")

    transcriptions: dict[str, str] = {}

    groups: dict[str, GeminiPostGroup] = {}

    current_group: GeminiPostGroup | None = None

    current_filename: str | None = None

    # Three states, not two. The transcription sits between an *opening*
    # fence -- which is ```` ```text ````, not a bare ```` ``` ```` -- and a
    # closing one, and treating the opening fence as the closing one
    # captures an empty string for every slide in the package. The extra
    # state is the whole difference between reading the OCR and silently
    # storing nothing.
    awaiting_fence = False
    in_transcription = False

    fence_lines: list[str] = []

    for line in body.splitlines():
        stripped = line.strip()

        heading = _POST_HEADING.match(line)

        if heading:
            current_group = GeminiPostGroup(group_id=heading.group(1))

            groups[current_group.group_id] = current_group

            current_filename = None
            awaiting_fence = False
            in_transcription = False
            fence_lines = []

            continue

        if current_group is None:
            continue

        if _TRANSCRIPTION.match(stripped):
            awaiting_fence = True
            in_transcription = False

            fence_lines = []

            # ``current_filename`` is deliberately *not* cleared here.
            # The document states the filename before the transcription,
            # and clearing it would silently drop every transcription the
            # package ever wrote -- which is all of them.
            continue

        if awaiting_fence:
            # The opening fence, ```` ```text ````. Anything else here
            # means the heading was not followed by a transcription, and
            # leaving the state alone is safer than guessing where one
            # begins.
            if stripped.startswith("```"):
                awaiting_fence = False
                in_transcription = True

            continue

        if in_transcription:
            # A closing fence ends the transcription. An unterminated
            # one runs to the end of the document, which is reported by
            # comparing what was captured against the record count.
            if stripped.startswith("```"):
                in_transcription = False

                if current_filename:
                    transcriptions[current_filename] = "\n".join(
                        fence_lines
                    ).strip()

                current_filename = None
                fence_lines = []

                continue

            fence_lines.append(line)

            continue

        slide_heading = _SLIDE_HEADING.match(line)

        if slide_heading:
            number = int(slide_heading.group("number"))

            current_group.filenames.append(f"__pending_slide_{number}")

            continue

        filename_match = _FILENAME.match(stripped)

        if filename_match:
            current_filename = filename_match.group("name").strip()

            if current_group.filenames:
                pending = current_group.filenames[-1]

                if pending.startswith("__pending_slide_"):
                    number = pending.rsplit("_", 1)[1]

                    current_group.filenames[-1] = (
                        f"{current_filename}#slide={number}"
                    )

            continue

        # The specific patterns are tried before the generic
        # ``- label: value`` one. ``_FIELD`` matches every line in this
        # block -- including ``- Subject / heading clue: **...**`` and
        # ``- Grouping confidence: **HIGH**`` -- and it handled only two
        # of them, so both were silently discarded before their own
        # patterns ever ran.
        subject = _SUBJECT.match(stripped)

        if subject:
            current_group.subject_clue = subject.group("value").strip()

            continue

        confidence = _CONFIDENCE.match(stripped)

        if confidence:
            if confidence.group("what").lower() == "grouping":
                current_group.grouping_confidence = (
                    confidence.group("value").strip()
                )

            else:
                current_group.ordering_confidence = (
                    confidence.group("value").strip()
                )

            continue

        field_match = _FIELD.match(stripped)

        if field_match:
            label = field_match.group("label").strip().lower()
            value = field_match.group("value").strip().strip("`").strip()

            if label == "activity id":
                current_group.activity_id = value

            elif label == "estimated images/slides":
                digits = re.sub(r"[^\d]", "", value)

                if digits:
                    current_group.declared_slide_count = int(digits)

            continue
    # Restore plain filenames from the temporary slide marker.
    for group in groups.values():
        restored: list[str] = []

        for entry in group.filenames:
            if "#slide=" in entry:
                restored.append(entry.split("#slide=", 1)[0])

            elif not entry.startswith("__pending_slide_"):
                restored.append(entry)

        group.filenames = restored
        group.actual_slide_count = len(restored)

    return transcriptions, groups, package_info_from_text(contents)


def is_boilerplate(text: str | None) -> bool:
    """
    Whether a section is the package's constant method statement.

    Both of the package's prose sections repeat verbatim for every
    slide. Storing them would attribute a sentence about method to a
    specific image, once per image.
    """

    if not text:
        return True

    body = " ".join(text.split()).strip().lower()

    if not body:
        return True

    for boilerplate in (
        VISUAL_DESCRIPTION_BOILERPLATE,
        SOURCE_EXPLANATION_BOILERPLATE,
    ):
        if " ".join(boilerplate.split()).lower() in body:
            return True

    return False


def attach_transcriptions(
    records: list[GeminiImageRecord],
    transcriptions: dict[str, str],
) -> list[GeminiImageRecord]:
    """
    Give each record its transcription, and judge it.

    Records the package never transcribed keep their ``ABSENT`` status.
    That is different from ``NO_TEXT``, which is a claim about an image
    somebody looked at, and collapsing the two would hide 114 findings
    behind 2,782 silences.
    """

    attached: list[GeminiImageRecord] = []

    for record in records:
        text = transcriptions.get(record.filename)

        if text is None:
            if record.ocr_status is OcrStatus.AVAILABLE:
                # The inventory says the text exists and the archive
                # does not have it. Recorded rather than assumed present.
                attached.append(
                    record.model_copy(
                        update={
                            "ocr_status": OcrStatus.UNRESOLVED,
                            "verdict_note": (
                                "the inventory reported AVAILABLE text "
                                "but the knowledge archive holds no "
                                "transcription for this filename"
                            ),
                            "raw_ocr_text": None,
                            "public_text": None,
                        }
                    )
                )

            else:
                # The inventory already stated a status for this image
                # and the markdown simply has no transcription for it.
                # That status is a finding about the image -- NO_TEXT,
                # FAILED, PENDING all assert something -- so it is kept.
                # Overwriting it with ABSENT would turn "somebody looked
                # and found nothing" into "nobody looked", which is the
                # one distinction this function exists to preserve.
                #
                # ABSENT is therefore left as it stands: it is what an
                # empty status column already parses to.
                attached.append(
                    record.model_copy(
                        update={
                            "raw_ocr_text": None,
                            "public_text": None,
                        }
                    )
                )

            continue

        if record.ocr_status is OcrStatus.AVAILABLE and (
            not text.strip() or is_boilerplate(text)
        ):
            # The package claimed text and supplied either nothing or a
            # restatement of its own method statement. Either way there
            # is no transcription here, and storing the method sentence
            # as one would be the worst of the three outcomes.
            attached.append(
                record.model_copy(
                    update={
                        "raw_ocr_text": None,
                            "public_text": None,
                        "ocr_status": OcrStatus.NO_TEXT,
                        "ocr_note": (
                            "the inventory reported AVAILABLE but the "
                            "supplied text was empty or a restatement "
                            "of the package's own method statement"
                        ),
                    }
                )
            )

            continue

        quality, evidence = assess(text)

        public, redactions = redact(text)

        note = evidence

        if redactions:
            note = f"{evidence}; {redact_note(redactions)}"

        attached.append(
            _with_fingerprint(
                record.model_copy(
                    update={
                        "raw_ocr_text": text,
                        "public_text": public,
                        "redactions": redactions,
                        "search_text": as_search_text(public),
                        "ocr_quality": quality,
                        "ocr_note": note,
                    }
                )
            )
        )

    return attached


def groups_from_records(
    records: list[GeminiImageRecord],
) -> dict[str, GeminiPostGroup]:
    """
    Rebuild the groups from the inventory rather than the markdown.

    The markdown's grouping is prose with counts in it; the inventory is
    one row per image with a group column. Where the two disagree, the
    inventory wins and the disagreement is visible, because a group is
    defined by which rows carry its identifier.
    """

    grouped: dict[str, GeminiPostGroup] = {}

    for record in records:
        group = grouped.get(record.post_id)

        if group is None:
            group = GeminiPostGroup(
                group_id=record.post_id,
                activity_id=record.activity_id,
            )

            grouped[record.post_id] = group

        group.filenames.append(record.filename)

    for group in grouped.values():
        group.actual_slide_count = len(group.filenames)

    return grouped


__all__ = [
    "DECLARED_CAVEAT",
    "INVENTORY_CSV",
    "KNOWLEDGE_MD",
    "PACKAGE_FILES",
    "QUESTION_MD",
    "SOURCE_EXPLANATION_BOILERPLATE",
    "TECHNICAL_MD",
    "TOPIC_INDEX_MD",
    "VISUAL_DESCRIPTION_BOILERPLATE",
    "GeminiPackageInfo",
    "PackageError",
    "attach_transcriptions",
    "groups_from_records",
    "is_boilerplate",
    "package_digest",
    "package_info",
    "package_info_from_text",
    "parse_inventory",
    "parse_knowledge",
    "read_package",
]
