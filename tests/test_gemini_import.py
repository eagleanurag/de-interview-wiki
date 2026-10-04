"""
The Gemini package as an import: parsing, cross-checking, the question
gate, provenance, search, rendering and idempotency.

Built on synthetic packages and synthetic archives, for the same reasons
``test_visual_enrichment.py`` gives. The real package describes 3,047
real images; a test that asserts things about those specific files cannot
express a rule, it can only confirm that they still exist. What these
tests pin down is behaviour: that a bare link is rejected, that a
column merge is not guessed at, that raw transcription survives
byte-for-byte, and that importing twice writes the same bytes twice.

Where a real excerpt from the package is used it is quoted exactly,
because these are the cases the design was built around. ``cn oh? Azure``
is a real candidate question, ``Describe the Explain the TASK`` is a real
column merge, and the two sentences of constant prose really do repeat
for every single slide.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.aggregation.consolidation import (
    DerivedClaim,
    TechnologyNode,
    consolidate,
)
from src.ai.enricher import WINDOWS_COMMAND_LINE_LIMIT
from src.gemini import bridge, derived, importer, ocr_quality, questions
from src.gemini.importer import public_record
from src.redaction import redact
from src.redaction import summarise as redact_note
from src.gemini.crosscheck import (
    ArchiveError,
    activity_id_for_filename,
    contained,
    cross_check,
    index_archive,
    load_archive_posts,
    slide_number_for_filename,
)
from src.gemini.importer import GapReason
from src.gemini.models import (
    CandidateVerdict,
    GeminiCandidateQuestion,
    GeminiImageRecord,
    MatchVerdict,
    OcrQuality,
    OcrStatus,
    SourceKind,
)
from src.gemini.package_reader import (
    PackageError,
    attach_transcriptions,
    groups_from_records,
    package_digest,
    parse_inventory,
    parse_knowledge,
    read_package,
)
from src.gemini.questions import build_candidates
from src.gemini.safety import PublicOutputError, assert_public, find_local_paths
from src.gemini.taxonomy import (
    parse_technical,
    parse_topic_index,
    resolve_fragments,
    unmatched_groups,
)
from src.models import InterviewQuestion, KnowledgePost, MediaItem, SourceInfo
from src.wiki.ocr_index import build_ocr_index, build_ocr_records, excerpt


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

INVENTORY_HEADER = (
    "post_id,activity_id,slide_number,filename,width,height,"
    "size_bytes,sha256,exact_duplicate_of,likely_preview_of,ocr_status"
)

GOOD_OCR = """USE THE S.T.A.R TECHNIQUE!

Describe the Explain the TASK
SITUATION you that needed to be
were in. done.
Detail the ACTION Reveal the
you took to achieve RESULTS of your
the task. actions.

Link 1: https://www_youtube.com/watch?v=lfN8RDA7kVA&t=1090s"""

SQL_OCR = """SQL JOINS

INNER JOIN B ON A.key = B.key

SELECT A.id, B.value
FROM A
INNER JOIN B ON A.key = B.key
WHERE B.value > 0"""

#: Verbatim from the package. The letters are not words; this is what a
#: 480x360 screenshot of a data-tools menu comes back as.
GARBLED_OCR = """bree ondey AEST
= "Data WERGRGUES ve KEKE vs
Lake House vs Mesh
-~_ » gt. ™
e-* 2%
Data Lake oi
ae
82% 8... ene eee"""

#: A two-column statement, as OCR renders it when the columns are run
#: together. Shaped like the package's ``Describe the Explain the TASK``:
#: two genuine question openings, nothing between them, and the source
#: interleaved too.
MERGED_OCR = """Explain the MERGE INTO statement Explain the target table
the source rows you the MERGE matches on
where they join what happens on a match"""

#: Same diagram transcribed correctly, which is what the source file
#: says when the columns were read one at a time.
MERGED_SOURCE = """Explain the MERGE INTO statement that merges the source
rows into the target table where they join on a matching key"""


#: A 64-character digest, as the inventory carries one.
SHA = "a" * 64


def inventory_row(
    post_id: str = "POST-001",
    activity_id: str = "7263033731471351808",
    slide: int = 0,
    filename: str | None = None,
    width: int = 800,
    height: int = 420,
    size: int = 146149,
    sha256: str = SHA,
    duplicate_of: str = "",
    preview_of: str = "",
    status: str = "AVAILABLE",
) -> str:
    """
    One inventory row, built rather than concatenated.

    Written this way after a fixture bug that is worth remembering:
    implicit string concatenation binds tighter than ``*``, so a
    multi-line ``"...jpg,800,420,"  "a" * 64`` is one literal repeated
    sixty-four times, not a literal followed by sixty-four a's. The row
    looked plausible and every field after the seventh was wrong.
    """

    name = filename or (
        f"activity_{activity_id}_slide_{slide}.jpg"
    )

    return ",".join(
        [
            post_id,
            activity_id,
            str(slide),
            name,
            str(width),
            str(height),
            str(size),
            sha256,
            duplicate_of,
            preview_of,
            status,
        ]
    )


def write_package(
    root: Path,
    *,
    inventory_rows: list[str] | None = None,
    knowledge: str | None = None,
    question_bank: str | None = None,
    technical: str | None = None,
    topic_index: str | None = None,
) -> Path:
    """
    A package on disk, in the shape the real one has.

    Every file is written even when the caller supplies no content for
    it, because ``read_package`` requires all five and a missing file is a
    different situation from an empty one.
    """

    root.mkdir(parents=True, exist_ok=True)

    rows = (
        inventory_rows
        if inventory_rows is not None
        else [inventory_row()]
    )

    (root / "LINKEDIN_IMAGE_INVENTORY_COMPLETE.csv").write_text(
        INVENTORY_HEADER + "\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )

    (root / "LINKEDIN_KNOWLEDGE_ARCHIVE_COMPLETE.md").write_text(
        knowledge if knowledge is not None else DEFAULT_KNOWLEDGE,
        encoding="utf-8",
    )

    (root / "LINKEDIN_INTERVIEW_QUESTION_BANK.md").write_text(
        question_bank if question_bank is not None else "",
        encoding="utf-8",
    )

    (root / "LINKEDIN_TECHNICAL_KNOWLEDGE.md").write_text(
        technical if technical is not None else "",
        encoding="utf-8",
    )

    (root / "LINKEDIN_TOPIC_INDEX.md").write_text(
        topic_index if topic_index is not None else "",
        encoding="utf-8",
    )

    return root


DEFAULT_KNOWLEDGE = f"""# LINKEDIN KNOWLEDGE ARCHIVE

## Archive Status

Stable post grouping and numeric slide ordering are preserved.

## Archive Summary

- Total uploaded images: **3,047**
- Activity groups/posts: **312**
- Standalone activity groups: **247**
- Multi-image groups: **65**
- Unique meaningful images after exact-byte duplicate removal: **2,782**
- Exact duplicate copies: **265**
- OCR available: **2,668**
- OCR no-text: **114**
- OCR failed/time-out: **0**
- OCR pending: **0**

---

# POST-001

## Source Identification

- Activity ID: `7263033731471351808`
- Estimated images/slides: **1**
- Subject / heading clue: **USE THE S.T.A.R TECHNIQUE!**
- Author/source: **Not confidently separable from the media set; see exact transcription.**
- Grouping confidence: **HIGH**
- Ordering confidence: **HIGH**

## Slide 1 — `slide_0`

**Filename:** `activity_7263033731471351808_slide_0.jpg`
**Dimensions:** 800 × 420 px
**Size:** 146,149 bytes
**OCR status:** **AVAILABLE**

### Exact Transcription

```text
{GOOD_OCR}
```

### Visual Description

Text was machine-transcribed. Non-text structure is not invented where it was not explicitly reviewed.

### Source-Derived Explanation

Limited to the captured source material; external knowledge was not used to fill omissions.

---
"""


def write_archive(
    root: Path,
    files: dict[str, bytes],
    posts: dict[str, dict] | None = None,
) -> Path:
    """
    A minimal archive: a metadata file and a media directory.

    Read-only as far as this test suite is concerned. Nothing here writes
    to an archive after creating it, which is the property the real one
    is required to have.
    """

    media = root / "media"
    media.mkdir(parents=True, exist_ok=True)

    for name, payload in files.items():
        (media / name).write_bytes(payload)

    records = []

    for post_id, body in (posts or {}).items():
        records.append(
            {
                "post_id": post_id,
                "text": body.get("text", ""),
                "media": {"saved_files": body.get("files", [])},
            }
        )

    (root / "posts_archive.json").write_text(
        json.dumps({"posts": records}), encoding="utf-8"
    )

    return root


def png_bytes(colour: bytes = b"\x89PNG\r\n\x1a\n") -> bytes:
    """A file named ``.jpg`` that is not one, which the archive has."""

    return colour + b"not really an image"


def make_post(identifier: str, paths: list[str]) -> KnowledgePost:
    """A committed post with the given media, and nothing else."""

    return KnowledgePost(
        id=identifier,
        source=SourceInfo(
            platform="linkedin",
            captured_at=datetime.now(timezone.utc),
        ),
        original_text="A post about joins.",
        media=[MediaItem(type="image", path=path) for path in paths],
    )


# ---------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------


def test_the_inventory_csv_is_parsed_into_one_record_per_row(tmp_path):
    package = write_package(
        tmp_path / "package",
        inventory_rows=[
            "POST-001,7263033731471351808,0,"
            "activity_7263033731471351808_slide_0.jpg,800,420,146149,"
            + "a" * 64
            + ",,,AVAILABLE",
            "POST-004,7287036149515112448,1,"
            "activity_7287036149515112448_slide_01.jpg,1200,1000,807866,"
            + "b" * 64
            + ",,,NO_TEXT",
        ],
    )

    records = parse_inventory(read_package(package))

    assert len(records) == 2

    first, second = records

    assert first.post_id == "POST-001"
    assert first.activity_id == "7263033731471351808"
    assert first.slide_number == 0
    assert first.slide_number_source == "package"
    assert first.ocr_status is OcrStatus.AVAILABLE
    assert first.width == 800
    assert first.height == 420
    assert first.size_bytes == 146149
    assert first.declared_sha256 == "a" * 64

    # Inconsistent zero padding on one post, exactly as the archive has
    # it. The CSV's own number wins and the filename is not re-parsed.
    assert second.slide_number == 1
    assert second.filename.endswith("_slide_01.jpg")
    assert second.ocr_status is OcrStatus.NO_TEXT


def test_a_non_numeric_slide_number_is_recorded_as_unresolved(tmp_path):
    package = write_package(
        tmp_path / "package",
        inventory_rows=[
            "POST-001,7263033731471351808,?,broken.jpg,0,0,0,,,,AVAILABLE"
        ],
    )

    records = parse_inventory(read_package(package))

    assert records[0].slide_number_source == "unresolved"


def test_an_unrecognised_ocr_status_becomes_unresolved_not_available(tmp_path):
    package = write_package(
        tmp_path / "package",
        inventory_rows=[
            "POST-001,7263033731471351808,0,a.jpg,1,1,1,"
            + "c" * 64
            + ",,,SOMETHING_NEW"
        ],
    )

    records = parse_inventory(read_package(package))

    # Treating an unknown state as a known one would report coverage the
    # package never claimed.
    assert records[0].ocr_status is OcrStatus.UNRESOLVED


def test_an_inventory_without_its_expected_columns_is_refused(tmp_path):
    """
    Reading the package and parsing it are separate steps.

    ``read_package`` checks the five files are present; the column check
    belongs to ``parse_inventory``, which is where the schema is
    actually interpreted. A file that exists but cannot be read as the
    inventory it claims to be has to fail at the parse, not be silently
    accepted as a package with no images.
    """

    package = tmp_path / "package"
    package.mkdir()

    (package / "LINKEDIN_IMAGE_INVENTORY_COMPLETE.csv").write_text(
        "post_id,filename\nPOST-001,a.jpg\n", encoding="utf-8"
    )

    for name in (
        "LINKEDIN_KNOWLEDGE_ARCHIVE_COMPLETE.md",
        "LINKEDIN_INTERVIEW_QUESTION_BANK.md",
        "LINKEDIN_TECHNICAL_KNOWLEDGE.md",
        "LINKEDIN_TOPIC_INDEX.md",
    ):
        (package / name).write_text("", encoding="utf-8")

    # All five files are present, so reading succeeds.
    contents = read_package(package)

    with pytest.raises(PackageError) as error:
        parse_inventory(contents)

    assert "activity_id" in str(error.value)


def test_a_missing_package_file_is_named_rather_than_guessed_around(tmp_path):
    package = write_package(tmp_path / "package")

    (package / "LINKEDIN_TOPIC_INDEX.md").unlink()

    with pytest.raises(PackageError) as error:
        read_package(package)

    assert "LINKEDIN_TOPIC_INDEX.md" in str(error.value)


def test_the_package_digest_covers_names_as_well_as_contents(tmp_path):
    first = write_package(tmp_path / "one")
    second = write_package(tmp_path / "two")

    left = package_digest(read_package(first))
    right = package_digest(read_package(second))

    assert left == right

    # A file that changed changes the digest, which is what makes a
    # skipped re-import safe rather than merely convenient.
    (second / "LINKEDIN_TOPIC_INDEX.md").write_text(
        "## SQL\n\nPOST-001\n", encoding="utf-8"
    )

    assert package_digest(read_package(second)) != left


# ---------------------------------------------------------------------
# The knowledge archive, and where the OCR actually lives
# ---------------------------------------------------------------------


def test_transcriptions_are_keyed_by_filename_not_by_post_and_slide(tmp_path):
    package = write_package(tmp_path / "package")

    transcriptions, groups, _ = parse_knowledge(read_package(package))

    assert "activity_7263033731471351808_slide_0.jpg" in transcriptions

    assert "USE THE S.T.A.R TECHNIQUE!" in (
        transcriptions["activity_7263033731471351808_slide_0.jpg"]
    )

    group = groups["POST-001"]

    assert group.activity_id == "7263033731471351808"
    assert group.declared_slide_count == 1
    assert group.subject_clue == "USE THE S.T.A.R TECHNIQUE!"
    assert group.ordering_confidence == "HIGH"


def test_a_transcription_keeps_every_line_exactly_as_written(tmp_path):
    package = write_package(tmp_path / "package")

    transcriptions, _, _ = parse_knowledge(read_package(package))

    body = transcriptions["activity_7263033731471351808_slide_0.jpg"]

    # Character for character, including the blank lines and the URL.
    # A transcription that has been tidied is no longer evidence of what
    # was on the slide.
    assert body == GOOD_OCR


def test_the_packages_own_summary_figures_are_kept_verbatim(tmp_path):
    package = write_package(tmp_path / "package")

    _, _, info = parse_knowledge(read_package(package))

    assert info.claimed_total_images == 3047
    assert info.claimed_groups == 312
    assert info.claimed_ocr_available == 2668
    assert info.claimed_ocr_no_text == 114
    assert info.claimed_exact_duplicates == 265
    assert info.claimed_unique_after_duplicates == 2782

    # The package states its own limit, and it is carried rather than
    # paraphrased so this project cannot overstate its coverage.
    assert "human-style semantic visual review" in info.declared_caveat


def test_the_constant_prose_sections_are_not_stored_as_descriptions(tmp_path):
    package = write_package(tmp_path / "package")

    records = attach_transcriptions(
        parse_inventory(read_package(package)),
        parse_knowledge(read_package(package))[0],
    )

    outcome = bridge.outcome_for(
        "post",
        records,
        "2026-01-01T00:00:00+00:00",
    )

    for analysis in outcome.visual.analyses:
        # Every post in the package carries the same two sentences about
        # method. Storing them would put a sentence about policy into
        # the knowledge base once per image.
        assert analysis.visual_summary is None
        assert analysis.diagram_description is None


def test_a_record_the_package_never_transcribed_is_absent_not_no_text(tmp_path):
    package = write_package(
        tmp_path / "package",
        inventory_rows=[
            "POST-001,7263033731471351808,0,a.jpg,1,1,1," + "a" * 64 + ",,,NO_TEXT"
        ],
    )

    records = attach_transcriptions(
        parse_inventory(read_package(package)), {}
    )

    # "There is no text here" and "nobody looked" are different facts,
    # and only the first is a finding about the image.
    assert records[0].ocr_status is OcrStatus.NO_TEXT


def test_available_in_the_inventory_but_absent_from_the_archive_is_a_finding(
    tmp_path,
):
    package = write_package(tmp_path / "package")

    records = attach_transcriptions(
        parse_inventory(read_package(package)), {}
    )

    assert records[0].ocr_status is OcrStatus.UNRESOLVED

    assert "no transcription for this filename" in (
        records[0].verdict_note
    )


def test_reimporting_the_same_package_produces_identical_fingerprints(tmp_path):
    package = write_package(tmp_path / "package")

    contents = read_package(package)

    first = attach_transcriptions(
        parse_inventory(contents), parse_knowledge(contents)[0]
    )

    second = attach_transcriptions(
        parse_inventory(contents), parse_knowledge(contents)[0]
    )

    assert [r.fingerprint for r in first] == [r.fingerprint for r in second]

    assert all(record.fingerprint for record in first)


# ---------------------------------------------------------------------
# OCR quality: named states, never a score
# ---------------------------------------------------------------------


def test_coherent_prose_is_readable():
    quality, evidence = ocr_quality.assess(SQL_OCR)

    assert quality is OcrQuality.READABLE
    assert "words" in evidence


def test_misread_english_is_not_detectable_as_garbled():
    """
    The package's real symbol-noise sample, and what it actually is.

    This test exists to stop the mistake being made twice. The obvious
    approach is a letter-share floor, and it is wrong: measured, this
    sample is 61 per cent letters and 17 per cent symbols, while the
    coherent SQL transcription is 79 and 9. The noise scores *better* on
    letter share, symbol share, short-word share, average word length and
    longest word run. OCR failing on a small screenshot produces
    confidently misread letters, which are the same shape as real ones.

    So it is not called garbled, and the wiki shows it as a transcription
    with no observable defect rather than claiming it is correct.
    """

    quality, evidence = ocr_quality.assess(GARBLED_OCR)

    assert quality is OcrQuality.READABLE
    assert "letters" in evidence
    assert "no observable defect" in ocr_quality.limitations()


def test_nothing_transcribed_is_empty():
    assert ocr_quality.assess(None)[0] is OcrQuality.EMPTY
    assert ocr_quality.assess("   ")[0] is OcrQuality.EMPTY


def test_a_repeated_line_is_detected_although_ordinary_noise_is_not():
    # The one symbol-free signal that does indicate a failed read.
    quality, evidence = ocr_quality.assess(
        "delta lake delta lake delta lake delta lake delta lake "
        "delta lake delta lake delta lake delta lake delta lake"
    )

    assert quality is OcrQuality.GARBLED
    assert "repeats" in evidence


def test_extreme_symbol_load_is_detected():
    quality, evidence = ocr_quality.assess(
        "» ™ © ® ± ≠ ≥ ≤ ƒ ∞ ≈ ∑ ∏ √ ∫ ∂ — " * 6
    )

    assert quality is OcrQuality.GARBLED
    assert "symbol" in evidence


def test_a_handful_of_words_is_fragmentary_not_garbled():
    quality, evidence = ocr_quality.assess("Git tool")

    # Real words, too few to carry a topic. That is a statement about
    # density, and it is one character analysis can actually support.
    assert quality is OcrQuality.FRAGMENTARY
    assert "too few" in evidence


def test_search_text_collapses_whitespace_and_nothing_else():
    body = "SELECT  A.id,\n\n  B.value\tFROM  A"

    # Code punctuation and case survive. Stripping parentheses or
    # lowercasing would break a search for COUNT(DISTINCT x), and these
    # slides are full of code.
    assert ocr_quality.as_search_text(body) == "SELECT A.id, B.value FROM A"


def test_only_empty_and_garbled_slides_are_offered_to_the_fallback():
    assert ocr_quality.needs_reprocessing(OcrQuality.EMPTY)
    assert ocr_quality.needs_reprocessing(OcrQuality.GARBLED)

    # A fragmentary slide is real text from a real picture. Spending a
    # model call to confirm a small diagram has three labels on it is how
    # a budget disappears.
    assert not ocr_quality.needs_reprocessing(OcrQuality.FRAGMENTARY)
    assert not ocr_quality.needs_reprocessing(OcrQuality.READABLE)


# ---------------------------------------------------------------------
# Cross-check against the archive
# ---------------------------------------------------------------------


def test_an_activity_is_recovered_from_a_filename():
    assert (
        activity_id_for_filename("activity_7263033731471351808_slide_0.jpg")
        == "7263033731471351808"
    )

    assert (
        slide_number_for_filename("activity_7263033731471351808_slide_01.jpg")
        == 1
    )


def test_the_same_post_numbered_two_ways_still_yields_two_slides(tmp_path):
    package = write_package(
        tmp_path / "package",
        inventory_rows=[
            "POST-004,7287036149515112448,0,"
            "activity_7287036149515112448_slide_0.jpg,480,360,30027,"
            + "d" * 64
            + ",,activity_7287036149515112448_slide_10.jpg,AVAILABLE",
            "POST-004,7287036149515112448,1,"
            "activity_7287036149515112448_slide_01.jpg,1200,1000,807866,"
            + "e" * 64
            + ",,,AVAILABLE",
        ],
    )

    records = parse_inventory(read_package(package))

    groups = groups_from_records(records)

    assert groups["POST-004"].actual_slide_count == 2


def test_a_file_the_archive_does_not_have_is_reported_not_dropped():
    record = GeminiImageRecord(
        post_id="POST-001",
        filename="absent.jpg",
        declared_sha256="f" * 64,
    )

    checked, report = cross_check([record], {})

    assert len(checked) == 1
    assert checked[0].verdict is MatchVerdict.MISSING_FROM_ARCHIVE
    assert report.missing_from_archive == 1


def test_a_digest_that_disagrees_with_the_archive_is_a_conflict():
    record = GeminiImageRecord(
        post_id="POST-001",
        activity_id="7263033731471351808",
        filename="a.jpg",
        declared_sha256="a" * 64,
    )

    checked, report = cross_check(
        [record],
        {"a.jpg": {"filename": "a.jpg", "sha256": "b" * 64,
                   "size_bytes": 0, "width": None, "height": None}},
    )

    # Kept, with the disagreement written beside it. Correcting the
    # record would hide the only evidence that the two ever disagreed.
    assert checked[0].verdict is MatchVerdict.CONFLICTING
    assert "digest differs" in checked[0].verdict_note
    assert report.sha256_mismatches


def test_a_dangling_duplicate_target_is_reported():
    record = GeminiImageRecord(
        post_id="POST-001",
        activity_id="7263033731471351808",
        filename="a.jpg",
        exact_duplicate_of="nowhere.jpg",
    )

    checked, report = cross_check(
        [record],
        {"a.jpg": {"filename": "a.jpg", "sha256": "a" * 64,
                   "size_bytes": 0, "width": None, "height": None}},
    )

    assert report.dangling_duplicate_targets
    assert "duplicate target names no file" in checked[0].verdict_note


def test_a_dangling_preview_target_is_reported():
    record = GeminiImageRecord(
        post_id="POST-001",
        activity_id="7263033731471351808",
        filename="a.jpg",
        likely_preview_of="nowhere.jpg",
    )

    checked, report = cross_check(
        [record],
        {"a.jpg": {"filename": "a.jpg", "sha256": "a" * 64,
                   "size_bytes": 0, "width": None, "height": None}},
    )

    assert report.dangling_preview_targets


def test_archive_media_the_package_never_mentioned_is_counted(tmp_path):
    records = [
        GeminiImageRecord(
            post_id="POST-001",
            activity_id="7263033731471351808",
            filename="known.jpg",
        )
    ]

    index = {
        "known.jpg": {"filename": "known.jpg", "sha256": "a" * 64,
                      "size_bytes": 0, "width": None, "height": None},
        "unknown.jpg": {"filename": "unknown.jpg", "sha256": "b" * 64,
                        "size_bytes": 0, "width": None, "height": None},
    }

    _, report = cross_check(records, index)

    assert report.archive_files_absent_from_package == 1
    assert report.absent_filenames == ["unknown.jpg"]


def test_an_archive_with_no_metadata_file_is_an_error_not_an_empty_map(tmp_path):
    with pytest.raises(ArchiveError):
        load_archive_posts(tmp_path)


def test_the_archive_is_only_ever_read(tmp_path):
    archive = write_archive(
        tmp_path / "archive",
        {"a.jpg": png_bytes()},
        {"7263033731471351808": {"files": ["a.jpg"]}},
    )

    before = {
        path.name: (path.stat().st_mtime_ns, path.stat().st_size)
        for path in sorted((archive / "media").iterdir())
    }

    index_archive(archive)
    load_archive_posts(archive)

    after = {
        path.name: (path.stat().st_mtime_ns, path.stat().st_size)
        for path in sorted((archive / "media").iterdir())
    }

    assert before == after


def test_a_path_escaping_the_media_root_is_refused(tmp_path):
    media = tmp_path / "media"
    media.mkdir()

    # The archive's own metadata is data, not instructions, and a path
    # like this resolves outside the root before any comparison happens.
    for candidate in (
        "../outside.jpg",
        "..\\outside.jpg",
        str(tmp_path / "outside.jpg"),
    ):
        with pytest.raises(ArchiveError):
            contained(media, candidate)


def test_a_path_inside_the_media_root_is_accepted(tmp_path):
    media = tmp_path / "media"
    media.mkdir()

    (media / "a.jpg").write_bytes(b"x")

    assert contained(media, "a.jpg").name == "a.jpg"


# ---------------------------------------------------------------------
# The question bank
# ---------------------------------------------------------------------


def bank(*entries: str) -> str:
    """A candidate bank in the real file's format."""

    body = ["# INTERVIEW QUESTION BANK", ""]

    for number, entry in enumerate(entries, start=1):
        body.extend(
            [
                f"## Question {number}",
                "",
                entry.strip(),
                "",
                "---",
                "",
            ]
        )

    return "\n".join(body)


def entry(
    question: str,
    *,
    excerpt: str = "",
    source: str = "POST-001, slide 0",
    declared_type: str = "Theory/Scenario",
    declared_difficulty: str = "Easy",
) -> str:
    return f"""**Question:** {question}

**Type:** {declared_type}

**Difficulty:** {declared_difficulty}

**Answer / source-bounded response:** {excerpt or question}

**Source:** {source}"""


def test_a_bare_link_is_rejected(tmp_path):
    package = write_package(
        tmp_path / "package",
        question_bank=bank(
            entry(
                "Link 1: https://www_youtube.com/watch?v=lfN8RDA7kVA&t=1090s",
                excerpt=GOOD_OCR,
            )
        ),
    )

    records = attach_transcriptions(
        parse_inventory(read_package(package)),
        parse_knowledge(read_package(package))[0],
    )

    built = build_candidates(read_package(package), records)

    assert built[0].verdict is CandidateVerdict.REJECT

    # Kept in full, so the rejection can be argued with.
    assert built[0].candidate_text.startswith("Link 1:")
    assert questions.accepted(built) == []


def test_symbol_noise_with_one_real_word_is_rejected(tmp_path):
    package = write_package(
        tmp_path / "package",
        question_bank=bank(entry("cn oh? Azure", excerpt=GARBLED_OCR)),
    )

    records = attach_transcriptions(
        parse_inventory(read_package(package)),
        parse_knowledge(read_package(package))[0],
    )

    built = build_candidates(read_package(package), records)

    assert built[0].verdict is CandidateVerdict.REJECT

    # The '?' came from OCR debris. Requiring a recognised opener rather
    # than a question mark is what keeps this out; a mark the writer did
    # not choose is not evidence of a question.
    assert "not question-like" in built[0].verdict_reason
    assert not questions.is_question_like("cn oh? Azure")[0]


def test_a_readable_question_is_accepted(tmp_path):
    package = write_package(
        tmp_path / "package",
        question_bank=bank(
            entry(
                "What is an INNER JOIN and when should you use one?",
                excerpt=SQL_OCR,
            )
        ),
    )

    records = attach_transcriptions(
        parse_inventory(read_package(package)),
        parse_knowledge(read_package(package))[0],
    )

    built = build_candidates(read_package(package), records)

    assert built[0].verdict is CandidateVerdict.ACCEPT
    assert built[0].question.startswith("What is an INNER JOIN")
    assert built[0].source_resolved is True


def test_a_column_merge_is_not_guessed_at(tmp_path):
    package = write_package(
        tmp_path / "package",
        question_bank=bank(
            entry("Describe the Explain the TASK", excerpt=GOOD_OCR)
        ),
    )

    records = attach_transcriptions(
        parse_inventory(read_package(package)),
        parse_knowledge(read_package(package))[0],
    )

    built = build_candidates(read_package(package), records)

    # Both readings are genuine, and only one appears verbatim in the
    # source. The honest destination is a person, not a confident guess
    # at what the speaker was going to say.
    assert built[0].verdict in {
        CandidateVerdict.REWRITE,
        CandidateVerdict.NEEDS_REVIEW,
    }

    if built[0].verdict is CandidateVerdict.REWRITE:
        # If it was rewritten, the rewrite must be in the source.
        assert "explain the task" in GOOD_OCR.lower()

    assert built[0].candidate_text == "Describe the Explain the TASK"


def test_a_merge_the_source_supports_is_narrowed_to_that_clause():
    """
    The real ``Describe the Explain the TASK`` case, end to end.

    Two genuine question openings with nothing between them is the
    observable signal that OCR ran two diagram cells together. The
    surviving clause is accepted only because ``explain the task``
    appears verbatim in the source transcription -- which is the whole
    difference between a repair and a guess.
    """

    raw = questions.RawEntry(1)

    raw.set("question", "Describe the Explain the TASK")

    verdict, _reason, question, note = questions.judge(raw, GOOD_OCR)

    assert verdict is CandidateVerdict.REWRITE

    # The surviving clause keeps the source's own capitalisation. Only
    # the leading letter is normalised; the words are not edited.
    assert question == "Explain the TASK"
    assert "columns together" in note
    assert "explain the task" in GOOD_OCR.lower()


def test_a_merge_the_source_does_not_support_is_left_for_a_person():
    raw = questions.RawEntry(1)

    # The same text, against a transcription that does not contain the
    # surviving clause. Nothing to narrow to that is not a guess, so the
    # entry waits for a person -- and must not be published as-is, which
    # is the failure this gate exists to prevent.
    raw.set("question", "Describe the Explain the TASK")

    verdict, reason, question, _note = questions.judge(raw, MERGED_SOURCE)

    assert verdict is CandidateVerdict.NEEDS_REVIEW
    assert question is None
    assert raw.get("question") == "Describe the Explain the TASK"
    assert "diagram columns together" in reason


def test_a_merged_fragment_is_never_published_as_a_question():
    """
    The measure that would have let it through.

    ``Describe the Explain the TASK`` is 86 per cent letters, so the
    quality assessment calls it readable. That is correct as a statement
    about letters and wrong as a statement about a question, which is
    why the merge check is separate from the quality thresholds.
    """

    assert questions.suspicious_merge("Describe the Explain the TASK")

    assert ocr_quality.assess("Describe the Explain the TASK")[
        0
    ] is OcrQuality.READABLE


def test_two_openers_in_an_ordinary_sentence_are_not_a_merge():
    assert not questions.suspicious_merge(
        "What is an INNER JOIN and when should you use one?"
    )


def test_a_well_formed_question_with_two_interrogatives_is_not_repaired():
    raw = questions.RawEntry(1)

    # Two openers, but this is a sentence with a question mark rather
    # than a column merge. The gate is the sentence punctuation, not the
    # accident that the source check happened to fail.
    raw.set(
        "question",
        "What is an INNER JOIN and when should you use one?",
    )

    verdict, _reason, question, _note = questions.judge(raw, MERGED_SOURCE)

    assert verdict is CandidateVerdict.ACCEPT
    assert question == "What is an INNER JOIN and when should you use one?"


def test_a_trailing_link_is_rewritten_only_when_the_remainder_is_in_the_source():
    package_source = (
        "Explain the MERGE INTO statement in this clause. "
        "https://example.invalid/video"
    )

    raw = questions.RawEntry(1)

    raw.set(
        "question",
        "Explain the MERGE INTO statement in this clause. "
        "https://example.invalid/video",
    )

    verdict, _reason, question, note = questions.judge(raw, package_source)

    assert verdict is CandidateVerdict.REWRITE
    assert question == (
        "Explain the MERGE INTO statement in this clause."
    )
    assert "trailing link" in note


def test_a_rewrite_that_the_source_does_not_support_is_not_made():
    raw = questions.RawEntry(1)

    # Entirely absent from the source text below.
    raw.set(
        "question",
        "How does a window function differ from a correlated subquery? "
        "https://example.invalid/x",
    )

    verdict, _reason, question, _note = questions.judge(
        raw, "SELECT id, RANK() OVER (PARTITION BY x) FROM t"
    )

    # Not rewritten, because doing so would put words on a slide that
    # does not contain them.
    assert verdict is not CandidateVerdict.REWRITE
    assert question is None or question == raw.get("question")


def test_a_question_whose_slide_does_not_resolve_needs_review():
    raw = questions.RawEntry(1)

    raw.set("question", "What is a window function?")

    verdict, reason, question, _ = questions.judge(raw, "")

    assert verdict is CandidateVerdict.NEEDS_REVIEW
    assert "did not resolve to an image" in reason
    assert question is None


def test_damaged_looking_question_text_is_accepted_because_it_cannot_be_told():
    """
    A documented limitation, pinned so it is not rediscovered as a bug.

    ``Explain the ~_ » gt. ™ 9 sos ©) distributed`` looks like debris and
    is accepted, because character analysis rates it clean -- the same
    finding as
    :func:`test_misread_english_is_not_detectable_as_garbled`. Rejecting
    it would mean a threshold nobody can justify, and every threshold
    that catches this also catches a real bullet-fragment slide.

    The gate does not rest on the quality verdict. It rests on the
    source-support requirement: a rewrite must appear verbatim in the
    slide's own transcription, which no fragment invented by a
    threshold can satisfy.
    """

    raw = questions.RawEntry(1)

    raw.set("question", "Explain the ~_ » gt. ™ 9 sos ©) distributed")

    verdict, _reason, question, _note = questions.judge(
        raw, "Explain the distributed system architecture"
    )

    assert verdict is CandidateVerdict.ACCEPT
    assert question is not None


def test_an_empty_candidate_is_rejected():
    raw = questions.RawEntry(1)

    raw.set("question", "")

    verdict, reason, question, _ = questions.judge(raw, "anything")

    assert verdict is CandidateVerdict.REJECT
    assert "no question text" in reason
    assert question is None


def test_the_packages_answers_are_never_promoted_to_answers(tmp_path):
    package = write_package(
        tmp_path / "package",
        question_bank=bank(
            entry(
                "What is an INNER JOIN?",
                excerpt=SQL_OCR,
            )
        ),
    )

    contents = read_package(package)

    records = attach_transcriptions(
        parse_inventory(contents), parse_knowledge(contents)[0]
    )

    built = build_candidates(contents, records)

    # The "answer" field is the slide transcription repeated. It is kept
    # as an excerpt and nothing else.
    assert built[0].source_excerpt == SQL_OCR
    assert built[0].question != SQL_OCR


def test_slide_numbering_is_resolved_against_the_inventory(tmp_path):
    package = write_package(
        tmp_path / "package",
        inventory_rows=[
            "POST-001,7263033731471351808,0,"
            "activity_7263033731471351808_slide_0.jpg,800,420,146149,"
            + "a" * 64
            + ",,,AVAILABLE"
        ],
        question_bank=bank(
            entry(
                "What is an INNER JOIN?",
                excerpt=GOOD_OCR,
                # One-based, while the inventory counts from zero.
                source="POST-001, slide 1",
            )
        ),
    )

    contents = read_package(package)

    records = attach_transcriptions(
        parse_inventory(contents), parse_knowledge(contents)[0]
    )

    built = build_candidates(contents, records)

    assert built[0].slide_convention == "one_based"
    assert built[0].filename == "activity_7263033731471351808_slide_0.jpg"


def test_a_question_citing_an_unknown_group_does_not_resolve(tmp_path):
    package = write_package(
        tmp_path / "package",
        question_bank=bank(
            entry("What is an INNER JOIN?", source="POST-999, slide 0")
        ),
    )

    contents = read_package(package)

    records = attach_transcriptions(
        parse_inventory(contents), parse_knowledge(contents)[0]
    )

    built = build_candidates(contents, records)

    assert built[0].source_resolved is False
    assert built[0].verdict is CandidateVerdict.NEEDS_REVIEW


def test_the_banks_constant_type_and_difficulty_are_kept_but_not_trusted(
    tmp_path,
):
    package = write_package(
        tmp_path / "package",
        question_bank=bank(
            entry("What is an INNER JOIN?", declared_type="Theory/Scenario",
                  declared_difficulty="Easy")
        ),
    )

    contents = read_package(package)

    records = attach_transcriptions(
        parse_inventory(contents), parse_knowledge(contents)[0]
    )

    built = build_candidates(contents, records)

    # Recorded because it is what the package said, and recorded because
    # inspection showed it is constant, which is why nothing downstream
    # treats it as a signal.
    assert built[0].declared_type == "Theory/Scenario"
    assert built[0].declared_difficulty == "Easy"


# ---------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------


TECHNICAL = """# CONSOLIDATED TECHNICAL KNOWLEDGE

## Topic Coverage

- **SQL** — 128 activity groups
- **Apache Spark** — 64 activity groups

## Post-by-Post Extraction

### POST-001

**Technologies/topics:** SQL

- Slide 0: USE THE S.T.A.R TECHNIQUE!

### POST-002

**Technologies/topics:** Azure Databricks, Databricks

- Slide 3: a quotation no transcription contains
"""


def test_technology_claims_and_their_quoted_support_are_both_kept():
    knowledge = parse_technical({"LINKEDIN_TECHNICAL_KNOWLEDGE.md": TECHNICAL})

    assert knowledge["POST-001"].technologies == ["SQL"]
    assert knowledge["POST-002"].technologies == [
        "Azure Databricks",
        "Databricks",
    ]


def test_similar_names_are_not_collapsed_into_one_concept():
    knowledge = parse_technical({"LINKEDIN_TECHNICAL_KNOWLEDGE.md": TECHNICAL})

    # Deciding these name one technology is the existing taxonomy's
    # judgement. A merge made here would be silent.
    assert len(knowledge["POST-002"].technologies) == 2


def test_a_repeated_technology_is_recorded_once():
    body = TECHNICAL.replace(
        "**Technologies/topics:** SQL",
        "**Technologies/topics:** SQL, sql, SQL",
    )

    knowledge = parse_technical({"LINKEDIN_TECHNICAL_KNOWLEDGE.md": body})

    assert knowledge["POST-001"].technologies == ["SQL"]


def test_a_fragment_is_attached_by_matching_words_not_the_slide_number():
    records = [
        GeminiImageRecord(
            post_id="POST-001",
            activity_id="7263033731471351808",
            filename="slide.jpg",
            slide_number=3,
            raw_ocr_text="USE THE S.T.A.R TECHNIQUE!",
        )
    ]

    knowledge = parse_technical({"LINKEDIN_TECHNICAL_KNOWLEDGE.md": TECHNICAL})

    resolved, notes = resolve_fragments(knowledge, records)

    keys = list(resolved["POST-001"].slide_fragments)

    # The package numbered it "Slide 0" and the inventory says 3. The
    # words decide, and the declared number is kept only as a note.
    assert keys == ["slide.jpg#declared_slide=0"]


def test_a_quotation_no_transcription_contains_is_reported_not_attached():
    records = [
        GeminiImageRecord(
            post_id="POST-001",
            activity_id="7263033731471351808",
            filename="slide.jpg",
            slide_number=0,
            raw_ocr_text="something else entirely",
        )
    ]

    knowledge = parse_technical({"LINKEDIN_TECHNICAL_KNOWLEDGE.md": TECHNICAL})

    resolved, notes = resolve_fragments(knowledge, records)

    assert resolved["POST-002"].slide_fragments == {}
    assert any("does not appear" in note for note in notes)


def test_a_quotation_in_several_transcriptions_is_left_unattached():
    shared = "MERGE INTO target USING source"

    records = [
        GeminiImageRecord(
            post_id="POST-002",
            activity_id="7287036149515112448",
            filename="one.jpg",
            slide_number=0,
            raw_ocr_text=shared,
        ),
        GeminiImageRecord(
            post_id="POST-002",
            activity_id="7287036149515112448",
            filename="two.jpg",
            slide_number=1,
            raw_ocr_text=shared,
        ),
    ]

    knowledge = parse_technical(
        {
            "LINKEDIN_TECHNICAL_KNOWLEDGE.md": TECHNICAL.replace(
                "- Slide 3: a quotation no transcription contains",
                f"- Slide 1: {shared}",
            )
        }
    )

    resolved, notes = resolve_fragments(knowledge, records)

    assert resolved["POST-002"].slide_fragments == {}
    assert any("cannot be attributed" in note for note in notes)


def test_the_topic_index_is_read_as_topic_to_groups():
    index = parse_topic_index(
        {
            "LINKEDIN_TOPIC_INDEX.md": (
                "# TOPIC INDEX\n\n"
                "## SQL\n\nPOST-004, POST-005, POST-008\n\n"
                "## Python\n\nPOST-005, POST-006\n"
            )
        }
    )

    assert index.topics["SQL"] == ["POST-004", "POST-005", "POST-008"]
    assert index.topics["Python"] == ["POST-005", "POST-006"]


def test_a_topic_referencing_an_unknown_group_is_reported():
    index = parse_topic_index(
        {"LINKEDIN_TOPIC_INDEX.md": "## SQL\n\nPOST-001, POST-317\n"}
    )

    knowledge = parse_technical({"LINKEDIN_TECHNICAL_KNOWLEDGE.md": TECHNICAL})

    # POST-001 is defined by the technical file; POST-317 is not. Something
    # was extracted from something this project cannot see. Dropping it
    # silently would make a topic look better covered than it is.
    assert unmatched_groups(index, knowledge) == ["POST-317"]


# ---------------------------------------------------------------------
# Source and derived separation
# ---------------------------------------------------------------------


def test_a_transcription_never_becomes_the_posts_own_text(tmp_path):
    records = attach_transcriptions(
        parse_inventory(read_package(write_package(tmp_path / "package"))),
        parse_knowledge(
            read_package(write_package(tmp_path / "package"))
        )[0],
    )

    post = make_post(
        "urn-li-saved-abc",
        ["media/activity_7263033731471351808_slide_0.jpg"],
    )

    index = bridge.GeminiIndex(records, [], "2026-01-01T00:00:00+00:00")

    bridge.inject(post, index.outcome_for(post), index.records_for(post))

    # The author's words are untouched. The transcription is additional
    # evidence about the post, never a substitute for it.
    assert post.original_text == "A post about joins."
    assert "USE THE S.T.A.R TECHNIQUE!" not in post.original_text


def test_the_transcription_lands_on_the_media_with_its_provenance(tmp_path):
    package = write_package(tmp_path / "package")

    records = attach_transcriptions(
        parse_inventory(read_package(package)),
        parse_knowledge(read_package(package))[0],
    )

    post = make_post(
        "urn-li-saved-abc",
        ["media/activity_7263033731471351808_slide_0.jpg"],
    )

    index = bridge.GeminiIndex(records, [], "2026-01-01T00:00:00+00:00")

    bridge.inject(post, index.outcome_for(post), index.records_for(post))

    item = post.media[0]

    assert item.extracted_text == GOOD_OCR
    assert item.extraction_method == bridge.PROCESSOR
    assert item.ocr_status == OcrStatus.AVAILABLE.value
    assert item.ocr_quality == OcrQuality.READABLE.value
    assert item.source_kind == SourceKind.IMAGE_OCR.value
    assert item.slide_count == 1
    assert item.sequence == 0


def test_the_ocr_quality_is_not_disguised_as_a_reading_confidence(tmp_path):
    package = write_package(tmp_path / "package")

    records = attach_transcriptions(
        parse_inventory(read_package(package)),
        parse_knowledge(read_package(package))[0],
    )

    outcome = bridge.outcome_for("p", records, "2026-01-01T00:00:00+00:00")

    for analysis in outcome.visual.analyses:
        # Confidence means "how sure the processor is that it read this".
        # A transcription's legibility is a different question with a
        # different answer, and the named quality states carry it.
        assert analysis.confidence == 0.0


def test_the_transcription_processor_is_distinct_from_the_vision_one():
    from src.visual.models import (
        PROCESSOR_CONFIGURATION as VISION_CONFIG,
        PROCESSOR_VERSION as VISION_VERSION,
    )

    # Different names and different versions, so the two can never
    # collide in a cache keyed by asset digest plus processor.
    assert bridge.PROCESSOR != "vision"
    assert bridge.CONFIGURATION != VISION_CONFIG

    assert (bridge.PROCESSOR_VERSION, bridge.CONFIGURATION) != (
        VISION_VERSION,
        VISION_CONFIG,
    )


def test_a_post_the_package_says_nothing_about_is_left_alone():
    index = bridge.GeminiIndex([], [], "")

    post = make_post("urn-li-saved-abc", ["media/activity_1_slide_0.jpg"])

    assert index.outcome_for(post) is None
    assert index.records_for(post) == {}
    assert index.digest_for(post) == ""


# ---------------------------------------------------------------------
# Gaps: what CP12 is left with
# ---------------------------------------------------------------------


def test_a_slide_with_no_text_is_a_gap_cp12_could_still_read():
    record = GeminiImageRecord(
        post_id="POST-001",
        filename="a.jpg",
        ocr_status=OcrStatus.NO_TEXT,
    )

    gap = importer.gaps_for(record)

    assert gap is not None
    assert gap.reason is GapReason.NO_TEXT
    assert gap.actionable is True


def test_an_unresolved_slide_is_a_gap():
    record = GeminiImageRecord(
        post_id="POST-001",
        filename="a.jpg",
        ocr_status=OcrStatus.UNRESOLVED,
    )

    assert importer.gaps_for(record).reason is GapReason.UNRESOLVED


def test_debris_is_a_gap_and_a_fragment_is_not():
    # The status has to say the text exists, or the record is ABSENT and
    # the gap is reported for a different reason.
    assert (
        importer.gaps_for(
            GeminiImageRecord(
                post_id="P",
                filename="a.jpg",
                ocr_status=OcrStatus.AVAILABLE,
                ocr_quality=OcrQuality.GARBLED,
            )
        ).reason
        is GapReason.DEBRIS
    )

    assert (
        importer.gaps_for(
            GeminiImageRecord(
                post_id="P",
                filename="a.jpg",
                ocr_status=OcrStatus.AVAILABLE,
                ocr_quality=OcrQuality.FRAGMENTARY,
            )
        )
        is None
    )


def test_a_readable_slide_is_not_a_gap():
    record = GeminiImageRecord(
        post_id="POST-001",
        filename="a.jpg",
        ocr_status=OcrStatus.AVAILABLE,
        ocr_quality=OcrQuality.READABLE,
        raw_ocr_text=SQL_OCR,
    )

    # Importing this package exists to avoid thousands of vision calls.
    assert importer.gaps_for(record) is None


def test_a_file_the_archive_lacks_is_reported_but_is_not_actionable():
    record = GeminiImageRecord(
        post_id="POST-001",
        filename="a.jpg",
        ocr_status=OcrStatus.AVAILABLE,
        ocr_quality=OcrQuality.READABLE,
        raw_ocr_text="Some text.",
        verdict=MatchVerdict.MISSING_FROM_ARCHIVE,
    )

    gap = importer.gaps_for(record)

    # There is no file on disk for a vision call to read, so calling this
    # a gap would be asking for work that cannot be done.
    assert gap.actionable is False
    assert gap.reason is GapReason.MISSING_FROM_ARCHIVE


def test_a_preview_is_only_a_preview_when_its_target_exists():
    records = [
        GeminiImageRecord(
            post_id="P",
            filename="activity_1_slide_0.jpg",
            likely_preview_of="activity_1_slide_01.jpg",
            slide_number=0,
            width=480,
            height=360,
        ),
        GeminiImageRecord(
            post_id="P",
            filename="activity_1_slide_01.jpg",
            slide_number=1,
            width=1200,
            height=1000,
        ),
    ]

    known = {record.filename for record in records}

    roles = {
        record.filename: bridge._role_for(record, known) for record in records
    }

    assert roles["activity_1_slide_0.jpg"].value == "thumbnail"
    assert roles["activity_1_slide_01.jpg"].value == "slide"


def test_an_unverifiable_preview_claim_does_not_discard_the_only_image():
    record = GeminiImageRecord(
        post_id="P",
        filename="activity_1_slide_0.jpg",
        likely_preview_of="absent.jpg",
        slide_number=0,
    )

    role = bridge._role_for(record, {"activity_1_slide_0.jpg"})

    # Marking it redundant on the strength of a claim about a file nobody
    # can check would throw away a post's only visual evidence.
    assert role.value != "thumbnail"


# ---------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------


def test_importing_the_same_package_twice_writes_identical_bytes(tmp_path):
    package = write_package(tmp_path / "package")

    archive = write_archive(
        tmp_path / "archive",
        {"activity_7263033731471351808_slide_0.jpg": png_bytes()},
        {"7263033731471351808": {"files": [
            "activity_7263033731471351808_slide_0.jpg"
        ]}},
    )

    output = tmp_path / "imported"

    first = importer.run(package, archive, output)
    body = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in sorted(output.rglob("*.json"))
    }

    second = importer.run(package, archive, output)

    after = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in sorted(output.rglob("*.json"))
    }

    assert first.wrote is True

    # The skip is decided by digest, so the second run does not re-read
    # or re-write anything.
    assert second.wrote is False
    assert body == after


def test_a_changed_package_is_a_real_change_and_is_reimported(tmp_path):
    package = write_package(tmp_path / "package")

    archive = write_archive(
        tmp_path / "archive",
        {"activity_7263033731471351808_slide_0.jpg": png_bytes()},
        {"7263033731471351808": {"files": [
            "activity_7263033731471351808_slide_0.jpg"
        ]}},
    )

    output = tmp_path / "imported"

    importer.run(package, archive, output)

    (package / "LINKEDIN_TOPIC_INDEX.md").write_text(
        "## SQL\n\nPOST-001\n", encoding="utf-8"
    )

    assert importer.run(package, archive, output).wrote is True


def test_reimporting_creates_no_duplicate_records(tmp_path):
    package = write_package(tmp_path / "package")

    archive = write_archive(
        tmp_path / "archive",
        {"activity_7263033731471351808_slide_0.jpg": png_bytes()},
        {"7263033731471351808": {"files": [
            "activity_7263033731471351808_slide_0.jpg"
        ]}},
    )

    output = tmp_path / "imported"

    importer.run(package, archive, output)
    importer.run(package, archive, output, force=True)

    loaded = importer.load_records(output)

    assert len(loaded) == 1
    assert loaded[0].filename == "activity_7263033731471351808_slide_0.jpg"


def test_a_stale_group_file_is_removed_on_reimport(tmp_path):
    package = write_package(tmp_path / "package")

    archive = write_archive(
        tmp_path / "archive",
        {"activity_7263033731471351808_slide_0.jpg": png_bytes()},
        {"7263033731471351808": {"files": [
            "activity_7263033731471351808_slide_0.jpg"
        ]}},
    )

    output = tmp_path / "imported"

    images = output / "images"
    images.mkdir(parents=True, exist_ok=True)
    (images / "POST-999.json").write_text('{"images": []}', encoding="utf-8")

    importer.run(package, archive, output)

    # A group that vanished from the package must not survive from an
    # earlier import, or the knowledge base would describe an activity
    # that is no longer in it.
    assert not (images / "POST-999.json").exists()


def test_an_unreadable_manifest_causes_a_reimport_rather_than_a_silent_skip(
    tmp_path,
):
    output = tmp_path / "imported"
    output.mkdir(parents=True, exist_ok=True)

    (output / "manifest.json").write_text("{not json", encoding="utf-8")

    assert importer.read_manifest(output) is None


# ---------------------------------------------------------------------
# No local paths in public output
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        r"C:\Users\eagleanurag\Downloads\linkedin_saved_archive",
        "C:\\Users\\someone\\file.jpg",
        r"\\server\share\file.jpg",
        "see Downloads\\archive for it",
        "run .venv\\Scripts\\python.exe",
        "see .agent/notes.md",
    ],
)
def test_local_path_shapes_are_found(text):
    assert find_local_paths(text)


@pytest.mark.parametrize(
    "text",
    [
        "Use a forward slash: media/activity_1_slide_0.jpg",
        "Set the value to 10.",
        "Documents and Desktop are ordinary words in a transcript.",
        "SELECT * FROM t WHERE x = 1",
    ],
)
def test_ordinary_text_is_not_mistaken_for_a_local_path(text):
    assert find_local_paths(text) == []


def test_a_payload_with_a_local_path_is_refused():
    with pytest.raises(PublicOutputError):
        assert_public(
            {"images": [{"path": "media/a.jpg", "note": r"C:\Users\x\y"}]},
            where="test",
        )


def test_a_mapping_key_carrying_a_path_is_also_refused():
    # A payload whose values are clean can still be useless if its keys
    # are not.
    with pytest.raises(PublicOutputError):
        assert_public({r"C:\Users\x\a.jpg": {"sha256": "abc"}}, where="test")


def test_a_clean_payload_passes():
    assert_public(
        {
            "images": [
                {
                    "path": "media/activity_1_slide_0.jpg",
                    "raw_ocr_text": "INNER JOIN B ON A.key = B.key",
                }
            ]
        },
        where="test",
    )


def test_a_record_never_stores_a_local_path(tmp_path):
    package = write_package(tmp_path / "package")

    records = parse_inventory(read_package(package))

    for record in records:
        # The logical path, so a published record never names a Windows
        # user account.
        assert record.path == f"media/{record.filename}"
        assert ":" not in record.path


# ---------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------


def ocr_model(*posts: KnowledgePost):
    """
    A site model carrying only what the OCR index reads.

    ``SiteModel`` is frozen, so it is constructed rather than filled in.
    Every field the OCR index does not touch is empty, which is the
    point: this tests the index, not the model.
    """

    from src.wiki.analysis import SiteModel

    return SiteModel(
        posts=tuple(posts),
        post_slugs=tuple(post.id for post in posts),
        topics=(),
        topic_slugs={},
        questions=(),
        concept_entries=(),
        technology_entries=(),
        concepts=(),
    )


def test_a_term_that_appears_only_inside_an_image_is_findable(tmp_path):
    post = make_post("urn-li-saved-abc", ["media/activity_1_slide_0.jpg"])

    post.media[0].extracted_text = SQL_OCR
    post.media[0].extraction_method = bridge.PROCESSOR
    post.media[0].sequence = 3
    post.media[0].ocr_status = OcrStatus.AVAILABLE.value
    post.media[0].ocr_quality = OcrQuality.READABLE.value

    records = build_ocr_records(ocr_model(post))

    assert len(records) == 1

    record = records[0]

    assert record["i"] == "urn-li-saved-abc"
    assert record["s"] == 4
    assert record["f"] == "activity_1_slide_0.jpg"
    assert "INNER JOIN B ON A.key = B.key" in record["x"]
    assert record["m"] == bridge.PROCESSOR


def test_search_text_is_not_truncated_so_a_late_term_is_still_found():
    body = ("filler " * 400) + "INNER JOIN"

    post = make_post("p", ["media/a.jpg"])

    post.media[0].extracted_text = body
    post.media[0].extraction_method = bridge.PROCESSOR

    records = build_ocr_records(ocr_model(post))

    # Truncated text cannot be found, and a search that silently misses
    # a term on a slide is worse than a larger index file.
    assert "INNER JOIN" in records[0]["x"]


def test_media_without_text_is_not_indexed():
    post = make_post("p", ["media/a.jpg"])

    assert build_ocr_records(ocr_model(post)) == []


def test_media_with_text_of_unknown_origin_is_not_indexed():
    post = make_post("p", ["media/a.jpg"])

    post.media[0].extracted_text = SQL_OCR

    # Indexing it would attribute the text to nobody.
    assert build_ocr_records(ocr_model(post)) == []


def test_the_ocr_index_records_that_it_is_machine_transcribed():
    post = make_post("p", ["media/a.jpg"])

    post.media[0].extracted_text = SQL_OCR
    post.media[0].extraction_method = bridge.PROCESSOR

    payload = build_ocr_index(ocr_model(post))

    assert payload["slides"] == 1
    assert "Machine transcription" in payload["caveat"]


def test_an_excerpt_is_cut_for_display_but_the_index_keeps_everything():
    post = make_post("p", ["media/a.jpg"])

    post.media[0].extracted_text = "x" * 5000
    post.media[0].extraction_method = bridge.PROCESSOR

    assert len(excerpt("x" * 5000)) < 5000
    assert len(build_ocr_records(ocr_model(post))[0]["x"]) == 5000


def test_the_index_carries_a_filename_and_never_a_directory(tmp_path):
    post = make_post("p", ["media/activity_1_slide_0.jpg"])

    post.media[0].extracted_text = SQL_OCR
    post.media[0].extraction_method = bridge.PROCESSOR

    record = build_ocr_records(ocr_model(post))[0]

    assert "/" not in record["f"]
    assert find_local_paths(record["f"]) == []


# ---------------------------------------------------------------------
# Derived technologies in the consolidation
# ---------------------------------------------------------------------


def test_a_technology_read_off_a_slide_is_attached_to_that_slide():
    from src.gemini.models import GeminiPostKnowledge

    records = [
        GeminiImageRecord(
            post_id="POST-001",
            filename="slide.jpg",
            slide_number=2,
            raw_ocr_text=SQL_OCR,
        )
    ]

    entry = GeminiPostKnowledge(group_id="POST-001", technologies=["SQL"])

    claims = derived.claims_for_post(entry, records)

    assert len(claims) == 1
    assert claims[0].name == "SQL"
    assert claims[0].filename == "slide.jpg"
    assert claims[0].slide_number == 2
    assert claims[0].source_kind == "image_ocr"
    assert claims[0].evidence


def test_a_claim_no_transcription_supports_is_still_recorded_but_marked():
    from src.gemini.models import GeminiPostKnowledge

    records = [
        GeminiImageRecord(
            post_id="POST-001",
            filename="slide.jpg",
            slide_number=0,
            raw_ocr_text="something unrelated",
        )
    ]

    entry = GeminiPostKnowledge(group_id="POST-001", technologies=["SQL"])

    claims = derived.claims_for_post(entry, records)

    # Dropping it would make this project's coverage quietly smaller than
    # the evidence available. Recording it unattached keeps the
    # difference between a claim and a citation visible.
    assert len(claims) == 1
    assert claims[0].filename == ""
    assert claims[0].evidence == derived.NO_SLIDE_EVIDENCE


def test_a_short_technology_does_not_match_inside_a_longer_word():
    # "Git" appearing inside "GitHub" is not evidence that the slide was
    # about Git.
    assert derived.mentions("We use GitHub Actions", "Git") is False
    assert derived.mentions("We use git for version control", "Git") is True


def test_a_derived_claim_joins_the_same_node_as_a_text_detected_one():
    post = make_post("p", ["media/a.jpg"])

    post.ai_analysis.topics = ["Databases"]
    post.ai_analysis.derived_technologies = ["SQL"]

    post.original_text = "Today we look at SQL window functions."

    index = consolidate(
        [post],
        derived_technologies={
            "p": [
                DerivedClaim(
                    name="SQL",
                    source_kind="image_ocr",
                    filename="a.jpg",
                    slide_number=0,
                )
            ]
        },
    )

    node = next(n for n in index.technologies if n.name == "SQL")

    # One place to look, and the weaker evidence still attached to it.
    assert node.derived
    assert node.derived[0].filename == "a.jpg"


def test_a_technology_read_off_a_slide_reaches_both_artifacts():
    """
    The knowledge base and the site must not disagree.

    The site builder reads the same post field the consolidation reads,
    so a slide-only technology is either in both or in neither. Reading
    the imported package directly in one place and not the other is how a
    knowledge base starts claiming coverage its own site does not show.
    """

    post = make_post("p", ["media/activity_1_slide_0.jpg"])

    post.original_text = "A slide-heavy post about joins."
    post.ai_analysis.derived_technologies = ["Apache Spark"]

    index = consolidate([post])

    assert any(
        node.name == "Apache Spark" for node in index.technologies
    )

    from src.wiki.analysis import _build_technologies

    entries = _build_technologies((post,), ("p",))

    assert any(entry.label == "Apache Spark" for entry in entries)


def test_a_technology_no_transcription_mentions_is_not_claimed():
    index = bridge.GeminiIndex(
        [
            GeminiImageRecord(
                post_id="POST-001",
                activity_id="7263033731471351808",
                filename="slide.jpg",
                slide_number=0,
                raw_ocr_text="something unrelated entirely",
            )
        ],
        [],
        "",
        {"POST-001": ["Apache Spark"]},
    )

    post = make_post(
        "p", ["media/activity_7263033731471351808_slide.jpg"]
    )

    # The package attributes it to the post; no image in the post
    # mentions it. Reproducing it would put a term into the knowledge
    # base that no source supports.
    assert index.technologies_for(post) == []


def test_a_technology_a_transcription_does_mention_is_claimed():
    index = bridge.GeminiIndex(
        [
            GeminiImageRecord(
                post_id="POST-001",
                activity_id="7263033731471351808",
                filename="slide.jpg",
                slide_number=0,
                raw_ocr_text=SQL_OCR,
            )
        ],
        [],
        "",
        {"POST-001": ["SQL", "Apache Spark"]},
    )

    post = make_post(
        "p", ["media/activity_7263033731471351808_slide_0.jpg"]
    )

    assert index.technologies_for(post) == ["SQL"]


def test_the_derived_field_is_omitted_when_empty():
    # The 490 posts committed before any of this existed must serialise
    # exactly as they did.
    payload = make_post("p", []).model_dump(mode="json")

    assert "derived_technologies" not in payload["ai_analysis"]


def test_the_derived_field_is_present_when_used():
    post = make_post("p", [])

    post.ai_analysis.derived_technologies = ["SQL"]

    payload = post.model_dump(mode="json")

    assert payload["ai_analysis"]["derived_technologies"] == ["SQL"]


def test_a_technology_node_deduplicates_an_identical_claim():
    node = TechnologyNode(name="SQL", slug="sql")

    claim = DerivedClaim(
        name="SQL", filename="a.jpg", slide_number=0
    )

    node.add_derived(claim)
    node.add_derived(claim)

    # The same slide offered once per post that carries it would otherwise
    # put forty copies of one reading on the page.
    assert len(node.derived) == 1


def test_no_import_means_no_derived_claims(tmp_path, monkeypatch):
    import src.gemini.derived as module

    monkeypatch.setattr(module, "DEFAULT_OUTPUT", tmp_path / "absent")

    assert module.derived_claims_for_posts([make_post("p", ["media/a.jpg"])]) == {}


def test_the_claim_counts_separate_supported_from_unsupported(
    tmp_path, monkeypatch
):
    import src.gemini.derived as module
    from src.gemini.models import GeminiPostKnowledge

    knowledge = {
        "POST-001": GeminiPostKnowledge(
            group_id="POST-001", technologies=["SQL"]
        ),
        "POST-002": GeminiPostKnowledge(
            group_id="POST-002", technologies=["Apache Spark"]
        ),
    }

    records = [
        GeminiImageRecord(
            post_id="POST-001",
            filename="slide.jpg",
            slide_number=0,
            raw_ocr_text=SQL_OCR,
        )
    ]

    monkeypatch.setattr(module, "DEFAULT_OUTPUT", tmp_path)
    monkeypatch.setattr(module, "load_knowledge", lambda base=None: knowledge)
    monkeypatch.setattr(module, "load_records", lambda base=None: records)

    counts = module.claim_counts()

    # Reporting the two together would make the second kind look like the
    # first.
    assert counts["claims_with_a_slide"] == 1
    assert counts["claims_without_one"] == 1
    assert counts["posts"] == 2


# ---------------------------------------------------------------------
# Freshness
# ---------------------------------------------------------------------


def write_result(tmp_path, **fingerprint) -> Path:
    """A worker result carrying only the fingerprint."""

    target = tmp_path / "cloud_worker_p.json"

    target.write_text(
        json.dumps({"_enrichment": fingerprint}), encoding="utf-8"
    )

    return target


def test_a_corrected_transcription_invalidates_the_post_it_informs(tmp_path):
    from src.pipeline.freshness import (
        ENRICHER_VERSION,
        OCR_PROCESSOR_VERSION,
        _reusable,
    )

    target = write_result(
        tmp_path,
        source_digest="src",
        enricher_version=ENRICHER_VERSION,
        ocr_digest="old",
        ocr_processor_version=OCR_PROCESSOR_VERSION,
    )

    assert _reusable(target, "src", "", "new") is None


def test_an_unchanged_transcription_reuses_the_result(tmp_path):
    from src.pipeline.freshness import (
        ENRICHER_VERSION,
        OCR_PROCESSOR_VERSION,
        _reusable,
    )

    target = write_result(
        tmp_path,
        source_digest="src",
        enricher_version=ENRICHER_VERSION,
        ocr_digest="same",
        ocr_processor_version=OCR_PROCESSOR_VERSION,
    )

    assert _reusable(target, "src", "", "same") is not None


def test_a_changed_transcription_does_not_invalidate_the_visual_digest(
    tmp_path,
):
    from src.pipeline.freshness import (
        ENRICHER_VERSION,
        VISUAL_PROCESSOR_VERSION,
        _reusable,
    )

    target = write_result(
        tmp_path,
        source_digest="src",
        enricher_version=ENRICHER_VERSION,
        visual_digest="vis",
        visual_processor_version=VISUAL_PROCESSOR_VERSION,
    )

    # The two sources are compared independently. A change to the
    # imported transcription is not a change to what the vision
    # processor read, and treating it as one would re-enrich posts for
    # no reason.
    assert _reusable(target, "src", "vis", "") is not None


def test_a_refresh_keeps_the_reading_derived_fields(tmp_path):
    from src.pipeline.freshness import _refresh_provenance

    stored = {
        "media": [
            {
                "type": "image",
                "path": "media/a.jpg",
                "extracted_text": "USE THE S.T.A.R TECHNIQUE!",
                "sequence": 0,
                "sha256": "abc",
                "extraction_method": "gemini_ocr",
                "ocr_status": "AVAILABLE",
                "ocr_quality": "readable",
                "ocr_note": "measured",
                "source_kind": "image_ocr",
                "slide_count": 3,
            }
        ],
        "original_text": "old",
    }

    current = make_post("p", ["media/a.jpg"])

    refreshed = _refresh_provenance(stored, current)

    item = refreshed["media"][0]

    # Republishing a machine's garbled reading as though it were plain
    # fact is exactly what these fields exist to prevent.
    assert item["ocr_status"] == "AVAILABLE"
    assert item["ocr_quality"] == "readable"
    assert item["ocr_note"] == "measured"
    assert item["extracted_text"] == "USE THE S.T.A.R TECHNIQUE!"


# ---------------------------------------------------------------------
# Questions: deduplication and provenance
# ---------------------------------------------------------------------


def test_a_question_the_post_already_has_is_not_added_twice():
    from src.pipeline.orchestrate import _add_ocr_questions

    post = make_post("p", ["media/activity_1_slide_0.jpg"])

    from src.models import InterviewQuestion

    post.interview_questions.append(
        InterviewQuestion(
            question="What is an inner join?",
            type="theory",
            difficulty="easy",
        )
    )

    _add_ocr_questions(
        post,
        [
            GeminiCandidateQuestion(
                index=1,
                candidate_text="What is an INNER JOIN?",
                verdict=CandidateVerdict.ACCEPT,
                question="What is an INNER JOIN?",
                filename="activity_1_slide_0.jpg",
                slide_number=0,
            )
        ],
    )

    # Case and punctuation are not what makes a question the same
    # question; the existing dedup already knows that.
    assert len(post.interview_questions) == 1


def test_an_accepted_question_records_where_it_came_from():
    from src.gemini.models import GeminiCandidateQuestion
    from src.pipeline.orchestrate import _add_ocr_questions

    post = make_post("p", ["media/activity_1_slide_0.jpg"])

    _add_ocr_questions(
        post,
        [
            GeminiCandidateQuestion(
                index=1,
                candidate_text="What is an INNER JOIN?",
                verdict=CandidateVerdict.ACCEPT,
                question="What is an INNER JOIN?",
                source_excerpt=SQL_OCR,
                filename="activity_1_slide_0.jpg",
                slide_number=3,
            )
        ],
    )

    question = post.interview_questions[0]

    assert question.source_kind == SourceKind.IMAGE_OCR.value
    assert question.answer_source == "source_excerpt"

    # The slide is named, so the claim can be checked.
    assert "activity_1_slide_0.jpg" in question.source_note
    assert "3" in question.source_note

    # The answer is an excerpt of the transcription, introduced as one.
    # OCR is not an answer and must not be presented as if it were.
    assert "Source excerpt" in question.answer
    assert "INNER JOIN B ON A.key = B.key" in question.answer


def test_a_question_with_no_source_text_gets_no_invented_answer():
    from src.pipeline.orchestrate import _add_ocr_questions

    post = make_post("p", ["media/activity_1_slide_0.jpg"])

    _add_ocr_questions(
        post,
        [
            GeminiCandidateQuestion(
                index=1,
                candidate_text="What is an INNER JOIN?",
                verdict=CandidateVerdict.ACCEPT,
                question="What is an INNER JOIN?",
                source_excerpt="",
            )
        ],
    )

    assert "too damaged" in post.interview_questions[0].answer


def test_a_rejected_candidate_never_reaches_a_post():
    from src.pipeline.orchestrate import _add_ocr_questions

    post = make_post("p", ["media/activity_1_slide_0.jpg"])

    _add_ocr_questions(
        post,
        [
            GeminiCandidateQuestion(
                index=1,
                candidate_text="cn oh? Azure",
                verdict=CandidateVerdict.REJECT,
                question=None,
            )
        ],
    )

    assert post.interview_questions == []


def test_only_accepted_candidates_are_carried_into_the_index():
    index = bridge.GeminiIndex(
        [],
        [
            GeminiCandidateQuestion(
                index=1,
                candidate_text="What is an INNER JOIN?",
                verdict=CandidateVerdict.ACCEPT,
                question="What is an INNER JOIN?",
            ),
            GeminiCandidateQuestion(
                index=2,
                candidate_text="Link 1: https://example.invalid",
                verdict=CandidateVerdict.REJECT,
            ),
            GeminiCandidateQuestion(
                index=3,
                candidate_text="Describe the Explain the TASK",
                verdict=CandidateVerdict.NEEDS_REVIEW,
            ),
        ],
    )

    assert len(index.accepted) == 1


# ---------------------------------------------------------------------
# Wiki rendering
# ---------------------------------------------------------------------


def render(post: KnowledgePost) -> str:
    """The post's page, rendered."""

    from src.wiki.pages import render_post_detail

    return render_post_detail(ocr_model(post), post, post.id)


def transcribed_post(
    identifier: str = "urn-li-saved-abc",
    *,
    extracted_text: str | None = "The grain table stores one row per booking.",
    ocr_status: str = OcrStatus.AVAILABLE.value,
    ocr_quality: str = OcrQuality.READABLE.value,
    ocr_note: str = "128 chars, 21 words, 74% letters, 6% symbols",
    slide_count: int = 1,
    sequence: int = 0,
    role: str = "slide",
) -> KnowledgePost:
    """
    A post carrying one machine transcription.

    Rebuilt rather than kept: the helper sat between two of the tests
    that were rewritten against the data instead of the page, and
    removing the page left nothing between them to define it.

    The fields it sets are the ones the reader-facing page stopped
    showing and the record still has to carry -- provenance, the
    judgement, the evidence and the position in the carousel.
    """

    post = make_post(identifier, ["media/activity_1_slide_0.jpg"])

    media = post.media[0]
    media.extracted_text = extracted_text
    media.extraction_method = bridge.PROCESSOR
    media.source_kind = "image_ocr"
    media.ocr_status = ocr_status
    media.ocr_quality = ocr_quality
    media.ocr_note = ocr_note
    media.slide_count = slide_count
    media.sequence = sequence
    media.role = role

    return post


def test_a_transcription_is_marked_as_a_machine_reading_in_the_data():
    """
    The property moves rather than disappears.

    The knowledge page used to print "transcribed from this image by a
    machine" beside every transcription. It no longer prints
    transcriptions at all, which is a stronger guarantee for a reader:
    nothing a machine produced can be mistaken for something the author
    wrote, because none of it is on the page.

    What still has to hold is that the *record* says so, because that is
    what a reader, an auditor or a future renderer relies on.
    """

    media = transcribed_post().media[0]

    assert media.extraction_method == "gemini_ocr"
    assert media.source_kind == "image_ocr"
    assert media.ocr_status == OcrStatus.AVAILABLE.value


def test_the_ocr_status_quality_and_evidence_are_kept_on_the_record():
    post = transcribed_post(
        ocr_quality=OcrQuality.GARBLED.value,
        ocr_status=OcrStatus.AVAILABLE.value,
        ocr_note=(
            "168 chars, 33 words, 61% letters, 17% symbols -- mostly "
            "symbols rather than words"
        ),
    )

    media = post.media[0]

    # The judgement and the measurement behind it are both retained. A
    # garbled transcription with no note is indistinguishable from a
    # clean one once it is out of the importer, and the note is what a
    # later renderer would need to say so honestly.
    assert media.ocr_quality == OcrQuality.GARBLED.value
    assert "mostly symbols rather than words" in media.ocr_note
    assert "%" in media.ocr_note


def test_no_text_and_absent_remain_different_claims():
    """
    "Nobody looked" and "there is nothing there" are different facts.

    Both used to be spelled out in prose on the page. The page no longer
    lists slides, so the distinction has to live in the status alone --
    and it does, which is what this now checks.
    """

    empty = transcribed_post(
        extracted_text=None,
        ocr_status=OcrStatus.NO_TEXT.value,
        ocr_quality=OcrQuality.EMPTY.value,
    ).media[0]

    absent = transcribed_post(
        extracted_text=None,
        ocr_status=OcrStatus.ABSENT.value,
        ocr_quality=OcrQuality.EMPTY.value,
    ).media[0]

    assert empty.ocr_status == OcrStatus.NO_TEXT.value
    assert absent.ocr_status == OcrStatus.ABSENT.value
    assert empty.ocr_status != absent.ocr_status


def test_the_slide_position_is_kept():
    post = transcribed_post(slide_count=4, sequence=2)

    media = post.media[0]

    assert media.sequence == 2
    assert media.slide_count == 4


def test_a_preview_is_still_marked_as_one():
    assert transcribed_post(role="thumbnail").media[0].role == "thumbnail"


def test_no_rendered_page_presents_a_transcription_as_knowledge():
    """
    The strongest form of "label it as a machine reading".

    The old page listed every slide with its transcription, its OCR
    status, its slide number and whether it was a preview. All of that is
    gone from the reader's view, so a transcription cannot be quoted as
    something a person wrote -- which is the failure the labelling was
    there to prevent.
    """

    page = render(transcribed_post())

    assert transcribed_post().media[0].extracted_text not in page

    for marker in (
        "transcribed from this image",
        "ocr_status",
        "Model-assisted content",
        "Original source material",
    ):
        assert marker not in page


def test_a_hostile_transcription_cannot_reach_a_page_as_markup():
    """
    Escaping, checked at the boundary that still exists.

    Transcriptions are untrusted third-party text. They used to be
    escaped at the point of insertion into the media list on the post
    page. That page no longer renders them, so there is no insertion
    point left to escape at -- which is the strongest form of the
    guarantee: the payload has nowhere to land.

    The text is still held as data, unchanged, because rewriting it
    would misrepresent what the package said. What must not happen is
    markup reaching a document.
    """

    hostile = (
        "MERGE <script>alert('x')</script> "
        "<img src=x onerror=alert(1)>"
    )

    page = render(transcribed_post(extracted_text=hostile))

    assert "<script>" not in page
    assert "<img src=x" not in page
    assert "alert('x')" not in page

    # Held verbatim as data: the importer does not alter what the
    # package wrote, because a sanitised copy under this name would be a
    # false claim about the source.
    stored = transcribed_post(extracted_text=hostile).media[0].extracted_text

    assert "<script>" in stored

def test_the_page_never_repeats_the_packages_constant_prose():
    page = render(transcribed_post())

    # Stored once per image would put a sentence about method into the
    # knowledge base attributed to an image it says nothing about.
    assert "Non-text structure is not invented" not in page
    assert "external knowledge was not used to fill omissions" not in page


def source_question_post() -> KnowledgePost:
    """
    A post carrying one question read off a slide.

    Rebuilt for the same reason as :func:`transcribed_post`: the helper
    sat immediately before a test that was rewritten, and the rewrite
    removed it along with the page assertions it served.

    The distinction it exists to protect -- a passage quoted from a slide
    is not an answer written to the question -- is unchanged, and the
    slide the passage came from is still recorded on the record.
    """

    post = make_post("urn-li-saved-abc", ["media/activity_1_slide_0.jpg"])

    post.media[0].extracted_text = SQL_OCR
    post.media[0].extraction_method = bridge.PROCESSOR
    post.media[0].source_kind = "image_ocr"
    post.media[0].ocr_status = OcrStatus.AVAILABLE.value
    post.media[0].ocr_quality = OcrQuality.READABLE.value

    post.interview_questions = [
        InterviewQuestion(
            question="What is an INNER JOIN?",
            type="theory",
            difficulty="easy",
            answer=SQL_OCR,
            answer_source="source_excerpt",
            source_note="activity_1_slide_0.jpg",
        )
    ]

    return post


def test_a_source_derived_question_is_labelled_and_located():
    """
    The label survives; the on-page location does not.

    A question read off a slide is not an answer to itself, and the
    reader is told so wherever it appears. That is unchanged. What the
    old test also checked -- that the page names the slide file the
    question came from -- was part of the media inventory, and an
    ``activity_7263033731471351808_slide_0.jpg`` on a study page is an
    internal identifier a reader cannot use.

    The provenance it was checking for is still on the record.
    """

    page = render(source_question_post())

    assert "What is an INNER JOIN?" in page

    # Labelled as a passage rather than written as an answer.
    assert "source excerpt" in page or "Quoted from the slide" in page

    question = source_question_post().interview_questions[0]

    assert question.answer_source == "source_excerpt"

    # And the slide it was read from is still recorded.
    assert question.source_note == "activity_1_slide_0.jpg"

def test_the_page_never_contains_a_local_path(tmp_path):
    post = transcribed_post()

    page = render(post)

    assert find_local_paths(page) == []


# ---------------------------------------------------------------------
# Injection adds to the post; it does not invent it
# ---------------------------------------------------------------------
#
# Found by running the importer over the real archive rather than by
# reading the code. The archive holds 65 files no post references -- a
# carousel a post lists one slide of, for instance -- and the package
# transcribes every image the archive holds. Injecting by asset list
# therefore attached 2,735 media items that were never the post's.
#
# The second effect was quieter and worse. The largest carousel reached a
# 46,030 character prompt, and Windows refuses a command line over 32,767
# characters by failing process creation, which Python reports as
# FileNotFoundError. Enrichment of those posts failed with an error naming
# the OpenCode executable -- the executable was present and worked, and
# the message pointed an operator at the wrong system entirely.


def _records(count: int) -> list:
    """Transcriptions for one activity, ``count`` slides of it."""

    return [
        GeminiImageRecord(
            post_id="P",
            filename=f"activity_1_slide_{n}.jpg",
            declared_sha256=f"{n:064x}",
            ocr_status=OcrStatus.AVAILABLE,
            raw_ocr_text=f"Slide {n} text.",
            public_text=f"Slide {n} text.",
        )
        for n in range(count)
    ]


def test_injection_never_adds_media_the_post_does_not_have():
    """The package describes the archive; the post describes the post."""

    records = _records(6)
    index = bridge.GeminiIndex(records, [], "2026-01-01T00:00:00+00:00")

    # The post has one slide. The package read six.
    post = make_post("urn-li-saved-abc", ["media/activity_1_slide_0.jpg"])

    bridge.inject(
        post,
        index.outcome_for(post),
        index.records_for(post),
    )

    assert len(post.media) == 1


def test_the_slide_the_post_does_have_still_gets_its_transcription():
    """Not skipped as collateral: the one slide that IS the post's."""

    records = _records(6)
    index = bridge.GeminiIndex(records, [], "2026-01-01T00:00:00+00:00")

    post = make_post("urn-li-saved-abc", ["media/activity_1_slide_0.jpg"])

    bridge.inject(
        post, index.outcome_for(post), index.records_for(post)
    )

    assert "Slide 0 text." in post.media[0].extracted_text


def test_media_the_archive_holds_but_the_post_does_not_is_reported():
    """
    Skipped, not dropped.

    The import still records every transcription, so nothing is lost --
    but the count that had no post to attach to is a fact about the
    archive and the posts disagreeing, and it should be visible rather
    than silently absorbed.
    """

    records = _records(6)
    index = bridge.GeminiIndex(records, [], "2026-01-01T00:00:00+00:00")

    post = make_post("urn-li-saved-abc", ["media/activity_1_slide_0.jpg"])

    outcome = index.outcome_for(post)

    bridge.inject(post, outcome, index.records_for(post))

    assert outcome.unmatched == 5


def test_the_visual_stage_still_adds_media_it_actually_read():
    """
    The other half of the contract.

    ``add_missing=False`` is for the package only. The archive-driven
    visual stage read those files, so they are the post's content and
    must be attached -- otherwise the fallback path would silently stop
    describing images at all.
    """

    from src.visual.stage import inject as inject_media

    records = _records(3)
    index = bridge.GeminiIndex(records, [], "2026-01-01T00:00:00+00:00")

    post = make_post("urn-li-saved-abc", ["media/activity_1_slide_0.jpg"])

    outcome = index.outcome_for(post)

    inject_media(post, outcome)

    assert len(post.media) == 3
    assert outcome.unmatched == 0


def test_an_over_long_prompt_is_reported_as_such_rather_than_as_a_missing_binary():
    """
    The guard, exercised.

    Without it the post fails inside ``subprocess.run`` with a
    ``FileNotFoundError`` that :class:`OpenCodeClient` re-raises as
    "the executable could not be started" -- pointing at an executable
    that is installed and working.
    """

    from src.ai.enricher import AIEnricher, PromptTooLongError

    post = make_post("urn-li-saved-abc", ["media/activity_1_slide_0.jpg"])

    post.original_text = "x" * (WINDOWS_COMMAND_LINE_LIMIT + 100)

    with pytest.raises(PromptTooLongError):
        AIEnricher().enrich(post)


def test_transcription_from_an_image_the_post_lacks_never_reaches_the_post():
    """
    The property a reader depends on, stated as one assertion.

    Not "the count is right" -- that a number matches an expectation --
    but that no slide's words can appear on a post that never attached
    that slide. The first version of this injected every transcription
    the package held, which put 62 slides of text onto a post with one
    image; nothing about the count would have caught that, because the
    count was never checked against the post.
    """

    records = _records(6)
    index = bridge.GeminiIndex(records, [], "2026-01-01T00:00:00+00:00")

    post = make_post("urn-li-saved-abc", ["media/activity_1_slide_0.jpg"])

    bridge.inject(
        post, index.outcome_for(post), index.records_for(post)
    )

    everything = " ".join(
        item.extracted_text or "" for item in post.media
    )

    assert "Slide 0 text." in everything

    for absent in range(1, 6):
        assert f"Slide {absent} text." not in everything


def test_the_count_of_unattached_transcriptions_is_reported_by_the_orchestrator():
    """
    Surfaced rather than absorbed.

    2,735 of this project's 3,047 transcriptions describe images no post
    attaches. They are not lost -- they are in the import -- but a run
    that says nothing about them invites the reader to assume all 3,047
    reached a post.
    """

    import inspect

    from src.pipeline.orchestrate import Settled, _enrich_one

    assert "ocr_unmatched" in Settled.__dataclass_fields__

    signature = inspect.signature(_enrich_one)

    # The count is part of what the orchestrator is told, not something
    # it has to go looking for afterwards.
    assert len(signature.return_annotation.split(",")) == 3

def test_a_prompt_this_large_cannot_be_passed_as_a_command_line_argument():
    """
    The ceiling this bug hit, named so the reason is findable.

    A prompt past this cannot be passed as an argument on Windows, and
    the resulting ``FileNotFoundError`` is indistinguishable from a
    missing executable. The largest prompt in this checkout is 7,894
    characters; anything approaching this ceiling means one post is
    carrying text that is not its own.
    """

    assert WINDOWS_COMMAND_LINE_LIMIT == 32767


# ---------------------------------------------------------------------
# Local path redaction
# ---------------------------------------------------------------------
#
# Written against the real corpus, because the OCR damage is not the
# shape anyone would guess. Three lines of the same W3Schools tutorial
# slide, read by the same engine on the same page, are a local path in
# three different ways:
#
#     c:\Users\Your Name>python          colon, single backslash
#     c:\\Users\Your Name>python         colon, backslash doubled
#     c\\Users\Your Name>python          **no colon at all**
#
# A filter anchored on ``X:`` catches the first two and misses the third
# entirely, which then has its doubled backslash pair eaten as a UNC
# path and leaves a stray ``c`` in the published text.


@pytest.mark.parametrize(
    "raw,expected",
    [
        # The prompt case, with the colon intact. Split so no piece of the
        # literal ends in a backslash, which a raw string cannot.
        (
            "C:\\Users\\Your Name\\AppData\\Local\\Programs\\Python\\"
            "Python36-32\\Scripts>pip install requests",
            "<LOCAL_WINDOWS_PATH>>pip install requests",
        ),
        # Backslash doubled by OCR.
        (
            r"c:\\Users\\Your Name>python --version",
            "<LOCAL_WINDOWS_PATH>>python --version",
        ),
        # Colon dropped entirely.
        (
            r"c\\Users\Your Name>python",
            "<LOCAL_WINDOWS_PATH>>python",
        ),
        (
            r"C\Users\Your Name>python myfile.py",
            "<LOCAL_WINDOWS_PATH>>python myfile.py",
        ),
        # A real UNC share.
        (r"\\buildserver\c$\builds\out.txt", "<UNC_PATH>"),
        # An environment variable whose value is a local path.
        (r"%LOCALAPPDATA%\Programs\Python\python.exe", "<LOCAL_PATH_ENV_VAR>"),
        # This project's own directories.
        (r".venv\Scripts\python.exe -m pytest", "<LOCAL_PROJECT_PATH>"),
    ],
)
def test_a_local_path_is_replaced_and_the_command_survives(raw, expected):
    public, count = redact(raw)

    assert public == expected
    assert count >= 1


@pytest.mark.parametrize(
    "text",
    [
        # A Windows-looking word must not be enough. These are all real
        # lines from this corpus that an earlier, looser filter broke.
        "if is=n//2 or j==n//2:",
        "m=m//10;q=q//10",
        "tasks involving I/O operations (e.g., network requests).",
        "*2024-04-24', '%d/%m/%Y') \"2024-04-24\", '%d/%m/%Y'",
        "x = a\\b",
        # Not local paths at all, and required to survive verbatim.
        "spark-submit --master local[*] --class MyJob jar",
        "abfss://container@storage.dfs.core.windows.net/raw/sales",
        "https://www.python.org/downloads/",
        "s3://my-bucket/data/2024/",
        "mongodb://localhost:27017/",
        "SELECT COUNT(*) FROM orders WHERE dt = '2024-01-01';",
        "df = spark.read.parquet('s3a://bucket/path')",
        "self.assertEqual(x, y)",
    ],
)
def test_legitimate_technical_content_is_not_touched(text):
    public, count = redact(text)

    assert public == text
    assert count == 0


def test_a_redacted_transcription_is_still_searchable_for_its_command():
    raw = r"C:\Users\Your Name\Scripts>pip install requests"

    public, _ = redact(raw)

    # The point of replacing only the path: the technical content of the
    # slide survives, so a reader searching for the command finds it.
    assert "pip install requests" in public


def test_redaction_reports_how_many_paths_it_replaced():
    text = (
        r"copy C:\Users\Alice\a.txt D:\Users\Bob\b.txt "
        r"and \\server\share\c.txt"
    )

    _public, count = redact(text)

    assert count == 3


def test_the_summary_does_not_call_sanitised_text_verbatim():
    note = redact_note(2)

    assert "2 local filesystem paths" in note
    assert "replaced" in note
    assert "verbatim" not in note
    assert redact_note(0) == ""


def test_a_record_keeps_the_verbatim_text_separate_from_the_public_text(tmp_path):
    """
    Provenance, not just filtering.

    The verbatim transcription stays available in memory and is never
    written; the public copy carries a marker and a count. Calling the
    sanitised string "verbatim" would be a false claim about the source.
    """

    package = write_package(
        tmp_path / "package",
        knowledge=DEFAULT_KNOWLEDGE.replace(
            "USE THE S.T.A.R TECHNIQUE!",
            r"C:\Users\Your Name>python --version",
        ),
    )

    contents = read_package(package)

    records = attach_transcriptions(
        parse_inventory(contents), parse_knowledge(contents)[0]
    )

    record = records[0]

    # Verbatim: still exactly what the package wrote.
    assert r"C:\Users\Your Name" in record.raw_ocr_text

    # Public: no local path, and a marker instead.
    assert r"C:\Users" not in record.public_text
    assert "<LOCAL_WINDOWS_PATH>" in record.public_text
    assert record.redactions >= 1

    # Search is built from the public text, so a search can never surface
    # a string the reader cannot see.
    assert "<LOCAL_WINDOWS_PATH>" in record.search_text


def test_the_written_record_does_not_contain_the_verbatim_field():
    payload = public_record(
        GeminiImageRecord(
            post_id="P",
            filename="a.jpg",
            raw_ocr_text=r"C:\Users\Alice\secret.txt",
            public_text="<LOCAL_WINDOWS_PATH>",
        )
    )

    assert "raw_ocr_text" not in payload
    assert payload["public_text"] == "<LOCAL_WINDOWS_PATH>"


def test_the_published_text_path_never_uses_the_verbatim_one():
    """
    The record field is not the only route to publication.

    ``extracted_text`` reaches ``post.json``, the knowledge base and the
    site, so the bridge has to publish the sanitised copy. Asserted on the
    bridge's own output rather than trusted.
    """

    record = GeminiImageRecord(
        post_id="P",
        filename="activity_1_slide_0.jpg",
        declared_sha256="a" * 64,
        raw_ocr_text=r"C:\Users\Alice\Scripts>pip install requests",
        public_text="<LOCAL_WINDOWS_PATH>>pip install requests",
    )

    outcome = bridge.outcome_for("p", [record], "2026-01-01T00:00:00+00:00")

    analysis = outcome.visual.analyses[0]

    assert r"C:\Users" not in (analysis.extracted_text or "")
    assert "pip install requests" in (analysis.extracted_text or "")

    items = bridge.to_media_items([record])

    assert r"C:\Users" not in (items[0].extracted_text or "")


def test_original_text_never_receives_a_transcription(tmp_path):
    package = write_package(
        tmp_path / "package",
        knowledge=DEFAULT_KNOWLEDGE.replace(
            "USE THE S.T.A.R TECHNIQUE!",
            r"C:\Users\Your Name>python --version",
        ),
    )

    contents = read_package(package)

    records = attach_transcriptions(
        parse_inventory(contents), parse_knowledge(contents)[0]
    )

    post = make_post(
        "urn-li-saved-abc",
        ["media/activity_7263033731471351808_slide_0.jpg"],
    )

    index = bridge.GeminiIndex(records, [], "2026-01-01T00:00:00+00:00")

    bridge.inject(post, index.outcome_for(post), index.records_for(post))

    # Unchanged, sanitised or not. The author's words are the author's.
    assert post.original_text == "A post about joins."

    assert "<LOCAL_WINDOWS_PATH>" not in post.original_text


def test_the_public_output_guard_is_still_armed():
    """
    The guard must still refuse, not merely be bypassed.

    Redaction removes the paths that are found. The guard is what catches
    the ones that are not, so weakening it would remove the only thing
    standing between the next unfamiliar OCR shape and a published
    Windows user directory.
    """

    assert find_local_paths(r"C:\Users\Someone\file.txt")

    with pytest.raises(PublicOutputError):
        assert_public(
            {"note": r"C:\Users\Someone\file.txt"}, where="test"
        )


# ---------------------------------------------------------------------
# Import structure
# ---------------------------------------------------------------------
#
# Run in a fresh interpreter on purpose. An in-process import proves
# nothing here: by the time a test runs, every module is already
# initialised, so a cycle that only bites on a cold import -- which is
# exactly what `python -m src.pipeline` is -- would pass unnoticed.
#
# The original defect was one of these. `src/gemini/bridge.py` imported
# `src.pipeline.visual` for one small helper, and importing any submodule
# of `src.pipeline` executes that package's `__init__.py`, which imports
# the orchestrator, which imports the bridge back. Whichever of the two a
# user reached first, the other was found half-built.

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_python(statement: str) -> subprocess.CompletedProcess:
    """One statement in a fresh interpreter, from the repository root."""

    return subprocess.run(
        [sys.executable, "-c", statement],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.parametrize(
    "statement",
    [
        "import src.gemini.bridge",
        "import src.pipeline",
        "import src.gemini; import src.pipeline",
        "import src.pipeline; import src.gemini.bridge",
        "import src.gemini.derived",
        "import src.visual.stage",
        "import src.pipeline.visual",
    ],
    ids=[
        "bridge-alone",
        "pipeline-alone",
        "gemini-then-pipeline",
        "pipeline-then-gemini",
        "derived-alone",
        "visual-stage-alone",
        "pipeline-visual-alone",
    ],
)
def test_importing_the_source_and_pipeline_layers_does_not_cycle(statement):
    """
    Both layers import, in either order, in a cold interpreter.

    Order matters and both are checked. ``src.pipeline`` may depend on
    ``src.gemini`` -- it does, deliberately, to use the imported
    transcriptions -- but not the reverse. That direction is what the
    orchestrator needs in order to consult a source layer, and asserting
    both orders catches an inversion reintroduced from either side.
    """

    result = run_python(statement)

    assert result.returncode == 0, (
        f"{statement!r} failed:\n{result.stdout}\n{result.stderr}"
    )


def test_the_gemini_layer_does_not_import_the_pipeline_layer():
    """
    The structural half of the fix, checked directly.

    A passing import is not sufficient evidence on its own: a future edit
    could reintroduce ``from src.pipeline.visual import ...`` into the
    bridge *lazily*, inside a function, where no import-order test would
    ever exercise it. It would work in every test run and fail again the
    first time that function was reached during a cold import. So the
    boundary itself is asserted: no module under ``src/gemini`` may name
    ``src.pipeline`` anywhere in its source.
    """

    offenders: list[str] = []

    for path in sorted((REPO_ROOT / "src" / "gemini").glob("*.py")):
        source = path.read_text(encoding="utf-8")

        for number, line in enumerate(source.splitlines(), start=1):
            stripped = line.strip()

            if stripped.startswith("#"):
                continue

            if "src.pipeline" in stripped and (
                stripped.startswith("import ")
                or stripped.startswith("from ")
                or stripped.startswith("import(")
            ):
                offenders.append(f"{path.name}:{number}: {stripped}")

    assert offenders == [], (
        "the imported-package layer must not depend on the pipeline "
        "layer; the pipeline imports it, not the reverse:\n  "
        + "\n  ".join(offenders)
    )


def test_a_media_path_still_maps_to_its_activity_after_the_move():
    """
    The relocated helper must behave exactly as the original did.

    The re-export in ``src.pipeline.visual`` and the definition in
    ``src.visual.assets`` have to agree, because the whole visual stage
    keys on the value this returns: ``VisualRunner.posts_with_media``
    tests membership against the archive's own ``post_id``, and this
    archive keys its media-bearing posts as ``activity_<id>``. Measured
    against ``posts_archive.json``: 485 records, of which the ones with
    media are ``activity_7509801763013595136`` and so on -- prefix
    included, which is why the prefix is preserved here.
    """

    from src.pipeline.visual import archive_post_id
    from src.visual.assets import activity_for

    for path in (
        "media/activity_7263033731471351808_slide_0.jpg",
        "media/activity_7263033731471351808_slide_01.jpg",
        "media/urn_7263033731471351808_original.pdf",
        "media/something_else.jpg",
    ):
        assert archive_post_id(path) == activity_for(path)

    # The prefix is the archive's own key and must survive the move.
    assert activity_for("media/activity_7263033731471351808_slide_7.jpg") == (
        "activity_7263033731471351808"
    )


def test_the_relocated_inject_is_the_same_function_the_pipeline_reexports():
    from src.pipeline.visual import inject as reexported
    from src.visual.stage import inject as canonical

    assert reexported is canonical