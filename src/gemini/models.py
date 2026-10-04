"""
What the Gemini package says, and how far it can be trusted.

The package is derived data produced by a machine reading pictures. It
is not the post, and it is not a human's transcription. Everything in
this package exists to keep four things apart that the package itself
runs together:

* what the LinkedIn author wrote,
* what a machine read off a picture,
* what a model concluded from that reading,
* what this project's enricher derived afterwards.

They are different kinds of claim with different authority, and a reader
who cannot tell them apart will believe a machine misread of a
low-resolution screenshot as much as they would believe the author.

Two specific traps in this package shaped the model.

**Its "Visual Description" and "Source-Derived Explanation" sections are
constant.** Every post carries the identical two sentences: *"Text was
machine-transcribed. Non-text structure is not invented where it was not
explicitly reviewed."* and *"Limited to the captured source material;
external knowledge was not used to fill omissions."* They are a
statement of policy, not a description of any picture. Storing them as
``visual_summary`` would put a sentence about method into the knowledge
base once per slide, attributed to a slide it says nothing about. So
they are recognised and dropped rather than stored, and
:attr:`GeminiImageRecord.visual_description` is only ever populated from
something that differs between records.

**Its "Answer" field is the OCR echoed back.** In the question bank the
answer is the slide transcription repeated verbatim, not an answer. It
is therefore kept as ``source_excerpt`` and never promoted to an answer.

Nothing here decides whether a transcription is *good*. That judgement
belongs to :mod:`src.gemini.ocr_quality`, which works from observable
signals in the text, and a transcription's quality is recorded beside it
rather than used to silently drop it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class SourceKind(str, Enum):
    """
    What kind of claim a piece of text is.

    The single most important distinction in this project. A reader who
    knows a line came from a slide reading rather than from the author
    can weigh it; a reader who does not will quote it as something the
    author said.
    """

    #: Written by the LinkedIn author. The only authoritative source.
    AUTHOR_SOURCE = "author_source"

    #: Machine transcription of a picture's text. Verbatim, unverified,
    #: and frequently damaged on small images.
    IMAGE_OCR = "image_ocr"

    #: A model's interpretation of a picture, distinct from its words.
    VISUAL_DERIVED = "visual_derived"

    #: This project's own enrichment, derived from all of the above.
    AI_ENRICHMENT = "ai_enrichment"


class OcrStatus(str, Enum):
    """
    What the package says about an image's text.

    ``ABSENT`` is this project's addition and means the package said
    nothing at all about an image that exists in the archive. It is
    distinct from ``NO_TEXT`` -- a package that claims an image has no
    text and a package that never looked at it are different facts, and
    only the first is a finding about the image.
    """

    AVAILABLE = "AVAILABLE"
    NO_TEXT = "NO_TEXT"
    PARTIAL = "PARTIAL"
    UNRESOLVED = "UNRESOLVED"
    FAILED = "FAILED"
    PENDING = "PENDING"
    ABSENT = "ABSENT"


class MatchVerdict(str, Enum):
    """
    How a package record stands against the real archive.

    The archive is authoritative for what exists; the package is
    authoritative only for what it read out of a picture. Every record
    gets a verdict, and no record is dropped for having a bad one.
    """

    #: The file exists, and its recorded bytes match what is on disk.
    MATCHED = "matched"

    #: The package names a file the archive does not have.
    MISSING_FROM_ARCHIVE = "missing_from_archive"

    #: The archive holds several files this record could be.
    AMBIGUOUS = "ambiguous"

    #: The package and the archive disagree about something objective.
    CONFLICTING = "conflicting"


class CandidateVerdict(str, Enum):
    """
    What to do with a question the package proposed.

    The package is explicit that its questions come from "question-like
    or instruction-like text captured from the uploaded slides", and
    inspection bears that out: one entry is a bare YouTube URL and
    another is the symbol noise ``cn oh? Azure``. Neither is an
    interview question, and publishing them would put noise in front of
    a reader looking for something to study.
    """

    #: Readable, genuinely a question, and traceable to a slide.
    ACCEPT = "accept"

    #: The intent is recoverable from the source but OCR damaged the
    #: wording. Rewritable only from what the source supports.
    REWRITE = "rewrite"

    #: Not a question, or too damaged to recover safely.
    REJECT = "reject"

    #: Possibly useful, but the intent cannot be reconstructed without
    #: a person. Kept and surfaced rather than guessed at.
    NEEDS_REVIEW = "needs_review"


class OcrQuality(str, Enum):
    """
    What is observably true of a transcription.

    Every member is computed from the text itself or from the image's
    size. No member is a score invented to order images, because a
    number nobody can trace is how a weak transcription displaces a
    strong one and nobody can say why.
    """

    #: Substantial, connected prose.
    READABLE = "readable"

    #: Real words, but sparse or fragmentary.
    FRAGMENTARY = "fragmentary"

    #: Dominated by symbols, stray characters and OCR debris.
    GARBLED = "garbled"

    #: Nothing to judge.
    EMPTY = "empty"


class GeminiPackageInfo(BaseModel):
    """Which package, and what it claimed about itself."""

    package_name: str = "LINKEDIN_KNOWLEDGE_ARCHIVE_PACKAGE"
    source_path: str = ""

    #: The package's own headline figures, kept verbatim so a report can
    #: compare them against what was actually found without the original
    #: being lost.
    claimed_total_images: int | None = None
    claimed_groups: int | None = None
    claimed_ocr_available: int | None = None
    claimed_ocr_no_text: int | None = None
    claimed_ocr_failed: int | None = None
    claimed_ocr_pending: int | None = None
    claimed_unique_after_duplicates: int | None = None
    claimed_exact_duplicates: int | None = None

    #: The package's own words about its limits, kept because this
    #: project must not overstate them.
    declared_caveat: str = ""

    imported_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class GeminiImageRecord(BaseModel):
    """
    One image as the package describes it, cross-checked against disk.

    ``raw_ocr_text`` is whatever the package wrote, character for
    character. It is never corrected, trimmed, or tidied: a corrected
    transcription is no longer evidence of what was on the slide, and
    the damaged original is what tells a reader the image was hard to
    read. :attr:`search_text` exists alongside it for indexing and is
    derived, never authoritative.
    """

    #: The package's own grouping identifier, e.g. ``POST-004``.
    post_id: str

    #: The LinkedIn activity identifier, e.g. ``7263033731471351808``.
    activity_id: str = ""

    #: The media filename, e.g.
    #: ``activity_7263033731471351808_slide_01.jpg``.
    filename: str

    #: Logical path used everywhere the image is referred to. Never a
    #: local filesystem path: these records are committed and published.
    path: str = ""

    #: Zero-based, from the package's own ordering.
    slide_number: int = 0

    slide_number_source: Literal["package", "filename", "unresolved"] = (
        "package"
    )

    width: int | None = None
    height: int | None = None
    size_bytes: int = 0

    #: The package's digest of the image, kept so a mismatch against the
    #: archive is detectable rather than assumed away.
    declared_sha256: str = ""

    #: The archive's digest of the file. Empty when the file was absent.
    actual_sha256: str = ""

    #: The package's exact-duplicate claim, as a filename.
    exact_duplicate_of: str = ""

    #: The package's preview claim, as a filename. Recorded as claimed,
    #: never acted on: see :mod:`src.gemini.crosscheck`.
    likely_preview_of: str = ""

    ocr_status: OcrStatus = OcrStatus.ABSENT

    #: Verbatim transcription. Never altered.
    #:
    #: Held in memory only. It is **not** written to any tracked or
    #: published file, because a transcription of a Windows command
    #: prompt contains a local filesystem path and that file is committed
    #: and deployed. :attr:`public_text` is what gets written; this is
    #: what it was derived from.
    #:
    #: The two are kept separate so that "verbatim" stays a true word.
    #: A sanitised copy presented under this name would be a claim that
    #: nothing was changed, and something was.
    raw_ocr_text: str | None = None

    #: The transcription as it may be published: local filesystem paths
    #: replaced by an explicit marker, everything else character for
    #: character. This is the field that reaches tracked output, the
    #: knowledge base and the site.
    public_text: str | None = None

    #: How many local paths :attr:`public_text` had replaced. Non-zero
    #: means the published text is not identical to the source, and a
    #: reader is entitled to know that without having to diff it.
    redactions: int = 0

    #: Derived, for search. Whitespace collapsed and nothing else, taken
    #: from :attr:`public_text` rather than from the verbatim text, so a
    #: search can never surface a string that was removed from what the
    #: reader can actually see.
    search_text: str = ""

    #: Always ``IMAGE_OCR`` when present. Named so that no downstream
    #: reader has to infer the origin.
    source_kind: SourceKind = SourceKind.IMAGE_OCR

    #: How the text was obtained.
    extraction_method: Literal["gemini_ocr"] = "gemini_ocr"

    #: Observable properties of the transcription.
    ocr_quality: OcrQuality = OcrQuality.EMPTY
    ocr_note: str = ""

    verdict: MatchVerdict = MatchVerdict.MATCHED
    verdict_note: str = ""

    #: Deterministically derived from the fields above, so re-importing
    #: an unchanged package produces byte-identical output.
    fingerprint: str = ""

    @property
    def readable_text(self) -> str:
        """
        The transcription, preferring the verbatim copy.

        Present because the two are not always both available. During an
        import both exist and the verbatim one is the better source for
        verification. After a reload from disk only :attr:`public_text`
        survives, because the verbatim copy is deliberately not written.

        Reading :attr:`raw_ocr_text` directly is therefore a bug waiting
        to happen: it is ``None`` for every record loaded from a previous
        import, and a consumer that forgets sees no text at all rather
        than an error. That is not hypothetical -- it is how the
        technology report came to claim that none of 578 claims had a
        naming slide when 263 of them do.
        """

        if self.raw_ocr_text is not None:
            return self.raw_ocr_text

        return self.public_text or ""


class GeminiPostGroup(BaseModel):
    """
    One reconstructed post, as the package grouped the activity.

    ``group_id`` is the package's ``POST-nnn``. It is kept as the
    package's own namespace and never substituted for the project's own
    post identifier, because the two are not the same thing: this
    project hashes the archive identifier to keep it off the published
    site.
    """

    group_id: str
    activity_id: str = ""
    declared_slide_count: int = 0
    actual_slide_count: int = 0
    subject_clue: str = ""
    grouping_confidence: str = ""
    ordering_confidence: str = ""
    filenames: list[str] = Field(default_factory=list)
    verdict: MatchVerdict = MatchVerdict.MATCHED
    verdict_note: str = ""


class GeminiCandidateQuestion(BaseModel):
    """
    A question the package proposed, with the judgement on it.

    ``candidate_text`` is preserved whatever happens, so a rejected
    entry can be audited rather than merely disappeared. ``question`` is
    only populated when the verdict is ACCEPT or REWRITE, and
    ``rewrite_note`` says what was changed when it is a rewrite.
    """

    index: int

    #: Exactly as the package wrote it.
    candidate_text: str

    #: The package's own fields. Both were constant across inspection
    #: and are kept only to record that fact.
    declared_type: str = ""
    declared_difficulty: str = ""

    #: The package's "answer", which is the slide transcription repeated.
    #: Never promoted to an answer.
    source_excerpt: str = ""

    group_id: str = ""
    activity_id: str = ""
    slide_number: int | None = None

    #: Which numbering the package's own "slide N" turned out to mean.
    #: The knowledge archive prints ``## Slide 1`` for a file whose own
    #: name says ``slide_0``, so the two disagree by one on the same
    #: slide. Resolved against the inventory rather than assumed, and
    #: recorded because the difference decides which picture a question
    #: is attributed to.
    slide_convention: Literal[
        "zero_based", "one_based", "unresolved"
    ] = "unresolved"

    filename: str = ""

    #: True when the question's slide resolved to a specific image and
    #: the source could therefore be checked.
    source_resolved: bool = False

    verdict: CandidateVerdict = CandidateVerdict.NEEDS_REVIEW
    verdict_reason: str = ""

    #: Only for ACCEPT and REWRITE.
    question: str | None = None
    rewrite_note: str = ""

    source_kind: SourceKind = SourceKind.IMAGE_OCR
    extraction_method: Literal["gemini_ocr"] = "gemini_ocr"
    fingerprint: str = ""


class GeminiPostKnowledge(BaseModel):
    """
    The package's per-post technology claims, cross-referenced.

    ``slide_fragments`` are the package's own short quotations from each
    slide. They are kept because a technology claim is only worth as
    much as the text it rests on, and a reader who cannot see that text
    has been asked to take the claim on trust.
    """

    group_id: str
    activity_id: str = ""
    technologies: list[str] = Field(default_factory=list)
    slide_fragments: dict[str, str] = Field(default_factory=dict)
    #: Group ids the package referenced that have no image record.
    unresolved_group_ids: list[str] = Field(default_factory=list)
    source_kind: SourceKind = SourceKind.VISUAL_DERIVED


class GeminiTopicIndex(BaseModel):
    """
    The package's topic index, as topic to group ids.

    Kept as the package stated it. Two topics whose names differ only in
    case are the same topic; two that differ otherwise are not merged
    here, because deciding that "Azure Databricks" and "Databricks" are
    the same technology is the project's existing taxonomy's judgement
    to make and not this package's.
    """

    topics: dict[str, list[str]] = Field(default_factory=dict)


class CrossCheckReport(BaseModel):
    """
    What was compared, and what did not agree.

    Every disagreement is listed. None is resolved silently, because the
    whole value of a cross-check is that the mismatches survive into the
    report where a person can see them.
    """

    records_total: int = 0
    matched: int = 0
    missing_from_archive: int = 0
    ambiguous: int = 0
    conflicting: int = 0

    #: Archive media the package never mentioned.
    archive_files_absent_from_package: int = 0
    absent_filenames: list[str] = Field(default_factory=list)

    #: Declared digests that disagree with the archive's own.
    sha256_mismatches: list[str] = Field(default_factory=list)

    #: Dimensions or sizes that disagree with the archive's.
    dimension_mismatches: list[str] = Field(default_factory=list)

    #: Groups whose slide count disagrees with the file count.
    slide_count_mismatches: list[str] = Field(default_factory=list)

    #: Duplicate and preview claims naming a file with no record.
    dangling_duplicate_targets: list[str] = Field(default_factory=list)
    dangling_preview_targets: list[str] = Field(default_factory=list)

    #: Every record not plainly matched, for review.
    notes: list[str] = Field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return {
            "records_total": self.records_total,
            "matched": self.matched,
            "missing_from_archive": self.missing_from_archive,
            "ambiguous": self.ambiguous,
            "conflicting": self.conflicting,
            "archive_files_absent_from_package": (
                self.archive_files_absent_from_package
            ),
            "sha256_mismatches": len(self.sha256_mismatches),
            "dimension_mismatches": len(self.dimension_mismatches),
            "slide_count_mismatches": len(self.slide_count_mismatches),
            "dangling_duplicate_targets": len(self.dangling_duplicate_targets),
            "dangling_preview_targets": len(self.dangling_preview_targets),
        }


class ImportManifest(BaseModel):
    """
    What one import run did, so the next one knows what it already has.

    Written to ``data/imported/gemini/manifest.json``. The package
    digest is what makes a re-import cheap to skip: the same package
    yields the same digest, and a changed package is a real change
    rather than a re-read of the same bytes.

    The archive is fingerprinted too, because the package describes the
    archive and a re-exported archive changes what the package means
    even when the package is untouched.
    """

    manifest_version: int = 1
    package: GeminiPackageInfo = Field(
        default_factory=GeminiPackageInfo
    )

    #: SHA-256 over the five package files' contents.
    package_digest: str = ""

    #: SHA-256 over the archive filenames and their digests.
    archive_digest: str = ""

    #: A cheaper fingerprint -- names, sizes and modification times --
    #: used only to decide whether a re-import can be skipped without
    #: re-reading 288 MB. Weaker than ``archive_digest`` and never
    #: quoted as evidence; see
    #: :func:`src.gemini.importer.archive_fingerprint`.
    fingerprint: str = ""

    records: int = 0
    groups: int = 0
    candidate_questions: int = 0
    ocr_available: int = 0
    ocr_no_text: int = 0
    ocr_unresolved: int = 0
    duplicates: int = 0
    previews: int = 0
    unreadable: int = 0

    imported_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def summary(self) -> str:
        return (
            f"{self.records} image record(s) across {self.groups} group(s); "
            f"{self.ocr_available} with OCR, {self.ocr_no_text} with no text, "
            f"{self.ocr_unresolved} unresolved; "
            f"{self.duplicates} duplicate(s), {self.previews} preview(s)"
        )


__all__ = [
    "CandidateVerdict",
    "CrossCheckReport",
    "GeminiCandidateQuestion",
    "GeminiImageRecord",
    "GeminiPackageInfo",
    "GeminiPostGroup",
    "GeminiPostKnowledge",
    "GeminiTopicIndex",
    "ImportManifest",
    "MatchVerdict",
    "OcrQuality",
    "OcrStatus",
    "SourceKind",
]
