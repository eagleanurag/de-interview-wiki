"""
Handing imported transcriptions to the pipeline as if vision had read them.

The important property of this module is what it does *not* do: it does
not add a second way for a post to get its media text. It builds the
objects ``inject()`` already expects -- a ``VisualAsset`` per file and a
``VisualAnalysis`` per transcription -- and everything downstream is
unchanged. The enricher builds its prompt from
``media.extracted_text``; the grounding check measures against the same
string; the wiki already renders a media item. None of that needed to
know where the text came from, which is why none of it had to change.

What *is* new is the provenance. A vision analysis and a Gemini
transcription are both "text read off a picture", but they are not
equally trustworthy and they must not be indistinguishable in the
output. Three things keep them apart:

**The processor name.** ``gemini_ocr`` rather than ``vision``, carried
into ``extraction_method`` and rendered by the wiki. A stored analysis
is keyed by asset digest *and* processor *and* configuration, so the two
never collide in the cache and either can be re-read without
invalidating the other.

**The confidence field is left at zero.** The tempting thing is to put
the OCR quality here. It would be wrong twice over: it is a different
measure from the one ``confidence`` means, and a transcription's
quality is a judgement about text rather than about whether the
processor succeeded at reading. The named ``ocr_quality`` states carry
that instead.

**The package's constant prose is not stored.** Every post in the
package carries the same two sentences under "Visual Description" and
"Source-Derived Explanation". They are a statement of method, not a
description of any slide, and writing them into ``visual_summary`` would
put a sentence about policy into the knowledge base once per image,
attributed to an image it says nothing about. They are dropped at parse
time, and ``visual_summary`` stays ``None``.

Nothing here reads a file, calls a model, or touches the archive. The
transcriptions have already been cross-checked against it; this step
only reshapes them.
"""

from __future__ import annotations

from pathlib import Path

from src.gemini.crosscheck import activity_id_for_filename
from src.gemini.derived import mentions
from src.gemini.models import (
    GeminiCandidateQuestion,
    GeminiImageRecord,
    MatchVerdict,
    OcrStatus,
    SourceKind,
)
from src.redaction import summarise as redact_note
from src.models import KnowledgePost, MediaItem
from src.visual.dedupe import (
    asset_digest,
    mark_duplicates,
    redundant_assets,
)
from src.visual.models import (
    AssetRole,
    PostVisual,
    ProcessingState,
    VisualAnalysis,
    VisualAsset,
)
from src.visual.stage import PostOutcome
from src.visual.stage import inject as inject_media


#: Bumped when the transcription contract changes. Kept separate from
#: CP12's processor version so that changing this path never invalidates
#: stored vision analyses, and vice versa.
PROCESSOR_VERSION = "1"

#: What produced these analyses, as a cache key and as a label.
CONFIGURATION = "gemini-ocr-import"

PROCESSOR = "gemini_ocr"


def logical_path(filename: str) -> str:
    """
    The path a record is referred to by, everywhere.

    Relative to the post's media directory, which is the same
    convention the committed posts use. A local absolute path would name
    a Windows user account in a published document, so nothing in this
    package ever constructs one.
    """

    return f"media/{filename}"


def _role_for(record: GeminiImageRecord, known: set[str]) -> AssetRole:
    """
    What this file is, from the package's claim plus the archive's names.

    A preview is only called a preview when the slide it previews is
    actually in the same group. The package's ``likely_preview_of``
    column is a claim, and a claim about a file that is not present
    cannot be acted on -- and marking such a file redundant would discard
    a post's only image.
    """

    if record.likely_preview_of and record.likely_preview_of in known:
        return AssetRole.THUMBNAIL

    if record.exact_duplicate_of:
        return AssetRole.SLIDE

    if Path(record.filename).stem.rsplit("_slide_", 1)[-1].isdigit():
        return AssetRole.SLIDE

    return AssetRole.IMAGE


def asset_for(
    record: GeminiImageRecord,
    role: AssetRole,
    sequence_count: int,
    known: set[str],
) -> VisualAsset:
    """One file, described without interpreting it."""

    state = ProcessingState.PROCESSED

    note = ""

    if record.verdict is MatchVerdict.MISSING_FROM_ARCHIVE:
        state = ProcessingState.FAILED

        note = (
            "the package transcribed this file but the archive does not "
            "have it"
        )

    elif record.ocr_status is OcrStatus.NO_TEXT:
        note = "the package reports no text in this image"

    elif record.ocr_status is OcrStatus.ABSENT:
        note = "the package said nothing about this image"

    elif record.ocr_status in {
        OcrStatus.UNRESOLVED,
        OcrStatus.FAILED,
        OcrStatus.PENDING,
    }:
        note = record.verdict_note or "the package could not read this image"

    if (
        record.likely_preview_of
        and record.likely_preview_of not in known
        and role is not AssetRole.THUMBNAIL
    ):
        # The package calls this a preview of a file that is not among
        # the records for this activity. Recorded and not acted on,
        # because marking it redundant would discard a post's only image
        # on the strength of a claim about a file nobody can check.
        note = (
            f"{note} The package calls this a preview of "
            f"{record.likely_preview_of}, which is not among the records "
            "for this activity."
        ).strip()

    return VisualAsset(
        path=logical_path(record.filename),
        filename=record.filename,
        media_type="image",
        # The package records no detected format, and the archive's
        # extension is known to lie for 536 files, so this stays empty
        # rather than repeating a name the bytes do not support.
        format="",
        width=record.width,
        height=record.height,
        byte_size=record.size_bytes,
        sha256=record.actual_sha256 or record.declared_sha256,
        sequence=record.slide_number,
        sequence_count=sequence_count,
        role=role,
        declared_role=role,
        state=state,
        note=note,
    )


def analysis_for(
    record: GeminiImageRecord,
    analysed_at: str,
) -> VisualAnalysis | None:
    """
    One transcription, in the shape the pipeline consumes.

    ``None`` when the record has no content identity, because an
    analysis whose ``source_asset_hash`` is empty cannot be attributed
    to a file and would collapse against every other unattributable one.
    The slide stays visible as an asset with a note; what is withheld is
    an attribution that could not be made.
    """

    digest = record.actual_sha256 or record.declared_sha256

    if not digest:
        return None

    return VisualAnalysis(
        source_asset_hash=digest,
        source_path=logical_path(record.filename),
        sequence=record.slide_number,
        # The publishable transcription, not the verbatim one. A
        # transcription of a Windows command prompt contains a local
        # filesystem path, and this text ends up in ``post.json``, in the
        # knowledge base and on the published site. The verbatim text
        # stays in memory as the fingerprint's input and is never
        # written; the substitution is recorded on the media item so the
        # page can say a path was replaced.
        extracted_text=record.public_text or record.raw_ocr_text,
        # Deliberately empty: the package's prose sections are the same
        # two sentences for every slide and describe no slide.
        visual_summary=None,
        diagram_description=None,
        processor=PROCESSOR,
        processor_version=PROCESSOR_VERSION,
        processor_configuration=CONFIGURATION,
        # Zero on purpose. Confidence means "how sure the processor is
        # that it read this", and a transcription's legibility is a
        # different question carrying a different answer. Conflating them
        # would let a legibility judgement into the one number the
        # pipeline treats as a reading success.
        confidence=0.0,
        analysed_at=analysed_at,
    )


def outcome_for(
    post_id: str,
    records: list[GeminiImageRecord],
    analysed_at: str,
) -> PostOutcome:
    """
    A post's outcome, ready for the existing ``inject()``.

    Duplicates and previews are marked by the same functions CP12 uses,
    so a post read by vision and a post read from the package produce
    the same shape and the same notion of what is redundant.
    """

    ordered = sorted(
        records, key=lambda record: (record.slide_number, record.filename)
    )

    known = {record.filename for record in ordered}

    assets = [
        asset_for(record, _role_for(record, known), len(ordered), known)
        for record in ordered
    ]

    mark_duplicates(assets)

    redundant_assets(assets)

    analyses: list[VisualAnalysis] = []

    failures: list[str] = []

    for record in ordered:
        analysis = analysis_for(record, analysed_at)

        if analysis is None:
            failures.append(record.filename)

            continue

        analyses.append(analysis)

    visual = PostVisual(
        post_id=post_id,
        assets=assets,
        analyses=analyses,
        failures=failures,
        calls=0,
        reused=len(analyses),
        processor_version=PROCESSOR_VERSION,
        processor_configuration=CONFIGURATION,
        analysed_at=analysed_at,
    )

    return PostOutcome(
        post_id=post_id,
        visual=visual,
        # Nothing was read now; these records were produced earlier by a
        # different tool. Reporting them as fresh reads would make the
        # numbers a reader sees a count of work this project did not do.
        processed=0,
        reused=len(analyses),
        calls=0,
        retries=0,
        failures=len(failures),
    )


def index_by_activity(
    records: list[GeminiImageRecord],
) -> dict[str, list[GeminiImageRecord]]:
    """
    Records grouped by the activity their filename names.

    Grouped by activity rather than by the package's own ``POST-nnn``,
    because the committed posts and CP12 both key on the activity, and
    a second key would mean a second way to say which post a slide
    belongs to.

    Keyed on the canonical bare identifier, which is what both sides
    reduce to: the package's ``activity_id`` column already is bare, and
    the committed post's media filename is stripped to match. Keying on
    either spelling alone would silently group nothing.
    """

    grouped: dict[str, list[GeminiImageRecord]] = {}

    for record in records:
        activity = record.activity_id

        if not activity:
            activity = activity_id_for_filename(record.filename) or ""

        if not activity:
            continue

        grouped.setdefault(activity, []).append(record)

    return grouped


def inject(
    post: KnowledgePost,
    outcome: PostOutcome,
    records: dict[str, GeminiImageRecord],
) -> KnowledgePost:
    """
    Put transcriptions into the post's media, in memory.

    Wraps :func:`src.visual.stage.inject` rather than reimplementing it,
    then adds the provenance fields that only this source can supply.
    Doing the media mutation in one place means the ordering, the
    duplicate handling and the "this stage adds to the post, it does not
    replace it" rule are not restated and later diverged.

    The wrapped function lives in ``src.visual`` rather than
    ``src.pipeline`` for the same reason this module does not import the
    pipeline at all: a source layer that imports the orchestration layer
    inverts the dependency, and the orchestrator imports this module.
    """

    # add_missing=False, and the reason is in :func:`src.visual.stage.inject`:
    # the package describes every image the archive holds for a post's
    # activity, including the 65 files no post references, and attaching
    # those fabricated media -- one carousel reached a 46,030 character
    # prompt, past the Windows command-line limit.
    post = inject_media(post, outcome, add_missing=False)

    slide_count = len(outcome.visual.assets)

    for item in post.media:
        name = Path(item.path).name

        record = records.get(name)

        if record is None:
            continue

        item.ocr_status = record.ocr_status.value
        item.ocr_quality = record.ocr_quality.value
        item.source_kind = SourceKind.IMAGE_OCR.value
        item.slide_count = slide_count

        # A local path was replaced in what the reader can see. Stated
        # on the item so the page says so, rather than presenting
        # sanitised text as if nothing had been removed.
        if record.redactions:
            item.ocr_note = "; ".join(
                part
                for part in (
                    item.ocr_note,
                    redact_note(record.redactions),
                )
                if part
            )

        # The evidence, kept even when the transcription is unusable:
        # a reader who wants to know why a slide reads as it does needs
        # the reason, not just the verdict.
        if record.ocr_note:
            item.ocr_note = "; ".join(
                part
                for part in (item.ocr_note, record.ocr_note)
                if part
            )

        if record.verdict is not MatchVerdict.MATCHED:
            item.ocr_note = "; ".join(
                part
                for part in (item.ocr_note, record.verdict_note)
                if part
            )

    post.media.sort(
        key=lambda entry: (entry.sequence, Path(entry.path).name)
    )

    return post


def media_for_post(
    post: KnowledgePost,
    grouped: dict[str, list[GeminiImageRecord]],
) -> list[GeminiImageRecord]:
    """
    The records belonging to one committed post.

    Matched on the activity in the media filename, which is the only
    link the committed post carries. A post whose files the package
    never mentioned gets an empty list and is left entirely alone.
    """

    for item in post.media:
        activity = activity_id_for_filename(item.path)

        if activity and activity in grouped:
            return grouped[activity]

    return []


def questions_for_post(
    post: KnowledgePost,
    grouped: dict[str, list[GeminiImageRecord]],
    candidates: list[GeminiCandidateQuestion],
) -> list[GeminiCandidateQuestion]:
    """
    The accepted candidates belonging to one post.

    Filtered to ACCEPT and REWRITE only, and matched on activity. A
    NEEDS_REVIEW or REJECT entry is still imported, still stored in full
    and still visible in the question report; it simply does not reach
    the knowledge base, because publishing a fragment as an interview
    question is the specific failure this gate exists to prevent.
    """

    activities = {
        activity_id_for_filename(item.path)
        for item in post.media
        if activity_id_for_filename(item.path)
    }

    if not activities:
        return []

    return [
        candidate
        for candidate in candidates
        if candidate.question
        and candidate.verdict.value in {"accept", "rewrite"}
        and candidate.activity_id in activities
    ]


def ocr_digest(outcome: PostOutcome | None) -> str:
    """
    The fingerprint of what the transcriptions contribute to a post.

    Covers each contributing asset's content digest, its position and its
    role, plus the processor version and configuration -- the same
    construction :func:`src.pipeline.visual.digest_for` uses, with this
    package's own version and configuration so the two can never be
    confused for one another.

    Empty for a post the package says nothing about, which is what makes
    a post that never needed this layer behave exactly as it did before
    it existed.
    """

    if outcome is None or not outcome.visual.assets:
        return ""

    marked = list(outcome.visual.assets)

    mark_duplicates(marked)

    redundant_assets(marked)

    return asset_digest(
        marked,
        processor_version=PROCESSOR_VERSION,
        configuration=CONFIGURATION,
    )


class GeminiIndex:
    """
    The imported transcriptions, indexed for the enrichment stage.

    Loaded once and consulted per post, because loading is the expensive
    part and a run asks about several hundred posts. Nothing here is
    per-post mutable, so one instance can serve every worker thread.

    ``None`` for a post with no records is the answer more often than not:
    the package covers 312 activities and the corpus is larger, so most
    posts have nothing to contribute and must be left exactly as they
    were.
    """

    def __init__(
        self,
        records: list[GeminiImageRecord],
        candidates: list[GeminiCandidateQuestion] | None = None,
        analysed_at: str = "",
        technologies: list[str] | None = None,
    ) -> None:
        self.by_activity = index_by_activity(records)

        self.by_filename = {record.filename: record for record in records}

        self.candidates = list(candidates or [])

        self.accepted = [
            candidate
            for candidate in self.candidates
            if candidate.question
            and candidate.verdict.value in {"accept", "rewrite"}
        ]

        self.analysed_at = analysed_at

        #: The package's own per-post technology claims, for checking against
        #: what the transcriptions actually say. Grouped rather than
        #: global, because two posts can share a technology and a claim
        #: about one post must not attach to another.
        self.technologies_by_group: dict[str, list[str]] = {}

        for group_id, names in (technologies or {}).items():
            self.technologies_by_group[group_id.upper()] = list(names)

        self._outcomes: dict[str, PostOutcome] = {}

    def __len__(self) -> int:
        return len(self.by_activity)

    def outcome_for(self, post: KnowledgePost) -> PostOutcome | None:
        """
        The outcome for one post, or ``None`` when the package has
        nothing for it.

        Cached per post identifier. The outcome is a pure function of the
        records, so building it twice would produce two equal objects and
        one more chance for them to differ.
        """

        if post.id in self._outcomes:
            return self._outcomes[post.id]

        records = media_for_post(post, self.by_activity)

        if not records:
            return None

        outcome = outcome_for(post.id, records, self.analysed_at)

        self._outcomes[post.id] = outcome

        return outcome

    def records_for(self, post: KnowledgePost) -> dict[str, GeminiImageRecord]:
        """This post's records, keyed by filename."""

        return {
            record.filename: record
            for record in media_for_post(post, self.by_activity)
        }

    def questions_for(self, post: KnowledgePost) -> list[GeminiCandidateQuestion]:
        """The accepted questions belonging to this post."""

        return questions_for_post(post, self.by_activity, self.accepted)

    def technologies_for(self, post: KnowledgePost) -> list[str]:
        """
        Technologies this post's slide images name, and nothing else.

        A name is included only when some transcription for this post
        actually contains it. The package's per-post technology list is
        a broader claim -- it attributes a technology to a post -- and
        reproducing it unqualified would put a term into the knowledge
        base that no image in the post mentions.

        Deliberately not merged with the post's own text technologies:
        that list is what the author said, this is what a machine read
        off a picture, and the two are kept distinguishable all the way
        to the page.
        """

        names: list[str] = []

        records = media_for_post(post, self.by_activity)

        claimed: set[str] = set()

        for record in records:
            claimed.update(
                self.technologies_by_group.get(record.post_id.upper(), [])
            )

        for record in records:
            text = record.readable_text

            if not text:
                continue

            for name in sorted(claimed):
                if name in names:
                    continue

                if mentions(text, name):
                    names.append(name)

        return names

    def digest_for(self, post: KnowledgePost) -> str:
        """The OCR fingerprint for one post, for the freshness check."""

        return ocr_digest(self.outcome_for(post))


def load_index(output: str | Path | None = None) -> GeminiIndex | None:
    """
    The imported package, from a previous import.

    ``None`` when nothing has been imported, which is a normal state and
    not an error: this layer is optional, and a project that never ran an
    import should enrich exactly as it did before.
    """

    from src.gemini.importer import (
        DEFAULT_OUTPUT,
        load_candidates as load_candidates_,
        load_knowledge,
        load_records,
        read_manifest,
    )

    base = Path(output) if output is not None else DEFAULT_OUTPUT

    manifest = read_manifest(base)

    if manifest is None:
        return None

    technologies = {
        group_id: list(entry.technologies)
        for group_id, entry in load_knowledge(base).items()
    }

    return GeminiIndex(
        load_records(base),
        load_candidates_(base),
        analysed_at=manifest.imported_at,
        technologies=technologies,
    )


def to_media_items(records: list[GeminiImageRecord]) -> list[MediaItem]:
    """
    Transcriptions as media items, for callers that want them directly.

    Used by the search index and the wiki's post pages, which need the
    transcription and its status without the whole visual-stage
    machinery. Kept separate from :func:`inject` because a media item
    on its own is not provenance-checked against a post.
    """

    items: list[MediaItem] = []

    for record in sorted(
        records, key=lambda entry: (entry.slide_number, entry.filename)
    ):
        items.append(
            MediaItem(
                type="image",
                path=logical_path(record.filename),
                # Publishable text only, for the same reason as
                # ``analysis_for``: this reaches the site.
                extracted_text=record.public_text or record.raw_ocr_text,
                extraction_method=PROCESSOR,
                sequence=record.slide_number,
                sha256=record.actual_sha256 or record.declared_sha256 or None,
                ocr_status=record.ocr_status.value,
                ocr_quality=record.ocr_quality.value,
                ocr_note=record.ocr_note or None,
                source_kind=SourceKind.IMAGE_OCR.value,
                slide_count=len(records),
                description=(
                    f"image, {record.width or '?'}x{record.height or '?'} "
                    f"pixels, slide {record.slide_number + 1}"
                ),
            )
        )

    return items


__all__ = [
    "CONFIGURATION",
    "PROCESSOR",
    "PROCESSOR_VERSION",
    "GeminiIndex",
    "analysis_for",
    "asset_for",
    "index_by_activity",
    "inject",
    "load_index",
    "logical_path",
    "media_for_post",
    "ocr_digest",
    "outcome_for",
    "questions_for_post",
    "to_media_items",
]