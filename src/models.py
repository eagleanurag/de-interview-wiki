from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class MediaItem(BaseModel):
    type: Literal["image", "pdf", "video", "other"]
    path: str
    description: str | None = None
    extracted_text: str | None = None

    #: Position in the post's own sequence, zero-based.
    #:
    #: Added with visual enrichment. A post carrying a carousel needs to
    #: say which image came first, and the filename cannot be trusted to
    #: say it: this archive numbers one post's slides ``_slide_0`` and
    #: ``_slide_01`` on the same post, and pads some to three digits and
    #: others to one. Defaulted to zero so every existing post still
    #: validates, which is what lets the field arrive without rewriting
    #: 490 committed documents.
    sequence: int = 0

    #: What the asset is, when known: slide, thumbnail, document, image.
    role: str | None = None

    #: Digest of the file's bytes, when it has been read. Recorded so a
    #: visual claim in ``extracted_text`` can be traced to the exact file
    #: it came from, and so a changed file is detectable without
    #: re-reading every other one.
    sha256: str | None = None

    #: How this asset's content was obtained: ``vision``, ``ocr``, or
    #: unset for a file never read beyond its metadata.
    extraction_method: str | None = None

    # -- Provenance of a machine transcription -----------------------
    #
    # A transcription of a slide and a sentence the author typed are
    # different kinds of claim, and a reader who cannot tell them apart
    # will quote a misread of a 480-pixel screenshot as though the
    # author said it. These four fields are what let the wiki say which
    # one this is.
    #
    # Optional and omitted when unset, so the 490 posts committed before
    # any of this existed still validate and serialise byte-for-byte as
    # they did.

    #: What the transcription claims about the image: ``AVAILABLE``,
    #: ``NO_TEXT``, ``PARTIAL``, ``UNRESOLVED``, or ``ABSENT`` for an
    #: image a tool said nothing about. ``ABSENT`` is deliberately not
    #: folded into ``NO_TEXT``: "there is no text here" and "nobody
    #: looked" are different facts and only the first is a finding.
    ocr_status: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )

    #: What is observably true of the transcription: ``readable``,
    #: ``fragmentary``, ``garbled``, or ``empty``. A named state rather
    #: than a score, because a number would need a threshold and the
    #: threshold would be an invention.
    ocr_quality: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )

    #: The evidence behind ``ocr_quality``, written out. Present so the
    #: judgement can be argued with rather than merely accepted.
    ocr_note: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )

    #: Which kind of source produced the text on this item:
    #: ``author_source``, ``image_ocr``, ``visual_derived`` or
    #: ``ai_enrichment``.
    source_kind: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )

    #: How many images the post has in total, so a three-slide post can
    #: be told from a truncated three-hundred-slide one. Per post rather
    #: than per item, and recorded on every item so no reader has to
    #: reconstruct it.
    slide_count: int | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )


class SourceInfo(BaseModel):
    platform: str
    url: str | None = None
    captured_at: datetime
    author: str | None = None

    # How the content actually reached the project. A post is either
    # collected by an authorized automated run, or supplied by a person
    # as a file. The two are not interchangeable, so the difference is
    # recorded rather than left to be guessed from the platform: a
    # LinkedIn post that a user exported and saved says so.
    #
    # Omitted from a dump when it is not set, because a post that does
    # not declare a capture method has nothing to declare, and a null in
    # every entry of the published knowledge base would be noise.
    capture_method: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )

    # When the source itself published the content, exactly as it
    # rendered it. A string rather than a datetime because a relative
    # form such as "2 days ago" is only resolvable against the capture
    # date, and a wrong timestamp would be worse than the original.
    published_at: str | None = None


class AIAnalysis(BaseModel):
    summary: str | None = None
    topics: list[str] = Field(default_factory=list)
    subtopics: list[str] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    image_descriptions: list[str] = Field(default_factory=list)

    #: Technologies named only inside an attached image, each one present
    #: in that image's own transcription.
    #:
    #: A separate list rather than an addition to ``concepts`` because
    #: the evidence is weaker and must not look identical to a claim the
    #: author made in the post body. ``detect_technologies`` reads the
    #: post text only, and deliberately so -- a term appearing in a
    #: post's *body* is the author saying it. This list says "a
    #: machine read this off an attached picture", which is a different
    #: claim and is kept apart.
    #:
    #: Both the knowledge base and the site read this, so the two cannot
    #: disagree about which posts cover a technology. Excluded when
    #: empty, which is every post with no images.
    derived_technologies: list[str] = Field(
        default_factory=list,
        exclude_if=lambda value: not value,
    )


class InterviewQuestion(BaseModel):
    question: str
    type: Literal[
        "theory",
        "coding",
        "scenario",
        "architecture",
        "troubleshooting",
    ]
    difficulty: Literal["easy", "medium", "hard"]
    answer: str | None = None

    # -- Where a question and its answer came from -------------------
    #
    # A question read off a slide is not a question a person wrote, and
    # a transcription is not an answer. Without these two fields the
    # knowledge base would present both as though they were authored
    # here, which is the specific misreading this whole layer is
    # arranged to prevent.

    #: ``ai_enriched`` for a question this project's enricher produced,
    #: ``image_ocr`` for one read out of a slide's transcribed text.
    source_kind: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )

    #: ``ai_enriched`` for an answer composed from the whole post, or
    #: ``source_excerpt`` for a passage quoted out of the slide the
    #: question was read from. The distinction is the difference between
    #: something reasoned out and something copied, and a reader is
    #: entitled to know which they have.
    answer_source: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )

    #: Which slide, in prose: the file and its position. Kept as a
    #: sentence rather than as separate fields because it is only ever
    #: read as a sentence, on the card, under the question.
    source_note: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )


class Classification(BaseModel):
    domain: str = "Data Engineering"
    primary_topic: str | None = None
    secondary_topics: list[str] = Field(default_factory=list)
    interview_relevant: bool = False


class EnrichmentFingerprint(BaseModel):
    """
    What an enrichment describes and which version produced it.

    Excluded from the canonical knowledge base: it records how this
    pipeline processed a post, not what the post says, and publishing
    it would put build metadata into the reader-facing output.

    ``source_digest`` covers the post's own material: its text and the
    bytes of the files attached to it. ``visual_digest`` covers what was
    read *out of* those files -- their sequence as well as their content
    -- so that reordering a carousel invalidates the enrichment without
    a changed text or a changed file, and so that upgrading the visual
    processor re-derives the knowledge that depended on it.

    Kept apart on purpose. Folding the visual stage into the source
    digest would have made the enrichment depend on provider output,
    and a provider that is not byte-identical between runs would then
    invalidate every post on every run.
    """

    source_digest: str = ""
    enricher_version: str = ""
    enriched_at: str = ""

    #: Digest of the post's visual assets, their content and their
    #: order, plus the processor version and configuration that read
    #: them. Empty for a post with no images, which is most of them.
    visual_digest: str = ""

    visual_processor_version: str = ""

    #: Digest of machine transcriptions contributed from outside the
    #: post -- the imported image-knowledge package -- plus the
    #: version that produced them.
    #:
    #: A third field rather than an extension of ``visual_digest``
    #: because these are different sources with different authority. The
    #: vision processor reads this project's own copy of an image on
    #: demand; the imported package is a third party's transcription of
    #: those same images, produced elsewhere and at another time. Folding
    #: them together would mean a change to either silently
    #: invalidate enrichment derived from the other, and would make the
    #: recorded version claim both were used when only one was.
    #:
    #: Empty for every post the package says nothing about, which is
    #: every post with no images and most of the rest.
    ocr_digest: str = ""

    ocr_processor_version: str = ""


class SavedItemProvenance(BaseModel):
    """
    Which saved item a post came from, and how complete it was.

    A saved list is a list of links, and a link is not a post. This
    records the difference so a reader can tell material that was
    captured from material that was only linked, and can get back to the
    item it came from.

    Kept out of a post that did not come from a saved list, rather than
    written empty, because a block of nulls on every post would say
    "this was saved" about posts that were not.
    """

    saved_item_id: str = ""
    saved_date: str | None = None
    canonical_url: str | None = None
    url_kind: str | None = None
    capture_state: str | None = None
    capture_match: str | None = None

    # How complete the capture is, read off what it contained: a link
    # on its own, a screenshot, a document, or text. Not a score,
    # because a number with nothing behind it cannot be checked.
    capture_quality: str | None = None

    capture_notes: list[str] = Field(default_factory=list)
    saved_notes: str | None = None
    metadata_only: bool = False


class KnowledgePost(BaseModel):
    id: str
    source: SourceInfo
    original_text: str = ""
    media: list[MediaItem] = Field(default_factory=list)

    ai_analysis: AIAnalysis = Field(default_factory=AIAnalysis)

    interview_questions: list[InterviewQuestion] = Field(
        default_factory=list
    )

    classification: Classification = Field(
        default_factory=Classification
    )

    # Where this post was loaded from. Excluded from the serialized
    # form because it is a fact about this machine, not about the
    # content: a knowledge base that carried it would stop being
    # portable the moment it was written anywhere else. Media paths
    # stay relative and are resolved against this.
    directory: str = Field(default="", exclude=True)

    # Which content the analysis describes, and which version of the
    # enrichment produced it. Lets a later run enrich only what
    # changed instead of paying for the model on every post.
    enrichment: EnrichmentFingerprint = Field(
        default_factory=EnrichmentFingerprint, exclude=True
    )

    # Which saved item this post came from, for a post that came from
    # one. Unlike the fields above this is part of the published
    # knowledge base: it is a fact about where the content came from,
    # not about how this pipeline processed it, and a reader needs it to
    # tell a captured post from a link. Omitted when there is none, so a
    # post that was never saved does not carry a null claiming it was.
    saved_item: SavedItemProvenance | None = Field(
        default=None, exclude_if=lambda value: value is None
    )