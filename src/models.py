from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class MediaItem(BaseModel):
    type: Literal["image", "pdf", "video", "other"]
    path: str
    description: str | None = None
    extracted_text: str | None = None


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
    """

    source_digest: str = ""
    enricher_version: str = ""
    enriched_at: str = ""


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