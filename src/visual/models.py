"""
What a visual asset is, and what was understood from it.

Two records, kept apart on purpose. An asset is a fact about a file:
where it came from, how many bytes, what shape, which slide it was. An
analysis is a claim about that asset's contents, made by a named
processor at a named version, with the digest of the asset it read.

Separating them is what makes incremental processing decidable. The
asset is compared by content; the analysis is reused when the content,
the processor and the configuration are all unchanged. Merging the two
would mean a processor upgrade invalidated nothing and a changed file
invalidated the wrong thing.

Provenance is carried explicitly rather than implied by nesting. Every
analysis names the asset digest it was made from and the sequence
number it came from, so a claim in the knowledge base can be walked back
to a slide. A claim that cannot name its slide does not belong here.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class AssetRole(str, Enum):
    """
    What an asset is, in the post's own terms.

    ``thumbnail`` is the one that earns its keep. The archive carries a
    low-resolution preview alongside the full carousel for 65 posts, and
    treating that preview as an independent slide would both double the
    work and let a 480x360 copy of slide one stand in for the real thing.
    """

    SLIDE = "slide"
    THUMBNAIL = "thumbnail"
    DOCUMENT = "document"
    IMAGE = "image"
    DIAGRAM = "diagram"
    SCREENSHOT = "screenshot"
    UNKNOWN = "unknown"


class ProcessingState(str, Enum):
    """
    Where one asset got to.

    ``failed`` is a real state rather than an absence, because a slide
    that could not be read has to stay visible. A hundred-slide post
    with ninety-nine understood and one unreadable is a different
    knowledge base from one with a hundred, and only an explicit record
    distinguishes them from a post that was never processed.
    """

    PENDING = "pending"
    PROCESSED = "processed"
    FAILED = "failed"
    SKIPPED = "skipped"
    REUSED = "reused"


class VisualAsset(BaseModel):
    """
    One file, described without interpreting it.

    Dimensions and format come from the bytes rather than the filename,
    which matters because 536 of the archive's 3,047 files are named
    ``.jpg`` and are not.
    """

    #: Path relative to the post directory, so the record is portable.
    path: str

    #: The archive's own filename, kept because it is the only thing
    #: that ties this asset back to a file a person can find.
    filename: str = ""

    media_type: Literal["image", "pdf", "video", "other"] = "image"

    #: Detected from content, never from the extension.
    format: str = ""

    width: int | None = None
    height: int | None = None
    byte_size: int = 0

    #: Content identity. The single thing reuse is decided on.
    sha256: str = ""

    #: Position in the post's visual narrative, zero-based, numeric.
    sequence: int = 0

    #: How many assets the post has in total, so a reader can tell a
    #: three-slide post from a truncated three-hundred-slide one.
    sequence_count: int = 1

    role: AssetRole = AssetRole.UNKNOWN

    #: The role before analysis, from the archive's own naming. Retained
    #: because it is what decides which assets are redundant, and it
    #: must not be able to change once analysis has run.
    declared_role: AssetRole = AssetRole.UNKNOWN

    state: ProcessingState = ProcessingState.PENDING

    #: Set when this asset's content is already known from another
    #: asset, by digest. The record keeps the relationship rather than
    #: deleting the asset, so a reader can see the slide existed and
    #: that it was not analysed twice.
    duplicate_of: str | None = None

    #: True for the asset whose analysis stands for a duplicate group.
    is_primary: bool = True

    #: Why an asset was skipped, when it was. Empty for anything that
    #: ran.
    note: str = ""

    @property
    def megapixels(self) -> float:
        if not self.width or not self.height:
            return 0.0

        return (self.width * self.height) / 1_000_000

    def identity(self) -> str:
        """
        The key a cached analysis is stored under.

        Asset digest, processor version and configuration together,
        because any of the three changing means the stored answer is no
        longer the answer that would be produced now.
        """

        return f"{self.sha256}:{PROCESSOR_VERSION}:{PROCESSOR_CONFIGURATION}"


#: Bumped when the analysis contract or the prompt changes. A stored
#: analysis from another version is not reusable, for the same reason a
#: stored enrichment from another version is not.
PROCESSOR_VERSION = "1"

#: What the processor was configured to do, folded into the reuse key so
#: that turning on diagram analysis re-reads slides it once transcribed
#: as plain text.
PROCESSOR_CONFIGURATION = "vision+bounded-batch"


class CodeBlock(BaseModel):
    """
    Code read out of an image.

    Marked as derived and kept verbatim. The language is only filled in
    when the processor was confident; guessing produces confidently
    wrong facts about how a technology is used, which is worse than
    leaving it blank.
    """

    language: str | None = None
    code: str
    #: How the text was obtained: transcription, vision, or both.
    method: str = "vision"


class VisualAnalysis(BaseModel):
    """
    What one processor understood from one asset.

    Every field is optional except the provenance, because partial
    understanding is the normal case and a record that insists on
    completeness invents the parts that were not read.
    """

    #: Digest of the asset this describes. Never empty: an analysis
    #: that cannot name its source is not evidence.
    source_asset_hash: str

    #: The asset's path, for display and for walking back to the file.
    source_path: str = ""

    #: The slide's position, so a claim names where it appeared.
    sequence: int = 0

    extracted_text: str | None = None

    visual_summary: str | None = None

    diagram_description: str | None = None

    code_blocks: list[CodeBlock] = Field(default_factory=list)

    #: Only what was legible on the slide. Not inferred, not completed.
    detected_technologies: list[str] = Field(default_factory=list)

    detected_topics: list[str] = Field(default_factory=list)

    detected_concepts: list[str] = Field(default_factory=list)

    #: Questions the slide itself raises or answers. The post-level
    #: synthesis decides which of these become knowledge; a slide is not
    #: a question generator on its own.
    interview_questions: list[str] = Field(default_factory=list)

    #: 0.0 to 1.0, and reported rather than trusted. A slide the
    #: processor admits it could not read is a different thing from one
    #: it read cleanly, and averaging them would hide the difference.
    confidence: float = 0.0

    processor: str = "vision"

    processor_version: str = PROCESSOR_VERSION

    #: Configuration the analysis was produced under.
    processor_configuration: str = PROCESSOR_CONFIGURATION

    #: When it ran, so a cache can be reasoned about without git.
    analysed_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    #: Set when this analysis was reused rather than produced.
    reused: bool = False

    #: Where the asset sat in a batch, when it was batched.
    batch_index: int | None = None

    @property
    def is_usable(self) -> bool:
        """
        Whether this analysis says anything at all.

        A slide can be readable and carry no text -- a photograph, a
        chart with no labels. That is a successful analysis of an
        uninformative asset, and it is different from a failed one.
        """

        return bool(
            self.extracted_text
            or self.visual_summary
            or self.diagram_description
            or self.code_blocks
        )

    def as_source_text(self) -> str:
        """
        The analysis rendered for the enrichment prompt.

        Deliberately plain. It is handed to the same grounding check as
        the post body, so anything stated here is measured against the
        post's own text and must survive that check like anything else.
        """

        parts: list[str] = []

        if self.extracted_text:
            parts.append(self.extracted_text.strip())

        for block in self.code_blocks:
            if block.code.strip():
                parts.append(block.code.strip())

        if self.diagram_description:
            parts.append(self.diagram_description.strip())

        if self.visual_summary:
            parts.append(self.visual_summary.strip())

        return "\n\n".join(part for part in parts if part)


class PostVisual(BaseModel):
    """
    Everything understood from one post's images, in slide order.

    Held per post rather than folded into the post's own analysis,
    because it is derived from files rather than said by the author. The
    post body remains the authoritative source; this is additional
    evidence about it, and the distinction survives into the wiki.
    """

    post_id: str

    assets: list[VisualAsset] = Field(default_factory=list)

    analyses: list[VisualAnalysis] = Field(default_factory=list)

    #: Assets that could not be read, with the reason. Never inferred
    #: content standing in for them.
    failures: list[str] = Field(default_factory=list)

    #: Model calls this post cost, and how many were reused from cache.
    calls: int = 0
    reused: int = 0

    #: Digest of the asset set and its ordering. Decides whether the
    #: post's visual stage needs running again at all.
    source_digest: str = ""

    processor_version: str = PROCESSOR_VERSION

    processor_configuration: str = PROCESSOR_CONFIGURATION

    analysed_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def ordered_analyses(self) -> list[VisualAnalysis]:
        """Analyses in slide order, deduplicated assets collapsed."""

        seen: set[str] = set()
        ordered: list[VisualAnalysis] = []

        for analysis in sorted(
            self.analyses, key=lambda item: item.sequence
        ):
            if analysis.source_asset_hash in seen:
                continue

            seen.add(analysis.source_asset_hash)
            ordered.append(analysis)

        return ordered

    def coverage(self) -> dict[str, int]:
        """What was understood, counted. Missing categories read as zero."""

        assets = self.assets
        analyses = self.ordered_analyses()

        return {
            "assets": len(assets),
            "slides": sum(
                1
                for asset in assets
                if asset.role is AssetRole.SLIDE
            ),
            "thumbnails": sum(
                1
                for asset in assets
                if asset.role is AssetRole.THUMBNAIL
            ),
            "duplicates": sum(
                1 for asset in assets if asset.duplicate_of
            ),
            "analysed": len(analyses),
            "with_text": sum(
                1 for analysis in analyses if analysis.extracted_text
            ),
            "with_code": sum(
                1
                for analysis in analyses
                if analysis.code_blocks
            ),
            "with_diagram": sum(
                1
                for analysis in analyses
                if analysis.diagram_description
            ),
            "failed": sum(
                1
                for asset in assets
                if asset.state is ProcessingState.FAILED
            ),
        }


__all__ = [
    "AssetRole",
    "CodeBlock",
    "PROCESSOR_CONFIGURATION",
    "PROCESSOR_VERSION",
    "PostVisual",
    "ProcessingState",
    "VisualAnalysis",
    "VisualAsset",
]
