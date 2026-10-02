"""
The Saved Item record.

A Saved Item is one thing the user saved, and it may or may not have
content behind it. That distinction is the whole point of the model: an
export of LinkedIn's saved list gives a URL and a date and nothing
else, and pretending otherwise would put an invented post body into
the knowledge base.

So a Saved Item carries what the user actually supplied, records
whether content was found for it, and stays metadata-only when it was
not. It never claims a body it does not have.

Nothing here holds a credential. A Saved Item is derived from a file
the user exported; there is no password, cookie, token or session
state in this model, and the fields that would hold them if they
existed are not part of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from src.ingestion.saved_items.urls import NormalizedUrl


class SavedItemState(str, Enum):
    """
    Where a Saved Item is in the pipeline.

    The states are ordered by how much has happened to the item, so a
    run can report progress by counting them and a later run can pick up
    where an earlier one stopped. Every state is written down, so
    resuming never depends on guessing.
    """

    #: Discovered from a manifest, with no content looked for yet.
    PENDING = "pending"

    #: Content was found for it and read, but not stored as a post.
    CAPTURED = "captured"

    #: Stored as a post in ``data/posts``.
    IMPORTED = "imported"

    #: Enriched: the enrichment stage has produced an analysis.
    ENRICHED = "enriched"

    #: Something went wrong. The reason is recorded, never guessed at.
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in {
            SavedItemState.ENRICHED,
            SavedItemState.FAILED,
        }

    @property
    def rank(self) -> int:
        order = {
            SavedItemState.PENDING: 0,
            SavedItemState.CAPTURED: 1,
            SavedItemState.IMPORTED: 2,
            SavedItemState.ENRICHED: 3,
            SavedItemState.FAILED: -1,
        }

        return order[self]


#: The kind recorded on an item that has no URL and is identified by an
#: identifier instead.
#:
#: Says how the item was identified, not where it came from. A local
#: archive is the common case, but the name would be wrong for any other
#: list that carries identifiers rather than links, and the source id
#: already carries its own namespace for that.
KIND_IDENTIFIED_BY_ID = "identified_by_id"


class CaptureMethod(str, Enum):
    """
    How the content actually entered the system.

    Recorded because it is the difference between material this project
    read and material a person supplied. Nothing here claims anything
    was collected from LinkedIn automatically.
    """

    #: A list the user exported from LinkedIn.
    USER_EXPORT = "user_export"

    #: Content the user captured or saved themselves.
    USER_PROVIDED = "user_provided"

    #: A page the user saved from their browser and supplied as HTML.
    USER_SAVED_PAGE = "user_saved_page"

    #: A bundle the user assembled with text and media.
    USER_BUNDLE = "user_bundle"

    def to_platform_source(self) -> str:
        return "linkedin"


@dataclass
class SavedItem:
    """
    One saved item.

    ``original_url`` is kept beside the canonical form so a reader can
    see exactly what was saved, and ``source_id`` is the only thing
    deduplication keys on, because it is derived from the canonical URL.
    """

    original_url: str
    canonical_url: str
    source_id: str
    kind: str
    identifier: str | None

    saved_date: str | None = None
    title: str | None = None
    author: str | None = None
    notes: str | None = None

    capture_method: CaptureMethod = CaptureMethod.USER_EXPORT

    state: SavedItemState = SavedItemState.PENDING

    #: A bundle this item was pointed at by the manifest, when the
    #: manifest names one instead of relying on the directory name.
    bundle: str | None = None

    #: Set once content has been read for this item.
    content_path: str | None = None
    content_digest: str | None = None
    media_paths: list[str] = field(default_factory=list)

    #: How complete the capture is, read off what it actually contains.
    #: A link on its own, a screenshot and a transcript are different
    #: things to a reader, and calling them all "content" would hide the
    #: difference that matters.
    capture_quality: str = "metadata_only"

    #: What was actually found in the capture, and what was not. Kept
    #: so a report can say "a screenshot with no text" rather than
    #: implying a body was recovered.
    capture_notes: list[str] = field(default_factory=list)

    #: The post this item became, once it has been stored.
    post_id: str | None = None

    failure_reason: str | None = None

    created_at: str = ""
    updated_at: str = ""

    #: Whether the user supplied content, as opposed to only a link.
    @property
    def has_content(self) -> bool:
        return bool(self.content_digest)

    @property
    def is_metadata_only(self) -> bool:
        return not self.has_content

    @property
    def is_linkedin_content(self) -> bool:
        return self.kind not in {"external", "linkedin_other"}

    @property
    def is_external(self) -> bool:
        return self.kind == "external"

    def touch(self) -> "SavedItem":
        self.updated_at = datetime.now(timezone.utc).isoformat()

        if not self.created_at:
            self.created_at = self.updated_at

        return self

    def merge(self, other: "SavedItem") -> "SavedItem":
        """
        Fold a freshly discovered copy of this item into this one.

        The stored record wins for anything it already knows, because a
        later export carries less than the run that read the content.
        A field that is empty here and present in the new record is
        filled, because losing it would be losing the user's data.
        """

        for name in (
            "saved_date",
            "title",
            "author",
            "notes",
            "bundle",
        ):
            if not getattr(self, name) and getattr(other, name):
                setattr(self, name, getattr(other, name))

        if other.content_digest and other.content_digest != self.content_digest:
            # The content behind this link changed, so the stored
            # analysis no longer describes it.
            self.content_digest = other.content_digest
            self.content_path = other.content_path
            self.media_paths = list(other.media_paths)
            self.capture_notes = list(other.capture_notes)
            self.capture_quality = other.capture_quality
            self.state = SavedItemState.CAPTURED

        elif other.content_digest and not self.content_digest:
            self.content_digest = other.content_digest
            self.content_path = other.content_path
            self.media_paths = list(other.media_paths)
            self.capture_notes = list(other.capture_notes)
            self.capture_quality = other.capture_quality
            self.state = SavedItemState.CAPTURED

        if other.state.rank > self.state.rank:
            self.state = other.state

        if not self.failure_reason and other.failure_reason:
            self.failure_reason = other.failure_reason

        return self.touch()

    def as_dict(self) -> dict:
        """The stored form. Contains no credential, by construction."""

        return {
            "source_id": self.source_id,
            "original_url": self.original_url,
            "canonical_url": self.canonical_url,
            "kind": self.kind,
            "identifier": self.identifier,
            "saved_date": self.saved_date,
            "title": self.title,
            "author": self.author,
            "notes": self.notes,
            "capture_method": self.capture_method.value,
            "state": self.state.value,
            "bundle": self.bundle,
            "content_path": self.content_path,
            "content_digest": self.content_digest,
            "capture_quality": self.capture_quality,
            "media": list(self.media_paths),
            "capture_notes": list(self.capture_notes),
            "post_id": self.post_id,
            "failure_reason": self.failure_reason,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "SavedItem":
        return cls(
            original_url=str(payload.get("original_url") or ""),
            canonical_url=str(payload.get("canonical_url") or ""),
            source_id=str(payload.get("source_id") or ""),
            kind=str(payload.get("kind") or "unknown"),
            identifier=payload.get("identifier"),
            saved_date=payload.get("saved_date"),
            title=payload.get("title"),
            author=payload.get("author"),
            notes=payload.get("notes"),
            capture_method=_capture_method(
                payload.get("capture_method")
            ),
            state=_state(payload.get("state")),
            bundle=payload.get("bundle"),
            content_path=payload.get("content_path"),
            content_digest=payload.get("content_digest"),
            capture_quality=_quality(payload.get("capture_quality")),
            media_paths=[
                str(item) for item in (payload.get("media") or [])
            ],
            capture_notes=[
                str(item) for item in (payload.get("capture_notes") or [])
            ],
            post_id=payload.get("post_id"),
            failure_reason=payload.get("failure_reason"),
            created_at=str(payload.get("created_at") or ""),
            updated_at=str(payload.get("updated_at") or ""),
        )

    @classmethod
    def from_url(
        cls,
        normalized: NormalizedUrl,
        *,
        capture_method: CaptureMethod = CaptureMethod.USER_EXPORT,
    ) -> "SavedItem":
        return cls(
            original_url=normalized.original,
            canonical_url=normalized.canonical,
            source_id=normalized.source_id,
            kind=normalized.kind,
            identifier=normalized.identifier,
            capture_method=capture_method,
        ).touch()

    @classmethod
    def from_source_id(
        cls,
        source_id: str,
        *,
        kind: str = KIND_IDENTIFIED_BY_ID,
        identifier: str | None = None,
        capture_method: CaptureMethod = CaptureMethod.USER_EXPORT,
    ) -> "SavedItem":
        """
        An item identified by an identifier rather than by a link.

        For material that genuinely has no URL. A local archive of saved
        posts holds a third of its records with no permalink, because the
        card they were captured from did not expose one, and there is
        nothing in the media filenames or the media URLs that would
        recover it.

        Inventing a URL for those would be the one thing this model
        exists to prevent: the wiki would render a link that goes
        nowhere and a reader would believe the project had one. So the
        item says it has no link, is identified by the identifier it does
        have, and is stored with an empty ``canonical_url`` -- which the
        renderer already treats as "render no link".

        The identifier is required and is not checked against a URL
        namespace, because the caller is stating the identity rather
        than deriving it. Callers that are deriving an identity use
        :func:`source_id_for`, which hashes a canonical URL.
        """
        if not source_id or not source_id.strip():
            raise ValueError(
                "an item with no URL still needs an identifier to be "
                "identified by"
            )

        return cls(
            original_url="",
            canonical_url="",
            source_id=source_id.strip(),
            kind=kind,
            identifier=identifier,
            capture_method=capture_method,
        ).touch()


def _state(value: object) -> SavedItemState:
    if isinstance(value, SavedItemState):
        return value

    try:
        return SavedItemState(str(value))
    except ValueError:
        # An unknown state is read as pending rather than trusted,
        # because acting on a state this version does not understand
        # would be worse than redoing the work.
        return SavedItemState.PENDING


def _capture_method(value: object) -> CaptureMethod:
    if isinstance(value, CaptureMethod):
        return value

    try:
        return CaptureMethod(str(value))
    except ValueError:
        return CaptureMethod.USER_EXPORT


def _quality(value: object) -> str:
    """
    A recorded capture quality, or the most cautious reading of it.

    An unrecognised value is read as ``metadata_only`` rather than
    trusted. A stored quality that claims text the pipeline has never
    seen would put a transcript in front of a reader who then finds a
    bare link, and the cautious reading is the one that cannot lie.
    """

    from src.ingestion.saved_items.bundles import (
        CAPTURE_QUALITIES,
        QUALITY_METADATA_ONLY,
    )

    if isinstance(value, str) and value in CAPTURE_QUALITIES:
        return value

    return QUALITY_METADATA_ONLY