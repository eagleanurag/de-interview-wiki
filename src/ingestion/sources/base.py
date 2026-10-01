"""
The source contract.

A source discovers content the user is already authorized to see,
normalizes it into the repository's post shape, and stops for a human
the moment it meets a security challenge.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from src.ingestion.post_document import (
    MediaEntry,
    PostDocument,
    media_path_for,
    media_type_for,
)


class StopReason(str, Enum):
    """Why a collection run ended.

    Recorded on every run so a partial collection is never mistaken for
    a complete one.
    """

    MAX_POSTS = "max_posts"
    UNTIL_DATE = "until_date"
    EXHAUSTED = "source_exhausted"
    NO_NEW_CONTENT = "no_new_content"
    SCROLL_LIMIT = "scroll_limit"
    SECURITY_CHALLENGE = "security_challenge"
    LAYOUT_CHANGED = "layout_changed"
    DRY_RUN = "dry_run"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


class CollectionStopped(RuntimeError):
    """
    A collection run ended for a recorded reason.

    Carries the reason so it can be written to a checkpoint instead of
    being lost in a traceback.
    """

    def __init__(
        self,
        reason: StopReason,
        message: str = "",
        *,
        state: "CollectionState | None" = None,
    ) -> None:
        super().__init__(message or reason.value)
        self.reason = reason
        self.state = state


class SecurityChallenge(CollectionStopped):
    """
    The source presented a challenge that must not be bypassed.

    CAPTCHA, MFA, OTP, account restriction or an explicit access denial.
    The correct response is always to stop and ask the human.
    """

    def __init__(
        self,
        kind: str,
        message: str = "",
        *,
        state: "CollectionState | None" = None,
    ) -> None:
        super().__init__(
            StopReason.SECURITY_CHALLENGE,
            message or f"security challenge: {kind}",
            state=state,
        )
        self.kind = kind


@dataclass
class CollectedPost:
    """
    One post as a source saw it, before persistence.

    ``media`` holds local files the source already has. ``media_urls``
    holds remote references a browser source found but has not
    downloaded. Both are optional; a post with neither is still valid.
    """

    source_post_id: str
    text: str
    url: str | None = None
    published_at: str | None = None
    author: str | None = None
    media: list = field(default_factory=list)
    media_urls: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def to_document(
        self,
        *,
        post_id: str,
        platform: str,
        captured_at: str | None = None,
    ) -> PostDocument:
        """
        Normalize into the repository's post shape.

        ``post_id`` is the already-normalized identifier the collector
        derived from ``source_post_id``. It is passed in rather than
        derived here, so the collector keeps ownership of the one
        mapping that decides where a post lives.

        Values are only filled in when the source actually provided
        them. Nothing is invented, so a missing author or timestamp
        stays missing rather than being fabricated.

        Media files are recorded by name only. The importer copies the
        bytes into the post directory, which is what keeps a source
        unable to write outside ``data/posts/``.
        """

        document = PostDocument.new(
            post_id,
            text=self.text,
            platform=platform,
            url=self.url,
            author=self.author,
            captured_at=captured_at,
        )

        for item in self.media:
            path = Path(item)

            document.upsert_media(
                MediaEntry(
                    type=media_type_for(path.name),
                    path=media_path_for(path.name),
                )
            )

        return document


@dataclass
class CollectionState:
    """Counts and cursor for a resumable run."""

    discovered: int = 0
    persisted: int = 0
    duplicates: int = 0
    failed: int = 0
    last_post_id: str = ""
    last_url: str = ""
    stopped_because: str = ""
    started_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict:
        return {
            "discovered": self.discovered,
            "persisted": self.persisted,
            "duplicates": self.duplicates,
            "failed": self.failed,
            "last_post_id": self.last_post_id,
            "last_url": self.last_url,
            "stopped_because": self.stopped_because,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CollectionState":
        return cls(
            discovered=int(payload.get("discovered", 0) or 0),
            persisted=int(payload.get("persisted", 0) or 0),
            duplicates=int(payload.get("duplicates", 0) or 0),
            failed=int(payload.get("failed", 0) or 0),
            last_post_id=str(payload.get("last_post_id") or ""),
            last_url=str(payload.get("last_url") or ""),
            stopped_because=str(payload.get("stopped_because") or ""),
            started_at=str(payload.get("started_at") or ""),
            updated_at=str(payload.get("updated_at") or ""),
        )


class Source(ABC):
    """
    Base class for a content source.

    Subclasses implement discovery and extraction. Persistence is handled
    by the collector, so a source only has to produce documents.
    """

    #: Short identifier stored on each post's source metadata.
    name: str = "source"

    #: Written to ``post.json`` as ``source.platform``.
    platform: str = "unknown"

    @abstractmethod
    def discover(self, **limits: object):
        """
        Yield discovered content.

        Must stop when its limits are reached and must raise
        :class:`SecurityChallenge` rather than attempt to get past a
        challenge.
        """

    def close(self) -> None:
        """Release any resources. Safe to call more than once."""

    def __enter__(self) -> "Source":
        return self

    def __exit__(self, *exception: object) -> None:
        self.close()