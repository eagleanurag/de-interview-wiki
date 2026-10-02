"""
Identity for records in a local LinkedIn archive.

A saved-items item is identified by its canonical URL, which is a good
identity when there is a URL. A third of this archive has none: the
record was captured from a card that did not expose a permalink, and
neither the media filenames nor the original media URLs carry one. That
is a real fact about the record, not a gap to be papered over.

So a record without a permalink is identified by the archive's own
identifier, namespaced so it can never collide with a URL-derived one,
and it is recorded as having no URL. Nothing here invents a link, and
the two identity kinds stay distinguishable in the manifest, the report
and the published page.

The two are deliberately different in shape. A URL-derived id is
``urn:li:saved:<16 hex>`` and this module never produces that form; an
archive-only id is ``urn:li:archive:<16 hex>``. A test asserts the two
cannot be confused, because an id that looked like a URL-derived one
would make a reader believe a link existed.
"""

from __future__ import annotations

import hashlib
import re

from src.ingestion.saved_items.urls import (
    SavedItemUrlError,
    normalize_linkedin_url,
    source_id_for,
)


#: The namespace for an item the archive could not give a URL.
ARCHIVE_NAMESPACE = "urn:li:archive:"

#: The kind an archive-only item carries.
#:
#: Namespaced rather than reusing the saved-items kind, so a reader of a
#: manifest can tell the two identity schemes apart by the shape of the
#: id alone. The value itself is the generic one the model uses for any
#: item identified without a link.
KIND_ARCHIVE_ONLY = "identified_by_id"

#: A key that puts a record's own identity ahead of its content, so two
#: records that happen to share text stay two items.
CONTENT_NAMESPACE = "urn:li:archive-content:"

_WORD = re.compile(r"[^a-z0-9]+")


def digest(value: str, namespace: str) -> str:
    """
    A stable short digest under a namespace.

    Sixteen hex characters is the same width the URL-derived identity
    uses, so the two are interchangeable in a directory name and a URL
    segment. The namespace is part of the hashed input, so an archive id
    can never equal a URL-derived id even for identical text.
    """

    return hashlib.sha256(f"{namespace}:{value}".encode()).hexdigest()[:16]


def content_fingerprint(text: str) -> str:
    """
    A digest of a post's whole text, for detecting repeated content.

    Normalised rather than raw so two renderings of the same post
    collapse: case, punctuation and whitespace are not differences
    anyone would call a different post.

    The *whole* text is hashed. A prefix was tried first and produced a
    false positive, folding two posts that agree for the first few
    hundred characters and diverge afterwards into one. A duplicate that
    silently swallows a second post is worse than a duplicate that gets
    reported, so the digest covers everything.
    """

    normalised = _WORD.sub(" ", str(text or "").lower()).strip()

    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


class ArchiveIdentity:
    """
    How one archive record is identified, and what it honestly claims.

    ``has_permalink`` is stored rather than inferred from the URL being
    empty, because "the archive had no permalink" and "the permalink was
    dropped because it was unusable" are different facts and only one of
    them is true of these records.
    """

    __slots__ = (
        "source_id",
        "canonical_url",
        "original_url",
        "kind",
        "identifier",
        "has_permalink",
        "activity_id",
    )

    def __init__(
        self,
        *,
        source_id: str,
        canonical_url: str,
        original_url: str,
        kind: str,
        identifier: str | None,
        has_permalink: bool,
        activity_id: str | None,
    ) -> None:
        self.source_id = source_id
        self.canonical_url = canonical_url
        self.original_url = original_url
        self.kind = kind
        self.identifier = identifier
        self.has_permalink = has_permalink
        self.activity_id = activity_id

    @property
    def bundle_name(self) -> str:
        """
        The folder name this record's capture lives in.

        A filesystem-safe rendering of the identity, so a post whose id
        contains a colon does not have to be renamed by hand before it
        can be a directory.
        """

        return self.source_id.replace(":", "-")

    def as_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "canonical_url": self.canonical_url,
            "original_url": self.original_url,
            "kind": self.kind,
            "identifier": self.identifier,
            "has_permalink": self.has_permalink,
            "activity_id": self.activity_id,
        }


def activity_id_of(post_id: str) -> str | None:
    """
    The LinkedIn activity number in an archive post id, if it has one.

    The archive names an activity-backed post ``activity_<digits>``. The
    number is the one part of such a post that LinkedIn itself would
    recognise, so it is kept separately for provenance even when the
    record carries no permalink to put it in.
    """

    match = re.fullmatch(r"activity_(\d{6,})", str(post_id or ""))

    return match.group(1) if match else None


def identify(
    post_id: str,
    permalink: object,
    *,
    text: str = "",
) -> ArchiveIdentity:
    """
    Work out one record's identity.

    A usable permalink is normalized the same way a saved-items URL is,
    so an archive record and a saved-items row pointing at the same post
    produce the same ``source_id`` and deduplicate against each other
    rather than existing twice.

    A record whose permalink is absent gets an archive-namespaced id. A
    record whose permalink is present but unusable also gets one, and is
    marked as having no usable permalink rather than as having had none:
    the difference is worth keeping because it is the difference between
    the archive losing information and the archive being wrong.
    """

    identifier = str(post_id or "").strip() or None

    activity = activity_id_of(post_id)

    raw = "" if permalink is None else str(permalink).strip()

    if not raw:
        return ArchiveIdentity(
            source_id=f"{ARCHIVE_NAMESPACE}{digest(identifier or text, ARCHIVE_NAMESPACE)}",
            canonical_url="",
            original_url="",
            kind=KIND_ARCHIVE_ONLY,
            identifier=identifier,
            has_permalink=False,
            activity_id=activity,
        )

    try:
        normalized = normalize_linkedin_url(raw)

    except SavedItemUrlError:
        # The archive recorded a permalink it could not use. Falling back
        # to the archive identity keeps the post rather than dropping it,
        # and the unusable value is preserved as the original so nothing
        # the archive said is lost.
        return ArchiveIdentity(
            source_id=f"{ARCHIVE_NAMESPACE}{digest(identifier or raw, ARCHIVE_NAMESPACE)}",
            canonical_url="",
            original_url=raw,
            kind=KIND_ARCHIVE_ONLY,
            identifier=identifier,
            has_permalink=False,
            activity_id=activity,
        )

    return ArchiveIdentity(
        source_id=source_id_for(normalized.canonical),
        canonical_url=normalized.canonical,
        original_url=normalized.original,
        kind=normalized.kind,
        identifier=identifier,
        has_permalink=True,
        activity_id=activity or normalized.identifier,
    )


__all__ = [
    "ARCHIVE_NAMESPACE",
    "CONTENT_NAMESPACE",
    "KIND_ARCHIVE_ONLY",
    "ArchiveIdentity",
    "activity_id_of",
    "content_fingerprint",
    "digest",
    "identify",
]
