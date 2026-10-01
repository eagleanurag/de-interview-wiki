"""
The capture file contract.

One folder, one saved item, one small JSON file saying which. It exists
so a capture can be dropped in without the user having to work out a
naming scheme, and it is deliberately minimal: the URL is the only
required field, because it is the only thing that cannot be worked out
from the rest of the folder.

Nothing here is required beyond the URL. Every other field is something
the capture may or may not know, and a missing one is filled in from
the manifest or left absent rather than guessed at.

Credential-shaped fields are refused and reported, never stored. A
capture is a folder of notes; there is nothing in it that should ever
hold a password, and a file that does is not something to copy into a
committed knowledge base.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from src.ingestion.saved_items.urls import (
    SavedItemUrlError,
    normalize_linkedin_url,
)


#: Files that name which saved item a capture belongs to, in precedence
#: order.
CAPTURE_NAMES = ("capture.json", "saved-item.json", "post.json")

#: Keys that may carry the link, in precedence order. An alias is
#: accepted because the file is written by hand and by export tools, and
#: refusing a spelling would only teach the user the wrong one.
CAPTURE_URL_KEYS = (
    "url",
    "canonical_url",
    "canonicalUrl",
    "link",
    "permalink",
    "source_url",
)

#: Keys that may carry the identifier.
CAPTURE_ID_KEYS = (
    "source_id",
    "sourceId",
    "id",
    "urn",
    "saved_item_id",
)

#: Keys that may carry the captured text.
CAPTURE_TEXT_KEYS = (
    "text",
    "content",
    "body",
    "original_text",
    "post_text",
)

CAPTURE_TITLE_KEYS = ("title", "headline", "name")
CAPTURE_AUTHOR_KEYS = ("author", "authorName", "author_name", "byline")
CAPTURE_NOTES_KEYS = ("notes", "note", "annotation")
CAPTURE_DATE_KEYS = ("captured_at", "capturedAt", "published_at", "date")

#: Keys that may say how the content was captured. Recorded, never
#: trusted for anything but reporting: the pipeline decides what actually
#: happened by looking at what the folder contains.
CAPTURE_METHOD_KEYS = ("capture_method", "captureMethod")

#: Field names that are never read, whatever a capture claims.
#:
#: Matched on the normalized name, so ``access_token``, ``Access Token``
#: and ``accessToken`` are all caught. Refused rather than ignored, so
#: the user is told their file holds something it should not.
CREDENTIAL_FIELDS = frozenset(
    {
        "password",
        "passwd",
        "pass",
        "secret",
        "token",
        "accesstoken",
        "refreshtoken",
        "bearer",
        "cookie",
        "cookies",
        "setcookie",
        "session",
        "sessionid",
        "sessionkey",
        "storagestate",
        "authorization",
        "auth",
        "apikey",
        "apisecret",
        "clientsecret",
        "privatekey",
        "csrftoken",
    }
)

#: Suffixes that are never capture files, whatever they are named.
_NOT_CAPTURE_SUFFIXES = frozenset({".json.tmp", ".tmp", ".bak"})


class CaptureError(ValueError):
    """Raised when a capture file exists but cannot be used."""


def normalize_field(value: object) -> str:
    """
    Reduce a field name to a comparison form.

    Case, spacing and the choice between ``_`` and ``-`` are all
    differences a person typing by hand or an export tool makes freely,
    and none of them changes what the field is.
    """
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def credential_fields(payload: dict) -> list[str]:
    """Field names in a capture that name a credential."""
    return sorted(
        str(key)
        for key in payload
        if normalize_field(key) in CREDENTIAL_FIELDS
    )


@dataclass
class Capture:
    """
    One capture file, read.

    ``claims`` is what the file says it belongs to; ``fields`` is what
    it declares beyond that. A capture with only a URL is complete and
    usable. A capture with a credential field is still readable, but the
    field is dropped and reported rather than kept.
    """

    url: str | None = None
    source_id: str | None = None
    text: str = ""
    title: str | None = None
    author: str | None = None
    notes: str | None = None
    captured_at: str | None = None
    published_at: str | None = None
    capture_method: str | None = None

    rejected_fields: list[str] = field(default_factory=list)
    unreadable_reason: str = ""
    path: Path | None = None

    @property
    def is_empty(self) -> bool:
        return not any(
            (
                self.url,
                self.source_id,
                self.text.strip(),
                self.title,
                self.author,
                self.notes,
            )
        )

    def as_dict(self) -> dict:
        """The declared fields, with anything refused left out."""
        return {
            key: value
            for key, value in {
                "url": self.url,
                "source_id": self.source_id,
                "title": self.title,
                "author": self.author,
                "notes": self.notes,
                "captured_at": self.captured_at,
                "published_at": self.published_at,
                "capture_method": self.capture_method,
            }.items()
            if value
        }


def first_string(payload: dict, keys: tuple[str, ...]) -> str | None:
    """The first non-empty string among several accepted key names."""
    for key in keys:
        value = payload.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

        # A date is often written unquoted in JSONL.
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)

    return None


def read_capture_file(path: str | Path) -> Capture:
    """
    Read one capture file.

    A file that is not an object, or is not valid JSON, is reported
    rather than guessed at. A URL that does not normalize is dropped
    rather than stored, because a link that cannot be resolved cannot
    identify a saved item and keeping it would only produce a confusing
    failure later.
    """
    target = Path(path)

    capture = Capture(path=target)

    if target.name.lower().endswith(tuple(_NOT_CAPTURE_SUFFIXES)):
        capture.unreadable_reason = (
            "this looks like a temporary file, not a capture"
        )

        return capture

    try:
        raw = target.read_text(encoding="utf-8-sig", errors="replace")

    except OSError as exc:
        capture.unreadable_reason = f"{type(exc).__name__}: {exc}"

        return capture

    try:
        payload = json.loads(raw)

    except json.JSONDecodeError as exc:
        capture.unreadable_reason = (
            f"not valid JSON ({exc.msg} at line {exc.lineno})"
        )

        return capture

    if not isinstance(payload, dict):
        capture.unreadable_reason = (
            f"expected a JSON object, found {type(payload).__name__}"
        )

        return capture

    capture.rejected_fields = credential_fields(payload)

    raw_url = first_string(payload, CAPTURE_URL_KEYS)

    if raw_url:
        try:
            capture.url = normalize_linkedin_url(raw_url).canonical

        except SavedItemUrlError as exc:
            # Named, not stored. The capture is still read for its text
            # and media; it just cannot be matched by this link.
            capture.rejected_fields.append(f"url: {exc}")

    capture.source_id = first_string(payload, CAPTURE_ID_KEYS)
    capture.title = first_string(payload, CAPTURE_TITLE_KEYS)
    capture.author = first_string(payload, CAPTURE_AUTHOR_KEYS)
    capture.notes = first_string(payload, CAPTURE_NOTES_KEYS)
    capture.captured_at = first_string(payload, CAPTURE_DATE_KEYS)
    capture.capture_method = first_string(
        payload, CAPTURE_METHOD_KEYS
    )
    capture.text = first_string(payload, CAPTURE_TEXT_KEYS) or ""

    if not capture.captured_at:
        capture.published_at = None

    return capture


def find_capture_file(directory: str | Path) -> Path | None:
    """The capture file in a directory, if it has one."""
    base = Path(directory)

    if base.is_file() and base.suffix.lower() in {".json", ".jsonl"}:
        return base

    for name in CAPTURE_NAMES:
        candidate = base / name

        if candidate.is_file():
            return candidate

    return None


def capture_json_example(
    url: str = "https://www.linkedin.com/posts/...",
) -> str:
    """
    A capture file the user can copy, containing nothing they must not.

    One field, because one field is all that is required. A longer
    example would imply the rest is expected, and a user copying a
    template full of empty fields would fill in a title they do not have
    rather than leaving it out.
    """
    return json.dumps({"url": url}, indent=2)


__all__ = [
    "CAPTURE_AUTHOR_KEYS",
    "CAPTURE_DATE_KEYS",
    "CAPTURE_ID_KEYS",
    "CAPTURE_NAMES",
    "CAPTURE_NOTES_KEYS",
    "CAPTURE_TEXT_KEYS",
    "CAPTURE_TITLE_KEYS",
    "CAPTURE_URL_KEYS",
    "CREDENTIAL_FIELDS",
    "Capture",
    "CaptureError",
    "capture_json_example",
    "credential_fields",
    "find_capture_file",
    "first_string",
    "normalize_field",
    "read_capture_file",
]
