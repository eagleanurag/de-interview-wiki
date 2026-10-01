"""
Read a Saved Items manifest.

The user exports their saved list, or writes it down, in whatever form
their tool produces. This reads CSV, TSV, plain text and JSON or JSONL
without asking them to reshape it first.

Two rules run through all of it:

* A row is understood only when its URL can be found. Nothing is
  invented for a row that cannot be read, and the problem is reported
  rather than guessed at.
* A record with a URL and nothing else is a real, useful record. It is
  kept as metadata-only rather than being dropped, because a saved link
  is something the user chose to keep.

No reader here fetches anything. Everything comes from a file.
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from src.ingestion.saved_items.model import (
    CaptureMethod,
    SavedItem,
)
from src.ingestion.saved_items.urls import (
    SavedItemUrlError,
    normalize_linkedin_url,
)


class ManifestError(ValueError):
    """Raised when a manifest cannot be read at all."""


@dataclass
class ReadIssue:
    """One thing the reader could not understand."""

    location: str
    message: str

    def __str__(self) -> str:
        return f"{self.location}: {self.message}"


@dataclass
class ManifestRead:
    """What a manifest yielded, and what it could not."""

    items: list[SavedItem] = field(default_factory=list)
    issues: list[ReadIssue] = field(default_factory=list)
    source_format: str = "unknown"

    def __len__(self) -> int:
        return len(self.items)


#: Header spellings accepted for each field. Matching is on a
#: normalized form, so ``Saved Date``, ``saved_date`` and ``SAVED
#: DATE`` are one header. The first spelling is the one used in
#: messages.
URL_HEADERS = (
    "url",
    "link",
    "saved url",
    "saved url link",
    "linkedin url",
    "linkedin link",
    "url link",
    "post url",
    "item url",
    "permalink",
)

DATE_HEADERS = (
    "saved date",
    "date saved",
    "saved on",
    "saved at",
    "date",
    "timestamp",
    "created",
    "created at",
)

TITLE_HEADERS = ("title", "name", "headline", "subject")

AUTHOR_HEADERS = ("author", "saved by", "owner", "poster", "creator")

NOTES_HEADERS = ("notes", "note", "comment", "annotation", "description")

ID_HEADERS = (
    "saved item id",
    "item id",
    "id",
    "urn",
    "source id",
    "source_post_id",
)

#: A column naming the capture bundle that belongs to a row, for a
#: saved list that points at the content it saved alongside itself.
BUNDLE_HEADERS = (
    "bundle",
    "bundle path",
    "capture",
    "capture path",
    "capture bundle",
    "content",
    "content path",
    "folder",
    "directory",
    "dir",
)

#: Fields that are never read into a Saved Item, whatever a header
#: claims. A saved list does not contain credentials, and a manifest
#: that does is not something to copy into the knowledge base.
FORBIDDEN_HEADERS = (
    "password",
    "passwd",
    "secret",
    "token",
    "cookie",
    "cookies",
    "session",
    "storage state",
    "storage_state",
    "access token",
    "auth header",
    "authorization",
    "api key",
    "api_key",
    "bearer",
)


def read_manifest(path: str | Path) -> ManifestRead:
    """Read a manifest, choosing the reader from its content."""

    target = Path(path)

    if not target.is_file():
        raise ManifestError(f"the manifest does not exist: {target}")

    suffix = target.suffix.lower()

    text = target.read_text(
        encoding="utf-8-sig", errors="replace"
    )

    if suffix == ".jsonl":
        return _read_jsonl(text, target)

    if suffix == ".json":
        return _read_json(text, target)

    if suffix == ".tsv":
        return _read_delimited(text, target, delimiter="\t")

    if suffix == ".csv":
        return _read_delimited(text, target, delimiter=",")

    if suffix == ".txt":
        return _read_plain(text, target)

    # An unknown extension is decided by content rather than refused, so
    # a user who exported without a suffix still gets their data in.
    if _looks_like_json(text):
        return _read_json(text, target)

    if _looks_like_delimited(text):
        return _read_delimited(text, target, delimiter=",")

    return _read_plain(text, target)


# ---------------------------------------------------------------------
# Delimited
# ---------------------------------------------------------------------


def _read_delimited(
    text: str,
    path: Path,
    *,
    delimiter: str,
) -> ManifestRead:
    result = ManifestRead(
        source_format="tsv" if delimiter == "\t" else "csv"
    )

    # A UTF-8 byte-order mark survives strip of the wrong field name
    # only on the first cell, so it is removed here too.
    text = text.lstrip("﻿")

    reader = csv.reader(io.StringIO(text), delimiter=delimiter)

    rows = list(reader)

    if not rows:
        return result

    header = rows[0]
    mapping = _map_header(header, result)

    for number, row in enumerate(rows[1:], start=2):
        if not any((cell or "").strip() for cell in row):
            continue

        fields = _row_to_fields(row, mapping)

        item = _item_from_fields(
            fields,
            f"{path.name} line {number}",
            result,
        )

        if item is not None:
            result.items.append(item)

    return result


def _map_header(
    header: list[str],
    result: ManifestRead,
) -> dict[str, str]:
    """
    Work out which column holds which field.

    A column whose header names a credential is skipped rather than
    read, and noted, so a manifest that was exported from somewhere
    unexpected does not quietly move a secret into the pipeline.
    """

    mapping: dict[str, str] = {}

    for index, cell in enumerate(header):
        label = _normalize_header(cell)

        if not label:
            continue

        if any(_compact(bad) in label for bad in FORBIDDEN_HEADERS):
            result.issues.append(
                ReadIssue(
                    location=f"column {index + 1}",
                    message=(
                        f"{str(cell).strip()!r} names a credential "
                        "and was not read"
                    ),
                )
            )
            continue

        for field_name, spellings in (
            ("url", URL_HEADERS),
            ("saved_date", DATE_HEADERS),
            ("title", TITLE_HEADERS),
            ("author", AUTHOR_HEADERS),
            ("notes", NOTES_HEADERS),
            ("source_id", ID_HEADERS),
            ("bundle", BUNDLE_HEADERS),
        ):
            if field_name in mapping:
                continue

            if label in {_compact(name) for name in spellings}:
                mapping[field_name] = str(index)
                break

    if "url" not in mapping:
        # A manifest with no URL column cannot describe a saved item,
        # so the file itself is reported rather than each row.
        result.issues.append(
            ReadIssue(
                location="header",
                message=(
                    "no URL column found; looked for "
                    + ", ".join(URL_HEADERS[:4])
                ),
            )
        )

    return mapping


def _row_to_fields(
    row: list[str],
    mapping: dict[str, str],
) -> dict[str, str]:
    fields: dict[str, str] = {}

    for name, index in mapping.items():
        try:
            value = row[int(index)]
        except (IndexError, ValueError):
            continue

        if value is None:
            continue

        cleaned = value.strip()

        if cleaned:
            fields[name] = cleaned

    return fields


# ---------------------------------------------------------------------
# JSON and JSONL
# ---------------------------------------------------------------------


def _read_json(text: str, path: Path) -> ManifestRead:
    result = ManifestRead(source_format="json")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ManifestError(
            f"{path.name} is not valid JSON: {exc}"
        ) from exc

    records: list[object]

    if isinstance(payload, list):
        records = payload

    elif isinstance(payload, dict):
        # An object wrapping the rows, which is how several exports are
        # shaped.
        for key in ("items", "saved", "savedItems", "data", "results"):
            nested = payload.get(key)

            if isinstance(nested, list):
                records = nested
                break

        else:
            records = [payload]

    else:
        raise ManifestError(
            f"{path.name} must contain an object or a list of objects"
        )

    for number, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            result.issues.append(
                ReadIssue(
                    location=f"{path.name} record {number}",
                    message="not an object",
                )
            )
            continue

        fields = _fields_from_object(record, result, number)

        item = _item_from_fields(
            fields, f"{path.name} record {number}", result
        )

        if item is not None:
            result.items.append(item)

    return result


def _read_jsonl(text: str, path: Path) -> ManifestRead:
    result = ManifestRead(source_format="jsonl")

    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()

        if not stripped:
            continue

        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as exc:
            result.issues.append(
                ReadIssue(
                    location=f"{path.name} line {number}",
                    message=f"not valid JSON ({exc.msg})",
                )
            )
            continue

        if not isinstance(record, dict):
            result.issues.append(
                ReadIssue(
                    location=f"{path.name} line {number}",
                    message="not an object",
                )
            )
            continue

        fields = _fields_from_object(record, result, number)

        item = _item_from_fields(
            fields, f"{path.name} line {number}", result
        )

        if item is not None:
            result.items.append(item)

    return result


def _fields_from_object(
    record: dict,
    result: ManifestRead,
    number: int,
) -> dict[str, str]:
    """
    Pull the known fields out of a record.

    Aliases are matched the same way a header row is, so a JSON export
    that uses ``savedDate`` or ``saved_date`` both work.
    """

    fields: dict[str, str] = {}

    normalized = {
        _compact(key): value for key, value in record.items()
    }

    for field_name, spellings in (
        ("url", URL_HEADERS),
        ("saved_date", DATE_HEADERS),
        ("title", TITLE_HEADERS),
        ("author", AUTHOR_HEADERS),
        ("notes", NOTES_HEADERS),
        ("source_id", ID_HEADERS),
        ("bundle", BUNDLE_HEADERS),
    ):
        if field_name in fields:
            continue

        for spelling in spellings:
            key = _compact(spelling)

            if key in normalized:
                value = normalized[key]

                if isinstance(value, (str, int, float)):
                    text = str(value).strip()

                    if text:
                        fields[field_name] = text

                break

    for label in FORBIDDEN_HEADERS:
        if any(_compact(label) in key for key in normalized):
            result.issues.append(
                ReadIssue(
                    location=f"record {number}",
                    message=(
                        f"a {label!r} field was present and was not read"
                    ),
                )
            )

    return fields


# ---------------------------------------------------------------------
# Plain text
# ---------------------------------------------------------------------

_URL_IN_TEXT = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)


def _read_plain(text: str, path: Path) -> ManifestRead:
    """
    Read a plain text list of links.

    One URL per line, with everything after the URL on the same line
    treated as a note, because that is how a list pasted into a text
    file usually reads. A line with no URL is reported and skipped.
    """

    result = ManifestRead(source_format="txt")

    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()

        if not stripped or stripped.startswith("#"):
            continue

        match = _URL_IN_TEXT.search(stripped)

        if not match:
            result.issues.append(
                ReadIssue(
                    location=f"{path.name} line {number}",
                    message="no URL on this line",
                )
            )
            continue

        url = match.group(0)

        # Trailing punctuation is part of the sentence, not the URL.
        url = url.rstrip(".,;)]}'\"")

        fields = {"url": url}

        remainder = stripped[match.end() :].strip(" \t-—:|")

        if remainder:
            fields["notes"] = remainder

        item = _item_from_fields(
            fields, f"{path.name} line {number}", result
        )

        if item is not None:
            result.items.append(item)

    return result


# ---------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------


def _item_from_fields(
    fields: dict[str, str],
    location: str,
    result: ManifestRead,
) -> SavedItem | None:
    """Build one Saved Item, reporting anything unusable."""

    raw_url = fields.get("url")

    if not raw_url:
        result.issues.append(
            ReadIssue(
                location=location,
                message="no URL, so the row does not describe a saved item",
            )
        )
        return None

    try:
        normalized = normalize_linkedin_url(raw_url)
    except SavedItemUrlError as exc:
        result.issues.append(
            ReadIssue(location=location, message=str(exc))
        )
        return None

    item = SavedItem.from_url(normalized)

    item.saved_date = fields.get("saved_date")
    item.title = fields.get("title")
    item.author = fields.get("author")
    item.notes = fields.get("notes")
    item.bundle = fields.get("bundle")

    return item


def _normalize_header(value: object) -> str:
    """
    Reduce a header or key to a comparison form.

    Case, spacing and the choice between ``_`` and ``-`` are all
    differences an export tool makes freely, and none of them changes
    which column a field is in.
    """

    return _compact(value)


def _compact(value: object) -> str:
    """
    Strip a header or key down to its letters and digits.

    ``saved date``, ``saved_date``, ``saved-date`` and ``savedDate``
    are one header. Only the letters are kept, so no separator an
    export tool happens to choose can change which column a field is
    in. Word boundaries are not recoverable from a JSON key, so this
    deliberately over-matches rather than miss a real column.
    """

    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _looks_like_json(text: str) -> bool:
    stripped = text.lstrip().lstrip("﻿")

    return stripped.startswith(("{", "["))


def _looks_like_delimited(text: str) -> bool:
    first = text.splitlines()[0] if text.splitlines() else ""

    return (
        "," in first
        or "\t" in first
        and not first.strip().startswith("http")
    )


def iter_manifest(paths: list[str | Path]) -> Iterator[ManifestRead]:
    """Read several manifests in order."""

    for entry in paths:
        yield read_manifest(entry)