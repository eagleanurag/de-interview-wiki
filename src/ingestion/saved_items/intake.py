"""
Build or update a Saved Items list from simple input.

A user who exported their saved posts has a list. A user who never did
has a text file and a browser, and what they have is rarely a CSV with
the right headers. This turns either into the one list the importer
reads, without asking them to reshape it first.

Three inputs are recognised, and only because their structure is
unambiguous:

* plain text, one link per line, with anything after the link kept as a
  note;
* a comma or tab separated file, whose columns are matched by name;
* a bookmark export, which is either HTML with links in it or JSON in
  the shape browsers write.

A fourth, a PDF of saved posts, is **not** recognised. It is a document
whose links are not reliably extractable, and a wrong guess at which
line was which post would attach the wrong body to the right link. It is
reported as unsupported rather than half-read.

What this never does: overwrite a capture, or touch an enrichment. It
writes the list, and nothing else. The list is the user's own record of
what they saved, so an existing one is merged into rather than replaced,
and a link that is already there keeps whatever the earlier export said
about it.
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

from src.ingestion.saved_items.model import SavedItem
from src.ingestion.saved_items.readers import (
    ManifestError,
    ManifestRead,
    read_manifest,
)
from src.ingestion.saved_items.urls import (
    SavedItemUrlError,
    normalize_linkedin_url,
)


#: What the command did, for the summary it prints.
ADDED = "added"
UPDATED = "updated"
KEPT = "kept"
SKIPPED = "skipped"


@dataclass
class MergeResult:
    """
    What merging a list into an existing one did.

    Counts rather than a list of changes, because the interesting
    question is how many links are new, not which ones.
    """

    added: int = 0
    updated: int = 0
    kept: int = 0
    skipped: int = 0
    issues: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.added + self.updated + self.kept + self.skipped

    def line(self) -> str:
        return (
            f"added {self.added}, updated {self.updated}, "
            f"kept {self.kept}, skipped {self.skipped}"
        )

    def as_dict(self) -> dict:
        return {
            "added": self.added,
            "updated": self.updated,
            "kept": self.kept,
            "skipped": self.skipped,
            "issues": list(self.issues),
        }


#: Formats a user might point this at, and whether the format is
#: recognised. A file whose format is not recognised is named as such
#: rather than read hopefully.
SUPPORTED_SUFFIXES = (".txt", ".csv", ".tsv", ".json", ".jsonl", ".html", ".htm")

#: Named so the refusal can say what to do instead.
UNSUPPORTED_MESSAGE = (
    "A PDF of saved posts is not read. Which line of a paginated "
    "document belongs to which post cannot be established reliably, and "
    "attaching the wrong text to the right link is worse than not "
    "importing. Export the links, or copy them into a .txt file, one "
    "per line."
)


def merge_into(existing: list[SavedItem], incoming: list[SavedItem]) -> MergeResult:
    """
    Fold a freshly read list into the one on file.

    A link already present keeps what it had, because the earlier record
    usually came from an export that carried more, or from a run that
    read a capture. A link whose metadata is empty here and filled in
    the new list is filled, because that is the user's own data being
    recovered rather than overwritten.
    """
    result = MergeResult()

    by_url = {item.canonical_url: item for item in existing}

    for item in incoming:
        known = by_url.get(item.canonical_url)

        if known is None:
            existing.append(item)
            by_url[item.canonical_url] = item
            result.added += 1
            continue

        filled = False

        for name in ("saved_date", "title", "author", "notes", "bundle"):
            if not getattr(known, name) and getattr(item, name):
                setattr(known, name, getattr(item, name))
                filled = True

        if filled:
            result.updated += 1
        else:
            result.kept += 1

    return result


# ---------------------------------------------------------------------
# Bookmark exports
# ---------------------------------------------------------------------


_LINK_IN_HTML = re.compile(
    r"""<a\b[^>]*?href\s*=\s*["']([^"']+)["'][^>]*>(.*?)</a>""",
    re.IGNORECASE | re.DOTALL,
)

_TAG = re.compile(r"<[^>]+>")

#: Hosts a bookmark export is expected to contain. A file full of other
#: links is not a saved-posts export, and reading it as one would put
#: unrelated pages into the knowledge base.
_EXPECTED_HOST = "linkedin.com"


class _LinkCollector(HTMLParser):
    """
    The links in a page, with the text that labelled them.

    A bookmark file is mostly whitespace between tags, so the label is
    whatever was collected between one anchor and the next. Reading the
    text after an anchor as the label of the anchor before it would put
    every title one link too early.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)

        self.links: list[tuple[str, str]] = []

        self._open: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag != "a":
            return

        attributes = {
            key.lower(): (value or "") for key, value in attrs
        }

        self._flush()

        href = attributes.get("href", "").strip()

        if href:
            self._open = href

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._open is not None and data.strip():
            self._text.append(data.strip())

    def _flush(self) -> None:
        if self._open is None:
            return

        href = self._open
        label = re.sub(r"\s+", " ", " ".join(self._text)).strip()

        self._open = None
        self._text = []

        self.links.append((href, label))

    def close(self) -> None:
        self._flush()

        super().close()


def read_bookmarks(path: Path) -> ManifestRead:
    """
    Read a browser bookmark export.

    Only the links are taken, and only when the file is a page of
    bookmarks rather than an arbitrary page: a file with no LinkedIn
    links in it is reported instead of imported, because an export the
    user pointed at by mistake should not become a knowledge base.
    """
    result = ManifestRead(source_format="bookmarks")

    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")

    except OSError as exc:
        raise ManifestError(f"{path.name} could not be read: {exc}") from exc

    collector = _LinkCollector()

    try:
        collector.feed(text)
        collector.close()

    except Exception:  # noqa: BLE001
        # A bookmark file truncated by a crashed export still holds
        # whatever links it got before it stopped.
        pass

    if not collector.links:
        # Fall back to a plain scan, for an export whose anchors are not
        # well formed.
        collector.links = [
            (href, "")
            for href, _ in _LINK_IN_HTML.findall(text)
        ]

    seen: set[str] = set()

    for href, label in collector.links:
        cleaned = href.strip()

        if not cleaned or cleaned in seen:
            continue

        seen.add(cleaned)

        if _EXPECTED_HOST not in cleaned.lower():
            continue

        try:
            item = SavedItem.from_url(normalize_linkedin_url(cleaned))

        except SavedItemUrlError as exc:
            result.issues.append(
                _issue(str(path), cleaned, str(exc))
            )
            continue

        if label and not item.title:
            item.title = _TAG.sub("", label).strip()[:200] or None

        result.items.append(item)

    if not result.items:
        result.issues.append(
            _issue(
                str(path),
                "no usable LinkedIn links",
                "The file was read as a bookmark export and no LinkedIn "
                "post links were found in it.",
            )
        )

    return result


def _issue(location: str, subject: str, message: str):
    from src.ingestion.saved_items.readers import ReadIssue

    return ReadIssue(location=f"{location}: {subject}", message=message)


# ---------------------------------------------------------------------
# Writing the list
# ---------------------------------------------------------------------

#: The columns written, in a fixed order so two runs produce the same
#: file and a diff means something changed.
COLUMNS = ("URL", "Saved Date", "Title", "Author", "Notes", "Bundle")


def render_csv(items: list[SavedItem]) -> str:
    """
    Write the list as CSV.

    The canonical URL is written, not the original. A list that kept
    whatever the user typed would hold four spellings of the same post
    and every later run would have to normalize them again. The original
    is not lost: it is in the manifest, per item.
    """
    buffer = io.StringIO()

    writer = csv.writer(buffer, lineterminator="\n")

    writer.writerow(COLUMNS)

    for item in sorted(items, key=lambda entry: entry.canonical_url):
        writer.writerow(
            [
                item.canonical_url,
                item.saved_date or "",
                item.title or "",
                item.author or "",
                item.notes or "",
                item.bundle or "",
            ]
        )

    return buffer.getvalue()


def write_list(
    path: str | Path,
    items: list[SavedItem],
) -> Path:
    """
    Write the list, atomically.

    Atomically because this is the file the whole workflow hangs off. A
    list truncated by an interrupted write would send the next run
    importing items that are already imported, or dropping the ones it
    cannot see.
    """
    from src.ingestion.collect import replace_file

    target = Path(path)

    target.parent.mkdir(parents=True, exist_ok=True)

    temporary = target.with_suffix(target.suffix + ".tmp")

    temporary.write_text(render_csv(items), encoding="utf-8")

    replace_file(temporary, target)

    return target


# ---------------------------------------------------------------------
# Reading whatever the user pointed at
# ---------------------------------------------------------------------


def read_input(path: str | Path) -> ManifestRead:
    """
    Read a list, a bookmark export, or a file of links.

    Chooses by extension where the extension is meaningful and by
    content where it is not, so a file called ``saved.dat`` holding
    bookmarks is still recognised.
    """
    target = Path(path)

    if not target.is_file():
        raise ManifestError(f"the file does not exist: {target}")

    suffix = target.suffix.lower()

    if suffix == ".pdf":
        raise ManifestError(UNSUPPORTED_MESSAGE)

    if suffix in {".html", ".htm"}:
        return read_bookmarks(target)

    if suffix in {".txt", ".csv", ".tsv", ".json", ".jsonl"}:
        return read_manifest(target)

    # No useful extension, or an unfamiliar one. Read the head and
    # decide from what is actually in the file.
    try:
        head = target.read_text(
            encoding="utf-8-sig", errors="replace"
        )[:4096]

    except OSError as exc:
        raise ManifestError(f"{target.name} could not be read: {exc}") from exc

    lowered = head.lstrip().lower()

    if lowered.startswith(("{", "[")):
        return read_manifest(target)

    if "<a " in lowered or "<!doctype html" in lowered or "<html" in lowered:
        return read_bookmarks(target)

    return read_manifest(target)


__all__ = [
    "ADDED",
    "COLUMNS",
    "KEPT",
    "MergeResult",
    "SKIPPED",
    "UPDATED",
    "merge_into",
    "read_bookmarks",
    "read_input",
    "render_csv",
    "write_list",
]
