"""
Content bundles for Saved Items.

A saved list tells the pipeline what the user chose to keep. It does not
contain the posts themselves, so most items arrive as a URL and a date
and nothing more. That is a true statement about the data, and this
module keeps it true: an item stays metadata-only until content is
actually supplied beside it.

When the user does supply content, it arrives as a directory:

.. code-block:: text

    data/incoming/saved-items/
        manifest.csv
        urn-li-saved-1a2b3c4d5e6f7a8b/
            capture.json
            content.md
            screenshot.png
            document.pdf

A bundle is tied to an item three ways, and the first that applies
wins: by the item's identifier as a directory name, by the source id,
or by a ``url`` inside ``capture.json`` that normalizes to the item's
canonical URL. A manifest may also name the bundle directly in a
column, which is how a saved list that points at its own captures is
read.

Nothing here reaches the network. A bundle is a directory the user
filled in, and the only thing read from it is what is actually in it.
A corrupt file, an empty page and a screenshot with no text are all
reported as what they are rather than described as successful
captures.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

from src.ingestion.saved_items import capture as _capture
from src.ingestion.saved_items.model import SavedItem
from src.ingestion.saved_items.urls import (
    SavedItemUrlError,
    normalize_linkedin_url,
)
from src.ingestion.sources.manual import (
    DOCUMENT_SUFFIXES,
    IMAGE_SUFFIXES,
    MEDIA_SUFFIXES,
    TEXT_SUFFIXES,
)


#: HTML pages the user saved from a browser.
HTML_SUFFIXES = frozenset({".html", ".htm"})

#: Directories that hold bundles rather than being one.
#:
#: A capture inbox is a folder of capture folders, and reading the
#: folder as a single bundle would merge every saved post into one post
#: and invent a document the user never wrote. Naming the container keeps
#: the directory-first rule intact: the inbox is a container, and the
#: folders inside it are the bundles.
CONTAINER_NAMES = frozenset({"captures", "capture", "inbox", "bundles"})

#: Every format a bundle may hold.
BUNDLE_SUFFIXES = (
    TEXT_SUFFIXES | HTML_SUFFIXES | MEDIA_SUFFIXES | frozenset({".json", ".jsonl"})
)

#: A JSON file inside a bundle, which is how a bundle says which item it
#: belongs to and may also carry the text itself. The names and the
#: shape are defined once, in the capture contract, and used from there
#: so the reader, the matcher and the diagnostics cannot disagree.
CAPTURE_NAMES = _capture.CAPTURE_NAMES

#: Files that are never part of a capture, whatever they are named.
IGNORED_SUFFIXES = frozenset(
    {
        ".tmp",
        ".bak",
        ".swp",
        ".log",
        ".py",
        ".yaml",
        ".yml",
        ".toml",
        ".lock",
        ".partial",
        ".part",
        ".crdownload",
    }
)

#: Tags whose text is structure or script, not content.
_DROPPED_TAGS = frozenset(
    {"script", "style", "noscript", "svg", "template", "head"}
)


class BundleError(ValueError):
    """Raised when a bundle cannot be used and the caller must say so."""


#: How complete a capture actually is.
#:
#: Every value here is something the pipeline observed rather than a
#: score it invented. There is deliberately no confidence number: a
#: figure with no basis behind it is worse than naming the situation,
#: because a reader has no way to tell a measured 0.9 from a guessed
#: one.
QUALITY_METADATA_ONLY = "metadata_only"
QUALITY_IMAGE_ONLY = "image_only"
QUALITY_DOCUMENT_ONLY = "document_only"
QUALITY_PARTIAL = "partial"
QUALITY_TEXT = "text"
QUALITY_TEXT_AND_MEDIA = "text_and_media"

#: Every value, for validating one that arrives from elsewhere.
CAPTURE_QUALITIES = (
    QUALITY_METADATA_ONLY,
    QUALITY_IMAGE_ONLY,
    QUALITY_DOCUMENT_ONLY,
    QUALITY_PARTIAL,
    QUALITY_TEXT,
    QUALITY_TEXT_AND_MEDIA,
)


@dataclass
class CapturedContent:
    """
    What was actually found for one Saved Item.

    ``text`` is empty when the bundle held no readable text, which is
    the normal case for a bundle of screenshots. ``media`` lists what
    came with it. ``notes`` are what could not be read rather than what
    was wrong with the item, so a partial capture is still a capture.
    """

    text: str = ""
    media: list[Path] = field(default_factory=list)
    title: str | None = None
    author: str | None = None
    published_at: str | None = None
    notes: list[str] = field(default_factory=list)
    unreadable: list[str] = field(default_factory=list)

    @property
    def has_text(self) -> bool:
        return bool(self.text.strip())

    @property
    def has_media(self) -> bool:
        return bool(self.media)

    @property
    def has_anything(self) -> bool:
        return self.has_text or self.has_media

    def describe(self) -> str:
        """One line naming what was found, for the report."""

        parts: list[str] = []

        if self.has_text:
            parts.append(f"text ({len(self.text.strip())} chars)")

        if self.media:
            parts.append(f"{len(self.media)} media file(s)")

        if self.unreadable:
            parts.append(f"{len(self.unreadable)} unreadable")

        return ", ".join(parts) if parts else "no content"

    def quality(self) -> str:
        """
        How complete this capture is.

        Read off what is actually in it, so a reader can tell a
        transcript from a link from a screenshot from a document. A
        capture that carries text but also has a file that would not
        open is ``partial`` rather than ``text``, because the difference
        between "this is all there is" and "this is all that could be
        read" is the one a reader needs.
        """

        if not self.has_anything:
            return QUALITY_METADATA_ONLY

        if not self.has_text:
            documents = [
                path
                for path in self.media
                if path.suffix.lower() in DOCUMENT_SUFFIXES
            ]

            if len(documents) == len(self.media):
                return QUALITY_DOCUMENT_ONLY

            return QUALITY_IMAGE_ONLY

        if self.unreadable:
            return QUALITY_PARTIAL

        if self.media:
            return QUALITY_TEXT_AND_MEDIA

        return QUALITY_TEXT


def directory_names(item: SavedItem) -> tuple[str, ...]:
    """
    The names a bundle directory may use for this item.

    A source id contains colons, which a directory name cannot hold, so
    the punctuation-free form is accepted too. The short digest is
    accepted as well, because that is what a person copying an id from
    a report would type.
    """

    names: list[str] = []

    if item.source_id:
        names.append(item.source_id)
        names.append(_safe_name(item.source_id))

        digest = item.source_id.rsplit(":", 1)[-1]

        if digest:
            names.append(digest)

    if item.identifier:
        names.append(_safe_name(item.identifier))

    if item.post_id:
        names.append(item.post_id)
        names.append(_safe_name(item.post_id))

    ordered: list[str] = []

    for name in names:
        cleaned = name.strip()

        if cleaned and cleaned not in ordered:
            ordered.append(cleaned)

    return tuple(ordered)


_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_name(value: str) -> str:
    """A filesystem-safe form of an identifier."""

    return _SAFE_NAME.sub("-", value).strip("-")


# ---------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------


def discover_bundles(root: str | Path) -> list[Path]:
    """
    Every bundle directory beneath a root.

    Directory-first, like the manual source: the root holds bundles and
    a directory is a bundle. A single file at the root is a bundle
    too, so a user who drops one saved page in still gets it read.

    A directory named as a container is descended into rather than read
    as a bundle, so a capture inbox is a container of captures and not
    one enormous capture.

    Anything that resolves outside the root is left out entirely rather
    than returned and refused later. A capture is user-supplied content
    and must not be able to name a file anywhere on the machine, and the
    cheapest place to guarantee that is before the path is ever read.

    Returns paths relative to the root, so a discovered bundle can be
    handed straight back to :func:`read_bundle` with the same root and
    resolve to the same place.
    """
    base = Path(root)

    if not base.is_dir():
        raise BundleError(f"the bundle root does not exist: {base}")

    bundles: list[Path] = []

    def consider(directory: Path) -> None:
        """Add the bundles in a directory, descending into containers."""
        for entry in sorted(directory.iterdir(), key=lambda item: item.name):
            if entry.name.startswith("."):
                continue

            if not entry.is_dir():
                if (
                    entry.suffix.lower() in BUNDLE_SUFFIXES
                    and not _is_state_file(entry)
                ):
                    bundles.append(entry.relative_to(base))

                continue

            if not _within(entry, base):
                # A link out of the drop zone. Not listed, not read, and
                # reported by the index as a problem rather than being
                # silently absent.
                continue

            if entry.name.lower() in CONTAINER_NAMES:
                # A container, not a bundle. Depth is bounded so a
                # folder nested inside itself cannot loop forever.
                if len(entry.relative_to(base).parts) < _MAX_DEPTH:
                    consider(entry)

                continue

            if _has_content(entry):
                bundles.append(entry.relative_to(base))

    consider(base)

    return bundles


def discover_rejected(root: str | Path) -> list[Path]:
    """
    Entries directly inside a root that resolve outside it.

    Reported rather than ignored, so a link the user made is visible as
    a problem instead of appearing to be a capture that was never found.
    """
    base = Path(root)

    if not base.is_dir():
        return []

    rejected: list[Path] = []

    for entry in sorted(base.iterdir(), key=lambda item: item.name):
        if entry.name.startswith("."):
            continue

        if entry.is_dir() and not _within(entry, base):
            rejected.append(entry)

        elif entry.is_file() and entry.is_symlink():
            if not _within(entry, base):
                rejected.append(entry)

    return rejected


def _has_content(directory: Path) -> bool:
    """Whether a directory holds anything a bundle may hold."""

    for entry in directory.rglob("*"):
        if not entry.is_file():
            continue

        if entry.name.startswith("."):
            continue

        if entry.suffix.lower() in BUNDLE_SUFFIXES:
            return True

    return False


#: How deep a container may nest. Bounded so a loop of directories
#: cannot make discovery unbounded, and generous enough for an inbox
#: inside a dated folder.
_MAX_DEPTH = 4

#: Files that are this project's own state rather than anything the user
#: captured. Read as a bundle they would be attached to whichever item
#: happened to sort first, which is how a record of saved items ends up
#: quoted as the body of a LinkedIn post.
STATE_FILES = frozenset(
    {
        "saved-items-manifest.json",
        "saved-items-manifest.json.tmp",
        "saved-items-state.json",
    }
)


def _is_state_file(path: Path) -> bool:
    """Whether a file is pipeline state rather than a capture."""
    return path.name.lower() in STATE_FILES


def _association_file(resolved: Path) -> Path | None:
    """The capture file that claims an item, when there is one."""

    if resolved.is_file() and resolved.suffix.lower() in {
        ".json",
        ".jsonl",
    }:
        return resolved

    if not resolved.is_dir():
        return None

    for name in CAPTURE_NAMES:
        candidate = resolved / name

        if candidate.is_file() and _within(candidate, resolved):
            return candidate

    return None


def read_bundle(path: str | Path, *, root: str | Path) -> CapturedContent:
    """
    Read one bundle, refusing anything outside the root.

    The containment check is on the resolved path, so a symlink
    pointing out of the drop zone is refused the same way a ``..`` in a
    name is. A user-supplied directory is still not trusted to stay
    inside the directory the user pointed at.

    The same check is applied to every file inside the bundle, because
    a bundle that is legitimately placed can still hold a link that
    reaches out of the drop zone, and a capture must not be able to
    name a file anywhere on the machine.
    """
    base = Path(root).resolve()

    target = Path(path)

    resolved = target.resolve() if target.is_absolute() else (base / target).resolve()

    if not _within(resolved, base):
        raise BundleError(
            f"the bundle escapes the bundle root: {path}"
        )

    if not resolved.exists():
        raise BundleError(f"the bundle does not exist: {path}")

    if resolved.is_file():
        return _from_single_file(resolved)

    return _from_directory(resolved, base=resolved)


def _within(path: Path, base: Path) -> bool:
    """
    Whether a resolved path really is inside a directory.

    Both sides are compared as resolved paths, so a symlink is judged by
    where it lands rather than by where it is written.
    """
    try:
        resolved = path.resolve()
        anchor = base.resolve()

    except OSError:
        return False

    return resolved == anchor or anchor in resolved.parents


def _from_single_file(path: Path) -> CapturedContent:
    content = CapturedContent()

    _absorb_file(path, content)

    return content


def _from_directory(directory: Path, *, base: Path) -> CapturedContent:
    content = CapturedContent()

    structured: dict | None = None

    for name in CAPTURE_NAMES:
        candidate = directory / name

        if candidate.is_file() and _within(candidate, directory):
            structured = _read_json(candidate, content)

            if structured is not None:
                break

    for entry in sorted(directory.rglob("*"), key=lambda item: str(item)):
        if not entry.is_file():
            continue

        if entry.name.startswith("."):
            continue

        if entry.name in CAPTURE_NAMES:
            continue

        if entry.suffix.lower() in IGNORED_SUFFIXES:
            continue

        # A file written inside the bundle can still be a link to
        # somewhere else entirely. It is reported and skipped, so the
        # capture is smaller than the directory appears to be rather
        # than silently reading outside the drop zone.
        if not _within(entry, base):
            content.unreadable.append(
                f"{entry.name}: resolves outside the bundle and was not read"
            )
            continue

        _absorb_file(entry, content)

    if structured is not None:
        _apply_structured(structured, content)

    if not content.has_anything and not content.notes:
        content.notes.append(
            "the bundle held no readable text or media"
        )

    return content


def _absorb_file(path: Path, content: CapturedContent) -> None:
    """Fold one file into the content, isolating its failures."""

    suffix = path.suffix.lower()

    if suffix in MEDIA_SUFFIXES:
        if _media_is_readable(path, content):
            content.media.append(path)

        return

    if suffix in HTML_SUFFIXES:
        _absorb_html(path, content)

        return

    if suffix == ".jsonl":
        _absorb_jsonl(path, content)

        return

    if suffix in TEXT_SUFFIXES:
        text = _read_text(path, content)

        if text.strip():
            content.text = _join(content.text, text)

        return

    # A ``.json`` that is not a capture name is still a file the user
    # put here, so its text is read rather than dropped. Structurally
    # invalid JSON is reported.
    if suffix == ".json":
        text = _read_text(path, content)

        if text.strip():
            content.text = _join(content.text, text)

        return


def _media_is_readable(path: Path, content: CapturedContent) -> bool:
    """
    Whether a media file is one the pipeline can actually use.

    A corrupt file is reported and skipped, so it still ends up in the
    post as a preserved file but is not claimed to have been read. An
    image with no text is kept as an image: this project does not do
    OCR, and saying otherwise would be inventing content.
    """
    try:
        size = path.stat().st_size
    except OSError as exc:
        content.unreadable.append(f"{path.name}: {_brief(exc)}")
        return False

    if size == 0:
        content.unreadable.append(f"{path.name}: the file is empty")
        return False

    suffix = path.suffix.lower()

    if suffix in DOCUMENT_SUFFIXES:
        if not _pdf_is_readable(path, content):
            return False

    elif suffix in IMAGE_SUFFIXES and not _image_is_readable(path, content):
        return False

    return True


def _pdf_is_readable(path: Path, content: CapturedContent) -> bool:
    """Whether a PDF opens, and how much text it actually yields."""

    from src.processing.pdf_processor import extract_pdf_text

    try:
        text = extract_pdf_text(path)

    except Exception as exc:  # noqa: BLE001
        # A PDF the extractor cannot open is preserved but not read.
        # The file is still the user's material and is not discarded.
        content.unreadable.append(
            f"{path.name}: the PDF could not be opened ({_brief(exc)})"
        )
        return False

    if not text.strip():
        # A scanned PDF opens cleanly and has no selectable text. That
        # is a real property of the file, and it is reported as such
        # rather than dressed up as an empty capture.
        content.notes.append(
            f"{path.name}: opened, but it has no selectable text"
        )
        return True

    content.text = _join(content.text, text)

    return True


def _image_is_readable(path: Path, content: CapturedContent) -> bool:
    """Whether an image can be identified from its own header."""

    try:
        with path.open("rb") as handle:
            header = handle.read(16)

    except OSError as exc:
        content.unreadable.append(f"{path.name}: {_brief(exc)}")
        return False

    if not _looks_like_image(path, header):
        content.unreadable.append(
            f"{path.name}: not a readable image"
        )
        return False

    content.notes.append(
        f"{path.name}: image kept as-is; no OCR is available, "
        "so it contributes no text"
    )

    return True


_IMAGE_SIGNATURES: tuple[tuple[bytes, ...], ...] = (
    (b"\x89PNG\r\n\x1a\n",),
    (b"\xff\xd8\xff",),
    (b"GIF87a",),
    (b"GIF89a",),
    (b"BM",),
)


def _looks_like_image(path: Path, header: bytes) -> bool:
    """Recognise an image from its magic bytes."""

    if any(header.startswith(signature) for signature in _IMAGE_SIGNATURES):
        return True

    # WebP is a RIFF container with "WEBP" at offset 8.
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return True

    return False


def _absorb_html(path: Path, content: CapturedContent) -> None:
    """Read text and metadata out of a page the user saved."""

    text = _read_text(path, content)

    parsed = _parse_html(text)

    if parsed.title and not content.title:
        content.title = parsed.title

    if parsed.author and not content.author:
        content.author = parsed.author

    if parsed.published_at and not content.published_at:
        content.published_at = parsed.published_at

    if parsed.text:
        content.text = _join(content.text, parsed.text)

    if not parsed.has_content:
        content.notes.append(
            f"{path.name}: the page held no post text; "
            "nothing was fetched to fill the gap"
        )


def _absorb_jsonl(path: Path, content: CapturedContent) -> None:
    """Read a JSON Lines file, one record per line."""

    text = _read_text(path, content)

    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()

        if not stripped:
            continue

        try:
            record = json.loads(stripped)

        except json.JSONDecodeError as exc:
            content.unreadable.append(
                f"{path.name} line {number}: not valid JSON ({exc.msg})"
            )
            continue

        if isinstance(record, dict):
            _apply_structured(record, content)


def _read_json(path: Path, content: CapturedContent) -> dict | None:
    """Read a structured capture, reporting why it could not be used."""

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8-sig", errors="replace")
        )

    except (OSError, json.JSONDecodeError) as exc:
        content.unreadable.append(f"{path.name}: {_brief(exc)}")
        return None

    if not isinstance(payload, dict):
        content.unreadable.append(
            f"{path.name}: expected an object"
        )
        return None

    return payload


def _apply_structured(payload: dict, content: CapturedContent) -> None:
    """
    Fold a structured record into the content.

    A capture file may carry the text itself, and it may name the item
    it belongs to. Only fields that are present are read; a capture
    with a URL and no text contributes the URL and nothing more.

    Read through the capture contract rather than a second list of key
    names, so the fields accepted here and the fields a user is told
    about cannot drift apart. A field that names a credential is dropped
    and reported.
    """
    for entry in _capture.credential_fields(payload):
        content.unreadable.append(
            f"{entry}: names a credential and was not read"
        )

    text = _capture.first_string(payload, _capture.CAPTURE_TEXT_KEYS)

    if text and text.strip():
        content.text = _join(content.text, text)

    title = _capture.first_string(payload, _capture.CAPTURE_TITLE_KEYS)

    if title and not content.title:
        content.title = title

    author = _capture.first_string(payload, _capture.CAPTURE_AUTHOR_KEYS)

    if author and not content.author:
        content.author = author

    published = _capture.first_string(
        payload, _capture.CAPTURE_DATE_KEYS
    )

    if published and not content.published_at:
        content.published_at = published

    notes = _capture.first_string(payload, _capture.CAPTURE_NOTES_KEYS)

    if notes and notes.strip():
        content.text = _join(content.text, notes)


def _read_text(path: Path, content: CapturedContent) -> str:
    """Read a text file, isolating a read failure to that one file."""

    try:
        return path.read_text(encoding="utf-8-sig", errors="replace")

    except OSError as exc:
        content.unreadable.append(f"{path.name}: {_brief(exc)}")

        return ""


_ABSOLUTE_PATH = re.compile(r"[A-Za-z]:\\[^\s'\"]*|/[^\s'\"]*")


def _brief(exc: BaseException) -> str:
    """
    A short, portable description of a failure.

    The raw message from a library usually quotes the absolute path it
    was given, which would put the user's home directory into a capture
    note and from there into a committed post. Naming the exception
    class and its last few words keeps the real cause and drops the
    local layout.
    """

    text = f"{type(exc).__name__}: {exc}"

    text = _ABSOLUTE_PATH.sub(lambda match: Path(match.group(0)).name, text)

    text = re.sub(r"\s+", " ", text).strip()

    if len(text) > 160:
        text = text[:157].rstrip() + "..."

    return text


def _join(existing: str, addition: str) -> str:
    """Append text without doubling what is already there."""

    addition = addition.strip()

    if not addition:
        return existing

    if not existing.strip():
        return addition

    return existing.rstrip() + "\n\n" + addition


# ---------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------


@dataclass
class ParsedPage:
    """What a saved page actually contained."""

    text: str = ""
    title: str | None = None
    author: str | None = None
    published_at: str | None = None
    has_content: bool = False


#: ``<meta>`` values worth reading, by the name a page uses for them.
_META_FIELDS: dict[str, str] = {
    "og:title": "title",
    "twitter:title": "title",
    "og:description": "description",
    "twitter:description": "description",
    "og:url": "url",
    "article:published_time": "published",
    "og:updated_time": "published",
    "author": "author",
    "article:author": "author",
    "og:site_name": "site",
}


class _PageReader(HTMLParser):
    """
    Pull the content out of a saved page.

    Only what the file contains is read. Nothing is looked up, and a
    field the page does not carry stays empty rather than being filled
    from a default or a guess.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)

        self.meta: dict[str, str] = {}
        self.parts: list[str] = []
        self.times: list[str] = []

        self._skip = 0
        self._in_json_ld = False
        self._json_ld: list[str] = []

    @property
    def json_ld(self) -> str:
        """The structured data the page carried, as written."""

        return "".join(self._json_ld)

    def handle_starttag(self, tag: str, attrs: list) -> None:
        attributes = {
            key.lower(): (value or "") for key, value in attrs
        }

        if tag == "meta":
            self._read_meta(attributes)

            return

        if tag == "script":
            script_type = attributes.get("type", "").lower()

            if script_type in {
                "application/ld+json",
                "application/json+ld",
            }:
                self._in_json_ld = True

            self._skip += 1

            return

        if tag in _DROPPED_TAGS:
            self._skip += 1
            return

        if tag == "time":
            moment = attributes.get("datetime")

            if moment:
                self.times.append(moment.strip())

        if tag in {"p", "div", "article", "section", "li", "br", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            if self._in_json_ld:
                self._in_json_ld = False

            self._skip = max(0, self._skip - 1)

            return

        if tag in _DROPPED_TAGS:
            self._skip = max(0, self._skip - 1)

    def handle_data(self, data: str) -> None:
        if self._in_json_ld:
            self._json_ld.append(data)

            return

        if self._skip:
            return

        if data.strip():
            self.parts.append(data)

    def close(self) -> None:  # type: ignore[override]
        super().close()

    def _read_meta(self, attributes: dict[str, str]) -> None:
        key = (
            attributes.get("property")
            or attributes.get("name")
            or attributes.get("itemprop")
            or ""
        ).strip().lower()

        if not key:
            return

        value = (attributes.get("content") or "").strip()

        if not value:
            return

        self.meta.setdefault(key, value)


def _parse_html(html: str) -> ParsedPage:
    """Read a saved page, preferring what the page states about itself."""
    reader = _PageReader()

    try:
        reader.feed(html)
        reader.close()

    except Exception:  # noqa: BLE001
        # A page truncated by a crashed save is still worth reading for
        # whatever it did contain.
        pass

    page = ParsedPage()

    for key, value in reader.meta.items():
        field_name = _META_FIELDS.get(key)

        if field_name == "title" and value and not page.title:
            page.title = value

        elif field_name == "author" and value and not page.author:
            page.author = value

        elif field_name == "published" and value and not page.published_at:
            page.published_at = value

    for moment in reader.times:
        if moment and not page.published_at:
            page.published_at = moment

    body = _clean_text("".join(reader.parts))

    description = reader.meta.get("og:description", "").strip()

    twitter = reader.meta.get("twitter:description", "").strip()

    if len(twitter) > len(description):
        description = twitter

    # The page's own description is the post text, because that is what
    # the page declares itself to be about, and a saved LinkedIn page is
    # mostly navigation chrome around it. It is only preferred when it
    # says something; a one-word description is not a body, and the
    # visible text is then a better reading of the same file.
    if len(description) >= _SUBSTANTIAL:
        page.text = description

    elif body:
        page.text = body

    elif reader.json_ld:
        page.text = _clean_text(_from_json_ld(reader.json_ld))

    page.has_content = bool(page.text.strip())

    return page


#: How long a declared description has to be before it is trusted as
#: the content. Below this, a page that says more in its body wins.
_SUBSTANTIAL = 40


_JSON_LD_TEXT_KEYS = ("articleBody", "text", "description", "headline")
_JSON_LD_NAME_KEYS = ("name", "givenName")


def _from_json_ld(blob: str) -> str:
    """Read text out of structured data the page carries."""
    try:
        payload = json.loads(blob)

    except json.JSONDecodeError:
        return ""

    return _search_json_ld(payload)


def _search_json_ld(payload: object) -> str:
    if isinstance(payload, dict):
        for key in _JSON_LD_TEXT_KEYS:
            value = payload.get(key)

            if isinstance(value, str) and value.strip():
                return value

        for value in payload.values():
            found = _search_json_ld(value)

            if found:
                return found

    elif isinstance(payload, list):
        for entry in payload:
            found = _search_json_ld(entry)

            if found:
                return found

    return ""


def _json_ld_name(payload: object) -> str:
    if isinstance(payload, dict):
        author = payload.get("author")

        if isinstance(author, dict):
            for key in _JSON_LD_NAME_KEYS:
                value = author.get(key)

                if isinstance(value, str) and value.strip():
                    return value

            return _json_ld_name(author)

        if isinstance(author, str) and author.strip():
            return author

        for value in payload.values():
            found = _json_ld_name(value)

            if found:
                return found

    elif isinstance(payload, list):
        for entry in payload:
            found = _json_ld_name(entry)

            if found:
                return found

    return ""


_WHITESPACE = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def _clean_text(text: str) -> str:
    """Tidy extracted text without changing what it says."""

    text = text.replace("\xa0", " ")

    text = _WHITESPACE.sub(" ", text)

    lines = [line.strip() for line in text.split("\n")]

    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


# ---------------------------------------------------------------------
# Association
# ---------------------------------------------------------------------


@dataclass
class BundleAssociation:
    """A bundle, and the item it claims to belong to."""

    path: Path
    url: str | None = None
    source_id: str | None = None
    directory_name: str = ""
    readable: bool = True
    reason: str = ""

    @property
    def claim_keys(self) -> tuple[str, ...]:
        keys: list[str] = []

        for value in (self.source_id, self.directory_name):
            if value:
                keys.append(value)
                keys.append(_safe_name(value))

        if self.directory_name:
            keys.append(self.directory_name)

        ordered: list[str] = []

        for key in keys:
            if key and key not in ordered:
                ordered.append(key)

        return tuple(ordered)


def association_for(
    path: str | Path,
    *,
    root: str | Path,
) -> BundleAssociation:
    """
    Work out which item a bundle belongs to.

    The directory name and the source id inside it are the primary
    claims. A ``url`` is read too, but only when the file can be
    normalized, so a bundle cannot attach itself to an item by
    containing an unusable link.
    """
    base = Path(root)

    target = Path(path)

    resolved = target.resolve() if target.is_absolute() else (base / target).resolve()

    association = BundleAssociation(
        path=resolved,
        directory_name=(
            resolved.name if resolved.is_dir() else _containing_name(resolved, base)
        ),
    )

    notes: CapturedContent = CapturedContent()

    claim_file = _association_file(resolved)

    structured: dict | None = None

    if claim_file is not None:
        structured = _read_json(claim_file, notes)

    if structured is None:
        return association

    raw_id = _first_string(structured, ("source_id", "sourceId", "id", "urn"))

    if raw_id:
        association.source_id = raw_id

    raw_url = _first_string(
        structured, ("url", "canonical_url", "canonicalUrl", "link", "permalink")
    )

    if raw_url:
        try:
            association.url = normalize_linkedin_url(raw_url).canonical

        except SavedItemUrlError:
            # An unusable URL is not a claim. The bundle is still read,
            # it just is not attached by it.
            association.readable = True
            association.reason = (
                f"the url {raw_url[:60]!r} could not be normalized, "
                "so the bundle was matched by name only"
            )

    return association


#: Why a bundle matched, in the order a match is preferred. A name is
#: cheapest and least ambiguous, so it is tried first.
MATCH_BY_NAME = "directory name"
MATCH_BY_ID = "source id in capture.json"
MATCH_BY_URL = "canonical url in capture.json"
MATCH_BY_MANIFEST = "bundle named in the manifest"
MATCH_NONE = "no bundle"


class BundleIndex:
    """
    Every bundle under a root, indexed by what each one claims.

    Built once, because the alternative is catastrophic at any real
    size: asking each item to consider each bundle reopens each bundle's
    ``capture.json`` every time, so a thousand items against a thousand
    captures is a million file reads before anything is imported. An
    index reads each capture file exactly once and then answers in
    constant time, which is what makes a large saved list usable at all.

    A bundle is claimed by at most one item, so two items can never end
    up sharing the same capture and being given the same body.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

        #: Every bundle, in a stable order.
        self.bundles: list[Path] = discover_bundles(self.root)

        #: Key to the bundle that claims it, in preference order.
        self._by_key: dict[str, list[tuple[int, str]]] = {}
        self._by_url: dict[str, tuple[int, str]] = {}

        #: What each bundle declared, for the orphan report.
        self.associations: dict[str, BundleAssociation] = {}

        self.problems: list[str] = []

        self._claimed: set[str] = set()

        # Checked at the top level and one level down, because that is
        # where a capture inbox puts its captures and where a link out
        # of the drop zone would be.
        for rejected in discover_rejected(self.root):
            self.problems.append(
                f"{rejected.name}: resolves outside the capture root"
            )

        for container in sorted(
            (
                entry
                for entry in self.root.iterdir()
                if entry.is_dir() and not entry.is_symlink()
                and entry.name.lower() in CONTAINER_NAMES
            ),
            key=lambda item: item.name,
        ):
            for rejected in discover_rejected(container):
                self.problems.append(
                    f"{rejected.name}: resolves outside the capture root"
                )

        self._build()

    # -----------------------------------------------------------------
    # Building
    # -----------------------------------------------------------------

    def _build(self) -> None:
        for position, bundle in enumerate(self.bundles):
            key = str(bundle)

            try:
                association = association_for(bundle, root=self.root)

            except (BundleError, OSError) as exc:
                self.problems.append(f"{bundle.name}: {exc}")
                continue

            self.associations[key] = association

            if association.url:
                # First claim wins, so a duplicate capture file cannot
                # take a bundle away from the folder named for it.
                self._by_url.setdefault(
                    association.url, (position, MATCH_BY_URL)
                )

            for claim in association.claim_keys:
                self._by_key.setdefault(claim, []).append(
                    (position, MATCH_BY_NAME)
                )

            if association.source_id:
                for claim in (association.source_id, _safe_name(
                    association.source_id
                )):
                    self._by_key[claim] = [
                        (position, MATCH_BY_ID)
                    ] + self._by_key.get(claim, [])

    # -----------------------------------------------------------------
    # Lookup
    # -----------------------------------------------------------------

    def lookup_keys(self, item: "object") -> list[str]:
        """
        Every key an item could be found by, most specific first.

        The item's own identifier is preferred over its name, because a
        folder named after the id is an unambiguous claim and a folder
        named after a human-readable title is not.
        """

        keys: list[str] = []

        def add(value: object) -> None:
            if not isinstance(value, str):
                return

            cleaned = value.strip()

            if cleaned and cleaned not in keys:
                keys.append(cleaned)

        add(getattr(item, "source_id", None))
        add(getattr(item, "identifier", None))
        add(getattr(item, "post_id", None))

        for name in (
            getattr(item, "source_id", None),
            getattr(item, "identifier", None),
            getattr(item, "post_id", None),
        ):
            if isinstance(name, str) and name.strip():
                add(_safe_name(name))

        return keys

    def find(self, item: "object") -> tuple[str, str] | None:
        """
        The bundle that belongs to an item, and why it was chosen.

        Returns the bundle as a root-relative name and the reason, so
        the post can record how its capture was found.
        """
        for key in self.lookup_keys(item):
            candidates = self._by_key.get(key)

            if not candidates:
                continue

            for position, reason in candidates:
                name = str(self.bundles[position])

                if name not in self._claimed:
                    self._claimed.add(name)

                    return name, reason

        url = getattr(item, "canonical_url", None)

        if isinstance(url, str) and url:
            found = self._by_url.get(url)

            if found is not None:
                position, reason = found

                name = str(self.bundles[position])

                if name not in self._claimed:
                    self._claimed.add(name)

                    return name, reason

        return None

    def find_by_name(self, name: str) -> tuple[str, str] | None:
        """
        The bundle a manifest names directly.

        A manifest that points at its own captures is the most explicit
        claim there is, so it is honoured even when a folder is named
        after something else.
        """
        candidate = Path(name)

        for bundle in self.bundles:
            if str(bundle) == name:
                return name, MATCH_BY_MANIFEST

        if (self.root / candidate).exists():
            for bundle in self.bundles:
                if str(bundle) == str(candidate):
                    return name, MATCH_BY_MANIFEST

        return None

    def unclaimed(self) -> list[str]:
        """Bundles no item has taken, which is what an orphan is."""

        return [
            str(bundle)
            for bundle in self.bundles
            if str(bundle) not in self._claimed
        ]

    def claim(self, name: str) -> bool:
        """
        Take a bundle for an item that already knows which one it wants.

        Used when a manifest names its own capture, which is a claim
        made directly by the user rather than inferred. Recorded the same
        way as any other claim, so a bundle cannot end up serving two
        items.
        """
        if name in self._claimed:
            return False

        self._claimed.add(name)

        return True

    def __len__(self) -> int:
        return len(self.bundles)


def _first_string(payload: dict, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = payload.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def _containing_name(path: Path, base: Path) -> str:
    """
    The name a loose file's directory would be matched by.

    A file dropped straight into the root has no directory of its own to
    name, so it can only be matched by what it says inside it.
    """
    try:
        relative = path.parent.resolve().relative_to(base.resolve())

    except ValueError:
        return ""

    return relative.name if len(relative.parts) == 1 else ""


def content_digest(content: CapturedContent) -> str:
    """
    A stable digest of a capture.

    Text and media are both hashed by content, not by file name, so
    replacing a screenshot with a different screenshot is recognised as
    a change. That is the whole purpose of the digest: it decides
    whether the enrichment already on file still describes this item.

    An empty capture digests to an empty string, not to the hash of
    nothing. A truthy digest is what marks an item as having content,
    so a bundle of unreadable files must not be counted as a capture
    that succeeded.
    """

    if not content.has_anything:
        return ""

    digest = hashlib.sha256()

    digest.update(content.text.strip().encode("utf-8", "replace"))

    for path in sorted(content.media, key=lambda item: item.name):
        digest.update(b"\x00")
        digest.update(path.name.encode("utf-8", "replace"))

        try:
            payload = path.read_bytes()

        except OSError:
            # An unreadable file still identifies the capture by name.
            digest.update(b"<unreadable>")
            continue

        digest.update(hashlib.sha256(payload).digest())

    return digest.hexdigest()
