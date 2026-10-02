"""
Read a local LinkedIn archive.

The archive is a JSON list of saved-post records and a folder of media
that belongs to them. This module reads it and nothing else: it does not
fetch anything, does not open a browser, and never writes to the archive
directory. Every path it resolves is checked against the media root
before it is touched, because a record is untrusted input even when the
person who wrote it is the person reading it.

One record failing costs that record. The archive is a list of hundreds
of independent posts, and a single unreadable row must not cost the
other four hundred and eighty-four, so every record is read inside its
own isolation and whatever went wrong is recorded against it.

Media is indexed once, by name, rather than searched for per record. The
obvious alternative is to look each referenced file up by listing the
media directory, which is a directory walk per post and turns an import
into thousands of walks over the same folder. The index is built with a
single pass and a size and digest read per file, both streamed, so the
peak memory cost is one file's first block rather than the whole set.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from src.ingestion.linkedin_archive.identity import (
    ArchiveIdentity,
    content_fingerprint,
    identify,
)


#: The archive's own manifest file.
ARCHIVE_JSON = "posts_archive.json"

#: The directory of already-downloaded assets beside it.
MEDIA_DIR = "media"

#: A directory inside the archive this module must never read.
#:
#: A browser session is credentials wearing a different hat. It is named
#: here so the refusal is a constant someone can read rather than an
#: omission someone has to notice, and so a test can assert the name is
#: still here.
FORBIDDEN_DIRECTORIES = ("chrome_session",)

#: Read in blocks, so a large asset is hashed without being held whole.
_BLOCK = 1 << 20

#: Signatures for the formats this archive is expected to hold. Read
#: from the bytes rather than believed from the extension: a third of
#: this archive's images are named ``.jpg`` and are not JPEGs.
_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", "jpeg"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"BM", "bmp"),
    (b"%PDF", "pdf"),
    (b"RIFF", "webp"),
    (b"<?xml", "svg"),
    (b"<svg", "svg"),
)


class ArchiveError(ValueError):
    """Raised when the archive as a whole cannot be read."""


class MediaError(ValueError):
    """Raised when one media file cannot be used, and named as such."""


@dataclass
class MediaAsset:
    """
    One file in the archive's media directory.

    ``name`` is the archive's own filename and is what a record refers
    to. ``detected_format`` is what the bytes actually are, which is not
    always the same thing: the archive names every asset ``.jpg`` and a
    third of them are PNG or GIF. Both are recorded, because a reader who
    is told a file is a JPEG and handed a PNG is being told something
    false, and because the extension is what the record refers to and so
    cannot simply be changed.
    """

    name: str
    path: Path
    size: int = 0
    digest: str = ""
    detected_format: str = "unknown"
    readable: bool = False
    problem: str = ""

    @property
    def is_image(self) -> bool:
        return self.detected_format in {
            "jpeg",
            "png",
            "gif",
            "bmp",
            "webp",
        }

    @property
    def is_document(self) -> bool:
        return self.detected_format == "pdf"

    @property
    def is_svg(self) -> bool:
        return self.detected_format == "svg"

    @property
    def extension_agrees(self) -> bool:
        """Whether the name describes the bytes."""
        suffix = self.path.suffix.lower().lstrip(".")

        if not suffix:
            return False

        if suffix == "jpg":
            return self.detected_format == "jpeg"

        if suffix == "jpeg":
            return self.detected_format == "jpeg"

        if suffix == "webp":
            return self.detected_format == "webp"

        return suffix == self.detected_format

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "size": self.size,
            "sha256": self.digest,
            "detected_format": self.detected_format,
            "readable": self.readable,
            "extension_agrees": self.extension_agrees,
            "problem": self.problem,
        }


@dataclass
class MediaIndex:
    """
    Every asset under the media root, looked up by name.

    Built in one pass. A name the archive refers to but the folder does
    not hold is kept as a miss rather than being dropped, so the report
    can say which post lost which file.
    """

    root: Path
    assets: dict[str, MediaAsset] = field(default_factory=dict)
    by_digest: dict[str, list[str]] = field(default_factory=dict)
    rejected: list[str] = field(default_factory=list)
    unreadable: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.assets)

    @property
    def total_bytes(self) -> int:
        return sum(asset.size for asset in self.assets.values())

    def get(self, name: object) -> MediaAsset | None:
        """The asset a record refers to, or None when it is not there."""
        if not isinstance(name, str):
            return None

        return self.assets.get(Path(name.strip()).name)

    def shared_by_digest(self, name: str) -> list[str]:
        """Other names holding byte-identical content."""
        asset = self.assets.get(name)

        if asset is None or not asset.digest:
            return []

        return [
            other
            for other in self.by_digest.get(asset.digest, [])
            if other != name
        ]

    def unique_names(self) -> int:
        """How many distinct files, counting identical content once."""
        return len(self.by_digest)


@dataclass
class ArchiveRecord:
    """
    One post, as the archive recorded it, plus what was worked out.

    ``raw_text`` is the archive's text, byte for byte. Nothing in this
    pipeline rewrites it; normalization, where any is wanted, happens
    downstream and keeps the original.
    """

    post_id: str
    identity: ArchiveIdentity
    raw_text: str = ""
    author: str | None = None
    headline: str | None = None
    profile_url: str | None = None
    scraped_at: str | None = None
    relative_time: str | None = None

    media: list[MediaAsset] = field(default_factory=list)
    missing_media: list[str] = field(default_factory=list)
    original_media_urls: list[str] = field(default_factory=list)

    claimed_media_count: int = 0
    claimed_multi_slide: bool = False
    notes: list[str] = field(default_factory=list)

    content_digest: str = ""
    duplicate_of: str | None = None

    @property
    def has_text(self) -> bool:
        return bool(self.raw_text.strip())

    @property
    def has_media(self) -> bool:
        return bool(self.media)

    @property
    def is_duplicate(self) -> bool:
        return self.duplicate_of is not None

    def note(self, message: str) -> None:
        if message and message not in self.notes:
            self.notes.append(message)


@dataclass
class ArchiveFailure:
    """
    One record that could not be read, and why.

    Carries the stage so a report can say where it broke, and whether it
    is worth retrying, because a malformed row is not going to become
    well-formed on its own while a file that was busy might.
    """

    index: int
    post_id: str
    stage: str
    error_type: str
    message: str
    recoverable: bool = False

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "post_id": self.post_id,
            "stage": self.stage,
            "error_type": self.error_type,
            "message": self.message,
            "recoverable": self.recoverable,
        }


@dataclass
class Archive:
    """
    A read archive: its records, its media index, and what went wrong.

    ``path`` is kept for reporting and is never written to. It is
    deliberately the only place the archive's location appears, so a
    report that accidentally renders it is easy to notice.
    """

    path: Path
    media: MediaIndex
    records: list[ArchiveRecord] = field(default_factory=list)
    failures: list[ArchiveFailure] = field(default_factory=list)
    total_seen: int = 0

    @property
    def duplicates(self) -> list[ArchiveRecord]:
        return [record for record in self.records if record.is_duplicate]

    def unique(self) -> list[ArchiveRecord]:
        return [
            record for record in self.records if not record.is_duplicate
        ]


# ---------------------------------------------------------------------
# The archive on disk
# ---------------------------------------------------------------------


def json_path_for(root: str | Path) -> Path:
    return Path(root) / ARCHIVE_JSON


def media_root_for(root: str | Path) -> Path:
    return Path(root) / MEDIA_DIR


def assert_readable_root(root: str | Path) -> Path:
    """
    Check the archive root before anything is read from it.

    Refuses a root that does not exist, and names the one directory
    inside an archive this module will not open. A caller who points
    this at the archive's parent still gets a useful answer, because the
    refusal names the file it wanted.
    """

    base = Path(root).expanduser()

    if not base.exists():
        raise ArchiveError(f"the archive directory does not exist: {base}")

    if not base.is_dir():
        raise ArchiveError(f"the archive path is not a directory: {base}")

    manifest = base / ARCHIVE_JSON

    if not manifest.is_file():
        raise ArchiveError(
            f"no {ARCHIVE_JSON} in {base}. Point this at the directory "
            f"that contains it."
        )

    return base


def build_media_index(root: str | Path) -> MediaIndex:
    """
    Index every file under the media directory, once.

    Each file is identified from its first bytes and hashed in blocks, so
    the cost is one streaming read per file regardless of size. A file
    that cannot be opened is recorded and skipped rather than failing the
    import, because one unreadable image is not a reason to lose four
    hundred posts.
    """

    base = Path(root).expanduser().resolve()
    media_root = base / MEDIA_DIR

    index = MediaIndex(root=media_root)

    if not media_root.is_dir():
        return index

    for path in sorted(media_root.rglob("*")):
        # A link is checked before anything else, and before the
        # is_file test, because a link to a directory is not a file and
        # would otherwise be skipped in silence -- which is the one
        # outcome a reader cannot tell apart from "there was nothing
        # there".
        if path.is_symlink() and not _within(path, media_root):
            index.rejected.append(path.name)
            continue

        if not path.is_file():
            continue

        if not _within(path, media_root):
            index.rejected.append(path.name)
            continue

        asset = _describe(path)

        if not asset.readable:
            index.unreadable.append(path.name)

        index.assets[asset.name] = asset

        if asset.digest:
            index.by_digest.setdefault(asset.digest, []).append(asset.name)

    return index


def _describe(path: Path) -> MediaAsset:
    """Read one file's facts, streaming so size does not matter."""

    asset = MediaAsset(name=path.name, path=path)

    try:
        asset.size = path.stat().st_size

        hasher = hashlib.sha256()

        with path.open("rb") as handle:
            header = handle.read(64)

            hasher.update(header)

            while True:
                block = handle.read(_BLOCK)

                if not block:
                    break

                hasher.update(block)

        asset.digest = hasher.hexdigest()
        asset.detected_format = _sniff(header)
        asset.readable = asset.size > 0

        if not asset.readable:
            asset.problem = "the file is empty"

        elif asset.detected_format == "unknown":
            asset.problem = "the file is not a format this project reads"

    except OSError as exc:
        asset.problem = f"{type(exc).__name__}: {exc}"

    return asset


def _sniff(header: bytes) -> str:
    """What a file's first bytes say it is."""
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "webp"

    for signature, name in _SIGNATURES:
        if header.startswith(signature):
            return name

    return "unknown"


def _within(candidate: Path, root: Path) -> bool:
    """
    Whether a path really is inside a root.

    Both sides are resolved first, so a symbolic link or a ``..``
    segment cannot make a path outside the root look like one inside it.
    This is the single containment check in this module; every read
    goes through it.
    """

    try:
        resolved = candidate.resolve()
        base = root.resolve()

    except OSError:
        return False

    try:
        resolved.relative_to(base)

    except ValueError:
        return False

    return True


# ---------------------------------------------------------------------
# The records
# ---------------------------------------------------------------------


def read_archive(
    root: str | Path,
    *,
    include_duplicates: bool = True,
) -> Archive:
    """
    Read the whole archive.

    The JSON is loaded and validated at the top level first, because a
    file that is not a list of objects is a different problem from a
    record with a missing field, and conflating them would report a
    corrupt archive as hundreds of malformed posts.

    Each record is then read on its own. A record that cannot be read is
    recorded as a failure and the rest continue, which is the property
    that makes a five-hundred-post import survivable.
    """

    base = assert_readable_root(root)
    manifest = json_path_for(base)

    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArchiveError(
            f"{ARCHIVE_JSON} is not valid JSON ({exc.msg} at line "
            f"{exc.lineno}, column {exc.colno})"
        ) from exc
    except OSError as exc:
        raise ArchiveError(
            f"{ARCHIVE_JSON} could not be read: {exc}"
        ) from exc

    if isinstance(payload, dict) and isinstance(payload.get("posts"), list):
        # Tolerated because an archive written by a tool that wraps its
        # export is still an archive. Read the list, say so.
        payload = payload["posts"]

    if not isinstance(payload, list):
        raise ArchiveError(
            f"{ARCHIVE_JSON} holds a {type(payload).__name__}, not a list "
            "of post records"
        )

    archive = Archive(
        path=base,
        media=build_media_index(base),
        total_seen=len(payload),
    )

    for index, entry in enumerate(payload):
        try:
            record = read_record(entry, archive.media, index=index)

        except Exception as exc:  # noqa: BLE001
            archive.failures.append(
                ArchiveFailure(
                    index=index,
                    post_id=_salvage_id(entry),
                    stage="record",
                    error_type=type(exc).__name__,
                    message=str(exc)[:400],
                    recoverable=False,
                )
            )
            continue

        if record is not None:
            archive.records.append(record)

    _mark_duplicates(archive.records)

    if not include_duplicates:
        archive.records = archive.unique()

    return archive


def read_record(
    entry: object,
    media: MediaIndex,
    *,
    index: int = 0,
) -> ArchiveRecord | None:
    """
    Read one record.

    A record with no post id is not a post; it cannot be identified,
    deduplicated or resumed, and a record the pipeline cannot name is a
    record it will import again next run. It is refused rather than
    given a synthetic id, and the refusal is reported.
    """

    if not isinstance(entry, dict):
        raise ArchiveError(
            f"record {index} is a {type(entry).__name__}, not an object"
        )

    post_id = str(entry.get("post_id") or "").strip()

    if not post_id:
        raise ArchiveError(
            f"record {index} has no post_id, so it cannot be identified"
        )

    identity = identify(
        post_id,
        entry.get("permalink"),
        text=str(entry.get("text") or ""),
    )

    record = ArchiveRecord(
        post_id=post_id,
        identity=identity,
        raw_text=str(entry.get("text") or ""),
        scraped_at=_text_or_none(entry.get("scraped_at")),
        relative_time=_text_or_none(entry.get("relative_time")),
    )

    author = entry.get("author")

    if isinstance(author, dict):
        record.author = _text_or_none(author.get("name"))
        record.headline = _text_or_none(author.get("headline"))
        record.profile_url = _text_or_none(author.get("profile_url"))
    elif author is not None:
        record.note("the author was not an object and was not read")

    if not record.author or record.author.strip().lower() == "unknown":
        # Kept as whatever the archive said, because rewriting it to
        # nothing would lose the fact that the archive failed to find an
        # author, which is different from the post having no author.
        record.note(
            "the archive recorded no usable author name for this post"
        )

    media_block = entry.get("media")

    if not isinstance(media_block, dict):
        record.note("the media block was missing or not an object")
        media_block = {}

    record.claimed_media_count = _as_int(media_block.get("total_media_count"))
    record.claimed_multi_slide = bool(media_block.get("is_multi_slide"))

    record.original_media_urls = [
        url
        for url in (
            _text_or_none(item) for item in _as_list(media_block.get("original_urls"))
        )
        if url
    ]

    _attach_media(record, media_block, media)

    record.content_digest = content_fingerprint(record.raw_text)

    if not record.has_text:
        record.note("the archive recorded no text for this post")

    return record


def _attach_media(
    record: ArchiveRecord,
    media_block: dict,
    media: MediaIndex,
) -> None:
    """
    Resolve the files a record refers to.

    The archive's file list is a claim, not a fact: this checks it
    against the folder. A file that is listed twice, which this archive
    does for three carousel posts whose pagination failed, is counted
    once and the repeat is noted, because two references to one file is
    not two files.
    """

    names = _as_list(media_block.get("saved_files"))

    seen: set[str] = set()

    for entry in names:
        if not isinstance(entry, str) or not entry.strip():
            record.note("a media entry was empty or not a filename")
            continue

        name = entry.strip()
        base = Path(name).name

        if base != name.strip():
            # A path where a name was expected. The basename is used and
            # the fact is recorded, so a record cannot make the importer
            # read a file outside the media directory by writing a path.
            record.note(
                f"the media reference {name!r} included a path; only its "
                "filename was used"
            )

        if name in seen:
            record.note(
                f"{name} is listed more than once in this record and was "
                "counted once"
            )
            continue

        seen.add(name)

        asset = media.get(name)

        if asset is None:
            record.missing_media.append(name)
            record.note(f"{name} is listed by the archive but not in the "
                        "media folder")
            continue

        record.media.append(asset)

        if not asset.readable:
            record.note(f"{name} could not be read: {asset.problem}")

        elif not asset.extension_agrees:
            suffix = asset.path.suffix or "no extension"

            record.note(
                f"{name} is named {suffix} but its content is "
                f"{asset.detected_format}"
            )

    if record.claimed_multi_slide and len(record.media) < 2:
        record.note(
            "the archive marked this as multi-slide but supplied "
            f"{len(record.media)} distinct file(s), so the remaining "
            "slides were not collected"
        )

    if record.claimed_media_count and (
        record.claimed_media_count != len(record.media)
    ):
        record.note(
            f"the archive counted {record.claimed_media_count} media "
            f"item(s) and {len(record.media)} were usable"
        )


def _mark_duplicates(records: list[ArchiveRecord]) -> None:
    """
    Point repeated content at the record that keeps it.

    Only whole-text matches count. A record that is dropped as a
    duplicate is not deleted: it stays in the archive and in the report,
    and the surviving record names it, so nothing the archive held is
    lost and a reader can see that two captures of one post existed.

    The keeper is chosen by what is most useful to lose nothing: a
    record with a permalink beats one without, and a record with media
    beats one without. That is the whole of the policy, and it is stated
    here because a merge that silently discards the only copy of a URL
    would be a data loss dressed as a tidy-up.
    """

    by_content: dict[str, list[ArchiveRecord]] = {}

    for record in records:
        if not record.content_digest:
            continue

        by_content.setdefault(record.content_digest, []).append(record)

    for group in by_content.values():
        if len(group) < 2:
            continue

        keeper = max(
            group,
            key=lambda record: (
                record.identity.has_permalink,
                bool(record.media),
                len(record.raw_text),
                record.post_id,
            ),
        )

        for record in group:
            if record is keeper:
                continue

            record.duplicate_of = keeper.post_id
            record.note(
                f"the same text appears as {keeper.post_id}, which keeps "
                "this record's place; both are reported"
            )


# ---------------------------------------------------------------------
# Small readers
# ---------------------------------------------------------------------


def _as_list(value: object) -> list:
    if isinstance(value, list):
        return value

    return []


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        return 0

    if isinstance(value, int):
        return value

    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())

    return 0


def _text_or_none(value: object) -> str | None:
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None

    if value is None:
        return None

    return str(value)


def _salvage_id(entry: object) -> str:
    """The best identifier available on a record that failed to read."""
    if isinstance(entry, dict):
        for key in ("post_id", "id", "urn"):
            value = entry.get(key)

            if isinstance(value, str) and value.strip():
                return value.strip()[:80]

    return "(no identifier)"


__all__ = [
    "ARCHIVE_JSON",
    "FORBIDDEN_DIRECTORIES",
    "MEDIA_DIR",
    "Archive",
    "ArchiveError",
    "ArchiveFailure",
    "ArchiveRecord",
    "MediaAsset",
    "MediaError",
    "MediaIndex",
    "assert_readable_root",
    "build_media_index",
    "json_path_for",
    "media_root_for",
    "read_archive",
    "read_record",
]
