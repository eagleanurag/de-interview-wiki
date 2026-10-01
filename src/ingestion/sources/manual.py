"""
The manual source.

Content the user has already captured and authorized: text they pasted,
a file they dropped in, a document, a screenshot, or a bundle directory
holding any of those. This is the default source because it needs no
credentials and no network, which is what makes the rest of the
pipeline testable without any live source.

It is a first-class source, not a fallback. A bundle may be:

* a directory holding ``capture.json`` or ``post.json``
* a directory holding a text file, with any images and PDFs beside it
* a directory holding only media, such as a PDF or a screenshot
* a single text file
* a single ``.jsonl`` export, one post per line

Anything the user drops into a directory is discovered without the
importer needing to understand the internal post schema, which is the
point: the user supplies material, not JSON.

Everything here is deterministic. A bundle's identifier comes from the
capture data when it declares one and from a content digest otherwise,
so importing the same bundle twice refreshes one post instead of
creating two, and a changed bundle is recognised as changed.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime, timezone
from pathlib import Path

from src.ingestion.errors import InvalidPostError
from src.ingestion.sources.base import (
    CollectedPost,
    CollectionStopped,
    CollectionState,
    StopReason,
    Source,
)


#: Structured capture files, in precedence order.
CAPTURE_NAMES = ("capture.json", "post.json")

#: Files that carry the post's text.
TEXT_SUFFIXES = frozenset({".txt", ".md", ".markdown"})

#: Media the post may own.
IMAGE_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
)
DOCUMENT_SUFFIXES = frozenset({".pdf"})

#: Everything a post may own as media.
MEDIA_SUFFIXES = IMAGE_SUFFIXES | DOCUMENT_SUFFIXES

#: Files that are never treated as a bundle of their own.
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
    }
)

#: Keys a capture may use for the same field. Ordered so the first
#: present wins, which keeps a capture that uses an alias working
#: without the caller guessing the schema.
TEXT_KEYS = ("text", "original_text", "content", "body")
ID_KEYS = (
    "source_post_id",
    "id",
    "urn",
    "post_id",
    "external_id",
)
URL_KEYS = ("url", "source_url", "link", "permalink")
PUBLISHED_KEYS = ("published_at", "publishedAt", "date", "timestamp")
AUTHOR_KEYS = ("author", "authorName", "author_name", "byline")
MEDIA_KEYS = ("media", "images", "attachments", "files")

#: Keys consumed into fields rather than carried as extras.
_CONSUMED_KEYS = frozenset(
    TEXT_KEYS
    + ID_KEYS
    + URL_KEYS
    + PUBLISHED_KEYS
    + AUTHOR_KEYS
    + MEDIA_KEYS
)


class ManualSource(Source):
    """
    Reads posts from a directory of capture bundles.

    Directories are walked in sorted order so a run is reproducible.
    """

    name = "manual"
    platform = "manual"

    def __init__(
        self,
        bundle_root: str | Path,
        *,
        platform: str | None = None,
    ) -> None:
        self.bundle_root = Path(bundle_root)
        self._platform = platform

    # -----------------------------------------------------------------
    # Discovery
    # -----------------------------------------------------------------

    def discover(self, **limits: object):
        """
        Yield one collected post per bundle, in sorted order.

        Respects ``max_posts`` and ``since``/``until`` the same way the
        browser source does, so both sources behave identically for the
        collector.
        """

        max_posts = limits.get("max_posts")
        since = limits.get("since")
        until = limits.get("until")

        if not self.bundle_root.exists():
            raise CollectionStopped(
                StopReason.FAILED,
                f"Manual source root does not exist: "
                f"{self.bundle_root}",
            )

        produced = 0

        for bundle in self._bundles():
            if isinstance(max_posts, int) and produced >= max_posts:
                raise CollectionStopped(
                    StopReason.MAX_POSTS,
                    f"Reached the configured limit of {max_posts}.",
                )

            for collected in self._read_bundle(bundle):
                if not self._within_window(
                    collected.published_at, since=since, until=until
                ):
                    continue

                collected.extra["bundle"] = str(bundle)

                yield collected

                produced += 1

                if isinstance(max_posts, int) and produced >= max_posts:
                    raise CollectionStopped(
                        StopReason.MAX_POSTS,
                        f"Reached the configured limit of {max_posts}.",
                    )

        raise CollectionStopped(StopReason.EXHAUSTED)

    def _bundles(self) -> list[Path]:
        """
        Every bundle under the root, in a stable order.

        Discovery works on directories, top down. A directory that
        directly holds text, media or a capture is one bundle and its
        files belong to it; a directory that only holds other
        directories is not a bundle itself, so a nested layout does not
        turn the parent into a post.

        Working top down is what stops a bundle's own images from each
        becoming a separate post, which is what treating files
        individually would do.
        """

        if self.bundle_root.is_file():
            return [self.bundle_root]

        bundles: list[Path] = []

        for directory in self._directories():
            if directory == self.bundle_root:
                # The root is a container of bundles, not a bundle. Two
                # notes dropped side by side are two posts; merging them
                # would invent a document the user never wrote.
                bundles.extend(self._loose_files(directory))
                continue

            files = [
                path
                for path in sorted(directory.iterdir())
                if path.is_file()
            ]

            if not files:
                continue

            names = {path.name for path in files}

            if names & set(CAPTURE_NAMES):
                # The directory is the bundle; the capture is not also
                # a post of its own.
                bundles.append(directory)
                continue

            exports = [
                path
                for path in files
                if path.suffix.lower() == ".jsonl"
            ]

            if exports:
                # An export is a bundle in its own right, one post per
                # line, so its lines stay separable.
                bundles.extend(exports)
                continue

            text = [
                path
                for path in files
                if path.suffix.lower() in TEXT_SUFFIXES
            ]

            media = [
                path
                for path in files
                if path.suffix.lower() in MEDIA_SUFFIXES
            ]

            payloads = [
                path
                for path in files
                if path.suffix.lower() == ".json"
            ]

            if text or media:
                bundles.append(directory)
                continue

            # A directory holding only loose JSON has no declared
            # structure, so each object is offered as a bundle in its
            # own right.
            bundles.extend(payloads)

        return _dedupe(bundles)

    def _loose_files(self, directory: Path) -> list[Path]:
        """
        Files sitting directly in the root.

        Each is its own bundle, because the root holds bundles rather
        than being one.
        """

        bundles: list[Path] = []

        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue

            if path.name in CAPTURE_NAMES:
                bundles.append(directory)
                continue

            suffix = path.suffix.lower()

            if suffix in IGNORED_SUFFIXES:
                continue

            if (
                suffix in TEXT_SUFFIXES
                or suffix == ".jsonl"
                or suffix == ".json"
                or suffix in MEDIA_SUFFIXES
            ):
                bundles.append(path)

        return bundles

    def _directories(self) -> list[Path]:
        """Every directory under the root, shallowest first."""

        found = [
            self.bundle_root,
            *self.bundle_root.rglob("*"),
        ]

        directories = [
            path for path in found if path.is_dir()
        ]

        return sorted(
            directories,
            key=lambda path: (len(path.parts), str(path)),
        )

    # -----------------------------------------------------------------
    # Reading
    # -----------------------------------------------------------------

    def _read_bundle(self, bundle: Path) -> list[CollectedPost]:
        """
        Read one bundle into one or more collected posts.

        A JSONL export yields one post per line, because that is the
        shape the format exists to express. Merging them into a single
        post would produce one document that answers to none of them.
        """

        if bundle.is_file():
            suffix = bundle.suffix.lower()

            if suffix == ".jsonl":
                return self._read_jsonl_bundle(bundle)

            if suffix in TEXT_SUFFIXES:
                collected = self._read_text_bundle(bundle)

                return [collected] if collected is not None else []

            if suffix == ".json":
                payload = self._read_json(bundle)

                return [self._from_payload(payload, bundle.parent)]

            if suffix in IMAGE_SUFFIXES or suffix in DOCUMENT_SUFFIXES:
                return [self._read_media_bundle(bundle.parent)]

            return []

        capture = next(
            (
                bundle / name
                for name in CAPTURE_NAMES
                if (bundle / name).is_file()
            ),
            None,
        )

        if capture is not None:
            payload = self._read_json(capture)

            return [self._from_payload(payload, bundle)]

        text = self._directory_text(bundle)
        media = self._directory_media(bundle)

        if not text and not media:
            return []

        collected = self._from_files(bundle, text, media)

        if collected.text or collected.media:
            return [collected]

        # A file with neither content nor media is nothing. Storing it
        # would publish an empty page.
        return []

    def _read_jsonl_bundle(self, bundle: Path) -> list[CollectedPost]:
        """
        Read a JSON Lines export: one post per line.

        A malformed line is recorded and skipped rather than aborting
        the run, because losing one bad record is better than losing the
        export.
        """

        collected: list[CollectedPost] = []
        problems: list[str] = []

        for number, line in enumerate(
            bundle.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines(),
            start=1,
        ):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                problems.append(f"{bundle.name} line {number}")
                continue

            if not isinstance(payload, dict):
                problems.append(f"{bundle.name} line {number}: not an object")
                continue

            entry = self._from_payload(payload, bundle, line_number=number)

            if entry is not None:
                collected.append(entry)

        if problems and not collected:
            raise InvalidPostError(
                f"{bundle.name} has no usable records: "
                + ", ".join(problems[:5])
            )

        for entry in collected:
            entry.extra["bundle"] = str(bundle)

        if problems:
            # Reported so a truncated export is visible rather than
            # silently short.
            collected[0].extra["jsonl_problems"] = problems[:20]

        return collected

    def _read_text_bundle(self, bundle: Path) -> CollectedPost | None:
        """
        A single text file, with any media sitting beside it.

        Returns None when the file has no content, so an empty note is
        not stored as an empty post.
        """

        text = bundle.read_text(
            encoding="utf-8", errors="replace"
        ).strip()

        media = self._sibling_media(bundle)

        if not text and not media:
            return None

        return self._from_files(bundle, text, media)

    def _read_media_bundle(self, directory: Path) -> CollectedPost:
        """
        A directory holding only media.

        The post has no text of its own yet. That is not a failure: a
        PDF or a screenshot carries its content, and the media stage is
        what turns it into text. Until then the post exists with its
        provenance and its media, which is what keeps it reachable.
        """

        media = self._directory_media(directory)

        return self._from_files(directory, "", media)

    def _from_payload(
        self,
        payload: dict,
        bundle: Path,
        *,
        line_number: int | None = None,
    ) -> CollectedPost | None:
        """Build a post from a capture object."""

        text = _first_string(payload, TEXT_KEYS)

        identifier = _first_string(payload, ID_KEYS)

        if not identifier:
            # No declared identity, so derive one from the content.
            # A hash is stable across runs, which is what makes a
            # repeated import refresh rather than duplicate.
            identifier = content_digest(
                text, [str(item) for item in self._media_paths(payload, bundle)]
            )

        media = self._media_paths(payload, bundle)

        extra = {
            key: value
            for key, value in payload.items()
            if key not in _CONSUMED_KEYS
        }

        if line_number is not None:
            extra["jsonl_line"] = line_number

        return CollectedPost(
            source_post_id=str(identifier),
            text=text,
            url=_first_string(payload, URL_KEYS),
            published_at=_first_string(payload, PUBLISHED_KEYS),
            author=_first_string(payload, AUTHOR_KEYS),
            media=media,
            extra=extra,
        )

    def _from_files(
        self,
        bundle: Path,
        text: str,
        media: list[Path],
    ) -> CollectedPost:
        """
        Build a post from files on disk.

        The identifier is the bundle's own name when it is one, and a
        content digest otherwise, so a named bundle keeps a readable id
        and an anonymous one still has a stable one.
        """

        stem = bundle.stem if bundle.is_file() else bundle.name

        readable = _ID_SAFE.sub("-", stem.strip().lower()).strip("-")

        digest = content_digest(text, [str(item) for item in media])

        if not readable or readable in {"content", "capture", "post"}:
            identifier = f"manual-{digest[:12]}"
        else:
            identifier = f"{readable}-{digest[:8]}"

        return CollectedPost(
            source_post_id=identifier,
            text=text,
            media=media,
            extra={
                # The digest is recorded so a later run can tell an
                # unchanged bundle from an edited one.
                "content_digest": digest,
                "bundle_name": bundle.name,
            },
        )

    # -----------------------------------------------------------------
    # Files
    # -----------------------------------------------------------------

    def _directory_text(self, directory: Path) -> str:
        """Every text file in a bundle, joined in name order."""

        parts: list[str] = []

        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue

            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue

            content = path.read_text(
                encoding="utf-8", errors="replace"
            ).strip()

            if content:
                parts.append(content)

        return "\n\n".join(parts)

    def _directory_media(self, directory: Path) -> list[Path]:
        """Every media file in a bundle, in name order."""

        return [
            path
            for path in sorted(directory.iterdir())
            if path.is_file()
            and path.suffix.lower() in MEDIA_SUFFIXES
        ]

    def _sibling_media(self, bundle: Path) -> list[Path]:
        """Media sitting beside a single file."""

        parent = bundle.parent

        return [
            path
            for path in sorted(parent.iterdir())
            if path.is_file()
            and path != bundle
            and path.suffix.lower() in MEDIA_SUFFIXES
        ]

    def _media_paths(
        self,
        payload: dict,
        bundle: Path,
    ) -> list[Path]:
        """
        Media named by a capture.

        A relative name resolves against the bundle. A name that tries
        to escape the bundle is refused rather than followed, because a
        capture is user-supplied content and must not be able to name a
        file anywhere on the machine.
        """

        raw = None

        for key in MEDIA_KEYS:
            if payload.get(key):
                raw = payload[key]
                break

        if not isinstance(raw, list):
            return []

        base = bundle if bundle.is_dir() else bundle.parent

        resolved: list[Path] = []

        for item in raw:
            if isinstance(item, dict):
                item = item.get("path") or item.get("file")

            if not isinstance(item, str) or not item.strip():
                continue

            candidate = Path(item.strip())

            if candidate.is_absolute():
                continue

            target = (base / candidate).resolve()

            try:
                target.relative_to(base.resolve())
            except ValueError:
                # Escapes the bundle directory.
                continue

            if target.is_file():
                resolved.append(target)

        return resolved

    def _read_json(self, path: Path) -> dict:
        try:
            payload = json.loads(
                path.read_text(
                    encoding="utf-8", errors="replace"
                )
            )
        except json.JSONDecodeError as exc:
            raise InvalidPostError(
                f"{path} is not valid JSON: {exc}"
            ) from exc

        if not isinstance(payload, dict):
            raise InvalidPostError(
                f"{path} must contain a JSON object"
            )

        return payload

    @property
    def platform(self) -> str:  # type: ignore[override]
        return self._platform or self.name

    @staticmethod
    def _within_window(
        published_at: str | None,
        *,
        since: object,
        until: object,
    ) -> bool:
        if not published_at or (since is None and until is None):
            return True

        moment = _parse_moment(published_at)

        if moment is None:
            # An unparsable timestamp is kept rather than dropped, so a
            # format change never silently loses content.
            return True

        lower = _parse_moment(str(since)) if since else None
        upper = _parse_moment(str(until)) if until else None

        if lower and moment < lower:
            return False

        if upper and moment > upper:
            return False

        return True


_ID_SAFE = re.compile(r"[^a-z0-9._-]+")


def _dedupe(paths: list[Path]) -> list[Path]:
    """Unique paths, first-seen order, resolved for comparison."""

    seen: set[Path] = set()
    unique: list[Path] = []

    for path in paths:
        try:
            key = path.resolve()
        except OSError:
            key = path

        if key in seen:
            continue

        seen.add(key)
        unique.append(path)

    return unique


def content_digest(text: str, media: list[str]) -> str:
    """
    A stable digest of a post's text and its media.

    Used for identity when a bundle declares none, and to tell an edited
    bundle from an unchanged one. The media entries are sorted so a
    different file order does not read as a different post, and the
    file *names* are hashed rather than their contents so a large PDF is
    not read twice.
    """

    digest = hashlib.sha256()

    digest.update(text.strip().encode("utf-8", "replace"))

    for name in sorted(media):
        digest.update(b"\x00")
        digest.update(Path(name).name.encode("utf-8", "replace"))

    return digest.hexdigest()


def _first_string(payload: dict, keys: tuple[str, ...]) -> str:
    """The first non-empty string among several accepted key names."""

    for key in keys:
        value = payload.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def _optional_str(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()

    return None


def _parse_moment(value: str) -> datetime | None:
    """
    Parse a timestamp to a naive UTC value.

    Normalized to naive UTC because sources are inconsistent about
    offsets, and comparing an offset-aware value with a naive one
    raises. Naive UTC keeps every comparison well defined.
    """

    text = value.strip()

    if not text:
        return None

    candidate = text.replace("Z", "+00:00")

    for attempt in (candidate, text[:10]):
        try:
            parsed = datetime.fromisoformat(attempt)
        except ValueError:
            continue

        if parsed.tzinfo is None:
            return parsed

        return parsed.astimezone(timezone.utc).replace(tzinfo=None)

    return None


def parse_day(value: str) -> date:
    """Parse a ``--since``/``--until`` day boundary."""

    parsed = _parse_moment(value)

    if parsed is None:
        raise ValueError(
            f"Not a valid date: {value!r}. Use YYYY-MM-DD."
        )

    return parsed.date()


def merge_states(
    previous: CollectionState | None,
    current: CollectionState,
) -> CollectionState:
    """Add a resumed run's counts to the recorded totals."""

    if previous is None:
        return current

    return CollectionState(
        discovered=previous.discovered + current.discovered,
        persisted=previous.persisted + current.persisted,
        duplicates=previous.duplicates + current.duplicates,
        failed=previous.failed + current.failed,
        last_post_id=current.last_post_id
        or previous.last_post_id,
        last_url=current.last_url or previous.last_url,
        stopped_because=current.stopped_because,
        started_at=previous.started_at or current.started_at,
        updated_at=current.updated_at,
    )
