"""
The ingestion layer: turning a manual capture into a post.

A capture is whatever a person already has after reading an interview
post by hand: some text, a screenshot or two, sometimes a PDF of the
notes. This module turns that into the one committed shape the rest
of the pipeline understands::

    data/posts/<post_id>/post.json     authored content
    data/posts/<post_id>/media/*       media files

Everything is additive and idempotent. Re-running an import refreshes
the authored fields, leaves enrichment alone, and re-adding an
unchanged media file is a no-op, so a capture can be imported again
after one more screenshot is added without duplicating anything.

Whether the result is a post the pipeline can actually use is decided
by :mod:`src.ingestion.validation`, which is asked before anything is
written, and again by the test suite over the committed posts.

The layer is entirely local. Content is added by hand or by an
explicitly authorised process: there is deliberately no scraper here,
no network client, and nowhere to put a credential.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from src.ingestion.errors import (
    IngestionError,
    InvalidPostError,
    MediaConflictError,
    PostExistsError,
    PostNotFoundError,
    UnsupportedMediaError,
)
from src.ingestion.post_document import (
    CLASSIFICATION_KEYS,
    MEDIA_DIRECTORY_NAME,
    POST_FILE_NAME,
    MediaEntry,
    PostDocument,
    media_path_for,
    media_type_for,
    normalize_post_id,
)
from src.ingestion.validation import (
    LEVEL_ERROR,
    ValidationIssue,
    ValidationReport,
    media_files,
    validate_document,
    validate_post_directory,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_POSTS_ROOT = REPOSITORY_ROOT / "data" / "posts"

# Files that hold the capture's text rather than its media. They are
# never imported as media, so a bundle with both notes.md and a
# screenshot does not end up serving the notes as an attachment.
CAPTURE_TEXT_SUFFIXES = (".md", ".txt", ".markdown")

# Preferred capture text file names, in the order they are looked for.
CAPTURE_TEXT_NAMES = (
    "notes.md",
    "notes.txt",
    "post.md",
    "post.txt",
    "text.md",
    "text.txt",
    "transcript.md",
    "transcript.txt",
)

# Directories inside data/posts that are never posts.
IGNORED_DIRECTORY_PREFIXES = (".", "_")

_UNSAFE_NAME_CHARACTERS = re.compile(r"[^A-Za-z0-9._-]+")
_REPEATED_UNDERSCORES = re.compile(r"_{2,}")

MAX_HASH_CHUNK = 1024 * 1024


def posts_root(root: str | Path | None = None) -> Path:
    """The directory holding one subdirectory per post."""

    if root is None:
        return DEFAULT_POSTS_ROOT

    return Path(root)


def post_directory(
    post_id: str,
    root: str | Path | None = None,
) -> Path:
    """Where a post lives, without requiring that it exists."""

    return posts_root(root) / normalize_post_id(post_id)


# ---------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class PostSummary:
    """What discovery can tell about one post directory."""

    post_id: str
    directory: Path
    has_post_file: bool
    text_length: int = 0
    media_count: int = 0
    declared_media: int = 0

    @property
    def media_names(self) -> tuple[str, ...]:
        media_directory = (
            self.directory / MEDIA_DIRECTORY_NAME
        )

        if not media_directory.is_dir():
            return ()

        return tuple(
            sorted(
                path.name
                for path in media_directory.iterdir()
                if path.is_file()
            )
        )


def discover_posts(
    root: str | Path | None = None,
) -> list[PostSummary]:
    """
    List post directories in a stable order.

    This mirrors the discovery glob used by the pipeline, so what the
    importer sees is what a worker will see. A directory without a
    post.json is still reported, because it is either a mistake worth
    flagging or a post somebody started and has not finished.
    """

    base = posts_root(root)

    if not base.is_dir():
        return []

    summaries: list[PostSummary] = []

    for directory in sorted(base.iterdir()):
        if not directory.is_dir():
            continue

        name = directory.name

        if name.startswith(IGNORED_DIRECTORY_PREFIXES):
            continue

        post_file = directory / POST_FILE_NAME

        if not post_file.is_file():
            summaries.append(
                PostSummary(
                    post_id=name,
                    directory=directory,
                    has_post_file=False,
                )
            )
            continue

        try:
            document = PostDocument.load(directory)
        except IngestionError:
            # An unreadable post is reported by validation with a
            # precise message, rather than making discovery fail.
            summaries.append(
                PostSummary(
                    post_id=name,
                    directory=directory,
                    has_post_file=True,
                )
            )
            continue

        summaries.append(
            PostSummary(
                post_id=name,
                directory=directory,
                has_post_file=True,
                text_length=len(document.original_text),
                media_count=len(media_files(directory)),
                declared_media=len(document.media()),
            )
        )

    return summaries


# ---------------------------------------------------------------------
# Authoring
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class MediaImportResult:
    """What one media import actually changed."""

    post_id: str
    directory: Path
    added: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    renamed: tuple[tuple[str, str], ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.added)


@dataclass(frozen=True)
class PostImportResult:
    """What one post import actually changed."""

    post_id: str
    directory: Path
    created: bool
    text_from: str = ""
    media: MediaImportResult | None = None
    notes: tuple[str, ...] = ()

    @property
    def media_added(self) -> tuple[str, ...]:
        return self.media.added if self.media else ()


def create_post(
    post_id: str,
    *,
    text: str = "",
    root: str | Path | None = None,
    platform: str | None = None,
    url: str | None = None,
    author: str | None = None,
    captured_at: str | None = None,
    domain: str | None = None,
    primary_topic: str | None = None,
    secondary_topics: tuple[str, ...] = (),
    interview_relevant: bool | None = None,
    overwrite: bool = False,
    allow_empty: bool = False,
) -> PostDocument:
    """
    Create a new, valid post directory.

    The scaffold is a complete post.json, so the worker discovers the
    post and enriches it without any further editing. ``overwrite``
    exists for regenerating a scaffold, and refuses to silently
    destroy authored content otherwise.

    A post with neither text nor media is refused by default, because
    it would publish an empty page. ``allow_empty`` creates the shell
    for content that is still being captured, and the validation step
    keeps flagging it until the content lands.
    """

    directory = post_directory(post_id, root)
    identifier = normalize_post_id(post_id)

    if (directory / POST_FILE_NAME).exists() and not overwrite:
        raise PostExistsError(
            f"{identifier} already exists at {directory}. "
            "Use `add-media` or `import` to update it, or pass "
            "--overwrite to replace it."
        )

    document = PostDocument.new(
        identifier,
        text=text,
        **(
            _new_post_options(
                platform=platform,
                url=url,
                author=author,
                captured_at=captured_at,
                domain=domain,
                primary_topic=primary_topic,
                secondary_topics=secondary_topics,
                interview_relevant=interview_relevant,
            )
        ),
    )

    if not text.strip() and not allow_empty:
        raise InvalidPostError(
            f"{identifier} would have no content. Pass the captured "
            "text with --text or --text-file, or --allow-empty to "
            "create a shell and fill it in later."
        )

    _refuse_invalid(document, identifier, require_content=False)

    document.save(directory)

    return document


def add_media(
    post_id: str,
    sources: list[str | Path] | tuple[str | Path, ...],
    *,
    root: str | Path | None = None,
    description: str = "",
    force: bool = False,
) -> MediaImportResult:
    """
    Attach media files to an existing post.

    A source may be a file or a directory; a directory contributes its
    own files, ignoring hidden files and subdirectories, because a
    subdirectory of ``media/`` is enrichment output rather than a
    capture.

    Adding a file whose content is already present is a no-op, so this
    is safe to re-run. A file whose content differs is refused unless
    ``force`` is set, so a capture can never quietly replace a
    different image.
    """

    directory = post_directory(post_id, root)
    identifier = normalize_post_id(post_id)

    if not (directory / POST_FILE_NAME).is_file():
        raise PostNotFoundError(
            f"{identifier} has no {POST_FILE_NAME} at {directory}. "
            "Create the post first, for example with "
            f"`python -m src.ingestion.cli new {identifier}`."
        )

    document = PostDocument.load(directory)

    added, skipped, renamed = _ingest_media(
        document,
        directory,
        _expand_sources(sources),
        description=description,
        force=force,
    )

    document.save(directory)

    return MediaImportResult(
        post_id=identifier,
        directory=directory,
        added=tuple(added),
        skipped=tuple(skipped),
        renamed=tuple(renamed),
    )


def import_post(
    post_id: str,
    bundle: str | Path,
    *,
    root: str | Path | None = None,
    text: str | None = None,
    text_file: str | Path | None = None,
    platform: str | None = None,
    url: str | None = None,
    author: str | None = None,
    captured_at: str | None = None,
    domain: str | None = None,
    primary_topic: str | None = None,
    secondary_topics: tuple[str, ...] = (),
    interview_relevant: bool | None = None,
    description: str = "",
    force: bool = False,
) -> PostImportResult:
    """
    Import one manual capture as a post.

    A bundle is either a directory holding a capture or a single text
    file. A directory may contain its own ``post.json``, in which case
    that document is the base, so a hand-written post can be imported
    and enriched without editing it into the new shape first.

    The post is created when it does not exist and refreshed when it
    does. Enrichment output is never overwritten by an import.
    """

    source = Path(bundle)
    identifier = normalize_post_id(post_id)
    directory = post_directory(identifier, root)

    if not source.exists():
        raise IngestionError(
            f"capture bundle does not exist: {source}"
        )

    notes: list[str] = []
    created = not (directory / POST_FILE_NAME).is_file()

    document = _base_document(source, identifier, directory)

    if document.post_id != identifier:
        notes.append(
            f"capture declares id {document.post_id!r}; imported as "
            f"{identifier!r}"
        )

        # The post id given on the command line is the identity the
        # pipeline, the job ids and the generated site will use, so it
        # is adopted here rather than producing a post that validates
        # against nothing.
        document.post_id = identifier
        document.data["id"] = identifier

    text_from = _apply_text(
        document,
        text=text,
        text_file=text_file,
        bundle=source,
    )

    document.merge_source(
        platform=platform,
        url=url,
        author=author,
        captured_at=captured_at,
    )

    if domain or primary_topic or secondary_topics or (
        interview_relevant is not None
    ):
        _apply_classification(
            document,
            domain=domain,
            primary_topic=primary_topic,
            secondary_topics=secondary_topics,
            interview_relevant=interview_relevant,
        )

    if document.has_enrichment():
        notes.append(
            "existing enrichment was kept; re-run the worker to "
            "refresh it"
        )

    media_sources = _capture_media(source)

    if media_sources:
        added, skipped, renamed = _ingest_media(
            document,
            directory,
            media_sources,
            description=description,
            force=force,
        )

        media = MediaImportResult(
            post_id=identifier,
            directory=directory,
            added=tuple(added),
            skipped=tuple(skipped),
            renamed=tuple(renamed),
        )
    else:
        media = None

    _refuse_invalid(document, identifier)

    document.save(directory)

    return PostImportResult(
        post_id=identifier,
        directory=directory,
        created=created,
        text_from=text_from,
        media=media,
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------
# Validating a tree of posts
# ---------------------------------------------------------------------
def validate_posts(
    *,
    root: str | Path | None = None,
) -> ValidationReport:
    """
    Validate every post a worker would pick up.

    Errors describe posts that will fail, or silently degrade, in the
    pipeline. Warnings describe posts that will work but are not
    described as clearly as they could be, so they never fail a run.
    """

    base = posts_root(root)

    if not base.is_dir():
        raise IngestionError(
            f"posts directory does not exist: {base}"
        )

    posts = discover_posts(base)
    issues: list[ValidationIssue] = []

    if not posts:
        issues.append(
            ValidationIssue(
                level=LEVEL_ERROR,
                post_id="-",
                message=(
                    f"no posts found under {base}; the pipeline "
                    "requires at least one data/posts/<id>/post.json"
                ),
            )
        )

    for summary in posts:
        issues.extend(validate_post_directory(summary.directory))

    return ValidationReport(
        root=base,
        posts=tuple(posts),
        issues=tuple(issues),
    )


def validate_post(
    post_id: str,
    *,
    root: str | Path | None = None,
) -> list[ValidationIssue]:
    """Validate one post directory."""

    return validate_post_directory(
        post_directory(post_id, root)
    )


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------


def _new_post_options(
    *,
    platform: str | None,
    url: str | None,
    author: str | None,
    captured_at: str | None,
    domain: str | None,
    primary_topic: str | None,
    secondary_topics: tuple[str, ...],
    interview_relevant: bool | None,
) -> dict:
    """Forward only the options the caller actually supplied."""

    options: dict = {}

    if platform is not None:
        options["platform"] = platform

    if url is not None:
        options["url"] = url

    if author is not None:
        options["author"] = author

    if captured_at is not None:
        options["captured_at"] = captured_at

    if domain is not None:
        options["domain"] = domain

    if primary_topic is not None:
        options["primary_topic"] = primary_topic

    if secondary_topics:
        options["secondary_topics"] = tuple(secondary_topics)

    if interview_relevant is not None:
        options["interview_relevant"] = interview_relevant

    return options


def _base_document(
    bundle: Path,
    post_id: str,
    directory: Path,
) -> PostDocument:
    """
    The document an import starts from.

    A bundle may carry its own post.json, which wins, so a hand-written
    capture is imported exactly as authored. Otherwise an existing post
    is refreshed in place rather than replaced, which is what keeps
    enrichment and earlier media declarations across re-imports. A
    genuinely new post starts from a fresh document.
    """

    if bundle.is_dir():
        bundled = bundle / POST_FILE_NAME

        if bundled.is_file():
            return PostDocument.load_file(bundled)

    if (directory / POST_FILE_NAME).is_file():
        return PostDocument.load(directory)

    return PostDocument.new(post_id)


def _apply_text(
    document: PostDocument,
    *,
    text: str | None,
    text_file: str | Path | None,
    bundle: Path,
) -> str:
    """
    Set the authored text and report where it came from.

    Explicit text wins over a bundled notes file, and a bundled notes
    file wins over whatever the post already contained, because the
    capture is the fresher copy of what was read.
    """

    if text is not None:
        document.set_original_text(text)
        return "--text"

    if text_file:
        path = Path(text_file)

        if not path.is_file():
            raise IngestionError(f"text file does not exist: {path}")

        document.set_original_text(
            path.read_text(encoding="utf-8-sig").strip()
        )

        return path.as_posix()

    found = _find_capture_text(bundle)

    if found is not None:
        document.set_original_text(
            found.read_text(encoding="utf-8-sig").strip()
        )

        return found.as_posix()

    return ""


def _find_capture_text(bundle: Path) -> Path | None:
    """
    Locate the capture's text file.

    The well-known names win. Otherwise a single text file is used, so
    a bundle named `captured.md` still imports, while a bundle with
    several unrelated text files is left for the human to name
    explicitly.
    """

    if bundle.is_file():
        return bundle if _is_capture_text(bundle) else None

    if not bundle.is_dir():
        return None

    for name in CAPTURE_TEXT_NAMES:
        candidate = bundle / name

        if candidate.is_file():
            return candidate

    candidates = sorted(
        path
        for path in bundle.iterdir()
        if path.is_file() and _is_capture_text(path)
    )

    if len(candidates) == 1:
        return candidates[0]

    return None


def _is_capture_text(path: Path) -> bool:
    return path.suffix.lower() in CAPTURE_TEXT_SUFFIXES


def _capture_media(bundle: Path) -> list[Path]:
    """
    The media files inside a capture bundle.

    Text files and the bundle's own post.json are content rather than
    attachments, and hidden files are machine noise.
    """

    if not bundle.is_dir():
        return []

    media: list[Path] = []

    for path in sorted(bundle.iterdir()):
        if not path.is_file():
            continue

        if path.name.startswith("."):
            continue

        if path.name == POST_FILE_NAME or _is_capture_text(path):
            continue

        media.append(path)

    return media


def _expand_sources(
    sources: list[str | Path] | tuple[str | Path, ...],
) -> list[Path]:
    """
    Flatten files and directories into a list of media files.

    A directory is expanded the same way a capture bundle is, so
    pointing at a capture directory never turns its notes into an
    attachment. A file named explicitly is always taken literally,
    because that is an unambiguous instruction.
    """

    if not sources:
        raise IngestionError("no media source was given")

    expanded: list[Path] = []

    for source in sources:
        path = Path(source)

        if path.is_dir():
            expanded.extend(_capture_media(path))
            continue

        if not path.is_file():
            raise IngestionError(
                f"media source does not exist: {path}"
            )

        expanded.append(path)

    return expanded


def _ingest_media(
    document: PostDocument,
    directory: Path,
    sources: list[Path],
    *,
    description: str,
    force: bool,
) -> tuple[list[str], list[str], list[tuple[str, str]]]:
    """Copy media into the post and declare it. Shared by both paths."""

    media_directory = directory / MEDIA_DIRECTORY_NAME

    added: list[str] = []
    skipped: list[str] = []
    renamed: list[tuple[str, str]] = []

    for source in sources:
        name = safe_media_name(source.name)

        if name != source.name:
            renamed.append((source.name, name))

        target = media_directory / name

        if target.exists():
            if _same_content(source, target):
                skipped.append(name)
                continue

            if not force:
                raise MediaConflictError(
                    f"{name} already exists in {directory.name} with "
                    "different content. Pass --force to replace it."
                )

        media_directory.mkdir(parents=True, exist_ok=True)

        shutil.copy2(source, target)

        added.append(name)

        document.upsert_media(
            MediaEntry(
                type=media_type_for(name),
                path=media_path_for(name),
                description=description.strip(),
            )
        )

    return added, skipped, renamed


def safe_media_name(name: str) -> str:
    """
    Reduce a captured file name to a portable, committable name.

    A screenshot called ``Screenshot 2026-01-01 at 10.15.42.png`` is
    renamed rather than refused, because refusing it would make the
    import unusable, but the final name is always reported so the
    rename is visible.
    """

    base = Path(name).name.strip()

    if not base:
        raise UnsupportedMediaError(
            "media file has no usable name"
        )

    cleaned = _UNSAFE_NAME_CHARACTERS.sub("_", base)
    cleaned = _REPEATED_UNDERSCORES.sub("_", cleaned)
    cleaned = cleaned.strip("._-")

    if not cleaned:
        raise UnsupportedMediaError(
            f"media file name has nothing portable left: {name!r}"
        )

    return cleaned


def _same_content(left: Path, right: Path) -> bool:
    try:
        if left.stat().st_size != right.stat().st_size:
            return False
    except OSError:
        return False

    return _digest(left) == _digest(right)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(MAX_HASH_CHUNK), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _apply_classification(
    document: PostDocument,
    *,
    domain: str | None,
    primary_topic: str | None,
    secondary_topics: tuple[str, ...],
    interview_relevant: bool | None,
) -> None:
    """Update classification without discarding other topics."""

    existing = document.data.get("classification")

    if not isinstance(existing, dict):
        existing = {
            "domain": "Data Engineering",
            "primary_topic": None,
            "secondary_topics": [],
            "interview_relevant": False,
        }

    if domain is not None:
        existing["domain"] = domain

    if primary_topic is not None:
        existing["primary_topic"] = primary_topic or None

    if secondary_topics:
        merged = [
            topic
            for topic in existing.get("secondary_topics", [])
            if isinstance(topic, str)
        ]

        for topic in secondary_topics:
            if topic and topic not in merged:
                merged.append(topic)

        existing["secondary_topics"] = merged

    if interview_relevant is not None:
        existing["interview_relevant"] = bool(interview_relevant)

    document.data["classification"] = {
        key: existing.get(key) for key in CLASSIFICATION_KEYS
    }


def _refuse_invalid(
    document: PostDocument,
    post_id: str,
    *,
    require_content: bool = True,
) -> None:
    """
    Refuse to write a post that is already broken.

    The check runs before anything is written, so a rejected import
    leaves the posts tree exactly as it was rather than half-ingested.
    """

    issues = [
        issue
        for issue in validate_document(
            document,
            post_id=post_id,
            require_content=require_content,
        )
        if issue.is_error
    ]

    if not issues:
        return

    details = "; ".join(issue.message for issue in issues[:5])

    raise InvalidPostError(
        f"{post_id} is not a usable post: {details}"
    )


