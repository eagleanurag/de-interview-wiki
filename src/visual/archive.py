"""
Reading the archive, without trusting it.

``posts_archive.json`` is a file this project did not write, describing
files on a machine this project does not own. Everything it says is
therefore treated as a claim to be checked rather than an instruction to
be followed.

Three properties hold throughout:

* the archive is opened read-only and never written, renamed or moved;
* every path it names is resolved and confirmed to sit inside the media
  root before being opened;
* the browser session directory is never read, listed or referenced. It
  sits beside the media and is not part of this system's inputs, and
  ``FORBIDDEN`` below exists so that a future edit which starts reaching
  for it fails loudly instead of quietly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.visual.assets import (
    UnsafePath,
    build_asset,
    contained_media_path,
    order_assets,
)

#: Directory names that must never be read from. The archive keeps a
#: Chrome profile beside the media, and nothing in this project has any
#: reason to open it.
#:
#: Note what this list is *not*: a reason to refuse the archive. Those
#: directories sitting beside the media is simply how this archive is
#: laid out, and refusing to process 3,047 images because an off-limits
#: folder happens to share a parent directory would make the stage
#: impossible to run for a reason that has nothing to do with the media.
#: The rule is about what is *opened*, which is enforced on every path
#: at the point of use by ``contained_media_path``.
FORBIDDEN_DIRECTORIES = {
    "chrome_session",
    "User Data",
    "Default",
    "Profile 1",
}

#: Individual files that must never be read, wherever they are found.
FORBIDDEN_FILES = {
    "Cookies",
    "Login Data",
    "Web Data",
    "Local State",
    "History",
    "Bookmarks",
}


def assert_readable(target: Path) -> Path:
    """
    Refuse a path that names browser state.

    Applied at the point of use rather than at the archive root, so the
    check is on the thing about to be opened. Returns the path so it can
    wrap a resolution.
    """

    parts = {part.lower() for part in Path(target).parts}
    name = Path(target).name

    if parts & {entry.lower() for entry in FORBIDDEN_DIRECTORIES}:
        raise ArchiveError(
            f"{target} is inside browser state, which this system does "
            "not read. Media and metadata only."
        )

    if name in FORBIDDEN_FILES:
        raise ArchiveError(
            f"{target} is a browser profile file, which this system does "
            "not read. Media and metadata only."
        )

    return Path(target)


class ArchiveError(RuntimeError):
    """The archive could not be read as a whole."""


@dataclass(frozen=True)
class ArchivePost:
    """One record, with the parts this stage needs."""

    post_id: str
    text: str
    permalink: str | None
    saved_files: tuple[str, ...]
    declared_count: int
    is_multi_slide: bool
    original_urls: tuple[str, ...] = ()

    @property
    def has_media(self) -> bool:
        return bool(self.saved_files)


@dataclass
class Archive:
    """
    The archive, indexed by post.

    Constructed from the metadata file alone. Media is described lazily,
    per post, because describing 3,047 files hashes 287 MB and there is
    no reason to do that for a post whose enrichment is already current.
    """

    root: Path
    posts: dict[str, ArchivePost] = field(default_factory=dict)
    #: Filenames present on disk but referenced by no record.
    orphans: tuple[str, ...] = ()

    @property
    def media_root(self) -> Path:
        return self.root / "media"

    @property
    def metadata_path(self) -> Path:
        return self.root / "posts_archive.json"

    def post_ids_with_media(self) -> list[str]:
        """Posts that have at least one saved file, in a stable order."""

        return sorted(
            post_id
            for post_id, post in self.posts.items()
            if post.has_media
        )


def _check_not_session(media_root: Path) -> None:
    """
    Confirm the media directory is not itself browser state.

    The archive root is allowed to contain a ``chrome_session`` folder --
    it does, and that is its normal layout. What must not happen is being
    pointed *at* one of those folders as though it were the media, which
    is the mistake this catches.
    """

    resolved = media_root.resolve()

    for part in resolved.parts:
        if part.lower() in {
            entry.lower() for entry in FORBIDDEN_DIRECTORIES
        }:
            raise ArchiveError(
                f"The media directory resolves inside browser state "
                f"({part}). This system reads media and metadata only, so "
                "the archive is almost certainly pointed at the wrong "
                "directory."
            )

    assert_readable(resolved)


def load_archive(root: str | Path) -> Archive:
    """
    Read ``posts_archive.json``.

    A missing or malformed file is an error rather than an empty
    archive, because "the archive is empty" and "the archive could not be
    read" lead to opposite decisions about whether to delete anything.
    """

    base = Path(root).expanduser()

    if not base.is_dir():
        raise ArchiveError(f"Archive root does not exist: {base}")

    metadata = base / "posts_archive.json"

    if not metadata.is_file():
        raise ArchiveError(f"Archive metadata is missing: {metadata}")

    media = base / "media"

    if not media.is_dir():
        raise ArchiveError(f"Archive media directory is missing: {media}")

    _check_not_session(media)

    try:
        payload = json.loads(metadata.read_text(encoding="utf-8"))

    except (OSError, json.JSONDecodeError) as exc:
        raise ArchiveError(f"Could not read {metadata}: {exc}") from exc

    if isinstance(payload, dict):
        records = (
            payload.get("posts")
            or payload.get("records")
            or payload.get("items")
            or []
        )

    elif isinstance(payload, list):
        records = payload

    else:
        raise ArchiveError(
            f"{metadata} holds a {type(payload).__name__}, not a list of "
            "records"
        )

    posts: dict[str, ArchivePost] = {}
    referenced: set[str] = set()
    duplicates: list[str] = []

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue

        post_id = str(
            record.get("post_id") or record.get("id") or ""
        ).strip()

        if not post_id:
            continue

        if post_id in posts:
            # Recorded rather than merged. Two records for one post is
            # an archive problem, and silently taking the second would
            # hide it while quietly choosing which version is truth.
            duplicates.append(post_id)
            continue

        media_record = record.get("media") or {}

        if not isinstance(media_record, dict):
            media_record = {}

        saved = tuple(
            str(name)
            for name in (media_record.get("saved_files") or [])
            if str(name).strip()
        )

        referenced.update(saved)

        declared = media_record.get("total_media_count")

        posts[post_id] = ArchivePost(
            post_id=post_id,
            text=str(record.get("text") or ""),
            permalink=(
                str(record.get("permalink"))
                if record.get("permalink")
                else None
            ),
            saved_files=saved,
            declared_count=(
                int(declared) if isinstance(declared, int) else len(saved)
            ),
            is_multi_slide=bool(media_record.get("is_multi_slide")),
            original_urls=tuple(
                str(url)
                for url in (media_record.get("original_urls") or [])
                if str(url).strip()
            ),
        )

    if duplicates:
        raise ArchiveError(
            f"{len(duplicates)} duplicate post id(s) in {metadata}, "
            f"the first being {duplicates[0]}"
        )

    try:
        on_disk = {path.name for path in media.iterdir() if path.is_file()}

    except OSError as exc:
        raise ArchiveError(f"Could not list {media}: {exc}") from exc

    return Archive(
        root=base,
        posts=posts,
        orphans=tuple(sorted(on_disk - referenced)),
    )


def assets_for_post(archive: Archive, post_id: str):
    """
    Every file belonging to one post, described and ordered.

    Includes files the metadata does not reference. The archive omits 65
    low-resolution previews, and those are real files on disk belonging
    to the post -- seeing them is what allows them to be recognised as
    previews rather than as missing slides.
    """

    from src.visual.assets import SLIDE_NAME

    post = archive.posts.get(post_id)

    if post is None:
        return []

    media_root = archive.media_root
    referenced = set(post.saved_files)

    # Anything on disk whose stem matches this post's stem, plus
    # whatever the metadata names explicitly.
    candidates: dict[str, bool] = {}

    for name in sorted(referenced):
        candidates[name] = True

    try:
        entries = sorted(media_root.iterdir())

    except OSError:
        entries = []

    for entry in entries:
        if not entry.is_file():
            continue

        stem = SLIDE_NAME.match(entry.stem)

        if stem and stem.group("stem") == post_id:
            candidates[entry.name] = entry.name in referenced

    assets = []

    for name in sorted(candidates):
        is_referenced = candidates[name]

        try:
            path = contained_media_path(media_root, name)

        except UnsafePath:
            # Recorded as a failed asset rather than dropped, so a
            # refused path is visible instead of looking like a post
            # with fewer images.
            from src.visual.models import (
                AssetRole,
                ProcessingState,
                VisualAsset,
            )

            assets.append(
                VisualAsset(
                    path=f"media/{name}",
                    filename=name,
                    state=ProcessingState.FAILED,
                    role=AssetRole.UNKNOWN,
                    note="Refused: the path does not resolve inside the "
                    "media root",
                )
            )

            continue

        assets.append(
            build_asset(
                path,
                media_root,
                referenced=is_referenced,
                declared_count=max(post.declared_count, len(candidates)),
                sequence_count=len(candidates),
            )
        )

    return order_assets(assets)


def resolve_asset_path(archive: Archive, asset) -> Path:
    """
    The file on disk for one asset, re-checked at the point of use.

    Containment and the browser-state rule are both applied here rather
    than trusted from discovery. Between discovery and use there is a
    model call, and a path from the archive is never trusted because it
    was checked once.
    """

    from src.visual.assets import UnsafePath

    try:
        resolved = contained_media_path(archive.media_root, asset.filename)

    except UnsafePath as exc:
        raise ArchiveError(
            f"{asset.filename}: {exc}"
        ) from exc

    return assert_readable(resolved)


__all__ = [
    "FORBIDDEN_DIRECTORIES",
    "FORBIDDEN_FILES",
    "Archive",
    "ArchiveError",
    "ArchivePost",
    "assert_readable",
    "assets_for_post",
    "load_archive",
    "resolve_asset_path",
]
