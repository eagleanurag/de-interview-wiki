"""
Checking the package against the real archive.

Two sources with different authority, and the asymmetry is the whole
point of this module.

**The archive decides what exists.** Whether a file is there, what its
bytes hash to, how big it is, what shape it is, and which activity it
belongs to. Nothing the package says can change any of that, and a
disagreement means the package is wrong about something objective --
which is worth recording rather than correcting in place.

**The package decides only what it read out of a picture.** That an
image contains the words "USE THE S.T.A.R TECHNIQUE!" is a claim about
pixels, and no amount of checking bytes will confirm or refute it. It
is stored as the package's claim, attributed to the package.

The archive is opened read-only. Nothing here writes, renames or moves
anything, and no path from the archive's own metadata is followed
without being resolved and confirmed to sit inside the media root --
the same containment rule CP12 applies, for the same reason.

Every disagreement lands in the report. A record whose digest disagrees
with the file on disk is reported as conflicting and kept, because
"Gemini hashed a different version of this image" is a real possibility
and the record is still useful when it is visible.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from src.gemini.models import (
    CrossCheckReport,
    GeminiImageRecord,
    MatchVerdict,
)


#: The archive's own naming, reused rather than restated so that a change
#: to the convention cannot leave two parsers disagreeing.
SLIDE_NAME = re.compile(r"^(?P<stem>.+?)_slide_(?P<digits>\d+)$")


class ArchiveError(RuntimeError):
    """The archive could not be read as a whole."""


def _leading_bytes(path: Path, count: int = 8) -> bytes:
    """The first few bytes, for saying why a file was refused."""

    try:
        with path.open("rb") as handle:
            return handle.read(count)

    except OSError:
        return b""


def contained(media_root: Path, candidate: str | Path) -> Path:
    """
    Resolve a filename inside the media root, or refuse it.

    The archive's metadata is a file this project did not write, so its
    paths are data. Comparing two *resolved* paths is what defeats
    ``..``, absolute paths, drive letters, UNC paths and symlinks in one
    step, because resolution has already followed the last two.
    """

    root = Path(media_root).resolve()

    raw = Path(candidate)

    joined = raw if raw.is_absolute() else root / raw

    try:
        resolved = joined.resolve(strict=False)

    except (OSError, RuntimeError) as exc:
        raise ArchiveError(
            f"could not resolve {candidate!r}: {exc}"
        ) from exc

    try:
        resolved.relative_to(root)

    except ValueError as exc:
        raise ArchiveError(
            f"{candidate!r} resolves to {resolved}, outside {root}"
        ) from exc

    return resolved


def load_archive_posts(archive_root: str | Path) -> dict[str, dict]:
    """
    The archive's own records, keyed by post identifier.

    Read-only. A malformed or missing file raises rather than returning
    an empty mapping, because "the archive is empty" and "the archive
    could not be read" lead to opposite decisions.
    """

    base = Path(archive_root).expanduser()

    metadata = base / "posts_archive.json"

    if not metadata.is_file():
        raise ArchiveError(f"archive metadata is missing: {metadata}")

    if not (base / "media").is_dir():
        raise ArchiveError(f"archive media directory is missing: {base / 'media'}")

    try:
        payload = json.loads(metadata.read_text(encoding="utf-8"))

    except (OSError, json.JSONDecodeError) as exc:
        raise ArchiveError(f"could not read {metadata}: {exc}") from exc

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
            f"{metadata} holds a {type(payload).__name__}, not a list"
        )

    posts: dict[str, dict] = {}

    for record in records:
        if not isinstance(record, dict):
            continue

        post_id = str(record.get("post_id") or record.get("id") or "").strip()

        if not post_id or post_id in posts:
            continue

        media = record.get("media") or {}

        if not isinstance(media, dict):
            media = {}

        posts[post_id] = {
            "post_id": post_id,
            "text": str(record.get("text") or ""),
            "media": media,
            "saved_files": [
                str(name)
                for name in (media.get("saved_files") or [])
                if str(name).strip()
            ],
        }

    return posts


#: Filename stems in this archive are ``activity_<id>`` or
#: ``urn_li_activity_<id>``. The package's own ``activity_id`` column
#: carries the same identifier with no prefix at all, and the archive's
#: ``post_id`` is the prefixed form again.
#:
#: Measured, not assumed: the first row of the package inventory gives
#: ``activity_id`` ``7263033731471351808`` against filename
#: ``activity_7263033731471351808_slide_0.jpg``, and 315 of the 485
#: archive records key on ``activity_<id>``. Comparing those spellings
#: directly -- which an earlier version of this module did -- reports
#: every record as conflicting and every group as missing, which is a
#: finding about a prefix rather than about the data.
ACTIVITY_PREFIXES = ("urn_li_activity_", "activity_")


def activity_id_for_filename(filename: str) -> str | None:
    """
    The activity a media filename belongs to, in canonical bare form.

    Bare, because that is the form the package's own column uses and the
    form two independent sources agree on. Prefixes are stripped rather
    than compared, so a cross-check compares identifiers rather than
    spellings.

    None for a file whose stem carries no activity at all, rather than
    the stem itself: an unrecognised name is not an activity, and
    returning the raw stem would let it be compared against real ones.
    """

    stem = Path(str(filename)).stem

    match = SLIDE_NAME.match(stem)

    candidate = match.group("stem") if match else stem

    for prefix in ACTIVITY_PREFIXES:
        if candidate.startswith(prefix):
            return candidate[len(prefix):] or None

    return None


def archive_post_key(activity_id: str) -> str:
    """
    The archive's own ``post_id`` spelling for an activity.

    The inverse of :func:`activity_id_for_filename`, and needed because
    the archive's keys are prefixed. Used only for looking an activity up
    in the archive; the prefix is never written into a record.
    """

    text = str(activity_id or "").strip()

    if not text:
        return ""

    for prefix in ACTIVITY_PREFIXES:
        if text.startswith(prefix):
            return text

    return f"activity_{text}"

def slide_number_for_filename(filename: str) -> int | None:
    """The slide number in a filename, as an integer."""

    match = SLIDE_NAME.match(Path(str(filename)).stem)

    if not match:
        return None

    return int(match.group("digits"))


def index_archive(archive_root: str | Path) -> dict[str, dict]:
    """
    Every media file on disk, described from its bytes.

    Keyed by filename. The digests are taken here rather than trusted
    from the package so that the cross-check compares two independent
    measurements instead of the package against itself.
    """

    media_root = Path(archive_root).expanduser() / "media"

    if not media_root.is_dir():
        raise ArchiveError(f"media directory is missing: {media_root}")

    index: dict[str, dict] = {}

    for path in sorted(media_root.iterdir()):
        if not path.is_file():
            continue

        hasher = hashlib.sha256()

        with path.open("rb") as handle:
            while True:
                block = handle.read(1 << 20)

                if not block:
                    break

                hasher.update(block)

        width = height = None
        detected = ""

        try:
            from PIL import Image

            Image.MAX_IMAGE_PIXELS = 250_000_000

            with Image.open(path) as image:
                width, height = image.size
                detected = (image.format or "").lower()

        except Exception:
            # Unreadable is a finding, not a reason to stop indexing.
            detected = ""

        index[path.name] = {
            "filename": path.name,
            "sha256": hasher.hexdigest(),
            "size_bytes": path.stat().st_size,
            "width": width,
            "height": height,
            "format": detected,
            "leading_bytes": _leading_bytes(path),
        }

    return index


def archive_digest(index: dict[str, dict]) -> str:
    """
    One digest over the archive's filenames and contents.

    Over file *identity*, not metadata: a re-exported archive whose
    files are byte-identical has not changed what the package describes,
    and treating it as changed would force a full re-import for nothing.
    """

    digest = hashlib.sha256()

    for name in sorted(index):
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(index[name]["sha256"].encode("ascii"))

    return digest.hexdigest()


def cross_check(
    records: list[GeminiImageRecord],
    archive_index: dict[str, dict],
    archive_posts: dict[str, dict] | None = None,
) -> tuple[list[GeminiImageRecord], CrossCheckReport]:
    """
    Compare every record against the archive, and say where they differ.

    Returns the records with a verdict attached and the report. Records
    are never dropped and never edited: a conflicting record is kept as
    it arrived, with the disagreement written beside it, because a
    corrected record would hide the only evidence that the two sources
    ever disagreed.
    """

    archive_posts = archive_posts or {}

    report = CrossCheckReport(records_total=len(records))

    known_ids = set(archive_posts)

    referenced = {
        name
        for post in archive_posts.values()
        for name in post["saved_files"]
    }

    checked: list[GeminiImageRecord] = []

    for record in records:
        filename = record.filename

        entry = archive_index.get(filename)

        if entry is None:
            # The package names a file the archive does not have.
            # Reported, and kept: it may be a file added since the
            # package was produced, and either way the package's claim
            # about it cannot be checked.
            report.missing_from_archive += 1

            checked.append(
                record.model_copy(
                    update={
                        "verdict": MatchVerdict.MISSING_FROM_ARCHIVE,
                        "verdict_note": (
                            "the archive holds no file with this name, so "
                            "the package's transcription cannot be "
                            "verified against anything"
                        ),
                    }
                )
            )

            continue

        problems: list[str] = []

        record_sha = entry["sha256"]

        if record.declared_sha256 and record.declared_sha256 != record_sha:
            problems.append(
                "the package's digest differs from the archive's"
            )

            report.sha256_mismatches.append(
                f"{filename}: package {record.declared_sha256[:12]} "
                f"vs archive {record_sha[:12]}"
            )

        if (
            record.size_bytes
            and entry["size_bytes"]
            and record.size_bytes != entry["size_bytes"]
        ):
            problems.append("the byte size differs from the archive's")

        if (
            record.width
            and entry["width"]
            and record.height
            and entry["height"]
            and (record.width, record.height) != (entry["width"], entry["height"])
        ):
            problems.append("the dimensions differ from the archive's")

            report.dimension_mismatches.append(
                f"{filename}: package {record.width}x{record.height} "
                f"vs archive {entry['width']}x{entry['height']}"
            )

        derived = activity_id_for_filename(filename)

        # Both sides in canonical bare form: the package's column is bare
        # and ``activity_id_for_filename`` strips the filename prefix.
        if record.activity_id and derived and record.activity_id != derived:
            problems.append(
                "the activity id does not match the one in the filename"
            )

        if known_ids:
            # Gemini's own grouping id is not expected to appear in the
            # archive, which keys on activity ids. What must hold is that
            # the activity the filename names exists -- looked up in the
            # archive's own prefixed spelling.
            key = archive_post_key(derived) if derived else ""

            if key and key not in known_ids:
                problems.append(
                    "the activity named by the filename is not in the "
                    "archive"
                )

        if record.exact_duplicate_of and record.exact_duplicate_of not in archive_index:
            problems.append("the duplicate target names no file on disk")

            report.dangling_duplicate_targets.append(
                f"{filename} -> {record.exact_duplicate_of}"
            )

        if record.likely_preview_of and record.likely_preview_of not in archive_index:
            problems.append("the preview target names no file on disk")

            report.dangling_preview_targets.append(
                f"{filename} -> {record.likely_preview_of}"
            )

        if problems:
            report.conflicting += 1

            checked.append(
                record.model_copy(
                    update={
                        "actual_sha256": record_sha,
                        "verdict": MatchVerdict.CONFLICTING,
                        "verdict_note": "; ".join(problems),
                    }
                )
            )

            continue

        report.matched += 1

        checked.append(
            record.model_copy(
                update={"actual_sha256": record_sha, "verdict": MatchVerdict.MATCHED}
            )
        )

    # Two different gaps, and conflating them would hide one.

    # Files on disk that the package never mentioned. Not an error: the
    # package is a third party's work and may legitimately have skipped
    # something. Recorded so the gap is visible rather than assumed
    # away.
    package_filenames = {record.filename for record in records}

    absent = sorted(set(archive_index) - package_filenames)

    report.archive_files_absent_from_package = len(absent)
    report.absent_filenames = absent[:200]

    # Files the archive's own metadata does not reference -- the
    # low-resolution previews, which is a fact about the archive rather
    # than about the package. Kept as a note because it explains why the
    # two file sets differ.
    unreferenced = sorted(set(archive_index) - referenced)

    if unreferenced:
        report.notes.append(
            f"{len(unreferenced)} file(s) on disk are not referenced by "
            "the archive's own metadata; they are previews that arrived "
            "before the carousel was saved, and the package includes them"
        )

    return checked, report


def group_verdicts(
    groups: dict,
    records: list[GeminiImageRecord],
    archive_posts: dict[str, dict],
) -> dict:
    """
    Attach a verdict to each group.

    A group's slides are its filenames. Where the package's declared
    count disagrees with the rows that carry its identifier, that is
    recorded -- it means the markdown summary and the inventory describe
    different things, and which is right is not settled here.
    """

    by_group: dict[str, list[GeminiImageRecord]] = {}

    for record in records:
        by_group.setdefault(record.post_id, []).append(record)

    result: dict = {}

    for group_id, group in groups.items():
        members = by_group.get(group_id, [])

        notes: list[str] = []

        verdict = MatchVerdict.MATCHED

        if group.declared_slide_count and members:
            if len(members) != group.declared_slide_count:
                notes.append(
                    f"the summary claims {group.declared_slide_count} "
                    f"slide(s) but {len(members)} record(s) carry this "
                    "group id"
                )

                verdict = MatchVerdict.CONFLICTING

        activity = group.activity_id or (
            members[0].activity_id if members else ""
        )

        if (
            activity
            and archive_posts
            and archive_post_key(activity) not in archive_posts
        ):
            notes.append(
                f"the activity {activity} is not in the archive's posts"
            )

            verdict = MatchVerdict.MISSING_FROM_ARCHIVE

        absent = [
            record.filename
            for record in members
            if record.verdict is MatchVerdict.MISSING_FROM_ARCHIVE
        ]

        if absent and verdict is MatchVerdict.MATCHED:
            verdict = MatchVerdict.MISSING_FROM_ARCHIVE

        result[group_id] = group.model_copy(
            update={
                "verdict": verdict,
                "verdict_note": "; ".join(notes),
                "actual_slide_count": len(members),
            }
        )

    return result


__all__ = [
    "ArchiveError",
    "activity_id_for_filename",
    "archive_digest",
    "contained",
    "cross_check",
    "group_verdicts",
    "index_archive",
    "load_archive_posts",
    "slide_number_for_filename",
]
