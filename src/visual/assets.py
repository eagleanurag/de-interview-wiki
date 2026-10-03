"""
Finding an image in the archive, safely, and describing it honestly.

Three things are decided here and nowhere else.

**Whether a file may be read at all.** Every path originates in a JSON
file the project did not write, so it is treated as hostile: resolved,
then checked to be inside the media root by comparing resolved paths
rather than by inspecting the string. That defeats ``..``, absolute
paths, drive letters, UNC paths and symlinks in one comparison, because
``Path.resolve`` has already followed the last two by the time the
comparison happens.

**What order the slides are in.** The archive names them
``{post}_slide_0``, ``_slide_01``, ``_slide_02`` and, in 65 posts, both
the one-digit and two-digit forms of the same carousel. Lexical sorting
puts ``_slide_100`` before ``_slide_20``, and mixing padding widths makes
that worse rather than better. So the sequence comes from the digits
parsed as an integer, and nothing else decides it.

**Whether the bytes are what the filename claims.** 536 of 3,047 files
are named ``.jpg`` and are PNG or GIF. Format and dimensions are read
from the content, so a mislabelled file is processed as what it is.

Nothing here writes, and the archive is opened read-only throughout.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from src.visual.models import (
    AssetRole,
    ProcessingState,
    VisualAsset,
)

#: The archive's own naming. The digits are the sequence and nothing
#: else is, because the padding is inconsistent across the archive.
SLIDE_NAME = re.compile(r"^(?P<stem>.+?)_slide_(?P<digits>\d+)$")

#: A document the archive saved alongside the images.
DOCUMENT_NAME = re.compile(r"^(?P<stem>.+)_original\.(?P<ext>pdf)$", re.I)

#: Content types recognised from magic bytes rather than extensions.
#: Checked in order, longest signature first.
_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"%PDF", "pdf"),
    (b"RIFF", "webp"),
)

_JPEG = b"\xff\xd8\xff"


class UnsafePath(ValueError):
    """
    A path that resolves outside the media root.

    A distinct type so containment failures are never mistaken for a
    missing file. They have opposite causes and opposite responses: a
    missing file is an archive that is incomplete, and an escaping path
    is an archive that is not to be trusted.
    """


def contained_media_path(media_root: Path, candidate: str | Path) -> Path:
    """
    Resolve a path from the archive and refuse it if it escapes.

    The comparison is between two resolved paths, which is the only form
    that catches all of the escapes at once. Rejecting ``..`` as a
    substring would miss a symlink, and checking the string starts with
    the root would miss both a symlink and a different drive.
    """

    root = Path(media_root).resolve()

    raw = Path(candidate)

    # An absolute path in the JSON is treated as relative to the root
    # first, because the archive records bare filenames and a future
    # export might record a full path from another machine.
    joined = raw if raw.is_absolute() else root / raw

    try:
        resolved = joined.resolve(strict=False)

    except (OSError, RuntimeError) as exc:
        raise UnsafePath(
            f"Could not resolve {candidate!r} inside the media root: {exc}"
        ) from exc

    try:
        resolved.relative_to(root)

    except ValueError as exc:
        raise UnsafePath(
            f"{candidate!r} resolves to {resolved}, which is outside "
            f"the media root {root}"
        ) from exc

    # Absence is deliberately not checked here. A file that is not there
    # is a gap in the archive; a file that resolves outside the root is
    # a path that must not be trusted, and they need different words and
    # different responses. Conflating them reported a missing slide as
    # a refusal, which points an operator at the security rule instead
    # of at the archive. The caller discovers absence by touching the
    # file, and reports it as absence.
    return resolved


def sniff_format(head: bytes) -> str:
    """
    The format implied by the first bytes.

    Returns ``""`` for anything unrecognised rather than guessing from
    the extension, so an unknown file is described as unknown instead of
    inheriting a claim the content does not support.
    """

    if head.startswith(_JPEG):
        return "jpeg"

    for signature, name in _MAGIC:
        if head.startswith(signature):
            if name == "webp" and head[8:12] != b"WEBP":
                continue

            return name

    stripped = head.lstrip()[:64].lower()

    if stripped.startswith(b"<?xml") or stripped.startswith(b"<svg"):
        return "svg"

    return ""


def digest_file(path: Path) -> str:
    """
    SHA-256 of a file's bytes, read in blocks.

    Content only, never the name. Two slides with identical pixels are
    the same slide whether they arrived in different posts, and content
    is what lets that be noticed.
    """

    hasher = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 20)

            if not block:
                break

            hasher.update(block)

    return hasher.hexdigest()


def describe_image(path: Path) -> tuple[str, int | None, int | None]:
    """
    Format, width and height, read from the content.

    Pillow is opened with a size guard and never asked to decode the
    pixels, so a decompression bomb cannot exhaust memory: the header
    alone gives the dimensions. ``Image.open`` is lazy, and the format
    comes from the plugin that matched the magic bytes.
    """

    from PIL import Image

    # Pillow's own bomb guard, raised rather than configured away.
    Image.MAX_IMAGE_PIXELS = 250_000_000

    try:
        with Image.open(path) as image:
            width, height = image.size

            detected = sniff_format(_leading_bytes(path)) or (
                image.format or ""
            ).lower()

    except Exception:
        # A file that will not open is a corrupt asset, reported as one
        # by the caller. It is never allowed to stop the post.
        return sniff_format(_leading_bytes(path)), None, None

    return detected, width, height


def classify_role(
    filename: str,
    declared_count: int,
    *,
    referenced: bool = True,
) -> AssetRole:
    """
    What an asset is in the post's own terms, from naming alone.

    Only the distinction that matters for work is drawn here: a
    standalone preview against a carousel slide. Whether an image is a
    diagram or a photograph is a question for the processor, which can
    see it; this stage cannot, and guessing would be dishonest.

    The rule is the archive's own: ``saved_files`` lists the carousel
    slides and omits the single ``_slide_0`` that arrived with the post
    before the carousel was saved. An unreferenced ``_slide_0`` is
    therefore a preview, and its role stays that way whatever it turns
    out to depict.
    """

    if DOCUMENT_NAME.match(filename):
        return AssetRole.DOCUMENT

    match = SLIDE_NAME.match(Path(filename).stem)

    if not match:
        return AssetRole.IMAGE

    index = int(match.group("digits"))

    if not referenced:
        return AssetRole.THUMBNAIL

    # A post whose only asset is numbered 0 is a single image, not a
    # carousel whose first slide happens to be zero.
    if declared_count <= 1:
        return AssetRole.IMAGE

    return AssetRole.SLIDE


def sequence_for(filename: str) -> int:
    """
    A slide's position, from the digits in its name.

    An unnumbered file sorts to the front at zero, so it cannot displace
    a numbered slide and the ordering stays total.
    """

    match = SLIDE_NAME.match(Path(filename).stem)

    if not match:
        return 0

    return int(match.group("digits"))


def build_asset(
    path: Path,
    media_root: Path,
    *,
    referenced: bool = True,
    declared_count: int = 1,
    sequence_count: int | None = None,
) -> VisualAsset:
    """
    One asset, described from its bytes.

    A file that cannot be read still produces an asset, in the ``failed``
    state with the reason recorded. Producing nothing would make an
    unreadable slide indistinguishable from one that was never there,
    which is the opposite of what a reader needs.
    """

    filename = path.name

    role = classify_role(
        filename, declared_count, referenced=referenced
    )

    asset = VisualAsset(
        path=f"media/{filename}",
        filename=filename,
        media_type="image",
        sequence=sequence_for(filename),
        sequence_count=sequence_count or declared_count or 1,
        role=role,
        declared_role=role,
    )

    try:
        resolved = contained_media_path(media_root, filename)

    except UnsafePath as exc:
        asset.state = ProcessingState.FAILED
        asset.note = f"Refused: {exc}"

        return asset

    try:
        byte_size = resolved.stat().st_size

    except OSError as exc:
        asset.state = ProcessingState.FAILED
        asset.note = f"Could not stat: {exc}"

        return asset

    asset.byte_size = byte_size

    try:
        asset.sha256 = digest_file(resolved)

    except OSError as exc:
        asset.state = ProcessingState.FAILED
        asset.note = f"Could not read: {exc}"

        return asset

    image_format, width, height = describe_image(resolved)

    asset.format = image_format
    asset.width = width
    asset.height = height

    if image_format == "pdf":
        asset.media_type = "pdf"
        asset.role = AssetRole.DOCUMENT

    elif image_format == "svg":
        # Recognised but never rendered here. An SVG can carry script,
        # and rasterising it is the visual stage's decision to make
        # under its own sandbox, not this one's to make silently.
        asset.media_type = "other"
        asset.role = AssetRole.IMAGE
        asset.note = (
            "SVG: recognised but not rasterised here; "
            "script is never executed"
        )

    elif not image_format or width is None:
        asset.state = ProcessingState.FAILED
        asset.note = (
            "Not a recognised image or document "
            f"(leading bytes: {_leading_bytes(resolved)!r})"
        )

    return asset


def _leading_bytes(path: Path, count: int = 8) -> bytes:
    """The first few bytes, for explaining why a file was refused."""

    try:
        with path.open("rb") as handle:
            return handle.read(count)

    except OSError:
        return b""


def order_assets(assets: list[VisualAsset]) -> list[VisualAsset]:
    """
    Assets in presentation order.

    Numeric on the sequence, then by filename so that two assets
    claiming the same position -- which happens when a post has both
    ``_slide_0`` and ``_slide_00`` -- still order identically between
    runs. Determinism here is not cosmetic: it is what makes a stored
    digest comparable.
    """

    return sorted(assets, key=lambda asset: (asset.sequence, asset.filename))


__all__ = [
    "UnsafePath",
    "build_asset",
    "classify_role",
    "contained_media_path",
    "describe_image",
    "digest_file",
    "order_assets",
    "sequence_for",
    "sniff_format",
]
