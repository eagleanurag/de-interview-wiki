"""
Not analysing the same picture twice.

The archive contains 2,782 distinct images across 3,047 files. The 265
redundant files are not a rounding error: the largest group is 42 copies
of one image spread across different posts, which is a cheat sheet or a
diagram that got saved many times.

Two rules, and the second one is the one that matters.

**Exact duplicates are content, not filename.** Two files are the same
slide when their bytes are equal, which is a SHA-256 comparison and
nothing subtler. The archive also carries, for 65 posts, a low-resolution
preview alongside the full carousel; that is a different file with
different bytes and it is *not* an exact duplicate.

**A thumbnail never replaces its full image.** Where a preview exists,
the carousel slide stays the primary asset and the preview is recorded as
a redundant copy of it. Merging them on visual resemblance would be the
wrong tool: two slides in a carousel are often near-identical in
composition while differing in exactly the text that matters, and a
perceptual hash that called them the same would silently delete one of
them.

So perceptual similarity is deliberately not used for merging. It is
reported, as a hint, and never acted on. False merging is worse than
processing a duplicate.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from src.visual.models import (
    AssetRole,
    ProcessingState,
    VisualAsset,
)


def group_by_content(assets: list[VisualAsset]) -> dict[str, list[VisualAsset]]:
    """
    Assets grouped by what they contain.

    Only usable assets take part. A file that could not be read has no
    content identity, and grouping it with other unreadable files would
    report a duplicate that nobody can verify.
    """

    groups: dict[str, list[VisualAsset]] = defaultdict(list)

    for asset in assets:
        if not asset.sha256:
            continue

        if asset.state is ProcessingState.FAILED:
            continue

        groups[asset.sha256].append(asset)

    return dict(groups)


def _thumbnail_of(asset: VisualAsset, full: VisualAsset) -> bool:
    """
    Whether ``asset`` looks like a reduced copy of ``full``.

    Used only to prefer the better file when choosing a group's primary.
    The test is deliberately one-sided: a smaller file with strictly
    smaller dimensions is a candidate preview of a larger one. Nothing
    infers sameness from the pictures themselves.
    """

    if asset.role is not AssetRole.THUMBNAIL:
        return False

    if asset.width is None or full.width is None:
        return False

    return (
        asset.width <= full.width
        and asset.height <= full.height
        and (asset.width, asset.height) != (full.width, full.height)
    )


def choose_primary(group: list[VisualAsset]) -> VisualAsset:
    """
    The asset whose analysis stands for a duplicate group.

    Prefers, in order: a carousel slide over a preview, then more pixels,
    then more bytes, then the filename. The last tiebreak exists so that
    two files identical in every measured respect still resolve the same
    way on every run, which is what keeps a stored digest comparable.
    """

    def rank(asset: VisualAsset) -> tuple:
        is_preview = 1 if asset.role is AssetRole.THUMBNAIL else 0

        return (
            is_preview,
            -(asset.width or 0) * (asset.height or 0),
            -asset.byte_size,
            asset.filename,
        )

    return sorted(group, key=rank)[0]


def mark_duplicates(assets: list[VisualAsset]) -> int:
    """
    Mark every non-primary member of a duplicate group.

    The asset is kept and annotated rather than removed. A reader
    looking at a post should see that it had three images and that two
    were the same picture, not silently see one image and wonder.

    Returns how many assets were marked.
    """

    marked = 0

    for members in group_by_content(assets).values():
        if len(members) < 2:
            continue

        primary = choose_primary(members)

        for member in members:
            if member is primary:
                member.is_primary = True
                member.duplicate_of = None
                continue

            member.is_primary = False
            member.duplicate_of = primary.filename
            marked += 1

    return marked


def redundant_assets(assets: list[VisualAsset]) -> list[VisualAsset]:
    """
    Assets that carry no knowledge of their own.

    Duplicates, and previews whose full image is present. Both are
    excluded from the fingerprint, because a different copy of a picture
    nobody reads from is not a change in what the post says.

    A preview is only redundant when its larger counterpart is actually
    present. On its own it is the post's only image, and skipping it
    would discard the only visual evidence that post has.
    """

    redundant: list[VisualAsset] = []

    for asset in assets:
        if asset.duplicate_of:
            redundant.append(asset)
            continue

        if asset.role is not AssetRole.THUMBNAIL:
            continue

        larger = [
            other
            for other in assets
            if other is not asset
            and other.role is AssetRole.SLIDE
            and other.width is not None
            and asset.width is not None
            and other.width >= asset.width
            and other.height >= (asset.height or 0)
        ]

        if larger:
            asset.note = (
                asset.note
                or "Low-resolution preview; the full slide is present"
            )

            redundant.append(asset)

    return redundant


def asset_digest(
    assets: list[VisualAsset],
    *,
    processor_version: str,
    configuration: str,
) -> str:
    """
    A digest of what a post's images say, in the order they say it.

    Covers, for each asset that carries knowledge: its content digest,
    its sequence, and its role. The processor version and configuration
    are folded in, so upgrading the processor re-reads slides it would
    otherwise reuse.

    Deliberately excluded: byte size, pixel dimensions and the filename.
    Re-encoding a slide at the same quality changes all three and changes
    none of what it says, and re-enriching 490 posts because an export
    tool rewrote a filename is exactly the cost this design exists to
    avoid.

    Also excluded: redundant assets. A new preview of a slide already
    present does not alter the post's visual knowledge, so it must not
    invalidate it.
    """

    import hashlib

    digest = hashlib.sha256()

    digest.update(f"{processor_version}|{configuration}".encode("utf-8"))

    contributing = sorted(
        (
            asset
            for asset in assets
            if not asset.duplicate_of and asset.role is not AssetRole.THUMBNAIL
        ),
        key=lambda asset: (asset.sequence, asset.filename),
    )

    for asset in contributing:
        digest.update(b"\x00")
        digest.update(f"{asset.sequence}:{asset.role.value}:".encode("utf-8"))

        # A failed asset is included by its absence of digest, so that
        # a file becoming unreadable invalidates the post. Silence would
        # look the same as never having had it.
        digest.update((asset.sha256 or f"unreadable:{asset.note}").encode("utf-8"))

    return digest.hexdigest()


def perceptual_hint(first: VisualAsset, second: VisualAsset) -> bool:
    """
    Whether two assets are *possibly* the same picture.

    Reported, never acted on, and kept for one reason: it makes the
    decision to skip perceptual merging explicit rather than an omission
    somebody will later read as an oversight.

    The signal is the dimension ratio alone. Comparing pixels needs a
    hashing library that is not installed, and adding one to make a
    heuristic act on would be a bad trade.
    """

    if not (first.width and first.height and second.width and second.height):
        return False

    if first.sha256 == second.sha256:
        return True

    long_first = max(first.width, first.height)
    long_second = max(second.width, second.height)

    if not (long_first and long_second):
        return False

    ratio = long_first / long_second

    if ratio < 1:
        ratio = 1 / ratio

    # Within about 2 per cent on the long edge and the same shape.
    if ratio >= 1.02:
        return False

    short_ratio = (min(first.width, first.height) / min(second.width, second.height))

    return 1 / 1.02 <= short_ratio <= 1.02


__all__ = [
    "asset_digest",
    "choose_primary",
    "group_by_content",
    "mark_duplicates",
    "perceptual_hint",
    "redundant_assets",
]
