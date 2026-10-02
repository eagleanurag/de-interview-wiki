"""
Turn a read archive into the drop zone the existing importer reads.

The archive is not imported by a second pipeline. It is written out in
the form CP8 and CP9 already consume -- a manifest of saved items and a
folder of capture bundles per item -- and handed to ``SavedItemsSource``.
That reuse is the point: bundle association, content fingerprints,
capture quality, the manifest as a resume record, and the reconcile pass
that marks what really landed are all already written and tested, and a
parallel path would be a second thing to keep correct.

Nothing here writes to the archive. Media is hard-linked where the
volume allows and copied where it does not, so the archive stays the one
copy of the original bytes and the working drop zone costs no more space
than a name. A hard link is a second name for the same bytes, not a
copy, so a later edit to either would change both -- which is another
reason the archive is treated as read-only and never opened for writing.

Only a manifest and capture folders are written. An earlier version also
wrote a side-car naming what the drop zone was made from, and it broke
every tool that reads the inbox: a saved-items drop zone is read by
trying each supported file as a candidate list, so a foreign JSON in
the root becomes a list of rows with no URL, and the validator duly
reported several hundred of them. An inbox has one contract -- everything
in it is a candidate list -- and the side-car was not honouring it.

The provenance it carried is not lost. Every row carries an
archive-namespaced source id, every capture file names the archive post
it came from, and the import report holds the counts.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from src.ingestion.linkedin_archive.archive import (
    Archive,
    ArchiveRecord,
)
from src.ingestion.saved_items.model import CaptureMethod, SavedItem
from src.ingestion.saved_items.urls import try_normalize


#: The directory inside the drop zone that holds the captures.
CAPTURES = "captures"

#: The list the existing saved-items reader consumes.
MANIFEST = "manifest.csv"

#: Columns written, in a fixed order so two runs produce the same file.
#: ``Bundle`` points an item at its own capture, the most explicit claim
#: available and the cheapest to match. ``Source ID`` is what a record
#: with no permalink is identified by, and it is written for every row
#: rather than only the ones that need it, so the column's presence never
#: depends on which records happened to lack a link.
COLUMNS = (
    "URL",
    "Source ID",
    "Saved Date",
    "Title",
    "Author",
    "Notes",
    "Bundle",
)


@dataclass
class PrepareResult:
    """
    What preparing the drop zone did.

    Counts rather than a list of everything, because the interesting
    question is how many posts are ready and what went wrong, not which
    ones.
    """

    root: Path
    written: int = 0
    skipped_duplicates: int = 0
    unchanged: int = 0
    media_linked: int = 0
    media_copied: int = 0
    media_shared: int = 0
    problems: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "root": str(self.root),
            "written": self.written,
            "skipped_duplicates": self.skipped_duplicates,
            "unchanged": self.unchanged,
            "media_linked": self.media_linked,
            "media_copied": self.media_copied,
            "media_shared": self.media_shared,
            "problems": list(self.problems),
        }


def prepare(
    archive: Archive,
    destination: str | Path,
    *,
    include_duplicates: bool = False,
) -> PrepareResult:
    """
    Write the archive out as a saved-items drop zone.

    Duplicates are left out of the drop zone by default. The record that
    keeps a duplicate's place is written instead, and the dropped record
    is named in that one's notes, so the second capture is not lost so
    much as it is attributed to the one that was kept.

    Every record the drop zone claims contributes a manifest row, whether
    or not its capture folder had to be rewritten. The manifest is the
    list; a capture folder is the content behind one entry. Rewriting
    the manifest from only the records that changed would leave a
    re-run with a header and no rows, which reads as an empty inbox and
    imports nothing if the stored state is ever cleared.
    """

    base = Path(destination).expanduser()
    captures = base / CAPTURES

    base.mkdir(parents=True, exist_ok=True)
    captures.mkdir(parents=True, exist_ok=True)

    result = PrepareResult(root=base)

    rows: list[list[str]] = [list(COLUMNS)]

    for record in archive.records:
        if record.is_duplicate and not include_duplicates:
            result.skipped_duplicates += 1
            continue

        try:
            row, rewritten = _write_record(
                record, captures, archive, result
            )

        except Exception as exc:  # noqa: BLE001
            result.problems.append(
                f"{record.post_id}: {type(exc).__name__}: {exc}"
            )
            continue

        rows.append(row)

        if rewritten:
            result.written += 1
        else:
            result.unchanged += 1

    _write_manifest(base, rows, result)

    return result


def _write_record(
    record: ArchiveRecord,
    captures: Path,
    archive: Archive,
    result: PrepareResult,
) -> tuple[list[str], bool]:
    """
    Write one record's capture folder and return its manifest row.

    The row is returned whether or not the folder had to be rewritten;
    the boolean says whether it was. An unchanged capture folder is
    left alone, which is what makes a re-run cheap, but the record still
    belongs in the manifest because the manifest is the list rather than
    a log of this run's edits.
    """

    bundle = record.identity.bundle_name
    folder = captures / bundle

    payload = _capture_payload(record, archive)

    body = payload["text"]

    # The fingerprint covers the text and the *content* of each media
    # file, not merely its name. A file replaced under the same name is
    # a change the reader needs to see, and the index already digested
    # every asset to build the sharing report, so including the digest
    # costs one more hash of a value already in hand rather than
    # another pass over twenty-three megabytes.
    fingerprint = _fingerprint(body, record.media)

    marker = folder / "capture.json"

    if marker.is_file():
        try:
            existing = json.loads(marker.read_text(encoding="utf-8"))

        except (OSError, json.JSONDecodeError):
            existing = {}

        if isinstance(existing, dict) and existing.get(
            "_archive_fingerprint"
        ) == fingerprint:
            # Already prepared from this exact content. The folder is
            # left alone, so a second import does not touch five
            # hundred of them to change nothing, but the row is still
            # built and returned: the manifest has to keep listing what
            # the drop zone holds, not only what this run wrote.
            return _row_for(record, bundle), False

    folder.mkdir(parents=True, exist_ok=True)

    text_file = folder / "content.md"
    text_file.write_text(body, encoding="utf-8")

    keep = _place_media(record, folder, archive, result)

    marker.write_text(
        json.dumps(
            {
                "url": record.identity.canonical_url or "",
                "source_id": record.identity.source_id,
                "title": _title_for(record),
                "author": record.author or "",
                # Deliberately no "text" here. The body already lives in
                # content.md, and the bundle reader joins every text
                # source it finds, so carrying it in both would store the
                # post twice over. This file names the post; content.md
                # holds it.
                "_archive_post_id": record.post_id,
                "_archive_permalink_present": record.identity.has_permalink,
                "_archive_scraped_at": record.scraped_at or "",
                "_archive_fingerprint": fingerprint,
                "_archive_media": keep,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return _row_for(record, bundle), True


def _row_for(record: ArchiveRecord, bundle: str) -> list[str]:
    """
    The manifest row for one record.

    A record with no permalink contributes an empty URL cell and is
    matched by its ``Source ID``, which is what it is identified by. The
    row is built the same way whether or not the capture folder had to
    be written, so a re-run produces the same manifest it produced the
    first time rather than a shorter one.
    """

    return [
        record.identity.canonical_url or "",
        record.identity.source_id,
        record.scraped_at or "",
        _title_for(record) or "",
        record.author or "",
        "; ".join(record.notes) if record.notes else "",
        bundle,
    ]


def _capture_payload(record: ArchiveRecord, archive: Archive) -> dict:
    """
    The text and media one record contributes.

    The archive's text is written exactly as it was recorded. It is not
    trimmed into a summary, reordered, or cleaned up, because the point
    of the archive is what the post said, and a reader comparing the
    knowledge base against LinkedIn needs to find the same words.
    """
    return {
        "text": record.raw_text if record.has_text else "",
        "media": [asset.name for asset in record.media],
    }


def _fingerprint(body: str, media: list) -> str:
    """
    What a prepared capture depends on.

    The post's text and the name and content digest of every media file
    it carries. The digest is already computed by the media index, so
    this is arithmetic on a value in hand rather than a second read of
    every image.
    """
    import hashlib

    hasher = hashlib.sha256()

    hasher.update(body.encode("utf-8"))

    for asset in sorted(media, key=lambda item: item.name):
        hasher.update(b"\0")
        hasher.update(asset.name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update((asset.digest or "").encode("utf-8"))

    return hasher.hexdigest()[:32]


def _place_media(
    record: ArchiveRecord,
    folder: Path,
    archive: Archive,
    result: PrepareResult,
) -> list[dict]:
    """
    Put a record's media in its capture folder.

    Identical bytes referenced by two posts are written once per post
    folder, because each post's media travels with that post and a
    shared file would mean a post whose media disappears when another is
    deleted. The sharing is recorded rather than deduplicated away, so
    the report can say how much of the archive was the same picture more
    than once.
    """

    placed: list[dict] = []

    for asset in record.media:
        name = asset.name

        source = asset.path

        target = folder / name

        try:
            _place_one(source, target)

        except OSError as exc:
            placed.append(
                {
                    "name": name,
                    "placed": False,
                    "problem": f"{type(exc).__name__}: {exc}",
                }
            )
            record.note(f"{name} could not be placed: {exc}")
            continue

        entry = {
            "name": name,
            "placed": True,
            "size": source.stat().st_size,
        }

        shared = archive.media.shared_by_digest(name)

        if shared:
            entry["identical_to"] = shared
            result.media_shared += 1

        if target.stat().st_ino == source.stat().st_ino:
            entry["linked"] = True
            result.media_linked += 1
        else:
            entry["copied"] = True
            result.media_copied += 1

        placed.append(entry)

    return placed


def _place_one(source: Path, target: Path) -> None:
    """
    Make ``target`` a name for ``source``'s bytes.

    A hard link where the volume allows, because the two names are then
    the same file and the drop zone costs no space. A copy where it does
    not, or where the link is refused, so this works across volumes and
    on a filesystem without links.
    """
    if target.is_file():
        try:
            if target.stat().st_size == source.stat().st_size:
                return

        except OSError:
            pass

        target.unlink()

    try:
        os.link(source, target)

    except (OSError, NotImplementedError, AttributeError):
        shutil.copyfile(source, target)


def _title_for(record: ArchiveRecord) -> str | None:
    """
    A label for the manifest, taken from the post itself.

    The first line of the post, which is what a reader would recognise
    it by. Not invented and not from the archive, which never recorded
    one: the archive has no title field at all, and the first line is
    the only thing in the record that names its subject.
    """
    for line in record.raw_text.splitlines():
        cleaned = line.strip().lstrip("#").strip()

        if cleaned:
            return cleaned[:200]

    return None


def _write_manifest(
    base: Path,
    rows: list[list[str]],
    result: PrepareResult,
) -> None:
    """
    Write the manifest the existing reader consumes.

    A record with no permalink contributes an empty URL cell and is
    matched by its ``Bundle`` column, which is an explicit claim of its
    own capture. A reader that requires a URL would refuse the row; this
    one does not, because the drop zone already knows where the content
    is and a missing link is not a reason to lose the post.
    """
    import csv
    import io

    buffer = io.StringIO()

    writer = csv.writer(buffer, lineterminator="\n")

    for row in rows:
        writer.writerow(row)

    target = base / MANIFEST

    temporary = target.with_suffix(".csv.tmp")

    temporary.write_text(buffer.getvalue(), encoding="utf-8")

    os.replace(temporary, target)


def to_saved_item(record: ArchiveRecord) -> SavedItem | None:
    """
    The saved item an archive record becomes, when read as a list row.

    Used by the importer's own plan view so the archive can be checked
    before anything is written. A record with no usable URL has no row
    in the manifest to become, and returns None rather than a
    half-populated item.
    """
    normalized = try_normalize(record.identity.canonical_url)

    if normalized is None:
        return None

    return SavedItem.from_url(
        normalized,
        capture_method=CaptureMethod.USER_BUNDLE,
    )


__all__ = [
    "CAPTURES",
    "COLUMNS",
    "MANIFEST",
    "PrepareResult",
    "prepare",
    "to_saved_item",
]
