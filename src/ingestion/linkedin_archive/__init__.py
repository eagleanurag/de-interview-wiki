"""
Import a local LinkedIn archive into the knowledge base.

A folder holding a saved-post export the person already has, and the
media that came with it. This package reads it, works out what each
record honestly claims, and writes it out in the form the saved-items
importer already reads. It collects nothing, opens no browser, reads no
credential, and never writes to the archive it was pointed at.

The division of work is deliberate. Identity and validation live in
:mod:`identity` and :mod:`archive`; writing the drop zone lives in
:mod:`prepare`; the numbers live in :mod:`report`. Everything past the
reading stage is the existing architecture, reused rather than
reimplemented, so bundle association, content fingerprints, capture
quality and the resume record all behave the way they already do.
"""

from __future__ import annotations

from src.ingestion.linkedin_archive.archive import (
    ARCHIVE_JSON,
    FORBIDDEN_DIRECTORIES,
    MEDIA_DIR,
    Archive,
    ArchiveError,
    ArchiveFailure,
    ArchiveRecord,
    MediaAsset,
    MediaIndex,
    build_media_index,
    read_archive,
)
from src.ingestion.linkedin_archive.identity import (
    ARCHIVE_NAMESPACE,
    KIND_ARCHIVE_ONLY,
    ArchiveIdentity,
    content_fingerprint,
    identify,
)
from src.ingestion.linkedin_archive.prepare import (
    MANIFEST,
    PrepareResult,
    prepare as prepare_drop_zone,
)
from src.ingestion.saved_items.model import KIND_IDENTIFIED_BY_ID
from src.ingestion.linkedin_archive.report import (
    ImportReport,
    build_report,
    write_report,
)


__all__ = [
    "ARCHIVE_JSON",
    "ARCHIVE_NAMESPACE",
    "FORBIDDEN_DIRECTORIES",
    "KIND_ARCHIVE_ONLY",
    "KIND_IDENTIFIED_BY_ID",
    "MANIFEST",
    "MEDIA_DIR",
    "Archive",
    "ArchiveError",
    "ArchiveFailure",
    "ArchiveIdentity",
    "ArchiveRecord",
    "ImportReport",
    "MediaAsset",
    "MediaIndex",
    "PrepareResult",
    "build_media_index",
    "build_report",
    "content_fingerprint",
    "identify",
    "prepare_drop_zone",
    "read_archive",
    "write_report",
]
