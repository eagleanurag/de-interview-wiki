"""
Local ingestion of LinkedIn Saved Items.

This package turns a list the user exported, and content the user
captured, into Knowledge Posts. It reads files and nothing else. There
is no browser, no network call, no session, and no dependency on how
LinkedIn's web interface happens to be laid out today.

The boundary is deliberate. A saved list is data the user already has.
This package is how that data enters the knowledge pipeline, so the
pipeline keeps working whether or not the website changes, and the
project never needs to fetch a saved post to know what the user saved.
"""

from src.ingestion.saved_items.model import (
    CaptureMethod,
    SavedItem,
    SavedItemState,
)
from src.ingestion.saved_items.readers import (
    ManifestError,
    ManifestRead,
    ReadIssue,
    read_manifest,
)
from src.ingestion.saved_items.urls import (
    NormalizedUrl,
    SavedItemUrlError,
    normalize_linkedin_url,
    source_id_for,
)

__all__ = [
    "CaptureMethod",
    "ManifestError",
    "ManifestRead",
    "NormalizedUrl",
    "ReadIssue",
    "SavedItem",
    "SavedItemState",
    "SavedItemUrlError",
    "normalize_linkedin_url",
    "read_manifest",
    "source_id_for",
]
