"""
Errors raised by the ingestion layer.

Every message here is written for the person adding a post by hand.
They are looking at a capture bundle on their laptop, not at a
traceback, so an error has to say what to change.
"""

from __future__ import annotations


class IngestionError(RuntimeError):
    """Base class for every ingestion failure."""


class PostExistsError(IngestionError):
    """Raised when creating a post that already exists."""


class PostNotFoundError(IngestionError):
    """Raised when a post directory does not exist."""


class InvalidPostError(IngestionError):
    """Raised when a post cannot be used or written as authored."""


class MediaConflictError(IngestionError):
    """Raised when adding media that would overwrite other content."""


class UnsupportedMediaError(IngestionError):
    """Raised when a media file cannot be given a portable name."""
