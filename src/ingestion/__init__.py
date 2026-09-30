"""
The ingestion layer.

Everything needed to turn a manually captured interview post into the
committed shape the rest of the pipeline reads:

* :mod:`src.ingestion.post_document` owns ``post.json``
* :mod:`src.ingestion.importer` creates, updates and validates posts
* :mod:`src.ingestion.post_loader` reads a post for the worker
* :mod:`src.ingestion.cli` exposes all of it on the command line

Content is added by hand or by an explicitly authorised process. There
is no scraper here and no network client, by design.
"""

from src.ingestion.errors import (
    IngestionError,
    InvalidPostError,
    MediaConflictError,
    PostExistsError,
    PostNotFoundError,
    UnsupportedMediaError,
)
from src.ingestion.importer import (
    DEFAULT_POSTS_ROOT,
    MediaImportResult,
    PostImportResult,
    PostSummary,
    add_media,
    create_post,
    discover_posts,
    import_post,
    post_directory,
    posts_root,
    validate_post,
    validate_posts,
)
from src.ingestion.post_document import (
    MEDIA_DIRECTORY_NAME,
    POST_FILE_NAME,
    MediaEntry,
    PostDocument,
    normalize_post_id,
)
from src.ingestion.post_loader import load_post
from src.ingestion.validation import (
    ValidationIssue,
    ValidationReport,
    describe_report,
)


__all__ = [
    "DEFAULT_POSTS_ROOT",
    "IngestionError",
    "InvalidPostError",
    "MEDIA_DIRECTORY_NAME",
    "POST_FILE_NAME",
    "MediaConflictError",
    "MediaEntry",
    "MediaImportResult",
    "PostDocument",
    "PostExistsError",
    "PostImportResult",
    "PostNotFoundError",
    "PostSummary",
    "UnsupportedMediaError",
    "ValidationIssue",
    "ValidationReport",
    "add_media",
    "create_post",
    "describe_report",
    "discover_posts",
    "import_post",
    "load_post",
    "normalize_post_id",
    "post_directory",
    "posts_root",
    "validate_post",
    "validate_posts",
]
