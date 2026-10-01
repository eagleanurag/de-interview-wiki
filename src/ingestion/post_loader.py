"""
Reading a normalized post for the worker.

The structure is the one the repository already ships::

    post_directory/
    ├── post.json
    └── media/
        ├── image.png
        └── document.pdf

``post.json`` may declare its media, in which case the declaration is
honoured and its description is kept. Files that are present in
``media/`` but not declared are still discovered, so a post dropped in
by hand behaves exactly as before.

This module is the reader the enrichment pipeline depends on. Writing
posts is the job of :mod:`src.ingestion.importer`; nothing here
modifies a post directory.
"""

from datetime import datetime
from pathlib import Path

from src.ingestion.post_document import (
    IMAGE_EXTENSIONS,
    PDF_EXTENSIONS,
    PostDocument,
    resolve_media_path,
)
from src.models import (
    AIAnalysis,
    Classification,
    InterviewQuestion,
    KnowledgePost,
    MediaItem,
    SourceInfo,
)


def load_post(post_directory: str | Path) -> KnowledgePost:
    """
    Load a normalized knowledge post from a local post directory.

    Expected structure:

        post_directory/
        ├── post.json
        └── media/
            ├── image.png
            └── document.pdf
    """

    post_directory = Path(post_directory)

    if not post_directory.exists():
        raise FileNotFoundError(
            f"Post directory does not exist: {post_directory}"
        )

    post_file = post_directory / "post.json"

    if not post_file.exists():
        raise FileNotFoundError(
            f"post.json not found in: {post_directory}"
        )

    document = PostDocument.load(post_directory)

    raw_data = document.data

    media_items = _resolve_media(document, post_directory)

    source_data = raw_data.get("source", {})

    source = SourceInfo(
        platform=source_data.get("platform", "unknown"),
        url=source_data.get("url"),
        captured_at=_parse_datetime(
            source_data.get("captured_at")
        ),
        author=source_data.get("author"),
        # Kept verbatim rather than parsed: a relative form such as
        # "2 days ago" is only resolvable against the capture date.
        published_at=source_data.get("published_at"),
    )

    ai_data = raw_data.get("ai_analysis", {})

    ai_analysis = AIAnalysis(
        summary=ai_data.get("summary"),
        topics=ai_data.get("topics", []),
        subtopics=ai_data.get("subtopics", []),
        concepts=ai_data.get("concepts", []),
        image_descriptions=ai_data.get(
            "image_descriptions", []
        ),
    )

    classification_data = raw_data.get(
        "classification", {}
    )

    classification = Classification(
        domain=classification_data.get(
            "domain",
            "Data Engineering",
        ),
        primary_topic=classification_data.get(
            "primary_topic"
        ),
        secondary_topics=classification_data.get(
            "secondary_topics", []
        ),
        interview_relevant=classification_data.get(
            "interview_relevant",
            False,
        ),
    )

    questions = [
        InterviewQuestion.model_validate(question)
        for question in raw_data.get(
            "interview_questions",
            [],
        )
    ]

    return KnowledgePost(
        id=raw_data["id"],
        source=source,
        original_text=raw_data.get(
            "original_text",
            "",
        ),
        media=media_items,
        ai_analysis=ai_analysis,
        interview_questions=questions,
        classification=classification,
    )


def _resolve_media(
    document: PostDocument,
    post_directory: Path,
) -> list[MediaItem]:
    """
    Build the media list for a post.

    Declared media comes first, in the order the author declared it,
    with any description preserved. Everything else in ``media/`` is
    discovered, so an undeclared file is still ingested exactly as it
    was before declarations existed.
    """

    media_items: list[MediaItem] = []
    declared_paths: set[Path] = set()

    for entry in document.media():
        target = resolve_media_path(post_directory, entry.path)

        declared_paths.add(target)

        media_items.append(
            _media_item(
                media_type=entry.type,
                path=target,
                description=entry.description or None,
            )
        )

    for item in _discover_media(post_directory / "media"):
        if Path(item.path) in declared_paths:
            continue

        media_items.append(item)

    return media_items


def _media_item(
    *,
    media_type: str,
    path: Path,
    description: str | None = None,
) -> MediaItem:
    return MediaItem(
        type=media_type,
        path=str(path),
        description=description,
    )


def _discover_media(
    media_directory: Path,
) -> list[MediaItem]:
    """Discover media files inside a post's media directory."""

    if not media_directory.exists():
        return []

    media_items: list[MediaItem] = []

    for file_path in sorted(media_directory.iterdir()):

        if not file_path.is_file():
            continue

        extension = file_path.suffix.lower()

        if extension in IMAGE_EXTENSIONS:
            media_type = "image"

        elif extension in PDF_EXTENSIONS:
            media_type = "pdf"

        else:
            media_type = "other"

        media_items.append(
            MediaItem(
                type=media_type,
                path=str(file_path),
            )
        )

    return media_items


def _parse_datetime(value: str | None) -> datetime:
    """Parse an ISO-8601 datetime."""

    if not value:
        return datetime.now().astimezone()

    return datetime.fromisoformat(value)
