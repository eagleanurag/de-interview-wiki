import json
from datetime import datetime
from pathlib import Path

from src.models import (
    AIAnalysis,
    Classification,
    InterviewQuestion,
    KnowledgePost,
    MediaItem,
    SourceInfo,
)


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".gif",
    ".bmp",
}

PDF_EXTENSIONS = {
    ".pdf",
}


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

    raw_data = json.loads(
        post_file.read_text(encoding="utf-8")
    )

    media_items = _discover_media(post_directory / "media")

    source_data = raw_data.get("source", {})

    source = SourceInfo(
        platform=source_data.get("platform", "unknown"),
        url=source_data.get("url"),
        captured_at=_parse_datetime(
            source_data.get("captured_at")
        ),
        author=source_data.get("author"),
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


def _discover_media(
    media_directory: Path,
) -> list[MediaItem]:
    """Discover supported media files inside a post's media directory."""

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