"""
Checking that a post is one the pipeline can use.

The ingestion layer can write a post that a worker would choke on, or
that would publish an empty page, so every authored document is checked
against the post contract before it is written. The same checks run
over the committed posts, through the command line and through the test
suite, so a post that would break a worker does not reach `main`.

Findings come in two levels. An *error* describes a post that will
fail, or silently degrade, downstream. A *warning* describes a post
that will work but is not described as clearly as it could be, so a
warning never fails a run.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from src.ingestion.errors import IngestionError, InvalidPostError
from src.ingestion.post_document import (
    DOCUMENT_KEYS,
    MEDIA_DIRECTORY_NAME,
    MEDIA_TYPES,
    POST_FILE_NAME,
    MediaEntry,
    PostDocument,
    media_type_for,
    normalize_post_id,
    resolve_media_path,
    type_matches_extension,
)
from src.models import InterviewQuestion


if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance
    from src.ingestion.importer import PostSummary


LEVEL_ERROR = "error"
LEVEL_WARNING = "warning"

# Bound a listing so a mistaken capture cannot flood an issue comment.
MAX_LISTED_IN_MESSAGE = 5

# ---------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationIssue:
    """One finding about one post."""

    level: str
    post_id: str
    message: str

    @property
    def is_error(self) -> bool:
        return self.level == LEVEL_ERROR

    def __str__(self) -> str:
        return f"{self.level}: {self.post_id}: {self.message}"


@dataclass(frozen=True)
class ValidationReport:
    """The result of validating a whole posts tree."""

    root: Path
    posts: tuple[PostSummary, ...] = ()
    issues: tuple[ValidationIssue, ...] = ()

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(
            issue for issue in self.issues if issue.is_error
        )

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(
            issue
            for issue in self.issues
            if not issue.is_error
        )

    @property
    def ok(self) -> bool:
        return not self.errors


# ---------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------


def validate_post_directory(
    directory: str | Path,
) -> list[ValidationIssue]:
    """Validate one post directory, reporting every problem found."""

    path = Path(directory)
    post_id = path.name

    if not (path / POST_FILE_NAME).is_file():
        return [
            ValidationIssue(
                level=LEVEL_WARNING,
                post_id=post_id,
                message=(
                    f"no {POST_FILE_NAME}; this directory is not "
                    "ingested by the pipeline"
                ),
            )
        ]

    try:
        document = PostDocument.load(path)
    except IngestionError as exc:
        return [
            ValidationIssue(
                level=LEVEL_ERROR,
                post_id=post_id,
                message=str(exc),
            )
        ]

    issues = list(
        validate_document(
            document, post_id=post_id, directory=path
        )
    )

    issues.extend(_undeclared_media_issues(document, path))

    return issues


def validate_document(
    document: PostDocument,
    *,
    post_id: str = "",
    directory: str | Path | None = None,
    require_content: bool = True,
) -> list[ValidationIssue]:
    """
    Check one document against the post contract.

    ``directory`` enables the checks that need the files to be on disk.
    Without it the document is checked in the abstract, which is what
    an import does before it writes anything.

    ``require_content`` is only ever turned off for an explicitly
    requested empty scaffold. Validation of the committed tree always
    requires content.
    """

    identifier = post_id or document.post_id
    issues: list[ValidationIssue] = []

    issues.extend(_check_identity(document, identifier))
    issues.extend(_check_source(document, identifier))

    if require_content or document.media():
        issues.extend(_check_text(document, identifier))

    issues.extend(_check_media(document, identifier, directory))
    issues.extend(_check_questions(document, identifier))
    issues.extend(_check_classification(document, identifier))
    issues.extend(_check_unknown_keys(document, identifier))

    return issues


# ---------------------------------------------------------------------
# The media directory
# ---------------------------------------------------------------------


def media_files(directory: Path) -> list[Path]:
    """
    The media files inside a post.

    A directory inside media/ is enrichment output, not a capture, so
    only files are returned.
    """
    media_directory = directory / MEDIA_DIRECTORY_NAME

    if not media_directory.is_dir():
        return []

    return [
        path
        for path in sorted(media_directory.iterdir())
        if path.is_file()
    ]


# ---------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------


def _check_identity(
    document: PostDocument,
    post_id: str,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    declared = document.post_id

    try:
        normalize_post_id(declared)
    except InvalidPostError as exc:
        issues.append(
            ValidationIssue(
                level=LEVEL_ERROR,
                post_id=post_id,
                message=f"invalid post id: {exc}",
            )
        )

    if post_id and declared != post_id:
        issues.append(
            ValidationIssue(
                level=LEVEL_ERROR,
                post_id=post_id,
                message=(
                    f"post.json declares id {declared!r} but lives in "
                    f"directory {post_id!r}; the pipeline and the "
                    "generated site use both, so they must agree"
                ),
            )
        )

    return issues


def _check_source(
    document: PostDocument,
    post_id: str,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    raw = document.data.get("source")

    if raw is None:
        return [
            ValidationIssue(
                level=LEVEL_ERROR,
                post_id=post_id,
                message="missing 'source'; say where this was captured",
            )
        ]

    if not isinstance(raw, dict):
        return [
            ValidationIssue(
                level=LEVEL_ERROR,
                post_id=post_id,
                message="'source' must be a JSON object",
            )
        ]

    platform = raw.get("platform")

    if not isinstance(platform, str) or not platform.strip():
        issues.append(
            ValidationIssue(
                level=LEVEL_ERROR,
                post_id=post_id,
                message=(
                    "source.platform is required, for example "
                    "'manual' for a hand-captured post"
                ),
            )
        )

    for key in ("url", "author"):
        value = raw.get(key)

        if value is not None and not isinstance(value, str):
            issues.append(
                ValidationIssue(
                    level=LEVEL_ERROR,
                    post_id=post_id,
                    message=f"source.{key} must be a string",
                )
            )

    captured_at = raw.get("captured_at")

    if captured_at is not None:
        if not isinstance(captured_at, str):
            issues.append(
                ValidationIssue(
                    level=LEVEL_ERROR,
                    post_id=post_id,
                    message=(
                        "source.captured_at must be an "
                        "ISO-8601 string"
                    ),
                )
            )
        else:
            try:
                datetime.fromisoformat(captured_at)
            except ValueError:
                issues.append(
                    ValidationIssue(
                        level=LEVEL_ERROR,
                        post_id=post_id,
                        message=(
                            "source.captured_at is not ISO-8601: "
                            f"{captured_at!r}"
                        ),
                    )
                )

    return issues


def _check_text(
    document: PostDocument,
    post_id: str,
) -> list[ValidationIssue]:
    text = document.data.get("original_text")

    if text is None:
        text = ""

    if not isinstance(text, str):
        return [
            ValidationIssue(
                level=LEVEL_ERROR,
                post_id=post_id,
                message="original_text must be a string",
            )
        ]

    if not text.strip() and not document.media():
        return [
            ValidationIssue(
                level=LEVEL_ERROR,
                post_id=post_id,
                message=(
                    "a post needs original_text or at least one "
                    "media file, otherwise it contributes nothing"
                ),
            )
        ]

    return []


def _check_media(
    document: PostDocument,
    post_id: str,
    directory: str | Path | None,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    raw = document.data.get("media")

    if raw is None:
        return []

    if not isinstance(raw, list):
        return [
            ValidationIssue(
                level=LEVEL_ERROR,
                post_id=post_id,
                message="'media' must be a list of media entries",
            )
        ]

    seen: set[str] = set()

    for index, entry in enumerate(raw):
        try:
            media = MediaEntry.from_dict(entry, post_id=post_id)
        except InvalidPostError as exc:
            issues.append(
                ValidationIssue(
                    level=LEVEL_ERROR,
                    post_id=post_id,
                    message=f"media[{index}]: {exc}",
                )
            )
            continue

        if media.type not in MEDIA_TYPES:
            issues.append(
                ValidationIssue(
                    level=LEVEL_ERROR,
                    post_id=post_id,
                    message=(
                        f"media[{index}] has unknown type "
                        f"{media.type!r}; use one of "
                        f"{', '.join(sorted(MEDIA_TYPES))}"
                    ),
                )
            )
            continue

        if media.path in seen:
            issues.append(
                ValidationIssue(
                    level=LEVEL_ERROR,
                    post_id=post_id,
                    message=(
                        f"media path {media.path!r} is declared "
                        "twice"
                    ),
                )
            )
            continue

        seen.add(media.path)

        # Checked syntactically, so it applies before anything is
        # written as well as after: a post.json can never point the
        # loader at a file outside its own post directory.
        if _escapes_post(media.path):
            issues.append(
                ValidationIssue(
                    level=LEVEL_ERROR,
                    post_id=post_id,
                    message=(
                        f"media[{index}] path {media.path!r} must be "
                        "relative to the post directory and stay "
                        "inside it"
                    ),
                )
            )
            continue

        name = media.path.rsplit("/", 1)[-1]

        if not type_matches_extension(media.type, name):
            issues.append(
                ValidationIssue(
                    level=LEVEL_ERROR,
                    post_id=post_id,
                    message=(
                        f"media[{index}] declares type "
                        f"{media.type!r} but {name!r} is "
                        f"{media_type_for(name)!r}"
                    ),
                )
            )
            continue

        if directory is None:
            continue

        try:
            target = resolve_media_path(directory, media.path)
        except InvalidPostError as exc:
            issues.append(
                ValidationIssue(
                    level=LEVEL_ERROR,
                    post_id=post_id,
                    message=f"media[{index}]: {exc}",
                )
            )
            continue

        if not target.is_file():
            issues.append(
                ValidationIssue(
                    level=LEVEL_ERROR,
                    post_id=post_id,
                    message=(
                        f"declared media file is missing: "
                        f"{media.path}"
                    ),
                )
            )

    return issues


def _escapes_post(relative: str) -> bool:
    """
    Whether a declared media path leaves its post directory.

    Purely syntactic, so it can be checked on a document that has not
    been written yet. Absolute paths and `..` segments both qualify.
    """

    candidate = Path(relative)

    return bool(
        candidate.is_absolute()
        or candidate.anchor
        or ".." in candidate.parts
    )


def _check_questions(
    document: PostDocument,
    post_id: str,
) -> list[ValidationIssue]:
    raw = document.data.get("interview_questions")

    if raw is None:
        return []

    if not isinstance(raw, list):
        return [
            ValidationIssue(
                level=LEVEL_ERROR,
                post_id=post_id,
                message="'interview_questions' must be a list",
            )
        ]

    issues: list[ValidationIssue] = []

    for index, entry in enumerate(raw):
        try:
            InterviewQuestion.model_validate(entry)
        except Exception as exc:  # noqa: BLE001 - pydantic raises here
            detail = str(exc).splitlines()

            issues.append(
                ValidationIssue(
                    level=LEVEL_ERROR,
                    post_id=post_id,
                    message=(
                        f"interview_questions[{index}] is invalid: "
                        f"{detail[0] if detail else exc}"
                    ),
                )
            )

    return issues


def _check_classification(
    document: PostDocument,
    post_id: str,
) -> list[ValidationIssue]:
    raw = document.data.get("classification")

    if raw is None:
        return []

    if not isinstance(raw, dict):
        return [
            ValidationIssue(
                level=LEVEL_ERROR,
                post_id=post_id,
                message="'classification' must be a JSON object",
            )
        ]

    issues: list[ValidationIssue] = []

    domain = raw.get("domain")

    if domain is not None and not isinstance(domain, str):
        issues.append(
            ValidationIssue(
                level=LEVEL_ERROR,
                post_id=post_id,
                message="classification.domain must be a string",
            )
        )

    primary = raw.get("primary_topic")

    if primary is not None and not isinstance(primary, str):
        issues.append(
            ValidationIssue(
                level=LEVEL_ERROR,
                post_id=post_id,
                message="classification.primary_topic must be a string",
            )
        )

    secondary = raw.get("secondary_topics")

    if secondary is not None:
        if not isinstance(secondary, list) or not all(
            isinstance(topic, str) for topic in secondary
        ):
            issues.append(
                ValidationIssue(
                    level=LEVEL_ERROR,
                    post_id=post_id,
                    message=(
                        "classification.secondary_topics must be a "
                        "list of strings"
                    ),
                )
            )

    relevant = raw.get("interview_relevant")

    if relevant is not None and not isinstance(relevant, bool):
        issues.append(
            ValidationIssue(
                level=LEVEL_ERROR,
                post_id=post_id,
                message=(
                    "classification.interview_relevant must be true "
                    "or false"
                ),
            )
        )

    return issues


def _check_unknown_keys(
    document: PostDocument,
    post_id: str,
) -> list[ValidationIssue]:
    """
    Flag keys the pipeline does not read.

    A warning rather than an error: a typo such as `ai-analsis` would
    otherwise sit unnoticed in a committed file, while a future field
    must not break today's tooling.
    """

    unknown = [
        key
        for key in document.data
        if key not in DOCUMENT_KEYS
    ]

    if not unknown:
        return []

    return [
        ValidationIssue(
            level=LEVEL_WARNING,
            post_id=post_id,
            message=(
                "unrecognised top-level "
                f"key(s): {', '.join(sorted(unknown))}; nothing in "
                "the pipeline reads them"
            ),
        )
    ]


def _undeclared_media_issues(
    document: PostDocument,
    directory: Path,
) -> list[ValidationIssue]:
    """
    Warn about media on disk that post.json does not declare.

    Those files are still ingested, because the loader discovers them,
    so this is never an error. Declaring them is what records a
    description and keeps the manifest honest.
    """

    declared = {
        entry.path.rsplit("/", 1)[-1] for entry in document.media()
    }

    undeclared = [
        path.name
        for path in media_files(directory)
        if path.name not in declared
    ]

    if not undeclared:
        return []

    listed = ", ".join(
        undeclared[:MAX_LISTED_IN_MESSAGE]
    )

    if len(undeclared) > MAX_LISTED_IN_MESSAGE:
        listed += (
            f" and {len(undeclared) - MAX_LISTED_IN_MESSAGE} more"
        )

    return [
        ValidationIssue(
            level=LEVEL_WARNING,
            post_id=directory.name,
            message=(
                f"undeclared media file(s): {listed}; they are still "
                "ingested, but declaring them records a description"
            ),
        )
    ]


# ---------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------


def describe_report(report: ValidationReport) -> str:
    """A one-line summary of a validation run."""

    if report.ok:
        return (
            f"{len(report.posts)} post(s) valid, "
            f"{len(report.warnings)} warning(s)"
        )

    return (
        f"{len(report.posts)} post(s) checked, "
        f"{len(report.errors)} error(s), "
        f"{len(report.warnings)} warning(s)"
    )
