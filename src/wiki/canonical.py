"""
Validation of the canonical knowledge base produced by aggregation.

Only the fields the public site is allowed to expose are modelled here.
`skipped_files` is deliberately not modelled: it contains worker job
manifest paths and must never reach the published site. Pydantic drops
unknown keys, so those entries cannot leak through this object.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from src.models import KnowledgePost


class WikiError(RuntimeError):
    """Raised when the wiki cannot be generated safely."""


class CanonicalStats(BaseModel):
    """Aggregation counters. Not rendered on the site."""

    result_files_found: int = 0
    posts_aggregated: int = 0
    files_skipped: int = 0


class CanonicalKnowledgeBase(BaseModel):
    """
    The canonical `knowledge_base.json` contract.

    Post payloads reuse `src.models.KnowledgePost` so the wiki can
    never drift from the schema the workers actually produce.
    """

    schema_version: int = 1
    generated_at: str | None = None
    stats: CanonicalStats = Field(
        default_factory=CanonicalStats
    )
    posts: list[KnowledgePost] = Field(
        default_factory=list
    )


def load_canonical(
    input_path: str | Path,
) -> CanonicalKnowledgeBase:
    """
    Read and validate a canonical knowledge base.

    Raises WikiError with an actionable message for a missing file,
    unreadable file, malformed JSON, or a payload that does not match
    the canonical contract.
    """

    path = Path(input_path)

    if not path.exists():
        raise WikiError(
            f"Knowledge base not found: {path}. Generate one first, "
            f"for example: python -m src.aggregation.aggregator "
            f"--input-dir <dir> --output <path>"
        )

    if path.is_dir():
        raise WikiError(
            f"Knowledge base path is a directory, expected a JSON "
            f"file: {path}"
        )

    try:
        raw_text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise WikiError(
            f"Could not read knowledge base {path}: {exc}"
        ) from exc

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise WikiError(
            f"Knowledge base is not valid JSON ({path}): {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise WikiError(
            f"Knowledge base must be a JSON object with a 'posts' "
            f"list, got {type(payload).__name__}: {path}"
        )

    if "posts" not in payload:
        raise WikiError(
            f"Knowledge base is missing the required 'posts' list: "
            f"{path}"
        )

    if not isinstance(payload["posts"], list):
        raise WikiError(
            f"Knowledge base 'posts' must be a list, got "
            f"{type(payload['posts']).__name__}: {path}"
        )

    try:
        knowledge_base = CanonicalKnowledgeBase.model_validate(
            payload
        )
    except ValidationError as exc:
        raise WikiError(
            f"Knowledge base failed validation ({path}):\n{exc}"
        ) from exc

    _reject_duplicate_post_ids(knowledge_base)

    return knowledge_base


def _reject_duplicate_post_ids(
    knowledge_base: CanonicalKnowledgeBase,
) -> None:
    """Duplicate IDs would collapse into one page and lose content."""

    seen: set[str] = set()
    duplicates: list[str] = []

    for post in knowledge_base.posts:
        if post.id in seen:
            duplicates.append(post.id)
        seen.add(post.id)

    if duplicates:
        raise WikiError(
            f"Knowledge base contains duplicate post IDs: "
            f"{', '.join(sorted(set(duplicates)))}"
        )
