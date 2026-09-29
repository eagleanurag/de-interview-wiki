"""
Compact browser-readable search index.

The index is deliberately small and deliberately narrow. It contains
only what a search result needs to render: identifiers, human labels,
attribution, and the text a reader might search for.

It deliberately does **not** contain aggregation counters, job
manifests, output paths, attempt numbers, or any other pipeline
metadata. Answers are omitted too, because question text is what a
reader searches on and answers would multiply the payload.

Keys are short because this file is fetched by every search.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.wiki.analysis import SiteModel
from src.wiki.components import (
    INDEX_SOURCE_EXCERPT_LIMIT,
    INDEX_SUMMARY_LIMIT,
    collapse_whitespace,
    truncate,
)
from src.wiki.naming import post_page


INDEX_FORMAT_VERSION = 1


def build_search_records(
    model: SiteModel,
) -> list[dict]:
    """
    Build one compact record per post.

    Post order follows the model's canonical ID order, so the index is
    byte-for-byte reproducible for a given input.
    """

    records = []

    for post, slug in zip(model.posts, model.post_slugs):
        source = post.source

        records.append(
            {
                # Identifier and where to go.
                "i": post.id,
                "u": post_page(slug),
                # Attribution, so results stay sourced.
                "p": source.platform,
                "a": source.author or "",
                "d": source.captured_at.strftime("%Y-%m-%d"),
                # Searchable content.
                "s": _text(post.ai_analysis.summary, INDEX_SUMMARY_LIMIT),
                "x": _text(post.original_text, INDEX_SOURCE_EXCERPT_LIMIT),
                "tp": list(post.ai_analysis.topics),
                "sb": list(post.ai_analysis.subtopics),
                "c": list(post.ai_analysis.concepts),
                "q": [
                    collapse_whitespace(question.question)
                    for question in post.interview_questions
                ],
                # Cheap counters for result cards.
                "n": len(post.interview_questions),
            }
        )

    return records


def build_search_index(model: SiteModel) -> dict:
    """The full index document, including counts used by the UI."""

    records = build_search_records(model)

    return {
        "v": INDEX_FORMAT_VERSION,
        "posts": len(records),
        "questions": sum(record["n"] for record in records),
        "records": records,
    }


def write_search_index(
    model: SiteModel,
    path: Path,
) -> Path:
    """Write `search-index.json` next to the generated pages."""

    payload = json.dumps(
        build_search_index(model),
        ensure_ascii=False,
        separators=(",", ":"),
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{payload}\n", encoding="utf-8")

    return path


def _text(value: object, limit: int) -> str:
    excerpt, _ = truncate(value, limit)

    return excerpt
