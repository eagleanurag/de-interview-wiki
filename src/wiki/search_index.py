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
from src.wiki.naming import (
    concept_page,
    post_page,
    technology_page,
    topic_page,
)


INDEX_FORMAT_VERSION = 2

#: What a record points at. A reader searching "delta lake" should find
#: the post, the topic, the concept and the technology, not only the
#: post.
KIND_POST = "p"
KIND_TOPIC = "t"
KIND_CONCEPT = "c"
KIND_TECHNOLOGY = "x"


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
                "k": KIND_POST,
                # Attribution, so results stay sourced.
                "p": source.platform,
                "a": source.author or "",
                "d": source.captured_at.strftime("%Y-%m-%d"),
                # When the source published it, as rendered. Kept
                # separate from "d" because a relative form is not a
                # date, and overwriting one with the other would lose
                # information.
                "pb": source.published_at or "",
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

    records.extend(_group_records(model))

    return records


def _group_records(model: SiteModel) -> list[dict]:
    """
    One record per topic, concept and technology.

    Search that only covers posts cannot find a concept that appears
    in several posts, because nothing links the posts together except
    the concept itself. These records are what make the consolidated
    knowledge reachable by name.

    Answers are still excluded, for the same reason as on posts: it
    is the label a reader searches for.
    """

    records: list[dict] = []

    for topic in model.topics:
        records.append(
            {
                "i": f"topic:{topic.label}",
                "u": topic.page,
                "k": KIND_TOPIC,
                "p": "wiki",
                "a": "",
                "d": "",
                "pb": "",
                "s": "",
                "x": "",
                "tp": [topic.label],
                "sb": [],
                "c": list(topic.concepts),
                "q": [],
                "n": topic.question_count,
            }
        )

    for concept in model.concept_entries:
        records.append(
            {
                "i": f"concept:{concept.label}",
                "u": concept.page,
                "k": KIND_CONCEPT,
                "p": "wiki",
                "a": "",
                "d": "",
                "pb": "",
                "s": "",
                "x": "",
                "tp": list(concept.topics),
                "sb": [],
                "c": [concept.label],
                "q": [],
                "n": 0,
            }
        )

    for technology in model.technology_entries:
        records.append(
            {
                "i": f"technology:{technology.label}",
                "u": technology.page,
                "k": KIND_TECHNOLOGY,
                "p": "wiki",
                "a": "",
                "d": "",
                "pb": "",
                "s": "",
                "x": "",
                "tp": list(technology.topics),
                "sb": [],
                "c": [technology.label],
                "q": [],
                "n": technology.question_count,
            }
        )

    return records


def build_search_index(model: SiteModel) -> dict:
    """The full index document, including counts used by the UI."""

    records = build_search_records(model)

    by_kind: dict[str, int] = {}

    for record in records:
        kind = record.get("k", KIND_POST)
        by_kind[kind] = by_kind.get(kind, 0) + 1

    return {
        "v": INDEX_FORMAT_VERSION,
        "posts": sum(
            1 for record in records
            if record.get("k", KIND_POST) == KIND_POST
        ),
        "topics": by_kind.get(KIND_TOPIC, 0),
        "concepts": by_kind.get(KIND_CONCEPT, 0),
        "technologies": by_kind.get(KIND_TECHNOLOGY, 0),
        "questions": sum(
            record["n"]
            for record in records
            if record.get("k", KIND_POST) == KIND_POST
        ),
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
