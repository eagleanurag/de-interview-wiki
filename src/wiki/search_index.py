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

from src.wiki.curriculum import build_curriculum
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

#: Curriculum kinds. A reader searching "broadcast join" wants the
#: subject, the subtopic and the questions -- in that order of
#: usefulness -- not a post record that happens to mention it.
KIND_SUBJECT = "s"
KIND_SUBTOPIC = "b"
KIND_REVISION_QUESTION = "q"


def build_search_records(
    model: SiteModel,
) -> list[dict]:
    """
    Build one compact record per post.

    Post order follows the model's canonical ID order, so the index is
    byte-for-byte reproducible for a given input.
    """

    records = []

    # Technologies by the post that carries them. Without this a search
    # for "Azure Data Factory" finds the technology page and nothing
    # else, so the posts that actually discuss it stay unreachable by the
    # name a reader would type.
    technologies_by_slug: dict[str, list[str]] = {}

    for technology in model.technology_entries:
        for slug in technology.post_slugs:
            technologies_by_slug.setdefault(slug, []).append(
                technology.label
            )

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
                "t": sorted(technologies_by_slug.get(slug, [])),
                "q": [
                    collapse_whitespace(question.question)
                    for question in post.interview_questions
                ],
                # Cheap counters for result cards.
                "n": len(post.interview_questions),
            }
        )

    # The standalone topic, concept and technology records used to be
    # added here, 8,942 of them. Their pages are no longer generated, so
    # indexing them would have left the search index pointing at files
    # that do not exist -- a broken link behind every one.
    #
    # Nothing is lost as searchable text. Concepts, topics, subtopics and
    # technologies are still carried on each post record above (``c``,
    # ``tp``, ``sb``, ``t``), so "broadcast join" still matches the posts
    # and the technologies that discuss it; they simply no longer have a
    # page of their own to send a reader to.
    #
    # The revision pages are what a reader actually searches against,
    # and the JS already ranks them above the post records.

    curriculum = build_curriculum(list(model.posts))
    revision = build_curriculum_records(curriculum)
    records.extend(revision)

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
                "t": [],
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
                "t": [],
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
                "t": [technology.label],
                "q": [],
                "n": technology.question_count,
            }
        )

    return records


def build_curriculum_records(curriculum) -> list[dict]:
    """
    Subjects, subtopics and revision questions, as search records.

    Answers are indexed as well as question text, because "how do I find
    consecutive rows" is often how someone remembers a question they
    cannot otherwise place. Each record carries the breadcrumb of
    subject and subtopic so a result can say where it lives rather than
    leaving the reader to work it out.

    Concepts are deliberately *not* here. They are labels lifted from
    slide text -- "Running-total DAX measure" next to "No window
    function" -- and indexing three thousand of them put the noise a
    reader typed above the question they wanted. They remain reachable
    from the subtopic pages and the concepts index.
    """

    from src.wiki.revision import questions_page, subject_index_page
    from src.wiki.revision import subject_page as subject_path

    records: list[dict] = []

    for major in curriculum.majors:
        records.append(
            {
                "i": f"subject:{major.major.name}",
                "u": subject_path(major.major.slug,
                                  major.subtopics[0].slug),
                "k": KIND_SUBJECT,
                "p": "wiki",
                "a": "",
                "d": "",
                "pb": "",
                "s": major.major.blurb,
                "x": major.major.name,
                "tp": [major.major.name],
                "sb": [],
                "c": [],
                "t": [],
                "q": [],
                "n": major.question_count,
                # Lets a result read "SQL / Window Functions" rather than
                # "SQL", which is the difference between a result you can
                # act on and one you have to open.
                "bc": major.major.name,
            }
        )

        for node in major.subtopics:
            path = subject_path(node.major.slug, node.slug)

            # The concepts this subtopic teaches, as searchable text.
            #
            # They used to be indexed as 3,112 records of their own --
            # one page each, and the reason the concepts index existed.
            # With those pages gone, a search for "broadcast join" had
            # nowhere to land on Spark / Joins, because "Broadcast
            # joins" and "Broadcast Hash Join" are concept labels and
            # the subtopic record carried none of them. It reached the
            # questions and the posts instead, which is the answer one
            # hop later than the reader needed.
            #
            # Carrying them on the subtopic that teaches them puts the
            # label where it belongs and keeps the noise out: these are
            # the concepts under one subtopic, not three thousand across
            # the site.
            taught = sorted(
                concept.label for concept in node.concepts.values()
            )

            records.append(
                {
                    "i": f"subtopic:{node.major.name}/{node.subtopic.name}",
                    "u": path,
                    "k": KIND_SUBTOPIC,
                    "p": "wiki",
                    "a": "",
                    "d": "",
                    "pb": "",
                    "s": f"{node.major.name}: {node.subtopic.name}",
                    "x": node.subtopic.name,
                    "tp": [node.major.name],
                    "sb": [],
                    "c": taught,
                    "t": [],
                    "q": [],
                    "n": node.question_count,
                    "bc": f"{node.major.name} / {node.subtopic.name}",
                }
            )

    for question in sorted(
        curriculum.questions.values(), key=lambda item: item.text
    ):
        node = curriculum.subtopic(
            question.major_slug, question.subtopic_slug
        )

        crumb = question.subtopic_slug.replace("-", " ")

        if node is not None:
            crumb = f"{node.major.name} / {node.subtopic.name}"

        records.append(
            {
                "i": f"question:{question.id}",
                "u": subject_path(question.major_slug,
                                  question.subtopic_slug),
                "k": KIND_REVISION_QUESTION,
                "p": "wiki",
                "a": "",
                "d": "",
                "pb": "",
                # The question is the label; the answer is the excerpt.
                "s": question.text,
                "x": question.answer.text if question.answer.text else "",
                "tp": [crumb],
                "sb": [],
                "c": [],
                "t": [],
                "q": [],
                "n": 1,
                "bc": crumb,
            }
        )

    del questions_page, subject_index_page

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
        # Curriculum counts, kept separate from "questions" above.
        # That figure counts question slots on post cards; this one
        # counts the merged revision questions. Different numbers for
        # the same corpus, and conflating them is how a search result
        # count stops matching what a reader sees.
        "subjects": by_kind.get(KIND_SUBJECT, 0),
        "subtopics": by_kind.get(KIND_SUBTOPIC, 0),
        "revision_questions": by_kind.get(KIND_REVISION_QUESTION, 0),
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
