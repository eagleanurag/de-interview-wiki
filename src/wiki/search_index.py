"""
The search index, built over revision units rather than over the corpus.

It used to index 490 posts, 2,573 topics, 3,112 concepts, 44
technologies and 2,016 questions -- 11,616 records, every one of them
pointing at a page. That was true while the site had those pages. It no
longer does, so the index has to describe what is published: 22 revision
units and the 1,902 questions on them.

Two record kinds survive.

**A unit** (``b``) -- its title, its one-sentence summary, the concepts it
comes back to, and the sections it covers. This is what a search for
"broadcast join" should land on: Spark and PySpark, because "Broadcast
Hash Join" is one of its concepts.

**A question** (``q``) -- its text and its answer, pointing at the unit
page it is answered on. Answers are indexed because "how do I find
consecutive rows" is often how someone remembers a question they cannot
otherwise place, and because the answer is where the useful text is.

Records for posts, topics, concepts and technologies are gone. Not
filtered out: they pointed at pages that are not generated, and an
index entry that leads nowhere is worse than a missing one.

The OCR index is written separately and is unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.wiki.revision import _collapse
from src.wiki.revision_model import Unit, Units

INDEX_FORMAT_VERSION = 3

#: How much of a question's answer is carried. Enough for a search hit
#: to be recognisable, short enough that the index stays small -- it is
#: fetched whole on the first search.
INDEX_ANSWER_LIMIT = 400

#: How many concepts a unit record carries.
#:
#: All of them, deliberately. Capping at the top 24 was tried and
#: measured: 521 concepts were reachable instead of 3,056, so a search
#: for "Merge Tree" or "Z-Order" found nothing at all. Concept labels are
#: the vocabulary the corpus thinks in, and a label a reader can type but
#: not search is the worst of both worlds -- it looks supported and is
#: not findable. The extra weight is about 80 KB of short strings on an
#: index already fetched whole.
INDEX_UNIT_CONCEPTS = 0

#: Kept so the record shape the browser script reads is unchanged.
KIND_UNIT = "b"
KIND_QUESTION = "q"


def _unit_record(unit: Unit) -> dict:
    # Every concept, or none of them -- see INDEX_UNIT_CONCEPTS.
    concepts = (
        unit.top_concepts(INDEX_UNIT_CONCEPTS)
        if INDEX_UNIT_CONCEPTS
        else sorted(unit.concepts)
    )

    return {
        "i": unit.title,
        "u": f"revision/{unit.slug}.html",
        "k": KIND_UNIT,
        "p": unit.group,
        "a": "",
        "d": "",
        "pb": "",
        "s": unit.summary,
        # The concepts the unit comes back to. "Broadcast Hash Join" is
        # how a search for "broadcast join" finds this unit; without it
        # the unit is only findable by its own title.
        "c": concepts,
        "tp": [unit.group],
        "sb": unit.sections,
        "q": [],
        "t": [],
        "n": unit.question_count,
        "bc": unit.title,
    }


def _question_record(unit: Unit, question, path: str) -> dict:
    answer = _collapse(question.answer.text)[:INDEX_ANSWER_LIMIT]

    # The subtopic the question was placed in, which is finer than the
    # unit and is what a reader searching for it is likely to know. It is
    # also what search.js reads for the "topics" filter.
    topic = question.subtopic_slug.replace("-", " ")

    return {
        "i": question.text,
        "u": path,
        "k": KIND_QUESTION,
        "p": unit.group,
        "a": "",
        "d": "",
        "pb": "",
        "s": answer,
        "c": [],
        "tp": [unit.group],
        "sb": [topic],
        "q": [question.text],
        "t": [],
        "n": 1,
        "bc": f"{unit.title} / {topic}",
    }


def build_unit_records(units: Units) -> list[dict]:
    """
    Every unit, and every question on it.

    Order is unit order then question text, so the index is byte-for-byte
    reproducible and two builds can be compared.
    """

    records: list[dict] = []

    for unit in units.units:
        records.append(_unit_record(unit))

        # Which page of the unit each question actually sits on, so the
        # result links to the answer rather than to page one.
        for page in unit.pages():
            for question in sorted(page.questions, key=lambda q: q.text):
                records.append(_question_record(unit, question, page.path))

    return records


def build_index(units: Units) -> dict:
    """The index document, including the counts the UI shows."""

    records = build_unit_records(units)

    return {
        "v": INDEX_FORMAT_VERSION,
        "units": units.unit_count,
        "pages": units.page_count,
        "questions": units.kept,
        # Retained at zero so anything reading the old keys finds a
        # number it can trust rather than a KeyError. These are the
        # archive families, and nothing in them is published.
        "posts": 0,
        "topics": 0,
        "concepts": 0,
        "technologies": 0,
        "subjects": units.unit_count,
        "subtopics": units.unit_count,
        "revision_questions": units.kept,
        "records": records,
    }


def write_search_index(units: Units, path: Path) -> Path:
    """Write ``search-index.json`` next to the generated pages."""

    payload = json.dumps(
        build_index(units),
        ensure_ascii=False,
        separators=(",", ":"),
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")

    return path