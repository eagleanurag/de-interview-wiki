"""
The revision pages: what a person preparing for an interview actually reads.

Four pages and one repeating shape. Home answers "where do I start", the
subjects index answers "what is covered", a subtopic page answers "what
do I revise on this", the questions index answers "let me browse", and
search answers "I know the word, not the place".

The old site also had concepts, technologies, saved items, per-post
archive pages and a post index. Those are still generated -- they hold
real material and deleting them would throw away work -- but none of them
is in the primary navigation, because a reader revising for an interview
does not choose between "Concepts" and "Technologies" before they know
what they are looking for.

**Counts are revision aids, not inventory.** A subtopic shows how many
questions it holds because that tells you whether it is worth your time.
Post identifiers, OCR counts, hashes and enrichment fingerprints are not
shown at all: they describe how the site was built, and a reader does not
need them to revise.

**Answers expand with ``<details>``.** No framework, no JavaScript. The
whole interaction is one semantic element the browser already implements.

**Provenance is present but out of the way.** A question shows "seen in N
posts", and the originals are behind one more click. That is enough to
check a claim without making the archive the interface.
"""

from __future__ import annotations

import re

from src.wiki.components import esc
from src.wiki.curriculum import (
    ANSWER_EXCERPT,
    ANSWER_INSUFFICIENT,
    Curriculum,
    RevisionQuestion,
    SubtopicNode,
)
from src.wiki.layout import render_document
from src.wiki.naming import href


#: Shown above every question. A reader who does not know this should be
#: able to trust it, and the wording says plainly what it is.
CAVEAT = (
    "Questions and answers here are generated from LinkedIn posts "
    "captured for revision. Wording is sometimes imperfect, and the "
    "answers are a study aid rather than a source of truth."
)

#: How many concepts a subtopic page lists before deferring the rest.
#: A page with a thousand labels is the archive again, which is the thing
#: this design exists to stop.
CONCEPTS_ON_PAGE = 40

#: Questions shown on a subtopic page before deferring.
QUESTIONS_ON_PAGE = 60


def subject_page(major_slug: str, subtopic_slug: str) -> str:
    return f"topics/{major_slug}/{subtopic_slug}.html"


def subject_index_page() -> str:
    return "subjects.html"


def questions_page() -> str:
    return "questions.html"


def _collapse(text: object) -> str:
    """Whitespace collapsed to single spaces."""

    return re.sub(r"\s+", " ", str(text or "")).strip()


def _count(n: int, singular: str, plural: str | None = None) -> str:
    if n == 1:
        return f"1 {singular}"

    return f"{n:,} {plural or singular + 's'}"


def _json_list(values) -> str:
    """
    A JSON array, escaped once.

    Escaping happens here and *not* again at the call site. Doing it in
    both places produced ``&amp;quot;`` in the attribute, which is
    invisible in a rendered page and breaks any script that reads the
    value back.
    """

    import json

    return esc(json.dumps(list(values), ensure_ascii=False))


TYPE_LABELS = {
    "theory": "Theory",
    "coding": "Coding",
    "scenario": "Scenario",
    "architecture": "Architecture",
    "troubleshooting": "Troubleshooting",
}


def _type_label(value: str) -> str:
    """A question type, spelled out.

    Falls back to the raw value for anything the schema does not
    define, so an unexpected type renders rather than disappearing.
    """

    key = str(value or "").strip().lower()

    return TYPE_LABELS.get(key, str(value).capitalize() or "Question")


def _difficulty_badge(difficulty: str) -> str:
    return (
        '<span class="badge badge-quiet">'
        f"{esc(str(difficulty).capitalize())}</span>"
    )


def _question_block(
    question: RevisionQuestion, page: str
) -> str:
    """
    One question and its answer, collapsed until asked for.

    The summary is the question text. The answer is inside the
    ``<details>`` so the page is scannable -- a subtopic with sixty
    questions should read as a list of sixty questions, not sixty
    paragraphs.
    """

    sources = _count(question.source_count, "source")

    meta = [_difficulty_badge(question.difficulty)]

    if question.source_count > 1:
        meta.append(f'<span class="muted">{esc(sources)}</span>')

    answer = question.answer

    if answer.source == ANSWER_INSUFFICIENT:
        body = (
            '<p class="muted">No answer is available for this question. '
            "It was recovered from a slide image, and there was not enough "
            "text on the slide to answer it from.</p>"
        )
    elif answer.is_thin:
        # Stated rather than padded. A one-line answer written to fill
        # the space teaches nothing and reads as though it teaches
        # something.
        body = (
            f'<div class="prose">{esc(answer.text)}</div>'
            '<p class="muted answer-thin">This answer is very short. It is '
            "shown as captured rather than expanded, because expanding it "
            "would mean inventing the missing part.</p>"
        )
    else:
        body = f'<div class="prose">{esc(answer.text)}</div>'

        if answer.source == ANSWER_EXCERPT:
            body += (
                '<p class="muted">Quoted from the slide this question was '
                "read from, rather than written as an answer to it.</p>"
            )

    # The data-* attributes are the contract the questions page filter
    # has always had, and dropping them silently broke filtering rather
    # than reporting it. They are the same values the old cards carried.
    attributes = (
        f' data-difficulty="{esc(question.difficulty)}"'
        f' data-type="{esc(question.type)}"'
        f' data-topics="{_json_list(question.topics)}"'
    )

    meta = [
        f'<span class="badge badge-quiet">{esc(_type_label(question.type))}</span>'
    ] + meta

    return (
        f'<details class="question"{attributes}>'
        f"<summary><span class=\"question-text\">{esc(question.text)}</span>"
        f'<span class="badge-row">{_join(meta)}</span></summary>'
        f'<div class="question-body">{body}'
        f"{_provenance(question, page)}"
        "</div></details>"
    )


def _pluralise(label: str) -> str:
    """
    Pluralise the head of a source description, not the whole string.

    ``"LinkedIn post, saved from a list"`` reads as a noun phrase with a
    qualifier behind a comma, so the ``s`` belongs on the noun in front
    of it. Appending to the end gave "saved from a lists".
    """

    head, comma, qualifier = label.partition(",")

    return f"{head}s{comma}{qualifier}"


def _provenance(question: RevisionQuestion, page: str) -> str:
    """
    Where a question came from, behind a click.

    The post identifiers stay exactly where they were -- every question
    keeps its full ``post_ids`` tuple, and nothing about how a question
    is filed or merged has changed. They are simply not printed.

    ``urn-li-archive-c7812291ac93dd56`` is an internal key. A reader
    cannot type it back to anything, cannot ask it a question, and sees
    it as a defect on a page otherwise written for people; it was the
    one place the revision pages leaked the implementation. What can be
    told instead is true and useful: how many posts the question was
    asked in, and what kind of source each one was.
    """

    if not question.post_ids:
        return ""

    quantity = question.source_count
    sources = question.sources

    tally: dict[str, int] = {}

    for label in sources:
        tally[label] = tally.get(label, 0) + 1

    if len(tally) == 1:
        (label, count), = tally.items()

        # "1 LinkedIn post" and "3 LinkedIn posts". The label is a noun
        # phrase that may carry a qualifier after a comma
        # ("LinkedIn post, saved from a list"), so only the head is
        # pluralised -- appending to the whole string produced "saved
        # from a lists".
        plural = label if count == 1 else _pluralise(label)

        sentence = f"This question appears in {esc(f'{count} {plural}')}."

    else:
        # Mixed provenance: name each kind rather than repeating one.
        described = ", ".join(
            sorted(
                f"{esc(label)} ({esc(str(count))})"
                for label, count in tally.items()
            )
        )

        sentence = (
            f"This question appears in "
            f"{esc(_count(quantity, 'post'))}, from {described}."
        )

    return (
        '<details class="provenance">'
        "<summary>Show source</summary>"
        f'<p class="muted">{sentence}</p>'
        "</details>"
    )


def _join(parts: list[str]) -> str:
    return "".join(parts)


def _concept_list(node: SubtopicNode) -> str:
    """Concepts under a subtopic, capped and with the remainder counted."""

    concepts = sorted(
        node.concepts.values(), key=lambda item: (-len(item.post_ids), item.label)
    )

    if not concepts:
        return ""

    shown = concepts[:CONCEPTS_ON_PAGE]
    hidden = len(concepts) - len(shown)

    badges = "".join(
        '<span class="badge badge-quiet">'
        f"{esc(concept.label)}</span>"
        for concept in shown
    )

    more = ""
    if hidden > 0:
        more = (
            f'<p class="muted">{_count(hidden, "further concept")} on '
            "this subtopic are listed in the full concepts index.</p>"
        )

    return (
        '<div class="subblock">'
        f"<h3>Concepts ({_count(len(concepts), 'concept')})</h3>"
        f'<div class="badge-row">{badges}</div>'
        f"{more}</div>"
    )


def render_subject(curriculum: Curriculum, node: SubtopicNode) -> str:
    """One subtopic: its concepts, its questions, and where to go next."""

    questions = curriculum.questions_in(
        node.major.slug, node.subtopic.slug
    )

    shown = questions[:QUESTIONS_ON_PAGE]
    hidden = len(questions) - len(shown)

    blocks = [_question_block(question, subject_page(
        node.major.slug, node.subtopic.slug
    )) for question in shown]

    if hidden > 0:
        blocks.append(
            f'<p class="muted">{_count(hidden, "further question")} on '
            f"this subtopic are in the <a href=\""
            f'{esc(href(subject_page(node.major.slug, node.subtopic.slug), questions_page()))}'
            '">questions index</a>.</p>'
        )

    related = curriculum.related(node)

    related_html = ""
    if related:
        links = "".join(
            f'<li><a class="text-link" href="{esc(href(subject_page(node.major.slug, node.subtopic.slug), subject_page(other.major.slug, other.slug)))}">'
            f"{esc(other.subtopic.name)}</a></li>"
            for other in related
        )
        related_html = (
            '<div class="subblock"><h3>Related</h3>'
            f'<ul class="plain-list">{links}</ul></div>'
        )

    breadcrumb = (
        '<nav class="breadcrumb" aria-label="Breadcrumb">'
        f'<a href="{esc(href(subject_page(node.major.slug, node.subtopic.slug), "index.html"))}">Home</a>'
        " &rsaquo; "
        f'<a href="{esc(href(subject_page(node.major.slug, node.subtopic.slug), subject_index_page()))}">Subjects</a>'
        f" &rsaquo; {esc(node.major.name)}"
        "</nav>"
    )

    body = "".join(
        [
            breadcrumb,
            f"<h1>{esc(node.major.name)} &rsaquo; {esc(node.subtopic.name)}</h1>",
            f'<p class="lede">{esc(node.major.blurb)}</p>',
            f'<p class="counts">{_count(node.question_count, "question")} '
            f"&middot; {_count(len(node.concepts), 'concept')} "
            f"&middot; {_count(node.post_count, 'source')}</p>",
            _concept_list(node),
            '<div class="subblock">'
            f"<h2>Interview questions ({_count(node.question_count, 'question')})</h2>"
            f"{''.join(blocks) or '<p class=\"muted\">No questions yet.</p>'}"
            "</div>",
            related_html,
            f'<p class="muted caveat">{esc(CAVEAT)}</p>',
        ]
    )

    return render_document(
        page=subject_page(node.major.slug, node.subtopic.slug),
        title=f"{node.major.name} / {node.subtopic.name}",
        description=(
            f"{node.subtopic.name} questions and concepts under "
            f"{node.major.name}, for data engineering interview revision."
        ),
        body=body,
    )


def render_subject_index(curriculum: Curriculum) -> str:
    """Every subject, with what is inside it."""

    blocks = []

    for major in curriculum.majors:
        rows = []

        for node in major.subtopics:
            rows.append(
                '<li class="subject-row">'
                f'<a class="text-link" href="{esc(href(subject_index_page(), subject_page(node.major.slug, node.slug)))}">'
                f"{esc(node.subtopic.name)}</a>"
                f'<span class="muted">{_count(node.question_count, "question")}'
                f" &middot; {_count(len(node.concepts), 'concept')}</span>"
                "</li>"
            )

        blocks.append(
            '<section class="subject-card">'
            f"<h2>{esc(major.major.name)}</h2>"
            f'<p class="muted">{esc(major.major.blurb)}</p>'
            f'<p class="counts">{_count(major.question_count, "question")} '
            f"&middot; {_count(major.post_count, "source")}</p>"
            f'<ul class="plain-list">{"".join(rows)}</ul>'
            "</section>"
        )

    body = "".join(
        [
            "<h1>Subjects</h1>",
            '<p class="lede">Everything here, grouped the way you would '
            "revise it rather than the way it was captured.</p>",
            "".join(blocks),
            f'<p class="muted caveat">{esc(CAVEAT)}</p>',
        ]
    )

    return render_document(
        page=subject_index_page(),
        title="Subjects",
        description=(
            "Data engineering interview revision, grouped by subject "
            "and subtopic."
        ),
        body=body,
    )


def _filter_panel(curriculum: Curriculum, page: str) -> str:
    """
    Topic, type and difficulty filters, built from the data.

    Kept from the questions page this model replaced. Grouping questions
    by subject answers "what should I revise"; a difficulty filter answers
    "what can I be asked right now", and the two are worth having
    together. Every option is a value that occurs, so the panel never
    offers a filter that would return nothing.
    """

    topics: list[str] = []
    types: list[str] = []
    difficulties: list[str] = []

    for question in curriculum.questions.values():
        for label in question.topics:
            if label not in topics:
                topics.append(label)

        if question.type not in types:
            types.append(question.type)

        if question.difficulty not in difficulties:
            difficulties.append(question.difficulty)

    def options(values: list[str]) -> str:
        return "".join(
            f'<option value="{esc(value)}">{esc(value)}</option>'
            for value in sorted(values)
        )

    return (
        '<div class="filter-panel">'
        '<label for="filter-topic">Topic</label>'
        f'<select id="filter-topic" name="topic">{options(topics)}</select>'
        '<label for="filter-type">Type</label>'
        f'<select id="filter-type" name="type">'
        f'{options(types)}</select>'
        '<label for="filter-difficulty">Difficulty</label>'
        f'<select id="filter-difficulty" name="difficulty">'
        f'{options(difficulties)}</select>'
        "</div>"
    )


def render_questions(curriculum: Curriculum, page: str = "questions.html") -> str:
    """Every question, grouped by subject and subtopic."""

    blocks = []

    for major in curriculum.majors:
        if not major.question_count:
            continue

        sections = []

        for node in major.subtopics:
            questions = curriculum.questions_in(major.slug, node.slug)

            if not questions:
                continue

            items = "".join(
                _question_block(question, page) for question in questions
            )

            sections.append(
                f"<h3>{esc(node.subtopic.name)} "
                f'<span class="muted">{_count(len(questions), "question")}</span></h3>'
                f"{items}"
            )

        if sections:
            blocks.append(
                f'<section class="subject-card"><h2>{esc(major.major.name)}</h2>'
                f"{''.join(sections)}</section>"
            )

    if not blocks:
        blocks.append(
            '<p class="empty-state">No interview questions yet.</p>'
        )

    body = "".join(
        [
            "<h1>Interview questions</h1>",
            '<p class="lede">Every question, filed once. Answers open '
            "in place.</p>",
            _filter_panel(curriculum, page),
            "".join(blocks),
            f'<p class="muted caveat">{esc(CAVEAT)}</p>',
        ]
    )

    return render_document(
        page=page,
        title="Interview questions",
        description=(
            "Data engineering interview questions, grouped by subject "
            "and subtopic."
        ),
        body=body,
    )


def _recent_posts(model, page: str, limit: int = 6) -> str:
    """
    A short list of recently captured posts.

    Kept from the home page this replaced, and kept small. A reader who
    has just exported a batch wants to see it arrived; a reader revising
    for an interview wants the subjects above. Six titles, no counts and
    no diagnostics, is the part of the archive that a revision site can
    honestly keep.
    """

    if not model.posts:
        return (
            '<div class="subblock"><h2>Recently captured</h2>'
            '<p class="empty-state">No posts yet. Run the discovery stage '
            "to import captured posts.</p></div>"
        )

    rows = []

    for post, slug in list(zip(model.posts, model.post_slugs))[-limit:]:
        title = _collapse(
            getattr(post.ai_analysis, "summary", "")
            or post.original_text
            or slug
        )[:90]

        rows.append(
            '<li><a class="text-link" '
            f'href="{esc(href(page, f"posts/{slug}.html"))}">{esc(title)}</a></li>'
        )

    return (
        '<div class="subblock"><h2>Recently captured</h2>'
        f'<ul class="plain-list">{"".join(rows)}</ul></div>'
    )


def _stat_row(curriculum: Curriculum) -> str:
    """
    Counts that help decide where to start.

    Subjects, subtopics and questions. Not posts, topics or concepts:
    those count what the corpus *is*, which is inventory rather than
    guidance, and an archive count on the front page is how this site
    came to feel like an archive.
    """

    tiles = (
        ("Subjects", curriculum.major_count),
        ("Subjects and areas", curriculum.subtopic_count),
        ("Interview questions", curriculum.question_count),
    )

    cells = "".join(
        f'<p class="stat-value">{value}</p>'
        f'<p class="stat-label">{esc(label)}</p>'
        for label, value in tiles
    )

    return f'<div class="stat-row">{cells}</div>'


def render_home(
    curriculum: Curriculum,
    model=None,
    page: str = "index.html",
) -> str:
    """
    The front door.

    A title, one sentence about what this is, and the subjects. Counts
    appear where they help decide where to start and nowhere else -- no
    post counts, no OCR counts, no build metadata.
    """

    cards = []

    for major in curriculum.majors:
        cards.append(
            '<a class="subject-tile" '
            f'href="{esc(href(page, subject_page(major.slug, major.subtopics[0].slug)))}">'
            f"<h2>{esc(major.major.name)}</h2>"
            f'<p class="muted">{esc(major.major.blurb)}</p>'
            f'<p class="counts">{_count(major.question_count, "question")} '
            f"&middot; {len(major.subtopics)} subjects</p>"
            "</a>"
        )

    body = "".join(
        [
            "<h1>Data Engineering Interview Wiki</h1>",
            '<p class="lede">Revision material for data engineering '
            "interviews, organised so you can walk from a subject down to "
            "the questions that test it.</p>",
            _stat_row(curriculum),
            f'<div class="subject-tiles">{"".join(cards)}</div>',
            '<div class="subblock">'
            '<h2>Start here</h2>'
            '<ul class="plain-list">'
            f'<li><a class="text-link" href="{esc(href(page, subject_index_page()))}">'
            "Browse every subject and subtopic</a></li>"
            f'<li><a class="text-link" href="{esc(href(page, questions_page()))}">'
            "Browse every question</a></li>"
            "</ul></div>",
            _recent_posts(model, page) if model is not None else "",
            f'<p class="muted caveat">{esc(CAVEAT)}</p>',
        ]
    )

    return render_document(
        page=page,
        title="Data Engineering Interview Wiki",
        description=(
            "Data engineering interview revision: SQL, Spark, "
            "Databricks, Python, Azure Data Factory, AWS and more, with "
            "grouped questions and answers."
        ),
        body=body,
    )


def curriculum_pages(curriculum: Curriculum, model=None) -> dict[str, str]:
    """Every reader-facing page, keyed by its output path."""

    pages = {
        "index.html": render_home(curriculum, model),
        subject_index_page(): render_subject_index(curriculum),
        questions_page(): render_questions(curriculum),
    }

    for major in curriculum.majors:
        for node in major.subtopics:
            pages[subject_page(node.major.slug, node.slug)] = (
                render_subject(curriculum, node)
            )

    return pages


__all__ = [
    "CAVEAT",
    "CONCEPTS_ON_PAGE",
    "QUESTIONS_ON_PAGE",
    "curriculum_pages",
    "questions_page",
    "render_home",
    "render_questions",
    "render_subject",
    "render_subject_index",
    "subject_index_page",
    "subject_page",
]
