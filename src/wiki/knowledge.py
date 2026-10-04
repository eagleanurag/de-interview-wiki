"""
The knowledge page: what a captured post turned out to be about.

This replaces a page that described the *capture* rather than the
knowledge. It was titled ``urn-li-saved-ffccf4f7771b97bd`` and opened
with a table of Platform, Captured, Published, Author and Post ID,
followed by a "Saved item" card explaining how the content arrived, a
"Classification" table of Domain / Interview relevant / Primary topic /
Secondary topics, the original post text, and a media inventory with
filenames.

Every one of those was true and none of it helped anybody revise. The
metadata is not deleted -- it is still in the model, still written to
the knowledge base, still used for traceability, regeneration, auditing
and deduplication. It is simply not what this page is *for*. A reader
who wants the capture record asks for it, once, deliberately.

What is left is a study page: a name for the thing, a sentence about it,
the concepts worth remembering, the questions it raises with their
answers, somewhere to go next, and a one-line note about where it came
from.

The navigation is the hierarchy, not the label dump. "Topics: ...
Subtopics: ... Concepts: ... Classification: ..." was the same
information as four database columns; ``SQL / Joins / Broadcast Join``
is the same information as somewhere to go.
"""

from __future__ import annotations

import re
from datetime import datetime

from src.models import KnowledgePost
from src.wiki.components import (
    definition_list,
    esc,
    safe_link,
    truncate,
)
from src.wiki.curriculum import (
    Curriculum,
    _source_label,
    knowledge_title,
    placement_for,
)
from src.wiki.layout import render_document
from src.wiki.naming import href
from src.redaction import redact
from src.redaction import summarise as redact_note
from src.wiki.naming import (
    CONCEPTS_PAGE,
    INDEX_PAGE,
    QUESTIONS_PAGE,
    post_page,
)
from src.wiki.revision import subject_page

#: Concepts listed before the rest are dropped. A page of forty chips is
#: the badge wall this design exists to avoid.
CONCEPTS_SHOWN = 24

#: How many sibling knowledge pages to offer. Enough to be a next step,
#: few enough that the reader is not choosing between sixty.
RELATED_SHOWN = 6

#: Questions shown. A post contributes a handful; the cap exists so a
#: mis-enriched post cannot bury the page.
QUESTIONS_SHOWN = 40

#: Longest excerpt of the original post reproduced here.
#:
#: Behind a disclosure, but still: a page that prints a whole captured
#: post reproduces the archive on every page, which is the thing this
#: design is removing. The rest is one link away.
SOURCE_EXCERPT_LIMIT = 4000

#: How a capture arrived, in words a reader can act on.
#:
#: Kept because the distinction matters to someone deciding how much to
#: trust a page: material this project was handed is not the same claim
#: as material it went and collected. It used to head a "Saved item"
#: card on the post page. The card is gone -- it was one of five capture
#: sections in a row -- and the distinction moved into the capture
#: record, which is where someone checking provenance actually looks.
CAPTURE_METHOD_LABELS = {
    "user_export": "You exported this from your saved items",
    "user_provided": "You supplied the content for this",
    "user_saved_page": "You saved the page and supplied it here",
    "user_bundle": "You assembled this from your own notes",
}

SOURCE_PLATFORM_LABELS = {
    "linkedin": "LinkedIn",
    "x": "X",
    "twitter": "X",
    "blog": "Blog",
    "web": "Web",
}


def _published_label(post: KnowledgePost) -> str:
    """
    When the post was published, in a form worth reading.

    The capture stores whatever the source gave, which for this corpus
    is an ISO timestamp. Printed as recorded that is
    "2026-10-01T23:25:50Z" -- precise, and unreadable in a line whose
    job is to be glanced at. A timestamp is shortened to its date.

    Anything that is not a timestamp is left exactly as recorded. The
    source may have said "2 days ago", and converting that to a date
    would mean guessing at a reference point the page does not have.
    """

    raw = (post.source.published_at or "").strip()

    if not raw:
        return ""

    try:
        return datetime.fromisoformat(
            raw.replace("Z", "+00:00")
        ).strftime("%Y-%m-%d")

    except ValueError:
        return raw


def _platform(post: KnowledgePost) -> str:
    return SOURCE_PLATFORM_LABELS.get(
        post.source.platform, post.source.platform or "Source"
    )


def _answer_html(question) -> str:
    """
    An answer, with any example it already contains set apart.

    Fenced code is lifted into ``<pre>`` so it is legible and copyable.
    Everything else is prose, and stays prose -- the answer is the
    enricher's words, not rewritten here.

    A quoted passage is labelled. A question recovered from a slide comes
    with the text that was on that slide, and a reader who cannot tell
    that apart from an answer written to the question will quote the
    slide's wording as though it were a considered explanation.
    """

    answer = (question.answer or "").strip()

    if not answer:
        return (
            '<p class="muted">No answer was written for this question. '
            "It is listed because it was asked, not because there is "
            "something here to revise.</p>"
        )

    origin = ""

    if str(getattr(question, "answer_source", "") or "") == "source_excerpt":
        origin = (
            '<p class="muted">Quoted from the slide this question was read '
            "from, rather than written as an answer to it.</p>"
        )

    fenced = re.findall(r"```[a-zA-Z0-9_+.-]*\n(.*?)```", answer, re.S)

    remainder = re.sub(r"```[a-zA-Z0-9_+.-]*\n.*?```", "", answer, flags=re.S)

    prose = "\n".join(
        line.strip() for line in remainder.splitlines() if line.strip()
    )

    parts = []

    if prose:
        parts.append(f'<div class="prose">{esc(prose)}</div>')

    for block in fenced:
        parts.append(f"<pre><code>{esc(block.strip())}</code></pre>")

    if not parts:
        parts.append(
            '<p class="muted">The answer was code only, and it could not '
            "be read.</p>"
        )

    parts.append(origin)

    return "".join(parts)


def _key_concepts(post: KnowledgePost, page: str) -> str:
    """
    The things worth remembering, as a compact list.

    Concepts only. Topics and subtopics are already the breadcrumb and
    the hierarchy, so repeating them here as three more rows of chips
    would be the same information a third time.
    """

    analysis = post.ai_analysis

    concepts = [
        concept
        for concept in (analysis.concepts if analysis else ())
        if isinstance(concept, str) and concept.strip()
    ]

    if not concepts:
        return ""

    shown = concepts[:CONCEPTS_SHOWN]
    hidden = len(concepts) - len(shown)

    chips = "".join(
        f'<li><span class="chip chip-quiet">{esc(concept)}</span></li>'
        for concept in shown
    )

    # Used to read "and N more are listed in the concepts index", linking
    # to concepts.html. That page is no longer generated, so the link was
    # a dead end -- and a concept index is the wrong destination anyway:
    # a reader revising wants the subtopic that teaches the concept, not a
    # list of every label that contains it. The count stays, because it
    # tells the reader the chip row is a selection rather than everything.
    more = (
        f'<p class="muted">and {hidden} more from this post.</p>'
        if hidden > 0
        else ""
    )

    return (
        '<div class="subblock">'
        "<h2>Key concepts</h2>"
        f'<ul class="chip-row">{chips}</ul>'
        f"{more}</div>"
    )


def _questions(post: KnowledgePost, page: str) -> str:
    """The questions, closed until asked for."""

    questions = post.interview_questions[:QUESTIONS_SHOWN]

    if not questions:
        return (
            '<div class="subblock"><h2>Interview questions</h2>'
            '<p class="muted">No interview questions were generated '
            "for this post.</p></div>"
        )

    blocks = []

    for question in questions:
        badges = (
            '<span class="badge-row">'
            f'<span class="badge badge-quiet">'
            f"{esc(str(question.difficulty).capitalize())}</span>"
            f'<span class="badge badge-quiet">'
            f"{esc(str(question.type).capitalize())}</span>"
            "</span>"
        )

        blocks.append(
            '<details class="question">'
            f"<summary><span class=\"question-text\">"
            f"{esc(question.question)}</span>{badges}</summary>"
            f'<div class="question-body">'
            f"{_answer_html(question)}"
            "</div></details>"
        )

    hidden = len(post.interview_questions) - len(questions)

    more = (
        f'<p class="muted">and {hidden} more are in the '
        f'<a class="text-link" href="{esc(href(page, QUESTIONS_PAGE))}">'
        "questions index</a>.</p>"
        if hidden > 0
        else ""
    )

    return (
        '<div class="subblock">'
        f"<h2>Interview questions ({len(post.interview_questions)})</h2>"
        f"{''.join(blocks)}{more}</div>"
    )


def _related(
    post: KnowledgePost,
    curriculum: Curriculum | None,
    siblings: list[tuple[str, str]],
    page: str,
) -> str:
    """
    Where to go next: neighbouring subtopics, then neighbouring pages.

    A short list of links, never a graph. Drawing six thousand concepts
    as a visualisation is a data tool, and reaching for it is the
    archive-shaped instinct this page exists to resist.
    """

    blocks = []

    if curriculum is not None:
        major, subtopic = placement_for(post)

        node = curriculum.subtopic(major.slug, subtopic.slug)

        neighbours = curriculum.related(node, RELATED_SHOWN) if node else []

        if neighbours:
            links = "".join(
                '<li><a class="chip" href="'
                # ``other.major.slug``, not this page's subject: related()
                # borrows from other subjects when a subject has too few
                # subtopics to fill the row, and using this page's slug
                # for those links produced paths that do not exist.
                f'{esc(href(page, subject_page(other.major.slug, other.slug)))}">'
                f"{esc(other.subtopic.name)}</a></li>"
                for other in neighbours
            )

            blocks.append(
                '<p class="muted">More in this subject</p>'
                f'<ul class="chip-row">{links}</ul>'
            )

    if siblings:
        # Resolved against this page. sibling_pages returns root-relative
        # paths, and a knowledge page lives one directory down, so
        # written literally every one of these resolved to posts/posts/.
        links = "".join(
            f'<li><a class="chip" href="{esc(href(page, target))}">'
            f"{esc(label)}</a></li>"
            for label, target in siblings
        )

        blocks.append(
            '<p class="muted">Related topics</p>'
            f'<ul class="chip-row">{links}</ul>'
        )

    if not blocks:
        return ""

    return (
        '<div class="subblock"><h2>Related</h2>'
        f"{''.join(blocks)}</div>"
    )


def _source(post: KnowledgePost, contributing: int) -> str:
    """
    Attribution, one line, and the capture record on request.

    The line is not decoration. A reader deciding whether to trust a
    claim benefits from knowing it came from two posts rather than one,
    and from knowing they are two LinkedIn posts rather than a textbook.
    Everything past it -- the original text, the link, the files, the
    capture metadata, the identifier -- stays behind one disclosure,
    because it is for checking a claim rather than for revising.
    """

    platform = _platform(post)

    count = (
        "1 contributing post"
        if contributing == 1
        else f"{contributing} contributing posts"
    )

    bits = [esc(platform)]

    published = _published_label(post)

    if published:
        bits.append(esc(published))

    bits.append(esc(count))

    blocks = [
        '<div class="source-line">'
        # The separator is joined outside the escaping. Passing a joined
        # string through esc() escaped the entity too, and the line read
        # "LinkedIn &amp;middot; 2026-10-01" -- visible, and wrong.
        f"<span>Source: {' &middot; '.join(bits)}</span>"
        "</div>"
    ]

    detail = []

    original, truncated = truncate(
        post.original_text, SOURCE_EXCERPT_LIMIT
    )

    # Sanitised for display, and only for display. ``post.original_text``
    # is what the author wrote and is left exactly as it was: rewriting it
    # would misrepresent the source, and the model is where traceability
    # lives. The excerpt is the one place this text becomes a published
    # page, so this is where the guarantee belongs.

    # Measured: none of this project's 490 captured post bodies contains a
    # local filesystem path. It is closed because the corpus can change,
    # not because it was needed today -- and the same redactor already
    # protects machine transcriptions, which did contain them.
    original, replacements = redact(original)

    if original:
        notes = []

        if truncated:
            notes.append(
                '<p class="muted">Excerpt truncated for readability. The '
                "original post has the rest.</p>"
            )

        if replacements:
            notes.append(
                f'<p class="muted">{esc(redact_note(replacements))}</p>'
            )

        note = "".join(notes)

        detail.append(
            '<div class="subblock"><h3>What the post said</h3>'
            f'<div class="prose source-text">{esc(original)}</div>'
            f"{note}</div>"
        )
    else:
        detail.append(
            '<div class="subblock"><h3>What the post said</h3>'
            '<p class="muted">No original text was captured for this '
            "post.</p></div>"
        )

    url = safe_link(post.source.url)

    if url:
        detail.append(
            f'<p><a class="button button-ghost" href="{esc(url)}" '
            'rel="noopener noreferrer nofollow" target="_blank">'
            "Open the original post</a></p>"
        )
    else:
        detail.append(
            '<p class="muted">No source URL was recorded for this post.'
            "</p>"
        )

    if post.media:
        # Deliberately dropped. It was a count with no filenames, which
        # was already better than an inventory, but it is still archive
        # metadata: a reader revising cannot act on "3 images" and the
        # transcriptions behind them are not a study aid. What was
        # attached stays in the model and in the OCR index.
        pass

    # The capture record used to be Platform / Captured / How it arrived
    # / Author / Saved on / Saved item / Reference. Everything in that
    # list except the platform is either an identifier or metadata about
    # how the archive was built: a capture date, a capture method, a
    # saved-item id and the post's own key. None of it is something a
    # reader revising for an interview can use, and all of it is
    # retained in the model.
    #
    # What replaces it says the only thing that is both true and useful:
    # what kind of source this is. The same description the question
    # disclosure uses, from the same helper, so the two never drift.
    rows = [("Source", esc(_source_label(post)))]

    if post.source.author:
        rows.append(("Author", esc(post.source.author)))

    detail.append(
        '<div class="subblock"><h3>Source</h3>'
        + definition_list(rows)
        + "</div>"
    )

    blocks.append(
        '<details class="provenance">'
        "<summary>View source</summary>"
        f"{''.join(detail)}"
        "</details>"
    )

    return "".join(blocks)


def sibling_pages(
    post: KnowledgePost,
    posts: list[KnowledgePost],
    slugs: list[str],
    limit: int = RELATED_SHOWN,
) -> list[tuple[str, str]]:
    """
    Other pages that share a concept with this one.

    Chosen by shared concept rather than by subject, so "Related" is
    actually related: two pages under the same subtopic can have nothing
    in common, and two pages under different subtopics often share the
    idea a reader would want next.
    """

    mine = {
        concept
        for concept in (post.ai_analysis.concepts if post.ai_analysis else ())
        if isinstance(concept, str)
    }

    if not mine:
        return []

    scored: list[tuple[int, str, str]] = []

    for other, slug in zip(posts, slugs):
        if other.id == post.id:
            continue

        shared = mine & {
            concept
            for concept in (
                other.ai_analysis.concepts if other.ai_analysis else ()
            )
            if isinstance(concept, str)
        }

        if not shared:
            continue

        scored.append((-len(shared), knowledge_title(other), slug))

    scored.sort()

    return [
        (title, post_page(slug))
        for _count, title, slug in scored[:limit]
    ]


def _breadcrumb(
    post: KnowledgePost,
    page: str,
    curriculum: Curriculum | None,
) -> str:
    """
    Home / Subject / Subtopic, as a path rather than a label dump.

    Three hops, each resolved against this page. Written literally they
    would resolve inside ``posts/`` and 404, which is what the link test
    caught when this page was first written.

    The subject and the subtopic are the same destination, so they read
    as two positions on one link rather than two links to the same file.

    **The last hop is a link only when a page exists for it.** A subject
    page is written only for a subtopic that has questions or concepts
    to show, and a post whose own material landed in an empty one -- or
    in a knowledge base with no questions at all -- would otherwise
    breadcrumb straight to a 404. Plain text says the position without
    promising a destination.
    """

    major, subtopic = placement_for(post)

    target = href(page, subject_page(major.slug, subtopic.slug))

    node = (
        curriculum.subtopic(major.slug, subtopic.slug)
        if curriculum is not None
        else None
    )

    if node is None:
        subject = esc(major.name)
    else:
        subject = f'<a href="{esc(target)}">{esc(major.name)}</a>'

    return (
        '<nav class="breadcrumb" aria-label="Breadcrumb">'
        f'<a href="{esc(href(page, INDEX_PAGE))}">Home</a> &rsaquo; '
        f"{subject} &rsaquo; "
        f"{esc(subtopic.name)}"
        "</nav>"
    )


def render_knowledge(
    post: KnowledgePost,
    page: str,
    *,
    contributing: int = 1,
    curriculum: Curriculum | None = None,
    posts: list[KnowledgePost] | None = None,
    slugs: list[str] | None = None,
    generated_at: str | None = None,
) -> str:
    """One topic, as a study page."""

    title = knowledge_title(post)

    analysis = post.ai_analysis

    summary = (analysis.summary or "").strip() if analysis else ""

    siblings = (
        sibling_pages(post, posts, slugs)
        if posts is not None and slugs is not None
        else []
    )

    lede = summary or (
        "No summary was written for this topic. What follows is the "
        "questions it raises and what was captured alongside them."
    )

    body = "".join(
        [
            _breadcrumb(post, page, curriculum),
            "<h1>" + esc(title) + "</h1>",
            f'<p class="lede">{esc(summary)}</p>'
            if summary
            else f'<p class="muted">{esc(lede)}</p>',
            _key_concepts(post, page),
            _questions(post, page),
            _related(post, curriculum, siblings, page),
            _source(post, contributing),
            '<p class="muted caveat">Questions and answers here are '
            "generated from captured posts. Wording is sometimes "
            "imperfect, and the answers are a study aid rather than a "
            "source of truth.</p>",
        ]
    )

    return render_document(
        page=page,
        title=title,
        description=(summary[:300] or title),
        body=body,
        generated_at=generated_at,
    )


__all__ = [
    "CAPTURE_METHOD_LABELS",
    "CONCEPTS_SHOWN",
    "QUESTIONS_SHOWN",
    "RELATED_SHOWN",
    "SOURCE_EXCERPT_LIMIT",
    "render_knowledge",
    "sibling_pages",
]