"""
Rendering the revision guide.

One page per revision unit, with the same structure every time, so a
reader learns it once:

    Title
    Short revision summary
    Quick Revision      key concepts, definitions, differences, traps
    Interview Questions expandable answers, with an example where one exists
    Related Revision    links to the other units
    Source              one human-readable line

A unit too large to revise comfortably splits into pages of itself, and
every page carries Previous / Page N of M / Next. The split is a question
count, so it is the same on every build.

What this module deliberately does not do: render a page per source
post. Four hundred and ninety posts is the shape of the *corpus*, and
publishing one page per post is what made the site an archive with a
revision front end. The posts stay in the knowledge base, and a
revision page says how many contributed to it without naming any of
them.
"""

from __future__ import annotations

from src.wiki.components import esc
from src.wiki.layout import render_document
from src.wiki.naming import QUESTIONS_PAGE, href
from src.wiki.revision import (
    CAVEAT,
    _count,
    _question_block,
)
from src.wiki.revision_model import Unit, UnitPage, Units


#: Where the revision pages live. One directory, so a reader who lands
#: on one can reach a sibling without going through the home page, and
#: so the depth is known when building relative links.
REVISION_DIR = "revision"

#: The index of units, which is what the home page now links to.
UNITS_PAGE = "revision/index.html"

#: Concepts shown in Quick Revision before the list is truncated.
QUICK_CONCEPTS = 14

#: Sections listed in the contents block.
SECTIONS_SHOWN = 12


def unit_page(slug: str) -> str:
    """The canonical path of a unit's first page."""

    return f"{REVISION_DIR}/{slug}.html"


def unit_pages(units: Units) -> dict[str, str]:
    """Every published page, keyed by its output path."""

    pages: dict[str, str] = {
        "index.html": render_home(units),
        UNITS_PAGE: render_index(units),
        QUESTIONS_PAGE: render_all_questions(units),
    }

    for unit in units.units:
        for page in unit.pages():
            pages[page.path] = render_unit(units, unit, page)

    return pages


def render_all_questions(units: Units) -> str:
    """
    Every published question, in one list.

    Grouped by revision unit rather than by the taxonomy's subject and
    subtopic, so it lists exactly what the guide publishes -- 1,902
    questions -- rather than the curriculum's 2,016. The discarded 114
    are advice and hiring material, and a reader browsing for a question
    to revise does not want them here either.

    Each question links to the unit page it is answered on, because that
    is where the surrounding revision material is.
    """

    blocks = []

    for unit in units.units:
        ordered = sorted(unit.questions, key=lambda q: q.text)

        # The link is on the heading as well as on each question, because
        # a unit can have concepts and no questions -- the taxonomy gave
        # it content the filter refused, and it still has a page saying
        # so. With the link only on the questions, such a page was
        # generated, listed in the tree, and unreachable from here.
        link = (
            f'<a class="text-link" '
            f'href="{esc(href(QUESTIONS_PAGE, unit_page(unit.slug)))}">'
            f"{esc(unit.title)}</a>"
        )

        rows = "".join(
            f'<li><a class="text-link" '
            f'href="{esc(href(QUESTIONS_PAGE, unit_page(unit.slug)))}">'
            f"{esc(question.text)}</a></li>"
            for question in ordered
        )

        blocks.append(
            f"<h2>{link} "
            f'<span class="muted">{_count(unit.question_count, "question")}'
            "</span></h2>"
            f'<ul class="plain-list question-index">{rows}'
            + (
                ""
                if ordered
                else '<li class="muted">No questions; see the unit page.'
                "</li>"
            )
            + "</ul>"
        )

    if not blocks:
        blocks.append(
            '<p class="empty-state">No interview questions yet.</p>'
        )

    body = "".join(
        [
            "<h1>Interview questions</h1>",
            '<p class="lede">Every question in the guide, in one list. '
            "Each links to the unit it is answered on.</p>",
            f'<p class="counts">{_count(units.kept, "question")}</p>',
            "".join(blocks),
            f'<p class="muted caveat">{esc(CAVEAT)}</p>',
        ]
    )

    return render_document(
        page=QUESTIONS_PAGE,
        title="Interview questions",
        description=(
            "Every data engineering interview question in the guide, with "
            "a link to the revision unit that answers it."
        ),
        body=body,
    )


# --------------------------------------------------------------------- the
# page


def render_unit(units: Units, unit: Unit, page: UnitPage) -> str:
    """One page of one revision unit."""

    path = page.path

    blocks = [_question_block(question, path) for question in page.questions]

    title = unit.title

    if page.total > 1:
        title = f"{unit.title} — page {page.number} of {page.total}"

    body = "".join(
        [
            _breadcrumb(unit, path),
            f"<h1>{esc(title)}</h1>",
            f'<p class="lede">{esc(unit.summary)}</p>',
            f'<p class="counts">{_count(unit.question_count, "question")}'
            f" &middot; {_count(len(unit.concepts), 'concept')}"
            f" &middot; {_count(unit.post_count, 'source')}"
            + (
                f" &middot; page {_count(page.number, '')}"
                f"{' of %d' % page.total}"
                if page.total > 1
                else ""
            )
            + "</p>",
            _pagination(page),
            _quick_revision(unit, path),
            _contents(unit),
            '<div class="subblock">'
            + (
                f"<h2>Interview questions "
                f"({_count(len(page.questions), 'question')}"
                + (
                    f" of {_count(unit.question_count, 'question')}"
                    if page.total > 1
                    else ""
                )
                + ")</h2>"
            )
            + (
                "".join(blocks)
                or '<p class="muted">No questions on this page.</p>'
            )
            + "</div>",
            _pagination(page),
            _related(units, unit, path),
            f'<p class="muted caveat">{esc(CAVEAT)}</p>',
        ]
    )

    return render_document(
        page=path,
        title=title,
        description=unit.summary,
        body=body,
    )


def _breadcrumb(unit: Unit, page: str) -> str:
    return (
        '<nav class="breadcrumb" aria-label="Breadcrumb">'
        f'<a href="{esc(href(page, "index.html"))}">Home</a>'
        " &rsaquo; "
        f'<a href="{esc(href(page, UNITS_PAGE))}">{esc(unit.group)}'
        "</a>"
        f" &rsaquo; {esc(unit.title)}"
        "</nav>"
    )


def _pagination(page: UnitPage) -> str:
    """
    Previous / page N of M / next.

    On a unit that fits one page there is nothing to navigate and the row
    is omitted rather than rendered with dead ends -- "Previous" pointing
    at the page you are already on is worse than no navigation at all.
    """

    if page.total == 1:
        return ""

    previous = (
        f'<a class="text-link" rel="prev" '
        f'href="{esc(href(page.path, page.sibling_path(page.number - 1)))}">'
        "Previous</a>"
        if not page.is_first
        else "<span class=\"muted\">Previous</span>"
    )

    following = (
        f'<a class="text-link" rel="next" '
        f'href="{esc(href(page.path, page.sibling_path(page.number + 1)))}">'
        "Next</a>"
        if not page.is_last
        else "<span class=\"muted\">Next</span>"
    )

    return (
        '<nav class="pager" aria-label="Pages in this revision unit">'
        f"{previous}"
        f"<span>Page {page.number} of {page.total}</span>"
        f"{following}"
        "</nav>"
    )


def _quick_revision(unit: Unit, page: str) -> str:
    """
    The scannable part: what to remember before the questions.

    The concepts are the corpus's own labels for this unit, sorted by how
    many of its questions carry them, which puts the ones that recur
    first. They are not definitions -- the corpus did not write any, and
    inventing them would be the one thing this page must not do.
    """

    concepts = unit.top_concepts(QUICK_CONCEPTS)

    if not concepts:
        return ""

    chips = "".join(
        f'<li><span class="chip chip-quiet">{esc(concept)}</span></li>'
        for concept in concepts
    )

    hidden = len(unit.concepts) - len(concepts)

    note = (
        f'<p class="muted">and {hidden} more covered on this unit.</p>'
        if hidden > 0
        else ""
    )

    return (
        '<div class="subblock">'
        "<h2>Quick revision</h2>"
        '<p class="muted">The ideas this unit comes back to. Read the '
        "questions below for how they are asked.</p>"
        f'<ul class="chip-row">{chips}</ul>'
        f"{note}"
        "</div>"
    )


def _contents(unit: Unit) -> str:
    """
    The sections this unit covers, named.

    A unit is several of the taxonomy's subtopics, and a reader who
    knows the subtopic they want should be able to see that it is here.
    The names are the taxonomy's, so nothing is renamed for the page.
    """

    if len(unit.sections) < 2:
        return ""

    shown = unit.sections[:SECTIONS_SHOWN]
    hidden = len(unit.sections) - len(shown)

    items = "".join(f"<li>{esc(name)}</li>" for name in shown)

    if hidden > 0:
        items += f'<li class="muted">and {hidden} more</li>'

    return (
        '<div class="subblock">'
        "<h2>What this covers</h2>"
        f'<ul class="plain-list">{items}</ul>'
        "</div>"
    )


def _related(units: Units, unit: Unit, page: str) -> str:
    """
    The other revision units.

    Grouped, not a flat list of twenty-two links, because the useful
    next step from "SQL: Joins" is another SQL page or a platform page,
    and not a page about Kubernetes.
    """

    others = [other for other in units.units if other.slug != unit.slug]

    if not others:
        return ""

    same = [other for other in others if other.group == unit.group]

    elsewhere = [other for other in others if other.group != unit.group]

    blocks = []

    if same:
        links = "".join(
            f'<li><a class="text-link" '
            f'href="{esc(href(page, unit_page(other.slug)))}">'
            f"{esc(other.title)}</a></li>"
            for other in same
        )
        blocks.append(
            f"<h3>More in {esc(unit.group)}</h3>"
            f'<ul class="plain-list">{links}</ul>'
        )

    if elsewhere:
        links = "".join(
            f'<li><a class="text-link" '
            f'href="{esc(href(page, unit_page(other.slug)))}">'
            f"{esc(other.title)}</a></li>"
            for other in elsewhere[:10]
        )
        blocks.append(
            "<h3>Elsewhere</h3>"
            f'<ul class="plain-list">{links}</ul>'
        )

    return (
        '<div class="subblock">'
        "<h2>Related revision</h2>"
        f"{''.join(blocks)}"
        "</div>"
    )


# ----------------------------------------------------------------- indexes


def render_index(units: Units) -> str:
    """The whole curriculum as a tree: group, then revision unit."""

    blocks = []

    for group in units.groups:
        rows = []

        for unit in units.group_units(group):
            pages = (
                f" &middot; {_count(unit.page_count, 'page')}"
                if unit.page_count > 1
                else ""
            )

            rows.append(
                f'<li><a class="text-link" '
                f'href="{esc(href(UNITS_PAGE, unit_page(unit.slug)))}">'
                f"{esc(unit.title)}</a>"
                f'<span class="muted"> &mdash; '
                f'{esc(unit.summary)}</span>'
                f'<span class="muted counts">{_count(unit.question_count, "question")}'
                f"{pages}</span></li>"
            )

        blocks.append(
            f"<h2>{esc(group)}</h2>"
            f'<ul class="plain-list unit-list">{"".join(rows)}</ul>'
        )

    body = "".join(
        [
            "<h1>Revision units</h1>",
            '<p class="lede">Everything here is one thing a candidate '
            "revises. Each unit is a page; the larger ones run to more "
            "than one.</p>",
            f'<p class="counts">{_count(units.unit_count, "revision unit")}'
            f" &middot; {_count(units.page_count, 'page')}"
            f" &middot; {_count(units.kept, 'question')}</p>",
            "".join(blocks),
            f'<p class="muted caveat">{esc(CAVEAT)}</p>',
        ]
    )

    return render_document(
        page=UNITS_PAGE,
        title="Revision units",
        description=(
            "The revision curriculum: every subject and unit a data "
            "engineering candidate revises, one page each."
        ),
        body=body,
    )


def render_home(units: Units) -> str:
    """
    The home page is the curriculum.

    It used to report the corpus: how many posts, images and concepts
    were captured. That is a description of the archive, and a reader
    revising for an interview does not need it -- knowing there are
    3,047 slide images tells them nothing about joins. It is now the
    same tree as the units index, which is the thing they came for.
    """

    blocks = []

    for group in units.groups:
        rows = "".join(
            f'<li><a class="text-link" '
            f'href="{esc(href("index.html", unit_page(unit.slug)))}">'
            f"{esc(unit.title)}</a></li>"
            for unit in units.group_units(group)
        )

        blocks.append(
            f"<h2>{esc(group)}</h2>"
            f'<ul class="plain-list unit-list">{rows}</ul>'
        )

    body = "".join(
        [
            "<h1>Data Engineering Interview Revision</h1>",
            '<p class="lede">Start at the subject you are weakest in. '
            "Each unit is a page you can revise end to end.</p>",
            f'<p class="counts">{_count(units.unit_count, "revision unit")}'
            f" &middot; {_count(units.page_count, 'page')}"
            f" &middot; {_count(units.kept, 'question')}</p>",
            "".join(blocks),
            '<div class="subblock">'
            "<h2>Not revision material</h2>"
            f'<p class="muted">{_count(units.discarded, "question")} '
            "were left out: interview process and practice advice, "
            "behavioural coaching, job and career material, and "
            "announcements with nothing to revise. They are kept in the "
            "project's data, and are not part of this guide.</p>"
            "</div>",
            '<div class="subblock">'
            "<h2>Find something specific</h2>"
            '<ul class="plain-list">'
            f'<li><a class="text-link" href="search.html">Search</a> '
            "<span class=\"muted\">every question and answer</span></li>"
            f'<li><a class="text-link" href="{esc(href("index.html", QUESTIONS_PAGE))}">'
            "All questions</a> "
            "<span class=\"muted\">browse by subject and type</span></li>"
            "</ul>"
            "</div>",
            f'<p class="muted caveat">{esc(CAVEAT)}</p>',
        ]
    )

    return render_document(
        page="index.html",
        title="Data Engineering Interview Revision",
        description=(
            "A revision guide for data engineering interviews: SQL, "
            "Python, Spark, Databricks, Azure, AWS, data engineering "
            "practice, and the questions each one is asked."
        ),
        body=body,
    )