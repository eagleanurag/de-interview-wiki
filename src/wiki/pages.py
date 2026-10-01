"""
Page builders.

Every page renders from the `SiteModel`, so counts, ordering and links
stay consistent across the site. Nothing here reads pipeline metadata,
and every interpolated value is escaped by `src.wiki.components`.

All cross-page links go through `href(page, target)`, where `page` is
the site-relative path of the page being rendered. That is what keeps
links correct for pages nested under `posts/` and `topics/`.
"""

from __future__ import annotations

import json

from src.models import KnowledgePost
from src.wiki.analysis import (
    ConceptEntry,
    DIFFICULTIES,
    QuestionEntry,
    SiteModel,
    SUBTOPIC_KIND,
    TechnologyEntry,
    TopicEntry,
)
from src.wiki.components import (
    MEDIA_EXCERPT_LIMIT,
    POST_SOURCE_EXCERPT_LIMIT,
    SUMMARY_EXCERPT_LIMIT,
    TYPE_LABELS,
    badge,
    collapse_whitespace,
    definition_list,
    difficulty_badge,
    empty_state,
    esc,
    page_link,
    safe_link,
    section,
    stat_grid,
    stat_tile,
    topic_badges,
    truncate,
    type_badge,
)
from src.wiki.layout import render_document
from src.wiki.naming import (
    CONCEPTS_PAGE,
    INDEX_PAGE,
    NOT_FOUND_PAGE,
    QUESTIONS_PAGE,
    QUESTIONS_SCRIPT,
    SEARCH_INDEX_FILE,
    SEARCH_PAGE,
    SEARCH_SCRIPT,
    TECHNOLOGIES_PAGE,
    TOPICS_PAGE,
    href,
    post_page,
)


RECENT_POST_LIMIT = 12

SOURCE_PLATFORM_LABELS = {
    "linkedin": "LinkedIn",
}

GENERATED_CARD_SUBTITLE = (
    "Model-assisted content derived from the captured post. Treat "
    "it as a study aid, not as an authoritative source."
)

SOURCE_CARD_SUBTITLE = (
    "Captured third-party content, reproduced only as short excerpts "
    "for study and attribution."
)

#: How each capture method is described to a reader. The point of the
#: distinction is that material this project was given is not material
#: it went and collected, and a reader is entitled to know which they
#: are looking at.
CAPTURE_METHOD_LABELS = {
    "user_export": "You exported this from your saved items",
    "user_provided": "You supplied the content for this",
    "user_saved_page": "You saved the page and supplied it here",
    "user_bundle": "You assembled this from your own notes",
}

SAVED_ITEM_CARD_SUBTITLE = (
    "Traced back to the entry in your saved list, and to the capture "
    "that came with it."
)


def render_home(model: SiteModel) -> str:
    """Dashboard: totals plus the most recent knowledge entries."""

    page = INDEX_PAGE

    tiles = [
        stat_tile(
            model.post_count,
            "Posts",
            detail="Enriched knowledge entries",
        ),
        stat_tile(
            model.topic_count,
            "Topics",
            detail="Topics and subtopics",
        ),
        stat_tile(
            model.concept_count,
            "Concepts",
            detail="Distinct concepts named",
        ),
        stat_tile(
            model.question_count,
            "Interview questions",
            detail="Across every post",
        ),
    ]

    parts = [
        '<section class="hero">',
        "<h1>Data Engineering interview knowledge</h1>",
        '<p class="hero-lead">',
        esc(
            "Every entry here was captured from a public source, then "
            "enriched into a summary, concepts and interview "
            "questions you can revise from."
        ),
        "</p>",
        '<nav class="hero-actions" aria-label="Quick links">',
        _button_link(page, "Search the knowledge base", SEARCH_PAGE),
        _button_link(page, "Browse topics", TOPICS_PAGE),
        _button_link(page, "Practise questions", QUESTIONS_PAGE),
        "</nav>",
        "</section>",
        stat_grid(tiles, aria_label="Knowledge base totals"),
    ]

    if model.is_empty:
        parts.append(
            empty_state(
                "No posts yet",
                "The canonical knowledge base contains no posts. Run "
                "the worker pipeline, then regenerate this site.",
            )
        )
    else:
        parts.append(
            _post_card_grid(
                model,
                list(model.recent_pairs(RECENT_POST_LIMIT)),
                page,
                title="Recent knowledge entries",
                subtitle=(
                    "Most recently captured first. Every post is "
                    "reachable from the topics pages."
                ),
            )
        )

    parts.append(_explainer())

    return render_document(
        page=page,
        title="Home",
        description=model.summary(),
        body="".join(parts),
        generated_at=model.generated_at,
    )


def render_search(model: SiteModel) -> str:
    """Client-side search driven entirely by the generated index."""

    page = SEARCH_PAGE

    topic_options = "".join(
        f'<option value="{esc(entry.label)}">'
        f"{esc(entry.label)} ({entry.post_count})</option>"
        for entry in model.topics_of_kind("topic")
    )

    body = "".join(
        [
            "<h1>Search</h1>",
            '<p class="page-lead">',
            esc(
                "Searches summaries, original source text, topics, "
                "subtopics, concepts, interview questions and "
                "attribution across "
            ),
            f"<strong>{esc(model.post_count)}</strong> posts.",
            "</p>",
            '<form class="search-panel" role="search" '
            'onsubmit="return false">',
            '<label class="field">',
            '<span class="field-label">Search terms</span>',
            '<input id="search-input" class="search-input" '
            'type="search" autocomplete="off" '
            'placeholder="delta, z-ordering, broadcast join, '
            'small files...">',
            "</label>",
            '<label class="field">',
            '<span class="field-label">Topic</span>',
            '<select id="search-topic" class="search-select">',
            '<option value="">All topics</option>',
            topic_options,
            "</select>",
            "</label>",
            "</form>",
            '<p class="search-status" id="search-status" '
            'role="status" aria-live="polite">Loading index...</p>',
            '<div class="card-grid" id="search-results"></div>',
            "<noscript>",
            '<div class="empty-state">',
            "<h2>Search needs JavaScript</h2>",
            "<p>",
            esc(
                "This page filters a pre-generated index in the "
                "browser. With JavaScript disabled, browse the "
            ),
            page_link("topics page", TOPICS_PAGE, page),
            esc(" or the "),
            page_link("questions browser", QUESTIONS_PAGE, page),
            esc(" instead."),
            "</p>",
            "</div>",
            "</noscript>",
        ]
    )

    return render_document(
        page=page,
        title="Search",
        description=(
            f"Client-side search across {model.post_count} posts "
            f"and {model.concept_count} concepts."
        ),
        body=body,
        generated_at=model.generated_at,
        config={"indexUrl": SEARCH_INDEX_FILE},
        scripts=(SEARCH_SCRIPT,),
    )


def render_topics_index(model: SiteModel) -> str:
    """Alphabetical topic and subtopic directory."""

    page = TOPICS_PAGE

    if model.is_empty:
        return render_document(
            page=page,
            title="Topics",
            description="Topic directory.",
            body="".join(
                [
                    "<h1>Topics</h1>",
                    empty_state(
                        "No topics yet",
                        "Topics appear once the knowledge base "
                        "contains enriched posts.",
                    ),
                ]
            ),
            generated_at=model.generated_at,
        )

    parts = [
        "<h1>Topics</h1>",
        '<p class="page-lead">',
        esc(
            "Topics come from the enriched posts themselves, not "
            "from a fixed list. "
        ),
        f"<strong>{esc(model.topic_count)}</strong> ",
        esc(
            "topics and subtopics are available, each linking to its "
            "related posts."
        ),
        "</p>",
    ]

    parts.append(
        _topic_directory(
            model.topics_of_kind("topic"),
            page,
            "Topics",
            "Primary subjects named by the enrichment step.",
        )
    )

    parts.append(
        _topic_directory(
            model.topics_of_kind(SUBTOPIC_KIND),
            page,
            "Subtopics",
            "Finer-grained subjects named by the enrichment step.",
        )
    )

    return render_document(
        page=page,
        title="Topics",
        description=(
            f"Directory of {model.topic_count} data engineering "
            "topics and subtopics."
        ),
        body="".join(parts),
        generated_at=model.generated_at,
    )


def render_concepts_index(model: SiteModel) -> str:
    """Every concept the knowledge base has extracted."""

    page = CONCEPTS_PAGE

    entries = model.concept_entries

    if not entries:
        return render_document(
            page=page,
            title="Concepts",
            description="Concepts extracted from collected posts.",
            body=empty_state(
                "No concepts yet",
                "Concepts appear once posts have been enriched.",
            ),
        )

    shared = sum(
        1 for entry in entries if entry.post_count > 1
    )

    cards = []

    for entry in entries:
        cards.append(
            "".join(
                [
                    '<a class="card topic-card" ',
                    f'href="{esc(href(page, entry.page))}">',
                    f"<h3>{esc(entry.label)}</h3>",
                    '<div class="badge-row">',
                    badge(
                        str(entry.post_count),
                        "badge badge-quiet",
                        title="Posts mentioning this concept",
                    ),
                    badge(
                        str(len(entry.technologies)),
                        "badge badge-quiet",
                        title="Technologies it relates to",
                    ),
                    "</div>",
                    "</a>",
                ]
            )
        )

    body = "".join(
        [
            _explainer(),
            '<div class="stat-row">',
            stat_tile(len(entries), "Concepts"),
            stat_tile(shared, "Mentioned by more than one post"),
            stat_tile(model.technology_count, "Technologies"),
            "</div>",
            section(
                "All concepts",
                f'<div class="topic-grid">{"".join(cards)}</div>',
                subtitle=(
                    "Grouped from post analysis. Each concept keeps the "
                    "posts it came from."
                ),
            ),
        ]
    )

    return render_document(
        page=page,
        title="Concepts",
        description="Technical concepts extracted from collected posts.",
        body=body,
        generated_at=model.generated_at,
    )


def render_concept_detail(
    model: SiteModel,
    concept: ConceptEntry,
) -> str:
    """One concept: the posts that mention it, and what it relates to."""

    page = concept.page

    posts_by_slug = dict(zip(model.post_slugs, model.posts))

    pairs = [
        (posts_by_slug[slug], slug)
        for slug in concept.post_slugs
        if slug in posts_by_slug
    ]

    parts = [
        _breadcrumb(
            page,
            [("Concepts", CONCEPTS_PAGE), (concept.label, None)],
        ),
        '<section class="hero hero-compact">',
        '<div class="badge-row">',
        badge("Concept", "badge badge-kind"),
        "</div>",
        f"<h1>{esc(concept.label)}</h1>",
        "</section>",
        stat_grid(
            [
                stat_tile(concept.post_count, "Posts mentioning this"),
                stat_tile(len(concept.topics), "Related topics"),
                stat_tile(
                    len(concept.technologies), "Related technologies"
                ),
            ],
            aria_label=f"{concept.label} totals",
        ),
    ]

    if concept.technologies:
        parts.append(
            _related_links(
                page,
                concept.technologies,
                model,
                "technologies",
                "Related technologies",
            )
        )

    if concept.topics:
        parts.append(
            _related_links(
                page,
                concept.topics,
                model,
                "topics",
                "Related topics",
            )
        )

    if pairs:
        parts.append(
            _post_card_grid(
                model,
                pairs,
                page,
                title="Posts that mention this concept",
                subtitle=(
                    "The source material this concept was extracted "
                    "from, in canonical order."
                ),
            )
        )
    else:
        parts.append(
            empty_state(
                "No source posts",
                "This concept has no post to trace it back to.",
            )
        )

    return render_document(
        page=page,
        title=concept.label,
        description=(
            f"Posts mentioning {concept.label}, with the topics and "
            f"technologies around it."
        ),
        body="".join(parts),
        generated_at=model.generated_at,
    )


def render_technologies_index(model: SiteModel) -> str:
    """Every technology the knowledge base recognises in use."""

    page = TECHNOLOGIES_PAGE

    entries = model.technology_entries

    if not entries:
        return render_document(
            page=page,
            title="Technologies",
            description="Technologies the collected posts use.",
            body=empty_state(
                "No technologies yet",
                "Technologies appear once posts mention them.",
            ),
        )

    cards = []

    for entry in entries:
        cards.append(
            "".join(
                [
                    '<a class="card topic-card" ',
                    f'href="{esc(href(page, entry.page))}">',
                    f"<h3>{esc(entry.label)}</h3>",
                    '<div class="badge-row">',
                    badge(
                        str(entry.post_count),
                        "badge badge-quiet",
                        title="Posts using this technology",
                    ),
                    badge(
                        str(entry.question_count),
                        "badge badge-quiet",
                        title="Related questions",
                    ),
                    "</div>",
                    "</a>",
                ]
            )
        )

    body = "".join(
        [
            _explainer(),
            section(
                "All technologies",
                f'<div class="topic-grid">{"".join(cards)}</div>',
                subtitle=(
                    "Recognised from the text of collected posts, so "
                    "nothing appears without a post that mentions it."
                ),
            ),
        ]
    )

    return render_document(
        page=page,
        title="Technologies",
        description="Technologies the collected posts use.",
        body=body,
        generated_at=model.generated_at,
    )


def render_technology_detail(
    model: SiteModel,
    technology: TechnologyEntry,
) -> str:
    """One technology: the posts that use it and their questions."""

    page = technology.page

    posts_by_slug = dict(zip(model.post_slugs, model.posts))

    pairs = [
        (posts_by_slug[slug], slug)
        for slug in technology.post_slugs
        if slug in posts_by_slug
    ]

    parts = [
        _breadcrumb(
            page,
            [("Technologies", TECHNOLOGIES_PAGE), (technology.label, None)],
        ),
        '<section class="hero hero-compact">',
        '<div class="badge-row">',
        badge("Technology", "badge badge-kind"),
        "</div>",
        f"<h1>{esc(technology.label)}</h1>",
        "</section>",
        stat_grid(
            [
                stat_tile(technology.post_count, "Posts using this"),
                stat_tile(technology.question_count, "Related questions"),
                stat_tile(len(technology.topics), "Related topics"),
            ],
            aria_label=f"{technology.label} totals",
        ),
    ]

    if technology.topics:
        parts.append(
            _related_links(
                page,
                technology.topics,
                model,
                "topics",
                "Related topics",
            )
        )

    if pairs:
        parts.append(
            _post_card_grid(
                model,
                pairs,
                page,
                title="Posts using this technology",
                subtitle=(
                    "The collected posts that mention it, in canonical "
                    "order."
                ),
            )
        )
    else:
        parts.append(
            empty_state(
                "No source posts",
                "This technology has no post to trace it back to.",
            )
        )

    return render_document(
        page=page,
        title=technology.label,
        description=(
            f"Posts using {technology.label}, with the topics and "
            f"questions around it."
        ),
        body="".join(parts),
        generated_at=model.generated_at,
    )


def _related_links(
    page: str,
    labels: tuple[str, ...],
    model: SiteModel,
    kind: str,
    heading: str,
) -> str:
    """A row of links into another section, skipping unknown labels."""

    targets = _pages_for_labels(labels, model, kind)

    if not targets:
        return ""

    chips = "".join(
        '<li><a class="chip" href="{href}">{label}</a></li>'.format(
            href=esc(href(page, target)),
            label=esc(label),
        )
        for label, target in targets
    )

    return section(
        heading,
        f'<ul class="chip-row">{chips}</ul>',
        subtitle="Cross-links into the rest of the knowledge base.",
    )


def _pages_for_labels(
    labels: tuple[str, ...],
    model: SiteModel,
    kind: str,
) -> list[tuple[str, str]]:
    """
    Resolve labels to pages in another section.

    A label with no page is skipped rather than rendered as a dead
    link, which is what keeps every link on the site working.
    """

    if kind == "topics":
        lookup = {entry.label: entry.page for entry in model.topics}
    elif kind == "concepts":
        lookup = {
            entry.label: entry.page for entry in model.concept_entries
        }
    elif kind == "technologies":
        lookup = {
            entry.label: entry.page
            for entry in model.technology_entries
        }
    else:
        return []

    return [
        (label, lookup[label]) for label in labels if label in lookup
    ]


def render_topic_detail(
    model: SiteModel,
    topic: TopicEntry,
) -> str:
    """One topic: its posts, concepts and question volume."""

    page = topic.page

    posts_by_slug = dict(zip(model.post_slugs, model.posts))
    pairs = [
        (posts_by_slug[slug], slug)
        for slug in topic.post_slugs
        if slug in posts_by_slug
    ]

    kind_label = (
        "Subtopic"
        if topic.kind == SUBTOPIC_KIND
        else "Topic"
    )

    tiles = [
        stat_tile(topic.post_count, "Related posts"),
        stat_tile(topic.question_count, "Related questions"),
        stat_tile(
            len(topic.concepts), "Concepts in these posts"
        ),
    ]

    parts = [
        _breadcrumb(
            page,
            [("Topics", TOPICS_PAGE), (topic.label, None)],
        ),
        '<section class="hero hero-compact">',
        '<div class="badge-row">',
        badge(kind_label, "badge badge-kind"),
        "</div>",
        f"<h1>{esc(topic.label)}</h1>",
        "</section>",
        stat_grid(tiles, aria_label=f"{topic.label} totals"),
    ]

    if pairs:
        parts.append(
            _post_card_grid(
                model,
                pairs,
                page,
                title="Posts on this topic",
                subtitle=(
                    "Listed in canonical ID order so the page is "
                    "identical between builds."
                ),
            )
        )
    else:
        parts.append(
            empty_state(
                "No posts",
                "This topic has no posts in the current knowledge "
                "base.",
            )
        )

    if topic.concepts:
        chips = "".join(
            badge(concept, "badge badge-concept")
            for concept in topic.concepts
        )
        parts.append(
            section(
                "Concepts",
                f'<div class="badge-row">{chips}</div>',
                subtitle=(
                    "Distinct concepts named by the posts on this "
                    "topic."
                ),
            )
        )

    parts.append(
        '<p class="page-foot">'
        + esc("Looking for practice material? ")
        + page_link("Open the question browser", QUESTIONS_PAGE, page)
        + esc(" or ")
        + page_link("search the knowledge base", SEARCH_PAGE, page)
        + ".</p>"
    )

    return render_document(
        page=page,
        title=topic.label,
        description=(
            f"{kind_label} {topic.label}: {topic.post_count} posts, "
            f"{topic.question_count} interview questions and "
            f"{len(topic.concepts)} concepts."
        ),
        body="".join(parts),
        generated_at=model.generated_at,
    )


def render_questions(model: SiteModel) -> str:
    """
    Question browser with client-side filtering.

    Every question is rendered server-side with data attributes, so the
    page is complete and readable without JavaScript. The script only
    hides or reveals existing markup; it never injects content.
    """

    page = QUESTIONS_PAGE

    topic_values = sorted(
        {
            topic
            for question in model.questions
            for topic in question.topics
        },
        key=str.casefold,
    )
    type_counts = model.question_type_counts()
    difficulty_counts = model.difficulty_counts()

    topic_options = "".join(
        f'<option value="{esc(value)}">{esc(value)}</option>'
        for value in topic_values
    )

    difficulty_options = "".join(
        f'<option value="{esc(value)}">'
        f"{esc(value.capitalize())} "
        f"({difficulty_counts[value]})</option>"
        for value in DIFFICULTIES
        if difficulty_counts.get(value)
    )

    type_options = "".join(
        f'<option value="{esc(value)}">'
        f"{esc(TYPE_LABELS.get(value, value.capitalize()))} "
        f"({count})</option>"
        for value, count in type_counts.items()
    )

    cards = "".join(
        _question_card(question) for question in model.questions
    )

    empty_note = ""
    if not model.questions:
        empty_note = empty_state(
            "No interview questions yet",
            "Questions appear once the enrichment step has produced "
            "them for at least one post.",
        )

    body = "".join(
        [
            "<h1>Interview questions</h1>",
            '<p class="page-lead">',
            esc(
                "Filter by topic, difficulty and question type. The "
                "options come from the knowledge base itself, so the "
                "filter never offers a value the data does not "
                "contain."
            ),
            "</p>",
            '<form class="filter-panel" role="search" '
            'onsubmit="return false">',
            '<label class="field">',
            '<span class="field-label">Search</span>',
            '<input id="question-query" class="search-input" '
            'type="search" autocomplete="off" '
            'placeholder="filter questions...">',
            "</label>",
            _select_field(
                "question-topic", "Topic", topic_options
            ),
            _select_field(
                "question-difficulty",
                "Difficulty",
                difficulty_options,
            ),
            _select_field(
                "question-type", "Type", type_options
            ),
            '<button type="button" id="question-reset" '
            'class="button button-ghost">Reset</button>',
            "</form>",
            '<p class="search-status" id="question-status" '
            'role="status" aria-live="polite"></p>',
            f'<div class="question-list" id="question-list">{cards}'
            "</div>",
            empty_note,
        ]
    )

    return render_document(
        page=page,
        title="Interview questions",
        description=(
            f"{model.question_count} interview questions across "
            f"{model.post_count} posts, filterable by topic, "
            "difficulty and type."
        ),
        body=body,
        generated_at=model.generated_at,
        scripts=(QUESTIONS_SCRIPT,),
    )


def render_post_detail(
    model: SiteModel,
    post: KnowledgePost,
    slug: str,
) -> str:
    """One post: attribution, generated knowledge, source material."""

    page = post_page(slug)
    source = post.source

    platform = SOURCE_PLATFORM_LABELS.get(
        source.platform, source.platform
    )
    source_url = safe_link(source.url)

    pairs: list[tuple[str, str]] = [
        ("Platform", esc(platform)),
        (
            "Captured",
            esc(source.captured_at.strftime("%Y-%m-%d")),
        ),
    ]

    if source.published_at:
        # Shown as the source rendered it, which may be a relative form
        # such as "2 days ago". Converting that to a date would mean
        # guessing, so the original wording is kept.
        pairs.append(("Published", esc(source.published_at)))

    if source.author:
        pairs.append(("Author", esc(source.author)))

    pairs.append(("Post ID", f"<code>{esc(post.id)}</code>"))

    if source_url:
        source_link = (
            f'<a class="button button-ghost" '
            f'href="{esc(source_url)}" '
            f'rel="noopener noreferrer nofollow" target="_blank">'
            f"Open original source</a>"
        )
    else:
        source_link = (
            '<p class="muted">No source URL was recorded for this '
            "post.</p>"
        )

    description = collapse_whitespace(post.ai_analysis.summary)

    if not description:
        description = f"Knowledge entry {post.id}."

    body = "".join(
        [
            _breadcrumb(
                page,
                [
                    ("Home", INDEX_PAGE),
                    ("Topics", TOPICS_PAGE),
                    (post.id, None),
                ],
            ),
            '<section class="hero hero-compact">',
            f"<h1>{esc(post.id)}</h1>",
            '<div class="badge-row">',
            badge(platform, "badge badge-source"),
            badge(
                post.classification.domain or "Data Engineering",
                "badge badge-domain",
            ),
            "</div>",
            "</section>",
            section(
                "Source information",
                definition_list(pairs) + source_link,
                subtitle=(
                    "Attribution recorded at capture time. Only short "
                    "excerpts are reproduced here; read the original "
                    "for the full content."
                ),
            ),
            _saved_item_card(post),
            _generated_knowledge_card(model, post, page),
            _questions_card(post),
            _source_material_card(post),
            '<p class="page-foot">',
            page_link("Back to home", INDEX_PAGE, page),
            esc(" &middot; "),
            page_link("Browse topics", TOPICS_PAGE, page),
            esc(" &middot; "),
            page_link("Search the knowledge base", SEARCH_PAGE, page),
            "</p>",
        ]
    )

    return render_document(
        page=page,
        title=post.id,
        description=description[:300],
        body=body,
        generated_at=model.generated_at,
    )


def render_not_found(model: SiteModel) -> str:
    """`404.html`, used by GitHub Pages for unknown paths."""

    page = NOT_FOUND_PAGE

    body = "".join(
        [
            '<section class="hero">',
            "<h1>Page not found</h1>",
            '<p class="hero-lead">',
            esc(
                "That page is not part of this generated wiki. Try "
                "one of these instead."
            ),
            "</p>",
            '<nav class="hero-actions" aria-label="Site sections">',
            _button_link(page, "Home", INDEX_PAGE),
            _button_link(page, "Search", SEARCH_PAGE),
            _button_link(page, "Topics", TOPICS_PAGE),
            _button_link(page, "Questions", QUESTIONS_PAGE),
            "</nav>",
            "</section>",
        ]
    )

    return render_document(
        page=page,
        title="Page not found",
        description="Page not found.",
        body=body,
        generated_at=model.generated_at,
    )


# ---------------------------------------------------------------------
# Internal building blocks
# ---------------------------------------------------------------------


def _explainer() -> str:
    """Short orientation block explaining the site's provenance."""

    rows = [
        (
            "Posts",
            "One captured source post, enriched into a summary, "
            "concepts and interview questions.",
        ),
        (
            "Topics",
            "Subjects named by the enrichment step. Each topic page "
            "lists the posts that mention it.",
        ),
        (
            "Questions",
            "Generated practice questions, filterable by topic, "
            "difficulty and type.",
        ),
        (
            "Attribution",
            "Every post records its source platform, author and "
            "capture date. Source content is quoted only in short "
            "excerpts.",
        ),
    ]

    return section(
        "How this wiki is organised",
        definition_list(rows),
        css_class="card card-quiet",
    )


def _generated_knowledge_card(
    model: SiteModel,
    post: KnowledgePost,
    page: str,
) -> str:
    analysis = post.ai_analysis
    classification = post.classification
    slugs = model.topic_slugs

    blocks = []

    if analysis.summary:
        blocks.append(
            '<div class="prose">'
            f"<p>{esc(collapse_whitespace(analysis.summary))}</p>"
            "</div>"
        )

    knowledge_rows: list[tuple[str, str]] = []

    if analysis.topics:
        knowledge_rows.append(
            (
                "Topics",
                topic_badges(analysis.topics, page, slugs),
            )
        )

    if analysis.subtopics:
        knowledge_rows.append(
            (
                "Subtopics",
                topic_badges(analysis.subtopics, page, slugs),
            )
        )

    if knowledge_rows:
        blocks.append(definition_list(knowledge_rows))

    if analysis.concepts:
        chips = "".join(
            badge(concept, "badge badge-concept")
            for concept in analysis.concepts
        )
        blocks.append(
            '<div class="subblock">'
            "<h3>Concepts</h3>"
            f'<div class="badge-row">{chips}</div>'
            "</div>"
        )

    if analysis.image_descriptions:
        items = "".join(
            f"<li>{esc(description)}</li>"
            for description in analysis.image_descriptions
        )
        blocks.append(
            '<div class="subblock">'
            "<h3>Image descriptions</h3>"
            f'<ul class="plain-list">{items}</ul>'
            "</div>"
        )

    classification_rows: list[tuple[str, str]] = [
        (
            "Interview relevant",
            "Yes" if classification.interview_relevant else "No",
        )
    ]

    if classification.domain:
        classification_rows.insert(
            0, ("Domain", esc(classification.domain))
        )

    if classification.primary_topic:
        classification_rows.append(
            ("Primary topic", esc(classification.primary_topic))
        )

    if classification.secondary_topics:
        classification_rows.append(
            (
                "Secondary topics",
                " ".join(
                    badge(topic, "badge badge-quiet")
                    for topic in classification.secondary_topics
                ),
            )
        )

    blocks.append(
        '<div class="subblock">'
        "<h3>Classification</h3>"
        f"{definition_list(classification_rows)}"
        "</div>"
    )

    return section(
        "Generated knowledge",
        "".join(blocks),
        css_class="card card-generated",
        subtitle=GENERATED_CARD_SUBTITLE,
    )


def _questions_card(post: KnowledgePost) -> str:
    if not post.interview_questions:
        return section(
            "Interview questions",
            '<p class="muted">No interview questions were generated '
            "for this post.</p>",
        )

    cards = "".join(
        '<article class="question-card question-card-compact">'
        '<div class="badge-row">'
        f"{difficulty_badge(question.difficulty)}"
        f"{type_badge(question.type)}"
        "</div>"
        f"<h3>{esc(question.question)}</h3>"
        f"{_answer_block(question.answer)}"
        "</article>"
        for question in post.interview_questions
    )

    return section(
        f"Interview questions ({len(post.interview_questions)})",
        cards,
    )


def _saved_item_card(post: KnowledgePost) -> str:
    """
    The saved item a post came from.

    Nothing at all for a post that did not come from one: a section
    headed "saved item" on every page would imply every post was saved,
    and would bury the one section that actually tells a reader
    something.
    """
    saved = post.saved_item

    if saved is None or not saved.saved_item_id:
        return ""

    method = post.source.capture_method or ""

    # The reader gets the plain description and the recorded value is
    # kept alongside it, so the page says what happened in words and
    # remains traceable back to the manifest in the exact term it used.
    arrival = esc(
        CAPTURE_METHOD_LABELS.get(
            method, method.replace("_", " ") or "not recorded"
        )
    )

    if method:
        arrival += (
            f' <code class="muted" data-capture-method="{esc(method)}">'
            f"{esc(method)}</code>"
        )

    pairs: list[tuple[str, str]] = [
        ("How it arrived", arrival),
        ("Saved item", f"<code>{esc(saved.saved_item_id)}</code>"),
    ]

    if saved.saved_date:
        pairs.append(("Saved on", esc(saved.saved_date)))

    if saved.url_kind:
        pairs.append(("Link type", esc(saved.url_kind.replace("_", " "))))

    if saved.capture_match and saved.capture_match != "no bundle":
        pairs.append(("Content matched by", esc(saved.capture_match)))

    if saved.saved_notes:
        pairs.append(("Your note", esc(saved.saved_notes)))

    extras = ""

    if saved.capture_notes:
        items = "".join(
            f"<li>{esc(note)}</li>"
            for note in saved.capture_notes
            if note
        )

        if items:
            extras = (
                '<p class="muted">About this capture</p>'
                f"<ul>{items}</ul>"
            )

    if not extras:
        # The state is stated even when there is nothing to add, so a
        # reader is never left guessing whether something was missed.
        extras = (
            '<p class="muted">The capture was read in full; nothing '
            "was left unreadable.</p>"
        )

    return section(
        "Saved item",
        definition_list(pairs) + extras,
        css_class="card card-source",
        subtitle=SAVED_ITEM_CARD_SUBTITLE,
    )


def _source_material_card(post: KnowledgePost) -> str:
    original, truncated = truncate(
        post.original_text, POST_SOURCE_EXCERPT_LIMIT
    )

    blocks = []

    if original:
        note = ""
        if truncated:
            note = (
                '<p class="muted">Excerpt truncated for readability. '
                "Use the original source link for the full text.</p>"
            )

        blocks.append(
            f'<div class="prose source-text">{esc(original)}</div>'
            f"{note}"
        )
    else:
        blocks.append(
            '<p class="muted">No original text was captured for this '
            "post.</p>"
        )

    if post.media:
        blocks.append(_media_list(post))

    return section(
        "Original source material",
        "".join(blocks),
        css_class="card card-source",
        subtitle=SOURCE_CARD_SUBTITLE,
    )


def _media_list(post: KnowledgePost) -> str:
    items = []

    for media in post.media:
        details = [
            f'<span class="badge badge-quiet">'
            f"{esc(media.type)}</span>"
        ]

        if media.description:
            details.append(
                '<span class="media-desc">'
                f"{esc(collapse_whitespace(media.description))}"
                "</span>"
            )

        extracted, _ = truncate(
            media.extracted_text, MEDIA_EXCERPT_LIMIT
        )

        extracted_html = ""
        if extracted:
            extracted_html = (
                '<details class="media-extract">'
                "<summary>Extracted text from this file</summary>"
                f'<div class="prose">{esc(extracted)}</div>'
                "</details>"
            )

        path_html = ""
        if media.path:
            path_html = (
                f'<code class="media-path">{esc(media.path)}</code>'
            )

        items.append(
            '<li class="media-item">'
            f'<div class="badge-row">{"".join(details)}</div>'
            f"{path_html}"
            f"{extracted_html}"
            "</li>"
        )

    return (
        '<div class="subblock">'
        f"<h3>Media ({len(post.media)})</h3>"
        '<p class="muted">Files are referenced by their '
        "capture-time path and are not published with this site.</p>"
        f'<ul class="media-list">{"".join(items)}</ul>'
        "</div>"
    )


def _answer_block(answer: object) -> str:
    text = str(answer or "").strip()

    if not text:
        return '<p class="muted">No answer guidance recorded.</p>'

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if len(lines) <= 1:
        return f'<div class="prose answer">{esc(text)}</div>'

    items = "".join(f"<li>{esc(line)}</li>" for line in lines)

    return (
        '<details class="answer" open>'
        "<summary>What a strong answer covers</summary>"
        f'<ul class="plain-list">{items}</ul>'
        "</details>"
    )


def _question_card(question: QuestionEntry) -> str:
    topics_json = esc(
        json.dumps(list(question.topics), ensure_ascii=False)
    )

    return (
        f'<article class="question-card" id="{esc(question.key)}"'
        f' data-difficulty="{esc(question.difficulty)}"'
        f' data-type="{esc(question.question_type)}"'
        f' data-topics="{topics_json}"'
        f' data-post="{esc(question.post_id)}"'
        f' data-search="{esc(collapse_whitespace(question.question))}">'
        '<div class="badge-row">'
        f"{difficulty_badge(question.difficulty)}"
        f"{type_badge(question.question_type)}"
        "</div>"
        f"<h3>{esc(question.question)}</h3>"
        f"{_answer_block(question.answer)}"
        '<p class="question-source">From '
        f'<a class="text-link" '
        f'href="{esc(href(QUESTIONS_PAGE, question.page))}">'
        f"{esc(question.post_id)}</a></p>"
        "</article>"
    )


def _post_card(
    model: SiteModel,
    post: KnowledgePost,
    slug: str,
    page: str,
) -> str:
    analysis = post.ai_analysis

    summary, _ = truncate(analysis.summary, SUMMARY_EXCERPT_LIMIT)

    meta_bits = [
        post.source.platform,
        post.source.captured_at.strftime("%Y-%m-%d"),
    ]

    if post.source.author:
        meta_bits.append(post.source.author)

    return "".join(
        [
            '<article class="card post-card">',
            '<h3 class="post-card-title">',
            f'<a class="text-link" '
            f'href="{esc(href(page, post_page(slug)))}">'
            f"{esc(post.id)}</a>",
            "</h3>",
            '<p class="post-card-meta">',
            esc(" \u00b7 ".join(meta_bits)),
            "</p>",
            '<p class="post-card-summary">',
            esc(summary or "No summary recorded."),
            "</p>",
            '<div class="badge-row">',
            topic_badges(
                analysis.topics, page, model.topic_slugs
            ),
            badge(
                str(len(post.interview_questions)),
                "badge badge-quiet",
                title="Interview questions",
            ),
            badge(
                str(len(analysis.concepts)),
                "badge badge-quiet",
                title="Concepts",
            ),
            "</div>",
            "</article>",
        ]
    )


def _post_card_grid(
    model: SiteModel,
    pairs: list[tuple[KnowledgePost, str]],
    page: str,
    *,
    title: str,
    subtitle: str,
) -> str:
    if not pairs:
        return empty_state(title, "Nothing to show here yet.")

    cards = [
        _post_card(model, post, slug, page)
        for post, slug in pairs
    ]

    return section(
        title,
        f'<div class="card-grid">{"".join(cards)}</div>',
        subtitle=subtitle,
    )


def _topic_directory(
    entries: tuple[TopicEntry, ...],
    page: str,
    heading: str,
    intro: str,
) -> str:
    if not entries:
        return empty_state(
            f"No {heading.lower()} yet",
            "This section is empty in the current knowledge base.",
        )

    cards = []

    for entry in entries:
        cards.append(
            "".join(
                [
                    '<a class="card topic-card" ',
                    f'href="{esc(href(page, entry.page))}">',
                    f"<h3>{esc(entry.label)}</h3>",
                    '<div class="badge-row">',
                    badge(
                        str(entry.post_count),
                        "badge badge-quiet",
                        title="Related posts",
                    ),
                    badge(
                        str(entry.question_count),
                        "badge badge-quiet",
                        title="Related questions",
                    ),
                    "</div>",
                    "</a>",
                ]
            )
        )

    return section(
        heading,
        f'<div class="topic-grid">{"".join(cards)}</div>',
        subtitle=f"{intro} {len(entries)} listed.",
    )


def _breadcrumb(
    page: str,
    trail: list[tuple[str, str | None]],
) -> str:
    crumbs = [
        '<nav class="breadcrumb" aria-label="Breadcrumb"><ol>'
    ]

    for label, target in trail:
        if target:
            crumbs.append(
                f'<li><a href="{esc(href(page, target))}">'
                f"{esc(label)}</a></li>"
            )
        else:
            crumbs.append(
                '<li><span aria-current="page">'
                f"{esc(label)}</span></li>"
            )

    crumbs.append("</ol></nav>")

    return "".join(crumbs)


def _button_link(
    page: str,
    label: str,
    target: str,
) -> str:
    return (
        f'<a class="button" href="{esc(href(page, target))}">'
        f"{esc(label)}</a>"
    )


def _select_field(
    element_id: str,
    label: str,
    options: str,
) -> str:
    return (
        '<label class="field">'
        f'<span class="field-label">{esc(label)}</span>'
        f'<select id="{esc(element_id)}" class="search-select">'
        '<option value="">All</option>'
        f"{options}</select></label>"
    )
