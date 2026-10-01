"""
The Saved Items page.

One page, not one per item. A saved list runs to hundreds of links and
most of them never get a capture, so a page each would mean hundreds of
pages saying nothing at all. This shows what was actually captured,
grouped by how complete the capture was, and does not publish the
backlog: a link with nothing behind it has no page worth reading, and
it belongs in the drop zone where the person who saved it can act on it.

The grouping is the point. A post built from a screenshot and a post
built from a transcript are different things, and listing them the same
way would overstate whichever of the two is thinner.
"""

from __future__ import annotations

from src.wiki.components import (
    badge,
    definition_list,
    esc,
    href,
    section,
)
from src.wiki.layout import render_document
from src.wiki.naming import SAVED_ITEMS_PAGE


#: How each capture quality reads to a person, and what it means.
#:
#: The notes are not decoration. A reader looking at an image-only entry
#: needs to know that no text was recovered from it, or they will read
#: the summary as though it came from words the author wrote.
QUALITY_LABELS = {
    "text_and_media": "Captured text and media",
    "text": "Captured text",
    "partial": "Partly captured",
    "document_only": "Document only",
    "image_only": "Image only",
    "metadata_only": "Link only",
}

QUALITY_NOTES = {
    "text_and_media": (
        "The post's text was captured, with a file or two beside it."
    ),
    "text": "The post's text was captured.",
    "partial": (
        "Text was captured, but at least one supplied file could not be "
        "read. What is here is what could be opened; nothing was guessed "
        "at."
    ),
    "document_only": (
        "A document was captured but no text could be read from it, so "
        "there is no written body to summarise."
    ),
    "image_only": (
        "A screenshot was captured and kept as an image. No text was read "
        "from it, so the entry below says what it could not know."
    ),
    "metadata_only": "A saved link with no content behind it.",
}

#: The order the groups appear in: most complete first, because a reader
#: scanning the page wants the real captures at the top.
QUALITY_ORDER = (
    "text_and_media",
    "text",
    "partial",
    "document_only",
    "image_only",
    "metadata_only",
)

#: How each capture method reads. The distinction that matters to a
#: reader is between material this project was given and material it was
#: authorized to go and collect.
CAPTURE_METHOD_LABELS = {
    "user_export": "you exported this from your saved items",
    "user_provided": "you supplied the content",
    "user_saved_page": "you saved the page and supplied it",
    "user_bundle": "you assembled this from your own notes",
    "automated": "collected by an authorized run",
}


def _card(entry, page: str) -> str:
    """One captured saved item."""
    badges = [
        badge(
            QUALITY_LABELS.get(entry.quality, entry.quality),
            "badge badge-quiet",
            title="How complete the capture is",
        )
    ]

    if entry.interview_relevant:
        badges.append(
            badge("Interview relevant", "badge badge-domain")
        )

    if entry.question_count:
        badges.append(
            badge(
                str(entry.question_count),
                "badge badge-quiet",
                title="Questions from this post",
            )
        )

    facts: list[str] = []

    if entry.author:
        facts.append(f"by {esc(entry.author)}")

    if entry.saved_date:
        facts.append(f"saved {esc(entry.saved_date)}")

    if entry.captured_at:
        facts.append(f"captured {esc(entry.captured_at)}")

    if entry.capture_method:
        facts.append(
            esc(
                CAPTURE_METHOD_LABELS.get(
                    entry.capture_method,
                    entry.capture_method.replace("_", " "),
                )
            )
        )

    labels = "".join(
        f'<span class="badge badge-quiet">{esc(label)}</span>'
        for label in (
            *entry.topic_labels,
            *entry.technology_labels,
            *entry.concept_labels,
        )
    )

    origin = ""

    if entry.source_url:
        origin = (
            '<p class="muted">'
            f'<a href="{esc(entry.source_url)}" rel="noopener '
            'nofollow" target="_blank">Open the original source</a>'
            "</p>"
        )

    return "".join(
        [
            '<article class="card topic-card">',
            f'<div class="badge-row">{"".join(badges)}</div>',
            f'<h3><a href="{esc(href(page, entry.page))}">'
            f"{esc(entry.label)}</a></h3>",
            (
                f'<p class="muted">{" &middot; ".join(facts)}</p>'
                if facts
                else ""
            ),
            f'<div class="badge-row">{labels}</div>' if labels else "",
            origin,
            "</article>",
        ]
    )


def render_saved_items(model) -> str:
    """
    The saved items this knowledge base came from.

    Nothing at all is invented here: every entry is a post in the
    knowledge base that carries saved-item provenance, and every number
    is counted off those entries.
    """
    page = SAVED_ITEMS_PAGE

    entries = model.saved_items

    if not entries:
        return render_document(
            page=page,
            title="Saved Items",
            description="Posts captured from a saved-items list.",
            body=(
                '<section class="hero">'
                "<h1>Saved Items</h1></section>"
                + section(
                    "No saved items yet",
                    '<p class="muted">Posts imported from a saved-items '
                    "list appear here, each with the capture it came "
                    "from. Nothing is listed until a saved post has "
                    "actually been captured, and a link that was saved "
                    "but never captured has nothing to read.</p>",
                )
            ),
            generated_at=model.generated_at,
        )

    by_quality: dict[str, list] = {}

    for entry in entries:
        by_quality.setdefault(entry.quality, []).append(entry)

    sections = []

    for quality in QUALITY_ORDER:
        group = by_quality.get(quality)

        if not group:
            continue

        sections.append(
            section(
                f"{QUALITY_LABELS.get(quality, quality)} "
                f"({len(group)})",
                '<div class="topic-grid">'
                + "".join(_card(entry, page) for entry in group)
                + "</div>",
                subtitle=QUALITY_NOTES.get(quality, ""),
            )
        )

    relevant = sum(1 for entry in entries if entry.interview_relevant)

    questions = sum(entry.question_count for entry in entries)

    body = "".join(
        [
            '<section class="hero hero-compact">',
            "<h1>Saved Items</h1>",
            '<p class="muted">Posts captured from a list you exported '
            "from LinkedIn. Each one links back to the post it came "
            "from.</p>",
            "</section>",
            section(
                "At a glance",
                definition_list(
                    [
                        ("Captured items", esc(str(len(entries)))),
                        (
                            "Interview relevant",
                            esc(f"{relevant} of {len(entries)}"),
                        ),
                        (
                            "Questions generated",
                            esc(str(questions)),
                        ),
                    ]
                ),
                subtitle=(
                    "A saved link with no capture behind it is not listed. "
                    "It stays in your drop zone as a link, because there "
                    "is no content to read."
                ),
            ),
            *sections,
        ]
    )

    return render_document(
        page=page,
        title="Saved Items",
        description=(
            "Posts captured from a saved-items list, each with the "
            "source it came from."
        ),
        body=body,
        generated_at=model.generated_at,
    )


__all__ = [
    "CAPTURE_METHOD_LABELS",
    "QUALITY_LABELS",
    "QUALITY_NOTES",
    "QUALITY_ORDER",
    "render_saved_items",
]
