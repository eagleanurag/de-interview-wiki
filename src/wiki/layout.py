"""
The shared HTML shell: head, header, navigation and footer.

Every page is a complete document so the site can be opened straight
from the filesystem, and every in-site link is relative to the page
that contains it.
"""

from __future__ import annotations

import json

from src.wiki.components import SITE_NAME, esc
from src.wiki.naming import (
    INDEX_PAGE,
    CONCEPTS_PAGE,
    QUESTIONS_PAGE,
    SAVED_ITEMS_PAGE,
    REVISION_INDEX,
    SEARCH_PAGE,
    STYLE_FILE,
    TECHNOLOGIES_PAGE,
    TOPICS_PAGE,
    href,
)


#: The four things a reader revising for an interview reaches for.
#:
#: Concepts, Technologies, Topics and Saved Items used to be the primary
#: navigation, and that was the problem rather than the fix: a reader
#: who does not yet know whether the thing they half-remembers was filed
#: as a "concept" or a "technology" has to guess before they can search,
#: and all of those pages list labels the corpus generated rather than
#: subjects a person would revise.
#:
#: They were then demoted to a footer row, and finally stopped being
#: generated at all. The revision curriculum is the reader-facing
#: representation: a concept label belongs under the subtopic that
#: teaches it, a technology under the subject that uses it, and a saved
#: item is not revision material. Their pages, and every link into them,
#: are gone; the underlying concepts, technologies and saved-item
#: provenance remain in the knowledge base and in the search index's
#: data layer.
NAV_ITEMS = (
    ("Home", INDEX_PAGE),
    ("Subjects", REVISION_INDEX),
    ("Questions", QUESTIONS_PAGE),
    ("Search", SEARCH_PAGE),
)


def _head(
    *,
    page: str,
    title: str,
    description: str,
    config: dict | None,
) -> str:
    document_title = (
        SITE_NAME
        if title == SITE_NAME
        else f"{title} | {SITE_NAME}"
    )

    lines = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" '
        'content="width=device-width, initial-scale=1">',
        f"<title>{esc(document_title)}</title>",
        f'<meta name="description" content="{esc(description)}">',
        '<meta name="color-scheme" content="light dark">',
    ]

    if config is not None:
        payload = json.dumps(
            config, sort_keys=True, separators=(",", ":")
        )
        lines.append(
            '<script type="application/json" '
            f'id="wiki-config">{payload}</script>'
        )

    lines.append(
        '<link rel="stylesheet" '
        f'href="{esc(href(page, STYLE_FILE))}">'
    )
    lines.append("</head>")

    return "\n".join(lines)


def _navigation(page: str) -> str:
    items = []

    for label, target in NAV_ITEMS:
        current = (
            ' aria-current="page"' if target == page else ""
        )
        items.append(
            f'<li><a href="{esc(href(page, target))}"'
            f"{current}>{esc(label)}</a></li>"
        )

    return (
        '<nav class="site-nav" aria-label="Primary">'
        "<ul>" + "".join(items) + "</ul></nav>"
    )


def _header(page: str, tagline: str) -> str:
    return (
        '<header class="site-header">'
        '<div class="wrap header-inner">'
        f'<a class="brand" href="{esc(href(page, INDEX_PAGE))}">'
        f'<span class="brand-mark">DE</span>'
        f'<span class="brand-text">'
        f'<span class="brand-title">{esc(SITE_NAME)}</span>'
        f'<span class="brand-tagline">{esc(tagline)}</span>'
        "</span></a>"
        f"{_navigation(page)}"
        "</div>"
        "</header>"
    )


def _footer(page: str, generated_at: str | None) -> str:
    provenance = (
        "Source data generated "
        f"<code>{esc(generated_at)}</code>."
        if generated_at
        else "Source data generation time was not recorded."
    )

    # There is no secondary row any more. Concepts, Technologies, Topics
    # and Saved Items were the evidence behind the revision pages, and
    # while they were generated they were linked from here on every one
    # of the site's pages -- 2,601 links to four indexes of labels the
    # corpus invented. A reader who finished a subtopic and clicked
    # "Concepts" was back inside the archive the site was built to
    # replace. Their content is still in the knowledge base; it is just
    # no longer a place the reader is sent.
    return (
        '<footer class="site-footer">'
        '<div class="wrap">'
        f'<p>{esc(SITE_NAME)} &middot; {provenance}</p>'
        '<p class="footer-note">'
        "Static site generated from the canonical knowledge base. "
        "No data is fetched at runtime and no source content is "
        "reproduced beyond short attributed excerpts."
        "</p>"
        "</div>"
        "</footer>"
    )


def render_document(
    *,
    page: str,
    title: str,
    description: str,
    body: str,
    tagline: str = "Data Engineering interview preparation",
    generated_at: str | None = None,
    config: dict | None = None,
    scripts: tuple[str, ...] = (),
) -> str:
    """
    Render one complete HTML document.

    `scripts` are site-relative asset paths, loaded with a relative
    src so the page works from any mount point.
    """

    parts = [
        _head(
            page=page,
            title=title,
            description=description,
            config=config,
        ),
        "<body>",
        _header(page, tagline),
        f'<main class="wrap" id="main">{body}</main>',
        _footer(page, generated_at),
    ]

    for script in scripts:
        parts.append(
            f'<script src="{esc(href(page, script))}" '
            'defer></script>'
        )

    parts.append("</body>")
    parts.append("</html>")

    document = "\n".join(parts)

    if not document.endswith("\n"):
        document += "\n"

    return document
