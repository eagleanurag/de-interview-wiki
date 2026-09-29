"""
Small, reusable rendering helpers.

Everything that reaches the browser goes through `esc`. The knowledge
base contains third-party source text (scraped post bodies, OCR
output, model output), so it is treated as untrusted and escaped at the
point of insertion rather than trusted by convention.

Every URL is additionally scheme-checked before it becomes an href, so
a `javascript:` or `data:` URL in source data cannot become a live link.
"""

from __future__ import annotations

from html import escape

from src.wiki.naming import href, topic_page


SITE_NAME = "DE Interview Wiki"

ALLOWED_LINK_SCHEMES = frozenset({"http", "https"})

DIFFICULTY_LABELS = {
    "easy": "Easy",
    "medium": "Medium",
    "hard": "Hard",
}

TYPE_LABELS = {
    "theory": "Theory",
    "coding": "Coding",
    "scenario": "Scenario",
    "architecture": "Architecture",
    "troubleshooting": "Troubleshooting",
}

SUMMARY_EXCERPT_LIMIT = 260
INDEX_SUMMARY_LIMIT = 180
INDEX_SOURCE_EXCERPT_LIMIT = 1200
POST_SOURCE_EXCERPT_LIMIT = 2000
MEDIA_EXCERPT_LIMIT = 600


def esc(value: object) -> str:
    """Escape any value for insertion into HTML text or attributes."""

    if value is None:
        return ""

    return escape(str(value), quote=True)


def safe_link(url: object) -> str | None:
    """
    Return `url` only when it is an http(s) link.

    Relative, scheme-less, and non-web schemes return None so the
    renderer can fall back to plain text.
    """

    if not url:
        return None

    candidate = str(url).strip()

    if not candidate:
        return None

    if ":" not in candidate:
        return None

    scheme = candidate.split(":", 1)[0].strip().casefold()

    if scheme not in ALLOWED_LINK_SCHEMES:
        return None

    return candidate


def collapse_whitespace(text: object) -> str:
    return " ".join(str(text or "").split())


def truncate(
    text: object,
    limit: int,
) -> tuple[str, bool]:
    """
    Collapse and truncate text for an excerpt.

    Returns the excerpt and whether truncation happened, so callers
    can tell the reader that content was shortened.
    """

    cleaned = collapse_whitespace(text)

    if limit <= 0 or len(cleaned) <= limit:
        return cleaned, False

    window = cleaned[:limit]
    boundary = window.rfind(" ")

    if boundary > limit // 2:
        window = window[:boundary]

    return window.rstrip(" ,.;:-") + "\u2026", True


def excerpt(
    text: object,
    limit: int,
    *,
    empty: str = "Not recorded in the source post.",
) -> str:
    """Collapsed, truncated plain-text excerpt, escaped for output."""

    result, _ = truncate(text, limit)

    return esc(result) if result else esc(empty)


def badge(
    text: object,
    css_class: str,
    *,
    title: str | None = None,
) -> str:
    attributes = f' class="{esc(css_class)}"'

    if title:
        attributes += f' title="{esc(title)}"'

    return f"<span{attributes}>{esc(text)}</span>"


def topic_badges(
    labels: list[str] | tuple[str, ...],
    page: str,
    slugs: dict[str, str] | None = None,
) -> str:
    """
    Topic badges that link to their topic page.

    Labels without a topic page (for example subtopics, which get
    their own pages too) fall back to a non-linked badge.
    """

    lookup = slugs or {}

    rendered = []

    for label in labels:
        slug = lookup.get(label)

        if not slug:
            rendered.append(
                badge(label, "badge badge-topic", title="Topic")
            )
            continue

        rendered.append(
            f'<a class="badge badge-topic badge-link" '
            f'href="{esc(href(page, topic_page(slug)))}" '
            f'title="Topic: {esc(label)}">{esc(label)}</a>'
        )

    return "".join(rendered)


def difficulty_badge(value: object) -> str:
    difficulty = str(value or "").strip()

    if not difficulty:
        return ""

    label = DIFFICULTY_LABELS.get(
        difficulty, difficulty.capitalize()
    )

    return badge(
        label,
        f"badge badge-difficulty diff-{esc(difficulty)}",
        title=f"Difficulty: {label}",
    )


def type_badge(value: object) -> str:
    question_type = str(value or "").strip()

    if not question_type:
        return ""

    return badge(
        TYPE_LABELS.get(
            question_type, question_type.capitalize()
        ),
        "badge badge-type",
        title=f"Question type: {question_type}",
    )


def stat_tile(
    value: int,
    label: str,
    *,
    detail: str = "",
    css_class: str = "",
) -> str:
    detail_html = ""

    if detail:
        detail_html = (
            f'<p class="stat-detail">{esc(detail)}</p>'
        )

    classes = "stat-tile"

    if css_class:
        classes = f"{classes} {css_class}"

    return (
        f'<div class="{esc(classes)}">'
        f'<p class="stat-value">{esc(value)}</p>'
        f'<p class="stat-label">{esc(label)}</p>'
        f"{detail_html}"
        f"</div>"
    )


def stat_grid(tiles: list[str], *, aria_label: str) -> str:
    return (
        f'<div class="stat-grid" aria-label="{esc(aria_label)}">'
        + "".join(tiles)
        + "</div>"
    )


def empty_state(
    title: str,
    message: str,
) -> str:
    return (
        '<div class="empty-state">'
        f'<h2>{esc(title)}</h2>'
        f'<p>{esc(message)}</p>'
        "</div>"
    )


def page_link(
    label: str,
    target_page: str,
    from_page: str,
    *,
    css_class: str = "text-link",
) -> str:
    return (
        f'<a class="{esc(css_class)}" '
        f'href="{esc(href(from_page, target_page))}">'
        f"{esc(label)}</a>"
    )


def section(
    title: str,
    body: str,
    *,
    css_class: str = "card",
    subtitle: str = "",
) -> str:
    subtitle_html = ""

    if subtitle:
        subtitle_html = (
            f'<p class="section-subtitle">{esc(subtitle)}</p>'
        )

    return (
        f'<section class="{esc(css_class)}">'
        f"<h2>{esc(title)}</h2>"
        f"{subtitle_html}"
        f"{body}"
        "</section>"
    )


def definition_list(pairs: list[tuple[str, str]]) -> str:
    rows = "".join(
        f"<div class=\"def-row\">"
        f'<dt>{esc(key)}</dt>'
        f"<dd>{value}</dd>"
        "</div>"
        for key, value in pairs
        if value
    )

    if not rows:
        return ""

    return f'<dl class="def-list">{rows}</dl>'
