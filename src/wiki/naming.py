"""
Deterministic slugs and relative link helpers.

Every link in the generated site is relative to the page that contains
it. That makes the same output work three ways with no base URL
configuration at all:

* opened directly from the filesystem (``file://``)
* served from ``/``
* served from a GitHub Pages project path such as ``/repo-name/``

Repository owner and name never appear in application logic.
"""

from __future__ import annotations

import posixpath
import re
import unicodedata

from src.aggregation.consolidation import MAX_SLUG_LENGTH, slug_for


MAX_ID_LENGTH = 80

# Keeps word separators (underscore, hyphen, dot) so IDs stay legible.
_UNSAFE_ID_CHARACTERS = re.compile(r"[^A-Za-z0-9_.-]+")

ASSETS_DIR = "assets"
POSTS_DIR = "posts"
TOPICS_DIR = "topics"
CONCEPTS_DIR = "concepts"
TECHNOLOGIES_DIR = "technologies"

INDEX_PAGE = "index.html"
SEARCH_PAGE = "search.html"
QUESTIONS_PAGE = "questions.html"
NOT_FOUND_PAGE = "404.html"

#: The revision tree: Subject, then Subtopic. Named separately from
#: :data:`TOPICS_PAGE` because the two are different things and used to be
#: confused. "Topics" was the archive's grouping of posts by corpus
#: label; "subjects" is what a candidate revises. Only the latter is
#: generated.
SUBJECTS_PAGE = "subjects.html"

# Retained as names because the renderers and the model still describe
# the evidence layer. No page is written at any of these paths, so
# nothing links to them: see src.wiki.generator.
TOPICS_PAGE = "topics.html"
CONCEPTS_PAGE = "concepts.html"
TECHNOLOGIES_PAGE = "technologies.html"
SAVED_ITEMS_PAGE = "saved-items.html"

SEARCH_INDEX_FILE = f"{ASSETS_DIR}/search-index.json"

#: Fetched separately and only once a search has run. Split from the
#: main index because the transcriptions are large and most searches do
#: not need them; see :mod:`src.wiki.ocr_index`.
OCR_INDEX_FILE = f"{ASSETS_DIR}/ocr-index.json"
STYLE_FILE = f"{ASSETS_DIR}/style.css"
SEARCH_SCRIPT = f"{ASSETS_DIR}/search.js"
QUESTIONS_SCRIPT = f"{ASSETS_DIR}/questions.js"

MANIFEST_FILE = ".wiki-manifest.json"


def slugify(
    value: str,
    *,
    fallback: str = "untitled",
) -> str:
    """
    Reduce arbitrary text to a stable, URL-safe slug.

    Delegates to the aggregator rather than deriving one here. The
    knowledge base is the authoritative artifact and it records each
    label's slug, so the site has to produce the same string: a reader
    who follows a concept's ``slug`` out of the JSON must arrive at the
    page that was generated. Two implementations that agree today would
    be free to disagree the moment one of them gained a bound.
    """

    return slug_for(value, fallback=fallback)


def safe_id(value: str, *, fallback: str = "post") -> str:
    """
    Make a post ID safe to use as a file name.

    Post IDs are the canonical public identifier, and existing IDs use
    underscores (`sample_001`). Preserving them keeps generated URLs
    readable and predictable, and keeps the post page filename equal
    to the ID that appears in the page itself. Only characters that
    are genuinely unsafe in a path or a URL are replaced.

    Collisions are still resolved by the caller, because two distinct
    IDs can reduce to the same safe form.
    """

    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = normalized.encode(
        "ascii", "ignore"
    ).decode("ascii")

    candidate = _UNSAFE_ID_CHARACTERS.sub("-", ascii_only).strip("-")

    if len(candidate) > MAX_ID_LENGTH:
        candidate = candidate[:MAX_ID_LENGTH].rstrip("-")

    return candidate or fallback


def href(
    from_page: str,
    to_page: str,
) -> str:
    """
    Relative href from one generated page to another.

    Both arguments are site-relative POSIX paths. `posixpath` is used
    deliberately so links always use forward slashes regardless of the
    host operating system.
    """

    from_dir = posixpath.dirname(from_page) or "."

    return posixpath.relpath(to_page, from_dir)


class SlugRegistry:
    """
    Assigns unique slugs in insertion order.

    Two different labels can reduce to the same slug, which would make
    one page silently overwrite the other and break links. Collisions
    get a numeric suffix in first-come order, so the same ordered input
    always produces the same slugs.
    """

    def __init__(
        self,
        prefix: str = "",
        *,
        transform=slugify,
    ) -> None:
        self._prefix = prefix
        self._transform = transform
        self._counters: dict[str, int] = {}
        self._used: set[str] = set()

    def register(self, label: str) -> str:
        base = self._transform(label)
        candidate = f"{self._prefix}{base}"

        if candidate in self._used:
            index = self._counters.get(base, 0)

            while True:
                index += 1
                candidate = f"{self._prefix}{base}-{index}"

                if candidate not in self._used:
                    break

            self._counters[base] = index

        self._used.add(candidate)

        return candidate


def post_page(slug: str) -> str:
    """Site-relative path of a generated post page."""

    return f"{POSTS_DIR}/{slug}.html"


def topic_page(slug: str) -> str:
    """Site-relative path of a generated topic page."""

    return f"{TOPICS_DIR}/{slug}.html"


def concept_page(slug: str) -> str:
    """Site-relative path of a generated concept page."""

    return f"{CONCEPTS_DIR}/{slug}.html"


def technology_page(slug: str) -> str:
    """Site-relative path of a generated technology page."""

    return f"{TECHNOLOGIES_DIR}/{slug}.html"
