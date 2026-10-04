"""
Searching what the pictures say.

A separate index, loaded on demand, for one reason: size. The main
``search-index.json`` is fetched by every keystroke and deliberately
holds only enough to render a result card. The transcriptions are the
opposite -- 2,668 of them, some thousands of characters each -- and
folding them into the main index would make the file every search
downloads several times larger for the benefit of searches that mostly
never need it. So they get their own file, fetched once, on the first
query, and merged into the results.

The index is built from the post model rather than from the imported
records, which is a deliberate choice about dependencies. By the time
the site is generated the transcriptions have already been carried onto
``media.extracted_text`` by the pipeline, so this reads one source of
truth whether the text came from Gemini, from the CP12 vision
processor, or from both at different times. Reading the imported
package here instead would make the wiki depend on a file that may not
be on the machine that builds the site.

What is indexed is the *search* form, not the raw text. Whitespace is
collapsed and nothing else changes, because collapsing is the only
transformation that cannot alter what a term looks like -- stripping
punctuation would break ``COUNT(DISTINCT x)``, and these slides are
full of code.

A result carries the slide it was found on and the file it came from,
so "INNER JOIN" returns a post, a slide number and a filename rather
than an unexplained hit. Nothing here produces a filesystem path; the
filename is a bare name.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.wiki.analysis import SiteModel
from src.wiki.components import MEDIA_EXCERPT_LIMIT, collapse_whitespace
from src.wiki.naming import post_page


OCR_INDEX_FORMAT_VERSION = 1

#: Recorded next to the index so a reader knows an entry came from a
#: machine reading a picture and has not been checked by a person.
OCR_CAVEAT = (
    "Machine transcription of slide images. Text may be damaged, "
    "particularly on small or low-resolution images."
)


def build_ocr_records(model: SiteModel) -> list[dict]:
    """
    One record per transcribed slide.

    Only media that carries text *and* says how that text was obtained.
    A file that was read for its dimensions has no transcription and
    nothing to find; a file with text but no ``extraction_method`` has
    text whose origin is unknown, and indexing it would attribute it to
    nobody.
    """

    records: list[dict] = []

    for post, slug in zip(model.posts, model.post_slugs):
        for media in post.media:
            text = collapse_whitespace(media.extracted_text or "")

            if not text:
                continue

            if not media.extraction_method:
                continue

            records.append(
                {
                    # Which post, and where to read it.
                    "i": post.id,
                    "u": post_page(slug),
                    # One-based for display. The stored sequence is
                    # zero-based because that is how the archive orders
                    # slides, and a reader counting slides starts at one.
                    "s": media.sequence + 1,
                    # The bare filename, never the directory it sits in.
                    "f": _filename(media.path),
                    # Full search text: truncated text cannot be found.
                    "x": text,
                    # What the transcription claims, and what is
                    # observably true of it.
                    "st": media.ocr_status or "",
                    "ql": media.ocr_quality or "",
                    # How it was read, so the card can say so.
                    "m": media.extraction_method,
                }
            )

    records.sort(key=lambda record: (record["i"], record["s"], record["f"]))

    return records


def build_ocr_index(model: SiteModel) -> dict:
    """The full OCR index document."""

    records = build_ocr_records(model)

    return {
        "v": OCR_INDEX_FORMAT_VERSION,
        "caveat": OCR_CAVEAT,
        "slides": len(records),
        "records": records,
    }


def write_ocr_index(model: SiteModel, path: Path) -> Path:
    """Write ``ocr-index.json`` next to the generated pages."""

    payload = json.dumps(
        build_ocr_index(model),
        ensure_ascii=False,
        separators=(",", ":"),
    )

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(f"{payload}\n", encoding="utf-8")

    return path


def excerpt(text: str, limit: int = MEDIA_EXCERPT_LIMIT) -> str:
    """A displayable length of a transcription, cut at a boundary."""

    body = collapse_whitespace(text or "")

    if len(body) <= limit:
        return body

    return body[:limit].rstrip() + "…"


def _filename(path: str) -> str:
    """
    The last component of a media path.

    Written rather than imported so this module has no dependency on
    :mod:`pathlib` semantics for a string it does not open. The archive
    uses forward slashes in these paths and this project stores them
    that way regardless of which machine built the site.
    """

    return str(path or "").rsplit("/", 1)[-1]


__all__ = [
    "OCR_CAVEAT",
    "OCR_INDEX_FORMAT_VERSION",
    "build_ocr_index",
    "build_ocr_records",
    "excerpt",
    "write_ocr_index",
]