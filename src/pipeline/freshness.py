"""
Whether an existing enrichment result still describes its post.

Two decisions live here and nothing else does:

* **is this result reusable?** Its recorded fingerprint is compared with
  the post's current content digest, and the enricher version has to
  match as well. A result from an older contract is not reusable even
  when the analysis is largely right, because the whole point of the
  change was that the knowledge base must never hold something the
  current rules would have removed.
* **what has gone stale in a reused result?** Only the attribution --
  where the content came from and what files came with it. The analysis,
  the questions and the classification are exactly what the model was
  paid for and nothing about them has gone out of date, so replacing
  them would spend real time to arrive at the same answer.

A field the post has stopped claiming is removed rather than left, so a
post that has stopped claiming to have been captured stops claiming it.
"""

from __future__ import annotations

import json
from pathlib import Path


#: Bumped when the enrichment contract or the prompt changes.
#:
#: Version 3 added source grounding, so a result from version 2 may
#: contain a question about a technology the source never mentioned.
#: Those results are not reusable even though the underlying analysis is
#: largely right: the point of the check is that the knowledge base never
#: holds one, and a cached result would put it straight back.
ENRICHER_VERSION = "3"

#: The visual processor's version, imported rather than restated so that
#: the two cannot drift apart. A result enriched from images read by a
#: different processor is not the result the current processor would
#: produce, which is the same argument as the one above.
try:
    from src.visual.models import PROCESSOR_VERSION as VISUAL_PROCESSOR_VERSION

except ImportError:  # pragma: no cover - visual package always present
    VISUAL_PROCESSOR_VERSION = "1"

#: The imported-transcription processor's version, imported for the same
#: reason and with the same fallback. Kept distinct from
#: ``VISUAL_PROCESSOR_VERSION`` because the two read different sources
#: with different authority, and a result derived from one is not
#: automatically a result the other would produce.
try:
    from src.gemini.bridge import PROCESSOR_VERSION as OCR_PROCESSOR_VERSION

except ImportError:  # pragma: no cover - gemini package always present
    OCR_PROCESSOR_VERSION = "1"


def _reusable(
    target: Path,
    digest: str,
    visual_digest: str = "",
    ocr_digest: str = "",
) -> dict | None:
    """
    An existing worker result that still matches the current content.

    Read from the file rather than trusted from a stamp, because the
    stamp lives in the post and the result lives here, and the two can
    disagree if a run was interrupted between them.

    ``visual_digest`` is compared when given. A post whose images were
    re-read, reordered or replaced has different visual content even
    though its text and its file bytes are untouched, and reusing the
    enrichment would publish knowledge derived from slides that are no
    longer there.

    ``ocr_digest`` is the same argument one step further out. It covers
    transcriptions imported from outside the post, so a corrected
    transcription re-enriches the post it informs. It is compared
    independently of the visual digest because the two come from
    different sources: requiring both to match is what stops a change to
    either being mistaken for a change to the other.

    Each is compared only when the caller supplies one. A post with no
    images passes neither and is decided on its source alone, which is
    the behaviour every post had before visual enrichment existed.
    """

    try:
        payload = json.loads(target.read_text(encoding="utf-8"))

    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None

    recorded = payload.get("_enrichment") or {}

    if not isinstance(recorded, dict):
        return None

    if recorded.get("source_digest") != digest:
        return None

    if recorded.get("enricher_version") != ENRICHER_VERSION:
        return None

    if visual_digest:
        if recorded.get("visual_digest") != visual_digest:
            return None

        if recorded.get("visual_processor_version") != VISUAL_PROCESSOR_VERSION:
            return None

    if ocr_digest:
        if recorded.get("ocr_digest") != ocr_digest:
            return None

        if recorded.get("ocr_processor_version") != OCR_PROCESSOR_VERSION:
            return None

    return payload


def _refresh_provenance(
    payload: dict,
    post: object,
) -> dict:
    """
    Copy the current post's attribution onto a reused worker result.

    Only the fields that say where the content came from and what it
    looks like are replaced. The analysis, the questions and the
    classification are left exactly as they were, because they are what
    the model was paid for and nothing about them has gone stale.

    Media derived from reading a picture -- ``extracted_text`` read off a
    slide, its sequence, its content digest, how it was read -- is
    preserved rather than overwritten. The committed post carries the
    file but never its interpretation, so copying the post's media
    verbatim would strip the visual knowledge out of every reused result
    and republish the post as though its images had never been read.
    """

    current = post.model_dump(mode="json")

    for key in ("source", "media"):
        value = current.get(key)

        if key == "media" and value is not None:
            payload[key] = _merge_visual(value, payload.get(key) or [])

        elif value is None:
            payload.pop(key, None)

        else:
            payload[key] = value

    if current.get("saved_item") is None:
        payload.pop("saved_item", None)
    else:
        payload["saved_item"] = current["saved_item"]

    payload["original_text"] = current.get("original_text", "")

    return payload



#: Media fields that come from reading a file rather than from the post
#: describing it. Never overwritten by a refresh.
#:
#: The last four arrived with the imported image-knowledge package. They
#: are reading-derived for exactly the same reason ``extracted_text`` is,
#: and they matter more than the rest: they are what tells a reader that
#: a transcription is a transcription and how legible it looked. Dropping
#: them on refresh would republish a machine's garbled reading of a
#: 480-pixel screenshot as though it were plain fact.
_VISUAL_FIELDS = (
    "extracted_text",
    "sequence",
    "role",
    "sha256",
    "extraction_method",
    "ocr_status",
    "ocr_quality",
    "ocr_note",
    "source_kind",
    "slide_count",
)


def _merge_visual(
    current: list[dict],
    stored: list[dict],
) -> list[dict]:
    """
    Keep the stored reading of each picture while taking the post's own
    attribution for it.

    Keyed on the path, which is the one thing both sides agree on: the
    post names the file it attached, and the result names the file the
    text was read from. A file the post no longer references keeps
    nothing, because a post that has stopped claiming a picture should
    stop claiming what was read off it.
    """

    by_path = {
        entry.get("path"): entry
        for entry in stored
        if isinstance(entry, dict)
    }

    merged: list[dict] = []

    for entry in current:
        if not isinstance(entry, dict):
            continue

        previous = by_path.get(entry.get("path"))

        if not previous:
            merged.append(entry)
            continue

        combined = dict(entry)

        for name in _VISUAL_FIELDS:
            value = previous.get(name)

            if value:
                combined[name] = value

        merged.append(combined)

    return merged


__all__ = ["ENRICHER_VERSION", "_refresh_provenance", "_reusable"]