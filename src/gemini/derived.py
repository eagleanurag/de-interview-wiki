"""
The package's technology claims, matched to the posts they belong to.

The imported package names technologies per *activity*. This project's
posts are keyed by a hashed identifier derived from that activity, so
something has to connect the two, and this is where it happens.

The connection is made through the media filename, which both sides
carry. A post's ``media/activity_7263033731471351808_slide_0.jpg`` names
its activity, and the package's records for that activity name the same
file. Nothing else is trusted: the package's own ``POST-004`` is a
numbering of its own invention, kept only as provenance text.

**A claim is attributed to a slide only when the slide's own words
support it.** A technology the package lists for a post whose
transcriptions never mention it is still recorded, but with no slide
behind it and a note saying so. That is the difference between "the
package says this slide is about SQL" and "this slide says SQL", and
recording which of the two it is keeps the first from being read as the
second.

Three things this deliberately does not do.

* **It does not merge similar names.** ``Azure Data Factory`` and ``Data
  Factory`` stay two technologies. Deciding they name one concept is the
  existing taxonomy's judgement, and a merge made here would be silent
  and invisible.
* **It does not create topics or concepts.** The package asserts
  technologies. Asserting that ``SQL`` is a topic, or that ``Bayes'
  theorem`` is a concept, is a larger claim than it makes, and those stay
  with the enricher, which grounds them against the post's own text.
* **It does not run when nothing was imported.** Every entry point
  returns empty before touching the disk, so a project that never ran an
  import aggregates exactly as it did before.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from src.aggregation.consolidation import DerivedClaim
from src.gemini.crosscheck import activity_id_for_filename
from src.gemini.importer import (
    DEFAULT_OUTPUT,
    load_knowledge,
    load_records,
    read_manifest,
)
from src.gemini.models import GeminiImageRecord, GeminiPostKnowledge
from src.redaction import redact_text


#: What an unsupported claim says about itself. Fixed text rather than an
#: empty string, so a reader of the knowledge base sees the absence
#: instead of a blank that could mean anything.
NO_SLIDE_EVIDENCE = (
    "The package attributes this technology to the post, but no slide "
    "transcription for it mentions the name."
)


def _normalise(text: str) -> str:
    """Folded for comparison: case, accents and punctuation gone."""

    decomposed = unicodedata.normalize("NFKD", text.lower())

    stripped = "".join(
        char for char in decomposed if not unicodedata.combining(char)
    )

    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", stripped)).strip()


def mentions(text: str, term: str) -> bool:
    """
    Whether a transcription actually names a technology.

    Matched on the whole normalised term between spaces, so ``Git`` does
    not match one half of ``GitHub`` and a two-letter initial does not
    match everything. A technology whose letters merely appear across a
    word boundary is not evidence that the slide was about it.
    """

    needle = _normalise(term)

    haystack = _normalise(text)

    if not needle or not haystack:
        return False

    return f" {needle} " in f" {haystack} "


def claims_for_post(
    knowledge: GeminiPostKnowledge,
    records: list[GeminiImageRecord],
) -> list[DerivedClaim]:
    """
    Every claim one post contributes, with a slide behind each that can
    be found.

    One claim per supporting slide, so a technology appearing on three
    slides of a carousel is recorded three times and a reader can see it
    recurs rather than being told it does. A claim with no supporting
    slide is still made once, marked, because dropping it would make this
    project's coverage quietly smaller than the evidence available.
    """

    # Fragments the importer already resolved to a file. Their text is
    # the slide's own words, which is the strongest evidence available.
    fragments: dict[str, list[str]] = {}

    for key, text in knowledge.slide_fragments.items():
        filename = key.split("#", 1)[0]

        fragments.setdefault(filename, []).append(text)

    transcriptions = {
        record.filename: record.readable_text
        for record in records
    }

    slides_by_filename = {
        record.filename: record.slide_number for record in records
    }

    claims: list[DerivedClaim] = []

    for name in knowledge.technologies:
        supporting = sorted(
            filename
            for filename, text in transcriptions.items()
            if mentions(text, name)
        )

        if not supporting:
            claims.append(
                DerivedClaim(
                    name=name,
                    source_kind="image_ocr",
                    group_id=knowledge.group_id,
                    evidence=NO_SLIDE_EVIDENCE,
                )
            )

            continue

        for filename in supporting:
            quoted = " ".join(fragments.get(filename, [])).strip()

            evidence = quoted or (
                "the term appears in this slide's transcription"
            )

            # The quotation ends up in the knowledge base, so it is
            # sanitised on the way in rather than relying on the writer's
            # guard to reject the whole file over one slide's text.
            evidence = redact_text(evidence)

            claims.append(
                DerivedClaim(
                    name=name,
                    source_kind="image_ocr",
                    group_id=knowledge.group_id,
                    filename=filename,
                    slide_number=slides_by_filename.get(filename),
                    evidence=evidence,
                )
            )

    return claims


def derived_claims_for_posts(posts) -> dict[str, list[DerivedClaim]]:
    """
    Every post's claims, keyed by the post's own identifier.

    Matches each post to the package through the activity its media
    filenames name. Where that match fails nothing is inferred and the
    post contributes nothing, so a post whose files the package never
    mentioned is left exactly as the consolidation would have left it.
    """

    base = DEFAULT_OUTPUT

    if read_manifest(base) is None:
        return {}

    knowledge = load_knowledge(base)

    if not knowledge:
        return {}

    records = load_records(base)

    if not records:
        return {}

    by_activity: dict[str, list[GeminiImageRecord]] = {}

    for record in records:
        activity = record.activity_id

        if not activity:
            continue

        by_activity.setdefault(activity, []).append(record)

    result: dict[str, list[DerivedClaim]] = {}

    for post in posts:
        activities = {
            activity_id_for_filename(item.path)
            for item in post.media
            if activity_id_for_filename(item.path)
        }

        for activity in activities:
            members = by_activity.get(activity)

            if not members:
                continue

            for group_id in sorted(
                {record.post_id.upper() for record in members}
            ):
                entry = knowledge.get(group_id)

                if entry is None:
                    continue

                claims = claims_for_post(entry, members)

                if claims:
                    result.setdefault(post.id, []).extend(claims)

    return result


def claim_counts(
    output: str | Path | None = None,
) -> dict[str, int]:
    """
    How many claims are backed by a slide, and how many are not.

    The split is the number worth knowing. A claim with a slide behind it
    can be checked by reading that slide; one without is the package's
    assertion about a post, recorded so nothing is lost, and weaker.
    Reporting the two together would make the second kind look like the
    first.
    """

    base = Path(output) if output is not None else DEFAULT_OUTPUT

    knowledge = load_knowledge(base)

    records = load_records(base)

    by_group: dict[str, list[GeminiImageRecord]] = {}

    for record in records:
        by_group.setdefault(record.post_id.upper(), []).append(record)

    supported = 0
    unsupported = 0

    for group_id, entry in knowledge.items():
        claims = claims_for_post(entry, by_group.get(group_id.upper(), []))

        for claim in claims:
            if claim.filename:
                supported += 1
            else:
                unsupported += 1

    return {
        "claims_with_a_slide": supported,
        "claims_without_one": unsupported,
        "posts": len(knowledge),
    }


__all__ = [
    "NO_SLIDE_EVIDENCE",
    "claim_counts",
    "claims_for_post",
    "derived_claims_for_posts",
    "mentions",
]