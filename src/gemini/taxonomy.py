"""
The package's technology claims, and the topic index.

Two files, and they disagree in a way that matters.

``LINKEDIN_TOPIC_INDEX.md`` is a clean mapping: a topic heading followed
by the group identifiers that mention it.

``LINKEDIN_TECHNICAL_KNOWLEDGE.md`` is per post, and each post carries a
technology list plus short quotations attributed to particular slides::

    ### POST-004

    **Technologies/topics:** SQL, Apache Spark, Data Engineering

    - Slide 3: Probability-the Science of Uncertainty rior (Bayes rule)

The attribution is the problem. The inventory numbers its slides from
zero; the knowledge archive prints ``## Slide 1`` for the file named
``slide_0``; and this file's ``Slide 3`` is not reconcilable with either
from the text alone. A slide number that cannot be trusted is not used
to resolve anything, so every fragment here is attached to an image by
*matching its words against that image's transcription* and the number
is carried alongside as an unverified claim.

Content matching is used because it can be checked. A fragment that
appears in exactly one transcription is attached to that image, and
that is verifiable by anyone reading the record. A fragment that
appears in none, or in several, is recorded as unresolved rather than
assigned to whatever slide the number happened to name.

Topic names are not normalised beyond case. ``Azure Databricks`` and
``Databricks`` are different strings and this package gave no rule for
merging them, so they stay distinct and the project's existing taxonomy
decides. Collapsing them here would quietly discard a distinction
nobody asked this layer to make.
"""

from __future__ import annotations

import re
import unicodedata

from src.gemini.models import (
    GeminiImageRecord,
    GeminiPostKnowledge,
    GeminiTopicIndex,
    MatchVerdict,
)
from src.gemini.package_reader import TECHNICAL_MD, TOPIC_INDEX_MD
from src.redaction import redact_text


_COVERAGE = re.compile(
    r"^-\s*\*\*(?P<topic>[^*]+?)\*\*\s*[—–-]\s*(?P<count>\d+)\s*"
    r"(?P<unit>\S+.*?)\s*$"
)

_POST_HEADING = re.compile(r"^#{2,3}\s+(POST-\d+)\s*$", re.IGNORECASE)

_TECHNOLOGIES = re.compile(
    r"^\*\*Technologies/topics:\*\*\s*(?P<value>.+)$", re.IGNORECASE
)

_FRAGMENT = re.compile(
    r"^-\s*[Ss]lide\s+(?P<number>\d+)\s*:\s*(?P<text>.+)$"
)

_TOPIC_HEADING = re.compile(r"^##\s+(?P<topic>[^#\n]+?)\s*$")

_GROUP_REF = re.compile(r"POST-\d+", re.IGNORECASE)


def _normalise(text: str) -> str:
    """Folded for comparison only: case, accents and punctuation gone."""

    decomposed = unicodedata.normalize("NFKD", text.lower())

    stripped = "".join(
        char for char in decomposed if not unicodedata.combining(char)
    )

    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", stripped)).strip()


def _topic_key(name: str) -> str:
    """
    Identity of a topic name: case and surrounding space only.

    ``SQL`` and ``sql`` are the same topic. ``Azure Data Factory`` and
    ``Data Factory`` are not the same string, and deciding they are one
    concept is the existing taxonomy's job rather than this layer's.
    """

    return " ".join(name.split()).lower()


def parse_topic_index(contents: dict[str, str]) -> GeminiTopicIndex:
    """Topic to group identifiers, as stated."""

    body = contents.get(TOPIC_INDEX_MD, "")

    topics: dict[str, list[str]] = {}

    current: str | None = None

    for line in body.splitlines():
        stripped = line.strip()

        if not stripped:
            continue

        heading = _TOPIC_HEADING.match(stripped)

        if heading:
            current = heading.group("topic").strip()

            topics.setdefault(current, [])

            # A topic whose references sit on the heading line itself is
            # unusual but not malformed, so fall through and collect
            # them rather than dropping them.
            if not _GROUP_REF.search(stripped):
                continue

        if current is None:
            continue

        for reference in _GROUP_REF.findall(stripped):
            group = reference.upper()

            bucket = topics.setdefault(current, [])

            if group not in bucket:
                bucket.append(group)

    return GeminiTopicIndex(topics=topics)


def coverage(contents: dict[str, str]) -> dict[str, int]:
    """
    The package's own "N activity groups" counts.

    Kept verbatim and kept separate from the counts actually found. The
    package's numbers are a claim about its own work and are reported
    next to the measured ones so a discrepancy is visible instead of
    being resolved in either direction.
    """

    body = contents.get(TECHNICAL_MD, "")

    counts: dict[str, int] = {}

    in_coverage = False

    for line in body.splitlines():
        stripped = line.strip()

        if stripped.startswith("## Topic Coverage"):
            in_coverage = True

            continue

        if in_coverage and stripped.startswith("#"):
            break

        if not in_coverage:
            continue

        match = _COVERAGE.match(stripped)

        if not match:
            continue

        value = match.group("count").replace(",", "")

        if value.isdigit():
            counts[match.group("topic").strip()] = int(value)

    return counts


def parse_technical(
    contents: dict[str, str],
) -> dict[str, GeminiPostKnowledge]:
    """
    Per-post technology claims, with the fragments they rest on.

    The fragments are the reason this module exists at all. A reader told
    that POST-004 covers SQL and machine learning can only weigh that
    claim if they can see the two sentences it came from, so the
    sentences are stored beside the claim rather than discarded as
    detail.
    """

    body = contents.get(TECHNICAL_MD, "")

    knowledge: dict[str, GeminiPostKnowledge] = {}

    current: GeminiPostKnowledge | None = None

    for line in body.splitlines():
        stripped = line.strip()

        if not stripped:
            continue

        heading = _POST_HEADING.match(stripped)

        if heading:
            group = heading.group(1).upper()

            current = knowledge.get(group)

            if current is None:
                current = GeminiPostKnowledge(group_id=group)

                knowledge[group] = current

            continue

        if current is None:
            continue

        technologies = _TECHNOLOGIES.match(stripped)

        if technologies:
            names = [
                part.strip()
                for part in technologies.group("value").split(",")
                if part.strip()
            ]

            seen = {_topic_key(name) for name in current.technologies}

            for name in names:
                key = _topic_key(name)

                if key and key not in seen:
                    seen.add(key)

                    current.technologies.append(name)

            continue

        fragment = _FRAGMENT.match(stripped)

        if fragment:
            current.slide_fragments[fragment.group("number")] = (
                fragment.group("text").strip()
            )

    return knowledge


def resolve_fragments(
    knowledge: dict[str, GeminiPostKnowledge],
    records: list[GeminiImageRecord],
) -> tuple[dict[str, GeminiPostKnowledge], list[str]]:
    """
    Attach each quotation to the image whose transcription contains it.

    Matching on words rather than on the slide number beside the
    quotation, because that number uses a convention this package does
    not state consistently and a mis-attributed quote would put one
    slide's words under another slide's name.

    Resolution outcomes:

    * exactly one transcription contains the fragment -- attached, with
      the declared number kept in the key so the two cannot be
      confused;
    * none contains it -- left unattached and reported, because a
      quotation no transcription contains was probably paraphrased, and
      "this claim rests on words the package did not transcribe" is
      something a reader is entitled to know;
    * several contain it -- left unattached and reported. A quotation
      repeated across slides belongs to whichever slide was meant, and
      nothing here can tell them apart.

    Returns the knowledge and one note per fragment that could not be
    attached. The notes are returned rather than logged because
    discarding them would turn "we could not verify this" into "this is
    verified", which is precisely the confusion this layer exists to
    prevent.
    """

    by_group: dict[str, list[GeminiImageRecord]] = {}

    for record in records:
        by_group.setdefault(record.post_id.upper(), []).append(record)

    resolved: dict[str, GeminiPostKnowledge] = {}

    all_notes: list[str] = []

    for group, entry in knowledge.items():
        members = by_group.get(group.upper(), [])

        activity = members[0].activity_id if members else ""

        notes: list[str] = []

        kept: dict[str, str] = {}

        for number, text in entry.slide_fragments.items():
            needle = _normalise(text)

            if not needle:
                continue

            hits = [
                record
                for record in members
                if record.readable_text
                and needle in _normalise(record.readable_text)
            ]

            if len(hits) == 1:
                kept[f"{hits[0].filename}#declared_slide={number}"] = redact_text(text)

            elif not hits:
                notes.append(
                    f"{group} slide {number}: the quotation does not "
                    "appear in any transcription for this group, so it "
                    "is unattached and probably paraphrased"
                )

            else:
                notes.append(
                    f"{group} slide {number}: the quotation appears in "
                    f"{len(hits)} transcriptions, so it cannot be "
                    "attributed to one of them"
                )

        all_notes.extend(notes)

        resolved[group] = entry.model_copy(
            update={
                "activity_id": entry.activity_id or activity,
                "slide_fragments": kept,
                "unresolved_group_ids": sorted(
                    {
                        note.split(" slide ")[0].strip()
                        for note in notes
                    }
                )
                if notes
                else [],
            }
        )

    return resolved, all_notes


def unmatched_groups(
    index: GeminiTopicIndex,
    knowledge: dict[str, GeminiPostKnowledge],
) -> list[str]:
    """
    Groups the package references that no image record supports.

    A topic index pointing at ``POST-317`` when the inventory holds no
    such group is a real gap: something was extracted from something
    this project cannot see. Recorded rather than dropped, because the
    silent version of this is a topic that looks better covered than
    it is.
    """

    known = {group.upper() for group in knowledge}

    unresolved: list[str] = []

    for groups in index.topics.values():
        for group in groups:
            if group.upper() not in known and group not in unresolved:
                unresolved.append(group)

    return sorted(unresolved)


def verdict_for(
    entry: GeminiPostKnowledge,
    records: list[GeminiImageRecord],
) -> MatchVerdict:
    """A post's own verdict, from whether its claims have any support."""

    if not records:
        return MatchVerdict.MISSING_FROM_ARCHIVE

    return MatchVerdict.MATCHED


__all__ = [
    "coverage",
    "parse_technical",
    "parse_topic_index",
    "resolve_fragments",
    "unmatched_groups",
    "verdict_for",
]