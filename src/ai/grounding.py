"""
Is what the model wrote actually supported by the post?

The prompt already tells the model not to invent. A prompt is a request,
not a guarantee, and the failure it invites is specific: a post about
Delta Lake comes back with a summary, a concept list and three interview
questions, and somewhere in them is Kafka. A reader cannot tell that
apart from something the author actually said, because by that point it
is the same text in the same font.

So the response is checked against the post it came from, and anything
naming a technology the source never mentions is dropped. Dropped rather
than corrected, because there is no honest way to rewrite a question into
one the source supports without writing the question ourselves, and a
knowledge base that quietly contains questions we made up is worse than
one that contains fewer questions.

What is checked is narrow on purpose. A concept or a question naming a
*recognised* technology absent from the source is a fabrication that can
be shown. Anything else is left alone: deciding that a concept is
unsupported means judging meaning, and a check that guesses at meaning
will eventually throw away something true.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


#: Phrases that use a technology's name to mean something else.
#:
#: "Snowflake" is a warehouse and also the name of a dimensional
#: modelling technique. A post saying "snowflake schema" is talking about
#: star schemas, and treating that as a mention of the product would let
#: a question about the warehouse through on the strength of a word that
#: was never about it. The longer phrase is taken out before the bare
#: name is looked for, so each is judged on what it meant.
NAME_SHADOWED: dict[str, tuple[str, ...]] = {
    "Snowflake": ("snowflake schema",),
    "SQL": ("nosql",),
    "Azure": ("azure devops", "azure active directory"),
    "AWS": ("aws lambda",),
}


def _normalise(text: str) -> str:
    """Reduce text to comparable words."""
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _padded(text: str) -> str:
    """Normalised text with a space at each end, for whole-phrase tests."""
    return f" {_normalise(text)} "


@dataclass
class GroundingReport:
    """
    What grounding removed, and why.

    Kept so a post can say what was dropped. A concept that silently
    vanishes looks like a bug, and a reader comparing the source to the
    page deserves to know that the page is the one that was cut.
    """

    concepts: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not (self.concepts or self.questions or self.topics)

    @property
    def dropped(self) -> int:
        return len(self.concepts) + len(self.questions) + len(self.topics)

    def summary(self) -> str:
        """One line, for a capture note or a log."""
        if self.clean:
            return "every generated item is supported by the source"

        parts = []

        for label, items in (
            ("topic", self.topics),
            ("concept", self.concepts),
            ("question", self.questions),
        ):
            if items:
                parts.append(f"{len(items)} {label}")

        return (
            "dropped "
            + ", ".join(parts)
            + " naming a technology the source does not mention"
        )

    def as_dict(self) -> dict:
        return {
            "topics": list(self.topics),
            "concepts": list(self.concepts),
            "questions": list(self.questions),
        }


class Grounder:
    """
    Decides whether generated material is supported by the source.

    Built once from the source text, because the check runs against every
    concept and every question and re-deriving the recognised technologies
    each time would make enrichment quadratic in the length of the post.
    """

    def __init__(self, source_text: str) -> None:
        from src.aggregation.consolidation import TECHNOLOGIES

        self._technologies = {
            technology: tuple(
                phrase
                for phrase in (_normalise(m) for m in markers)
                if phrase
            )
            for technology, markers in TECHNOLOGIES.items()
        }

        self._shadows = {
            technology: tuple(
                phrase
                for phrase in (
                    _normalise(item) for item in NAME_SHADOWED.get(
                        technology, ()
                    )
                )
                if phrase
            )
            for technology in self._technologies
        }

        #: The technologies the source itself talks about.
        self.supported: set[str] = set(
            self.technologies_in(source_text)
        )

    def technologies_in(self, text: str) -> list[str]:
        """
        Every known technology a piece of text names.

        The one place recognition happens, so a mention in the source and
        a mention in a generated question are judged by the same rule.
        They have to be: a check that recognised technologies differently
        depending on which side of the comparison it sat on would reject
        questions about things the post plainly covers.

        A technology counts when any one of its phrases appears. Requiring
        all of them would fail on a post that says "Databricks" and never
        says "Photon", which is very nearly every post about Databricks.
        """
        haystack = _padded(text)

        found: list[str] = []

        for technology, phrases in self._technologies.items():
            if not phrases:
                continue

            body = haystack

            for shadow in self._shadows.get(technology, ()):
                body = body.replace(f" {shadow} ", " ")

            for phrase in phrases:
                if f" {phrase} " in body:
                    found.append(technology)
                    break

        return found

    def named(self, text: str) -> list[str]:
        """Which known technologies a piece of text names."""
        return self.technologies_in(text)

    def unsupported(self, text: str) -> list[str]:
        """
        Technologies named in ``text`` that the source does not mention.

        Phrases are matched whole, so "spark" is not found inside
        "sparkling" and "sql" is not found inside "postgresql". A false
        positive there would delete a perfectly good question about a
        subject the source really does discuss.
        """
        return [
            technology
            for technology in self.technologies_in(text)
            if technology not in self.supported
        ]

    def keep(self, text: str) -> bool:
        """Whether a generated item survives the check."""
        return not self.unsupported(text)


def check(
    source_text: str,
    *,
    topics: list[str],
    concepts: list[str],
    questions: list[str],
) -> tuple[list[str], list[str], list[str], GroundingReport]:
    """
    Remove generated items that name an unsupported technology.

    Returns the surviving topics, concepts and questions along with the
    report of what was removed, so the caller can record the difference
    rather than presenting a silently shortened list.
    """
    grounder = Grounder(source_text)

    report = GroundingReport()

    def keep_all(values: list[str], bucket: list[str]) -> list[str]:
        kept: list[str] = []

        for value in values:
            if grounder.keep(value):
                kept.append(value)
            else:
                bucket.append(value)

        return kept

    kept_topics = keep_all(list(topics), report.topics)
    kept_concepts = keep_all(list(concepts), report.concepts)
    kept_questions = keep_all(list(questions), report.questions)

    return kept_topics, kept_concepts, kept_questions, report


__all__ = [
    "NAME_SHADOWED",
    "Grounder",
    "GroundingReport",
    "check",
]
