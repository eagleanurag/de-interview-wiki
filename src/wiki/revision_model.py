"""
Building the revision model: units, their questions, and their pages.

Sits above :mod:`src.wiki.curriculum`, which sits above the knowledge
base. The chain is unchanged and nothing is thrown away:

    KnowledgePost -> Curriculum (14 subjects, 154 subtopics)
                  -> Units (23 revision units, paginated)
                  -> pages

The curriculum's job is to put every question in exactly one bucket, and
it needs 154 buckets to do that. The units' job is to say what a person
revises, and that is a coarser question, so most buckets merge and a few
units split across pages.

The provenance chain is untouched: every question still carries
``post_ids``, every unit still knows which posts contributed to it, and
the knowledge base is not read differently. What changes is what is
*published*.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.wiki.curriculum import Curriculum, RevisionQuestion
from src.wiki.relevance import (
    EXCLUDED_SUBTOPICS,
    NON_REVISION_SUBJECTS,
    exclusion_reason,
    relocation_target,
)
from src.wiki.units import (
    QUESTIONS_PER_PAGE,
    UNITS,
    UnitSpec,
    page_count,
)


@dataclass(frozen=True)
class UnitPage:
    """One page of one unit."""

    #: 1-based.
    number: int
    total: int
    slug: str
    questions: tuple[RevisionQuestion, ...]

    @property
    def path(self) -> str:
        """
        The published path.

        Page one is always the unsuffixed path, whether or not the unit
        has more pages, because that makes it the unit's canonical
        address: the index links to ``revision/sql-joins.html`` and that
        file exists whether the unit is one page or five. Numbering every
        page instead put the canonical link on a file nothing generated,
        which is 2,799 broken links on the home page alone.
        """

        return f"revision/{self._file()}"

    def _file(self) -> str:
        if self.total == 1 or self.number == 1:
            return f"{self.slug}.html"

        return f"{self.slug}-{self.number}.html"

    @property
    def is_first(self) -> bool:
        return self.number == 1

    @property
    def is_last(self) -> bool:
        return self.number == self.total

    def sibling_path(self, number: int) -> str:
        """The path of page ``number`` of the same unit."""

        if self.total == 1 or number == 1:
            return f"revision/{self.slug}.html"

        return f"revision/{self.slug}-{number}.html"


@dataclass
class Unit:
    """One revision unit, resolved against the curriculum."""

    spec: UnitSpec
    questions: list[RevisionQuestion] = field(default_factory=list)
    concepts: dict[str, int] = field(default_factory=dict)
    source_posts: set[str] = field(default_factory=set)
    #: The subtopic labels this unit covers, for the section headings.
    sections: list[str] = field(default_factory=list)

    @property
    def slug(self) -> str:
        return self.spec.slug

    @property
    def title(self) -> str:
        return self.spec.title

    @property
    def group(self) -> str:
        return self.spec.group

    @property
    def summary(self) -> str:
        return self.spec.summary

    @property
    def question_count(self) -> int:
        return len(self.questions)

    @property
    def page_count(self) -> int:
        return page_count(len(self.questions), QUESTIONS_PER_PAGE)

    @property
    def post_count(self) -> int:
        return len(self.source_posts)

    def pages(self) -> list[UnitPage]:
        """
        Split the unit's questions into pages, in a fixed order.

        Questions are sorted by text, not by whatever order the
        curriculum happened to produce them, so a rebuild splits the same
        questions onto the same pages. Without that the page a question
        lands on would depend on dictionary iteration order, and a reader
        who bookmarked "page 3" would come back to something else.
        """

        ordered = sorted(self.questions, key=lambda q: q.text)

        total = page_count(len(ordered), QUESTIONS_PER_PAGE)

        pages: list[UnitPage] = []

        for index in range(total):
            chunk = ordered[
                index * QUESTIONS_PER_PAGE : (index + 1) * QUESTIONS_PER_PAGE
            ]

            pages.append(
                UnitPage(
                    number=index + 1,
                    total=total,
                    slug=self.slug,
                    questions=tuple(chunk),
                )
            )

        return pages

    def top_concepts(self, limit: int = 12) -> list[str]:
        return sorted(self.concepts, key=lambda c: (-self.concepts[c], c))[
            :limit
        ]


@dataclass
class Units:
    """Every revision unit, plus the filtering that produced them."""

    units: list[Unit]
    kept: int
    discarded: int
    discard_reasons: dict[str, int]
    #: Subtopic slugs that declared a unit but had no questions, so no
    #: unit was created for them. Reported rather than hidden.
    unmapped: tuple[str, ...] = ()

    @property
    def unit_count(self) -> int:
        return len(self.units)

    @property
    def page_count(self) -> int:
        return sum(unit.page_count for unit in self.units)

    @property
    def question_count(self) -> int:
        return sum(len(unit.questions) for unit in self.units)

    @property
    def groups(self) -> list[str]:
        seen: list[str] = []

        for unit in self.units:
            if unit.group not in seen:
                seen.append(unit.group)

        return seen

    def group_units(self, group: str) -> list[Unit]:
        return [unit for unit in self.units if unit.group == group]

    def unit(self, slug: str) -> Unit | None:
        for unit in self.units:
            if unit.slug == slug:
                return unit

        return None


def build_units(curriculum: Curriculum) -> Units:
    """
    Assemble the revision model from the curriculum.

    Three passes, in this order, because each depends on the last:

    1. **Filter.** Every curriculum question is kept or discarded, and
       the discard is recorded with a reason. Discarding first means a
       question that appears in two subtopics cannot be counted twice,
       and that the number reported as "kept" is the number published.
    2. **Assign.** Each surviving question goes to the unit that declares
       its subtopic. A subtopic no unit claims is reported in
       ``unmapped`` -- silently dropping it would make the page count
       look better than the coverage is.
    3. **Collect concepts.** Concepts are gathered per unit rather than
       per subtopic, so a unit's "key concepts" is a vocabulary for the
       whole page instead of one corner of it.
    """

    reasons: dict[str, int] = {}
    kept_questions: dict[tuple[str, str], list[RevisionQuestion]] = {}
    relocated: dict[str, list[RevisionQuestion]] = {}

    discarded = 0

    for question in curriculum.questions.values():
        excluded_filing = (
            question.major_slug in NON_REVISION_SUBJECTS
            or (question.major_slug, question.subtopic_slug)
            in EXCLUDED_SUBTOPICS
        )

        why = exclusion_reason(
            question.text,
            question.major_slug,
            question.subtopic_slug,
        )

        if why is not None and excluded_filing:
            # Two questions filed under the search are really questions
            # about the work, and they are moved rather than dropped --
            # see src.wiki.relevance.RELOCATIONS for why rescuing them by
            # keyword kept 33 pieces of advice.
            target = relocation_target(question.text)

            if target is not None:
                relocated.setdefault(target, []).append(question)
                continue

            discarded += 1
            reasons[why] = reasons.get(why, 0) + 1

            continue

        if why is not None:
            discarded += 1
            reasons[why] = reasons.get(why, 0) + 1
            continue

        kept_questions.setdefault(
            (question.major_slug, question.subtopic_slug), []
        ).append(question)

    units: list[Unit] = []
    claimed: set[tuple[str, str]] = set()

    for spec in UNITS:
        unit = Unit(spec=spec)

        unit.questions.extend(relocated.get(spec.slug, ()))

        for major_slug, subtopic_slug in spec.sources:
            claimed.add((major_slug, subtopic_slug))

            for question in kept_questions.get((major_slug, subtopic_slug), ()):
                unit.questions.append(question)

            node = curriculum.subtopic(major_slug, subtopic_slug)

            if node is None:
                continue

            label = node.subtopic.name

            if label not in unit.sections:
                unit.sections.append(label)

            for concept in node.concepts.values():
                unit.concepts[concept.label] = (
                    unit.concepts.get(concept.label, 0) + 1
                )

            unit.source_posts.update(node.post_ids)

        if unit.questions or unit.concepts:
            units.append(unit)

    unmapped = tuple(
        f"{major}/{subtopic}"
        for (major, subtopic) in sorted(kept_questions)
        if (major, subtopic) not in claimed
    )

    return Units(
        units=units,
        kept=sum(len(unit.questions) for unit in units),
        discarded=discarded,
        discard_reasons=reasons,
        unmapped=unmapped,
    )