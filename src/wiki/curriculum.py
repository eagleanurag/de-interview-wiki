"""
The reader-facing knowledge model: a curriculum, not an archive.

The canonical knowledge base behind this has 1,560 distinct topics,
3,294 subtopics, 3,106 concepts and 1,938 questions. That is the right
amount of evidence and the wrong shape for a reader: it describes how
the material was labelled rather than what a person revising for an
interview needs to walk through.

This module puts a small, fixed hierarchy in front of it without
discarding any of it. Eight subjects, sixty-odd subtopics, concepts
underneath, questions underneath that. Every label the corpus produced is
still reachable, filed under the subject and subtopic it belongs to, and
the residual -- labels nothing could be placed by -- is counted in
:attr:`Curriculum.diagnostics` rather than quietly dropped.

Three decisions worth stating.

**Questions are merged across posts before they are placed.** The same
interview question appears in several posts, and a reader revising
"broadcast joins" wants one page of them, not five copies. Merging uses
the aggregation layer's own question key, so the curriculum cannot
disagree with the knowledge base about what counts as the same question.

**A question is placed by its own words where possible.** "What is
ROW_NUMBER() and how does it differ from RANK()?" places itself under SQL
→ Window Functions without needing its post to agree. Only when the
question's own text is unrecognisable does it inherit its post's
placement. That ordering matters because the post is a weaker signal: a
post about partitioning can carry a question about joins.

**Answers are not invented or padded.** An answer is either one this
project's enricher wrote, a passage quoted from the slide a question was
read off, or nothing -- and "nothing" is stated as such on the page rather
than filled with something plausible. That is the whole point of the
distinction between a source excerpt and an answer.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from src.aggregation.consolidation import _question_key
from src.models import KnowledgePost
from src.wiki.taxonomy import (
    MAJORS,
    Major,
    Subtopic,
    classify,
    slug_of,
)


#: An answer shorter than this says nothing a reader can revise from, so
#: it is labelled rather than presented as an answer.
THIN_ANSWER_CHARS = 60

#: How much of a concept's supporting text to show. Concepts are labels
#: drawn from a slide, and some of them are a whole slide long.
CONCEPT_EXCERPT_CHARS = 220

ANSWER_AI = "ai_enriched"
ANSWER_EXCERPT = "source_excerpt"
ANSWER_INSUFFICIENT = "insufficient"


@dataclass(frozen=True)
class Answer:
    """
    What a reader is shown under a question, and where it came from.

    ``source`` is never hidden and never embellished. A quoted passage
    from a slide is not an answer to the question and is not presented as
    one, because a reader who cannot tell the difference will believe the
    slide said something it did not.
    """

    text: str
    source: str
    excerpt: str = ""

    @property
    def is_thin(self) -> bool:
        return len(self.text.strip()) < THIN_ANSWER_CHARS

    @property
    def label(self) -> str:
        return {
            ANSWER_AI: "Model-written answer",
            ANSWER_EXCERPT: "Excerpt from the slide this was read from",
            ANSWER_INSUFFICIENT: "No answer available",
        }[self.source]


@dataclass(frozen=True)
class RevisionQuestion:
    """One question, wherever it came from, filed once."""

    id: str
    text: str
    type: str
    difficulty: str
    answer: Answer
    post_ids: tuple[str, ...]
    concepts: tuple[str, ...]
    #: The contributing posts' own topic labels. Carried so the
    #: questions page can keep its topic filter, which predates the
    #: curriculum and is worth more than the grouping replaced it.
    topics: tuple[str, ...]
    major_slug: str
    subtopic_slug: str

    #: A human-readable description of each contributing post, in the
    #: same order as :attr:`post_ids`.
    #:
    #: This exists so a page can attribute a question without printing the
    #: identifier behind it. ``urn-li-archive-c7812291ac93dd56`` is an
    #: internal key: it cannot be typed back to anything and it reads as a
    #: defect on a page written for people. The identifiers themselves are
    #: kept -- :attr:`post_ids` still holds every one, and nothing here is
    #: derived from stripping it -- so provenance is strengthened rather
    #: than lost.
    #:
    #: Defaulted, so the tests that construct a question directly are
    #: unaffected and an absent description degrades to a neutral noun
    #: rather than to an identifier.
    source_labels: tuple[str, ...] = ()

    @property
    def slug(self) -> str:
        return f"{slug_of(self.text)[:60]}-{_question_key(self.text)[:8]}"

    @property
    def source_count(self) -> int:
        return len(self.post_ids)

    @property
    def sources(self) -> tuple[str, ...]:
        """The descriptions, padded if a caller supplied none."""

        if self.source_labels:
            return self.source_labels

        return ("source post",) * self.source_count


@dataclass
class ConceptNode:
    """One concept, and the questions that test it."""

    label: str
    slug: str
    post_ids: list[str] = field(default_factory=list)
    question_ids: list[str] = field(default_factory=list)


@dataclass
class SubtopicNode:
    """One revisable subject area."""

    major: Major
    subtopic: Subtopic
    concepts: dict[str, ConceptNode] = field(default_factory=dict)
    question_ids: list[str] = field(default_factory=list)
    post_ids: list[str] = field(default_factory=list)

    @property
    def slug(self) -> str:
        return self.subtopic.slug

    @property
    def question_count(self) -> int:
        return len(self.question_ids)

    @property
    def post_count(self) -> int:
        return len(self.post_ids)


@dataclass
class MajorNode:
    """One subject, and its subtopics."""

    major: Major
    subtopics: list[SubtopicNode] = field(default_factory=list)

    @property
    def slug(self) -> str:
        return self.major.slug

    @property
    def question_count(self) -> int:
        return sum(node.question_count for node in self.subtopics)

    @property
    def post_count(self) -> int:
        return len(
            {post for node in self.subtopics for post in node.post_ids}
        )

    @property
    def concept_count(self) -> int:
        return len(
            {
                concept.slug
                for node in self.subtopics
                for concept in node.concepts.values()
            }
        )


@dataclass
class Curriculum:
    """Everything the reader-facing pages are built from."""

    majors: list[MajorNode]
    questions: dict[str, RevisionQuestion]
    diagnostics: dict[str, int] = field(default_factory=dict)

    @property
    def major_count(self) -> int:
        return len(self.majors)

    @property
    def subtopic_count(self) -> int:
        return sum(len(node.subtopics) for node in self.majors)

    @property
    def question_count(self) -> int:
        return len(self.questions)

    def major(self, slug: str) -> MajorNode | None:
        for node in self.majors:
            if node.slug == slug:
                return node

        return None

    def subtopic(
        self, major_slug: str, subtopic_slug: str
    ) -> SubtopicNode | None:
        node = self.major(major_slug)

        if node is None:
            return None

        for subtopic in node.subtopics:
            if subtopic.slug == subtopic_slug:
                return subtopic

        return None

    def questions_in(
        self, major_slug: str, subtopic_slug: str
    ) -> list[RevisionQuestion]:
        node = self.subtopic(major_slug, subtopic_slug)

        if node is None:
            return []

        return [
            self.questions[qid]
            for qid in node.question_ids
            if qid in self.questions
        ]

    def related(
        self, node: SubtopicNode, limit: int = 6
    ) -> list[SubtopicNode]:
        """
        Neighbouring subtopics, for a short "Related" list.

        Simple hyperlinks, deliberately. A graph visualisation over six
        thousand concepts is a data tool rather than a revision aid, and
        drawing one is the archive-shaped instinct this model exists to
        remove.
        """

        subject = self.major(node.major.slug)

        siblings = [
            other
            for other in (subject.subtopics if subject else [])
            if other.slug != node.slug
        ]

        if len(siblings) >= limit:
            return siblings[:limit]

        # Not enough within the subject to be worth showing; borrow from
        # the subjects with the most questions, largest first.
        for major in sorted(
            self.majors, key=lambda item: (-item.question_count, item.slug)
        ):
            if major.slug == node.major.slug:
                continue

            for other in major.subtopics:
                if len(siblings) >= limit:
                    return siblings

                if other.question_count:
                    siblings.append(other)

        return siblings


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _answer_for(question, post: KnowledgePost) -> Answer:
    """
    The answer for one question, without inventing one.

    An answer this project's enricher wrote is preferred. Failing that,
    the Gemini bridge may have attached a passage quoted from the slide
    the question was read from; that is shown and labelled as a passage,
    not promoted to an answer. Failing both, the page says there is
    nothing, which is a more useful thing to read than a fabrication.
    """

    del post

    text = _collapse(getattr(question, "answer", "") or "")

    if not text:
        return Answer(text="", source=ANSWER_INSUFFICIENT)

    source = getattr(question, "answer_source", "") or ANSWER_AI

    return Answer(text=text, source=source)


def _post_placement(post: KnowledgePost) -> tuple[Major, Subtopic]:
    """
    Where a post's material belongs, by its own labels.

    The most frequent placement across the post's topics, subtopics and
    concepts wins. Frequency rather than precedence because a post
    labelled with three subtopics is usually about the one it mentions
    most, and "first in a list" is an accident of generation order.
    """

    # A post that failed enrichment has no analysis at all, and
    # enrichment is allowed to fail. Reading through it unguarded turns
    # "this post was never analysed" into a crash while rendering, which
    # loses the post from the site entirely rather than filing it thinly.
    analysis = post.ai_analysis

    labels = []

    if analysis is not None:
        labels = list(analysis.topics) + list(analysis.subtopics)
        labels += list(analysis.concepts)

    placement = getattr(post.classification, "primary_topic", "")

    if placement:
        labels.insert(0, str(placement))

    tally: dict[tuple[str, str], int] = {}

    for label in labels:
        found = classify(label)

        key = (found.major.slug, found.subtopic.slug)

        tally[key] = tally.get(key, 0) + 1

    if not tally:
        return classify("").major, classify("").subtopic

    # Deterministic: highest count, then declared order of the subjects.
    ranked = sorted(
        tally.items(),
        key=lambda item: (-item[1], [m.slug for m in MAJORS].index(item[0][0])),
    )

    major_slug, subtopic_slug = ranked[0][0]

    for major in MAJORS:
        if major.slug != major_slug:
            continue

        for subtopic in major.subtopics:
            if subtopic.slug == subtopic_slug:
                return major, subtopic

    fallback = classify("")

    return fallback.major, fallback.subtopic


def _place_question(
    text: str, post: KnowledgePost
) -> tuple[Major, Subtopic, bool]:
    """
    Where a question belongs, preferring its own words.

    A question's text is a sharper signal than its post's labels: the
    post may be about partitioning and carry a question about joins, and
    the question is the thing the reader is revising.
    """

    own = classify(text)

    if not own.is_fallback:
        return own.major, own.subtopic, True

    major, subtopic = _post_placement(post)

    return major, subtopic, False


def _concepts_for(post: KnowledgePost) -> list[str]:
    return [
        label
        for label in post.ai_analysis.concepts
        if isinstance(label, str) and label.strip()
    ]


#: How a platform is written, for a string a reader will see.
#:
#: The model stores the lower-case machine value, and this is the only
#: place that turns it into prose.
_PLATFORM_NAMES = {
    "linkedin": "LinkedIn",
}


def _source_label(post: KnowledgePost) -> str:
    """
    Name the kind of source a post is, in words a reader can act on.

    Built from what the post already declares, never from its identifier.
    ``post.source.platform`` is the site the post came from, and
    ``post.saved_item`` records that a person exported it from a saved
    list rather than an automated run collecting it -- a distinction
    ``SourceInfo.capture_method`` documents as real and not
    interchangeable.

    What it deliberately does not include: the post identifier, the
    LinkedIn URL, the saved item's own id, and any capture date. Those
    stay in the model. A page that attributes a question does not need
    them, and printing them turns provenance into noise.
    """

    platform = (post.source.platform or "").strip()

    # ``SourceInfo.platform`` stores the lower-case machine value
    # ("linkedin"). Printing it capitalised gives "Linkedin", which is
    # not how the company writes its name, and this string is read by a
    # person. Anything unrecognised falls back to plain capitalisation
    # rather than being guessed at.
    name = _PLATFORM_NAMES.get(platform.lower(), platform.capitalize())

    if getattr(post, "saved_item", None) is not None:
        return f"{name} post, saved from a list"

    return f"{name} post"


def build_curriculum(posts: list[KnowledgePost]) -> Curriculum:
    """
    Assemble the reader-facing model from the enriched posts.

    Deterministic throughout: labels are counted, not iterated for
    side effects, and every collection is sorted before it is exposed. Two
    runs over the same posts produce the same curriculum, which is what
    makes the generated site reproducible.
    """

    #: Keyed by :attr:`RevisionQuestion.id`, because that is what the
    #: subtopic nodes reference. Keying by the dedup key instead looked
    #: correct and rendered every subtopic page with zero questions: the
    #: ids were present on the nodes but absent from this mapping.
    questions: dict[str, RevisionQuestion] = {}

    #: Question key -> merged record, so the same question from five posts
    #: becomes one entry carrying all five sources.
    merged: dict[str, dict] = {}

    merged_order: list[str] = []

    self_placed = 0
    inherited = 0

    # Read up front: the merge loop needs the posts' own topics for
    # every post, not only the ones seen so far.
    post_topics = {
        post.id: tuple(post.ai_analysis.topics) for post in posts
    }

    for post in posts:
        for question in post.interview_questions:
            key = _question_key(question.question)

            answer = _answer_for(question, post)

            record = merged.get(key)

            if record is None:
                major, subtopic, own = _place_question(
                    question.question, post
                )

                if own:
                    self_placed += 1
                else:
                    inherited += 1

                merged[key] = {
                    "text": question.question,
                    "type": question.type,
                    "difficulty": question.difficulty,
                    "answer": answer,
                    "post_ids": [post.id],
                    "concepts": set(),
                    # Seeded here as well as on the merge path. Filling
                    # it only when a duplicate turned up left every
                    # first-seen question with no topics, which emptied
                    # the topic filter and the data-topics attribute
                    # that the questions page has always carried.
                    "topics": set(post_topics.get(post.id, ())),
                    "major": major,
                    "subtopic": subtopic,
                }

                merged_order.append(key)

                continue

            # Same question seen again. Keep the more substantial answer
            # and every source.
            if (
                answer.text
                and len(answer.text) > len(record["answer"].text)
            ):
                record["answer"] = answer

            if post.id not in record["post_ids"]:
                record["post_ids"].append(post.id)

            record["topics"].update(post_topics.get(post.id, ()))

    # Concepts are a property of the post, so they are attached once per
    # post to every question that post contributed. Cheap, and it keeps a
    # concept visible from any question that touches it.
    post_concepts = {
        post.id: _concepts_for(post) for post in posts
    }

    posts_by_id = {post.id: post for post in posts}

    for key in merged_order:
        record = merged[key]

        concepts: set[str] = set()

        for post_id in record["post_ids"]:
            concepts.update(post_concepts.get(post_id, ()))

        record["concepts"] = sorted(concepts)

        question_id = f"q-{_slug_of_key(key)}"

        questions[question_id] = RevisionQuestion(
            id=question_id,
            text=_collapse(record["text"]),
            type=record["type"],
            difficulty=record["difficulty"],
            answer=record["answer"],
            post_ids=tuple(sorted(record["post_ids"])),
            concepts=tuple(record["concepts"]),
            topics=tuple(sorted(record["topics"])),
            major_slug=record["major"].slug,
            subtopic_slug=record["subtopic"].slug,
            source_labels=tuple(
                _source_label(posts_by_id[post_id])
                for post_id in sorted(record["post_ids"])
                if post_id in posts_by_id
            ),
        )

    # Every subject and subtopic gets a node up front, so that placing a
    # concept or a question is a lookup rather than a decision, and so
    # that "no node" can only mean the taxonomy itself is inconsistent.
    nodes: dict[tuple[str, str], SubtopicNode] = {}

    for major in MAJORS:
        for subtopic in major.subtopics:
            nodes[(major.slug, subtopic.slug)] = SubtopicNode(
                major=major, subtopic=subtopic
            )

    # Concepts, filed by their own classification.
    concepts_placed = 0

    for post in posts:
        for label in post_concepts.get(post.id, ()):
            found = classify(label)

            concepts_placed += 1

            node = nodes.get((found.major.slug, found.subtopic.slug))

            if node is None:
                continue

            concept = node.concepts.get(slug_of(label))

            if concept is None:
                concept = ConceptNode(label=label, slug=slug_of(label))

                node.concepts[concept.slug] = concept

            if post.id not in concept.post_ids:
                concept.post_ids.append(post.id)

    # Questions, into their subject and subtopic. Sorted first so the order
    # questions appear on a page is a function of their content, not of
    # whichever post happened to be aggregated first.
    for question in sorted(
        questions.values(),
        key=lambda item: (
            item.major_slug,
            item.subtopic_slug,
            item.text,
        ),
    ):
        node = nodes.get(
            (question.major_slug, question.subtopic_slug)
        )

        if node is None:
            continue

        node.question_ids.append(question.id)

        for post_id in question.post_ids:
            if post_id not in node.post_ids:
                node.post_ids.append(post_id)

    # Only subjects and subtopics with something in them are exposed. An
    # empty subtopic in a revision index is a dead end for a reader.
    majors: list[MajorNode] = []

    for major in MAJORS:
        major_node = MajorNode(major=major)

        for subtopic in major.subtopics:
            candidate = nodes[(major.slug, subtopic.slug)]

            if not (candidate.question_ids or candidate.concepts):
                continue

            major_node.subtopics.append(candidate)

        if major_node.subtopics:
            majors.append(major_node)

    raw_total = sum(
        len(post.interview_questions) for post in posts
    )

    return Curriculum(
        majors=majors,
        questions=questions,
        diagnostics={
            "posts": len(posts),
            "question_slots_before_merge": raw_total,
            "questions_after_merge": len(questions),
            "merged_away": raw_total - len(questions),
            "placed_by_own_words": self_placed,
            "placed_by_parent_post": inherited,
            "concepts": concepts_placed,
        },
    )


#: Longest title fragment taken from a summary when nothing better
#: exists. A title is a label, not a sentence, and a sentence that runs
#: to three lines above the content stops being one.
TITLE_FALLBACK_CHARS = 64

#: Fewest words a summary fragment may have and still be a title.
#: Below this the cut lands on an article or a conjunction, and a
#: title of one word is worse than nothing. Two is enough for the
#: short, clean sentences the unlabelled posts tend to have.
MIN_TITLE_WORDS = 2


def placement_for(post: KnowledgePost) -> tuple[Major, Subtopic]:
    """
    Where one post belongs, by its own labels.

    Public wrapper over the placement the builder already computes, so a
    post page and the subtopic page that lists it cannot disagree about
    which subject it is under. Two answers to one question, computed two
    ways, is how a page ends up filed under SQL while the navigation
    says Spark.
    """

    return _post_placement(post)


def knowledge_title(post: KnowledgePost) -> str:
    """
    A human-readable name for what a post turned out to be about.

    Never the identifier. ``urn-li-saved-ffccf4f7771b97bd`` is what the
    archive calls it; "SQL -- Hotel Booking Analytics" is what it is, and
    a reader revising for an interview needs the second one.

    Built from the classification's own primary topic where there is one,
    because that is the model's considered answer to "what is this".
    The subject is prefixed from the taxonomy, so the title reads as a
    position in the hierarchy rather than a label floating on its own.

    Two details are load-bearing. The subject name is stripped from the
    candidate, because "SQL" and "Hotel booking analysis with SQL" makes
    a title that says SQL twice. And the result is rejected if it looks
    like an identifier, because a fallback that can emit
    ``urn-li-...`` is not a fallback.
    """

    analysis = post.ai_analysis

    labelled = bool(
        (post.classification and post.classification.primary_topic)
        or (analysis and (analysis.topics or analysis.subtopics or analysis.concepts))
    )

    candidate = _title_candidate(post)

    subject = placement_for(post)[0].name

    if candidate:
        # Belt and braces. _looks_like_identifier has already been
        # applied to every candidate, and this is the last place an
        # identifier could reach a reader as a title.
        if _looks_like_identifier(candidate) or _looks_like_identifier(subject):
            return subject

        if not labelled:
            # No subject is claimed. Measured: 86 of 490 posts carry no
            # topic, subtopic, concept or primary topic at all -- they
            # are job announcements, personal updates and hiring posts.
            # The classifier files anything unrecognised under the
            # broadest subject, so prefixing one here produced "SQL" as
            # the title of a Goldman Sachs offer announcement. A title
            # that asserts a subject the post never mentioned is worse
            # than no title, because a reader revising SQL would find it.
            return candidate

        return f"{subject} \u2014 {candidate}"

    return subject


def _title_candidate(post: KnowledgePost) -> str:
    """The part of the title that is about the content, or nothing."""

    analysis = post.ai_analysis

    classification = post.classification

    raw = (
        (classification.primary_topic if classification else "")
        or _most_common(analysis.topics if analysis else ())
        or _most_common(analysis.subtopics if analysis else ())
        or _most_common(analysis.concepts if analysis else ())
    )

    # Every subject name, not only this post's own. "Hotel booking
    # analysis with SQL" filed under Data Engineering still says SQL
    # twice once the subject is prefixed, and the reader gains nothing
    # from the second mention.
    text = _strip_subjects(_collapse(raw))

    if text and not _looks_like_identifier(text):
        return text

    # Nothing usable was on the post -- its primary topic was, or reduced
    # to, the subject it is filed under. A concept is the next best
    # label, because a concept is a thing rather than a field value.
    concept = _strip_subjects(
        _collapse(_most_common(analysis.concepts if analysis else ()))
    )

    if concept and not _looks_like_identifier(concept):
        return concept

    # Last resort: a phrase cut from the summary. "SQL" on its own tells
    # a reader nothing about which SQL, so this is better -- but only if
    # what comes back is a phrase.
    from_summary = _from_summary(post)

    if from_summary:
        return from_summary

    return ""


def _strip_subjects(text: str) -> str:
    """
    Remove subject names from a phrase that also carries one.

    All of the taxonomy's, not just the post's own, because the phrases
    mix them freely: "Delta Lake query performance optimization" under
    Databricks, "Hotel booking analysis with SQL" under Data Engineering.
    Both already say their subject in the prefix.

    Each shape that occurs is handled rather than the bare name alone,
    because deleting the whole phrase would leave nothing. A dangling
    conjunction is removed afterwards -- stripping the leading word of
    "and Spark interview preparation" otherwise leaves the "and".
    """

    for word in _all_subject_words():
        escaped = re.escape(word)

        text = re.sub(
            rf"(?i)\b(?:with|using|in|for|on|and)\s+{escaped}\b",
            "",
            text,
        )

        text = re.sub(rf"(?i)^{escaped}\b[\s,:;-]*", "", text)

        text = re.sub(rf"(?i)\b{escaped}$", "", text)

    text = _collapse(text.strip(" \u2014 -:,."))

    # A leading conjunction left behind by the removal above.
    text = re.sub(r"(?i)^(?:and|or|with|for|in|on)\s+", "", text).strip()

    return text


def _all_subject_words() -> list[str]:
    """
    Every subject name and each of its significant words, longest first.

    Ordered by length so a longer alternative is removed before a
    shorter one can match inside it, and deduplicated so "Spark and
    PySpark" does not contribute "Spark" twice.
    """

    words: set[str] = set()

    for major in MAJORS:
        words.add(major.name)

        words.update(
            part for part in major.name.split() if part.lower() != "and"
        )

    return sorted(words, key=len, reverse=True)


def _most_common(labels: list[str] | tuple[str, ...]) -> str:
    """The most frequent label, ties broken alphabetically."""

    cleaned = [
        _collapse(label)
        for label in labels
        if isinstance(label, str) and label.strip()
    ]

    if not cleaned:
        return ""

    counts: dict[str, int] = {}

    for label in cleaned:
        counts[label] = counts.get(label, 0) + 1

    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _from_summary(post: KnowledgePost) -> str:
    """
    A label cut from the summary, or nothing.

    The awkwardness is all in the refusal. Model summaries here are
    multi-sentence paragraphs, so the first sentence is long, and cutting
    a long sentence to a title length lands mid-clause often enough to
    matter: with a blunt character limit the results included "SQL --
    The", "Databricks -- An" and "Spark and PySpark -- This". A title of
    one article is worse than no title, so a fragment shorter than
    :data:`MIN_TITLE_WORDS` words is discarded and the caller falls back
    to the subject alone.
    """

    analysis = post.ai_analysis

    summary = _collapse(analysis.summary if analysis else "")

    if not summary:
        return ""

    first = re.split(r"(?<=[.!?])\s", summary)[0].strip()

    if len(first) <= TITLE_FALLBACK_CHARS:
        candidate = first.rstrip(".!?")
    else:
        # Drop the last, possibly partial, word -- and only that.
        #
        # The earlier version was ``re.sub(r"[\s,;:].*$", "", chunk)``,
        # which removes from the *first* space onward and so kept exactly
        # one word. Every unlabelled post in the corpus fell back to its
        # summary and every one of them came back empty, which is why 86
        # of them were titled with the bare subject name.
        candidate = re.sub(r"\s+\S*$", "", first[:TITLE_FALLBACK_CHARS])

        # A trailing conjunction or preposition left by the cut.
        candidate = re.sub(
            r"(?i)\s+(?:and|or|with|for|in|to|of|from|by|at|on|as"
            r"|that|which|where|when|while|but|is|are|was|were)$",
            "",
            candidate,
        ).strip(" ,;:-")

        # And a leading run of framing words. Summaries here open
        # "The post announces...", "This is a hiring announcement...",
        # and stripping only the article leaves a title starting "post
        # announces", which reads as a sentence missing its subject.
        trimmed = candidate

        for _ in range(3):
            before = trimmed

            trimmed = re.sub(
                r"(?i)^(?:the|a|an|this|that|it|is|was|are|were|post|posts|"
                r"article|message|caption)\s+",
                "",
                trimmed,
            ).strip()

            if trimmed == before:
                break

        candidate = trimmed or candidate

        # Capitalised, because it is now a label rather than a clause.
        candidate = candidate[:1].upper() + candidate[1:]

    if len(candidate.split()) < MIN_TITLE_WORDS:
        return ""

    return candidate


def _looks_like_identifier(text: str) -> bool:
    """
    Whether a string is an archive identifier rather than a title.

    ``urn-li-saved-...``, ``activity_7263033731471351808_slide_0.jpg``,
    ``POST-029``. A fallback that can emit one of those is not a
    fallback, and this is the check that keeps that true.
    """

    if not text:
        return True

    if re.match(r"(?i)^urn[-_]", text):
        return True

    if re.match(r"(?i)^activity[_-]\d", text):
        return True

    if re.match(r"(?i)^post[-_]\d+$", text):
        return True

    # A long run of digits and separators with no spaces is a key, not a
    # phrase.
    return bool(re.fullmatch(r"[\w.:-]{12,}", text) and " " not in text)



def _slug_of_key(key: str) -> str:
    """
    A short, readable, collision-free identifier for a question key.

    The truncation alone is not an identifier. Eight questions were lost
    when two different keys produced the same 60-character prefix and
    the second overwrote the first, which is invisible unless the total
    is counted. The digest suffix makes the id a function of the whole
    key while the prefix stays readable in a URL.
    """

    stem = re.sub(r"[^a-z0-9]+", "-", key).strip("-")[:48] or "question"

    return f"{stem}-{hashlib.sha1(key.encode('utf-8')).hexdigest()[:8]}"


__all__ = [
    "ANSWER_AI",
    "ANSWER_EXCERPT",
    "ANSWER_INSUFFICIENT",
    "Answer",
    "ConceptNode",
    "Curriculum",
    "MajorNode",
    "RevisionQuestion",
    "SubtopicNode",
    "MIN_TITLE_WORDS",
    "THIN_ANSWER_CHARS",
    "TITLE_FALLBACK_CHARS",
    "build_curriculum",
    "knowledge_title",
    "placement_for",
]