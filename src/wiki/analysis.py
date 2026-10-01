"""
Derived, ordered views over the canonical knowledge base.

Pages render from this model rather than recomputing counts, topic
grouping and question lists per page. That keeps every page consistent
and makes the ordering rules explicit and testable in one place.

Nothing here invents schema fields. Every value is derived from fields
that exist in `src.models`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.aggregation.consolidation import (
    detect_technologies,
    post_topics,
)
from src.models import KnowledgePost
from src.wiki.canonical import CanonicalKnowledgeBase
from src.wiki.naming import (
    SlugRegistry,
    concept_page,
    post_page,
    safe_id,
    technology_page,
    topic_page,
)


DIFFICULTIES = ("easy", "medium", "hard")

TOPIC_KIND = "topic"
SUBTOPIC_KIND = "subtopic"


@dataclass(frozen=True)
class TopicEntry:
    """One topic or subtopic, with the posts that mention it."""

    label: str
    slug: str
    kind: str
    post_slugs: tuple[str, ...]
    concepts: tuple[str, ...]
    question_count: int

    @property
    def post_count(self) -> int:
        return len(self.post_slugs)

    @property
    def page(self) -> str:
        return topic_page(self.slug)

    @property
    def sort_key(self) -> tuple[str, int, str]:
        return (self.label.casefold(), self.post_count, self.label)


@dataclass(frozen=True)
class ConceptEntry:
    """One concept, with the posts that mention it."""

    label: str
    slug: str
    post_slugs: tuple[str, ...]
    topics: tuple[str, ...]
    technologies: tuple[str, ...]

    @property
    def post_count(self) -> int:
        return len(self.post_slugs)

    @property
    def page(self) -> str:
        return concept_page(self.slug)

    @property
    def sort_key(self) -> tuple[str, int, str]:
        return (self.label.casefold(), self.post_count, self.label)


@dataclass(frozen=True)
class TechnologyEntry:
    """One recognised technology and the posts that use it."""

    label: str
    slug: str
    post_slugs: tuple[str, ...]
    topics: tuple[str, ...]
    question_count: int

    @property
    def post_count(self) -> int:
        return len(self.post_slugs)

    @property
    def page(self) -> str:
        return technology_page(self.slug)

    @property
    def sort_key(self) -> tuple[str, int, str]:
        return (self.label.casefold(), self.post_count, self.label)


@dataclass(frozen=True)
class QuestionEntry:
    """One interview question, kept attached to its source post."""

    key: str
    post_id: str
    post_slug: str
    question: str
    answer: str
    question_type: str
    difficulty: str
    topics: tuple[str, ...]

    @property
    def page(self) -> str:
        return post_page(self.post_slug)

    @property
    def difficulty_rank(self) -> int:
        if self.difficulty in DIFFICULTIES:
            return DIFFICULTIES.index(self.difficulty)

        return len(DIFFICULTIES)

    @property
    def sort_key(self) -> tuple[str, int, str, str]:
        return (
            self.question_type.casefold(),
            self.difficulty_rank,
            self.post_id.casefold(),
            self.question,
        )


@dataclass(frozen=True)
class SiteModel:
    """Everything the pages need, fully ordered and counted."""

    posts: tuple[KnowledgePost, ...]
    post_slugs: tuple[str, ...]
    topics: tuple[TopicEntry, ...]
    topic_slugs: dict[str, str]
    questions: tuple[QuestionEntry, ...]
    concept_entries: tuple[ConceptEntry, ...]
    technology_entries: tuple[TechnologyEntry, ...]
    concepts: tuple[str, ...]
    generated_at: str | None
    stats: dict[str, int] = field(default_factory=dict)

    @property
    def post_count(self) -> int:
        return len(self.posts)

    @property
    def topic_count(self) -> int:
        return len(self.topics)

    @property
    def concept_count(self) -> int:
        return len(self.concepts)

    @property
    def concept_page_count(self) -> int:
        return len(self.concept_entries)

    @property
    def technology_count(self) -> int:
        return len(self.technology_entries)

    @property
    def question_count(self) -> int:
        return len(self.questions)

    @property
    def is_empty(self) -> bool:
        return not self.posts

    def recent_pairs(
        self,
        limit: int,
    ) -> tuple[tuple[KnowledgePost, str], ...]:
        """
        Most recently captured posts first, as (post, slug) pairs.

        Ordering is by capture timestamp descending, with the post ID
        breaking every tie, so the result never depends on the order
        the aggregator happened to emit. Posts with no capture
        timestamp sort last.
        """

        def recency_key(
            index: int,
        ) -> tuple[str, str]:
            post = self.posts[index]
            captured = post.source.captured_at

            # Timestamps may be naive or offset-aware, which cannot be
            # compared with each other, so they are compared as text.
            stamp = (
                "9999" if captured is None else captured.isoformat()
            )

            return (stamp, post.id.casefold())

        order = sorted(
            range(len(self.posts)),
            key=recency_key,
            reverse=True,
        )

        return tuple(
            (self.posts[index], self.post_slugs[index])
            for index in order[:limit]
        )

    def topics_of_kind(self, kind: str) -> tuple[TopicEntry, ...]:
        return tuple(
            entry for entry in self.topics if entry.kind == kind
        )

    def difficulty_counts(self) -> dict[str, int]:
        counts = {value: 0 for value in DIFFICULTIES}

        for question in self.questions:
            counts[question.difficulty] = (
                counts.get(question.difficulty, 0) + 1
            )

        return counts

    def question_type_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}

        for question in self.questions:
            counts[question.question_type] = (
                counts.get(question.question_type, 0) + 1
            )

        return dict(sorted(counts.items()))

    def concept_order_is_sorted(self) -> bool:
        return list(self.concepts) == sorted(
            self.concepts, key=str.casefold
        )

    def summary(self) -> str:
        """One-line description used in page metadata."""

        if self.is_empty:
            return (
                "Data Engineering interview knowledge base. "
                "No posts have been ingested yet."
            )

        parts = [
            f"{self.post_count} "
            f"{'post' if self.post_count == 1 else 'posts'}",
            f"{self.topic_count} topics",
            f"{self.concept_count} concepts",
            f"{self.question_count} interview questions",
        ]

        return (
            "Data Engineering interview knowledge base: "
            + ", ".join(parts)
            + "."
        )


def build_site_model(
    knowledge_base: CanonicalKnowledgeBase,
) -> SiteModel:
    """
    Derive the ordered site model from a canonical knowledge base.

    Deterministic for a given input: posts are ordered by ID, topics
    alphabetically, and questions by type then difficulty.
    """

    posts = tuple(
        sorted(
            knowledge_base.posts,
            key=lambda post: post.id.casefold(),
        )
    )

    # Post pages keep their ID as the file name where possible, so a
    # URL reads the same as the identifier it represents. The registry
    # still resolves the rare case of two distinct IDs that reduce to
    # the same safe form.
    post_registry = SlugRegistry(transform=safe_id)
    post_slugs = [
        post_registry.register(post.id) for post in posts
    ]

    topics = _build_topics(posts, post_slugs)

    concepts = sorted(
        {
            concept
            for post in posts
            for concept in post.ai_analysis.concepts
            if concept
        },
        key=str.casefold,
    )

    concept_entries = _build_concepts(posts, post_slugs)

    technology_entries = _build_technologies(posts, post_slugs)

    questions: list[QuestionEntry] = []

    for post, slug in zip(posts, post_slugs):
        topics_for_post = tuple(post_topics(post))

        for index, question in enumerate(post.interview_questions):
            questions.append(
                QuestionEntry(
                    key=f"{slug}#{index}",
                    post_id=post.id,
                    post_slug=slug,
                    question=question.question,
                    answer=question.answer or "",
                    question_type=question.type,
                    difficulty=question.difficulty,
                    topics=topics_for_post,
                )
            )

    return SiteModel(
        posts=posts,
        post_slugs=tuple(post_slugs),
        topics=topics,
        topic_slugs={
            entry.label: entry.slug for entry in topics
        },
        questions=tuple(
            sorted(questions, key=lambda item: item.sort_key)
        ),
        concepts=tuple(concepts),
        concept_entries=concept_entries,
        technology_entries=technology_entries,
        generated_at=knowledge_base.generated_at,
        stats=knowledge_base.stats.model_dump(),
    )


def _build_topics(
    posts: tuple[KnowledgePost, ...],
    post_slugs: tuple[str, ...],
) -> tuple[TopicEntry, ...]:
    """
    Group posts by topic and by subtopic.

    Grouping walks posts in canonical ID order, so the per-topic post
    ordering is stable regardless of the order the aggregator emitted.
    """

    registry = SlugRegistry()
    grouped: dict[str, dict[str, object]] = {}

    for post, slug in zip(posts, post_slugs):
        concepts = [
            concept
            for concept in post.ai_analysis.concepts
            if concept
        ]

        # Topics come from the shared helper, which is the same one the
        # consolidation layer uses. Deriving them separately here meant
        # the knowledge base could report a topic the site never
        # rendered, or the other way round.
        labels: list[tuple[str, str]] = [
            (TOPIC_KIND, topic) for topic in post_topics(post)
        ]

        labels += [
            (SUBTOPIC_KIND, subtopic)
            for subtopic in post.ai_analysis.subtopics
        ]

        for kind, label in labels:
            if not label or not label.strip():
                continue

            entry = grouped.get(label)

            if entry is None:
                entry = {
                    "kind": kind,
                    "slug": registry.register(label),
                    "post_slugs": [],
                    "concepts": [],
                    "question_count": 0,
                }
                grouped[label] = entry

            entry["post_slugs"].append(slug)
            entry["concepts"].extend(concepts)
            entry["question_count"] += len(
                post.interview_questions
            )

    entries = [
        TopicEntry(
            label=label,
            slug=str(entry["slug"]),
            kind=str(entry["kind"]),
            post_slugs=tuple(entry["post_slugs"]),
            concepts=tuple(
                sorted(set(entry["concepts"]), key=str.casefold)
            ),
            question_count=int(entry["question_count"]),
        )
        for label, entry in grouped.items()
    ]

    return tuple(
        sorted(
            entries,
            key=lambda entry: (
                entry.sort_key,
                entry.slug,
            ),
        )
    )


def _build_concepts(
    posts: tuple[KnowledgePost, ...],
    post_slugs: tuple[str, ...],
) -> tuple[ConceptEntry, ...]:
    """
    Group concepts across posts, keeping every post that mentions one.

    A concept mentioned by several posts is the case that matters: it
    is what turns a list of posts into a navigable concept, and the
    reason a reader can follow a concept back to each source.
    """

    registry = SlugRegistry()
    grouped: dict[str, dict[str, object]] = {}
    order: list[str] = []

    for post, slug in zip(posts, post_slugs):
        labels = post_topics(post)
        found = detect_technologies(post.original_text)

        for concept in post.ai_analysis.concepts:
            label = concept.strip()

            if not label:
                continue

            entry = grouped.get(label)

            if entry is None:
                entry = {
                    "slug": registry.register(label),
                    "posts": [],
                    "topics": [],
                    "technologies": [],
                }
                grouped[label] = entry
                order.append(label)

            posts_for: list[str] = entry["posts"]  # type: ignore[assignment]
            posts_for.append(slug)

            topics_for: list[str] = entry["topics"]  # type: ignore[assignment]
            technologies_for: list[str] = entry["technologies"]  # type: ignore[assignment]

            for topic in labels:
                if topic not in topics_for:
                    topics_for.append(topic)

            for technology in found:
                if technology not in technologies_for:
                    technologies_for.append(technology)

    entries = [
        ConceptEntry(
            label=label,
            slug=str(grouped[label]["slug"]),
            post_slugs=tuple(grouped[label]["posts"]),  # type: ignore[arg-type]
            topics=tuple(grouped[label]["topics"]),  # type: ignore[arg-type]
            technologies=tuple(grouped[label]["technologies"]),  # type: ignore[arg-type]
        )
        for label in order
    ]

    return tuple(sorted(entries, key=lambda item: item.sort_key))


def _build_technologies(
    posts: tuple[KnowledgePost, ...],
    post_slugs: tuple[str, ...],
) -> tuple[TechnologyEntry, ...]:
    """
    Group recognised technologies across posts.

    Detection is the shared one from the consolidation layer, so the
    site and the canonical knowledge base never disagree about which
    technologies a post uses. A technology only appears when the post
    text actually mentions it.
    """

    registry = SlugRegistry()
    grouped: dict[str, dict[str, object]] = {}

    for post, slug in zip(posts, post_slugs):
        topics = post_topics(post)

        for technology in detect_technologies(post.original_text):
            entry = grouped.setdefault(
                technology,
                {
                    "slug": registry.register(technology),
                    "posts": [],
                    "topics": [],
                    "questions": 0,
                },
            )

            posts_for: list[str] = entry["posts"]  # type: ignore[assignment]
            topics_for: list[str] = entry["topics"]  # type: ignore[assignment]

            posts_for.append(slug)

            for topic in topics:
                if topic not in topics_for:
                    topics_for.append(topic)

            entry["questions"] = int(entry["questions"]) + len(  # type: ignore[arg-type]
                post.interview_questions
            )

    entries = [
        TechnologyEntry(
            label=label,
            slug=str(entry["slug"]),
            post_slugs=tuple(entry["posts"]),  # type: ignore[arg-type]
            topics=tuple(entry["topics"]),  # type: ignore[arg-type]
            question_count=int(entry["questions"]),  # type: ignore[arg-type]
        )
        for label, entry in grouped.items()
    ]

    return tuple(sorted(entries, key=lambda item: item.sort_key))
