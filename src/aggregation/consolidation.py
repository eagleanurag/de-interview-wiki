"""
Consolidate individual posts into navigable knowledge areas.

A knowledge base of one page per post is a list, not a knowledge base.
This module builds the other half of it: topics, concepts,
technologies and questions that several posts contribute to, each
carrying the posts it came from so provenance is never lost.

Nothing here invents content. Every topic, concept and question is
taken from what a post's enrichment actually recorded, and an item
with no supporting post simply does not appear. Consolidation groups;
it does not summarise or merge away meaning.

Grouping is deterministic. Two runs over the same posts produce the
same nodes in the same order, so a diff in the repository means
something actually changed.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from src.models import KnowledgePost


class ConsolidationError(RuntimeError):
    """Raised when consolidation cannot be trusted."""


#: Technologies recognised in source material. Used only to recognise
#: what a post is about; a technology is only emitted when a post
#: actually mentions it.
TECHNOLOGIES: dict[str, tuple[str, ...]] = {
    "Databricks": (
        "databricks",
        "delta live tables",
        "photon",
        "unity catalog",
    ),
    "Apache Spark": ("apache spark", "spark sql", "pyspark", "spark"),
    "Delta Lake": ("delta lake", "deltalake", "delta lake"),
    "Azure": ("azure", "synapse", "data factory", "azure sql"),
    "AWS": ("aws", "redshift", "emr", "glue", "athena", "sagemaker"),
    "GCP": ("gcp", "bigquery", "dataflow", "dataproc"),
    "Apache Kafka": ("kafka", "event streaming", "change data capture"),
    "Apache Airflow": ("airflow", "dag orchestration"),
    "dbt": ("dbt",),
    "SQL": ("sql", "window function", "cte", "join strategy"),
    "Python": ("python", "pandas", "pyspark"),
    "Data Modelling": (
        "dimensional model",
        "star schema",
        "snowflake schema",
        "fact table",
        "dimension table",
        "data model",
    ),
    "Slowly Changing Dimensions": (
        "slowly changing dimension",
        "scd type",
        "scd2",
    ),
    "ETL and ELT": ("etl", "elt", "extract load transform"),
    "Lakehouse": ("lakehouse",),
    "Data Lake": ("data lake",),
    "Data Warehouse": ("data warehouse", "warehousing"),
    "Orchestration": ("orchestration", "scheduler", "workflow engine"),
    "Data Quality": ("data quality", "data contract", "validation framework"),
    "Data Governance": ("data governance", "lineage", "catalog"),
    "Streaming": ("streaming", "micro batch", "real time", "near real time"),
    "Distributed Systems": (
        "distributed system",
        "partitioning",
        "shuffle",
        "replication",
        "fault tolerance",
    ),
    "Performance Tuning": (
        "performance optimization",
        "performance tuning",
        "query optimization",
        "cluster utilization",
    ),
    "Testing": ("data test", "unit test", "integration test", "dbt test"),
    "CI CD": ("ci/cd", "continuous integration", "continuous delivery"),
}


#: Content kinds that are not technical knowledge. Recorded so a reader
#: can see why a post contributed nothing, rather than the post simply
#: vanishing.
NON_TECHNICAL_MARKERS: tuple[tuple[str, str], ...] = (
    ("job announcement", "job_announcement"),
    ("new journey", "job_announcement"),
    ("thrilled to share", "job_announcement"),
    ("i am happy to announce", "job_announcement"),
    ("joining", "job_announcement"),
    ("congratulations", "congratulation"),
    ("happy to announce", "announcement"),
    ("event", "event"),
    ("webinar", "event"),
    ("unleash", "event"),
    ("meetup", "event"),
    ("certification", "certification"),
    ("badge", "certification"),
    ("thank you", "appreciation"),
    ("happy birthday", "social"),
    ("followers", "social"),
)


def _slug(value: str) -> str:
    """
    A stable, filesystem- and URL-safe key for a label.

    Lowercase and hyphenated so it works as a directory name, a URL
    segment and a job id at once, and so two labels differing only in
    case consolidate into one node instead of two.
    """

    cleaned = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())

    return cleaned.strip("-") or "unknown"


def _normalise(value: str) -> str:
    """
    A comparison key that ignores case and punctuation.

    "Delta Lake", "delta lake" and "Delta-Lake" are one concept, and
    merging them is the whole point of consolidation.
    """

    return re.sub(r"[^a-z0-9]+", " ", value.strip().lower()).strip()


def detect_technologies(text: str) -> list[str]:
    """
    Which known technologies a post mentions.

    Recognition only. A technology is returned because the source text
    contains it, never because it is expected to be there.
    """

    haystack = text.lower()

    found: list[str] = []

    for technology, markers in TECHNOLOGIES.items():
        for marker in markers:
            if marker in haystack:
                found.append(technology)
                break

    return sorted(found)


def classify_content(post: KnowledgePost) -> str:
    """
    What kind of content a post is.

    Used to explain why a post contributed no knowledge, so it is never
    silently dropped. A post that teaches something is
    ``"technical"`` regardless of how the classifier is worded.
    """

    body = post.original_text.lower()

    for marker, kind in NON_TECHNICAL_MARKERS:
        if marker in body:
            return kind

    if post.classification.interview_relevant:
        return "technical"

    if post.ai_analysis.concepts or post.ai_analysis.topics:
        return "technical"

    # A post whose enrichment has not run yet is its own case, and it
    # is labelled as such. Filing it as "unclassified" would read as a
    # judgement about the content when it is really a statement about
    # the pipeline.
    if not (post.ai_analysis.summary or "").strip():
        return "unenriched"

    return "unclassified"


@dataclass
class TopicNode:
    """One consolidated knowledge area."""

    name: str
    slug: str
    post_ids: list[str] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)
    question_count: int = 0
    technologies: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "slug": self.slug,
            "post_ids": sorted(self.post_ids),
            "concepts": sorted(self.concepts),
            "technologies": sorted(self.technologies),
            "question_count": self.question_count,
        }


@dataclass
class ConceptNode:
    """One concept, with everything that mentions it."""

    name: str
    slug: str
    post_ids: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    technologies: list[str] = field(default_factory=list)

    @property
    def mention_count(self) -> int:
        return len(self.post_ids)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "slug": self.slug,
            "post_ids": sorted(self.post_ids),
            "topics": sorted(self.topics),
            "technologies": sorted(self.technologies),
            "mention_count": self.mention_count,
        }


@dataclass
class TechnologyNode:
    """One recognised technology and the posts that use it."""

    name: str
    slug: str
    post_ids: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    question_count: int = 0

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "slug": self.slug,
            "post_ids": sorted(self.post_ids),
            "topics": sorted(self.topics),
            "question_count": self.question_count,
        }


@dataclass
class QuestionNode:
    """One interview question, traceable to its source."""

    id: str
    question: str
    type: str
    difficulty: str
    answer: str
    post_id: str
    source_url: str | None
    topics: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "type": self.type,
            "difficulty": self.difficulty,
            "answer": self.answer,
            "post_id": self.post_id,
            "source_url": self.source_url,
            "topics": sorted(self.topics),
        }


@dataclass
class KnowledgeIndex:
    """The consolidated view of a set of posts."""

    topics: list[TopicNode] = field(default_factory=list)
    concepts: list[ConceptNode] = field(default_factory=list)
    technologies: list[TechnologyNode] = field(default_factory=list)
    questions: list[QuestionNode] = field(default_factory=list)
    content_kinds: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "topics": [node.as_dict() for node in self.topics],
            "concepts": [node.as_dict() for node in self.concepts],
            "technologies": [node.as_dict() for node in self.technologies],
            "questions": [node.as_dict() for node in self.questions],
            "content_kinds": dict(sorted(self.content_kinds.items())),
        }


def consolidate(posts: Iterable[KnowledgePost]) -> KnowledgeIndex:
    """
    Build the consolidated knowledge index.

    Every node keeps the posts it came from, so a reader can always get
    from a topic back to the original text it summarises.
    """

    index = KnowledgeIndex()

    topics: dict[str, TopicNode] = {}
    concepts: dict[str, ConceptNode] = {}
    technologies: dict[str, TechnologyNode] = {}

    for post in posts:
        index.content_kinds[post.id] = classify_content(post)

        post_topics = _post_topics(post)
        post_technologies = detect_technologies(post.original_text)

        for label in post_topics:
            key = _normalise(label)

            node = topics.get(key)

            if node is None:
                node = TopicNode(name=label.strip(), slug=_slug(label))
                topics[key] = node

            node.post_ids.append(post.id)

            for concept in post.ai_analysis.concepts:
                if concept not in node.concepts:
                    node.concepts.append(concept)

            node.question_count += len(post.interview_questions)

            for technology in post_technologies:
                if technology not in node.technologies:
                    node.technologies.append(technology)

        for concept in post.ai_analysis.concepts:
            key = _normalise(concept)

            node = concepts.get(key)

            if node is None:
                node = ConceptNode(
                    name=concept.strip(), slug=_slug(concept)
                )
                concepts[key] = node

            node.post_ids.append(post.id)

            for label in post_topics:
                if label not in node.topics:
                    node.topics.append(label)

            for technology in post_technologies:
                if technology not in node.technologies:
                    node.technologies.append(technology)

        for technology in post_technologies:
            node = technologies.get(technology)

            if node is None:
                node = TechnologyNode(
                    name=technology, slug=_slug(technology)
                )
                technologies[technology] = node

            node.post_ids.append(post.id)
            node.question_count += len(post.interview_questions)

            for label in post_topics:
                if label not in node.topics:
                    node.topics.append(label)

        for position, question in enumerate(post.interview_questions):
            index.questions.append(
                QuestionNode(
                    id=f"{post.id}-q{position + 1}",
                    question=question.question,
                    type=question.type,
                    difficulty=question.difficulty,
                    answer=question.answer or "",
                    post_id=post.id,
                    source_url=post.source.url,
                    topics=post_topics,
                )
            )

    # Sorted by name so the output does not depend on the order posts
    # arrived in.
    index.topics = sorted(
        topics.values(), key=lambda node: node.name.casefold()
    )
    index.concepts = sorted(
        concepts.values(), key=lambda node: node.name.casefold()
    )
    index.technologies = sorted(
        technologies.values(), key=lambda node: node.name.casefold()
    )
    index.questions.sort(key=lambda node: (node.post_id, node.id))

    return index


def _post_topics(post: KnowledgePost) -> list[str]:
    """
    The topics a post belongs under.

    The classification's own topic leads, because that is the model's
    judgement, and the analysed topics follow. Duplicates are removed
    by normalised form so one topic does not appear twice.
    """

    ordered: list[str] = []

    primary = (post.classification.primary_topic or "").strip()

    if primary:
        ordered.append(primary)

    for label in post.ai_analysis.topics:
        if label.strip():
            ordered.append(label.strip())

    for label in post.classification.secondary_topics:
        if label.strip():
            ordered.append(label.strip())

    seen: set[str] = set()
    unique: list[str] = []

    for label in ordered:
        key = _normalise(label)

        if not key or key in seen:
            continue

        seen.add(key)
        unique.append(label)

    return unique


def verify(index: KnowledgeIndex, posts: Iterable[KnowledgePost]) -> None:
    """
    Refuse a consolidated index that references a post it never saw.

    Consolidation can only ever reference posts that exist, so an
    unknown reference means a node was built wrongly rather than that
    the knowledge base is merely incomplete.
    """

    known = {post.id for post in posts}

    dangling: list[str] = []

    for group, nodes in (
        ("topic", index.topics),
        ("concept", index.concepts),
        ("technology", index.technologies),
    ):
        for node in nodes:
            for post_id in node.post_ids:
                if post_id not in known:
                    dangling.append(f"{group}/{node.slug} -> {post_id}")

    for question in index.questions:
        if question.post_id not in known:
            dangling.append(f"question/{question.id} -> {question.post_id}")

    if dangling:
        raise ConsolidationError(
            "Consolidation referenced unknown posts: "
            + ", ".join(sorted(set(dangling))[:10])
        )
