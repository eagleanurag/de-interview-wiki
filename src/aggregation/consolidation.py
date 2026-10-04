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
import unicodedata
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
    "Snowflake": ("snowflake", "snowpark", "virtual warehouse"),
    "Apache Iceberg": ("iceberg",),
    "Apache Hudi": ("hudi",),
    "Apache Flink": ("flink",),
    "Trino": ("trino",),
    "Presto": ("presto",),
    "Apache Beam": ("apache beam", "beam pipeline"),
    "Fivetran": ("fivetran",),
    "Airbyte": ("airbyte",),
    "PostgreSQL": ("postgresql", "postgres"),
    "MongoDB": ("mongodb",),
    "Tableau": ("tableau",),
    "Power BI": ("power bi", "powerbi"),
    "Terraform": ("terraform",),
    "Apache Hive": ("apache hive", "hive metastore"),
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


#: The longest slug a label may produce.
#:
#: Applied here rather than only where a page is written, because the
#: knowledge base is the authoritative artifact: a consumer that follows
#: a concept's ``slug`` to build a link must arrive at the page the site
#: actually generated. Two functions that truncate differently are one
#: function too many, and the disagreement only shows up once a corpus
#: has labels long enough to hit the bound -- which is why it survived
#: every test written against seven posts.
MAX_SLUG_LENGTH = 60


def slug_for(
    value: str,
    *,
    fallback: str,
    max_length: int = MAX_SLUG_LENGTH,
) -> str:
    """
    One slug function, used wherever a label becomes an identifier.

    Lowercase and hyphenated so the result works as a directory name, a
    URL segment and a job id at once, and so two labels differing only
    in case or punctuation consolidate into one node instead of two.

    Accented characters are decomposed and then dropped rather than
    deleted outright, so ``café`` and ``cafe`` land on the same slug and
    two spellings of one label do not become two topics.

    Bounded in length so a label that is a whole sentence does not
    become a file name no filesystem will accept.
    """

    normalized = unicodedata.normalize("NFKD", str(value))

    ascii_only = normalized.encode(
        "ascii", "ignore"
    ).decode("ascii")

    slug = re.sub(r"[^a-z0-9]+", "-", ascii_only.lower()).strip("-")

    if len(slug) > max_length:
        slug = slug[:max_length].rstrip("-")

    return slug or fallback


def _slug(value: str) -> str:
    """
    A stable, filesystem- and URL-safe key for a label.

    Named here because the knowledge base is built before anything is
    rendered, so this is where a label's identity is decided. The site
    asks this module for the same slug rather than deriving its own.
    """

    return slug_for(value, fallback="unknown")


__all__ = [
    "MAX_SLUG_LENGTH",
    "DerivedClaim",
    "normalise_label",
    "slug_for",
]


def normalise_label(value: str) -> str:
    """
    A comparison key that ignores case and punctuation.

    "Delta Lake", "delta lake" and "Delta-Lake" are one concept, and
    merging them is the whole point of consolidation.

    Public because a label's identity is decided here, and anything that
    renders those labels has to group them the same way. Grouping on the
    raw text instead produced one knowledge-base concept spread across
    several pages, each showing only the posts that happened to spell it
    that way.
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

    Read from the post's own text, so it is available whether or not
    the post has been enriched. A post that teaches something is
    ``"technical"`` regardless of how the classifier is worded.

    This is a statement about the content and nothing else. Whether the
    enrichment stage has run is a separate fact, reported in the
    knowledge base statistics, because folding the two together makes
    both unreadable: a post can be obviously a job announcement and
    obviously not yet enriched, and one label cannot say both.
    """

    body = post.original_text.lower()

    for marker, kind in NON_TECHNICAL_MARKERS:
        if marker in body:
            return kind

    if post.classification.interview_relevant:
        return "technical"

    if post.ai_analysis.concepts or post.ai_analysis.topics:
        return "technical"

    # Media text counts as content the post carries, so a post that is
    # only a document is still classifiable.
    if any((media.extracted_text or "").strip() for media in post.media):
        return "technical"

    return "unknown"


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


@dataclass(frozen=True)
class DerivedClaim:
    """
    A technology named by something other than the post's own text.

    The post body is the author's words and nothing else may be mixed
    into it, so a technology that appears only inside an attached image
    cannot reach the consolidation by being appended to the text. It
    arrives here instead, carrying where it came from.

    The provenance is not decoration. ``image_ocr`` means a machine read
    the word off a picture, which is weaker evidence than an author
    typing it, and a technology page that lists 128 groups for ``SQL``
    without saying how many of those were read off a slide is
    overstating its own coverage.
    """

    name: str

    #: ``image_ocr`` for a technology read out of a transcribed slide.
    source_kind: str = "image_ocr"

    #: The package's own grouping identifier for the post.
    group_id: str = ""

    filename: str = ""
    slide_number: int | None = None

    #: The slide's own words, when they could be matched to a specific
    #: transcription. Kept so the claim can be weighed rather than taken
    #: on trust.
    evidence: str = ""


@dataclass
class TechnologyNode:
    """One recognised technology and the posts that use it."""

    name: str
    slug: str
    post_ids: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    question_count: int = 0

    #: Technologies found only in an attached image, with the slide they
    #: were read from. Empty for any technology the post's own text
    #: mentions, which is most of them.
    derived: list[DerivedClaim] = field(default_factory=list)

    def add_derived(self, claim: DerivedClaim) -> None:
        """
        Record a claim, without repeating an identical one.

        The same slide is offered once per post that carries it, and a
        technology that appears in 40 posts would otherwise carry 40
        copies of the same reading.
        """

        key = (claim.filename, claim.slide_number, claim.name)

        for existing in self.derived:
            other = (existing.filename, existing.slide_number, existing.name)

            if other == key:
                return

        self.derived.append(claim)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "slug": self.slug,
            "post_ids": sorted(self.post_ids),
            "topics": sorted(self.topics),
            "question_count": self.question_count,
            "derived": [
                {
                    "name": claim.name,
                    "source_kind": claim.source_kind,
                    "group_id": claim.group_id,
                    "filename": claim.filename,
                    "slide_number": claim.slide_number,
                    "evidence": claim.evidence,
                }
                for claim in sorted(
                    self.derived,
                    key=lambda item: (
                        item.name,
                        item.filename,
                        item.slide_number if item.slide_number is not None else -1,
                    ),
                )
            ],
        }


@dataclass
class QuestionNode:
    """
    One interview question, traceable to every source it came from.

    ``post_ids`` lists them all rather than just the first, so merging
    a duplicate never loses the fact that two posts asked it.
    """

    id: str
    question: str
    type: str
    difficulty: str
    answer: str
    post_id: str
    source_url: str | None
    topics: list[str] = field(default_factory=list)
    post_ids: list[str] = field(default_factory=list)

    @property
    def also_asked_in(self) -> list[str]:
        return [post for post in self.post_ids if post != self.post_id]

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "type": self.type,
            "difficulty": self.difficulty,
            "answer": self.answer,
            "post_id": self.post_id,
            "post_ids": sorted(self.post_ids or [self.post_id]),
            "source_url": self.source_url,
            "topics": sorted(self.topics),
        }


@dataclass
class KnowledgeIndex:
    """The consolidated view of a set of posts."""

    # Whether each post has been analysed. Kept apart from
    # ``content_kinds`` because "this is a job announcement" and "this
    # has not been enriched yet" are different facts, and merging them
    # would force one label to carry both."""

    topics: list[TopicNode] = field(default_factory=list)
    subtopics: list[TopicNode] = field(default_factory=list)
    concepts: list[ConceptNode] = field(default_factory=list)
    technologies: list[TechnologyNode] = field(default_factory=list)
    questions: list[QuestionNode] = field(default_factory=list)
    content_kinds: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "topics": [node.as_dict() for node in self.topics],
            "subtopics": [node.as_dict() for node in self.subtopics],
            "concepts": [node.as_dict() for node in self.concepts],
            "technologies": [node.as_dict() for node in self.technologies],
            "questions": [node.as_dict() for node in self.questions],
            "content_kinds": dict(sorted(self.content_kinds.items())),
        }


#: Words that carry no meaning in a question, and stop two questions
#: from looking alike when only their framing differs.
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "what",
        "when",
        "which",
        "why",
        "with",
        "you",
        "your",
    }
)

#: How alike two questions must be before they are treated as the same
#: question. Deliberately high: merging two genuinely different
#: questions because they share vocabulary would lose one of them,
#: which is worse than showing a reader two similar questions.
DUPLICATE_THRESHOLD = 0.82


def _question_key(text: str) -> str:
    """
    A comparison key for a question.

    Case, punctuation and wording order all differ between two
    renderings of the same question, so none of them is allowed to make
    it look like a different one.
    """

    words = re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split()

    significant = [word for word in words if word not in _STOP_WORDS]

    return " ".join(sorted(significant))


def _content_words(text: str) -> frozenset:
    words = re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split()

    return frozenset(word for word in words if word not in _STOP_WORDS)


def _similar(left: frozenset, right: frozenset) -> float:
    """Jaccard overlap of two questions' content words."""

    if not left or not right:
        return 0.0

    union = left | right

    if not union:
        return 0.0

    return len(left & right) / len(union)


def _merge_questions(nodes: list["QuestionNode"]) -> list["QuestionNode"]:
    """
    Fold questions that are the same question into one.

    Two posts that both discuss the same idea will often produce very
    similar questions. Showing a reader the same question twice is
    noise, so the repeats are folded together and every post they came
    from is kept on the surviving question.

    Merging is conservative. Two questions that merely share vocabulary
    stay separate, because a wrong merge silently deletes a question a
    reader might have needed.
    """

    kept: list[QuestionNode] = []
    keys: list[frozenset] = []
    seen_exact: dict[str, QuestionNode] = {}

    for node in nodes:
        key = _question_key(node.question)

        existing_exact = seen_exact.get(key)

        if existing_exact is not None:
            for post_id in node.post_ids or [node.post_id]:
                if post_id not in existing_exact.post_ids:
                    existing_exact.post_ids.append(post_id)

            # The topics have to come with it: two posts asking the same
            # question sit under different topics, and a merged question
            # filed under only one of them would hide it from the other.
            for topic in node.topics:
                if topic not in existing_exact.topics:
                    existing_exact.topics.append(topic)

            continue

        words = _content_words(node.question)

        merged = False

        for candidate, candidate_words in zip(kept, keys):
            if _similar(words, candidate_words) < DUPLICATE_THRESHOLD:
                continue

            for post_id in node.post_ids or [node.post_id]:
                if post_id not in candidate.post_ids:
                    candidate.post_ids.append(post_id)

            for topic in node.topics:
                if topic not in candidate.topics:
                    candidate.topics.append(topic)

            seen_exact[key] = candidate
            merged = True
            break

        if merged:
            continue

        if not node.post_ids:
            node.post_ids = [node.post_id]

        kept.append(node)
        keys.append(words)
        seen_exact[key] = node

    return kept


def consolidate(
    posts: Iterable[KnowledgePost],
    derived_technologies: dict[str, list[DerivedClaim]] | None = None,
) -> KnowledgeIndex:
    """
    Build the consolidated knowledge index.

    Every node keeps the posts it came from, so a reader can always get
    from a topic back to the original text it summarises.

    ``derived_technologies`` supplies the per-slide detail for
    technologies that appear only inside an attached image, keyed by post
    identifier. The *names* come from the post's own
    ``ai_analysis.derived_technologies``, which both this function and the
    site builder read, so the two cannot disagree about which posts cover
    a technology. This parameter adds only the attribution: which slide,
    and what that slide's own words were.

    Detail without a name would be dropped and a name without detail
    would still be recorded, so the two halves are usable independently
    and neither is load-bearing for the other.
    """

    index = KnowledgeIndex()

    derived_technologies = derived_technologies or {}

    topics: dict[str, TopicNode] = {}
    subtopics: dict[str, TopicNode] = {}
    concepts: dict[str, ConceptNode] = {}
    technologies: dict[str, TechnologyNode] = {}

    for post in posts:
        index.content_kinds[post.id] = classify_content(post)

        topics_of_post = post_topics(post)
        post_technologies = detect_technologies(post.original_text)

        # A technology read off a slide is a real mention, and it counts
        # toward the technology's post list -- but it is added after the
        # text-derived ones so the node's own list keeps the author's
        # words first.
        for name in post.ai_analysis.derived_technologies:
            if name not in post_technologies:
                post_technologies.append(name)

        for claim in derived_technologies.get(post.id, []):
            if claim.name not in post_technologies:
                post_technologies.append(claim.name)

        for label in topics_of_post:
            key = normalise_label(label)

            node = topics.get(key)

            if node is None:
                node = TopicNode(name=label.strip(), slug=_slug(label))
                topics[key] = node

            # A subtopic is grouped the same way and carried
            # separately, so a reader reconciling the knowledge base
            # against the site sees where every page came from.
            for subtopic in post.ai_analysis.subtopics:
                sub_key = normalise_label(subtopic)

                sub_node = subtopics.get(sub_key)

                if sub_node is None:
                    sub_node = TopicNode(
                        name=subtopic.strip(), slug=_slug(subtopic)
                    )
                    subtopics[sub_key] = sub_node

                sub_node.post_ids.append(post.id)

                for concept in post.ai_analysis.concepts:
                    if concept not in sub_node.concepts:
                        sub_node.concepts.append(concept)

                sub_node.question_count += len(
                    post.interview_questions
                )


            node.post_ids.append(post.id)

            for concept in post.ai_analysis.concepts:
                if concept not in node.concepts:
                    node.concepts.append(concept)

            node.question_count += len(post.interview_questions)

            for technology in post_technologies:
                if technology not in node.technologies:
                    node.technologies.append(technology)

        for concept in post.ai_analysis.concepts:
            key = normalise_label(concept)

            node = concepts.get(key)

            if node is None:
                node = ConceptNode(
                    name=concept.strip(), slug=_slug(concept)
                )
                concepts[key] = node

            node.post_ids.append(post.id)

            for label in topics_of_post:
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

            for label in topics_of_post:
                if label not in node.topics:
                    node.topics.append(label)

        for claim in derived_technologies.get(post.id, []):
            node = technologies.get(claim.name)

            if node is not None:
                node.add_derived(claim)

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
                    topics=topics_of_post,
                    post_ids=[post.id],
                )
            )

    # Sorted by name so the output does not depend on the order posts
    # arrived in.
    index.topics = sorted(
        topics.values(), key=lambda node: node.name.casefold()
    )
    index.subtopics = sorted(
        subtopics.values(), key=lambda node: node.name.casefold()
    )
    index.concepts = sorted(
        concepts.values(), key=lambda node: node.name.casefold()
    )
    index.technologies = sorted(
        technologies.values(), key=lambda node: node.name.casefold()
    )
    index.questions = _merge_questions(index.questions)

    index.questions.sort(key=lambda node: (node.post_id, node.id))

    return index


def post_topics(post: KnowledgePost) -> list[str]:
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
        key = normalise_label(label)

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
        ("subtopic", index.subtopics),
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
