"""
Tests for knowledge consolidation.

Consolidation is what stops the knowledge base being a list of posts.
It groups topics, concepts and technologies that several posts
contribute to, keeps every group's provenance, and must never
introduce a claim no source supports.

These assert on behaviour: which posts end up in a group, whether
provenance survives, and whether the output is deterministic.
"""

from __future__ import annotations

import pytest

from src.aggregation.consolidation import (
    ConsolidationError,
    classify_content,
    consolidate,
    detect_technologies,
    verify,
)
from src.models import (
    AIAnalysis,
    Classification,
    InterviewQuestion,
    KnowledgePost,
    SourceInfo,
)
from datetime import datetime, timezone


def make_post(
    post_id: str,
    *,
    text: str = "Body text.",
    topics: tuple[str, ...] = (),
    subtopics: tuple[str, ...] = (),
    concepts: tuple[str, ...] = (),
    primary_topic: str | None = None,
    secondary_topics: tuple[str, ...] = (),
    questions: tuple[tuple[str, str, str, str], ...] = (),
    relevant: bool = False,
    url: str | None = None,
) -> KnowledgePost:
    """
    A valid post.

    Question tuples are (question, type, difficulty, answer), so a test
    can state a question without repeating the model construction.
    """

    return KnowledgePost(
        id=post_id,
        source=SourceInfo(
            platform="linkedin",
            url=url or f"https://example.invalid/{post_id}",
            captured_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
        original_text=text,
        media=[],
        ai_analysis=AIAnalysis(
            summary=f"Summary of {post_id}.",
            topics=list(topics),
            subtopics=list(subtopics),
            concepts=list(concepts),
        ),
        interview_questions=[
            InterviewQuestion(
                question=question,
                type=question_type,  # type: ignore[arg-type]
                difficulty=difficulty,  # type: ignore[arg-type]
                answer=answer,
            )
            for question, question_type, difficulty, answer in questions
        ],
        classification=Classification(
            domain="Data Engineering",
            primary_topic=primary_topic,
            secondary_topics=list(secondary_topics),
            interview_relevant=relevant,
        ),
    )


# ---------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------


def test_two_posts_about_one_topic_become_one_group():
    """The point of consolidation: shared material is grouped."""

    posts = [
        make_post("a", topics=("Databricks",), primary_topic="Databricks"),
        make_post("b", topics=("Databricks",), primary_topic="Databricks"),
    ]

    index = consolidate(posts)

    assert len(index.topics) == 1
    assert index.topics[0].post_ids == ["a", "b"]


def test_a_topic_naming_difference_still_consolidates():
    """
    "Delta Lake", "delta lake" and "Delta-Lake" are one topic. Splitting
    them would fragment the knowledge base over spelling.
    """

    posts = [
        make_post("a", topics=("Delta Lake",)),
        make_post("b", topics=("delta lake",)),
        make_post("c", topics=("Delta-Lake",)),
    ]

    index = consolidate(posts)

    assert len(index.topics) == 1
    assert index.topics[0].post_ids == ["a", "b", "c"]


def test_distinct_topics_stay_separate():
    posts = [
        make_post("a", topics=("Databricks",)),
        make_post("b", topics=("Airflow",)),
    ]

    index = consolidate(posts)

    assert [node.name for node in index.topics] == ["Airflow", "Databricks"]


def test_a_topic_repeated_inside_one_post_is_listed_once():
    """The primary topic also appearing in topics is not two entries."""

    posts = [
        make_post(
            "a",
            topics=("Databricks", "Databricks"),
            primary_topic="Databricks",
        ),
    ]

    index = consolidate(posts)

    assert len(index.topics) == 1
    assert index.topics[0].post_ids == ["a"]


def test_the_classification_topic_leads():
    posts = [
        make_post(
            "a",
            topics=("Performance optimization", "Databricks"),
            primary_topic="Databricks",
        ),
    ]

    index = consolidate(posts)

    # The model's own primary topic is first, so the strongest
    # judgement leads the group.
    assert index.topics[0].name == "Databricks"


def test_concepts_group_across_posts():
    posts = [
        make_post("a", concepts=("Partitioning",)),
        make_post("b", concepts=("Partitioning",)),
    ]

    index = consolidate(posts)

    assert len(index.concepts) == 1
    assert index.concepts[0].post_ids == ["a", "b"]
    assert index.concepts[0].mention_count == 2


def test_a_group_records_the_concepts_inside_it():
    posts = [
        make_post(
            "a",
            topics=("Databricks",),
            concepts=("Partitioning", "Z-Ordering"),
        ),
    ]

    index = consolidate(posts)

    assert index.topics[0].concepts == ["Partitioning", "Z-Ordering"]


def test_a_group_counts_the_questions_inside_it():
    posts = [
        make_post(
            "a",
            topics=("Databricks",),
            questions=(
                ("What is Z-Ordering?", "theory", "easy", "It sorts."),
                ("How does it help?", "troubleshooting", "medium", "Prunes."),
            ),
        ),
    ]

    index = consolidate(posts)

    assert index.topics[0].question_count == 2


# ---------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------


def test_every_group_records_where_it_came_from():
    """A group with no sources would be a claim nothing supports."""

    posts = [
        make_post("a", topics=("Databricks",), concepts=("Partitioning",)),
        make_post("b", topics=("Databricks",), concepts=("Partitioning",)),
    ]

    index = consolidate(posts)

    assert index.topics[0].post_ids == ["a", "b"]
    assert index.concepts[0].post_ids == ["a", "b"]


def test_a_question_points_at_its_post_and_source():
    posts = [
        make_post(
            "a",
            topics=("Databricks",),
            url="https://example.invalid/a",
            questions=(
                ("What is Z-Ordering?", "theory", "easy", "It sorts."),
            ),
        ),
    ]

    index = consolidate(posts)

    question = index.questions[0]

    assert question.post_id == "a"
    assert question.source_url == "https://example.invalid/a"
    assert question.topics == ["Databricks"]


def test_a_question_identifier_is_unique_across_posts():
    posts = [
        make_post(
            "a",
            questions=(("First?", "theory", "easy", "One."),),
        ),
        make_post(
            "b",
            questions=(("Second?", "theory", "easy", "Two."),),
        ),
    ]

    index = consolidate(posts)

    identifiers = [question.id for question in index.questions]

    assert len(identifiers) == len(set(identifiers))


def test_consolidation_refuses_to_reference_an_unknown_post():
    """
    A group naming a post that does not exist means the node was built
    wrongly, and a knowledge base that points at nothing is worse than
    one that admits it is incomplete.
    """

    posts = [make_post("a", topics=("Databricks",))]

    index = consolidate(posts)

    index.topics[0].post_ids.append("ghost")

    with pytest.raises(ConsolidationError) as error:
        verify(index, posts)

    assert "ghost" in str(error.value)


# ---------------------------------------------------------------------
# Technologies
# ---------------------------------------------------------------------


def test_a_technology_is_recognised_from_the_source_text():
    assert "Databricks" in detect_technologies(
        "We tuned the Databricks cluster for this workload."
    )


def test_no_technology_appears_without_mention():
    """
    Recognition only. An absent technology must not be listed, or the
    knowledge base would claim coverage that no post supports.
    """

    found = detect_technologies("A colleague announced a new job.")

    assert found == []


def test_a_technology_links_the_posts_that_use_it():
    posts = [
        make_post("a", text="Spark SQL and Databricks."),
        make_post("b", text="More about Databricks performance."),
    ]

    index = consolidate(posts)

    node = next(
        item for item in index.technologies if item.name == "Databricks"
    )

    assert node.post_ids == ["a", "b"]


@pytest.mark.parametrize(
    "text,technology",
    [
        ("delta lake tables", "Delta Lake"),
        ("we use pyspark daily", "Apache Spark"),
        ("window function usage", "SQL"),
        ("slowly changing dimension type 2", "Slowly Changing Dimensions"),
        ("kafka topic retention", "Apache Kafka"),
        ("airflow dag retries", "Apache Airflow"),
    ],
)
def test_each_supported_technology_is_recognised(text, technology):
    assert technology in detect_technologies(text)


# ---------------------------------------------------------------------
# Content classification
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,kind",
    [
        ("Thrilled to share with my network a new journey", "job_announcement"),
        ("Join us at our UNLEASH event tomorrow", "event"),
        ("Congratulations on the new role", "congratulation"),
        # "following" on its own is too ambiguous to be a marker: it
        # appears in technical writing ("following the partition"), so
        # the profile-widget wording is used instead.
        ("Followers 2,716", "social"),
    ],
)
def test_non_technical_content_is_named_rather_than_dropped(text, kind):
    """
    A post that teaches nothing is still stored and still traceable.
    It is labelled, so a reader can see why it contributed no
    knowledge, instead of silently disappearing.
    """

    assert classify_content(make_post("a", text=text)) == kind


def test_technical_content_is_recognised():
    post = make_post(
        "a",
        text="Z-Ordering improves Delta Lake pruning.",
        topics=("Databricks",),
        relevant=True,
    )

    assert classify_content(post) == "technical"


def test_content_that_fits_no_category_is_labelled_unknown():
    """
    Not classified is not the same as classified as nothing. The label
    records that nothing matched, which is honest.
    """

    assert classify_content(make_post("a", text="...")) == "unknown"


def test_every_post_gets_a_content_kind():
    posts = [make_post("a"), make_post("b", topics=("Databricks",))]

    index = consolidate(posts)

    assert set(index.content_kinds) == {"a", "b"}


# ---------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------


def test_consolidation_is_deterministic():
    """
    Two runs over the same posts must produce identical output, so a
    change in the repository always means something changed.
    """

    posts = [
        make_post("a", topics=("Databricks", "SQL")),
        make_post("b", topics=("Databricks",), concepts=("Skew",)),
        make_post("c", topics=("Airflow",)),
    ]

    first = consolidate(posts).as_dict()
    second = consolidate(list(reversed(posts))).as_dict()

    # Post order does not change the shape, only the ids are the same.
    assert first == second


def test_groups_are_sorted_by_name():
    posts = [
        make_post("a", topics=("Zebra", "Apple", "Mango")),
    ]

    index = consolidate(posts)

    assert [node.name for node in index.topics] == [
        "Apple",
        "Mango",
        "Zebra",
    ]


def test_no_group_is_empty():
    """
    A group with no posts is a topic with no source, which is exactly
    what must never be published.
    """

    posts = [
        make_post("a", topics=("Databricks",)),
        make_post("b", topics=("Airflow",)),
    ]

    index = consolidate(posts)

    for group in (index.topics, index.concepts, index.technologies):
        for node in group:
            assert node.post_ids, node.name


def test_nothing_is_invented_for_an_empty_knowledge_base():
    posts = []

    index = consolidate(posts)

    assert index.topics == []
    assert index.concepts == []
    assert index.technologies == []
    assert index.questions == []


def test_a_slug_is_safe_for_a_url_and_a_directory():
    posts = [
        make_post("a", topics=("C++ / Data Eng!",)),
    ]

    index = consolidate(posts)

    slug = index.topics[0].slug

    assert slug == "c-data-eng"
    assert "/" not in slug
    assert " " not in slug
