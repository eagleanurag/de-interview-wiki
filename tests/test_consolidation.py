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
    _slug,
    classify_content,
    consolidate,
    detect_technologies,
    normalise_label,
    slug_for,
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


# ---------------------------------------------------------------------
# Question deduplication
# ---------------------------------------------------------------------


def _post_with_questions(post_id: str, questions: list[str]):
    from src.models import InterviewQuestion

    return make_post(
        post_id,
        topics=("Spark",),
        concepts=("Partitioning",),
        questions=tuple(
            (question, "theory", "easy", "Because.") for question in questions
        ),
    )


def test_an_exactly_repeated_question_is_one_question():
    """
    Two posts that both ask the same thing should not show a reader the
    same question twice.
    """

    posts = [
        _post_with_questions("a", ["How does partitioning help?"]),
        _post_with_questions("b", ["How does partitioning help?"]),
    ]

    index = consolidate(posts)

    assert len(index.questions) == 1


def test_a_merged_question_keeps_both_sources():
    """
    Merging must not lose provenance: the reader still needs to know
    both posts asked it.
    """

    posts = [
        _post_with_questions("a", ["How does partitioning help?"]),
        _post_with_questions("b", ["How does partitioning help?"]),
    ]

    index = consolidate(posts)

    assert index.questions[0].post_ids == ["a", "b"]
    assert index.questions[0].also_asked_in == ["b"]


def test_the_same_question_in_different_wording_is_one_question():
    posts = [
        _post_with_questions("a", ["How does partitioning reduce scan?"]),
        _post_with_questions("b", ["Does partitioning reduce the scan?"]),
    ]

    index = consolidate(posts)

    assert len(index.questions) == 1


def test_genuinely_different_questions_are_all_kept():
    """
    A wrong merge silently deletes a question a reader might have
    needed, so distinct questions stay separate.
    """

    posts = [
        _post_with_questions(
            "a",
            [
                "How does partitioning help?",
                "How does bucketing help?",
            ],
        ),
        _post_with_questions("b", ["What causes data skew?"]),
    ]

    index = consolidate(posts)

    assert len(index.questions) == 3


def test_questions_sharing_only_keywords_are_kept_apart():
    """
    Sharing vocabulary is not the same as being the same question.
    """

    posts = [
        _post_with_questions(
            "a",
            [
                "Explain partitioning in Spark.",
                "What is partitioning in Spark used for?",
            ],
        ),
        _post_with_questions("b", ["How is a Spark cluster sized?"]),
    ]

    index = consolidate(posts)

    assert len(index.questions) == 3


def test_case_and_punctuation_do_not_make_a_new_question():
    posts = [
        _post_with_questions("a", ["How does partitioning help?"]),
        _post_with_questions("b", ["how does partitioning help"]),
    ]

    index = consolidate(posts)

    assert len(index.questions) == 1


def test_a_single_post_keeps_its_own_questions_distinct():
    posts = [_post_with_questions("a", ["First?", "Second?"])]

    index = consolidate(posts)

    assert len(index.questions) == 2


def test_merging_carries_the_topics_of_both_posts():
    from src.models import AIAnalysis, Classification, KnowledgePost, SourceInfo

    def post(post_id, topic, question):
        return KnowledgePost(
            id=post_id,
            source=SourceInfo(
                platform="manual",
                captured_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
            original_text="text",
            ai_analysis=AIAnalysis(
                summary="s", topics=[topic], concepts=["Partitioning"]
            ),
            interview_questions=[
                InterviewQuestion(
                    question=question,
                    type="theory",
                    difficulty="easy",
                    answer="a",
                )
            ],
            classification=Classification(
                domain="Data Engineering",
                primary_topic=topic,
                interview_relevant=True,
            ),
        )

    index = consolidate(
        [
            post("a", "Spark", "How does partitioning help?"),
            post("b", "Databricks", "How does partitioning help?"),
        ]
    )

    assert len(index.questions) == 1
    assert sorted(index.questions[0].topics) == ["Databricks", "Spark"]


def test_no_question_is_published_without_a_source():
    posts = [
        _post_with_questions("a", ["First?"]),
        _post_with_questions("b", ["First?"]),
    ]

    index = consolidate(posts)

    for question in index.questions:
        assert question.post_ids
        assert question.post_id


def _post(
    post_id: str,
    *,
    concepts: list[str] | None = None,
    topics: list[str] | None = None,
    text: str = "Delta Lake gives a data lake ACID guarantees.",
):
    """A collected post, for the consolidation tests to group."""

    return KnowledgePost(
        id=post_id,
        source=SourceInfo(
            platform="linkedin",
            url=f"https://www.linkedin.com/feed/update/{post_id}",
            captured_at="2026-10-02T04:30:00+00:00",
        ),
        original_text=text,
        media=[],
        ai_analysis=AIAnalysis(
            summary="Delta Lake notes.",
            topics=topics or ["Delta Lake"],
            subtopics=[],
            concepts=concepts or ["ACID"],
            image_descriptions=[],
        ),
        interview_questions=[],
        classification=Classification(
            domain="Data Engineering",
            primary_topic="Delta Lake",
            secondary_topics=[],
            interview_relevant=True,
        ),
    )


class TestLabelIdentity:
    """
    One slug function, and one grouping key, for the whole project.

    Both were duplicated: the aggregator derived a slug with no length
    bound while the site truncated at sixty characters, and the
    aggregator grouped concepts case-insensitively while the site
    grouped them on the exact text. Neither mattered against seven
    posts, because nothing was long enough to truncate and no two posts
    spelled a concept differently. Against five hundred it produced a
    knowledge base whose slugs did not address the pages the site
    generated, and concepts split across pages that each listed only the
    posts that happened to spell them that way.
    """

    def test_a_long_label_is_bounded(self):
        from src.aggregation.consolidation import MAX_SLUG_LENGTH

        label = (
            "Azure Data Factory data flows, incremental loading, "
            "triggers, and scheduling for a medallion pipeline"
        )

        slug = slug_for(label, fallback="unknown")

        assert len(slug) <= MAX_SLUG_LENGTH
        assert not slug.endswith("-")

    def test_the_bound_is_the_same_one_the_site_uses(self):
        from src.aggregation.consolidation import MAX_SLUG_LENGTH
        from src.wiki.naming import MAX_SLUG_LENGTH as site_max

        assert MAX_SLUG_LENGTH == site_max

    def test_the_knowledge_base_and_the_site_agree_on_a_long_slug(self):
        from src.wiki.naming import slugify

        label = (
            "Azure Data Factory data flows, incremental loading, "
            "triggers, and scheduling for a medallion pipeline"
        )

        # The aggregator's slug is what lands in the knowledge base, and
        # the site's is what becomes a file name. A consumer following
        # the stored slug has to arrive at the page.
        assert _slug(label) == slugify(label)

    def test_case_and_punctuation_do_not_make_two_concepts(self):
        assert normalise_label("Delta Lake") == normalise_label(
            "delta-lake"
        )
        assert normalise_label("ACID Compliance") == normalise_label(
            "acid compliance"
        )

    def test_accents_fold_rather_than_vanish(self):
        # Two spellings of one label must land on one slug, or a topic
        # appears twice under different names.
        assert slug_for("Café", fallback="x") == slug_for(
            "Cafe", fallback="x"
        )

    def test_a_label_with_no_usable_characters_uses_the_fallback(self):
        from src.wiki.naming import slugify

        assert slug_for("...", fallback="unknown") == "unknown"
        assert slugify("...", fallback="untitled") == "untitled"

    def test_two_spellings_of_one_concept_consolidate_to_one_node(self):
        first = _post(
            "urn:li:activity:1",
            concepts=["Delta Lake", "time travel"],
        )
        second = _post(
            "urn:li:activity:2",
            concepts=["delta lake", "Time Travel"],
        )

        index = consolidate([first, second])

        names = {node.name.casefold() for node in index.concepts}

        assert len(index.concepts) == 2, [
            node.name for node in index.concepts
        ]
        assert "delta lake" in names
        assert "time travel" in names

        lake = next(
            node
            for node in index.concepts
            if node.name.casefold() == "delta lake"
        )

        # Both posts, because both meant the same concept.
        assert set(lake.post_ids) == {
            "urn:li:activity:1",
            "urn:li:activity:2",
        }
