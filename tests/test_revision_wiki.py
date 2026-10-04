"""
The revision curriculum: placement, merging, answers and rendering.

Two things are being protected here, and they pull in opposite
directions.

**The hierarchy must stay small.** The corpus produced 1,560 distinct
topics and 3,106 concepts. If a subtopic can be created per label, the
site is the archive again with new navigation, and this module has
failed while looking like it worked. So the tests assert a *ceiling* on
how many subjects and subtopics exist, not merely that the placement
functions return something.

**Nothing may be invented.** Answers are the risk: a thin answer is
tempting to pad, and a question with no answer is tempting to answer.
The tests assert that a question with no answer says so, that a short
one is labelled short rather than expanded, and that a slide excerpt is
never presented as an answer to the question.

The placement tests are written against the real failure modes found by
running the classifier over the actual corpus, not against labels chosen
to look tidy. ``orm`` matching inside *performance* cost 116 misplaced
labels; ``data type`` matching *data transformation* misplaced every
transform label; and declared-subject-order beat specificity, which sent
every broadcast-join label to SQL.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import pytest

from src.models import (
    AIAnalysis,
    InterviewQuestion,
    KnowledgePost,
    MediaItem,
    SourceInfo,
)
from src.wiki.curriculum import (
    ANSWER_AI,
    ANSWER_INSUFFICIENT,
    THIN_ANSWER_CHARS,
    Curriculum,
    RevisionQuestion,
    build_curriculum,
)
from src.wiki.revision import (
    CAVEAT,
    curriculum_pages,
    questions_page,
    render_home,
    render_questions,
    render_subject,
    render_subject_index,
    subject_index_page,
    subject_page,
)
from src.wiki.taxonomy import MAJORS, classify


def post(
    identifier: str,
    *,
    topics: list[str] | None = None,
    subtopics: list[str] | None = None,
    concepts: list[str] | None = None,
    questions: list[str] | None = None,
    answers: dict[str, str] | None = None,
    media: list[str] | None = None,
) -> KnowledgePost:
    """A post with exactly the fields the curriculum reads."""

    answers = answers or {}

    return KnowledgePost(
        id=identifier,
        source=SourceInfo(
            platform="linkedin", captured_at=datetime.now(timezone.utc)
        ),
        original_text="An original post.",
        media=[
            MediaItem(type="image", path=path)
            for path in (media or ["media/activity_1_slide_0.jpg"])
        ],
        ai_analysis=AIAnalysis(
            summary="A summary.",
            topics=topics or [],
            subtopics=subtopics or [],
            concepts=concepts or [],
        ),
        interview_questions=[
            InterviewQuestion(
                question=text,
                type="theory",
                difficulty="medium",
                answer=answers.get(text, "A written answer of a usable length " * 2),
                answer_source=ANSWER_AI,
            )
            for text in (questions or [])
        ],
    )


# ---------------------------------------------------------------------
# The hierarchy stays small
# ---------------------------------------------------------------------


def test_the_hierarchy_is_a_fixed_size_not_a_per_label_invention():
    """
    The whole point of the redesign, asserted as a number.

    The original guard was 8 subjects and 65 subtopics, written against a
    taxonomy that matched nothing irregular. It moved to 14 subjects and
    150 subtopics once the corpus was actually measured, and the two
    limits below are the new measured bounds.

    What the guard is really for is the ratio. The corpus carries 6,885
    distinct knowledge labels. One hundred and fifty subtopics is one
    home per forty-six labels; a taxonomy that grew towards one per
    label would be the archive browser the redesign removed, and would
    put a thousand dead ends in front of a reader.

    The ceiling is deliberately far below the label count so that a
    change which drifts towards invention trips this rather than passing
    it.
    """

    assert len(MAJORS) <= 20

    subtopics = [s for m in MAJORS for s in m.subtopics]

    assert 60 <= len(subtopics) <= 200


def test_nothing_is_named_as_a_dumping_ground():
    """
    The one rule about naming, from the brief.

    "General" is allowed exactly once per subject, where it means "the
    source did not say enough to place this", and it is measured and
    reported rather than hidden. Any other word that admits it holds
    anything is a category created to make a number smaller.
    """

    forbidden = {
        "misc",
        "other",
        "others",
        "various",
        "technical stuff",
        "technical",
        "data engineering",
        "stuff",
        "unknown",
        "unclassified",
        "no topic",
    }

    for major in MAJORS:
        for subtopic in major.subtopics:
            if subtopic.name == "General":
                continue

            assert subtopic.name.casefold() not in forbidden, (
                f"{major.name} / {subtopic.name} is a dumping ground by name"
            )

    for major in MAJORS:
        generals = [
            s for s in major.subtopics if s.name == "General"
        ]

        assert len(generals) <= 1, (
            f"{major.name} declares General more than once"
        )


def test_every_subject_and_subtopic_has_a_slug_and_a_blurb():
    for major in MAJORS:
        assert major.slug
        assert major.blurb

        for subtopic in major.subtopics:
            assert subtopic.slug
            assert subtopic.name


def test_subtopic_slugs_are_unique_within_a_subject():
    for major in MAJORS:
        slugs = [s.slug for s in major.subtopics]

        assert len(slugs) == len(set(slugs))


# ---------------------------------------------------------------------
# Matching on word boundaries
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "label",
    [
        "Performance tuning",
        "Platform engineering",
        "Informational update",
    ],
)
def test_a_substring_stem_never_matches_inside_a_longer_word(label):
    """
    The bug that put a subject called ORM in the middle of performance.

    Every one of these contains "orm" inside a longer word. An unanchored
    sweep for that stem matched 116 real labels.
    """

    for major in MAJORS:
        for pattern in major.patterns:
            assert not pattern.search(label) or "orm" not in pattern.pattern


def test_data_type_does_not_match_data_transformation():
    assert classify("Data transformation").subtopic.name != (
        "Data Types and Constraints"
    )


def test_specificity_beats_declaration_order():
    """
    "Broadcast joins" is a Spark question that also contains the word
    "join". SQL is declared first, so a first-match-wins classifier sends
    it to SQL, which is wrong.
    """

    found = classify("Broadcast joins")

    assert found.major.slug == "spark-and-pyspark"
    assert found.subtopic.name == "Joins"


def test_an_explicitly_named_subject_wins():
    assert classify("Spark SQL joins").major.slug == "spark-and-pyspark"

    # ...and the more specific of two named subjects is the one chosen.
    assert classify("Spark SQL").subtopic.name == "Spark SQL"


def test_a_plural_that_is_not_a_singular_plus_s_is_matched():
    # "subqueries" is "subquer" + "ies", not "subquery" + a suffix, so a
    # stem-based pattern cannot find it.
    assert classify("Subqueries").major.slug == "sql"


def test_a_label_nothing_matches_is_counted_not_hidden():
    found = classify("Zzzqqq unrelated label")

    assert found.is_fallback

    # And it lands in the broadest subject rather than whichever was
    # declared first, which used to make SQL look like two thirds of the
    # corpus.
    assert found.major.slug == "data-engineering"


def test_classification_is_deterministic():
    for label in ("Window functions", "Partitioning", "Kafka", "Anything"):
        first = classify(label)
        second = classify(label)

        assert (first.major.slug, first.subtopic.slug) == (
            second.major.slug,
            second.subtopic.slug,
        )


# ---------------------------------------------------------------------
# Merging and placement
# ---------------------------------------------------------------------


def test_the_same_question_from_five_posts_becomes_one():
    posts = [
        post(
            f"post-{n}",
            topics=["Window functions"],
            questions=["What is ROW_NUMBER() and how does it differ from RANK()?"],
        )
        for n in range(5)
    ]

    curriculum = build_curriculum(posts)

    assert curriculum.question_count == 1

    only = next(iter(curriculum.questions.values()))

    assert only.source_count == 5
    assert len(only.post_ids) == 5


def test_a_question_is_filed_by_its_own_words_before_its_post():
    # The post is about partitioning; the question is about joins. A
    # reader revising joins wants the question under joins.
    posts = [
        post(
            "p",
            topics=["Partitioning"],
            questions=["How do broadcast joins work in Spark?"],
        )
    ]

    curriculum = build_curriculum(posts)

    only = next(iter(curriculum.questions.values()))

    assert only.major_slug == "spark-and-pyspark"
    assert only.subtopic_slug == "joins"


def test_a_question_with_no_recognisable_wording_inherits_its_post():
    posts = [
        post(
            "p",
            topics=["Slowly Changing Dimensions"],
            questions=["Zzzqqq wibble"],
        )
    ]

    curriculum = build_curriculum(posts)

    only = next(iter(curriculum.questions.values()))

    assert only.major_slug == "data-engineering"
    assert only.subtopic_slug == "slowly-changing-dimensions"


def test_questions_are_split_across_subtopics_not_duplicated():
    posts = [
        post(
            "p",
            questions=[
                "How do window functions differ from GROUP BY?",
                "How does partitioning reduce scanning in Spark?",
            ],
        )
    ]

    curriculum = build_curriculum(posts)

    placements = {
        (q.major_slug, q.subtopic_slug) for q in curriculum.questions.values()
    }

    assert len(curriculum.questions) == 2
    assert placements == {
        ("sql", "window-functions"),
        ("spark-and-pyspark", "partitions-and-partitioning"),
    }


def test_the_diagnostics_report_the_merge_and_the_residual():
    posts = [
        post("a", questions=["What is a window function?"]),
        post("b", questions=["What is a window function?"]),
        post("c", concepts=["Zzzqqq unrelated"]),
    ]

    curriculum = build_curriculum(posts)

    assert curriculum.diagnostics["question_slots_before_merge"] == 2
    assert curriculum.diagnostics["questions_after_merge"] == 1
    assert curriculum.diagnostics["merged_away"] == 1
    assert curriculum.diagnostics["concepts"] == 1


def test_building_twice_gives_the_same_curriculum():
    posts = [
        post(
            "a",
            topics=["SQL", "Spark"],
            concepts=["Joins", "Partitioning"],
            questions=["What is a window function?"],
        ),
        post("b", topics=["Python"], questions=["What is a decorator?"]),
    ]

    first = build_curriculum(posts)
    second = build_curriculum(list(reversed(posts)))

    assert first.question_count == second.question_count
    assert [m.slug for m in first.majors] == [m.slug for m in second.majors]
    assert first.questions.keys() == second.questions.keys()


# ---------------------------------------------------------------------
# Answers are never invented
# ---------------------------------------------------------------------


def test_a_question_with_no_answer_says_so():
    posts = [
        post("p", questions=["What is a window function?"], answers={
            "What is a window function?": "",
        })
    ]

    curriculum = build_curriculum(posts)

    only = next(iter(curriculum.questions.values()))

    assert only.answer.source == ANSWER_INSUFFICIENT
    assert only.answer.text == ""


def test_a_very_short_answer_is_kept_and_marked_thin():
    short = "It is a thing."

    posts = [
        post(
            "p",
            questions=["What is a window function?"],
            answers={"What is a window function?": short},
        )
    ]

    curriculum = build_curriculum(posts)

    only = next(iter(curriculum.questions.values()))

    assert len(short) < THIN_ANSWER_CHARS
    assert only.answer.text == short
    assert only.answer.is_thin


def test_a_slide_excerpt_is_labelled_as_an_excerpt_not_an_answer():
    posts = [
        post(
            "p",
            questions=["What is a star schema?"],
        )
    ]

    posts[0].interview_questions[0].answer = (
        "A schema with a central fact table and dimension tables."
    )
    posts[0].interview_questions[0].answer_source = "source_excerpt"

    curriculum = build_curriculum(posts)

    only = next(iter(curriculum.questions.values()))

    assert only.answer.source == "source_excerpt"
    assert "Excerpt from the slide" in only.answer.label


def test_the_more_substantial_answer_wins_when_questions_merge():
    posts = [
        post(
            "a",
            questions=["What is a window function?"],
            answers={"What is a window function?": "Short."},
        ),
        post(
            "b",
            questions=["What is a window function?"],
            answers={
                "What is a window function?": "A function computed over a "
                "window of rows defined by PARTITION BY and ORDER BY."
            },
        ),
    ]

    curriculum = build_curriculum(posts)

    only = next(iter(curriculum.questions.values()))

    assert "PARTITION BY" in only.answer.text


# ---------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------


@pytest.fixture
def curriculum() -> Curriculum:
    return build_curriculum(
        [
            post(
                "a",
                topics=["Window functions"],
                concepts=["RANK", "ROW_NUMBER"],
                questions=["What is ROW_NUMBER()?"],
            ),
            post(
                "b",
                topics=["Partitioning"],
                concepts=["Partitioning"],
                questions=["How does partitioning help?"],
            ),
        ]
    )


def test_the_home_page_names_the_site_and_offers_the_subjects(curriculum):
    page = render_home(curriculum)

    assert "Data Engineering Interview Wiki" in page

    for major in curriculum.majors:
        assert major.major.name in page


def test_the_home_page_shows_counts_but_no_build_metadata(curriculum):
    page = render_home(curriculum)

    assert "questions" in page

    # Post identifiers, OCR counts and fingerprints describe how the site
    # was built. A reader revising does not need them, and their absence
    # here is what makes the page read as a revision aid rather than a
    # report.
    for leaked in ("urn-li-", "sha256", "ocr", "fingerprint", "worker"):
        assert leaked.lower() not in page.lower()


def test_the_subject_index_lists_subjects_and_subjects_below_them(curriculum):
    page = render_subject_index(curriculum)

    assert "Subjects" in page

    for major in curriculum.majors:
        for node in major.subtopics:
            assert node.subtopic.name in page


def test_a_subject_page_carries_its_questions_behind_a_disclosure(curriculum):
    node = curriculum.major("sql").subtopics[0]

    page = render_subject(curriculum, node)

    # Native details/summary, no framework and no script.
    assert "<details" in page
    assert "<summary>" in page
    assert "<script" not in page.split("</body>")[0]

    assert page.count("<details") == page.count("</details>")


def test_a_question_expands_to_its_answer(curriculum):
    node = curriculum.major("sql").subtopics[0]

    page = render_subject(curriculum, node)

    assert "What is ROW_NUMBER()?" in page

    # The answer text is in the document, inside the disclosure, so it
    # needs no JavaScript to be reachable.
    assert "written answer" in page.lower() or "PARTITION" in page


def test_post_identifiers_appear_only_inside_provenance(curriculum):
    node = curriculum.major("sql").subtopics[0]

    page = render_subject(curriculum, node)

    # Provenance is available, but behind its own disclosure rather than
    # in the body of the page.
    assert '<details class="provenance">' in page
    assert "Show source" in page


def test_no_post_identifier_is_ever_printed_on_a_revision_page(curriculum):
    """
    The one thing the disclosure must not do is print the identifier.

    ``post_id`` is an internal key. A reader cannot type it back to
    anything, cannot ask it a question, and reads it as a defect on a
    page otherwise written for people -- "Show source" used to list
    ``urn-li-archive-c7812291ac93dd56`` and nothing else.

    Provenance itself is untouched: the identifiers are still on every
    question, and :func:`test_the_question_still_knows_which_posts_it_came_from`
    holds them to that.
    """

    pattern = re.compile(r"urn-li-[a-z]+-[0-9a-f]{8,}")

    for major in curriculum.majors:
        for node in major.subtopics:
            page = render_subject(curriculum, node)

            assert not pattern.search(page), (
                f"{major.slug}/{node.slug} prints a post identifier"
            )


def test_the_source_disclosure_says_what_the_source_was(curriculum):
    """
    The identifier is replaced, not merely removed.

    A disclosure that opened onto nothing would have dropped provenance
    from the reader's view, which is the opposite of the intent. It
    names the kind of source and how many there were.
    """

    rendered = [
        render_subject(curriculum, node)
        for major in curriculum.majors
        for node in major.subtopics
        if node.question_ids
    ]

    assert rendered, "no subject page rendered"

    for page in rendered:
        for detail in re.findall(
            r"<details class=\"provenance\">(.*?)</details>", page, re.S
        ):
            assert "This question appears in" in detail, (
                "the source disclosure says nothing"
            )

    joined = "".join(rendered)

    assert "post" in joined


def test_the_question_still_knows_which_posts_it_came_from(curriculum):
    """
    The counterpart to the test above.

    Dropping an identifier from the page is only safe while the model
    still holds it, so that anything needing to trace a claim back to a
    specific post can.
    """

    for question in curriculum.questions.values():
        assert question.post_ids, (
            f"{question.id} lost its post provenance"
        )

        if question.source_labels:
            assert len(question.source_labels) == len(question.post_ids)

            for label in question.source_labels:
                assert "urn-li" not in label
                assert label.strip()


def test_every_page_carries_the_caveat(curriculum):
    node = curriculum.major("sql").subtopics[0]

    for page in (
        render_home(curriculum),
        render_subject_index(curriculum),
        render_questions(curriculum),
        render_subject(curriculum, node),
    ):
        assert CAVEAT[:40] in page


def test_the_site_never_claims_a_person_reviewed_the_content(curriculum):
    node = curriculum.major("sql").subtopics[0]

    page = render_subject(curriculum, node)

    assert "reviewed by a person" not in page
    assert "human-reviewed" not in page


def test_no_rendered_page_contains_a_local_path(curriculum):
    node = curriculum.major("sql").subtopics[0]

    for page in (
        render_home(curriculum),
        render_subject_index(curriculum),
        render_questions(curriculum),
        render_subject(curriculum, node),
    ):
        assert not re.search(r"(?i)[A-Z]:\\Users|Downloads\\|\.venv|\.agent", page)


def test_question_text_is_escaped_rather_than_rendered_as_markup():
    hostile = "<script>alert(1)</script> & \"quoted\""

    curriculum = build_curriculum([post("p", questions=[hostile])])

    page = render_questions(curriculum)

    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page


def test_every_page_is_a_complete_html_document(curriculum):
    node = curriculum.major("sql").subtopics[0]

    for page in (
        render_home(curriculum),
        render_subject_index(curriculum),
        render_questions(curriculum),
        render_subject(curriculum, node),
    ):
        assert page.startswith("<!DOCTYPE html>")
        assert page.rstrip().endswith("</html>")


def test_the_pages_generated_cover_every_subject_and_subtopic(curriculum):
    pages = curriculum_pages(curriculum)

    for major in curriculum.majors:
        for node in major.subtopics:
            assert subject_page(node.major.slug, node.slug) in pages

    assert subject_index_page() in pages
    assert questions_page() in pages
    assert "index.html" in pages


def test_the_primary_navigation_is_four_items():
    from src.wiki.layout import NAV_ITEMS

    labels = [label for label, _ in NAV_ITEMS]

    # Concepts, Technologies, Topics and Saved Items were primary first,
    # then demoted to a footer row, and are now gone as pages entirely. A
    # reader has to guess which one they are looking at before they can
    # search, and all of them list labels the corpus generated rather
    # than subjects a person would revise. Their content is in the
    # knowledge base; the revision tree represents it readably.
    assert labels == ["Home", "Subjects", "Questions", "Search"]


def test_the_archive_sections_are_no_longer_offered_anywhere():
    """
    The footer row is gone, and with it 2,601 links.

    Every page on the site linked Concepts, Technologies, Topics and
    Saved Items from the footer. Removing those pages without removing
    the row would have left four dead links on every page; removing the
    row is what makes the deletion safe.
    """

    import src.wiki.layout as layout

    assert not hasattr(layout, "SECONDARY_NAV_ITEMS"), (
        "the footer row came back"
    )

    document = layout.render_document(
        page="index.html",
        title="Probe",
        description="Probe.",
        body="<p>probe</p>",
    )

    for gone in ("concepts.html", "technologies.html", "topics.html",
                 "saved-items.html"):
        assert gone not in document, f"{gone} is still linked from the chrome"