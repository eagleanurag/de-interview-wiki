"""
The knowledge page: what a reader sees, and what they must not.

The page this replaces was a description of a capture. It was titled
``urn-li-saved-ffccf4f7771b97bd`` and opened with Platform, Captured,
Published, Author and Post ID, then a "Saved item" card, then a
"Classification" table, then the original post text, then a media
inventory with filenames.

Every one of those fields is still in the model and still written. What
changed is that none of them is what the page is *for*, and the tests
here split cleanly along that line:

* **Reader-facing.** The forbidden sections are gone, the title is
  human-readable, the hierarchy is navigation, and the capture record is
  behind a disclosure.
* **Data-facing.** Nothing was deleted. ``post.id``, ``saved_item``,
  ``source``, ``original_text``, ``media`` and ``classification`` are all
  still there, still populated, and still round-trip. Simplifying a page
  is not a licence to lose the record the page was built from.

The second group is the one worth having. A redesign that quietly drops
provenance looks identical on the page and costs the project its ability
to regenerate, audit and deduplicate.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

import pytest

from src.models import (
    AIAnalysis,
    Classification,
    InterviewQuestion,
    KnowledgePost,
    MediaItem,
    SavedItemProvenance,
    SourceInfo,
)
from src.wiki.knowledge import (
    SOURCE_EXCERPT_LIMIT,
    render_knowledge,
    sibling_pages,
)
from src.wiki.curriculum import (
    _looks_like_identifier,
    knowledge_title,
    placement_for,
)


#: Sections the reader must not see. Named exactly as the archive page
#: named them, so a regression is recognisable rather than a diff.
FORBIDDEN_SECTIONS = (
    "Source information",
    "Saved item",
    "Original source material",
    "Generated knowledge",
    "Classification",
    "Interview relevant",
    "Primary topic",
    "Secondary topics",
    "How it arrived",
    "Content matched by",
    "Capture match",
)


def make_post(
    identifier: str = "urn-li-saved-ffccf4f7771b97bd",
    *,
    summary: str = (
        "A data analyst summarizes a completed SQL bootcamp project "
        "exploring hotel booking data."
    ),
    primary_topic: str = "Hotel booking analysis with SQL",
    concepts: list[str] | None = None,
    questions: list[InterviewQuestion] | None = None,
    text: str = "Captured post body.",
    media: list[MediaItem] | None = None,
    author: str = "A Poster",
    url: str = "https://www.linkedin.com/feed/update/urn:li:activity:1",
) -> KnowledgePost:
    return KnowledgePost(
        id=identifier,
        source=SourceInfo(
            platform="linkedin",
            captured_at=datetime(2026, 10, 2, tzinfo=timezone.utc),
            capture_method="user_provided",
            url=url,
            author=author,
        ),
        saved_item=SavedItemProvenance(saved_item_id="urn:li:saved:ff:cc"),
        original_text=text,
        media=media if media is not None else [
            MediaItem(type="image", path="media/activity_7326_slide_0.jpg")
        ],
        ai_analysis=AIAnalysis(
            summary=summary,
            topics=["SQL data analysis", "Hotel booking analytics"],
            subtopics=["Hotel occupancy analysis"],
            concepts=concepts
            if concepts is not None
            else ["Home-city booking frequency", "Monthly hotel fulfilment"],
        ),
        interview_questions=questions
        if questions is not None
        else [
            InterviewQuestion(
                question="How would you reproduce the top five customers?",
                type="scenario",
                difficulty="medium",
                answer=(
                    "Rank customers by their booking frequency within each "
                    "home city, then take the top five."
                ),
                answer_source="ai_enriched",
            )
        ],
        classification=Classification(
            domain="Data Analytics",
            primary_topic=primary_topic,
            secondary_topics=["Customer behavior analytics"],
            interview_relevant=True,
        ),
    )


@pytest.fixture
def page() -> str:
    return render_knowledge(make_post(), page="posts/sample.html")


# ---------------------------------------------------------------------
# Nothing internal is visible
# ---------------------------------------------------------------------


@pytest.mark.parametrize("section", FORBIDDEN_SECTIONS)
def test_no_capture_section_is_rendered(page, section):
    """
    Checked against the whole page now, not just the visible part.

    It used to be checked only against the text before the disclosure,
    on the reasoning that "Saved item" and "How it arrived" were fine
    behind a click. They are not: a capture date, a capture method, a
    saved-item id and the post's own key are all either identifiers or
    metadata about how the archive was built, and none of them helps a
    reader revise. They were removed from the disclosure too.
    """

    assert section not in page


def test_the_source_disclosure_says_what_the_source_was(page):
    """
    The replacement for the capture record.

    It has to say something. An empty disclosure would have dropped
    provenance from the reader's view, which is the opposite of what
    removing identifiers was for.
    """

    record = page[page.index('<details class="provenance">') :]

    assert "Source" in record
    assert "LinkedIn post" in record


def test_no_identifier_or_capture_metadata_survives_anywhere_on_the_page(page):
    """
    The whole page, including the disclosure.

    Four things were removed together because they are the same kind of
    thing: the Reference row printed ``post.id``, the Saved item row
    printed ``saved_item_id``, and Captured and How it arrived printed a
    capture timestamp and a capture method. A reader should not have to
    know what any of them is.

    Checked against the rendered text rather than the markup, because the
    source link is a real LinkedIn permalink and its href legitimately
    contains ``urn:li:activity:...``. That is the platform's own address
    for the post, not this project's internal key, and the brief allows
    a source link where one exists.
    """

    import re as _re

    text = _re.sub(r"<[^>]+>", " ", page)
    text = _re.sub(r"\s+", " ", text)

    for gone in (
        "urn-li-",
        "urn:li:",
        "activity_",
    ):
        assert gone not in text, f"{gone!r} is still visible on the page"

    # The capture-record labels are asserted as definition-list terms.
    # Checking the bare word would match the post's own body -- the
    # fixture's text is "Captured post body." -- so this looks for the
    # markup that actually renders a field label.
    labels = _re.findall(r"<dt>(.*?)</dt>", page)

    for label in labels:
        assert label in ("Source", "Author"), (
            f"{label!r} is a capture-record field that should be gone"
        )

    # And nothing carries the capture method as data.
    assert "data-capture-method" not in page

    # And no filename, in the markup either.
    assert "slide_0.jpg" not in page


def test_the_title_is_never_the_identifier(page):
    heading = page.split("<h1>")[1].split("</h1>")[0]

    assert "urn-li-saved" not in heading
    assert not _looks_like_identifier(heading)


def test_no_identifier_is_printed_before_the_disclosure(page):
    source = page.index('<details class="provenance">')

    assert "urn-li-saved" not in page[:source]


def test_the_original_text_is_still_reachable(page):
    """
    What the disclosure *is* for.

    The original post text and the link back to it are the two things a
    reader checking a claim actually needs, and both are permitted
    content rather than archive metadata.
    """

    record = page[page.index('<details class="provenance">') :]

    assert "What the post said" in record


def test_no_activity_id_or_filename_is_visible_before_the_disclosure(page):
    head = page[: page.index('<details class="provenance">')]

    assert "activity_" not in head
    assert ".jpg" not in head


def test_the_page_does_not_claim_a_person_reviewed_it(page):
    assert "reviewed by a person" not in page
    assert "human-reviewed" not in page


# ---------------------------------------------------------------------
# Titles
# ---------------------------------------------------------------------


def test_the_title_names_the_subject_and_the_subject_matter():
    title = knowledge_title(make_post())

    # Human-readable, positioned in the hierarchy, and the subject is
    # not said twice.
    assert title.endswith("Hotel booking analysis")
    assert "with SQL" not in title

    assert "urn" not in title.lower()


@pytest.mark.parametrize(
    "primary",
    [
        "SQL",
        "urn-li-saved-abcdef",
        "activity_7326889752870170625_slide_0.jpg",
        "POST-029",
        "",
    ],
)
def test_an_unusable_topic_never_becomes_the_title(primary):
    title = knowledge_title(make_post(primary_topic=primary))

    assert not _looks_like_identifier(title)
    assert "urn-li-saved" not in title
    assert "POST-029" not in title


def test_a_post_with_no_analysis_still_gets_a_title():
    post = make_post()
    post.ai_analysis = None

    title = knowledge_title(post)

    assert title
    assert not _looks_like_identifier(title)


# ---------------------------------------------------------------------
# Hierarchy as navigation
# ---------------------------------------------------------------------


def test_the_breadcrumb_is_three_hops_upwards(page):
    crumb = re.search(
        r'<nav class="breadcrumb".*?</nav>', page, re.S
    ).group(0)

    assert "Home" in crumb
    assert "aria-label=\"Breadcrumb\"" in crumb

    for reference in re.findall(r'href="([^"]+)"', crumb):
        assert reference.startswith("../")


def test_the_breadcrumb_reaches_the_posts_own_subtopic(page):
    major, subtopic = placement_for(make_post())

    crumb = re.search(
        r'<nav class="breadcrumb".*?</nav>', page, re.S
    ).group(0)

    assert esc(major.name) in crumb
    assert subtopic.name in crumb


def esc(value: str) -> str:
    """The escaping the renderer applies, so the test checks the page."""

    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def test_the_subtopic_appears_as_a_position_not_as_a_chip(page):
    """
    "Topics: ... Subtopics: ... Concepts: ..." as rows was the same
    information as four database columns. A subtopic is somewhere to go,
    so it is a link, not a label.
    """

    assert "Hotel occupancy analysis" not in page


def test_concepts_are_still_listed(page):
    assert "Home-city booking frequency" in page
    assert "Monthly hotel fulfilment" in page


# ---------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------


def test_questions_are_closed_until_asked_for(page):
    assert '<details class="question">' in page
    assert "<summary>" in page

    # One disclosure per question, balanced.
    assert page.count('<details class="question">') == page.count(
        "</details>"
    ) - page.count('<details class="provenance">')


def test_the_answer_is_in_the_page_not_fetched(page):
    assert "Rank customers by their booking frequency" in page


def test_a_quoted_passage_is_labelled_as_one():
    """
    Not an answer written to the question.

    A question recovered from a slide comes with whatever text was on
    that slide. Shown without saying so, a reader will quote the slide's
    wording as though it were a considered explanation.
    """

    post = make_post(
        questions=[
            InterviewQuestion(
                question="What is an INNER JOIN?",
                type="theory",
                difficulty="easy",
                answer="Rows with matching values in both tables.",
                answer_source="source_excerpt",
            )
        ]
    )

    page = render_knowledge(post, page="posts/sample.html")

    assert "Quoted from the slide" in page
    assert "rather than written as an answer" in page


def test_a_written_answer_is_not_labelled_as_a_quotation():
    post = make_post(
        questions=[
            InterviewQuestion(
                question="What is an INNER JOIN?",
                type="theory",
                difficulty="easy",
                answer=(
                    "An inner join keeps the rows whose join key matches "
                    "in both relations."
                ),
                answer_source="ai_enriched",
            )
        ]
    )

    page = render_knowledge(post, page="posts/sample.html")

    assert "Quoted from the slide" not in page


def test_an_empty_answer_says_so_rather_than_inventing_one():
    post = make_post(
        questions=[
            InterviewQuestion(
                question="What is a window function?",
                type="theory",
                difficulty="easy",
                answer="",
            )
        ]
    )

    page = render_knowledge(post, page="posts/sample.html")

    assert "No answer was written" in page


def test_a_code_example_is_set_apart_when_the_answer_has_one():
    """
    Shown when the answer contains code, which is what "where supported
    by source knowledge" means here.
    """

    post = make_post(
        questions=[
            InterviewQuestion(
                question="Write a window function query.",
                type="coding",
                difficulty="medium",
                answer=(
                    "Rank rows within each city."
                    "\n\n```sql\nSELECT city, id, ROW_NUMBER() "
                    "OVER (PARTITION BY city ORDER BY id) AS rn\n"
                    "FROM bookings;\n```"
                ),
                answer_source="ai_enriched",
            )
        ]
    )

    page = render_knowledge(post, page="posts/sample.html")

    assert "<pre><code>" in page
    assert "ROW_NUMBER()" in page


def test_the_answer_view_never_shows_a_transcription(page):
    """
    An OCR is not an answer. It is full of misread punctuation and
    half-spelled identifiers, and presenting it as an example would
    teach wrong code with more confidence than presenting nothing.
    """

    for marker in ("ocr_status", "ocr_quality", "image_ocr", "extracted_text"):
        assert marker not in page


def test_no_questions_says_so_explicitly():
    page = render_knowledge(
        make_post(questions=[]), page="posts/sample.html"
    )

    assert "No interview questions were generated" in page


# ---------------------------------------------------------------------
# Source, secondary
# ---------------------------------------------------------------------


def test_the_source_is_one_line_before_the_disclosure(page):
    line = re.search(r'<div class="source-line">.*?</div>', page, re.S).group(0)

    assert "LinkedIn" in line
    assert "1 contributing post" in line

    assert page.index("source-line") < page.index(
        '<details class="provenance">'
    )


def test_the_capture_record_is_collapsed(page):
    assert "<summary>View source</summary>" in page

    body = page.split('<details class="provenance">')[0]

    assert "Platform" not in body
    assert "Captured" not in body


def test_a_missing_link_is_stated_rather_than_left_silent():
    page = render_knowledge(
        make_post(url="javascript:alert(1)"), page="posts/sample.html"
    )

    assert "No source URL was recorded" in page

    # And the dangerous scheme appears nowhere at all.
    assert "javascript:" not in page


def test_attached_images_are_not_described_at_all(page):
    """
    Used to assert "1 image" appeared, behind the disclosure, with no
    filenames.

    The count has gone too. It was already better than an inventory, but
    it is still archive metadata: a reader revising cannot act on "1
    image", and the transcriptions behind it are not a study aid. What
    was attached stays in the model and in the OCR index.
    """

    head = page[: page.index('<details class="provenance">')]
    detail = page[page.index('<details class="provenance">') :]

    for text in (head, detail):
        assert "slide_0.jpg" not in text
        assert "activity_7326_slide_0.jpg" not in text
        assert "Attached images" not in text
        assert "1 image" not in text


def test_a_long_capture_is_excerpted_rather_than_reproduced():
    page = render_knowledge(
        make_post(text="word " * 4000), page="posts/sample.html"
    )

    assert "Excerpt truncated for readability" in page

    body = page.split('class="prose source-text"')[1]
    excerpt = body.split("</div>")[0]

    assert len(excerpt) < SOURCE_EXCERPT_LIMIT + 200


# ---------------------------------------------------------------------
# Related
# ---------------------------------------------------------------------


def test_related_pages_are_chosen_by_shared_concept():
    near = make_post(
        "urn-li-saved-near",
        concepts=["Home-city booking frequency", "Something else"],
    )
    far = make_post(
        "urn-li-saved-far",
        concepts=["Completely unrelated subject matter"],
    )
    self_post = make_post(concepts=["Home-city booking frequency"])

    found = sibling_pages(
        self_post, [self_post, near, far], ["a", "b", "c"]
    )

    targets = [target for _label, target in found]

    assert "posts/b.html" in targets
    assert "posts/c.html" not in targets
    assert "posts/a.html" not in targets


def test_a_post_with_no_concepts_has_no_related_pages():
    post = make_post()
    post.ai_analysis = None

    assert sibling_pages(post, [post], ["a"]) == []


# ---------------------------------------------------------------------
# Nothing was deleted
# ---------------------------------------------------------------------


def test_every_field_the_page_stopped_showing_is_still_on_the_model():
    """
    The whole point of separating the two.

    A redesign that drops provenance looks identical on the page and
    costs the project its ability to regenerate, audit and deduplicate.
    """

    post = make_post()

    assert post.id == "urn-li-saved-ffccf4f7771b97bd"
    assert post.saved_item is not None
    assert post.saved_item.saved_item_id == "urn:li:saved:ff:cc"
    assert post.source.platform == "linkedin"
    assert post.source.author == "A Poster"
    assert post.original_text == "Captured post body."
    assert len(post.media) == 1
    assert post.media[0].path.endswith(".jpg")
    assert post.classification.domain == "Data Analytics"
    assert post.classification.primary_topic == "Hotel booking analysis with SQL"
    assert post.classification.interview_relevant is True
    assert post.ai_analysis.concepts


def test_the_post_still_round_trips_through_the_canonical_form():
    post = make_post()

    payload = json.loads(post.model_dump_json())

    restored = KnowledgePost.model_validate(payload)

    assert restored.id == post.id
    assert restored.saved_item.saved_item_id == "urn:li:saved:ff:cc"
    assert restored.original_text == post.original_text
    assert restored.media[0].path == post.media[0].path
    assert restored.classification.domain == "Data Analytics"


def test_ocr_provenance_is_still_carried_on_the_media_item():
    post = make_post(
        media=[
            MediaItem(
                type="image",
                path="media/activity_7326_slide_0.jpg",
                extracted_text="Slide text.",
                ocr_status="AVAILABLE",
                ocr_quality="readable",
                source_kind="image_ocr",
            )
        ]
    )

    # The transcription is not rendered as an answer, but it is not
    # thrown away either.
    assert post.media[0].extracted_text == "Slide text."

    page = render_knowledge(post, page="posts/sample.html")

    assert "Slide text." not in page


# ---------------------------------------------------------------------
# Escaping
# ---------------------------------------------------------------------


def test_hostile_content_in_every_field_is_escaped():
    hostile = '<script>alert("x")</script> & <b>'

    post = make_post(
        summary=f"Summary {hostile}",
        primary_topic=f"Topic {hostile}",
        concepts=[f"Concept {hostile}"],
        text=f"Body {hostile}",
        questions=[
            InterviewQuestion(
                question=f"Question {hostile}",
                type="theory",
                difficulty="easy",
                answer=f"Answer {hostile}",
            )
        ],
    )

    page = render_knowledge(post, page="posts/sample.html")

    assert "<script>alert" not in page
    assert "<b>" not in page
    assert "&lt;script&gt;" in page
    assert "&amp;" in page


def test_no_local_path_reaches_the_page():
    post = make_post(text="Saved at C:\\Users\\Someone\\notes.txt")

    page = render_knowledge(post, page="posts/sample.html")

    assert not re.search(r"(?i)[A-Z]:\\Users", page)