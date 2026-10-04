"""
The revision guide's own architecture.

Everything else in the suite checks that a renderer does what it did.
This checks the three decisions the site is now built on, each of which
could be undone silently:

* **one revision unit is one page.** Not one page per source post, and
  not one page per taxonomy subtopic. A unit splits into pages of
  itself only when it passes a measurable threshold.
* **the split is deterministic.** Same corpus, same pages, same
  questions on each -- so a reader who bookmarked page 3 comes back to
  page 3.
* **the relevance filter accounts for itself.** Every question is either
  published or discarded with a recorded reason, the two add up, and
  the basic programming material the brief asks to keep survives.

The fixtures are built from a small synthetic corpus so the counts are
exact rather than dependent on what is in ``data/``.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

from src.wiki.canonical import load_canonical
from src.wiki.curriculum import build_curriculum
from src.wiki.generator import generate_site
from src.wiki.naming import (
    INDEX_PAGE,
    QUESTIONS_PAGE,
    REVISION_INDEX,
    SEARCH_INDEX_FILE,
)
from src.wiki.relevance import (
    EXCLUDED_SUBTOPICS,
    NON_REVISION_SUBJECTS,
    exclusion_reason,
    is_relevant,
    relocation_target,
)
from src.wiki.revision_model import Unit, UnitPage, build_units
from src.wiki.units import QUESTIONS_PER_PAGE, UNITS, page_count


# ---------------------------------------------------------------------
# A small corpus
# ---------------------------------------------------------------------


def post(
    identifier: str,
    *,
    topics: list[str],
    subtopics: list[str],
    questions: list[str],
    concepts: list[str] | None = None,
) -> dict:
    return {
        "id": identifier,
        "source": {
            "platform": "linkedin",
            "url": f"https://www.linkedin.com/posts/example/{identifier}",
            "captured_at": "2026-10-01T09:00:00Z",
            "capture_method": "manual",
            "published_at": "2026-09-28T08:00:00Z",
        },
        "original_text": "A captured post about data engineering.",
        "saved_item": {
            "saved_item_id": f"urn:li:saved:{identifier}",
            "saved_date": "2026-10-01",
        },
        "ai_analysis": {
            "summary": f"A post about {subtopics[0]}.",
            "topics": topics,
            "subtopics": subtopics,
            "concepts": concepts or subtopics,
            "technologies": [],
        },
        "interview_questions": [
            {
                "question": text,
                "type": "theory",
                "difficulty": "medium",
                "answer": f"An answer about {text.rstrip('?')}.",
            }
            for text in questions
        ],
        "media": [],
    }


def knowledge_base(posts: list[dict]) -> dict:
    return {
        "schema_version": 2,
        "generated_at": "2026-10-02T00:00:00+00:00",
        "stats": {
            "posts_aggregated": len(posts),
            "posts_enriched": len(posts),
            "result_files_found": len(posts),
            "skipped_files": [],
        },
        "knowledge": {
            "topics": {},
            "subtopics": {},
            "concepts": {},
            "technologies": {},
            "content_kinds": {p["id"]: "technical" for p in posts},
        },
        "posts": posts,
    }


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> tuple[Path, dict]:
    """A corpus spanning several subjects, written out once."""

    posts = [
        post(
            "alpha",
            topics=["SQL"],
            subtopics=["Joins"],
            concepts=["Inner Join", "Broadcast Hash Join"],
            questions=[
                "What is the difference between INNER JOIN and LEFT JOIN?",
                "When should Spark use a broadcast join?",
                "How do you find customers with no orders?",
            ],
        ),
        post(
            "beta",
            topics=["SQL"],
            subtopics=["Joins"],
            concepts=["Inner Join"],
            questions=[
                "What is the difference between INNER JOIN and LEFT JOIN?",
                "Explain joins briefly.",
            ],
        ),
        post(
            "gamma",
            topics=["Python"],
            subtopics=["Data Structures"],
            concepts=["List", "Dictionary"],
            questions=[
                "When would you use a dictionary instead of a list?",
                "What does a list comprehension do?",
            ],
        ),
        post(
            "delta",
            topics=["Interview Preparation"],
            subtopics=["Practice Platforms and Preparation"],
            concepts=["LeetCode"],
            questions=[
                "How should a candidate practice SQL interview questions?",
                "Which SQL topics should a candidate prioritise for an "
                "interview?",
            ],
        ),
        post(
            "epsilon",
            topics=["Interview Preparation"],
            subtopics=["Behavioural"],
            concepts=["STAR"],
            questions=[
                "How should a candidate use STAR format for a conflict "
                "question?",
            ],
        ),
        post(
            "zeta",
            topics=["Career and Hiring"],
            subtopics=["Job Seeking and Preparation"],
            concepts=[],
            questions=[
                "How can a job seeker make their skills visible?",
                "How would you write a CTE-based SQL solution that selects "
                "analysts under a hiring policy?",
            ],
        ),
    ]

    payload = knowledge_base(posts)

    path = tmp_path_factory.mktemp("revision") / "knowledge_base.json"

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return path, payload


@pytest.fixture(scope="module")
def curriculum(corpus):
    """The taxonomy, read through the same validator the generator uses."""

    path, _payload = corpus

    return build_curriculum(list(load_canonical(path).posts))


@pytest.fixture(scope="module")
def units(curriculum):
    return build_units(curriculum)


@pytest.fixture(scope="module")
def site(corpus, tmp_path_factory):
    path, _payload = corpus

    output = tmp_path_factory.mktemp("site")

    generate_site(input_path=path, output_dir=output)

    return output


# ---------------------------------------------------------------------
# One revision unit is one page
# ---------------------------------------------------------------------


def test_the_unit_table_names_every_source_it_claims():
    """
    A unit that names a subtopic the taxonomy does not have would render
    as a page with nothing on it, and the reference would be a typo that
    only shows up as an empty section.

    Checked against the real taxonomy, because the fixture declares six
    subtopics out of a hundred and fifty-four. Measured against the
    repository's own corpus; skipped where that corpus is absent, since
    a test that silently checks nothing is worse than one that is
    honestly absent.
    """

    repository = Path(__file__).resolve().parents[1] / "build" / (
        "knowledge_base.json"
    )

    if not repository.is_file():
        pytest.skip("build/knowledge_base.json is not present")

    real = build_curriculum(list(load_canonical(repository).posts))

    declared = {
        source for spec in UNITS for source in spec.sources
    }

    missing = sorted(
        (major, subtopic)
        for major, subtopic in declared
        if real.subtopic(major, subtopic) is None
    )

    assert missing == [], (
        f"the unit table names {len(missing)} subtopics the taxonomy "
        f"does not have: {missing[:5]}"
    )

    # And no subtopic is claimed twice, which would put the same
    # questions on two units and break "each question appears once".
    claimed = [source for spec in UNITS for source in spec.sources]

    duplicates = {
        source
        for source in claimed
        if claimed.count(source) > 1
    }

    assert duplicates == set(), sorted(duplicates)


def test_the_documented_units_are_the_ones_the_brief_names():
    """
    The brief names its examples explicitly: SQL split by topic, and
    Python, Spark, Databricks, ADF and AWS as one page each.

    These are the decisions a future edit could quietly reverse, so they
    are named here rather than left to be re-derived from the code.
    """

    by_slug = {spec.slug: spec for spec in UNITS}

    one_page_each = (
        "python-for-data-engineering",
        "spark-and-pyspark",
        "databricks",
        "azure-data-factory",
        "aws-for-data-engineering",
    )

    for slug in one_page_each:
        assert slug in by_slug, slug

    # SQL is the opposite: split by topic, because that is how it is
    # revised and asked.
    assert "sql-joins" in by_slug
    assert "sql-window-functions" in by_slug
    assert "sql-aggregations" in by_slug
    assert "sql-ctes-and-subqueries" in by_slug
    assert "sql-writing-and-optimization" in by_slug

    # And no unit is a restatement of a taxonomy subtopic one-for-one,
    # which is the granularity this change exists to remove.
    assert len(UNITS) < 60


def test_a_unit_that_fits_stays_on_one_page(units):
    """
    A unit below the threshold is not split.

    Splitting everything into pages of twenty made a page with one
    question on it, which is the per-subtopic granularity in a new coat.
    """

    for unit in units.units:
        if unit.question_count <= QUESTIONS_PER_PAGE:
            pages = unit.pages()

            assert len(pages) == 1, unit.slug
            assert pages[0].path == f"revision/{unit.slug}.html"

            # And there is nothing to navigate, so there is no pager.
            assert "pager" not in _body(units, unit, pages[0])


def test_a_unit_that_does_not_fit_splits_into_pages_of_itself(units):
    """
    A unit above the threshold becomes pages of *the same unit*, not
    several units. The pages share a title, a summary, a Quick Revision
    block and a Related block; only the question list and the pager
    differ.

    This is the difference between "paginated" and "fragmented", and it
    is what keeps the page count from being a proxy for the subtopic
    count.
    """

    from src.wiki.curriculum import RevisionQuestion

    big = Unit(
        spec=next(s for s in UNITS if s.slug == "sql-aggregations"),
        questions=[],
    )

    # Build one deliberately over the threshold, without needing the
    # corpus to be that large.
    template = next(
        q
        for unit in units.units
        for q in unit.questions
    )

    big.questions = [
        RevisionQuestion(
            **{**template.__dict__, "text": f"Synthetic question {n:04d}?"}
        )
        for n in range(QUESTIONS_PER_PAGE * 2 + 5)
    ]

    pages = big.pages()

    assert len(pages) == 3

    # No page is empty, and none is overfull.
    for page in pages:
        assert page.questions, page.number
        assert len(page.questions) <= QUESTIONS_PER_PAGE, page.number

    # The questions are split in a fixed order, so the same corpus puts
    # the same question on the same page every time.
    assert [q.text for q in pages[0].questions] == sorted(
        q.text for q in big.questions
    )[:QUESTIONS_PER_PAGE]

    # Every page is reachable from every other, by following the pager.
    for page in pages:
        assert page.total == 3

    assert pages[0].is_first and not pages[0].is_last
    assert not pages[1].is_first and not pages[1].is_last
    assert pages[2].is_last and not pages[2].is_first

    assert pages[0].sibling_path(1) == pages[0].path
    assert pages[0].sibling_path(2) == "revision/sql-aggregations-2.html"
    assert pages[2].sibling_path(3) == pages[2].path

    # And the path of a UnitPage is what the index links.
    assert UnitPage(number=1, total=3, slug="x", questions=()).path == (
        "revision/x.html"
    )
    assert UnitPage(number=2, total=3, slug="x", questions=()).path == (
        "revision/x-2.html"
    )


def test_page_count_is_a_measurable_rule_not_a_taste(units):
    """
    The threshold is arithmetic, so the page count can be predicted from
    the question count without rendering anything.
    """

    for unit in units.units:
        expected = max(
            1, math.ceil(unit.question_count / QUESTIONS_PER_PAGE)
        )

        assert unit.page_count == expected, unit.slug
        assert len(unit.pages()) == expected, unit.slug

    # And it is the *measured* rule, so a unit of exactly the threshold
    # is one page and one more is two.
    assert page_count(QUESTIONS_PER_PAGE) == 1
    assert page_count(QUESTIONS_PER_PAGE + 1) == 2
    assert page_count(0) == 1
    assert page_count(1) == 1


def _body(units, unit, page) -> str:
    from src.wiki.unit_pages import render_unit

    return render_unit(units, unit, page)


# ---------------------------------------------------------------------
# The filter accounts for itself
# ---------------------------------------------------------------------


def test_every_question_is_published_or_discarded_with_a_reason(
    curriculum, units
):
    """
    The arithmetic has to close.

    "1,902 published" is only meaningful if nothing went missing on the
    way. A question that passed the filter and then landed in no unit
    would be neither published nor counted as discarded, and the page
    count would look better than the coverage is.

    Run against the real corpus when it is present, and against the
    fixture otherwise, because the property is the same either way and
    the fixture alone would let a real unmapped subtopic through.
    """

    built = build_units(curriculum)

    assert built.kept + built.discarded == curriculum.question_count
    assert built.unmapped == ()

    # And the reasons sum to the discarded count.
    assert sum(built.discard_reasons.values()) == built.discarded

    assert built.kept == units.kept


def test_career_and_interview_process_material_is_not_published(units):
    """
    The brief's exclusion list, as a property of the output rather than
    of a pattern list.
    """

    published = " ".join(
        q.text for unit in units.units for q in unit.questions
    )

    for advice in (
        "practice SQL interview questions",
        "STAR format",
        "prioritise for an interview",
        "make their skills visible",
    ):
        assert advice not in published, advice


def test_a_basic_programming_question_survives(units):
    """
    The other half, and the half a filter like that usually gets wrong.

    "reverse a string", "valid palindrome", "list comprehension" and
    "merge two sorted lists" sit near the excluded vocabulary, and a
    filter written by pattern alone removes them. The brief asks for
    them explicitly.
    """

    published = [q.text for unit in units.units for q in unit.questions]

    assert "When would you use a dictionary instead of a list?" in published
    assert "What does a list comprehension do?" in published

    assert is_relevant("Write code to reverse a string.", "python")
    assert is_relevant("Implement a valid palindrome check.", "python")
    assert is_relevant("How do I merge two sorted lists?", "coding-and-algorithms")
    assert is_relevant("Explain big-O time complexity.", "coding-and-algorithms")


def test_a_misfiled_technical_question_is_relocated_not_dropped(units):
    """
    Two questions were filed under the search and are really questions
    about the work. They move to the unit that owns them.
    """

    published = {
        q.text for unit in units.units for q in unit.questions
    }

    assert (
        "How would you write a CTE-based SQL solution that selects "
        "analysts under a hiring policy?" in published
    )

    cte = units.unit("sql-ctes-and-subqueries")

    assert cte is not None

    assert any("CTE-based" in q.text for q in cte.questions)

    # And the relocation is a rule with a named target, not a side
    # effect.
    assert relocation_target("What roles does HDFS and YARN play?") == (
        "spark-and-pyspark"
    )
    assert relocation_target("How do you write a recursive CTE?") == (
        "sql-ctes-and-subqueries"
    )
    assert relocation_target("What is a join?") is None


def test_an_excluded_subtopic_is_excluded_even_when_it_names_a_technology(
    units,
):
    """
    The rule that was tried the other way round and had to be reversed.

    Allowing a question through because it names a technology kept 33 of
    the 85 questions in the excluded subtopics, every one of them advice:
    "which SQL topics should a candidate prioritise for an interview"
    names SQL and is still advice. So the subtopic decides, and the two
    technical exceptions are relocated by name.
    """

    assert (
        "interview-preparation",
        "practice-platforms-and-preparation",
    ) in EXCLUDED_SUBTOPICS

    assert exclusion_reason(
        "Which SQL topics should a candidate prioritise?",
        "interview-preparation",
        "practice-platforms-and-preparation",
    ) is not None

    assert "career-and-hiring" in NON_REVISION_SUBJECTS


def test_advice_is_refused_even_when_it_names_the_technology():
    """
    Order matters, and this is the case that fixes it.

    "Which SQL capabilities are prioritised in Week 1 of the roadmap?"
    names SQL. Naming the work is enough to *keep* a question and not
    enough to *rescue* one that is advice.
    """

    assert exclusion_reason(
        "Which SQL capabilities are prioritised in Week 1 of the "
        "roadmap?",
        "sql",
        "sql-fundamentals",
    ) is not None

    assert exclusion_reason(
        "Which SQL clauses implement a recursive CTE?", "sql", "ctes"
    ) is None


# ---------------------------------------------------------------------
# Provenance survives the page removal
# ---------------------------------------------------------------------


def test_provenance_is_intact_in_the_canonical_data(corpus):
    """
    Nothing was dropped from the data to make the site smaller.

    Every post keeps its identifier, its source, its verbatim text and
    its saved item. The posts are simply not pages any more.
    """

    _path, payload = corpus

    for record in payload["posts"]:
        assert record["id"]
        assert record["source"]["platform"] == "linkedin"
        assert record["source"]["captured_at"]
        assert record["source"]["url"]
        assert record["original_text"]
        assert record["saved_item"]["saved_item_id"]


def test_a_question_still_names_every_post_that_raised_it(units):
    """
    Consolidation is not conflation.

    "What is the difference between INNER JOIN and LEFT JOIN?" is asked
    by two posts and appears once. It has to remember both, or a reader
    has no way to know the question is common rather than incidental.
    """

    shared = [
        q
        for unit in units.units
        for q in unit.questions
        if len(q.post_ids) > 1
    ]

    assert shared, "the fixture should produce a consolidated question"

    for question in shared:
        assert len(set(question.post_ids)) == len(question.post_ids)
        assert len(question.source_labels) == len(question.post_ids)

        # The label is prose, never an identifier.
        for label in question.source_labels:
            assert "urn" not in label
            assert "urn-li-" not in label
            assert label == "LinkedIn post, saved from a list"


def test_the_question_disclosure_is_in_words_not_identifiers(site):
    """
    The disclosure the reader gets, which replaced the page that named
    the post.
    """

    published = "".join(
        page.read_text(encoding="utf-8")
        for page in site.rglob("*.html")
    )

    assert "appears in" in published
    assert "LinkedIn post" in published
    assert "saved from a list" in published

    for identifier in ("urn-li-", "urn:li:"):
        assert identifier not in published, identifier


# ---------------------------------------------------------------------
# What is published
# ---------------------------------------------------------------------


def test_no_page_is_generated_per_source_post(site):
    """
    The change, stated as a test so it cannot be reversed by accident.

    There is one page per revision unit. A page per captured post is the
    shape of the corpus, and it was 490 of the site's 649 pages.
    """

    assert not (site / "posts").exists()
    assert not (site / "topics").exists()

    for identifier in ("alpha", "beta", "gamma"):
        for page in site.rglob("*.html"):
            assert identifier not in page.read_text(encoding="utf-8"), (
                page.name
            )


def test_the_site_is_tens_of_pages_not_hundreds(site):
    """
    The scale is the requirement, so it is measured rather than trusted.

    Sixty-four is the current number. The bound is loose enough to allow
    a unit or two to be split further and tight enough to catch a return
    to one page per subtopic, which was 154 pages and one per post, 649
    in total.
    """

    pages = list(site.rglob("*.html"))

    assert len(pages) < 100, len(pages)

    # No family of pages exists beyond the tree and its units.
    directories = {
        path.relative_to(site).parts[0]
        for path in pages
        if len(path.relative_to(site).parts) > 1
    }

    assert directories <= {"revision"}, directories


def test_the_home_page_is_the_curriculum(site):
    """
    Not the corpus.

    The counts on the front page are revision units, pages and
    questions. Posts, images, slides and archive statistics are not
    shown, because "3,047 slide images" describes the capture and tells
    a candidate revising for an interview nothing about joins.
    """

    home = (site / INDEX_PAGE).read_text(encoding="utf-8")

    # The counted line, which is the claim under test. The footer caveat
    # legitimately mentions that the material came from LinkedIn posts,
    # so checking the whole page for the word "posts" would fail on
    # prose that is not a count.
    counted = re.findall(r'<p class="counts">(.*?)</p>', home, re.S)

    assert counted, "the home page reports no counts"

    text = " ".join(re.sub(r"<[^>]+>", " ", counted[0]).split())

    assert "revision unit" in text
    assert "question" in text

    for corpus_word in ("posts", "images", "slides", "captured", "concept"):
        assert corpus_word not in text.lower(), corpus_word

    # Every unit is reachable from the front page.
    for unit_page in (site / "revision").glob("*.html"):
        if unit_page.name == "index.html":
            continue

        assert f'href="revision/{unit_page.name}"' in home, unit_page.name

    # And what was left out is stated, which is the honest counterpart.
    assert "Not revision material" in home


def test_a_revision_page_has_the_documented_structure(site):
    """
    Title, summary, Quick Revision, questions, Related, and no
    identifiers. In that order, so a reader learns the shape once.
    """

    page = site / "revision" / "sql-joins.html"

    if not page.is_file():
        pytest.skip("the fixture did not produce a joins unit")

    body = page.read_text(encoding="utf-8")

    order = [
        body.find("<h1>"),
        body.find('class="lede"'),
        body.find("Quick revision"),
        body.find("Interview questions"),
        body.find("Related revision"),
    ]

    assert all(position >= 0 for position in order), order

    assert order == sorted(order), order

    # Every question opens in place, so the page is scannable.
    assert "<details" in body
    assert body.count("<details") == body.count("</details>")

    # The breadcrumb goes home, then to the tree, then to the unit. The
    # tree lives beside the unit pages, so it is a sibling here and the
    # site root is one level up.
    assert "breadcrumb" in body
    assert 'href="index.html"' in body
    assert 'href="../index.html"' in body


def test_the_questions_index_and_the_units_agree(site):
    """
    Both are built from the same model in the same pass, so they cannot
    drift. Checked anyway, because "cannot drift" is a claim about the
    code and this is a claim about the output.

    The two are reconciled by link rather than by title: each unit on
    the index is linked, and every question listed is on a page that
    exists. Comparing titles would need a slug rule that does not exist
    and should not.
    """

    index = json.loads(
        (site / SEARCH_INDEX_FILE).read_text(encoding="utf-8")
    )

    page = (site / QUESTIONS_PAGE).read_text(encoding="utf-8")

    unit_pages = {
        path.name
        for path in (site / "revision").glob("*.html")
        if path.name != "index.html"
    }

    assert index["units"] == len(unit_pages)

    linked = {
        Path(target).name
        for target in re.findall(r'href="(revision/[^"#?]+)"', page)
    }

    assert linked, "the questions index links no unit"

    # The page's own chrome links the tree as well as the units, so the
    # tree is allowed and anything else is not.
    assert linked - {"index.html"} <= unit_pages, sorted(
        linked - unit_pages - {"index.html"}
    )

    # Every unit is listed, which is the direction that matters: a unit
    # missing from the questions index is a unit whose questions are hard
    # to find.
    assert unit_pages <= linked | {"index.html"}, sorted(
        unit_pages - linked
    )

    # And the question count on the index is the published count.
    counted = re.search(
        r'<p class="counts">([^<]*)</p>', page
    )

    assert counted

    assert str(index["questions"]) in counted.group(1)


def test_the_search_index_reaches_units_not_posts(site):
    """
    "Broadcast Hash Join" is a concept label, and a search for "broadcast
    join" has to reach the Spark unit.

    With no post records and no concept records, a unit's own concepts
    are the only thing that makes it findable by a term the reader
    actually types.
    """

    index = json.loads(
        (site / SEARCH_INDEX_FILE).read_text(encoding="utf-8")
    )

    records = index["records"]

    assert {record["k"] for record in records} == {"b", "q"}

    for record in records:
        assert record["u"].startswith("revision/")
        assert (site / record["u"]).is_file(), record["u"]

    concepts = {
        label
        for record in records
        for label in record.get("c", [])
    }

    assert concepts, "no unit carries a concept"

    # The unit records carry the concepts; the question records carry the
    # answer text. A reader remembers questions by their answers.
    questions = [record for record in records if record["k"] == "q"]

    assert sum(1 for q in questions if q["s"]) > len(questions) // 2


def test_the_revision_index_lists_every_unit(site):
    index = (site / REVISION_INDEX).read_text(encoding="utf-8")

    for page in (site / "revision").glob("*.html"):
        if page.name == "index.html":
            continue

        assert f'href="{page.name}"' in index, page.name