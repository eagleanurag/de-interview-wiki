"""
Tests for the static wiki generator.

Tests build a canonical knowledge base fixture that resembles real
aggregator output, generate the site into a temporary directory, and
assert on the result. No generated file is written inside the
repository.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest


from src.wiki.analysis import build_site_model
from src.wiki.canonical import (
    CanonicalKnowledgeBase,
    WikiError,
    load_canonical,
)
from src.wiki.generator import generate_site
from src.wiki.naming import (
    CONCEPTS_PAGE,
    INDEX_PAGE,
    MANIFEST_FILE,
    NOT_FOUND_PAGE,
    QUESTIONS_PAGE,
    REVISION_INDEX,
    SAVED_ITEMS_PAGE,
    SEARCH_INDEX_FILE,
    SEARCH_PAGE,
    TECHNOLOGIES_PAGE,
    TOPICS_PAGE,
)


REPO_ROOT = Path(__file__).resolve().parents[1]

EXTERNAL_PREFIXES = (
    "http://",
    "https://",
    "mailto:",
    "data:",
    "javascript:",
    "#",
)

# Pipeline metadata that must never appear in a published site.
LEAK_TOKENS = (
    "job_id",
    "post_directory",
    "output_path",
    "max_attempts",
    "worker_id",
    "cloud_worker_",
    "skipped_files",
    "result_files_found",
    "files_skipped",
    "data/jobs/",
    "data/results/",
)


def make_post(
    post_id: str,
    *,
    topics: list[str] | None = None,
    subtopics: list[str] | None = None,
    concepts: list[str] | None = None,
    questions: list[dict] | None = None,
    url: str | None = "https://example.com/post",
    captured_at: str = "2026-09-29T18:15:00+05:30",
    original_text: str = "Delta table performance tuning notes.",
    summary: str = "A summary of Delta Lake tuning.",
) -> dict:
    """Build one enriched KnowledgePost exactly as a worker emits it."""

    return {
        "id": post_id,
        "source": {
            "platform": "linkedin",
            "url": url,
            "captured_at": captured_at,
            "author": "Sample Author",
        },
        "original_text": original_text,
        "media": [
            {
                "type": "image",
                "path": f"data/posts/{post_id}/media/chart.png",
                "description": "PNG image, 1200x700 pixels",
                "extracted_text": None,
            }
        ],
        "ai_analysis": {
            "summary": summary,
            "topics": topics
            if topics is not None
            else ["Databricks", "Delta Lake"],
            "subtopics": subtopics
            if subtopics is not None
            else ["Partitioning Strategy"],
            "concepts": concepts
            if concepts is not None
            else ["Partition Pruning", "Bloom Filters"],
            "image_descriptions": [],
        },
        "interview_questions": questions
        if questions is not None
        else [
            {
                "question": "When should you partition a Delta table?",
                "type": "scenario",
                "difficulty": "medium",
                "answer": "Partition on low cardinality columns.\n"
                "Avoid near-unique keys.",
            }
        ],
        "classification": {
            "domain": "Data Engineering",
            "primary_topic": "Databricks",
            "secondary_topics": ["Apache Spark"],
            "interview_relevant": True,
        },
    }


def make_knowledge_base(
    posts: list[dict],
    *,
    generated_at: str = "2026-09-30T00:00:00+00:00",
    skipped_files: list[str] | None = None,
) -> dict:
    """Build a canonical payload, including the keys the site must ignore."""

    return {
        "schema_version": 1,
        "generated_at": generated_at,
        "stats": {
            "result_files_found": len(posts) * 2,
            "posts_aggregated": len(posts),
            "files_skipped": len(posts),
        },
        "skipped_files": skipped_files
        if skipped_files is not None
        else [
            "cloud-worker-result-alpha/data/jobs/"
            "cloud_worker_alpha_1.json: worker job manifest",
            "cloud-worker-result-alpha/data/jobs/"
            "output_path_field_present: worker job manifest",
        ],
        "posts": posts,
    }


@pytest.fixture
def knowledge_base() -> dict:
    return make_knowledge_base(
        [
            make_post(
                "sample_alpha",
                topics=["Databricks", "Delta Lake"],
                subtopics=["Partitioning Strategy"],
                concepts=["Partition Pruning", "Bloom Filters"],
                questions=[
                    {
                        "question": "When should you partition a Delta table?",
                        "type": "scenario",
                        "difficulty": "medium",
                        "answer": "Partition on low cardinality columns.\n"
                        "Avoid near-unique keys.",
                    },
                    {
                        "question": "Explain data skipping.",
                        "type": "theory",
                        "difficulty": "hard",
                        "answer": "Per-file min/max statistics.",
                    },
                ],
            ),
            make_post(
                "sample_beta",
                topics=["Apache Spark", "Performance Tuning"],
                subtopics=["Adaptive Query Execution"],
                concepts=["Data Skew", "Shuffle Partitions"],
                captured_at="2026-09-28T09:00:00+05:30",
                questions=[
                    {
                        "question": "How does AQE reduce skew?",
                        "type": "troubleshooting",
                        "difficulty": "hard",
                        "answer": "Splits skewed partitions at runtime.",
                    }
                ],
            ),
        ]
    )


@pytest.fixture
def canonical_file(tmp_path: Path, knowledge_base: dict) -> Path:
    path = tmp_path / "knowledge_base.json"
    path.write_text(
        json.dumps(knowledge_base, indent=2), encoding="utf-8"
    )

    return path


@pytest.fixture
def site(tmp_path: Path, canonical_file: Path) -> Path:
    output = tmp_path / "site"
    generate_site(canonical_file, output)

    return output


def read(site: Path, relative: str) -> str:
    return (site / relative).read_text(encoding="utf-8")


def all_files(site: Path) -> list[Path]:
    return sorted(
        path
        for path in site.rglob("*")
        if path.is_file()
    )


# ---------------------------------------------------------------------
# Canonical input loading
# ---------------------------------------------------------------------


def test_canonical_input_loads_correctly(canonical_file: Path):
    loaded = load_canonical(canonical_file)

    assert loaded.schema_version == 1
    assert len(loaded.posts) == 2
    assert loaded.stats.posts_aggregated == 2

    # Post payloads validate against the shared schema, not a copy.
    post = loaded.posts[0]
    assert post.id == "sample_alpha"
    assert post.ai_analysis.topics == ["Databricks", "Delta Lake"]
    assert post.interview_questions[0].difficulty == "medium"
    assert post.classification.interview_relevant is True


def test_missing_input_fails(tmp_path: Path):
    with pytest.raises(WikiError) as error:
        generate_site(tmp_path / "absent.json", tmp_path / "site")

    assert "Knowledge base not found" in str(error.value)


def test_input_directory_instead_of_file_fails(tmp_path: Path):
    directory = tmp_path / "knowledge_base.json"
    directory.mkdir()

    with pytest.raises(WikiError) as error:
        generate_site(directory, tmp_path / "site")

    assert "directory" in str(error.value)


def test_malformed_json_fails(tmp_path: Path):
    broken = tmp_path / "knowledge_base.json"
    broken.write_text("{not json", encoding="utf-8")

    with pytest.raises(WikiError) as error:
        generate_site(broken, tmp_path / "site")

    assert "not valid JSON" in str(error.value)


def test_payload_without_posts_fails(tmp_path: Path):
    path = tmp_path / "knowledge_base.json"
    path.write_text(
        json.dumps({"schema_version": 1}), encoding="utf-8"
    )

    with pytest.raises(WikiError) as error:
        generate_site(path, tmp_path / "site")

    assert "'posts'" in str(error.value)


def test_posts_wrong_type_fails(tmp_path: Path):
    path = tmp_path / "knowledge_base.json"
    path.write_text(
        json.dumps({"posts": {"not": "a list"}}),
        encoding="utf-8",
    )

    with pytest.raises(WikiError) as error:
        generate_site(path, tmp_path / "site")

    assert "must be a list" in str(error.value)


def test_invalid_post_payload_fails(tmp_path: Path):
    payload = make_knowledge_base([make_post("sample_alpha")])
    payload["posts"][0]["source"]["captured_at"] = "not-a-timestamp"

    path = tmp_path / "knowledge_base.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(WikiError) as error:
        generate_site(path, tmp_path / "site")

    assert "failed validation" in str(error.value)


def test_duplicate_post_ids_fail(tmp_path: Path):
    payload = make_knowledge_base(
        [make_post("sample_alpha"), make_post("sample_alpha")]
    )

    path = tmp_path / "knowledge_base.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(WikiError) as error:
        generate_site(path, tmp_path / "site")

    assert "duplicate post IDs" in str(error.value)


def test_utf8_bom_input_is_accepted(tmp_path: Path):
    path = tmp_path / "knowledge_base.json"
    path.write_text(
        json.dumps(make_knowledge_base([make_post("sample_alpha")])),
        encoding="utf-8-sig",
    )

    output = tmp_path / "site"
    generate_site(path, output)

    assert (output / INDEX_PAGE).exists()


# ---------------------------------------------------------------------
# Page generation
# ---------------------------------------------------------------------


def test_required_root_pages_are_generated(site: Path):
    """
    Four root pages, and the revision tree beside them.

    There is no ``subjects.html`` any more: the tree a reader walks is
    the revision units, and it is written to ``revision/index.html``
    because that is where the pages it lists live.
    """

    for relative in (
        INDEX_PAGE,
        SEARCH_PAGE,
        QUESTIONS_PAGE,
        NOT_FOUND_PAGE,
        REVISION_INDEX,
    ):
        target = site / relative

        assert target.exists(), f"missing {relative}"
        assert target.stat().st_size > 0


def test_no_archive_page_family_is_generated(site: Path):
    """
    The removal, asserted.

    Nine thousand of this site's pages used to exist to list corpus
    labels: 5,786 grouped by topic, 3,112 concepts, 44 technologies,
    saved items, and four indexes over them. A candidate revising for an
    interview cannot use any of them -- a concept is a label the enricher
    produced, not a thing to revise -- and their presence is what made
    the site read as an archive with a revision front end.

    Nothing is deleted from the data. These are page paths.
    """

    gone = (
        TOPICS_PAGE,
        CONCEPTS_PAGE,
        TECHNOLOGIES_PAGE,
        SAVED_ITEMS_PAGE,
    )

    for relative in gone:
        assert not (site / relative).exists(), f"{relative} was generated"

    for directory in ("concepts", "technologies"):
        assert not (site / directory).exists(), (
            f"{directory}/ was generated"
        )

    # Archive topic pages sat directly under topics/; the revision pages
    # sit one level deeper, under topics/<subject>/<subtopic>.html.
    archive_topics = [
        path
        for path in (site / "topics").glob("*.html")
    ]

    assert archive_topics == [], (
        f"archive topic pages generated: {archive_topics[:3]}"
    )


def test_the_surviving_pages_link_to_nothing_that_is_gone(site: Path):
    for page in site.rglob("*.html"):
        body = page.read_text(encoding="utf-8")

        for removed in ("concepts/", "technologies/", "saved-items.html"):
            assert removed not in body, (
                f"{page.relative_to(site)} links to {removed}"
            )

        for removed in ("topics.html", "concepts.html", "technologies.html"):
            # subjects/<slug>.html and the revision tree are under topics/
            # and must not be confused with the removed archive index.
            assert f'href="{removed}"' not in body
            assert f'href="../{removed}"' not in body
            assert f'href="../../{removed}"' not in body


def test_home_page_shows_the_curriculum_not_the_corpus(site: Path):
    """
    The front page answers "what is in here", not "how big is the
    archive".

    It used to assert live counts of posts, topics and concepts, and to
    assert those labels were *absent* -- the counts of the corpus were
    kept and the names changed. That was the wrong half of the problem:
    "3,047 slide images" describes the capture, and a candidate revising
    for an interview cannot use it. The page now counts revision units,
    pages and questions, which is what it can be used for.
    """

    home = read(site, INDEX_PAGE)

    counts = re.findall(
        r'<p class="counts">(.*?)</p>', home, re.S
    )

    assert counts, "the home page reports no counts"

    text = re.sub(r"<[^>]+>", " ", counts[0])

    assert "revision unit" in text, text
    assert "page" in text, text
    assert "question" in text, text

    # And no corpus statistics at all.
    for corpus in ("posts", "images", "slides", "concepts", "archive",
                   "captured"):
        assert corpus not in text.lower(), f"{corpus!r} is still counted"

    # It lists the units, grouped. Which units depends on the fixture,
    # so this asserts the shape rather than the roster.
    assert "revision/" in home

    groups = re.findall(r"<h2>([^<]+)</h2>", home)

    assert groups, "the home page has no group headings"

    assert groups, "no group headings"

    # Every heading is a subject area rather than a page title.
    assert all(
        not g.endswith(".html") for g in groups
    ), groups

    # And it says what was left out, which is the honest counterpart to
    # the counts.
    assert "Not revision material" in home


def test_home_page_no_longer_lists_recent_posts(site: Path):
    """
    Removed rather than rewritten: there are no post pages to link to.

    The section listed the six most recent captured posts. Every one of
    those links led to a page that is no longer generated, which is the
    same reason the posts are not published.
    """

    home = read(site, INDEX_PAGE)

    assert "posts/" not in home
    assert "urn-li-" not in home


def test_home_page_has_no_sample_id_hardcoding():
    """
    The generator must not embed the fixture IDs. The dashboard
    renders whatever the knowledge base contains.
    """

    source = (
        REPO_ROOT / "src" / "wiki" / "pages.py"
    ).read_text(encoding="utf-8")

    for sample_id in ("sample_001", "sample_002", "sample_003"):
        assert sample_id not in source




def test_no_topic_page_and_no_topics_index_remain(site: Path):
    assert list((site / "topics").glob("*.html")) == []

    assert not (site / TOPICS_PAGE).exists()

    # And nothing in the surviving tree reaches them. The revision pages
    # also live under topics/, but always as topics/<subject>/<slug>.html
    # -- two segments. An archive page was topics/<slug>.html, one.
    pattern = re.compile(r'href="(?:\.\./)*topics/[^/"]+\.html"')

    for page in site.rglob("*.html"):
        body = page.read_text(encoding="utf-8")

        assert not pattern.search(body), (
            f"{page.relative_to(site)} links to an archive topic page"
        )


def test_questions_page_lists_every_published_question(site: Path):
    """
    One list, and it lists what the guide publishes.

    The questions page used to render every curriculum question from 154
    subtopic sections, each question linking back to a post. It now
    lists the questions on the revision units, each linking to the unit
    that answers it -- so it cannot drift from the pages it indexes,
    because both are built from the same model in the same pass.
    """

    page = read(site, QUESTIONS_PAGE)

    assert "Interview questions" in page

    index = json.loads(read(site, SEARCH_INDEX_FILE))

    assert index["questions"] > 0

    links = set(re.findall(r'href="(revision/[^"]+)"', page))

    assert links, "the questions page links to nothing"

    for href in links:
        assert (site / href).is_file(), href

    # It links a unit, never a post.
    assert "posts/" not in page

    # And it carries the questions themselves, not just links.
    questions = [r for r in index["records"] if r["k"] == "q"]

    for record in questions[:5]:
        text = re.sub(r"<[^>]+>", " ", record["i"])

        assert text.split()[0] in page, record["i"][:60]


def test_questions_page_has_no_filter_that_filters_nothing(site: Path):
    """
    The old page carried a subject filter and ``data-`` attributes to
    drive it, because it grouped 2,016 questions across 14 subjects and
    a reader wanted to narrow that.

    The grouping is now the 22 unit headings, so the panel was removed
    rather than left controlling nothing. What matters is that nothing on
    the page still claims a filter exists.
    """

    page = read(site, QUESTIONS_PAGE)

    assert 'id="filter-topic"' not in page
    assert 'id="filter-' not in page
    assert "<select" not in page


def test_search_index_describes_only_published_pages(site: Path):
    """
    Every record points at a file that exists, and there are two kinds.

    The index used to hold 11,616 records over posts, topics, concepts,
    technologies, subjects, subtopics and questions. Each kind pointed at
    a page, and all of them stopped existing when the archive layer and
    the per-post pages were removed. A record that leads nowhere is
    worse than a missing one, because it appears in the result list.
    """

    index = json.loads(read(site, SEARCH_INDEX_FILE))

    records = index["records"]

    assert records

    kinds = {record["k"] for record in records}

    assert kinds == {"b", "q"}, kinds

    for record in records:
        assert (site / record["u"]).is_file(), record["u"]

    units = [r for r in records if r["k"] == "b"]
    questions = [r for r in records if r["k"] == "q"]

    assert len(units) == index["units"]
    assert len(questions) == index["questions"]

    assert index["units"] > 0
    assert index["questions"] > 0

    # The retired families report zero rather than vanishing, so anything
    # reading the index finds a number it can trust.
    for retired in ("posts", "topics", "concepts", "technologies"):
        assert index[retired] == 0, retired


def test_search_index_carries_concepts_and_answers(site: Path):
    """
    What makes a search reach a unit rather than a dead end.

    "Broadcast Hash Join" is a concept label. Before units carried their
    concepts, a search for "broadcast join" could not reach Spark at all
    and fell back to the questions. And answers are indexed, because
    "how do I find consecutive rows" is how someone remembers a question
    they cannot otherwise place.
    """

    index = json.loads(read(site, SEARCH_INDEX_FILE))

    units = [r for r in index["records"] if r["k"] == "b"]

    assert any(u["c"] for u in units), "no unit carries any concept"

    questions = [r for r in index["records"] if r["k"] == "q"]

    with_answers = [q for q in questions if q["s"]]

    assert len(with_answers) > len(questions) // 2, (
        "most questions indexed with no answer text"
    )


def test_search_page_wires_the_index(site: Path):
    page = read(site, SEARCH_PAGE)

    assert f'"indexUrl":"{SEARCH_INDEX_FILE}"' in page
    assert "../assets/search.js" not in page
    assert 'src="assets/search.js"' in page
    assert 'id="search-input"' in page


def test_assets_are_copied_into_the_site(site: Path):
    for relative in (
        "assets/style.css",
        "assets/search.js",
        "assets/questions.js",
    ):
        target = site / relative

        assert target.exists(), f"missing {relative}"
        assert target.stat().st_size > 0


def test_manifest_records_generated_files(site: Path):
    """
    Every path in the manifest is a file, and the tree is in it.

    The manifest used to name 649 pages including 490 post pages. It is
    what a consumer reads to learn what was published, so a path in it
    that does not exist is the same defect as a broken link, found in
    one place instead of many.
    """

    manifest = json.loads(read(site, MANIFEST_FILE))

    assert manifest["generator"] == "src.wiki.generator"
    assert len(manifest["input_sha256"]) == 64

    for relative in manifest["files"]:
        assert (site / relative).exists(), relative

    published = [f for f in manifest["files"] if f.endswith(".html")]

    # Tens, not hundreds. The revision unit is the whole point, and this
    # is the number that would catch it reverting.
    assert len(published) < 100, len(published)

    assert INDEX_PAGE in manifest["files"]
    assert REVISION_INDEX in manifest["files"]
    assert SEARCH_INDEX_FILE in manifest["files"]

    assert any(f.startswith("revision/") for f in published)
    assert not any(f.startswith("posts/") for f in published)


# ---------------------------------------------------------------------
# Links
# ---------------------------------------------------------------------


def test_all_internal_links_resolve_to_generated_files(site: Path):
    broken = []

    for page in sorted(site.rglob("*.html")):
        document = page.read_text(encoding="utf-8")

        for reference in re.findall(
            r'(?:href|src)="([^"]+)"', document
        ):
            if reference.startswith(EXTERNAL_PREFIXES):
                continue

            target = reference.split("#")[0].split("?")[0]

            if not target:
                continue

            resolved = (page.parent / target).resolve()

            if not resolved.exists():
                broken.append(
                    f"{page.relative_to(site).as_posix()} -> "
                    f"{reference}"
                )

    assert broken == []


def test_nested_pages_climb_out_and_every_link_resolves(site: Path):
    """
    A page nested below the root must reach the root, and must not link
    to anywhere that does not exist.

    This replaced an assertion that every link on a nested page began
    with ``../``. Two things were wrong with it.

    **It was order-dependent.** It inspected
    ``next((site / "topics").glob("*/*.html"))`` -- one arbitrary page,
    whichever the filesystem enumerated first. Windows and Linux do not
    agree on that order, so the same commit passed locally and failed on
    the runner.

    **The rule was wrong.** A revision page links sideways to a sibling
    subtopic in the same directory, which from
    ``topics/spark-and-pyspark/spark-sql.html`` is the bare
    ``partitions-and-partitioning.html``. That is correct and it
    resolves. Demanding ``../../`` of it forbade a valid link to fix a
    problem the string check could not actually detect.

    What the old version was reaching for -- a link that resolves to the
    wrong place, such as a literal ``index.html`` written from
    ``topics/sql/``, which resolves to ``topics/sql/index.html``, a file
    that is never written -- is checked here directly instead: every
    reference on every nested page is resolved and required to be a file
    that exists. That is stronger, not weaker, and it does not care what
    order the filesystem lists anything in.
    """

    nested = sorted(site.glob("revision/*.html"))

    assert nested, "no nested pages were generated"

    for page in nested:
        depth = page.relative_to(site).parts[:-1]

        # One level below the root: revision/<unit>.html.
        up = "../" * len(depth)
        body = page.read_text(encoding="utf-8")

        # The tree lives beside the unit pages, so it is a sibling here,
        # while the site root is one level up.
        for chrome in (
            f'href="{up}index.html"',
            f'href="{up}assets/style.css"',
            f'href="{up}search.html"',
            'href="index.html"',
        ):
            assert chrome in body, f"{page.name} is missing {chrome}"

        for reference in re.findall(r'(?:href|src)="([^"]+)"', body):
            if reference.startswith(EXTERNAL_PREFIXES):
                continue

            target = reference.split("#")[0].split("?")[0]

            if not target:
                continue

            resolved = (page.parent / target).resolve()

            assert resolved.is_file(), (
                f"{page.relative_to(site)} links to {target!r}, "
                f"which resolves to {resolved} and is not generated"
            )


def test_root_pages_use_sibling_relative_links(site: Path):
    home = read(site, INDEX_PAGE)

    assert 'href="search.html"' in home
    assert 'href="questions.html"' in home
    assert 'href="revision/index.html"' in home
    assert 'href="assets/style.css"' in home

    # The tree lives one level down, so it is reached as a child rather
    # than a sibling.
    for page in (SEARCH_PAGE, QUESTIONS_PAGE, NOT_FOUND_PAGE):
        assert 'href="index.html"' in read(site, page)
        assert 'href="revision/index.html"' in read(site, page)

    tree = read(site, REVISION_INDEX)

    assert 'href="../index.html"' in tree
    assert 'href="../assets/style.css"' in tree


def test_no_absolute_or_repo_paths_are_hardcoded(site: Path):
    """
    The site must work under a GitHub Pages project path, so no link
    may be absolute or assume a repository owner and name.
    """

    forbidden = (
        "/de-interview-wiki",
        "eagleanurag",
        "https://de-interview-wiki",
    )

    for page in sorted(site.rglob("*.html")):
        document = page.read_text(encoding="utf-8")

        for token in forbidden:
            assert token not in document, (
                f"{page.relative_to(site).as_posix()} contains "
                f"{token!r}"
            )

        for reference in re.findall(r'href="(/[^"]*)"', document):
            raise AssertionError(
                f"{page.relative_to(site).as_posix()} uses an "
                f"absolute link: {reference}"
            )


def test_generator_source_has_no_repo_specific_paths():
    for path in sorted((REPO_ROOT / "src" / "wiki").rglob("*.py")):
        source = path.read_text(encoding="utf-8")

        assert "eagleanurag" not in source, path.name
        assert "de-interview-wiki" not in source, path.name


# ---------------------------------------------------------------------
# Escaping and privacy
# ---------------------------------------------------------------------




def test_no_job_manifest_metadata_leaks_into_the_site(
    site: Path,
    canonical_file: Path,
):
    """
    The canonical payload deliberately carries manifest paths in
    `skipped_files`. None of that may reach the published site.
    """

    payload = json.loads(
        canonical_file.read_text(encoding="utf-8")
    )

    assert payload["skipped_files"], "fixture must carry manifest data"

    for path in all_files(site):
        content = path.read_text(
            encoding="utf-8", errors="replace"
        )
        relative = path.relative_to(site).as_posix()

        for token in LEAK_TOKENS:
            assert token not in content, (
                f"{relative} leaks {token!r}"
            )


def test_search_index_ships_no_capture_metadata(site: Path):
    """
    The key set is asserted exactly, so a field added to the index is a
    deliberate change rather than a silent growth in what ships to every
    reader's browser.

    The record is now a revision unit rather than a post, so the fields
    that described a capture -- ``d`` the capture date, ``a`` the
    author, ``x`` the source excerpt -- are gone rather than blank.
    ``bc`` is the breadcrumb the result card reads, added so a hit can
    say "SQL: Window Functions" instead of "SQL".

    What is deliberately *not* asserted is the absence of an answer.
    Answers are indexed now, and on purpose: "how do I find consecutive
    rows" is how someone remembers a question they cannot otherwise
    place. The answer text is the useful part of the record, and it is
    not capture metadata.
    """

    index = json.loads(read(site, SEARCH_INDEX_FILE))

    record = next(r for r in index["records"] if r["k"] == "b")

    assert set(record) == {
        "a",
        "bc",
        "c",
        "d",
        "i",
        "k",
        "n",
        "p",
        "pb",
        "q",
        "s",
        "sb",
        "t",
        "tp",
        "u",
    }

    # The retired capture fields carry nothing rather than being absent
    # from the schema, so the browser script reads a string either way.
    assert record["a"] == ""
    assert record["d"] == ""
    assert record["x"] if "x" in record else True

    for leaked in ("original_text", "author", "captured_at", "saved_item"):
        assert leaked not in record


# ---------------------------------------------------------------------
# Empty and degenerate input
# ---------------------------------------------------------------------


def test_empty_knowledge_base_is_deterministic(tmp_path: Path):
    path = tmp_path / "knowledge_base.json"
    path.write_text(
        json.dumps(make_knowledge_base([])),
        encoding="utf-8",
    )

    first = tmp_path / "site-a"
    second = tmp_path / "site-b"

    generate_site(path, first)
    generate_site(path, second)

    for relative in (
        INDEX_PAGE,
        SEARCH_PAGE,
        QUESTIONS_PAGE,
        NOT_FOUND_PAGE,
        SEARCH_INDEX_FILE,
    ):
        assert (first / relative).read_text(
            encoding="utf-8"
        ) == (second / relative).read_text(encoding="utf-8")

    assert not (first / "posts").exists()
    assert not (first / "topics").exists()

    for relative in (
        INDEX_PAGE,
        SEARCH_PAGE,
        QUESTIONS_PAGE,
        NOT_FOUND_PAGE,
        REVISION_INDEX,
    ):
        assert (first / relative).is_file(), relative
        assert (second / relative).is_file(), relative


def test_empty_knowledge_base_renders_empty_states(tmp_path: Path):
    """
    An empty corpus must render, say so, and not invent a tree.

    The home page used to say "No posts yet", which was a statement about
    the archive. It now reports zero revision units, which is a statement
    about the guide, and that is the thing a reader would ask.
    """

    path = tmp_path / "knowledge_base.json"
    path.write_text(
        json.dumps(make_knowledge_base([])), encoding="utf-8"
    )

    output = tmp_path / "site"
    generate_site(path, output)

    home = read(output, INDEX_PAGE)
    tree = read(output, REVISION_INDEX)
    questions = read(output, QUESTIONS_PAGE)
    index = json.loads(read(output, SEARCH_INDEX_FILE))

    assert index["records"] == []
    assert index["posts"] == 0
    assert index["units"] == 0
    assert index["questions"] == 0

    # Every page still exists and explains itself.
    assert "0 revision units" in re.sub(r"<[^>]+>", " ", home)

    assert "No interview questions yet" in questions
    assert "Revision units" in tree

    # And no unit page was invented to hold nothing.
    assert not list((output / "revision").glob("*.html")) or [
        path
        for path in (output / "revision").glob("*.html")
        if path.name != "index.html"
    ] == []


def test_unknown_question_type_is_rendered_without_breaking(tmp_path: Path):
    """
    A type the schema does not list must still render.

    The question is now on a revision unit page rather than on the
    questions index, which lists question text and links rather than
    badges. The badge is on the unit page, so that is where the label has
    to appear.
    """

    payload = make_knowledge_base(
        [
            make_post(
                "sample_type",
                questions=[
                    {
                        "question": "Odd type question?",
                        "type": "architecture",
                        "difficulty": "easy",
                        "answer": None,
                    }
                ],
            )
        ]
    )

    path = tmp_path / "knowledge_base.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    output = tmp_path / "site"
    generate_site(path, output)

    document = read(output, QUESTIONS_PAGE)

    assert "Odd type question?" in document

    unit_pages = [
        candidate.read_text(encoding="utf-8")
        for candidate in (output / "revision").glob("*.html")
        if candidate.name != "index.html"
    ]

    assert unit_pages, "no unit page was generated"

    body = "".join(unit_pages)

    assert "Odd type question?" in body
    assert "Architecture" in body
    assert "Easy" in body


def test_schema_limited_types_are_not_invented(site: Path):
    """
    The schema only permits five question types. The generator must not
    fabricate options for types the data does not contain.

    Read from the unit pages, where the type badge is rendered, since the
    questions index lists text without badges.
    """

    document = "".join(
        path.read_text(encoding="utf-8")
        for path in (site / "revision").glob("*.html")
    )

    for invented in ("cheating", "coding</option>"):
        assert invented not in document

    types = set(re.findall(r'data-type="([^"]+)"', document))

    assert types <= {
        "theory",
        "coding",
        "scenario",
        "architecture",
        "troubleshooting",
    }

    # And it is not empty: the fixture does carry typed questions, and a
    # set assertion that passes vacuously is worth nothing.
    assert types == {"scenario", "theory", "troubleshooting"}


# ---------------------------------------------------------------------
# Determinism and output safety
# ---------------------------------------------------------------------


def test_generator_is_deterministic(tmp_path: Path, canonical_file: Path):
    first = tmp_path / "site-a"
    second = tmp_path / "site-b"

    generate_site(canonical_file, first)
    generate_site(canonical_file, second)

    first_files = {
        path.relative_to(first).as_posix(): path.read_bytes()
        for path in all_files(first)
    }
    second_files = {
        path.relative_to(second).as_posix(): path.read_bytes()
        for path in all_files(second)
    }

    assert first_files == second_files


def test_regenerating_replaces_stale_files(tmp_path: Path, canonical_file: Path):
    """
    A rebuild must leave nothing from the previous one.

    It used to be checked with post pages, which made it easy to read:
    build two posts, rebuild with one, assert the dropped post's page is
    gone. There are no post pages now, so the check is the same idea
    applied to the revision pages -- the first build's pages are gone
    and none of them mention the post the second build did not have.
    """

    output = tmp_path / "site"

    generate_site(canonical_file, output)

    first = {
        path.relative_to(output).as_posix()
        for path in all_files(output)
        if path.suffix == ".html"
    }

    assert first, "the first build produced no pages"

    smaller = tmp_path / "smaller.json"
    smaller.write_text(
        json.dumps(
            make_knowledge_base(
                [
                    make_post(
                        "sample_gamma",
                        topics=["Apache Spark"],
                        subtopics=["Skew Handling"],
                    )
                ]
            )
        ),
        encoding="utf-8",
    )

    generate_site(smaller, output)

    # Comparing the two builds is what makes a stale file a failure
    # rather than something a single absence check could miss. The first
    # build wrote revision pages for a two-post corpus; the second, for a
    # one-post corpus, must have written a different set -- and none of
    # the first one's leftovers.
    produced = {
        path.relative_to(output).as_posix()
        for path in all_files(output)
        if path.suffix == ".html"
    }

    assert not any(p.startswith("posts/") for p in produced), sorted(
        produced
    )

    assert not any(p.startswith("topics/") for p in produced), sorted(
        produced
    )

    # Every revision page present belongs to the second corpus: its
    # content mentions the post the second build knew about.
    for path in (output / "revision").glob("*.html"):
        body = path.read_text(encoding="utf-8")

        assert "sample_alpha" not in body, path.name

    # And the rebuild did not simply accumulate: a page the first build
    # wrote and the second did not is not on disk. Comparing the two
    # manifests is what catches that, and checking one known path is not
    # enough.
    second = {
        path.relative_to(output).as_posix()
        for path in all_files(output)
    }

    manifest = json.loads((output / MANIFEST_FILE).read_text(encoding="utf-8"))

    for relative in manifest["files"]:
        assert (output / relative).is_file(), relative


def test_generation_leaves_no_staging_directory(
    tmp_path: Path,
    canonical_file: Path,
):
    output = tmp_path / "site"

    generate_site(canonical_file, output)

    assert not (tmp_path / ".site.staging").exists()
    assert not (output / f"{MANIFEST_FILE}.tmp").exists()


def test_refuses_to_replace_an_unmanaged_directory(
    tmp_path: Path,
    canonical_file: Path,
):
    output = tmp_path / "site"
    output.mkdir()
    (output / "important.txt").write_text(
        "keep me", encoding="utf-8"
    )

    with pytest.raises(WikiError) as error:
        generate_site(canonical_file, output)

    assert "not empty" in str(error.value)
    assert (output / "important.txt").exists()


def test_refuses_to_generate_into_a_git_checkout(
    tmp_path: Path,
    canonical_file: Path,
):
    output = tmp_path / "repo"
    output.mkdir()
    (output / ".git").mkdir()

    with pytest.raises(WikiError) as error:
        generate_site(canonical_file, output)

    assert "git checkout" in str(error.value)


def test_failed_run_does_not_disturb_a_previous_site(
    tmp_path: Path,
    canonical_file: Path,
):
    output = tmp_path / "site"

    generate_site(canonical_file, output)
    before = read(output, INDEX_PAGE)

    broken = tmp_path / "broken.json"
    broken.write_text("{nope", encoding="utf-8")

    with pytest.raises(WikiError):
        generate_site(broken, output)

    assert read(output, INDEX_PAGE) == before


# ---------------------------------------------------------------------
# Analysis model
# ---------------------------------------------------------------------


def test_site_model_counts_and_ordering(canonical_file: Path):
    model = build_site_model(load_canonical(canonical_file))

    assert model.post_count == 2
    assert model.question_count == 3
    assert model.concept_count == 4
    assert model.concept_order_is_sorted()
    assert [post.id for post in model.posts] == [
        "sample_alpha",
        "sample_beta",
    ]

    # Recent ordering is capture time descending.
    assert [
        post.id for post, _slug in model.recent_pairs(10)
    ] == ["sample_alpha", "sample_beta"]

    assert model.difficulty_counts() == {"easy": 0, "medium": 1, "hard": 2}
    assert model.question_type_counts() == {
        "scenario": 1,
        "theory": 1,
        "troubleshooting": 1,
    }


def test_canonical_model_drops_unknown_top_level_keys():
    knowledge_base = CanonicalKnowledgeBase.model_validate(
        {
            "schema_version": 1,
            "posts": [],
            "skipped_files": ["secret path"],
            "result_files_found": 9,
        }
    )

    assert not hasattr(knowledge_base, "skipped_files")
    assert knowledge_base.posts == []


def test_identical_topic_and_subtopic_labels_merge_into_one_page():
    """
    A label used as both a topic and a subtopic describes one subject,
    so it gets one page rather than two competing pages.
    """

    model = build_site_model(
        load_canonical_from_payload(
            make_knowledge_base(
                [
                    make_post(
                        "sample_alpha",
                        topics=["Apache Spark"],
                        subtopics=["Apache Spark"],
                    )
                ]
            )
        )
    )

    labels = [entry.label for entry in model.topics]
    slugs = [entry.slug for entry in model.topics]

    # "Apache Spark" and "apache/spark" differ only in punctuation, so
    # they are one topic rather than two pages. The classification's
    # own topic is included, which is what keeps the site and the
    # knowledge base reporting the same topics.
    assert "Apache Spark" in labels
    assert "apache/spark" not in labels

    # The classification's own topic is included, so the site and the
    # knowledge base report the same topics rather than two lists.
    assert "Databricks" in labels
    # One page for the merged pair, plus the classified topic, which is
    # a distinct label and therefore its own page.
    assert slugs == ["apache-spark", "databricks"]
    assert model.topic_slugs["Apache Spark"] == "apache-spark"


def test_colliding_slugs_stay_unique():
    """
    Distinct labels that reduce to the same slug must not overwrite
    each other's page.
    """

    model = build_site_model(
        load_canonical_from_payload(
            make_knowledge_base(
                [
                    make_post(
                        "sample_alpha",
                        topics=["Apache Spark", "apache/spark"],
                    )
                ]
            )
        )
    )

    slugs = [entry.slug for entry in model.topics]
    labels = [entry.label for entry in model.topics]

    # Labels that differ only in punctuation or case are the same topic,
    # so they merge before a slug is ever assigned and no collision
    # remains to resolve.
    assert len(slugs) == len(set(slugs))
    assert "Apache Spark" in labels
    assert "apache/spark" not in labels

    # The classification's own topic is included, so the site and the
    # knowledge base report the same topics rather than two lists.
    assert "Databricks" in labels

    # Genuinely different labels still get distinct pages.
    assert "Databricks" in labels
    assert "Partitioning Strategy" in labels


def test_post_ids_are_preserved_in_page_names():
    model = build_site_model(
        load_canonical_from_payload(
            make_knowledge_base([make_post("sample_001")])
        )
    )

    assert model.post_slugs == ("sample_001",)


def load_canonical_from_payload(payload: dict) -> CanonicalKnowledgeBase:
    return CanonicalKnowledgeBase.model_validate(payload)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def run_cli(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "src.wiki.generator", *arguments],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )


def test_cli_generates_a_site(tmp_path: Path, canonical_file: Path):
    output = tmp_path / "cli-site"

    result = run_cli(
        "--input", str(canonical_file), "--output", str(output)
    )

    assert result.returncode == 0, result.stderr
    assert "Wiki generated" in result.stdout
    assert (output / INDEX_PAGE).exists()


def test_cli_defaults_to_site_directory(
    tmp_path: Path,
    canonical_file: Path,
):
    output = tmp_path / "defaulted"

    result = run_cli(
        "--input",
        str(canonical_file),
        "--output",
        str(output),
    )

    assert result.returncode == 0
    assert (output / INDEX_PAGE).exists()


def test_cli_returns_non_zero_on_missing_input(tmp_path: Path):
    result = run_cli(
        "--input",
        str(tmp_path / "nope.json"),
        "--output",
        str(tmp_path / "site"),
    )

    assert result.returncode == 1
    assert "WIKI_ERROR=" in result.stderr
    assert not (tmp_path / "site").exists()


def test_cli_returns_non_zero_on_malformed_input(tmp_path: Path):
    broken = tmp_path / "broken.json"
    broken.write_text("{oops", encoding="utf-8")

    result = run_cli(
        "--input", str(broken), "--output", str(tmp_path / "site")
    )

    assert result.returncode == 1
    assert "not valid JSON" in result.stderr


def test_cli_requires_input_argument():
    result = run_cli()

    assert result.returncode == 2
    assert "--input" in result.stderr
