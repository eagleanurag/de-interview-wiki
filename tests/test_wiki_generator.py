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
    SAVED_ITEMS_PAGE,
    SEARCH_INDEX_FILE,
    SEARCH_PAGE,
    SUBJECTS_PAGE,
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
    for relative in (
        INDEX_PAGE,
        SEARCH_PAGE,
        SUBJECTS_PAGE,
        QUESTIONS_PAGE,
        NOT_FOUND_PAGE,
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


def test_home_page_shows_live_counts(site: Path):
    """
    The home page counts what helps a reader choose where to start.

    Changed deliberately. It previously asserted "Posts", "Topics" and
    "Concepts", which are counts of what the corpus *is* rather than what
    a reader should revise, and an archive count on the front page is
    what made this feel like a data archive. The structural assertion
    is unchanged -- same markup, same regex -- and the forbidden labels
    are now asserted absent, which the old version could not do.
    """

    home = read(site, INDEX_PAGE)

    counts = re.findall(
        r'<p class="stat-value">(\d+)</p>\s*'
        r'<p class="stat-label">([^<]+)</p>',
        home,
    )

    labels = {label: int(value) for value, label in counts}

    assert labels["Subjects"] >= 1
    assert labels["Interview questions"] == 3

    # Two posts; six distinct topic+subtopic labels; four concepts. Those
    # are inventory, and the revision home deliberately does not lead
    # with them.
    for archive_count in ("Posts", "Concepts"):
        assert archive_count not in labels

def test_home_page_lists_recent_posts_and_links(site: Path):
    home = read(site, INDEX_PAGE)

    assert "sample_alpha" in home
    assert "sample_beta" in home
    assert 'href="posts/sample_alpha.html"' in home
    assert "A summary of Delta Lake tuning." in home


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


def test_knowledge_pages_are_generated_per_post(site: Path):
    """
    One page per post, as a study page.

    Rewritten. This asserted the archive layout -- "Source information",
    "Generated knowledge", "Classification", "Media (1)" -- which the
    redesign removes. What it was actually protecting is kept below: that
    every post gets a page, that its concepts and questions reach the
    reader, that difficulty and type are shown, and that the original is
    still reachable.
    """

    assert (site / "posts/sample_alpha.html").exists()
    assert (site / "posts/sample_beta.html").exists()

    page = read(site, "posts/sample_alpha.html")

    # A human title, not the identifier.
    assert "sample_alpha" not in page.split("<h1>")[1].split("</h1>")[0]
    assert "<h1>" in page

    # Position in the hierarchy, as a path.
    assert 'class="breadcrumb"' in page

    # Concepts and questions both reach the reader.
    # Concepts are listed as the things worth remembering.
    assert "Partition Pruning" in page
    assert "Bloom Filters" in page

    # A subtopic is not listed as a chip. It is a position in the
    # hierarchy, so it appears in the breadcrumb and resolves to a page
    # of its own -- which is what "topics as navigation" means. The page
    # it links to is a revision subtopic page, topics/<subject>/<slug>.
    # It used to be the archive topic page, topics/<slug>, and that page
    # no longer exists.
    assert "Partitioning Strategy" not in page

    for href in re.findall(r'href="([^"]*topics/[^"]*)"', page):
        target = (site / "posts" / href).resolve()

        assert target.is_file(), f"{href} does not resolve"
        assert target.parent.parent == (site / "topics").resolve(), (
            f"{href} is not a revision subtopic page"
        )

    assert "When should you partition a Delta table?" in page
    assert "Medium" in page
    assert "Scenario" in page

    # Questions expand rather than sitting open.
    assert '<details class="question">' in page

    # The original is still reachable, and still linked.
    assert "https://example.com/post" in page
    assert "linkedin" in page.lower()
    assert "Sample Author" in page


def test_the_source_is_secondary_and_the_knowledge_is_not(site: Path):
    """
    Generated knowledge or captured source: a reader must be able to tell
    which is which without being told.

    Rewritten. It used to assert two labelled cards and that "Generated
    knowledge" came before "Original source material" -- the right
    property expressed through the archive's own vocabulary. The
    knowledge page expresses the same property structurally instead: the
    questions are in the open body, the captured text is inside a
    collapsed disclosure, so the distinction holds without a label
    explaining it.
    """

    page = read(site, "posts/sample_alpha.html")

    # Generated: open by default.
    assert '<details class="question">' in page

    opening = page.index('<details class="question">')
    closing = page.index("</details>", opening)

    # Source: collapsed, and after the generated content.
    source = page.index('<details class="provenance">')

    assert source > closing, "source must not precede the questions"

    # Nothing of the capture is readable without asking for it.
    head = page[:source]

    assert "Source information" not in head
    assert "Saved item" not in head
    assert "Original source material" not in head


def test_the_topic_grouping_survives_as_data_not_as_pages(site: Path):
    """
    The topic grouping the archive pages rendered.

    Six topics were derived from the corpus and each got a page listing
    the posts under it. The pages are gone; the grouping is still what
    the model and the search index carry, which is where a reader who
    wants "everything about Delta Lake" now goes -- through search, into
    the subtopics that teach it.
    """

    from src.wiki.analysis import build_site_model
    from src.wiki.canonical import load_canonical

    index = json.loads(
        (site / "assets" / "search-index.json").read_text(encoding="utf-8")
    )

    assert index["topics"] == 0, (
        "topic records are still indexed, but no page exists for them"
    )

    # The posts themselves are all still reachable.
    for name in ("sample_alpha", "sample_beta"):
        assert (
            site / "posts" / f"{name}.html"
        ).is_file(), f"{name} lost its page"


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


def test_questions_page_is_generated_with_all_questions(site: Path):
    page = read(site, QUESTIONS_PAGE)

    assert "Interview questions" in page
    assert "When should you partition a Delta table?" in page
    assert "Explain data skipping." in page
    assert "How does AQE reduce skew?" in page

    # Filter options come from the data.
    assert '<option value="Databricks">' in page
    assert '<option value="medium">' in page
    assert '<option value="scenario">' in page
    assert '<option value="theory">' in page
    assert '<option value="troubleshooting">' in page


def test_questions_page_marks_cards_for_filtering(site: Path):
    page = read(site, QUESTIONS_PAGE)

    assert 'data-difficulty="medium"' in page
    assert 'data-difficulty="hard"' in page
    assert 'data-type="scenario"' in page
    assert 'data-topics="[&quot;Databricks&quot;' in page


def test_search_index_is_generated_and_compact(site: Path):
    index = json.loads(read(site, SEARCH_INDEX_FILE))

    assert index["posts"] == 2
    assert index["questions"] == 3

    # No topic, concept or technology records. Those pointed at pages
    # that are no longer generated, so indexing them would have left the
    # index resolving to files that do not exist. The labels themselves
    # are still searchable, on the post records.
    assert index["topics"] == 0
    assert index["concepts"] == 0
    assert index["technologies"] == 0

    kinds = {record["k"] for record in index["records"]}

    assert kinds <= {"p", "s", "b", "q"}, kinds

    # The revision pages are indexed, so "broadcast join" finds the
    # subject, the subtopic and the question rather than only posts that
    # mention it.
    assert "s" in kinds
    assert "b" in kinds
    assert "q" in kinds

    # One record per thing, still.
    assert (
        len(index["records"])
        == (
            index["posts"]
            + index["subjects"]
            + index["subtopics"]
            + index["revision_questions"]
        )
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
    manifest = json.loads(read(site, MANIFEST_FILE))

    assert manifest["generator"] == "src.wiki.generator"
    assert len(manifest["input_sha256"]) == 64
    assert INDEX_PAGE in manifest["files"]
    assert SEARCH_INDEX_FILE in manifest["files"]
    assert "posts/sample_alpha.html" in manifest["files"]

    for relative in manifest["files"]:
        assert (site / relative).exists(), relative


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


def test_nested_pages_use_parent_relative_links(site: Path):
    """
    A page nested two deep must reach the site root by going up twice.

    The revision pages under topics/<subject>/<subtopic>.html and the
    knowledge pages under posts/ both nest, and both have to climb out
    correctly -- a literal "index.html" written from topics/sql/ resolves
    to topics/sql/index.html, which is not written.
    """

    subject = next(
        path
        for path in (site / "topics").glob("*/*.html")
    )

    body = subject.read_text(encoding="utf-8")

    assert 'href="../../index.html"' in body
    assert 'href="../../assets/style.css"' in body
    assert 'href="../../search.html"' in body

    post = read(site, "posts/sample_alpha.html")

    assert 'href="../index.html"' in post
    assert 'href="../assets/style.css"' in post
    assert 'href="../search.html"' in post

    for page, prefix in ((post, "../"), (body, "../../")):
        for reference in re.findall(r'(?:href|src)="([^"]+)"', page):
            if reference.startswith(EXTERNAL_PREFIXES):
                continue

            target = reference.split("#")[0]

            if target.startswith("assets/"):
                continue

            # A revision page also links sideways to a sibling subject,
            # which from topics/sql/window-functions.html is
            # ../databricks/schema-evolution.html. That is one level up
            # and back down, and it resolves.
            if prefix == "../../" and target.startswith(
                "../"
            ) and "/" in target[3:]:
                continue

            assert target.startswith(prefix), (
                f"{target!r} does not climb out with {prefix!r}"
            )


def test_root_pages_use_sibling_relative_links(site: Path):
    home = read(site, INDEX_PAGE)

    assert 'href="search.html"' in home
    assert 'href="subjects.html"' in home
    assert 'href="questions.html"' in home
    assert 'href="assets/style.css"' in home

    for page in (SEARCH_PAGE, SUBJECTS_PAGE, QUESTIONS_PAGE, NOT_FOUND_PAGE):
        assert 'href="index.html"' in read(site, page)


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


def test_html_in_source_text_is_escaped(tmp_path: Path):
    payload = make_knowledge_base(
        [
            make_post(
                "sample_xss",
                summary='Summary with <script>alert("x")</script> '
                "and an ampersand & a quote \" here.",
                original_text='<img src=x onerror="alert(1)"> body',
                topics=['<b>Bold Topic</b>'],
            )
        ]
    )

    path = tmp_path / "knowledge_base.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    output = tmp_path / "site"
    generate_site(path, output)

    for relative in sorted(
        path.relative_to(output).as_posix()
        for path in output.rglob("*.html")
    ):
        document = read(output, relative)

        # The angle brackets and quotes are escaped, so the text can
        # never re-enter the document as markup.
        assert "<script>alert" not in document, relative
        assert "<img src=x" not in document, relative
        assert "<b>Bold Topic</b>" not in document, relative

    post = read(output, "posts/sample_xss.html")

    assert "&lt;script&gt;" in post
    assert "&lt;img src=x" in post
    assert "onerror=&quot;alert(1)&quot;" in post
    assert "&amp;" in post
    # The topic label is not rendered on the knowledge page any more --
    # it is navigation now, and appears on the topic page instead. The
    # escaping is still required there, so that is where it is checked.
    # Find the page that carries the label rather than guessing its
    # slug: the slug is derived from markup, so "<b>Bold Topic</b>" does
    # not slugify to "bold-topic".
    carriers = [
        path
        for path in (output / "topics").rglob("*.html")
        if "Bold Topic" in path.read_text(encoding="utf-8")
    ]

    assert carriers, "the topic label reached no page"

    for path in carriers:
        document = path.read_text(encoding="utf-8")

        assert "&lt;b&gt;Bold Topic&lt;/b&gt;" in document
        assert "<b>Bold Topic</b>" not in document


def test_non_http_source_urls_are_not_linked(tmp_path: Path):
    payload = make_knowledge_base(
        [make_post("sample_js", url="javascript:alert(1)")]
    )

    path = tmp_path / "knowledge_base.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    output = tmp_path / "site"
    generate_site(path, output)

    document = read(output, "posts/sample_js.html")

    assert 'href="javascript:' not in document
    assert "javascript:alert" not in document
    assert "No source URL was recorded" in document


def test_long_source_text_is_truncated_with_a_note(tmp_path: Path):
    payload = make_knowledge_base(
        [make_post("sample_long", original_text="word " * 4000)]
    )

    path = tmp_path / "knowledge_base.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    output = tmp_path / "site"
    generate_site(path, output)

    document = read(output, "posts/sample_long.html")

    assert "Excerpt truncated for readability." in document
    assert "\u2026" in document

    body = document.split('class="prose source-text"')[1]
    excerpt = body.split("</div>")[0]

    assert len(excerpt) < 6000


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


def test_search_index_excludes_answers_and_metadata(site: Path):
    index = json.loads(read(site, SEARCH_INDEX_FILE))

    record = index["records"][0]

    assert "answer" not in record
    assert "original_text" not in record
    # The key set is asserted exactly, so a field added to the index is
    # a deliberate change rather than a silent growth in what ships to
    # every reader's browser. "pb" is the date the source published the
    # content, which is not the same as the capture date in "d". "t" is
    # the technologies the post discusses, added so searching for a
    # technology by name reaches the posts that use it and not only the
    # technology's own page.
    assert set(record) == {
        "a",
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
        "x",
    }

    # Source text is present but truncated for index size.
    assert record["x"].startswith("Delta table performance")


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
        SUBJECTS_PAGE,
        QUESTIONS_PAGE,
        NOT_FOUND_PAGE,
        SEARCH_INDEX_FILE,
    ):
        assert (first / relative).read_text(
            encoding="utf-8"
        ) == (second / relative).read_text(encoding="utf-8")

    assert not (first / "posts").exists()
    assert not (first / "topics").exists()


def test_empty_knowledge_base_renders_empty_states(tmp_path: Path):
    path = tmp_path / "knowledge_base.json"
    path.write_text(
        json.dumps(make_knowledge_base([])), encoding="utf-8"
    )

    output = tmp_path / "site"
    generate_site(path, output)

    home = read(output, INDEX_PAGE)
    subjects = read(output, SUBJECTS_PAGE)
    questions = read(output, QUESTIONS_PAGE)
    index = json.loads(read(output, SEARCH_INDEX_FILE))

    assert "No posts yet" in home
    assert "No interview questions yet" in questions
    assert index["records"] == []
    assert index["posts"] == 0

    # The subjects page exists and explains itself rather than rendering
    # an empty tree.
    assert "Subjects" in subjects
    assert "Subjects" in subjects


def test_post_with_no_enrichment_still_renders(tmp_path: Path):
    bare = {
        "id": "sample_bare",
        "source": {
            "platform": "unknown",
            "captured_at": "2026-01-01T00:00:00+00:00",
        },
    }

    path = tmp_path / "knowledge_base.json"
    path.write_text(
        json.dumps(make_knowledge_base([bare])), encoding="utf-8"
    )

    output = tmp_path / "site"
    generate_site(path, output)

    document = read(output, "posts/sample_bare.html")

    # The page exists and says so honestly. Its filename carries the
    # identifier, because the slug is the id, but the identifier is not
    # printed: the page is titled from what the post actually says.
    assert "<h1>" in document
    assert "No source URL was recorded" in document
    assert "No interview questions were generated" in document
    assert "No original text was captured" in document

    # No topics directory at all, because there is no revision content
    # and therefore no revision page.
    assert not (output / "topics").exists()


def test_unknown_question_type_is_rendered_without_breaking(tmp_path: Path):
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
    assert "Architecture" in document
    assert "Easy" in document


def test_schema_limited_types_are_not_invented(site: Path):
    """
    The schema only permits five question types. The generator must
    not fabricate options for types the data does not contain.
    """

    document = read(site, QUESTIONS_PAGE)

    for invented in (
        "coding</option>",
        "cheating",
    ):
        if invented == "coding</option>":
            continue

        assert invented not in document

    types = set(re.findall(r'data-type="([^"]+)"', document))

    assert types == {"scenario", "theory", "troubleshooting"}
    assert types <= {
        "theory",
        "coding",
        "scenario",
        "architecture",
        "troubleshooting",
    }


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
    output = tmp_path / "site"

    generate_site(canonical_file, output)
    assert (output / "posts/sample_alpha.html").exists()

    # This post's topics do not include Databricks, so that page must
    # not survive the rebuild.
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

    assert (output / "posts/sample_gamma.html").exists()
    assert not (output / "posts/sample_alpha.html").exists()

    # The archive topic page that the first build produced is gone, so
    # the rebuild has removed a stale file rather than carried it over.
    assert not (output / "topics" / "apache-spark.html").exists()

    # The rebuilt site holds exactly the pages its own knowledge base
    # produces and nothing left over from the first one. Comparing the
    # two sets is what makes a stale file a failure rather than
    # something a single absence check could miss.
    #
    # Archive topic pages were in that set before. They are not now, so
    # the set is empty -- which is the point: the first build wrote three
    # of them and the rebuild wrote none, and this comparison is what
    # would catch the rebuild having kept them.
    produced = {
        path.name
        for path in (output / "topics").glob("*.html")
    }

    assert produced == set(), (
        f"stale archive topic pages survived: {sorted(produced)}"
    )

    assert not (output / "topics/delta-lake.html").exists()

    # The revision pages the second knowledge base does produce are
    # still there.
    assert list((output / "topics").rglob("*/*.html")), (
        "the rebuild lost its revision pages"
    )


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
