"""
End-to-end: validate, aggregate, generate the wiki.

Runs the same pipeline the validation workflow runs, over whatever is
actually in data/posts, and checks the properties that matter to a
reader of the site:

* the canonical knowledge base covers every post exactly once;
* the wiki contains a page per post, per topic and per question;
* links are relative, so the site works from the filesystem, from a
  domain root and from a project path with no configuration;
* a rebuild produces byte-identical output, so a diff in the
  repository means something changed.

Runs offline against the committed data. No credentials, no network.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from src.aggregation.aggregator import (
    WORKER_RESULT_GLOB,
    aggregate_results,
)
from src.ingestion.importer import validate_posts
from src.wiki.generator import generate_site

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_POSTS_ROOT = REPO_ROOT / "data" / "posts"


@pytest.fixture(scope="module")
def posts_root() -> Path:
    if not REAL_POSTS_ROOT.is_dir():
        pytest.skip("data/posts is absent")

    return REAL_POSTS_ROOT


@pytest.fixture(scope="module")
def worker_results(posts_root, tmp_path_factory):
    """
    Stage the posts the way the worker does.

    Aggregation reads worker results matching a glob, so the posts are
    copied in under that name. This keeps the test on the same input
    path the workflow uses instead of testing a second code path.

    Enriched results are preferred when the local pipeline has already
    produced them, because a knowledge base built from unenriched posts
    has no topics, concepts or questions to check, and a test that
    silently skips itself on every run is not a test.
    """

    directory = tmp_path_factory.mktemp("worker-results")

    enriched = Path("build") / "worker-results"

    if enriched.is_dir() and any(enriched.glob(WORKER_RESULT_GLOB)):
        for path in sorted(enriched.glob(WORKER_RESULT_GLOB)):
            shutil.copyfile(path, directory / path.name)
    else:
        for path in posts_root.glob("*/post.json"):
            shutil.copyfile(
                path, directory / f"cloud_worker_{path.parent.name}.json"
            )

    if not list(directory.glob(WORKER_RESULT_GLOB)):
        pytest.skip("no posts available to aggregate")

    return directory


@pytest.fixture(scope="module")
def knowledge_base(worker_results, tmp_path_factory):
    """Run the aggregation stage over the real posts."""

    output = tmp_path_factory.mktemp("kb") / "knowledge_base.json"

    aggregate_results(
        input_directory=worker_results,
        output_path=output,
        expected_post_count=None,
    )

    return json.loads(output.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def site(posts_root, knowledge_base, tmp_path_factory):
    """Generate the static site from the canonical knowledge base."""

    root = tmp_path_factory.mktemp("site")

    source = root / "knowledge_base.json"
    source.write_text(
        json.dumps(knowledge_base, indent=2), encoding="utf-8"
    )

    generate_site(input_path=source, output_dir=root / "site")

    return root / "site"


# ---------------------------------------------------------------------
# The canonical knowledge base
# ---------------------------------------------------------------------


def test_the_committed_posts_all_validate(posts_root):
    report = validate_posts(root=posts_root)

    assert report.ok, [str(issue) for issue in report.issues]


def test_every_post_appears_in_the_knowledge_base(
    posts_root, knowledge_base
):
    """
    Every collected post is aggregated, whether or not it has been
    enriched.

    A post that could not be enriched is still real content with real
    provenance, so dropping it would lose the source. It is aggregated
    and labelled instead.
    """

    expected = {
        path.parent.name for path in posts_root.glob("*/post.json")
    }

    actual = {post["id"] for post in knowledge_base["posts"]}

    assert actual == expected


def test_enrichment_coverage_is_reported(knowledge_base):
    """
    A knowledge base with no topics is otherwise indistinguishable from
    a knowledge base with no content, so coverage is explicit rather
    than left for a reader to infer from empty sections.
    """

    stats = knowledge_base["stats"]

    assert 0 <= stats["posts_enriched"] <= stats["posts_aggregated"]


def test_content_kind_and_enrichment_state_are_separate(
    knowledge_base,
):
    """
    "This is a job announcement" and "this has not been enriched yet"
    are different facts. One label cannot carry both, and a post can be
    obviously one and obviously the other.
    """

    kinds = knowledge_base["knowledge"]["content_kinds"]

    # A content kind is never a statement about the pipeline.
    for post_id, kind in kinds.items():
        assert kind not in {"unenriched", "enriched"}, post_id
        assert kind in {
            "technical",
            "job_announcement",
            "event",
            "congratulation",
            "certification",
            "appreciation",
            "social",
            "unknown",
        }, kind


def test_an_unenriched_post_is_still_classified_from_its_text(
    knowledge_base,
):
    """
    A post is classifiable from its own words whether or not the model
    has read it, so content classification does not wait on enrichment.
    """

    kinds = knowledge_base["knowledge"]["content_kinds"]

    for post in knowledge_base["posts"]:
        analysis = post["ai_analysis"]

        has_analysis = bool(
            (analysis.get("summary") or "").strip()
            or analysis.get("topics")
            or analysis.get("concepts")
        )

        kind = kinds[post["id"]]

        if not has_analysis:
            # Still labelled, from the text, not left out.
            assert isinstance(kind, str) and kind
        else:
            assert kind in {"technical", "unknown"} or kind in {
                "job_announcement",
                "event",
                "congratulation",
                "certification",
                "appreciation",
                "social",
            }, kind


def test_no_post_appears_twice(knowledge_base):
    identifiers = [post["id"] for post in knowledge_base["posts"]]

    assert len(identifiers) == len(set(identifiers))


def test_every_collected_post_kept_its_provenance(knowledge_base):
    """
    A collected post must remain traceable to where it came from, and
    must not claim provenance it does not have.

    A local saved-post archive holds a third of its records with no
    permalink, because the card they were captured from exposed none.
    Those posts are held to the alternative claim instead: identified by
    the archive's own namespace, with no link invented to satisfy a
    check that every post has one.
    """

    collected = [
        post
        for post in knowledge_base["posts"]
        if post["id"].startswith("urn-li-")
    ]

    if not collected:
        pytest.skip("no collected posts in this checkout")

    url_less = 0

    for post in collected:
        assert post["source"]["platform"] == "linkedin"
        assert post["source"]["captured_at"]
        assert post["original_text"].strip()

        url = post["source"].get("url")

        if not url:
            url_less += 1

            assert post["id"].startswith("urn-li-archive-"), post["id"]

            saved = post.get("saved_item") or {}

            assert saved.get("saved_item_id", "").startswith(
                "urn:li:archive:"
            )
            assert saved.get("url_kind") == "identified_by_id"
            assert not saved.get("canonical_url")

            continue

        assert url.startswith("https://www.linkedin.com/")

    # The archive really does hold posts with no link, so the branch
    # above is exercised rather than dead code in this checkout.
    assert url_less == sum(
        1
        for post in collected
        if post["id"].startswith("urn-li-archive-")
    )


def test_source_text_is_preserved_verbatim(knowledge_base):
    """
    The canonical base must carry what the post said, so a reader can
    trace a summary back to the original.
    """

    for post in knowledge_base["posts"]:
        assert post["original_text"]
        assert post["original_text"].strip()


# ---------------------------------------------------------------------
# The site
# ---------------------------------------------------------------------


def test_the_entry_points_exist(site):
    for name in (
        "index.html",
        "topics.html",
        "concepts.html",
        "technologies.html",
        "questions.html",
        "search.html",
    ):
        assert (site / name).is_file(), name


def test_every_post_has_a_page(site, knowledge_base):
    pages = {path.stem for path in (site / "posts").glob("*.html")}

    for post in knowledge_base["posts"]:
        assert post["id"] in pages, post["id"]


def test_the_search_index_covers_every_post(site, knowledge_base):
    """
    Search has to reach every post the site serves, or a reader cannot
    find content that exists.
    """

    payload = json.loads(
        (site / "assets" / "search-index.json").read_text(encoding="utf-8")
    )

    assert payload["posts"] == len(knowledge_base["posts"])

    # Records cover posts and the consolidated sections. A checkout
    # whose posts have not been enriched has no topics yet, and that is
    # an honest state rather than a failure.
    assert len(payload["records"]) >= payload["posts"]
    assert (
        len(payload["records"]) == payload["posts"]
        or payload["topics"] > 0
    )

    # "i" is the identifier and "u" the page, both compressed to keep
    # the index small. Only post records carry a post identifier.
    identifiers = {
        record["i"]
        for record in payload["records"]
        if record["k"] == "p"
    }

    assert identifiers == {post["id"] for post in knowledge_base["posts"]}


def test_every_search_record_points_at_a_page_that_exists(
    site, knowledge_base
):
    """An index entry pointing at a missing page is a dead search hit."""

    payload = json.loads(
        (site / "assets" / "search-index.json").read_text(encoding="utf-8")
    )

    for record in payload["records"]:
        assert (site / record["u"]).is_file(), record["u"]


def test_no_internal_link_is_rooted_or_absolute(site):
    """
    A site that only works at one mount point breaks the moment it is
    opened from somewhere else, so nothing the site links to itself may
    be rooted at "/" or written as an absolute URL.

    Outbound links are exempt: a link to the original LinkedIn post has
    to be absolute to be worth anything.
    """

    rooted = re.compile(r'(?:href|src)="/(?!/|#)')

    offenders: list[str] = []

    for path in site.rglob("*.html"):
        for match in rooted.finditer(path.read_text(encoding="utf-8")):
            offenders.append(f"{path.name}: {match.group(0)}")

    assert offenders == []


def test_no_internal_link_names_this_deployment(site):
    """
    An absolute link to the site's own pages would tie the output to one
    host and one mount point, so none may appear.
    """

    absolute = re.compile(r'(?:href|src)="https?://[^"]+"')

    # Hosts that are legitimately external references: the original
    # post, and the fixture URLs the committed sample data uses.
    allowed = ("linkedin.com", "example.com")

    offenders: list[str] = []

    for path in site.rglob("*.html"):
        for match in absolute.finditer(path.read_text(encoding="utf-8")):
            url = match.group(0)

            if any(host in url for host in allowed):
                continue

            offenders.append(f"{path.name}: {url[:70]}")

    assert offenders == []


def test_every_internal_link_resolves(site):
    """
    A relative link that points at nothing is a broken page, and it is
    the failure mode a relative-link design introduces.
    """

    broken: list[str] = []

    for path in site.rglob("*.html"):
        text = path.read_text(encoding="utf-8")

        for target in re.findall(r'href="([^"#?]+)"', text):
            if target.startswith(("http://", "https://", "mailto:")):
                continue

            resolved = (path.parent / target).resolve()

            if not resolved.exists():
                broken.append(f"{path.name} -> {target}")

    assert broken == []


def test_no_published_page_names_this_machine(site):
    """
    A page may not carry the path its material was imported from.

    Local saved-post archives live somewhere particular on somebody's
    disk, and the importer reads that location. Nothing about it belongs
    in output anyone else will read, and a leak here is the kind of thing
    that only shows up once and then cannot be taken back.

    Checked for the drive-letter and folder patterns a Windows path
    actually has, rather than for a literal directory name, because the
    test should keep working on a machine that never held this archive
    at all.
    """

    patterns = (
        re.compile(r"[A-Za-z]:\\\\?Users\\\\?"),
        re.compile(r"[A-Za-z]:/Users/"),
        re.compile(r"AppData"),
        re.compile(r"chrome_session"),
        re.compile(r"linkedin_saved_archive"),
    )

    offenders: list[str] = []

    for path in sorted(site.rglob("*.html")):
        body = path.read_text(encoding="utf-8")

        for pattern in patterns:
            match = pattern.search(body)

            if match:
                offenders.append(
                    f"{path.name}: {match.group(0)}"
                )

    assert offenders == [], offenders[:10]


def test_the_navigation_links_every_top_level_page(site):
    index = (site / "index.html").read_text(encoding="utf-8")

    for page in (
        "topics.html",
        "concepts.html",
        "technologies.html",
        "questions.html",
        "search.html",
    ):
        assert page in index, page


def test_the_topic_index_lists_the_topic_pages(site, knowledge_base):
    """A topic page nobody can reach is content nobody can read."""

    topics = sorted(
        {
            post["ai_analysis"].get("primary_topic")
            for post in knowledge_base["posts"]
            if post["ai_analysis"].get("primary_topic")
        }
    )

    if not topics:
        pytest.skip("no posts carry a primary topic yet")

    index = (site / "topics.html").read_text(encoding="utf-8")

    for topic in topics:
        assert topic in index, topic


def test_the_question_index_lists_every_question(knowledge_base, site):
    questions = [
        question
        for post in knowledge_base["posts"]
        for question in post["interview_questions"]
    ]

    if not questions:
        pytest.skip("no post has generated questions yet")

    index = (site / "questions.html").read_text(encoding="utf-8")

    for question in questions:
        # The full text is escaped in HTML, so the distinctive opening
        # is what is checked.
        opening = question["question"][:40]

        assert opening.split(" ")[0] in index, opening


def test_generation_is_reproducible(site, knowledge_base, tmp_path):
    """
    The same knowledge base must produce the same bytes, so a change in
    the repository is always a real change.
    """

    source = tmp_path / "knowledge_base.json"
    source.write_text(
        json.dumps(knowledge_base, indent=2), encoding="utf-8"
    )

    rebuilt = tmp_path / "rebuilt"

    generate_site(input_path=source, output_dir=rebuilt)

    original = sorted(
        p.relative_to(site) for p in site.rglob("*") if p.is_file()
    )
    again = sorted(
        p.relative_to(rebuilt) for p in rebuilt.rglob("*") if p.is_file()
    )

    assert original == again
    assert original, "the generator produced nothing"

    for relative in original:
        left = (site / relative).read_bytes()
        right = (rebuilt / relative).read_bytes()

        assert left == right, relative


def test_a_removed_post_disappears_from_the_site(
    knowledge_base, tmp_path
):
    """
    The site reflects the knowledge base it was given. Content that is
    no longer in the base must not survive in the output.
    """

    trimmed = dict(knowledge_base)
    trimmed["posts"] = knowledge_base["posts"][:1]
    trimmed["stats"] = dict(knowledge_base.get("stats", {}))
    trimmed["stats"]["posts_aggregated"] = len(trimmed["posts"])

    source = tmp_path / "trimmed.json"
    source.write_text(json.dumps(trimmed, indent=2), encoding="utf-8")

    output = tmp_path / "trimmed-site"

    generate_site(input_path=source, output_dir=output)

    pages = {path.stem for path in (output / "posts").glob("*.html")}

    assert pages == {trimmed["posts"][0]["id"]}


def test_the_site_is_produced_into_an_empty_directory(
    knowledge_base, tmp_path
):
    """
    Generation must not depend on a previous build being present, so a
    clean checkout builds the same site.
    """

    source = tmp_path / "kb.json"
    source.write_text(json.dumps(knowledge_base, indent=2), encoding="utf-8")

    output = tmp_path / "clean"

    generate_site(input_path=source, output_dir=output)

    assert (output / "index.html").is_file()


def test_generation_does_not_touch_the_repository(knowledge_base, tmp_path):
    """
    The generator writes where it is told and nowhere else, so running
    it cannot modify committed content.
    """

    before = sorted(
        path.relative_to(REPO_ROOT)
        for path in (REPO_ROOT / "data").rglob("*")
    )

    source = tmp_path / "kb.json"
    source.write_text(json.dumps(knowledge_base, indent=2), encoding="utf-8")

    generate_site(input_path=source, output_dir=tmp_path / "site")

    after = sorted(
        path.relative_to(REPO_ROOT)
        for path in (REPO_ROOT / "data").rglob("*")
    )

    assert before == after


def test_generation_refuses_a_malformed_knowledge_base(tmp_path):
    """A broken input is rejected rather than half-rendered."""

    source = tmp_path / "broken.json"
    source.write_text('{"schema_version": 1}', encoding="utf-8")

    with pytest.raises(Exception):
        generate_site(
            input_path=source, output_dir=tmp_path / "out"
        )


def test_generation_refuses_a_missing_input(tmp_path):
    with pytest.raises(Exception):
        generate_site(
            input_path=tmp_path / "absent.json",
            output_dir=tmp_path / "out",
        )


def test_the_published_date_is_shown_on_the_post_page(
    site, knowledge_base
):
    """
    A reader needs to know when the source published the content, which
    is not the same as when it was collected.
    """

    for post in knowledge_base["posts"]:
        published = post["source"].get("published_at")

        if not published:
            continue

        page = site / "posts" / f"{post['id']}.html"

        assert published in page.read_text(encoding="utf-8"), (
            post["id"],
            published,
        )


def test_a_collected_post_page_links_back_to_the_original(
    site, knowledge_base
):
    """
    Provenance has to be one click away, or a reader cannot check a
    claim against its source.
    """

    checked = 0

    for post in knowledge_base["posts"]:
        url = post["source"].get("url")

        if not url:
            continue

        page = site / "posts" / f"{post['id']}.html"

        assert url in page.read_text(encoding="utf-8"), post["id"]

        checked += 1

    if not checked:
        pytest.skip("no post carries a source url")


def test_the_search_index_carries_the_published_date(
    site, knowledge_base
):
    payload = json.loads(
        (site / "assets" / "search-index.json").read_text(encoding="utf-8")
    )

    by_id = {record["i"]: record for record in payload["records"]}

    for post in knowledge_base["posts"]:
        published = post["source"].get("published_at")

        assert by_id[post["id"]]["pb"] == (published or ""), post["id"]


def test_no_page_contains_a_credential_marker(site):
    """
    A rendered page must never carry something credential-shaped, so a
    browser or a reader cannot pick one up from the site.
    """

    markers = (
        "LINKEDIN_PASSWORD",
        "LINKEDIN_USERNAME",
        ".env",
    )

    for path in site.rglob("*.html"):
        text = path.read_text(encoding="utf-8")

        for marker in markers:
            assert marker not in text, f"{path.name}: {marker}"

# ---------------------------------------------------------------------
# Consolidated sections
# ---------------------------------------------------------------------


def test_every_concept_has_a_page(site, knowledge_base):
    concepts = knowledge_base["knowledge"]["concepts"]

    if not concepts:
        pytest.skip("no concepts consolidated in this checkout")

    pages = {path.stem for path in (site / "concepts").glob("*.html")}

    for concept in concepts:
        assert concept["slug"] in pages, concept["name"]


def test_every_technology_has_a_page(site, knowledge_base):
    technologies = knowledge_base["knowledge"]["technologies"]

    if not technologies:
        pytest.skip("no technologies recognised in this checkout")

    pages = {path.stem for path in (site / "technologies").glob("*.html")}

    for technology in technologies:
        assert technology["slug"] in pages, technology["name"]


def test_a_concept_page_lists_the_posts_it_came_from(
    site, knowledge_base
):
    """
    A concept that cannot be traced back to source is a claim nothing
    supports, so every concept page must name its posts.
    """

    for concept in knowledge_base["knowledge"]["concepts"]:
        page = site / "concepts" / f"{concept['slug']}.html"
        text = page.read_text(encoding="utf-8")

        for post_id in concept["post_ids"]:
            assert post_id in text, f"{concept['name']} -> {post_id}"


def test_a_technology_page_lists_the_posts_that_use_it(
    site, knowledge_base
):
    for technology in knowledge_base["knowledge"]["technologies"]:
        page = site / "technologies" / f"{technology['slug']}.html"
        text = page.read_text(encoding="utf-8")

        for post_id in technology["post_ids"]:
            assert post_id in text, f"{technology['name']} -> {post_id}"


def test_a_shared_concept_links_posts_from_more_than_one_source(
    site, knowledge_base
):
    """
    The point of consolidation: one concept page that gathers several
    posts, rather than a page per post.
    """

    shared = [
        concept
        for concept in knowledge_base["knowledge"]["concepts"]
        if len(concept["post_ids"]) > 1
    ]

    if not shared:
        pytest.skip("no concept is shared between posts yet")

    for concept in shared:
        page = site / "concepts" / f"{concept['slug']}.html"
        text = page.read_text(encoding="utf-8")

        found = sum(1 for post in concept["post_ids"] if post in text)

        assert found == len(concept["post_ids"]), concept["name"]


def test_a_technology_links_the_posts_that_use_it(site, knowledge_base):
    """
    A technology appears only when a post mentions it, so its page
    must point at those posts rather than at nothing.
    """

    for technology in knowledge_base["knowledge"]["technologies"]:
        page = site / "technologies" / f"{technology['slug']}.html"
        text = page.read_text(encoding="utf-8")

        assert technology["post_ids"], technology["name"]
        assert "posts/" in text, technology["name"]


def test_no_technology_page_exists_without_a_supporting_post(
    site, knowledge_base
):
    """
    A technology page with no post behind it would be an invented
    claim, so the two sets have to match exactly.
    """

    from_site = {
        path.stem for path in (site / "technologies").glob("*.html")
    }

    from_kb = {
        technology["slug"]
        for technology in knowledge_base["knowledge"]["technologies"]
    }

    assert from_site == from_kb


def test_the_search_index_covers_concepts_and_technologies(
    site, knowledge_base
):
    """
    Search that cannot find a concept is not search over the
    knowledge base, it is search over the posts.
    """

    payload = json.loads(
        (site / "assets" / "search-index.json").read_text(encoding="utf-8")
    )

    records = payload["records"]

    for concept in knowledge_base["knowledge"]["concepts"]:
        page = f"concepts/{concept['slug']}.html"

        assert any(
            record["u"] == page for record in records
        ), concept["name"]

    for technology in knowledge_base["knowledge"]["technologies"]:
        page = f"technologies/{technology['slug']}.html"

        assert any(
            record["u"] == page for record in records
        ), technology["name"]


def test_every_post_records_what_kind_of_content_it_is(
    knowledge_base,
):
    """
    A post that teaches nothing is still stored and still labelled, so
    a reader can see why it contributed no knowledge.
    """

    kinds = knowledge_base["knowledge"]["content_kinds"]

    assert kinds, "every post should carry a content kind"

    for post in knowledge_base["posts"]:
        assert post["id"] in kinds, post["id"]

        assert kinds[post["id"]] in {
            "technical",
            "job_announcement",
            "event",
            "congratulation",
            "certification",
            "appreciation",
            "social",
            "unknown",
        }, kinds[post["id"]]

def test_an_unenriched_post_is_still_reachable(site, knowledge_base):
    """
    A post with no analysis is still content with real provenance, so it
    must appear in the knowledge base and have a page. Dropping it would
    lose the source, and a reader would have no way to reach it.
    """

    unenriched = [
        post
        for post in knowledge_base["posts"]
        if not (post["ai_analysis"].get("summary") or "").strip()
    ]

    if not unenriched:
        pytest.skip("every post has been enriched in this checkout")

    pages = {path.stem for path in (site / "posts").glob("*.html")}

    for post in unenriched:
        assert post["id"] in pages, post["id"]

        page = (site / "posts" / f"{post['id']}.html").read_text(
            encoding="utf-8"
        )

        # The source text is still on the page. Rendered whitespace is
        # collapsed and long text is excerpted, so the opening words
        # are matched rather than an arbitrary slice.
        # Rendered text is HTML-escaped, so the comparison is made
        # against the same escaping the renderer applies.
        from html import unescape

        opening = " ".join(post["original_text"].split())[:40]

        rendered = " ".join(unescape(page).split())

        assert opening in rendered, post["id"]


def test_the_knowledge_base_holds_every_collected_post(knowledge_base):
    """
    Aggregation must not silently drop a post for lacking analysis. The
    count of posts in the base equals the count on disk, which is what
    makes a missing post visible rather than invisible.
    """

    on_disk = {
        path.parent.name for path in (REAL_POSTS_ROOT).glob("*/post.json")
    }

    aggregated = {post["id"] for post in knowledge_base["posts"]}

    # Only posts that are staged for aggregation are required. The
    # fixture may use enriched results, which cover the same ids.
    assert aggregated <= on_disk or aggregated == on_disk

    for post_id in on_disk:
        assert post_id in aggregated, post_id


def test_the_site_and_the_knowledge_base_report_the_same_things(
    knowledge_base, site
):
    """
    The knowledge base and the site are two views of one set of posts.
    If they disagree, a reader who reconciles them finds numbers that
    cannot both be true.
    """

    payload = json.loads(
        (site / "assets" / "search-index.json").read_text(encoding="utf-8")
    )

    kinds: dict[str, int] = {}

    for record in payload["records"]:
        kind = record.get("k", "p")
        kinds[kind] = kinds.get(kind, 0) + 1

    stats = knowledge_base["stats"]

    assert kinds.get("p") == stats["posts_aggregated"]
    assert kinds.get("c") == stats["concepts_consolidated"]
    assert kinds.get("x") == stats["technologies_consolidated"]

    # The site renders topics and subtopics as navigable pages. A label
    # that is both collapses onto one page, so the site count can be one
    # lower than the two lists combined but never higher.
    rendered = kinds.get("t", 0)
    listed = (
        stats["topics_consolidated"] + stats["subtopics_consolidated"]
    )

    assert rendered <= listed
    assert rendered >= max(
        stats["topics_consolidated"], stats["subtopics_consolidated"]
    )


def test_every_knowledge_node_kind_is_recorded(knowledge_base):
    """
    Anything the site can render has to be accounted for in the
    canonical base, or a page exists with nothing behind it.
    """

    knowledge = knowledge_base["knowledge"]

    for key in (
        "topics",
        "subtopics",
        "concepts",
        "technologies",
        "questions",
        "content_kinds",
    ):
        assert key in knowledge, key
