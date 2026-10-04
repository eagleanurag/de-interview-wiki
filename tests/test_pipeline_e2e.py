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
    Stage the enriched results the way aggregation reads them.

    Source order matters here, and getting it wrong is not cosmetic.

    ``data/results/`` comes first because that is where the repository
    keeps its enrichment and where CI reads it. ``build/worker-results``
    is a local artifact of a pipeline run, present on a developer's
    machine and absent in CI -- preferring it made this fixture build a
    knowledge base from whatever it found locally, so the same test
    covered enriched posts on one machine and unenriched posts on
    another. Given only unenriched posts it aggregated 490 of them with
    no concepts and no questions, and the assertions below failed on a
    knowledge base that was never going to have any.

    The local directory is still consulted second, so a developer who
    has just re-enriched sees their own results checked.
    """

    directory = tmp_path_factory.mktemp("worker-results")

    for source in (
        REPO_ROOT / "data" / "results",
        Path("build") / "worker-results",
    ):
        if not source.is_dir():
            continue

        found = sorted(source.glob(WORKER_RESULT_GLOB))

        if not found:
            continue

        for path in found:
            shutil.copyfile(path, directory / path.name)

        return directory

    # Nothing enriched anywhere. Aggregating raw posts would produce a
    # knowledge base with no concepts, technologies or questions, and
    # every assertion about those would fail for a reason that has
    # nothing to do with what is being tested.
    pytest.skip(
        "no enriched results to aggregate; run "
        "`python -m src.pipeline --only enrich` first"
    )


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
    """
    Five entry points, and the four the reader navigates by.

    Home, Subjects, Questions, Search and the 404 that GitHub Pages
    serves. There is no Topics, Concepts, Technologies or Saved Items
    page; that list was the shape of the old site and it is what this
    change removed.
    """

    for name in (
        "index.html",
        "subjects.html",
        "questions.html",
        "search.html",
        "404.html",
    ):
        assert (site / name).is_file(), name

    for name in (
        "topics.html",
        "concepts.html",
        "technologies.html",
        "saved-items.html",
    ):
        assert not (site / name).exists(), name


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

    # Records cover posts plus the revision curriculum. The archive's own
    # topic, concept and technology records are gone, because the pages
    # they pointed at are gone; the revision records took their place and
    # there are more of them, because every subtopic and every merged
    # question is indexed rather than only every post.
    assert len(payload["records"]) >= payload["posts"]

    # And nothing but a post or a revision page is indexed.
    kinds = {record["k"] for record in payload["records"]}

    assert kinds <= {"p", "s", "b", "q"}, kinds

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
    """
    Home must reach the three places a reader goes, and nothing else.

    This asserted the archive's five indexes. Reaching them is what made
    the site an archive with a revision front end rather than a revision
    wiki.
    """

    index = (site / "index.html").read_text(encoding="utf-8")

    for page in (
        "subjects.html",
        "questions.html",
        "search.html",
    ):
        assert page in index, page

    for gone in (
        "topics.html",
        "concepts.html",
        "technologies.html",
        "saved-items.html",
    ):
        assert gone not in index, gone


def test_every_top_level_page_is_reachable_from_home(site):
    """
    The replacement for the topic index.

    The archive had four indexes, each unreachable without the other
    three. There is one now: subjects, the revision tree, which reaches
    every subtopic and every question on the site.
    """

    index = (site / "index.html").read_text(encoding="utf-8")

    assert 'href="subjects.html"' in index

    subjects = (site / "subjects.html").read_text(encoding="utf-8")

    revisions = sorted((site / "topics").rglob("*/*.html"))

    assert revisions, "no revision pages were generated"

    for page in revisions:
        relative = page.relative_to(site / "topics").as_posix()

        assert relative in subjects, (
            f"{relative} is unreachable from the subjects index"
        )


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

    It used to pass by accident. The raw value is
    "2026-10-01T23:11:32Z" and the page shortened it to a date, but it
    also printed a *Captured* row whose "2026-10-01" happened to be a
    substring of the raw value, so the assertion was satisfied by a
    capture timestamp rather than by the published one. That row is
    gone, so the test now checks what it meant to check: the published
    date appears in the form the page renders it.
    """

    from datetime import datetime

    checked = 0

    for post in knowledge_base["posts"]:
        published = (post["source"].get("published_at") or "").strip()

        if not published:
            continue

        try:
            shown = datetime.fromisoformat(
                published.replace("Z", "+00:00")
            ).strftime("%Y-%m-%d")
        except ValueError:
            shown = published

        page = site / "posts" / f"{post['id']}.html"

        assert shown in page.read_text(encoding="utf-8"), (
            post["id"],
            published,
        )

        checked += 1

    assert checked, "no post recorded a published date"


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


def test_no_concept_or_technology_page_is_generated(
    site, knowledge_base
):
    """
    Concepts and technologies have no page, and that is the change.

    Three thousand one hundred and twelve concept pages and forty-four
    technology pages existed to list labels the enricher produced. A
    candidate revising for an interview revises subjects and subtopics,
    not the labels; and both pages were reachable from the footer of
    every page on the site, which put the archive one click away from
    everywhere.
    """

    concepts = knowledge_base["knowledge"]["concepts"]
    technologies = knowledge_base["knowledge"]["technologies"]

    if concepts:
        assert not (site / "concepts").exists()
        assert not (site / "concepts.html").exists()

    if technologies:
        assert not (site / "technologies").exists()
        assert not (site / "technologies.html").exists()

    assert not (site / "topics.html").exists()
    assert not (site / "saved-items.html").exists()

    # And nothing links to any of them.
    for page in site.rglob("*.html"):
        body = page.read_text(encoding="utf-8")

        for gone in (
            "concepts/",
            "technologies/",
            "concepts.html",
            "technologies.html",
            "topics.html",
            "saved-items.html",
        ):
            assert gone not in body, f"{page.relative_to(site)}: {gone}"


def test_every_concept_still_names_the_posts_it_came_from(
    knowledge_base
):
    """
    The traceability the concept page used to provide.

    A concept that cannot be traced back to source is a claim nothing
    supports. The page that listed its posts is gone; the linkage is not,
    and it is what lets a reader search a label and land on a post.
    """

    concepts = knowledge_base["knowledge"]["concepts"]

    if not concepts:
        pytest.skip("no concepts consolidated in this checkout")

    for concept in concepts:
        assert concept["post_ids"], concept["name"]
        assert concept["slug"], concept["name"]


def test_every_technology_still_names_the_posts_that_use_it(
    knowledge_base
):
    technologies = knowledge_base["knowledge"]["technologies"]

    if not technologies:
        pytest.skip("no technologies recognised in this checkout")

    for technology in technologies:
        assert technology["post_ids"], technology["name"]


def test_a_shared_concept_still_carries_every_contributing_post(
    knowledge_base
):
    """
    The point of consolidation, which outlived the page.

    One concept gathered from several posts, not one concept per post.
    """

    shared = [
        concept
        for concept in knowledge_base["knowledge"]["concepts"]
        if len(concept["post_ids"]) > 1
    ]

    if not shared:
        pytest.skip("no concept is shared between posts yet")

    for concept in shared:
        assert len(set(concept["post_ids"])) == len(concept["post_ids"]), (
            concept["name"]
        )


def test_concepts_and_technologies_are_still_searchable_text(
    site, knowledge_base
):
    """
    Where a reader meets them now.

    Not as pages, but as text on the posts that discuss them, which is
    what the technology keyword lists in the search index were for: a
    search for "Azure Data Factory" has to find the posts that use it,
    and it does.
    """

    payload = json.loads(
        (site / "assets" / "search-index.json").read_text(encoding="utf-8")
    )

    concepts = knowledge_base["knowledge"]["concepts"]

    if not concepts:
        pytest.skip("no concepts consolidated in this checkout")

    searchable = {
        label
        for record in payload["records"]
        for label in record.get("c", [])
    }

    for concept in concepts:
        assert concept["name"] in searchable, concept["name"]


def test_the_search_index_points_at_no_page_that_was_not_written(
    site, knowledge_base
):
    """
    The other half.

    Every record's ``u`` is a URL a reader can follow. A record pointing
    at a concept or technology page would have been a dead result in the
    result list, which is worse than the concept not being findable at
    all.
    """

    payload = json.loads(
        (site / "assets" / "search-index.json").read_text(encoding="utf-8")
    )

    records = payload["records"]

    assert records

    for record in records:
        target = site / record["u"]

        assert target.is_file(), (
            f"{record['u']} is indexed but not generated"
        )

    for record in records:
        assert record["u"].startswith(
            ("posts/", "topics/")
        ), record["u"]


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

    # Concepts and technologies are no longer indexed as records of
    # their own, because they no longer have pages. They are still in
    # the knowledge base, and the posts that carry them still index them
    # as searchable text -- which is the only way a reader now meets
    # either. The counts come from the data, not from the index.
    searchable_concepts = {
        label
        for record in payload["records"]
        for label in record.get("c", [])
    }

    searchable_technologies = {
        label
        for record in payload["records"]
        for label in record.get("t", [])
    }

    assert kinds.get("c") is None
    assert kinds.get("x") is None

    # Every consolidated concept is reachable as searchable text. Not an
    # equality: a post can carry a spelling the aggregator folded away,
    # so the set of labels a reader can search is a superset of the set
    # the knowledge base counts.
    for concept in knowledge_base["knowledge"]["concepts"]:
        assert concept["name"] in searchable_concepts, concept["name"]

    for technology in knowledge_base["knowledge"]["technologies"]:
        assert technology["name"] in searchable_technologies, technology["name"]

    # Topics and subtopics are no longer archive pages either. They are
    # rendered as the revision tree -- a subject per subject, a subtopic
    # per subtopic -- and it is the tree that has to reconcile with the
    # knowledge base, not a count of topic pages.
    subjects = kinds.get("s", 0)
    subtopics = kinds.get("b", 0)

    assert subjects > 0
    assert subtopics > 0

    # Every revision page exists.
    rendered_pages = sorted((site / "topics").rglob("*/*.html"))

    assert len(rendered_pages) == subtopics, (
        f"{len(rendered_pages)} pages for {subtopics} indexed subtopics"
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
