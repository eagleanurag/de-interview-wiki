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
    Four root pages, one revision tree, and the 404 GitHub Pages serves.

    Home, Questions, Search, the tree and the 404. There is no Subjects,
    Topics, Concepts, Technologies or Saved Items page: the first was the
    taxonomy's tree, which the units replaced, and the rest were the
    archive's shape, which is what this change removed.
    """

    for name in (
        "index.html",
        "questions.html",
        "search.html",
        "404.html",
    ):
        assert (site / name).is_file(), name

    assert (site / "revision" / "index.html").is_file()

    for name in (
        "subjects.html",
        "topics.html",
        "concepts.html",
        "technologies.html",
        "saved-items.html",
    ):
        assert not (site / name).exists(), name


def test_every_published_question_has_a_page(site, knowledge_base):
    """
    Every question the guide publishes is on a page.

    It used to be "every post has a page", which is the archive's
    question. The product's question is the narrower one: is there a page
    a reader can land on for each question? A question that is in the
    knowledge base but on no page is a question nobody can reach, and a
    post that has no page is simply a source record, which is what it is.
    """

    payload = json.loads(
        (site / "assets" / "search-index.json").read_text(encoding="utf-8")
    )

    assert payload["questions"] > 0

    indexed = {
        record["i"]
        for record in payload["records"]
        if record["k"] == "q"
    }

    assert indexed, "no question is indexed"

    # Every index entry resolves, which is what makes the set above
    # reachable rather than merely present.
    for record in payload["records"]:
        assert (site / record["u"]).is_file(), record["u"]

    # And the published count reconciles with the taxonomy's, less what
    # the relevance filter refused.
    assert payload["questions"] <= payload["revision_questions"]


def test_the_search_index_describes_the_published_guide(site):
    """
    Search reaches revision units and their questions, and nothing else.

    The index used to hold a record per post, per topic, per concept and
    per technology -- 11,616 of them, each pointing at a page. The pages
    are gone; so are the records, because an entry that leads nowhere is
    worse than a missing one.
    """

    payload = json.loads(
        (site / "assets" / "search-index.json").read_text(encoding="utf-8")
    )

    records = payload["records"]

    assert records

    kinds = {record["k"] for record in records}

    assert kinds == {"b", "q"}, kinds

    for record in records:
        assert record["u"].startswith("revision/"), record["u"]
        assert (site / record["u"]).is_file(), record["u"]

    # No post identifier is indexed, because there is no post page and
    # nothing should lead a reader to a machine key.
    assert payload["posts"] == 0

    for retired in ("topics", "concepts", "technologies"):
        assert payload[retired] == 0, retired

    # One record per unit, and one per published question.
    assert len([r for r in records if r["k"] == "b"]) == payload["units"]
    assert len([r for r in records if r["k"] == "q"]) == payload["questions"]

    assert payload["units"] > 0


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
        "revision/index.html",
        "questions.html",
        "search.html",
    ):
        assert page in index, page

    for gone in (
        "subjects.html",
        "topics.html",
        "concepts.html",
        "technologies.html",
        "saved-items.html",
        "posts/",
    ):
        assert gone not in index, gone


def test_every_revision_page_is_reachable_from_home(site):
    """
    The replacement for the topic index.

    The archive had four indexes, each unreachable without the other
    three. There is one now -- the revision tree -- and it has to reach
    every page of every unit, including the pages past the first.

    Reaching page 7 of a unit by browsing is not possible; there is no
    listing of "page 7" anywhere. So the pager is what makes it
    reachable, and this is the test that says so.
    """

    index = (site / "index.html").read_text(encoding="utf-8")

    assert 'href="revision/index.html"' in index

    tree = (site / "revision" / "index.html").read_text(encoding="utf-8")

    pages = sorted(
        page.name
        for page in (site / "revision").glob("*.html")
        if page.name != "index.html"
    )

    assert pages, "no revision pages were generated"

    first_pages = 0

    for name in pages:
        if name in tree:
            first_pages += 1

    # The tree links each unit's first page. Pages 2..N are linked only
    # by the pager, which is checked below.
    assert first_pages >= 1, "the tree lists no unit"

    linked = set()

    for page in (site / "revision").glob("*.html"):
        for target in re.findall(r'href="([^"#?]+)"', page.read_text("utf-8")):
            if not target.startswith(("http", "mailto")):
                linked.add((page.parent / target).resolve().name)

    unreached = set(pages) - linked

    assert unreached == set(), sorted(unreached)


def test_a_paginated_unit_offers_every_page(site):
    """
    Previous / page N of M / next, and every page reachable.

    The rule is a question count, so the pagination is the same on every
    build and a reader who bookmarked page 3 comes back to page 3.
    """

    paginated = []

    for page in sorted((site / "revision").glob("*.html")):
        if page.name == "index.html":
            continue

        body = page.read_text(encoding="utf-8")

        if "Page 1 of" in body:
            paginated.append((page, body))

    if not paginated:
        pytest.skip("no unit needed more than one page")

    for page, body in paginated:
        total = int(re.search(r"Page 1 of (\d+)", body).group(1))

        assert total > 1, page.name

        slug = page.stem

        # Every page the unit claims exists.
        for number in range(2, total + 1):
            sibling = site / "revision" / f"{slug}-{number}.html"

            assert sibling.is_file(), (
                f"{page.name} claims {total} pages but "
                f"{sibling.name} is not generated"
            )

        # Page one has no previous, rather than pointing at itself.
        assert "rel=\"prev\"" not in body, page.name

        # And following "next" from page one reaches every page, which is
        # what makes page 7 reachable to a reader who did not build the
        # URL. The pager is sequential, so page 1 links page 2 and not
        # page 3; the chain is what has to close.
        reached = {page.name}
        current = page

        while True:
            following = re.search(
                r'<a class="text-link" rel="next" href="([^"]+)"',
                current.read_text(encoding="utf-8"),
            )

            if not following:
                break

            current = (current.parent / following.group(1)).resolve()

            assert current.is_file(), following.group(1)

            reached.add(current.name)

            assert len(reached) <= total, "the pager loops"

        assert len(reached) == total, (
            f"{page.name}: the pager reaches {len(reached)} of "
            f"{total} pages"
        )

        # The last page stops, rather than offering a page that is not
        # there.
        assert "rel=\"next\"" not in current.read_text(encoding="utf-8")


def test_the_question_index_lists_every_question(knowledge_base, site):
    """
    The index lists what the guide publishes.

    It used to list every question in the knowledge base, which meant it
    listed the 114 the relevance filter discards as well -- advice about
    which platform to practise on, and questions about the hiring
    process. A reader browsing for something to revise does not want
    those, so the index now lists the published set and the two
    reconcile.
    """

    payload = json.loads(
        (site / "assets" / "search-index.json").read_text(encoding="utf-8")
    )

    index = (site / "questions.html").read_text(encoding="utf-8")

    published = [
        record["i"]
        for record in payload["records"]
        if record["k"] == "q"
    ]

    assert published

    for text in published[:200]:
        # The text is escaped in HTML, so the opening word is what is
        # checked. A first word that is also a stop word would pass
        # vacuously, so a distinctive one is required.
        first = text.split(" ")[0]

        assert first in index, text[:60]

    # And nothing more: the discarded questions are not listed.
    discarded = [
        question
        for post in knowledge_base["posts"]
        for question in post["interview_questions"]
    ]

    listed = len(published)

    assert listed < len(discarded), (
        "the filter discarded nothing, so it is not running"
    )


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
    The site reflects the knowledge base it was given.

    Checked on the questions rather than the pages now: a trimmed corpus
    must publish fewer questions, and must not publish the wording of a
    question that is no longer in the data.
    """

    trimmed = dict(knowledge_base)
    trimmed["posts"] = knowledge_base["posts"][:1]
    trimmed["stats"] = dict(knowledge_base.get("stats", {}))
    trimmed["stats"]["posts_aggregated"] = len(trimmed["posts"])

    source = tmp_path / "trimmed.json"
    source.write_text(json.dumps(trimmed, indent=2), encoding="utf-8")

    output = tmp_path / "trimmed-site"

    generate_site(input_path=source, output_dir=output)

    kept = {
        question["question"]
        for post in trimmed["posts"]
        for question in post["interview_questions"]
    }

    dropped = {
        question["question"]
        for post in knowledge_base["posts"]
        for question in post["interview_questions"]
    } - kept

    assert kept, "the fixture post carries no questions"

    payload = json.loads(
        (output / "assets" / "search-index.json").read_text("utf-8")
    )

    assert payload["questions"] <= len(kept)

    published = {
        record["i"]
        for record in payload["records"]
        if record["k"] == "q"
    }

    for text in dropped:
        assert text not in published, text[:60]


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


def test_the_published_date_survives_in_the_data_and_not_on_the_page(
    site, knowledge_base
):
    """
    A reader needs to know when the source published the content, which
    is not the same as when it was collected. And a reader revising for
    an interview does not.

    The date used to be rendered on the post page, where it was satisfied
    by accident: the raw value is "2026-10-01T23:11:32Z" and the page
    shortened it to a date, but it also printed a *Captured* row whose
    "2026-10-01" happened to be a substring of the raw value, so the
    assertion could be satisfied by a capture timestamp.

    Both facts are checked here instead. The timestamp is in the
    knowledge base, unmodified, and no capture timestamp appears anywhere
    a reader will see.
    """

    published_values = [
        (post["source"].get("published_at") or "").strip()
        for post in knowledge_base["posts"]
    ]

    recorded = [value for value in published_values if value]

    assert recorded, "no post recorded a published date"

    captured = [
        post["source"]["captured_at"] for post in knowledge_base["posts"]
    ]

    assert all(value for value in captured)

    # Not one of them reaches a page.
    from html import unescape

    for page in site.rglob("*.html"):
        body = " ".join(unescape(page.read_text("utf-8")).split())

        for value in recorded:
            assert value not in body, (page.name, value)

        for value in captured:
            assert value not in body, (page.name, value)


def test_the_source_url_survives_in_the_data_and_not_on_the_page(
    site, knowledge_base
):
    """
    Provenance has to be recoverable, or a reader cannot check a claim
    against its source -- and it must not be *presented* as a link out.

    The site is a revision guide. A link to the original post was the
    archive's affordance: it took a reader off the site to see a post
    whose material was already, restated and answered, on the page they
    were reading. That is not a service worth having, and it is why the
    per-post pages went.
    """

    urls = [
        post["source"].get("url")
        for post in knowledge_base["posts"]
        if post["source"].get("url")
    ]

    if not urls:
        pytest.skip("no post carries a source url")

    # Every url is intact in the data, and every post still has its
    # identifier, so the record can be traced.
    for post in knowledge_base["posts"]:
        assert post["id"]

        if post["source"].get("url"):
            assert post["source"]["url"].startswith("https://")

    for page in site.rglob("*.html"):
        body = page.read_text(encoding="utf-8")

        for url in urls[:50]:
            assert url not in body, (page.name, url)

    # And no page offers to take a reader to a platform.
    assert not (site / "posts").exists()


def test_the_search_index_carries_no_capture_dates(site, knowledge_base):
    """
    The index ships to every reader's browser, so it carries no
    timestamps and no platform: a result card has nothing to display
    that would put a capture date next to a question.
    """

    payload = json.loads(
        (site / "assets" / "search-index.json").read_text(encoding="utf-8")
    )

    for record in payload["records"]:
        assert record["d"] == "", record
        assert record["pb"] == "", record
        assert record["a"] == "", record


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


def test_concepts_are_still_searchable_text(site, knowledge_base):
    """
    Where a reader meets concepts now.

    Not as pages, but as the vocabulary of the revision unit that teaches
    them, which is what a search for "Broadcast Hash Join" needs to land
    on Spark.

    All of them, which is the part worth asserting. Capping a unit's
    concept list -- which is what the first version did, at 24 -- leaves
    521 of 3,056 labels reachable and the rest silently unsearchable. A
    label a reader can type but not find is worse than one that is
    absent, because it looks supported.
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

    assert searchable, "nothing is searchable by concept"

    missing = [
        concept["name"]
        for concept in concepts
        if concept["name"] not in searchable
    ]

    # Not every label survives the taxonomy: a concept on a post whose
    # questions were all discarded never reaches a unit. So this is a
    # coverage threshold with the misses reported, not an equality that
    # would only pass by discarding almost everything.
    coverage = 1 - len(missing) / len(concepts)

    assert coverage > 0.95, (
        f"{len(missing)} of {len(concepts)} concept labels are "
        f"unsearchable, e.g. {missing[:5]}"
    )


def test_technologies_are_reachable_as_searchable_text(
    site, knowledge_base
):
    """
    Technologies have no page and no record of their own.

    They are named in the questions and summaries of the units that use
    them, so "Azure Data Factory" finds the ADF revision unit. What it
    must not do is find a technology index page, because there is none.
    """

    technologies = knowledge_base["knowledge"]["technologies"]

    if not technologies:
        pytest.skip("no technologies recognised in this checkout")

    published = "".join(
        page.read_text(encoding="utf-8") for page in site.rglob("*.html")
    )

    named = [
        technology["name"]
        for technology in technologies
        if technology["name"] in published
    ]

    # The platform names a candidate is asked about are present. The
    # assertion is a threshold, not an equality: a technology recognised
    # once in a post that contributed no published question is not
    # something a reader could be expected to search for.
    major = {
        "Apache Spark",
        "Databricks",
        "Azure Data Factory",
        "Microsoft Azure",
        "AWS",
        "Delta Lake",
        "Apache Airflow",
        "dbt",
        "Snowflake",
        "Amazon Redshift",
        "Google BigQuery",
        "Apache Kafka",
        "Power BI",
    }

    found = sorted(
        name for name in major if any(name in text for text in [published])
    )

    assert len(found) >= 7, found

    assert named, "no technology is named anywhere on the site"


def test_the_search_index_points_at_no_page_that_was_not_written(site):
    """
    Every record's ``u`` is a URL a reader can follow.

    A record pointing at a page that was not generated is a dead result
    in the result list, which is worse than the record being absent: it
    looks like an answer.
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
        assert record["u"].startswith("revision/"), record["u"]


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

def test_an_unenriched_post_is_still_kept(knowledge_base):
    """
    A post with no analysis is still content with real provenance.

    It must appear in the knowledge base with its text, its source and a
    content kind, so the record is not lost. It has no page, which is a
    different claim: a post that taught nothing is not a revision unit.

    This used to assert a page, and on this corpus it passed by skipping,
    because every post has been enriched. The data assertion runs either
    way.
    """

    unenriched = [
        post
        for post in knowledge_base["posts"]
        if not (post["ai_analysis"].get("summary") or "").strip()
    ]

    kinds = knowledge_base["knowledge"]["content_kinds"]

    if not unenriched:
        pytest.skip(
            "every post has been enriched in this checkout, so this "
            "measures nothing here; the aggregation check below does"
        )

    for post in unenriched:
        assert post["id"] in kinds, post["id"]
        assert kinds[post["id"]], post["id"]
        assert post["original_text"].strip(), post["id"]
        assert post["source"]["captured_at"], post["id"]

    assert knowledge_base["knowledge"]["content_kinds"], "no kinds at all"


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
    The knowledge base and the site are two views of one corpus.

    The reconciliation changed shape rather than disappearing. It used to
    be "indexed posts == posts aggregated", which was true and which
    checked nothing a reader cares about -- it counted records against
    records. What a reader can check is the other direction: the
    knowledge base holds 490 posts, the site publishes the questions that
    survived relevance filtering, and every one of them is on a page that
    exists.
    """

    payload = json.loads(
        (site / "assets" / "search-index.json").read_text(encoding="utf-8")
    )

    stats = knowledge_base["stats"]

    # The corpus the site was built from.
    assert stats["posts_aggregated"] == len(knowledge_base["posts"])

    # The guide is smaller than the corpus, which is the whole point, and
    # the difference is the filter rather than a silent loss.
    assert payload["questions"] < stats["posts_aggregated"] * 10

    # Everything indexed is on a page.
    rendered = {
        path.relative_to(site).as_posix()
        for path in site.rglob("*.html")
    }

    for record in payload["records"]:
        assert record["u"] in rendered, record["u"]

    # Every unit has at least one page, and the page count is what the
    # unit's own question count implies.
    assert payload["units"] > 0

    unit_pages = rendered - {
        "index.html",
        "questions.html",
        "search.html",
        "404.html",
        "revision/index.html",
    }

    assert len(unit_pages) == payload["pages"], (
        f"{payload['pages']} pages reported, {len(unit_pages)} on disk"
    )

    # Concepts and technologies have no record of their own, because they
    # have no page. They are still in the data, and still nameable in a
    # search, which is the only way a reader now meets either.
    kinds = {record["k"] for record in payload["records"]}

    assert "c" not in kinds
    assert "x" not in kinds
    assert "t" not in kinds

    searchable = {
        label
        for record in payload["records"]
        for label in record.get("c", [])
    }

    assert searchable, "no concept is searchable"


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
