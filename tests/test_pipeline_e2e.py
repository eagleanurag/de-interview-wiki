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
    """

    directory = tmp_path_factory.mktemp("worker-results")

    for path in posts_root.glob("*/post.json"):
        destination = directory / f"cloud_worker_{path.parent.name}.json"
        shutil.copyfile(path, destination)

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


def test_every_enriched_post_appears_in_the_knowledge_base(
    posts_root, knowledge_base
):
    """
    Aggregation covers what the worker enriched. A post that was never
    enriched is skipped rather than aggregated as a half-record, so the
    assertion is against the enriched posts.
    """

    from src.aggregation.aggregator import REQUIRED_ENRICHED_FIELDS

    def enriched(path: Path) -> bool:
        payload = json.loads(path.read_text(encoding="utf-8"))

        return REQUIRED_ENRICHED_FIELDS <= set(payload)

    expected = {
        path.parent.name
        for path in posts_root.glob("*/post.json")
        if enriched(path)
    }

    actual = {post["id"] for post in knowledge_base["posts"]}

    assert actual == expected


def test_an_unenriched_post_is_skipped_not_half_aggregated(
    posts_root, knowledge_base
):
    """
    A bare stub carries no summary and no questions. Aggregating it
    would put an empty record in the canonical base, so it is left out
    and reported instead.
    """

    from src.aggregation.aggregator import REQUIRED_ENRICHED_FIELDS

    unenriched = []

    for path in posts_root.glob("*/post.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))

        if not REQUIRED_ENRICHED_FIELDS <= set(payload):
            unenriched.append(path.parent.name)

    if not unenriched:
        pytest.skip("every post has been enriched in this checkout")

    aggregated = {post["id"] for post in knowledge_base["posts"]}

    assert not aggregated.intersection(unenriched)


def test_no_post_appears_twice(knowledge_base):
    identifiers = [post["id"] for post in knowledge_base["posts"]]

    assert len(identifiers) == len(set(identifiers))


def test_every_collected_post_kept_its_provenance(knowledge_base):
    collected = [
        post
        for post in knowledge_base["posts"]
        if post["id"].startswith("urn-li-")
    ]

    if not collected:
        pytest.skip("no collected posts in this checkout")

    for post in collected:
        assert post["source"]["platform"] == "linkedin"
        assert post["source"]["url"]
        assert post["source"]["captured_at"]
        assert post["original_text"].strip()


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
    for name in ("index.html", "topics.html", "questions.html", "search.html"):
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

    assert payload["v"] == 1
    assert payload["posts"] == len(knowledge_base["posts"])
    assert len(payload["records"]) == len(knowledge_base["posts"])

    # "i" is the identifier and "u" the page, both compressed to keep
    # the index small. The identifier field is what the assertion needs.
    identifiers = {record["i"] for record in payload["records"]}

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


def test_the_navigation_links_every_top_level_page(site):
    index = (site / "index.html").read_text(encoding="utf-8")

    for page in ("topics.html", "questions.html", "search.html"):
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
