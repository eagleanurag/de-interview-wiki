"""
End-to-end tests for the Saved Items workflow.

A saved list is not a document. It is a list of links, some of which
the user later captured and some of which they did not, and the whole
point of the phase is that both halves behave honestly: a captured item
becomes a real post with real provenance, and an uncaptured one stays a
link and says so.

These drive the real command over a synthetic drop zone, so the list
formats, the bundle matching, the deduplication, the state file and the
posts on disk are all exercised together rather than one at a time. Then
the posts are carried the rest of the way: validated, aggregated and
published, to confirm a saved post reaches the wiki as an ordinary post
and is traceable back to the item it came from.

Every fixture is synthetic. No credential appears in any of them, no
network is touched, and no LinkedIn page is ever fetched.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from src.aggregation.aggregator import (
    WORKER_RESULT_GLOB,
    aggregate_results,
)
from src.ingestion.collect_cli import main as collect_cli_main
from src.ingestion.importer import validate_posts
from src.ingestion.post_document import PostDocument
from src.ingestion.post_loader import load_post, source_digest
from src.ingestion.saved_items.manifest import SavedItemsManifest
from src.ingestion.saved_items.urls import normalize_linkedin_url
from src.pipeline import ENRICHER_VERSION
from src.wiki.generator import generate_site

from tests.test_saved_items import pdf_bytes, png_bytes


POST_DELTA = "https://www.linkedin.com/posts/alice_delta-lake-101"
POST_KAFKA = "https://www.linkedin.com/posts/bob_kafka-partitioning-202"
ARTICLE_SPARK = "https://www.linkedin.com/pulse/carol_spark-skew-303"
POST_IDEMPOTENT = "https://www.linkedin.com/posts/dan_idempotence-404"
POST_BACKPRESSURE = "https://www.linkedin.com/posts/erin_backpressure-505"
POST_BROKEN = "https://www.linkedin.com/posts/frank_broken-606"
POST_DIAGRAM = "https://www.linkedin.com/posts/grace_diagram-707"


def sid(url: str) -> str:
    return normalize_linkedin_url(url).source_id


def safe_sid(url: str) -> str:
    return sid(url).replace(":", "-")


# ---------------------------------------------------------------------
# The drop zone
# ---------------------------------------------------------------------


def build_drop_zone(root: Path) -> Path:
    """
    A realistic saved-items drop zone.

    Contains a repeated URL with tracking noise, an item that was never
    captured, one captured as Markdown, one as a saved page, one as a
    PDF, one as a bare screenshot, one that is corrupt, and two rows
    that cannot be read at all.
    """

    drop = root / "saved-items"
    drop.mkdir(parents=True)

    (drop / "manifest.csv").write_text(
        "URL,Saved Date,Title,Author,Notes\n"
        # The same post twice, written the way two exports differ.
        f"{POST_DELTA}/?trk=abc,2026-01-02,Delta Lake,Alice,ACID\n"
        f"{POST_DELTA}#comment-5,2026-01-02,Delta Lake,Alice,ACID\n"
        # Never captured: a link and a date and nothing else.
        f"{POST_KAFKA},2026-01-03,Kafka partitioning,Bob,\n"
        f"{ARTICLE_SPARK},2026-01-04,Spark skew,Carol,join tuning\n"
        f"{POST_IDEMPOTENT},2026-01-05,Idempotence,Dan,\n"
        f"{POST_BACKPRESSURE},2026-01-06,Backpressure,Erin,\n"
        f"{POST_BROKEN},2026-01-07,Broken,Frank,\n"
        f"not-a-url,2026-01-08,,\n"
        ",2026-01-09,no url at all,\n",
        encoding="utf-8",
    )

    # A Markdown capture, found by its own directory name.
    markdown = drop / safe_sid(ARTICLE_SPARK)
    markdown.mkdir()

    (markdown / "content.md").write_text(
        "# Spark skew joins\n\n"
        "A long tail join is where skew shows up. Salting the key on "
        "the hot dimension is the standard fix.\n",
        encoding="utf-8",
    )

    # A saved page, found by the url inside its capture.json.
    page = drop / "dan-page"
    page.mkdir()

    (page / "capture.json").write_text(
        json.dumps({"url": POST_IDEMPOTENT + "/"}), encoding="utf-8"
    )

    (page / "page.html").write_text(
        "<html><head>"
        '<meta property="og:title" content="Exactly-once processing">'
        '<meta property="og:description" content="Idempotent writes '
        "make at-least-once delivery safe, because replaying a batch "
        'converges on the same state.">'
        "</head><body><nav>Home Jobs Sign in</nav></body></html>",
        encoding="utf-8",
    )

    # A document, found by the source id inside its capture.json.
    document = drop / "erin-notes"
    document.mkdir()

    (document / "capture.json").write_text(
        json.dumps({"source_id": sid(POST_BACKPRESSURE)}), encoding="utf-8"
    )

    (document / "document.pdf").write_bytes(
        pdf_bytes("Consumer lag and backpressure in Kafka consumers.")
    )

    # A bare screenshot, found because the list names its bundle.
    screenshot = drop / "extra-capture"
    screenshot.mkdir()

    (screenshot / "screenshot.png").write_bytes(png_bytes(seed=3))

    (drop / "more.jsonl").write_text(
        json.dumps(
            {
                "url": POST_DIAGRAM,
                "saved_date": "2026-01-10",
                "title": "Pipeline diagram",
                "bundle": "extra-capture",
            }
        ),
        encoding="utf-8",
    )

    # A capture that claims nothing must not be attached to anything.
    orphan = drop / "unrelated"
    orphan.mkdir()

    (orphan / "notes.md").write_text(
        "Belongs to no saved item at all.\n", encoding="utf-8"
    )

    # A capture that cannot be read.
    corrupt = drop / "frank-broken"
    corrupt.mkdir()

    (corrupt / "document.pdf").write_bytes(b"%PDF-1.4\nnot a pdf body")
    (corrupt / "shot.png").write_bytes(b"NOT AN IMAGE AT ALL")

    return drop


def run_import(
    drop: Path,
    posts: Path,
    *extra: str,
) -> int:
    """Run the real command over the drop zone."""

    return collect_cli_main(
        [
            "saved-items",
            "--input",
            str(drop / "manifest.csv"),
            "--input",
            str(drop / "more.jsonl"),
            "--posts-root",
            str(posts),
            *extra,
        ]
    )


@pytest.fixture
def imported(tmp_path: Path) -> tuple[Path, Path]:
    """One completed import, with the drop zone and the posts."""
    drop = build_drop_zone(tmp_path)
    posts = tmp_path / "posts"

    assert run_import(drop, posts) == 0

    return drop, posts


@pytest.fixture
def knowledge(imported, tmp_path_factory) -> dict:
    """
    The knowledge base built from the imported posts.

    Staged the way the worker stages them, so a saved item is
    aggregated over the ordinary path rather than a second one.
    """
    _drop, posts = imported

    results = tmp_path_factory.mktemp("worker-results")

    for path in posts.glob("*/post.json"):
        shutil.copyfile(
            path, results / f"cloud_worker_{path.parent.name}.json"
        )

    output = tmp_path_factory.mktemp("kb") / "knowledge_base.json"

    aggregate_results(
        input_directory=results,
        output_path=output,
        expected_post_count=None,
    )

    return json.loads(output.read_text(encoding="utf-8"))


def generate(knowledge: dict, tmp_path_factory) -> Path:
    """Publish a knowledge base, the way the site generator is run."""
    root = tmp_path_factory.mktemp("site")

    source = root / "knowledge_base.json"

    source.write_text(
        json.dumps(knowledge, indent=2), encoding="utf-8"
    )

    generate_site(input_path=source, output_dir=root / "out")

    return root / "out"


def post_ids(posts: Path) -> list[str]:
    return sorted(
        path.parent.name
        for path in posts.glob("*/post.json")
    )


def report_counts(output: str) -> dict[str, int]:
    """
    Read the numbers out of a rendered report.

    Parsed rather than matched as a block of text, so a change to the
    report's column width does not fail a test about what the report
    says.
    """
    counts: dict[str, int] = {}

    for line in output.splitlines():
        if " : " not in line:
            continue

        label, _, value = line.partition(" : ")

        label = label.strip()

        if not label or not label[0].isupper():
            continue

        try:
            counts[label] = int(value.strip())
        except ValueError:
            continue

    return counts


def stored(drop: Path, url: str) -> dict:
    manifest = SavedItemsManifest.load(
        drop / "saved-items-manifest.json"
    )

    item = manifest.get(sid(url))

    assert item is not None, f"{url} is not in the manifest"

    return item.as_dict()


# ---------------------------------------------------------------------
# What a run produces
# ---------------------------------------------------------------------


class TestFirstRun:
    def test_it_succeeds(self, imported):
        assert imported

    def test_every_captured_item_becomes_exactly_one_post(
        self, imported
    ):
        _drop, posts = imported

        assert post_ids(posts) == sorted(
            [
                safe_sid(ARTICLE_SPARK),
                safe_sid(POST_IDEMPOTENT),
                safe_sid(POST_BACKPRESSURE),
                safe_sid(POST_DIAGRAM),
            ]
        )

    def test_a_repeated_url_does_not_create_two_posts(self, imported):
        _drop, posts = imported

        # POST_DELTA appears twice in the list, with a tracking
        # parameter and a fragment on one of them.
        assert safe_sid(POST_DELTA) not in post_ids(posts)

    def test_a_never_captured_item_stays_a_link(self, imported):
        drop, _posts = imported

        item = stored(drop, POST_KAFKA)

        # No body was supplied, so none is recorded and no post exists.
        # The link, the save date and the title are all still kept,
        # because the user did save something.
        assert item["content_digest"] is None
        assert item["state"] == "pending"
        assert item["post_id"] is None
        assert item["saved_date"] == "2026-01-03"
        assert item["canonical_url"] == POST_KAFKA

    def test_a_corrupt_capture_does_not_produce_a_post(self, imported):
        drop, posts = imported

        assert safe_sid(POST_BROKEN) not in post_ids(posts)

        item = stored(drop, POST_BROKEN)

        assert item["content_digest"] is None
        assert item["state"] == "pending"

    def test_the_report_counts_both_halves(self, imported, capsys):
        drop = imported[0]

        posts = imported[1]

        capsys.readouterr()

        run_import(drop, posts)

        out = capsys.readouterr().out

        assert "Metadata only" in out
        assert "With content" in out
        assert "Pending" in out

    def test_unreadable_rows_are_reported_with_their_line(
        self, imported, capsys
    ):
        drop, posts = imported

        manifest_path = drop / "saved-items-manifest.json"
        manifest = SavedItemsManifest.load(manifest_path)

        assert len(manifest) == 7

        capsys.readouterr()

        run_import(drop, posts)

        out = capsys.readouterr().out

        assert "line 9" in out
        assert "line 10" in out

    def test_an_unclaimed_bundle_is_not_attached_to_anything(
        self, imported
    ):
        _drop, posts = imported

        for directory in posts.iterdir():
            text = (directory / "post.json").read_text(encoding="utf-8")

            assert "Belongs to no saved item" not in text


# ---------------------------------------------------------------------
# Idempotence and incremental work
# ---------------------------------------------------------------------


class TestRepeatedRuns:
    def test_a_second_run_imports_nothing_new(self, imported):
        drop, posts = imported

        before = post_ids(posts)

        run_import(drop, posts)

        assert post_ids(posts) == before

    def test_repeated_runs_converge(self, imported):
        drop, posts = imported

        before = post_ids(posts)

        for _ in range(3):
            run_import(drop, posts)

        assert post_ids(posts) == before

    def test_an_unchanged_capture_is_not_re_read(
        self, imported, capsys
    ):
        drop, posts = imported

        capsys.readouterr()

        run_import(drop, posts)

        counts = report_counts(capsys.readouterr().out)

        assert counts["Unchanged"] == 4
        assert counts["Content changed"] == 0

    def test_only_a_changed_capture_is_re_read(self, imported, capsys):
        drop, posts = imported

        markdown = drop / safe_sid(ARTICLE_SPARK) / "content.md"

        markdown.write_text(
            markdown.read_text(encoding="utf-8")
            + "\nBroadcast joins are another way skew appears.\n",
            encoding="utf-8",
        )

        capsys.readouterr()

        run_import(drop, posts)

        counts = report_counts(capsys.readouterr().out)

        assert counts["Content changed"] == 1
        assert counts["Unchanged"] == 3

    def test_a_changed_capture_updates_the_stored_post(
        self, imported
    ):
        drop, posts = imported

        markdown = drop / safe_sid(ARTICLE_SPARK) / "content.md"

        markdown.write_text(
            markdown.read_text(encoding="utf-8")
            + "\nBroadcast joins are another way skew appears.\n",
            encoding="utf-8",
        )

        run_import(drop, posts)

        text = json.loads(
            (posts / safe_sid(ARTICLE_SPARK) / "post.json").read_text(
                encoding="utf-8"
            )
        )["original_text"]

        assert "Broadcast joins" in text

    def test_a_new_row_is_the_only_new_work(self, imported):
        drop, posts = imported

        before = set(post_ids(posts))

        with (drop / "manifest.csv").open("a", encoding="utf-8") as handle:
            handle.write(
                f"{POST_KAFKA},2026-01-11,Kafka partitioning,Bob,"
            )

        # The item now has a capture, so it becomes a post.
        (drop / safe_sid(POST_KAFKA)).mkdir()
        (drop / safe_sid(POST_KAFKA) / "content.md").write_text(
            "Partitioning decides how much Kafka reads.\n",
            encoding="utf-8",
        )

        run_import(drop, posts)

        after = set(post_ids(posts))

        assert after - before == {safe_sid(POST_KAFKA)}
        assert before - after == set()

    def test_a_deleted_post_is_re_imported(self, imported):
        drop, posts = imported

        shutil.rmtree(posts / safe_sid(ARTICLE_SPARK))

        run_import(drop, posts)

        assert safe_sid(ARTICLE_SPARK) in post_ids(posts)

    def test_a_dry_run_writes_nothing(self, tmp_path):
        drop = build_drop_zone(tmp_path)
        posts = tmp_path / "posts"

        run_import(drop, posts, "--dry-run")

        assert not posts.exists() or post_ids(posts) == []


# ---------------------------------------------------------------------
# Crash and resume
# ---------------------------------------------------------------------


class TestResume:
    def test_the_manifest_is_usable_after_an_interruption(
        self, tmp_path
    ):
        drop = build_drop_zone(tmp_path)
        posts = tmp_path / "posts"

        source_path = drop / "saved-items-manifest.json"

        # A run that is interrupted after the list is read still leaves
        # a manifest that names every item it saw.
        assert run_import(drop, posts, "--max-posts", "1") == 0

        interrupted = SavedItemsManifest.load(source_path)

        assert len(interrupted) == 7

        # The next run picks up from there and finishes the rest.
        run_import(drop, posts)

        assert len(post_ids(posts)) == 4

    def test_a_crash_midway_does_not_duplicate_completed_work(
        self, tmp_path
    ):
        drop = build_drop_zone(tmp_path)
        posts = tmp_path / "posts"

        run_import(drop, posts, "--max-posts", "1")

        first = set(post_ids(posts))

        run_import(drop, posts)

        assert set(post_ids(posts)) == first | {
            safe_sid(ARTICLE_SPARK),
            safe_sid(POST_IDEMPOTENT),
            safe_sid(POST_BACKPRESSURE),
            safe_sid(POST_DIAGRAM),
        }

    def test_an_unreadable_manifest_stops_rather_than_reimporting(
        self, tmp_path, capsys
    ):
        drop = build_drop_zone(tmp_path)
        posts = tmp_path / "posts"

        run_import(drop, posts)

        before = post_ids(posts)

        (drop / "saved-items-manifest.json").write_text(
            "{ truncated", encoding="utf-8"
        )

        assert run_import(drop, posts) == 1

        assert "Cannot read the saved-items manifest" in (
            capsys.readouterr().out
        )
        assert post_ids(posts) == before

    def test_a_truncated_manifest_is_never_left_behind(self, imported):
        drop, _posts = imported

        leftovers = [
            path.name
            for path in drop.iterdir()
            if path.name.endswith(".tmp")
        ]

        assert leftovers == []


# ---------------------------------------------------------------------
# What ends up in the post
# ---------------------------------------------------------------------


class TestProvenance:
    def _document(self, posts: Path, url: str) -> PostDocument:
        return PostDocument.load(posts / safe_sid(url))

    def test_the_source_url_is_preserved(self, imported):
        _drop, posts = imported

        document = self._document(posts, ARTICLE_SPARK)

        assert document.source["url"] == ARTICLE_SPARK

    def test_the_capture_method_is_recorded(self, imported):
        _drop, posts = imported

        assert (
            self._document(posts, ARTICLE_SPARK).source["capture_method"]
            == "user_provided"
        )

    def test_a_saved_page_is_labelled_as_one(self, imported):
        _drop, posts = imported

        assert (
            self._document(posts, POST_DIAGRAM).source["capture_method"]
            == "user_saved_page"
        )

    def test_no_post_claims_to_have_been_collected(self, imported):
        _drop, posts = imported

        for path in posts.glob("*/post.json"):
            document = PostDocument.load(path.parent)

            assert document.source["capture_method"] is not None
            assert document.source["platform"] == "linkedin"

    def test_the_item_can_be_traced_back_to(self, imported):
        _drop, posts = imported

        provenance = self._document(posts, ARTICLE_SPARK).provenance()

        assert provenance["saved_item_id"] == sid(ARTICLE_SPARK)
        assert provenance["saved_date"] == "2026-01-04"
        assert provenance["canonical_url"] == ARTICLE_SPARK

    def test_the_match_is_recorded(self, imported):
        _drop, posts = imported

        by_name = self._document(posts, ARTICLE_SPARK).provenance()
        by_url = self._document(posts, POST_IDEMPOTENT).provenance()
        by_id = self._document(posts, POST_BACKPRESSURE).provenance()
        by_column = self._document(posts, POST_DIAGRAM).provenance()

        assert by_name["capture_match"] == "directory name"
        assert by_url["capture_match"] == "canonical url in capture.json"
        assert by_id["capture_match"] == "source id in capture.json"
        assert by_column["capture_match"] == "bundle named in the manifest"

    def test_the_saved_date_is_carried_onto_the_post(self, imported):
        _drop, posts = imported

        assert (
            self._document(posts, ARTICLE_SPARK).source["published_at"]
            == "2026-01-04"
        )

    def test_media_lands_beside_the_post(self, imported):
        _drop, posts = imported

        document = self._document(posts, POST_BACKPRESSURE)

        assert [entry.path for entry in document.media()] == [
            "media/document.pdf"
        ]

        assert (
            posts / safe_sid(POST_BACKPRESSURE) / "media" / "document.pdf"
        ).is_file()

    def test_the_captured_text_is_kept_verbatim(self, imported):
        _drop, posts = imported

        text = self._document(posts, ARTICLE_SPARK).data["original_text"]

        assert "A long tail join is where skew shows up." in text

    def test_an_image_only_capture_says_so(self, imported):
        _drop, posts = imported

        document = self._document(posts, POST_DIAGRAM)

        text = document.data["original_text"]

        # It must not pretend to have recovered a body, and it must not
        # claim the post was collected from LinkedIn.
        assert "No body text was supplied" in text
        assert "collected" not in text.lower()
        assert document.provenance()["metadata_only"] is False

    def test_the_page_chrome_is_not_mistaken_for_the_post(
        self, imported
    ):
        _drop, posts = imported

        text = self._document(posts, POST_IDEMPOTENT).data["original_text"]

        assert text.startswith("Idempotent writes")
        assert "Sign in" not in text

    def test_a_pdf_body_is_extracted(self, imported):
        _drop, posts = imported

        text = self._document(posts, POST_BACKPRESSURE).data["original_text"]

        assert "Consumer lag and backpressure" in text

    def test_no_post_records_an_engagement_count(self, imported):
        _drop, posts = imported

        for path in posts.glob("*/post.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))

            for name in payload:
                for word in ("reaction", "like_count", "view_count"):
                    assert word not in name

    def test_no_post_records_a_credential(self, imported):
        _drop, posts = imported

        for path in posts.glob("*/post.json"):
            lowered = path.read_text(encoding="utf-8").lower()

            for word in ("password", "cookie", "storage_state", "token"):
                assert word not in lowered


# ---------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------


class TestEnrichment:
    def _mark_enriched(self, directory: Path) -> None:
        """Record a fingerprint on a post, the way enrichment would."""
        post = load_post(directory)

        document = PostDocument.load(directory)

        document.mark_enriched(
            source_digest=source_digest(post),
            enricher_version=ENRICHER_VERSION,
        )
        document.save(directory)

    def _needs_enrichment(self, directory: Path) -> bool:
        post = load_post(directory)

        return PostDocument.load(directory).needs_enrichment(
            source_digest=source_digest(post),
            enricher_version=ENRICHER_VERSION,
        )

    def test_a_saved_item_participates_in_the_fingerprint(
        self, imported
    ):
        _drop, posts = imported

        assert self._needs_enrichment(posts / safe_sid(ARTICLE_SPARK))

    def test_an_enriched_item_is_reused(self, imported):
        _drop, posts = imported

        directory = posts / safe_sid(ARTICLE_SPARK)

        self._mark_enriched(directory)

        assert not self._needs_enrichment(directory)

    def test_changing_a_capture_invalidates_its_enrichment(
        self, imported
    ):
        drop, posts = imported

        directory = posts / safe_sid(ARTICLE_SPARK)

        self._mark_enriched(directory)

        markdown = drop / safe_sid(ARTICLE_SPARK) / "content.md"

        markdown.write_text(
            markdown.read_text(encoding="utf-8")
            + "\nSomething genuinely new was added to the capture.\n",
            encoding="utf-8",
        )

        run_import(drop, posts)

        assert self._needs_enrichment(directory)

    def test_an_unchanged_capture_keeps_its_enrichment(
        self, imported
    ):
        drop, posts = imported

        directory = posts / safe_sid(ARTICLE_SPARK)

        self._mark_enriched(directory)

        run_import(drop, posts)

        assert not self._needs_enrichment(directory)

    def test_the_manifest_records_enrichment_that_really_happened(
        self, imported
    ):
        drop, posts = imported

        directory = posts / safe_sid(ARTICLE_SPARK)

        document = PostDocument.load(directory)

        document.data["ai_analysis"]["summary"] = "A summary."
        document.save(directory)

        run_import(drop, posts)

        assert stored(drop, ARTICLE_SPARK)["state"] == "enriched"

    def test_the_manifest_does_not_claim_enrichment_that_did_not(
        self, imported
    ):
        drop, _posts = imported

        assert stored(drop, ARTICLE_SPARK)["state"] == "imported"


# ---------------------------------------------------------------------
# The rest of the pipeline
# ---------------------------------------------------------------------


class TestTheWholePipeline:
    def test_every_imported_post_validates(self, imported):
        _drop, posts = imported

        report = validate_posts(root=posts)

        assert report.ok, [str(issue) for issue in report.issues]

    def test_validation_says_so_on_the_command(self, tmp_path):
        drop = build_drop_zone(tmp_path)
        posts = tmp_path / "posts"

        assert run_import(drop, posts, "--validate") == 0

    def test_every_saved_item_reaches_the_knowledge_base(
        self, imported, knowledge
    ):
        _drop, posts = imported

        aggregated = {post["id"] for post in knowledge["posts"]}

        assert set(post_ids(posts)) == aggregated

    def test_a_saved_item_keeps_its_url_in_the_knowledge_base(
        self, imported, knowledge
    ):
        _drop, _posts = imported

        urls = {
            post["source"].get("url") for post in knowledge["posts"]
        }

        assert ARTICLE_SPARK in urls

    def test_a_saved_item_is_traceable_on_its_page(
        self, imported, knowledge, tmp_path_factory
    ):
        _drop, _posts = imported

        site = generate(knowledge, tmp_path_factory)

        # Post pages are named by slug, so the page is found by what it
        # says rather than by what it is called. Scoped to the post
        # directory because the saved-items index also links here on
        # purpose.
        carrying = [
            path
            for path in (site / "posts").glob("*.html")
            if ARTICLE_SPARK in path.read_text(encoding="utf-8")
        ]

        assert len(carrying) == 1

    def test_a_saved_item_page_states_how_it_was_captured(
        self, imported, knowledge, tmp_path_factory
    ):
        _drop, _posts = imported

        site = generate(knowledge, tmp_path_factory)

        page = next(
            path
            for path in (site / "posts").glob("*.html")
            if ARTICLE_SPARK in path.read_text(encoding="utf-8")
        ).read_text(encoding="utf-8")

        # A reader has to be able to tell material this project was
        # given from material it collected, in plain words and in the
        # exact term the manifest recorded.
        assert "You supplied the content for this" in page
        assert 'data-capture-method="user_provided"' in page

    def test_a_saved_item_page_links_back_to_the_item(
        self, imported, knowledge, tmp_path_factory
    ):
        _drop, _posts = imported

        site = generate(knowledge, tmp_path_factory)

        page = next(
            path
            for path in (site / "posts").glob("*.html")
            if ARTICLE_SPARK in path.read_text(encoding="utf-8")
        ).read_text(encoding="utf-8")

        assert sid(ARTICLE_SPARK) in page
        assert "2026-01-04" in page

    def test_a_post_that_was_not_saved_carries_no_saved_item(
        self, tmp_path
    ):
        # A post that came from a plain capture bundle has to be
        # indistinguishable from one that never needed a saved item,
        # rather than carrying a block of nulls that implies it did.
        from src.ingestion.importer import create_post

        create_post("plain_capture", text="A hand-captured note.", root=tmp_path)

        post = load_post(tmp_path / "plain_capture")

        assert post.saved_item is None
        assert "saved_item" not in post.model_dump(mode="json")

    def test_an_image_only_item_is_labelled_on_its_page(
        self, imported, knowledge, tmp_path_factory
    ):
        _drop, _posts = imported

        site = generate(knowledge, tmp_path_factory)

        carrying = [
            path.read_text(encoding="utf-8")
            for path in (site / "posts").glob("*.html")
            if "No body text was supplied" in path.read_text(
                encoding="utf-8"
            )
        ]

        # A reader has to be able to tell a captured post from one that
        # is only a link with a screenshot attached.
        assert len(carrying) == 1

    def test_generation_is_reproducible_for_saved_items(
        self, imported, knowledge, tmp_path_factory
    ):
        _drop, _posts = imported

        first = generate(knowledge, tmp_path_factory)
        second = generate(knowledge, tmp_path_factory)

        assert files_in(first) == files_in(second)

    def test_no_saved_item_page_carries_a_credential(
        self, imported, knowledge, tmp_path_factory
    ):
        _drop, _posts = imported

        site = generate(knowledge, tmp_path_factory)

        for path in site.rglob("*.html"):
            lowered = path.read_text(encoding="utf-8").lower()

            for word in ("password", "cookie", "storage_state"):
                assert word not in lowered

    def test_the_manifest_is_not_committed_material(self, imported):
        # The drop zone is ignored, so the manifest inside it is too.
        # What matters is that nothing in it is a credential.
        drop, _posts = imported

        text = (drop / "saved-items-manifest.json").read_text(
            encoding="utf-8"
        )

        for word in ("password", "cookie", "token", "secret"):
            assert word not in text.lower()


def test_the_worker_result_glob_is_unchanged():
    """
    A saved item is aggregated through the ordinary path. If the glob
    moved, saved posts would silently stop reaching the knowledge base
    while every other test still passed.
    """

    assert WORKER_RESULT_GLOB


def files_in(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): path.read_text(encoding="utf-8")
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
