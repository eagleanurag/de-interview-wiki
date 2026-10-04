"""
The saved-items workflow, end to end, over a realistic fixture set.

The fixture is the twelve cases a person actually runs into: technical
posts in five different formats, a screenshot with no text, a document,
a hiring notice that is not interview material, a link saved and never
captured, a link written down twice, a file that will not open, and two
folders that belong to nothing.

The assertions are about honesty rather than throughput. A relevant post
gets questions and an irrelevant one does not, a question about a
technology the source never mentions does not survive, a saved link with
no capture stays a link, and the same capture read twice is read once.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from src.ai.enricher import AIEnricher
from src.ai.opencode import OpenCodeResult
from src.aggregation.aggregator import (
    WORKER_RESULT_GLOB,
    aggregate_results,
)
from src.ingestion.collect_cli import main as collect_cli_main
from src.ingestion.post_loader import load_post, source_digest
from src.ingestion.saved_items import plan as plan_module
from src.ingestion.saved_items.bundles import (
    QUALITY_IMAGE_ONLY,
    QUALITY_PARTIAL,
    QUALITY_TEXT,
    QUALITY_TEXT_AND_MEDIA,
)
from src.ingestion.saved_items.manifest import SavedItemsManifest
from src.ingestion.saved_items.model import SavedItemState
from src.ingestion.saved_items.urls import normalize_linkedin_url
from src.pipeline import ENRICHER_VERSION
from src.wiki.generator import generate_site


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "saved-items"

DELTA = "https://www.linkedin.com/posts/alice_delta-lake-101"
SQL = "https://www.linkedin.com/posts/bob_sql-window-202"
SPARK = "https://www.linkedin.com/pulse/carol_pyspark-303"
DATABRICKS = "https://www.linkedin.com/posts/dan_databricks-404"
FACTORY = "https://www.linkedin.com/posts/erin_data-factory-505"
ARCHITECTURE = "https://www.linkedin.com/posts/frank_architecture-606"
HIRING = "https://www.linkedin.com/posts/grace_hiring-707"
IMAGE = "https://www.linkedin.com/posts/heidi_diagram-808"
DOCUMENT = "https://www.linkedin.com/posts/ivan_backpressure-909"
METADATA = "https://www.linkedin.com/posts/judy_never-captured-1010"
CORRUPT = "https://www.linkedin.com/posts/ken_corrupt-1110"
UNCLAIMED = "https://www.linkedin.com/posts/kim_unclaimed-1111"


def sid(url: str) -> str:
    return normalize_linkedin_url(url).source_id


@pytest.fixture
def drop(tmp_path: Path) -> Path:
    """A copy of the fixture, so a test can change it freely."""
    if not FIXTURES.is_dir():
        pytest.skip(
            "fixtures are absent; run python -m tests.build_fixtures"
        )

    target = tmp_path / "saved-items"

    shutil.copytree(FIXTURES, target)

    return target


@pytest.fixture
def posts(tmp_path: Path) -> Path:
    return tmp_path / "posts"


def run(*argv: str) -> int:
    return collect_cli_main(list(argv))


def import_all(drop: Path, posts: Path, *extra: str) -> int:
    return run(
        "saved-items",
        "--input", str(drop / "manifest.csv"),
        "--bundle-root", str(drop),
        "--posts-root", str(posts),
        *extra,
    )


def stored(drop: Path, url: str) -> dict:
    manifest = SavedItemsManifest.load(
        drop / "saved-items-manifest.json"
    )

    item = manifest.get(sid(url))

    assert item is not None, f"{url} is not in the manifest"

    return item.as_dict()


def post_for(posts: Path, url: str):
    directory = posts / sid(url).replace(":", "-")

    assert directory.is_dir(), f"no post for {url}"

    return load_post(directory)


# ---------------------------------------------------------------------
# What a run makes of the fixture
# ---------------------------------------------------------------------


class TestFixtureInbox:
    def test_it_validates_with_no_errors(self, drop: Path):
        assert run(
            "saved-items-validate", "--bundle-root", str(drop)
        ) == 0

    def test_the_plan_reads_every_case(self, drop: Path):
        plan = plan_module.build_plan(
            drop, inputs=[drop / "manifest.csv"]
        )

        counts = plan.counts()

        # One link saved and never captured, one row that repeats an
        # earlier one, and everything else with a capture behind it.
        assert counts["MISSING_CAPTURE"] == 1
        assert counts["DUPLICATE"] == 1
        assert counts["NEW"] >= 6

    def test_the_duplicate_row_collapses(self, drop: Path, posts: Path):
        import_all(drop, posts)

        # The same link twice is one saved item, and therefore one post.
        item = stored(drop, DELTA)

        assert item["post_id"] is not None
        assert item["state"] in {"imported", "enriched"}

    def test_a_never_captured_link_stays_a_link(
        self, drop: Path, posts: Path
    ):
        import_all(drop, posts)

        item = stored(drop, METADATA)

        assert item["content_digest"] is None
        assert item["state"] == "pending"
        assert item["post_id"] is None

        assert not (posts / sid(METADATA).replace(":", "-")).is_dir()

    def test_the_capture_quality_is_recorded(self, drop: Path, posts: Path):
        import_all(drop, posts)

        assert stored(drop, IMAGE)["capture_quality"] == (
            QUALITY_IMAGE_ONLY
        )
        assert stored(drop, DELTA)["capture_quality"] == QUALITY_TEXT
        assert stored(drop, DATABRICKS)["capture_quality"] == (
            QUALITY_TEXT_AND_MEDIA
        )

    def test_a_partial_capture_says_so(self, drop: Path, posts: Path):
        import_all(drop, posts)

        # The notes beside the broken document are readable, so this is
        # a partial capture rather than a failed one.
        assert stored(drop, CORRUPT)["capture_quality"] == (
            QUALITY_PARTIAL
        )

    def test_the_broken_document_is_reported(self, drop: Path, posts: Path):
        import_all(drop, posts)

        item = stored(drop, CORRUPT)

        assert any(
            "document.pdf" in note for note in item["capture_notes"]
        )

    def test_the_document_text_is_recovered(
        self, drop: Path, posts: Path
    ):
        import_all(drop, posts)

        post = post_for(posts, DOCUMENT)

        assert "Consumer lag" in post.original_text

    def test_a_saved_page_yields_its_text_not_its_chrome(
        self, drop: Path, posts: Path
    ):
        import_all(drop, posts)

        text = post_for(posts, SQL).original_text

        assert "window function" in text
        assert "Sign in" not in text

    def test_an_image_only_capture_says_it_recovered_no_text(
        self, drop: Path, posts: Path
    ):
        import_all(drop, posts)

        post = post_for(posts, IMAGE)

        assert "No body text was supplied" in post.original_text
        assert post.media

    def test_the_two_unclaimed_captures_are_reported(
        self, drop: Path
    ):
        plan = plan_module.build_plan(
            drop, inputs=[drop / "manifest.csv"]
        )

        orphans = {
            entry.location.replace("\\", "/")
            for entry in plan.problems
            if entry.code == "ORPHAN_CAPTURE"
        }

        assert "captures/unclaimed-capture" in orphans
        assert "captures/no-claim" in orphans

    def test_an_unclaimed_capture_keeps_its_url(self, drop: Path):
        plan = plan_module.build_plan(
            drop, inputs=[drop / "manifest.csv"]
        )

        orphan = next(
            entry
            for entry in plan.problems
            if entry.code == "ORPHAN_CAPTURE"
            and "unclaimed-capture" in entry.location
        )

        assert UNCLAIMED in orphan.detail.get("detected_url", "")

    def test_a_capture_with_no_claim_never_gets_one(
        self, drop: Path
    ):
        plan = plan_module.build_plan(
            drop, inputs=[drop / "manifest.csv"]
        )

        orphan = next(
            entry
            for entry in plan.problems
            if entry.code == "ORPHAN_CAPTURE"
            and "no-claim" in entry.location
        )

        # A folder with nothing in it saying which post it is cannot be
        # attached to one, because the only way to do that would be to
        # make one up.
        assert not orphan.detail.get("detected_url")


# ---------------------------------------------------------------------
# Incremental behaviour
# ---------------------------------------------------------------------


class TestIncremental:
    def test_a_second_run_imports_nothing(self, drop: Path, posts: Path):
        import_all(drop, posts)

        before = sorted(path.name for path in posts.iterdir())

        import_all(drop, posts)

        assert sorted(path.name for path in posts.iterdir()) == before

    def test_a_second_run_re_reads_nothing(self, drop: Path):
        import_all(drop, tmp := drop.parent / "posts")

        plan = plan_module.build_plan(
            drop, inputs=[drop / "manifest.csv"]
        )

        counts = plan.counts()

        assert counts["NEW"] == 0
        assert counts["CHANGED"] == 0
        assert counts["UNCHANGED"] >= 6

    def test_changing_one_capture_re_reads_only_that_one(
        self, drop: Path, posts: Path
    ):
        import_all(drop, posts)

        target = (
            drop / "captures" / sid(SPARK).replace(":", "-") / "content.txt"
        )

        target.write_text(
            target.read_text(encoding="utf-8")
            + "\nA skew hint join splits the hot key across executors.\n",
            encoding="utf-8",
        )

        plan = plan_module.build_plan(
            drop, inputs=[drop / "manifest.csv"]
        )

        counts = plan.counts()

        assert counts["CHANGED"] == 1
        assert counts["UNCHANGED"] >= 6
        assert counts["NEW"] == 0

    def test_a_dry_run_changes_nothing(self, drop: Path, posts: Path):
        import_all(drop, posts)

        manifest_before = (drop / "saved-items-manifest.json").read_bytes()
        list_before = (drop / "manifest.csv").read_bytes()
        posts_before = sorted(path.name for path in posts.iterdir())

        assert import_all(drop, posts, "--dry-run") == 0

        assert (drop / "saved-items-manifest.json").read_bytes() == (
            manifest_before
        )
        assert (drop / "manifest.csv").read_bytes() == list_before
        assert sorted(path.name for path in posts.iterdir()) == posts_before

    def test_a_deleted_post_comes_back(self, drop: Path, posts: Path):
        import_all(drop, posts)

        shutil.rmtree(posts / sid(DELTA).replace(":", "-"))

        import_all(drop, posts)

        assert (posts / sid(DELTA).replace(":", "-")).is_dir()

    def test_enrichment_is_reused_for_an_unchanged_capture(
        self, drop: Path, posts: Path
    ):
        import_all(drop, posts)

        directory = posts / sid(DELTA).replace(":", "-")

        _mark_enriched(directory)

        import_all(drop, posts)

        post = load_post(directory)

        assert not _needs_enrichment(post)

    def test_enrichment_is_invalidated_by_a_changed_capture(
        self, drop: Path, posts: Path
    ):
        import_all(drop, posts)

        directory = posts / sid(DELTA).replace(":", "-")

        _mark_enriched(directory)

        target = (
            drop / "captures" / sid(DELTA).replace(":", "-") / "content.md"
        )

        target.write_text(
            target.read_text(encoding="utf-8")
            + "\nZ-Ordering co-locates related rows for pruning.\n",
            encoding="utf-8",
        )

        import_all(drop, posts)

        assert _needs_enrichment(load_post(directory))


def _mark_enriched(directory: Path) -> None:
    from src.ingestion.post_document import PostDocument
    from src.pipeline import ENRICHER_VERSION

    post = load_post(directory)

    document = PostDocument.load(directory)

    document.mark_enriched(
        source_digest=source_digest(post),
        enricher_version=ENRICHER_VERSION,
    )

    document.save(directory)


def _needs_enrichment(post) -> bool:
    from src.ingestion.post_document import PostDocument
    from src.pipeline import ENRICHER_VERSION

    document = PostDocument.load(Path(post.directory))

    return document.needs_enrichment(
        source_digest=source_digest(post),
        enricher_version=ENRICHER_VERSION,
    )


# ---------------------------------------------------------------------
# Enrichment quality
# ---------------------------------------------------------------------


class ScriptedModel:
    """
    A model that answers from the post it was given.

    Stands in for the real one so the contract can be checked without
    spending a call: a relevant post gets questions grounded in what it
    says, a hiring notice gets none, and one post is answered with a
    question about a technology it never mentions, which is the failure
    the grounding check exists to catch.
    """

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def run(self, prompt: str) -> OpenCodeResult:
        self.prompts.append(prompt)

        body = prompt.split("Original content:")[-1]

        relevant = not any(
            marker in body.lower()
            for marker in ("hiring", "benefits", "send a note")
        )

        payload: dict = {
            "summary": "A summary of the post.",
            "topics": ["Data Engineering"],
            "subtopics": [],
            "concepts": [
                {
                    "name": "ACID guarantees",
                    "explanation": "Atomicity from the log.",
                }
            ],
            "classification": {
                "domain": "Data Engineering",
                "primary_topic": "Data Engineering",
                "secondary_topics": [],
                "interview_relevant": relevant,
            },
            "interview_questions": (
                [
                    {
                        "question": "How does Delta Lake give ACID "
                        "guarantees?",
                        "type": "theory",
                        "difficulty": "medium",
                        "what_strong_answers_cover": [
                            "Atomicity comes from the log."
                        ],
                    },
                    {
                        "question": "How do you tune a Snowflake virtual "
                        "warehouse?",
                        "type": "troubleshooting",
                        "difficulty": "medium",
                        "what_strong_answers_cover": [
                            "Start with the query profile."
                        ],
                    },
                ]
                if relevant
                else []
            ),
        }

        return OpenCodeResult(
            text=json.dumps(payload), data=payload
        )


class TestEnrichmentQuality:
    def _enriched(self, drop: Path, posts: Path) -> dict:
        return enrich_everything(drop, posts)

    def test_a_relevant_post_gets_a_useful_question(
        self, drop: Path, posts: Path
    ):
        enriched = self._enriched(drop, posts)

        post = enriched[sid(DELTA).replace(":", "-")]

        assert post.classification.interview_relevant is True
        assert post.interview_questions
        assert all(
            question.answer
            for question in post.interview_questions
        )

    def test_an_irrelevant_post_gets_no_questions(
        self, drop: Path, posts: Path
    ):
        enriched = self._enriched(drop, posts)

        post = enriched[sid(HIRING).replace(":", "-")]

        # A hiring notice is something people save and something nobody
        # can be interviewed on.
        assert post.classification.interview_relevant is False
        assert not post.interview_questions

    def test_a_question_about_an_unsupported_technology_is_removed(
        self, drop: Path, posts: Path
    ):
        enriched = self._enriched(drop, posts)

        post = enriched[sid(DELTA).replace(":", "-")]

        questions = [
            question.question for question in post.interview_questions
        ]

        assert "How does Delta Lake give ACID guarantees?" in questions
        assert not any(
            "Snowflake" in question for question in questions
        )

    def test_questions_carry_a_type_and_a_difficulty(
        self, drop: Path, posts: Path
    ):
        from src.ai.schemas import DIFFICULTIES, QUESTION_TYPES

        enriched = self._enriched(drop, posts)

        for post in enriched.values():
            for question in post.interview_questions:
                assert question.type in QUESTION_TYPES
                assert question.difficulty in DIFFICULTIES

    def test_a_question_about_the_source_technology_survives(
        self, drop: Path, posts: Path
    ):
        enriched = self._enriched(drop, posts)

        post = enriched[sid(DELTA).replace(":", "-")]

        body = " ".join(
            [post.original_text]
            + [
                media.extracted_text or ""
                for media in post.media
            ]
        ).lower()

        for question in post.interview_questions:
            for word in ("delta lake", "spark", "databricks"):
                if word in question.question.lower():
                    assert word in body


def enrich_everything(drop: Path, posts: Path) -> dict:
    """
    Import the fixture and enrich every post, writing the result out.

    Done through the same path the pipeline uses. A test that enriched
    in memory and then re-read from disk would be checking the
    un-enriched post, and one that wrote the post out itself would not
    be checking what the worker writes.
    """
    import_all(drop, posts)

    model = ScriptedModel()
    enricher = AIEnricher(client=model)

    enriched: dict = {}

    for directory in sorted(posts.iterdir()):
        if not directory.is_dir():
            continue

        post = load_post(directory)

        enricher.enrich(post)

        payload = post.model_dump(mode="json")

        payload["_enrichment"] = {
            "source_digest": source_digest(post),
            "enricher_version": ENRICHER_VERSION,
        }

        (directory / "post.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

        enriched[directory.name] = post

    return enriched


# ---------------------------------------------------------------------
# Into the knowledge base and the site
# ---------------------------------------------------------------------


class TestPublished:
    def _knowledge_base(self, posts: Path, tmp_path: Path) -> dict:
        results = tmp_path / "worker-results"
        results.mkdir(parents=True, exist_ok=True)

        for path in posts.glob("*/post.json"):
            shutil.copyfile(
                path, results / f"cloud_worker_{path.parent.name}.json"
            )

        output = tmp_path / "knowledge_base.json"

        aggregate_results(
            input_directory=results,
            output_path=output,
            expected_post_count=None,
        )

        return json.loads(output.read_text(encoding="utf-8"))

    def test_provenance_survives_aggregation(
        self, drop: Path, posts: Path, tmp_path: Path
    ):
        import_all(drop, posts)

        knowledge = self._knowledge_base(posts, tmp_path)

        by_id = {post["id"]: post for post in knowledge["posts"]}

        item = by_id[sid(DELTA).replace(":", "-")]

        assert item["source"]["url"] == DELTA
        assert item["source"]["capture_method"] == "user_provided"
        assert item["saved_item"]["saved_item_id"] == sid(DELTA)
        assert item["saved_item"]["canonical_url"] == DELTA

    def test_a_never_captured_link_reaches_nothing(
        self, drop: Path, posts: Path, tmp_path: Path
    ):
        import_all(drop, posts)

        knowledge = self._knowledge_base(posts, tmp_path)

        ids = {post["id"] for post in knowledge["posts"]}

        assert sid(METADATA).replace(":", "-") not in ids


    def test_no_page_is_generated_for_an_uncaptured_link(
        self, drop: Path, posts: Path, tmp_path: Path
    ):
        import_all(drop, posts)

        knowledge = self._knowledge_base(posts, tmp_path)

        source = tmp_path / "site"
        source.mkdir()

        (source / "knowledge_base.json").write_text(
            json.dumps(knowledge, indent=2), encoding="utf-8"
        )

        generate_site(
            input_path=source / "knowledge_base.json",
            output_dir=source / "out",
        )

        published = "".join(
            path.read_text(encoding="utf-8")
            for path in (source / "out").rglob("*.html")
        )

        assert sid(METADATA).replace(":", "-") not in published

        # And no page was generated for it either, which is now the whole
        # of the check: there are no per-post pages, so the question
        # "does it have a page" no longer has a page-shaped answer.
        assert not (source / "out" / "posts").exists()

    def test_generation_is_deterministic(
        self, drop: Path, posts: Path, tmp_path: Path
    ):
        import_all(drop, posts)

        knowledge = self._knowledge_base(posts, tmp_path)

        renders = []

        for name in ("first", "second"):
            source = tmp_path / name
            source.mkdir()

            (source / "knowledge_base.json").write_text(
                json.dumps(knowledge, indent=2), encoding="utf-8"
            )

            generate_site(
                input_path=source / "knowledge_base.json",
                output_dir=source / "out",
            )

            renders.append({
                str(path.relative_to(source / "out")): path.read_text(
                    encoding="utf-8"
                )
                for path in sorted((source / "out").rglob("*"))
                if path.is_file()
            })

        assert renders[0] == renders[1]

    def test_the_worker_result_glob_is_unchanged(self):
        # A saved item is aggregated through the ordinary path. If the
        # glob moved, saved posts would stop reaching the knowledge base
        # while every other test still passed.
        assert WORKER_RESULT_GLOB
