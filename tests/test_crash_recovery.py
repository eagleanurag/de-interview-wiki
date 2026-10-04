"""
Tests that every stage survives an interruption.

The property being checked is not "the function returned" but "the
next run resumes correctly". A stage that half-writes its output, or
that loses what it had already done, would not show up until a real
interruption, which is the worst time to find it.

Each test interrupts a stage the way a power loss would: partway
through, with the output left in whatever state that produced.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.aggregation.aggregator import (
    AggregationError,
    aggregate_results,
)
from src.ingestion.checkpoints import (
    CP5_AUTHENTICATION_READY,
    CP8_ENRICHMENT_COMPLETE,
    Checkpoint,
    current_file,
    read,
    write,
)
from src.ingestion.collect import Collector, CollectionLimits, read_state
from src.ingestion.importer import discover_posts, validate_posts
from src.ingestion.post_document import PostDocument
from src.ingestion.sources.base import CollectedPost
from src.ingestion.sources.manual import ManualSource
from src.wiki.generator import generate_site


# ---------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------


def make_bundle(root: Path, name: str, text: str) -> Path:
    bundle = root / name
    bundle.mkdir(parents=True)
    (bundle / "post.md").write_text(text, encoding="utf-8")

    return bundle


def test_an_interrupted_collection_resumes_without_duplicating(tmp_path):
    """
    Half the bundles are collected, then the run stops. The next run
    picks up the rest and creates no second copy of anything.
    """

    incoming = tmp_path / "incoming"

    for index in range(4):
        make_bundle(incoming, f"b{index}", f"Bundle {index} content.")

    posts = tmp_path / "posts"

    # First run collects two of them.
    Collector(
        ManualSource(incoming),
        root=posts,
        limits=CollectionLimits(max_posts=2),
    ).run()

    after_first = {path.name for path in posts.iterdir() if path.is_dir()}
    assert len(after_first) == 2

    # Second run collects everything.
    Collector(ManualSource(incoming), root=posts).run()

    after_second = {path.name for path in posts.iterdir() if path.is_dir()}

    assert len(after_second) == 4
    assert after_first < after_second


def test_interrupted_collection_state_is_reusable(tmp_path):
    """
    A state file left by a previous run is what makes a resume
    possible, so it has to be readable and must describe what happened.
    """

    incoming = tmp_path / "incoming"
    make_bundle(incoming, "b0", "Content.")

    posts = tmp_path / "posts"
    repository = tmp_path / "repository"

    Collector(
        ManualSource(incoming),
        root=posts,
        repository_root=repository,
    ).run()

    state = read_state(repository)

    assert state is not None
    assert state.persisted == 1
    assert state.last_post_id


def test_a_corrupt_state_file_does_not_stop_a_run(tmp_path):
    """
    A state file half-written by a power loss is unreadable. Losing the
    resume point is acceptable; refusing to collect is not.
    """

    incoming = tmp_path / "incoming"
    make_bundle(incoming, "b0", "Content.")

    posts = tmp_path / "posts"
    repository = tmp_path / "repository"

    state_file = repository / ".agent" / "checkpoints" / "collection.json"
    state_file.parent.mkdir(parents=True)
    state_file.write_text("{ truncated", encoding="utf-8")

    report = Collector(
        ManualSource(incoming),
        root=posts,
        repository_root=repository,
    ).run()

    assert report.state.persisted == 1


def test_a_partial_post_directory_is_repaired(tmp_path):
    """
    A post directory written before its document is complete is
    overwritten rather than left half-made.
    """

    incoming = tmp_path / "incoming"
    make_bundle(incoming, "b0", "Real content.")

    posts = tmp_path / "posts"
    posts.mkdir()

    # A directory with no post.json, as an interruption would leave.
    (posts / "leftover").mkdir()

    Collector(ManualSource(incoming), root=posts).run()

    report = validate_posts(root=posts)

    assert report.ok, [str(issue) for issue in report.issues]


# ---------------------------------------------------------------------
# Media
# ---------------------------------------------------------------------


def test_a_half_written_post_document_is_replaced(tmp_path):
    """
    The document layer writes atomically, so a document on disk is
    either the old one or the new one. A truncated file written by
    something else is replaced rather than trusted.
    """

    directory = tmp_path / "post"
    directory.mkdir()

    (directory / "post.json").write_text("{ truncated", encoding="utf-8")

    # The loader refuses rather than returning a half record.
    with pytest.raises(Exception):
        PostDocument.load_file(directory / "post.json")

    # And a fresh write produces something readable.
    PostDocument.new("post", text="Recovered.").save(directory)

    reloaded = PostDocument.load_file(directory / "post.json")

    assert reloaded.original_text == "Recovered."


# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------


def stage_results(tmp_path: Path, count: int) -> Path:
    from datetime import datetime, timezone

    from src.models import (
        AIAnalysis,
        Classification,
        KnowledgePost,
        SourceInfo,
    )

    results = tmp_path / "worker-results"
    results.mkdir(parents=True)

    for index in range(count):
        post = KnowledgePost(
            id=f"post-{index}",
            source=SourceInfo(
                platform="manual",
                captured_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
            original_text=f"Content {index}.",
            ai_analysis=AIAnalysis(
                summary=f"Summary {index}.",
                topics=["Spark"],
                concepts=["Partitioning"],
            ),
            interview_questions=[],
            classification=Classification(
                domain="Data Engineering",
                primary_topic="Spark",
                interview_relevant=True,
            ),
        )

        (results / f"cloud_worker_{post.id}.json").write_text(
            json.dumps(post.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )

    return results


def test_a_partial_knowledge_base_never_survives(tmp_path):
    """
    Aggregation stages into a temporary file and moves it into place, so
    an interrupted run leaves either the previous knowledge base or
    none, never a half-written one.
    """

    results = stage_results(tmp_path, 3)
    output = tmp_path / "knowledge_base.json"

    aggregate_results(input_directory=results, output_path=output)

    first = json.loads(output.read_text(encoding="utf-8"))

    assert first["stats"]["posts_aggregated"] == 3

    # No temporary file is left behind.
    assert list(tmp_path.glob("*.tmp")) == []


def test_a_failed_aggregation_leaves_the_previous_knowledge_base(
    tmp_path,
):
    """
    A run that fails must not destroy what the last good run produced.
    """

    results = stage_results(tmp_path, 2)
    output = tmp_path / "knowledge_base.json"

    aggregate_results(input_directory=results, output_path=output)

    before = output.read_text(encoding="utf-8")

    # A second run that cannot succeed.
    with pytest.raises(AggregationError):
        aggregate_results(
            input_directory=tmp_path / "absent",
            output_path=output,
        )

    assert output.read_text(encoding="utf-8") == before


# ---------------------------------------------------------------------
# Wiki
# ---------------------------------------------------------------------


def test_a_partial_site_never_serves(tmp_path):
    """
    The site is staged and swapped in, so an interrupted generation
    leaves the previous site rather than a mixture of two.
    """

    kb = tmp_path / "knowledge_base.json"
    kb.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "generated_at": "2026-01-01T00:00:00Z",
                "stats": {},
                "skipped_files": [],
                "posts": [],
                "knowledge": {
                    "topics": [],
                    "subtopics": [],
                    "concepts": [],
                    "technologies": [],
                    "questions": [],
                    "content_kinds": {},
                },
            }
        ),
        encoding="utf-8",
    )

    site = tmp_path / "site"

    generate_site(kb, site)

    marker = site / "index.html"
    marker.write_text("previous build", encoding="utf-8")

    with pytest.raises(Exception):
        generate_site(tmp_path / "absent.json", site)

    # The previous build is untouched.
    assert marker.read_text(encoding="utf-8") == "previous build"


# ---------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------


def test_a_checkpoint_is_readable_after_being_written(tmp_path):
    write(
        Checkpoint(
            checkpoint_id=CP5_AUTHENTICATION_READY,
            phase=CP5_AUTHENTICATION_READY,
            completed=["a"],
        ),
        tmp_path,
    )

    loaded = read(tmp_path)

    assert loaded is not None
    assert loaded.checkpoint_id == CP5_AUTHENTICATION_READY
    assert loaded.completed == ["a"]


def test_a_corrupt_checkpoint_is_treated_as_absent(tmp_path):
    """
    An unreadable checkpoint cannot be trusted, so it is treated as
    absent rather than as a state to resume from.
    """

    path = current_file(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("{ truncated", encoding="utf-8")

    assert read(tmp_path) is None


def test_a_checkpoint_never_stores_a_credential(tmp_path):
    from src.ingestion.checkpoints import CheckpointError

    with pytest.raises(CheckpointError):
        write(
            Checkpoint(
                checkpoint_id=CP5_AUTHENTICATION_READY,
                phase=CP5_AUTHENTICATION_READY,
                blocker="password: hunter2-not-a-real-value",
            ),
            tmp_path,
        )

    assert not current_file(tmp_path).exists()


def test_writing_a_checkpoint_is_atomic(tmp_path):
    write(
        Checkpoint(
            checkpoint_id=CP5_AUTHENTICATION_READY,
            phase=CP5_AUTHENTICATION_READY,
        ),
        tmp_path,
    )

    write(
        Checkpoint(
            checkpoint_id=CP5_AUTHENTICATION_READY,
            phase=CP5_AUTHENTICATION_READY,
            completed=["second"],
        ),
        tmp_path,
    )

    loaded = read(tmp_path)

    assert loaded.completed == ["second"]

    # No temporary file survives.
    assert list(current_file(tmp_path).parent.glob("*.tmp")) == []


def test_concurrent_writers_do_not_collide_on_one_temporary(tmp_path):
    """
    Enrichment runs one worker per post in threads and every worker
    writes the same checkpoint, so "atomic" has to mean atomic between
    threads and not only between one writer and a reader.

    It did not. The temporary name was derived from the target, so every
    writer shared one temporary: a rename could be pulled out from under
    another thread mid-flight, and on Windows a reader holding the
    destination open made the rename fail outright -- reported as a
    pipeline error with nothing to do with the pipeline. A full test run
    failed this way on a shared machine.

    Each write now gets a temporary of its own and the whole read/write
    pair is serialised in-process, with the retry left for the case a
    lock genuinely cannot cover: two pipeline runs on one machine.
    """

    import threading

    errors: list[BaseException] = []
    partial: list[str] = []

    def hammer(index: int) -> None:
        for _ in range(25):
            try:
                write(
                    Checkpoint(
                        checkpoint_id=CP8_ENRICHMENT_COMPLETE,
                        phase=CP8_ENRICHMENT_COMPLETE,
                        completed=[f"post-{index}"],
                    ),
                    tmp_path,
                )

                loaded = read(tmp_path)

                assert loaded is not None
                assert loaded.phase == CP8_ENRICHMENT_COMPLETE

                # And the bytes on disk are a whole document, never the
                # middle of one.
                raw = current_file(tmp_path).read_text(encoding="utf-8")

                json.loads(raw)

                if not raw.rstrip().endswith("}"):
                    partial.append(raw[-40:])

            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

    threads = [
        threading.Thread(target=hammer, args=(index,))
        for index in range(8)
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    assert errors == [], [repr(e) for e in errors[:3]]
    assert partial == []

    # No temporary survives a concurrent run either.
    assert list(current_file(tmp_path).parent.glob("*.tmp")) == []

    # And the surviving checkpoint is one of the ones written, not a
    # blend of two: the completed list is per-writer.
    loaded = read(tmp_path)

    assert loaded.completed[0].startswith("post-")


# ---------------------------------------------------------------------
# Incremental enrichment
# ---------------------------------------------------------------------


def test_an_interrupted_enrichment_leaves_earlier_results_usable(
    tmp_path,
):
    """
    Results already written stay usable when a later one fails, so a
    run interrupted partway still has most of its work.
    """

    from src.pipeline import ENRICHER_VERSION, _reusable

    results = tmp_path / "worker-results"
    results.mkdir()

    for index in range(3):
        (results / f"cloud_worker_post-{index}.json").write_text(
            json.dumps(
                {
                    "id": f"post-{index}",
                    "_enrichment": {
                        "source_digest": f"digest-{index}",
                        "enricher_version": ENRICHER_VERSION,
                    },
                }
            ),
            encoding="utf-8",
        )

    reusable = [
        index
        for index in range(3)
        if _reusable(
            results / f"cloud_worker_post-{index}.json",
            f"digest-{index}",
        )
        is not None
    ]

    assert reusable == [0, 1, 2]

    # A truncated fourth result is not reusable, and does not affect
    # the three that are.
    (results / "cloud_worker_post-3.json").write_text(
        "{ truncated", encoding="utf-8"
    )

    assert (
        _reusable(results / "cloud_worker_post-3.json", "digest-3") is None
    )

    for index in range(3):
        assert (
            _reusable(
                results / f"cloud_worker_post-{index}.json",
                f"digest-{index}",
            )
            is not None
        )