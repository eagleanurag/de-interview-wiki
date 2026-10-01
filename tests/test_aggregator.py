from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.aggregation.aggregator import (
    AggregationError,
    aggregate_results,
)


RUN_ID = "4242"

POST_IDS = ["sample_003", "sample_001", "sample_002"]


def make_post(
    post_id: str,
    *,
    summary: str | None = None,
) -> dict:
    """Build a valid enriched KnowledgePost payload."""

    return {
        "id": post_id,
        "source": {
            "platform": "linkedin",
            "url": f"https://example.com/{post_id}",
            "captured_at": "2026-09-29T18:30:00+05:30",
            "author": "Sample Author",
            "published_at": "Jan 15, 2025",
        },
        "original_text": f"Original text for {post_id}.",
        "media": [],
        "ai_analysis": {
            "summary": summary or f"Summary for {post_id}.",
            "topics": ["Databricks"],
            "subtopics": ["Partitioning"],
            "concepts": ["Partition Pruning"],
            "image_descriptions": [],
        },
        "interview_questions": [
            {
                "question": f"Explain {post_id} partitioning.",
                "type": "scenario",
                "difficulty": "medium",
                "answer": "Measure first.",
            }
        ],
        "classification": {
            "domain": "Data Engineering",
            "primary_topic": "Databricks",
            "secondary_topics": ["Apache Spark"],
            "interview_relevant": True,
        },
    }


def make_job_manifest(
    post_id: str,
    *,
    run_id: str = RUN_ID,
    status: str = "completed",
) -> dict:
    """Build a worker job manifest exactly like the workflow does."""

    job_id = f"cloud_worker_{post_id}_{run_id}"

    return {
        "job_id": job_id,
        "post_directory": f"data/posts/{post_id}",
        "output_path": f"data/results/{job_id}.json",
        "status": status,
        "attempt": 1,
        "max_attempts": 3,
        "created_at": "2026-09-29T13:41:56.180416Z",
        "started_at": "2026-09-29T13:41:56.180516Z",
        "completed_at": "2026-09-29T13:43:15.987723Z",
        "worker_id": "runner-worker",
        "error": None,
    }


def write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def write_worker_result(
    root: Path,
    post_id: str,
    *,
    artifact_name: str | None = None,
    run_id: str = RUN_ID,
    with_manifest: bool = True,
) -> Path:
    """
    Recreate the artifact layout produced by one matrix worker.

    The workflow uploads both the job manifest and the enriched result
    under data/jobs and data/results inside a per-post artifact.
    """

    artifact_directory = (
        root / (artifact_name or f"cloud-worker-result-{post_id}")
    )

    job_id = f"cloud_worker_{post_id}_{run_id}"

    if with_manifest:
        write_json(
            artifact_directory / "data" / "jobs" / f"{job_id}.json",
            make_job_manifest(post_id, run_id=run_id),
        )

    return write_json(
        artifact_directory / "data" / "results" / f"{job_id}.json",
        make_post(post_id),
    )


def write_worker_inputs(
    root: Path,
    post_ids: list[str] | None = None,
    *,
    with_manifests: bool = True,
) -> Path:
    """Populate a directory that looks like downloaded artifacts."""

    input_dir = root / "worker-results"
    input_dir.mkdir(parents=True, exist_ok=True)

    for post_id in post_ids or POST_IDS:
        write_worker_result(
            input_dir,
            post_id,
            with_manifest=with_manifests,
        )

    return input_dir


def read_knowledge_base(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_aggregates_multiple_worker_results(tmp_path):
    input_dir = write_worker_inputs(tmp_path)
    output_path = tmp_path / "out" / "knowledge_base.json"

    result = aggregate_results(input_dir, output_path)

    assert result == output_path
    assert output_path.exists()

    payload = read_knowledge_base(output_path)

    assert payload["schema_version"] == 1
    assert payload["posts"]
    assert payload["stats"]["posts_aggregated"] == len(POST_IDS)

    for post_id in POST_IDS:
        aggregated = next(
            post
            for post in payload["posts"]
            if post["id"] == post_id
        )
        expected = make_post(post_id)
        assert aggregated["source"] == expected["source"]
        assert (
            aggregated["ai_analysis"]["summary"]
            == expected["ai_analysis"]["summary"]
        )
        assert (
            len(aggregated["interview_questions"])
            == len(expected["interview_questions"])
        )
        assert aggregated["classification"] == (
            expected["classification"]
        )


def test_skips_worker_job_manifests(tmp_path):
    input_dir = tmp_path / "worker-results"
    input_dir.mkdir()

    write_worker_result(input_dir, "sample_001")
    write_worker_result(input_dir, "sample_002")

    output_path = tmp_path / "knowledge_base.json"

    aggregate_results(input_dir, output_path)

    payload = read_knowledge_base(output_path)

    assert payload["stats"]["posts_aggregated"] == 2
    assert payload["stats"]["files_skipped"] == 2

    post_ids = [post["id"] for post in payload["posts"]]
    assert sorted(post_ids) == ["sample_001", "sample_002"]

    skipped = "\n".join(payload["skipped_files"])
    assert "worker job manifest" in skipped
    assert "data/jobs/" in skipped

    for post in payload["posts"]:
        assert "job_id" not in post
        assert "output_path" not in post


def test_manifest_only_directory_produces_no_manifest_entries(
    tmp_path,
):
    input_dir = tmp_path / "worker-results"
    input_dir.mkdir()

    for post_id in POST_IDS:
        write_json(
            input_dir / f"cloud_worker_{post_id}_{RUN_ID}.json",
            make_job_manifest(post_id),
        )

    output_path = tmp_path / "knowledge_base.json"

    aggregate_results(
        input_dir,
        output_path,
        expected_post_count=0,
    )

    payload = read_knowledge_base(output_path)

    assert payload["posts"] == []
    assert payload["stats"]["posts_aggregated"] == 0
    assert payload["stats"]["files_skipped"] == len(POST_IDS)


def test_detects_duplicate_post_ids(tmp_path):
    input_dir = tmp_path / "worker-results"
    input_dir.mkdir()

    write_worker_result(
        input_dir,
        "sample_001",
        artifact_name="cloud-worker-result-sample_001",
        run_id="1000",
    )
    write_worker_result(
        input_dir,
        "sample_001",
        artifact_name="cloud-worker-result-sample_001-rerun",
        run_id="2000",
    )

    output_path = tmp_path / "knowledge_base.json"

    with pytest.raises(AggregationError) as error:
        aggregate_results(input_dir, output_path)

    assert "Duplicate post ID" in str(error.value)
    assert "sample_001" in str(error.value)
    assert not output_path.exists()


def test_rejects_invalid_knowledge_post(tmp_path):
    input_dir = tmp_path / "worker-results"
    input_dir.mkdir()

    write_worker_result(input_dir, "sample_001")

    broken = make_post("sample_002")
    broken["source"]["captured_at"] = "not-a-timestamp"

    write_json(
        input_dir
        / "cloud-worker-result-sample_002"
        / "data"
        / "results"
        / f"cloud_worker_sample_002_{RUN_ID}.json",
        broken,
    )

    output_path = tmp_path / "knowledge_base.json"

    with pytest.raises(AggregationError) as error:
        aggregate_results(input_dir, output_path)

    assert "Invalid KnowledgePost" in str(error.value)
    assert "sample_002" in str(error.value)
    assert not output_path.exists()


def test_rejects_unreadable_result_file(tmp_path):
    input_dir = tmp_path / "worker-results"
    input_dir.mkdir()

    write_worker_result(input_dir, "sample_001")

    broken_file = (
        input_dir
        / "cloud-worker-result-sample_002"
        / "data"
        / "results"
        / f"cloud_worker_sample_002_{RUN_ID}.json"
    )
    broken_file.parent.mkdir(parents=True, exist_ok=True)
    broken_file.write_text("{not json", encoding="utf-8")

    output_path = tmp_path / "knowledge_base.json"

    with pytest.raises(AggregationError) as error:
        aggregate_results(input_dir, output_path)

    assert "Could not read result file" in str(error.value)
    assert not output_path.exists()


def test_skips_json_arrays_and_partial_objects(tmp_path):
    input_dir = tmp_path / "worker-results"
    input_dir.mkdir()

    write_worker_result(input_dir, "sample_001")

    write_json(
        input_dir / "cloud_worker_note_1.json",
        ["not", "a", "post"],
    )
    write_json(
        input_dir / "cloud_worker_draft_1.json",
        {"id": "draft_1", "source": {}},
    )

    output_path = tmp_path / "knowledge_base.json"

    aggregate_results(input_dir, output_path)

    payload = read_knowledge_base(output_path)

    assert payload["stats"]["posts_aggregated"] == 1
    assert payload["stats"]["files_skipped"] == 3

    skipped = "\n".join(payload["skipped_files"])
    assert "not a JSON object" in skipped
    assert "missing enriched KnowledgePost fields" in skipped


def test_post_ordering_is_deterministic(tmp_path):
    payload_a = _aggregate_and_read(
        tmp_path / "case-a", POST_IDS
    )
    payload_b = _aggregate_and_read(
        tmp_path / "case-b", list(reversed(POST_IDS))
    )

    order_a = [post["id"] for post in payload_a["posts"]]
    order_b = [post["id"] for post in payload_b["posts"]]

    assert order_a == order_b
    assert order_a == sorted(POST_IDS)


def test_payload_is_stable_across_runs(tmp_path):
    first = _aggregate_and_read(tmp_path / "first", POST_IDS)
    second = _aggregate_and_read(tmp_path / "second", POST_IDS)

    first.pop("generated_at")
    second.pop("generated_at")

    assert first == second


def test_expected_post_count_detects_missing_result(tmp_path):
    input_dir = write_worker_inputs(tmp_path, ["sample_001"])
    output_path = tmp_path / "knowledge_base.json"

    with pytest.raises(AggregationError) as error:
        aggregate_results(
            input_dir,
            output_path,
            expected_post_count=3,
        )

    assert "Expected 3 aggregated post(s)" in str(error.value)
    assert not output_path.exists()

    aggregate_results(
        input_dir,
        output_path,
        expected_post_count=1,
    )

    assert read_knowledge_base(output_path)["posts"]


def test_missing_input_directory_raises(tmp_path):
    with pytest.raises(AggregationError) as error:
        aggregate_results(
            tmp_path / "absent",
            tmp_path / "knowledge_base.json",
        )

    assert "Input directory does not exist" in str(error.value)


def test_input_directory_without_worker_files_raises(tmp_path):
    input_dir = tmp_path / "worker-results"
    input_dir.mkdir()

    write_json(
        input_dir / "knowledge_base.json",
        {"posts": []},
    )

    with pytest.raises(AggregationError) as error:
        aggregate_results(
            input_dir,
            tmp_path / "knowledge_base.json",
        )

    assert "No worker JSON files" in str(error.value)


def test_temporary_file_is_removed_after_write(tmp_path):
    input_dir = write_worker_inputs(tmp_path, ["sample_001"])
    output_path = tmp_path / "knowledge_base.json"

    aggregate_results(input_dir, output_path)

    assert not output_path.with_suffix(".json.tmp").exists()


def test_utf8_bom_result_file_is_accepted(tmp_path):
    input_dir = tmp_path / "worker-results"
    input_dir.mkdir()

    target = (
        input_dir
        / "cloud-worker-result-sample_001"
        / "data"
        / "results"
        / f"cloud_worker_sample_001_{RUN_ID}.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(make_post("sample_001"), indent=2),
        encoding="utf-8-sig",
    )

    output_path = tmp_path / "knowledge_base.json"

    aggregate_results(input_dir, output_path)

    payload = read_knowledge_base(output_path)

    assert payload["stats"]["posts_aggregated"] == 1


def _aggregate_and_read(
    root: Path,
    post_ids: list[str],
) -> dict:
    input_dir = write_worker_inputs(root, post_ids)
    output_path = root / "knowledge_base.json"

    aggregate_results(input_dir, output_path)

    return read_knowledge_base(output_path)
