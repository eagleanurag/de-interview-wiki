"""
Structural checks for the cloud worker workflow.

These tests do not run the workflow. They assert the integration
contract between the YAML pipeline and the Python code, so a future
edit that breaks aggregation fails here instead of in a cloud run.
"""

from __future__ import annotations

from pathlib import Path

import pytest


yaml = pytest.importorskip("yaml")


WORKFLOW_PATH = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "workflows"
    / "run-python-worker.yml"
)


@pytest.fixture(scope="module")
def workflow() -> dict:
    return yaml.safe_load(
        WORKFLOW_PATH.read_text(encoding="utf-8")
    )


def step_names(job: dict) -> list[str]:
    return [step.get("name", "") for step in job["steps"]]


def upload_paths(step: dict) -> list[str]:
    """
    Split a multiline upload path.

    Line based, because GitHub expressions such as
    ${{ matrix.post_id }} contain spaces.
    """

    return [
        line.strip()
        for line in step["with"]["path"].strip().splitlines()
        if line.strip()
    ]


def find_step(job: dict, name: str) -> dict:
    for step in job["steps"]:
        if step.get("name") == name:
            return step

    raise AssertionError(
        f"Step {name!r} not found in job {job.get('name')!r}. "
        f"Available steps: {step_names(job)}"
    )


def test_workflow_parses_and_declares_all_jobs(workflow):
    assert "jobs" in workflow

    assert set(workflow["jobs"]) == {
        "discover",
        "worker",
        "aggregate",
    }


def test_discover_stays_dynamic(workflow):
    discover = workflow["jobs"]["discover"]

    assert set(discover["outputs"]) == {
        "matrix",
        "post_count",
    }

    discover_step = find_step(discover, "Discover post directories")

    assert 'glob("*/post.json")' in discover_step["run"]
    assert "post_count=" in discover_step["run"]


def test_worker_stays_parallel_and_fails_independently(workflow):
    worker = workflow["jobs"]["worker"]

    assert worker["needs"] == "discover"
    assert worker["strategy"]["fail-fast"] is False
    assert worker["strategy"]["max-parallel"] == 3
    assert "fromJSON" in str(worker["strategy"]["matrix"])


def test_worker_keeps_unique_per_post_artifacts(workflow):
    worker = workflow["jobs"]["worker"]

    upload = find_step(worker, "Upload worker result")
    assert upload["uses"] == "actions/upload-artifact@v4"

    assert (
        upload["with"]["name"]
        == "cloud-worker-result-${{ matrix.post_id }}"
    )
    assert upload["with"]["if-no-files-found"] == "error"

    assert upload_paths(upload) == [
        "data/jobs/"
        "cloud_worker_${{ matrix.post_id }}_"
        "${{ github.run_id }}.json",
        "data/results/"
        "cloud_worker_${{ matrix.post_id }}_"
        "${{ github.run_id }}.json",
    ]


def test_aggregate_runs_only_after_all_workers_succeed(workflow):
    aggregate = workflow["jobs"]["aggregate"]

    assert aggregate["needs"] == ["discover", "worker"]

    assert "if" not in aggregate
    assert "always()" not in str(aggregate.get("needs"))


def test_aggregate_downloads_all_worker_artifacts(workflow):
    aggregate = workflow["jobs"]["aggregate"]

    download = find_step(aggregate, "Download worker artifacts")

    assert download["uses"] == "actions/download-artifact@v4"
    assert (
        download["with"]["pattern"] == "cloud-worker-result-*"
    )
    assert (
        download["with"]["path"]
        == "aggregation/worker-results"
    )


def test_aggregate_downloads_worker_artifacts_without_flattening(
    workflow,
):
    """
    Each worker uploads its job manifest and its enriched result
    under data/jobs and data/results using the same file name.
    Merging every artifact into one directory would collapse those
    two paths onto one name and silently drop a file.
    """

    download = find_step(
        workflow["jobs"]["aggregate"],
        "Download worker artifacts",
    )

    assert download["with"].get("merge-multiple") is not True, (
        "Download worker artifacts must not set merge-multiple: "
        "true, because the per-post directory layout is what keeps "
        "the job manifest and the enriched result separate"
    )


def test_aggregate_runs_the_python_aggregator(workflow):
    aggregate = workflow["jobs"]["aggregate"]

    run_step = find_step(aggregate, "Run aggregation")

    assert "python -m src.aggregation.aggregator" in run_step["run"]
    assert (
        "--input-dir aggregation/worker-results"
        in run_step["run"]
    )
    assert (
        "--output aggregation/knowledge_base.json"
        in run_step["run"]
    )
    assert "--expected-post-count" in run_step["run"]

    assert (
        run_step["env"]["EXPECTED_POST_COUNT"]
        == "${{ needs.discover.outputs.post_count }}"
    )


def test_run_steps_define_the_env_vars_they_read(workflow):
    """
    A `run` script can only read an environment variable that its own
    step exports. Referencing EXPECTED_POST_COUNT from a step that
    does not define it raises KeyError at run time, so the step fails
    even though the knowledge base itself is valid.

    This is a generic invariant: it scans every job and every step
    instead of naming a step, so it also covers future steps.
    """

    referencing_steps = []

    for job_name, job in workflow["jobs"].items():
        for step in job["steps"]:
            if "EXPECTED_POST_COUNT" not in step.get(
                "run", ""
            ):
                continue

            referencing_steps.append(
                f"{job_name} / {step.get('name', '')}"
            )

            assert (
                "EXPECTED_POST_COUNT"
                in step.get("env", {})
            ), (
                f"{job_name} / {step.get('name', '')!r} reads "
                f"EXPECTED_POST_COUNT but does not define it in "
                f"that step's own env"
            )

    assert referencing_steps, (
        "No workflow step references EXPECTED_POST_COUNT, so this "
        "invariant is no longer being exercised"
    )


def test_aggregate_uploads_one_canonical_knowledge_base(workflow):
    aggregate = workflow["jobs"]["aggregate"]

    upload = find_step(aggregate, "Upload canonical knowledge base")

    assert upload["uses"] == "actions/upload-artifact@v4"
    assert (
        upload["with"]["path"]
        == "aggregation/knowledge_base.json"
    )
    assert upload["with"]["if-no-files-found"] == "error"

    upload_steps = [
        step
        for step in aggregate["steps"]
        if step.get("uses")
        == "actions/upload-artifact@v4"
    ]

    assert len(upload_steps) == 1


def test_worker_artifact_filenames_match_aggregator_glob(workflow):
    """
    The aggregator globs cloud_worker_*.json, so worker artifact
    file names must keep that prefix.
    """

    from src.aggregation.aggregator import (
        WORKER_RESULT_GLOB,
    )

    assert WORKER_RESULT_GLOB == "cloud_worker_*.json"

    worker = workflow["jobs"]["worker"]

    paths = upload_paths(
        find_step(worker, "Upload worker result")
    )

    assert len(paths) == 2

    for path in paths:
        assert "cloud_worker_" in path
        assert path.endswith(".json")
        assert "${{ matrix.post_id }}" in path
        assert "${{ github.run_id }}" in path


def test_workflow_never_pushes_or_commits(workflow):
    forbidden = (
        "git push",
        "git commit",
        "gh pr create",
        "curl -X POST",
    )

    for job_name, job in workflow["jobs"].items():
        assert job.get("permissions") == {
            "contents": "read"
        }, f"{job_name} should be read-only"

        for step in job["steps"]:
            script = step.get("run", "")

            for token in forbidden:
                assert token not in script, (
                    f"{job_name} / {step.get('name')} "
                    f"must not use {token!r}"
                )


def test_aggregate_ignores_job_manifests_in_downloaded_layout(
    workflow,
    tmp_path,
):
    """
    End-to-end check that the artifact layout produced by the
    download-artifact step is correctly understood by the aggregator.
    """

    from src.aggregation.aggregator import aggregate_results

    download = find_step(
        workflow["jobs"]["aggregate"],
        "Download worker artifacts",
    )

    root = tmp_path / download["with"]["path"]

    for post_id in ("sample_001", "sample_002"):
        artifact = root / f"cloud-worker-result-{post_id}"
        job_id = f"cloud_worker_{post_id}_4242"

        for kind in ("jobs", "results"):
            path = artifact / "data" / kind / f"{job_id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")

        (artifact / "data" / "results" / f"{job_id}.json").write_text(
            _post_json(post_id),
            encoding="utf-8",
        )

        (artifact / "data" / "jobs" / f"{job_id}.json").write_text(
            _manifest_json(post_id, job_id),
            encoding="utf-8",
        )

    output = tmp_path / "knowledge_base.json"

    aggregate_results(
        root,
        output,
        expected_post_count=2,
    )

    import json

    payload = json.loads(
        output.read_text(encoding="utf-8")
    )

    assert payload["stats"]["posts_aggregated"] == 2
    assert payload["stats"]["files_skipped"] == 2
    assert [post["id"] for post in payload["posts"]] == [
        "sample_001",
        "sample_002",
    ]


def _post_json(post_id: str) -> str:
    import json

    return json.dumps(
        {
            "id": post_id,
            "source": {
                "platform": "linkedin",
                "url": f"https://example.com/{post_id}",
                "captured_at": "2026-09-29T18:30:00+05:30",
            },
            "original_text": "text",
            "media": [],
            "ai_analysis": {"summary": "summary"},
            "interview_questions": [],
            "classification": {
                "domain": "Data Engineering",
                "interview_relevant": True,
            },
        },
        indent=2,
    )


def _manifest_json(post_id: str, job_id: str) -> str:
    import json

    return json.dumps(
        {
            "job_id": job_id,
            "post_directory": f"data/posts/{post_id}",
            "output_path": f"data/results/{job_id}.json",
            "status": "completed",
            "attempt": 1,
            "max_attempts": 3,
        },
        indent=2,
    )
