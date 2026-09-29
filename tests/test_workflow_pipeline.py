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
        "wiki",
        "deploy",
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
    """
    No job may push, commit, open a pull request, or call the API
    directly.

    Repository contents stay read-only everywhere. The only elevated
    permissions anywhere in the workflow are the Pages and OIDC pair
    on the deploy job, which the official Pages deployment requires
    and which grants nothing about the repository itself.
    """

    forbidden = (
        "git push",
        "git commit",
        "gh pr create",
        "gh release",
        "curl -X POST",
    )

    for job_name, job in workflow["jobs"].items():
        permissions = job.get("permissions", {})

        assert permissions.get("contents") in (None, "read"), (
            f"{job_name} must not have repository write access"
        )

        if job_name == "deploy":
            assert permissions == {
                "pages": "write",
                "id-token": "write",
            }
        else:
            assert permissions == {
                "contents": "read"
            }, f"{job_name} should be read-only"

        for step in job["steps"]:
            script = step.get("run", "")

            for token in forbidden:
                assert token not in script, (
                    f"{job_name} / {step.get('name')} "
                    f"must not use {token!r}"
                )


def test_pages_deployment_uses_the_official_actions(workflow):
    deploy = workflow["jobs"]["deploy"]

    assert deploy["runs-on"] == "ubuntu-latest"
    assert deploy["needs"] == ["wiki"]

    uses = [step.get("uses") for step in deploy["steps"]]

    assert uses == ["actions/deploy-pages@v4"]
    assert deploy["environment"]["name"] == "github-pages"
    assert (
        deploy["environment"]["url"]
        == "${{ steps.deployment.outputs.page_url }}"
    )

    upload = find_step(
        workflow["jobs"]["wiki"], "Upload Pages artifact"
    )

    assert upload["uses"] == "actions/upload-pages-artifact@v3"
    assert upload["with"]["path"] == "site"


def test_wiki_generation_happens_only_after_aggregation(workflow):
    """
    The wiki must never be built from a partial knowledge base.

    `wiki` depends on `aggregate` with no `if:` override, so the
    default success() applies: if aggregation fails, or if any worker
    failed and aggregation was therefore skipped, the wiki job is
    skipped too.
    """

    aggregate = workflow["jobs"]["aggregate"]
    wiki = workflow["jobs"]["wiki"]
    deploy = workflow["jobs"]["deploy"]

    assert wiki["needs"] == ["aggregate", "discover"]

    # No escape hatch that could run on failure.
    assert "if" not in wiki
    assert "always()" not in str(wiki.get("needs"))
    assert "always()" not in str(aggregate.get("needs"))
    assert "always()" not in str(deploy.get("needs"))

    # And the deploy waits on wiki.
    assert deploy["needs"] == ["wiki"]
    assert "if" not in deploy


def test_wiki_generation_consumes_the_canonical_artifact(
    workflow,
):
    """
    The wiki must read the knowledge base the aggregate job uploaded,
    not something reconstructed locally.
    """

    wiki = workflow["jobs"]["wiki"]

    download = find_step(wiki, "Download canonical knowledge base")

    assert download["uses"] == "actions/download-artifact@v4"
    assert (
        download["with"]["name"]
        == "knowledge-base-${{ github.run_id }}"
    )
    assert download["with"]["path"] == "aggregation"

    generate = find_step(wiki, "Generate static site")

    assert "python -m src.wiki.generator" in generate["run"]
    assert (
        "--input aggregation/knowledge_base.json"
        in generate["run"]
    )
    assert "--output site" in generate["run"]

    # The artifact name must match what the aggregate job uploads.
    aggregate = workflow["jobs"]["aggregate"]
    upload = find_step(
        aggregate, "Upload canonical knowledge base"
    )

    assert (
        upload["with"]["name"] == download["with"]["name"]
    )


def test_wiki_verification_blocks_partial_publication(workflow):
    """
    Publication must fail closed. The wiki job checks that the
    generated site exists and that its post count matches the
    discovered post count before the Pages artifact is uploaded.
    """

    wiki = workflow["jobs"]["wiki"]

    steps = [step.get("name") for step in wiki["steps"]]

    generate_index = steps.index("Generate static site")
    verify_index = steps.index("Verify generated site")
    upload_index = steps.index("Upload Pages artifact")

    assert generate_index < verify_index < upload_index

    verify = find_step(wiki, "Verify generated site")

    for required in (
        "site/index.html",
        "site/search.html",
        "site/topics.html",
        "site/questions.html",
        "site/assets/search-index.json",
    ):
        assert f"test -s {required}" in verify["run"]

    # The error text is wrapped across source lines, so compare on
    # whitespace-normalised text.
    # The messages are wrapped across YAML/Python source lines, so
    # match on stable fragments rather than whole sentences.
    normalized = " ".join(verify["run"].split())

    assert "Refusing to publish a" in normalized
    assert "partial knowledge base" in normalized

    # The count check must read the discovery output, and the verify
    # step must define the variable it reads.
    assert (
        verify["env"]["EXPECTED_POST_COUNT"]
        == "${{ needs.discover.outputs.post_count }}"
    )
    assert find_step(wiki, "Generate static site")["env"][
        "EXPECTED_POST_COUNT"
    ] == "${{ needs.discover.outputs.post_count }}"

    # The Pages upload must fail rather than deploy an empty site.
    upload = find_step(wiki, "Upload Pages artifact")

    assert "if-no-files-found" not in upload["with"]


def test_aggregation_still_fails_closed_on_post_count(workflow):
    """
    Adding the wiki must not weaken the existing aggregation guard.
    """

    aggregate = workflow["jobs"]["aggregate"]

    run_step = find_step(aggregate, "Run aggregation")

    assert "--expected-post-count" in run_step["run"]

    verify = find_step(aggregate, "Verify canonical knowledge base")

    normalized = " ".join(verify["run"].split())

    assert "Canonical knowledge base post count does" in normalized
    assert "not match discovered post count" in normalized
    assert "posts_aggregated" in verify["run"]


def test_worker_job_is_unchanged_by_the_wiki(workflow):
    """
    The wiki is additive. Worker discovery, parallelism, isolation
    and artifact naming must be exactly as before.
    """

    discover = workflow["jobs"]["discover"]
    worker = workflow["jobs"]["worker"]

    assert set(discover["outputs"]) == {"matrix", "post_count"}
    assert 'glob("*/post.json")' in find_step(
        discover, "Discover post directories"
    )["run"]

    assert worker["needs"] == "discover"
    assert worker["strategy"]["fail-fast"] is False
    assert worker["strategy"]["max-parallel"] == 3

    upload = find_step(worker, "Upload worker result")

    assert (
        upload["with"]["name"]
        == "cloud-worker-result-${{ matrix.post_id }}"
    )
    assert upload["with"]["if-no-files-found"] == "error"

    assert len(upload_paths(upload)) == 2


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
