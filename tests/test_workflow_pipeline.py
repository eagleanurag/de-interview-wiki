"""
Structural checks for the validation and deployment workflow.

These do not run the workflow. They assert the integration contract
between the YAML and the Python code, so a future edit that breaks
aggregation or publishing fails here rather than in a cloud run.

They used to assert something else. The workflow once fanned
enrichment out as one job per post; run 37026765769 showed that failing
six times out of twenty batches, every failure a provider response that
stopped partway, and the aggregation that would have used the other
fourteen never ran. Enrichment is local now and this file checks that
CI does not take it back.

So the contract asserted here is deliberately the opposite in one
respect and broader in the rest: no model, no fan-out, no credential --
and in exchange, checks the old workflow never had, namely that nothing
local reaches a published page, that two builds of the same input are
byte identical, and that no credential or session is tracked.
"""

from __future__ import annotations

from pathlib import Path

import pytest


yaml = pytest.importorskip("yaml")


REPO_ROOT = Path(__file__).resolve().parents[1]

WORKFLOW_PATH = (
    REPO_ROOT / ".github" / "workflows" / "run-python-worker.yml"
)

#: What the model is invoked through, wherever it is invoked from.
ENRICHMENT_CLIENT = "src.ai.opencode"


def workflow() -> dict:
    return yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))


def step_names(job: dict) -> list[str]:
    return [step.get("name", "") for step in job["steps"]]


def find_step(job: dict, name: str) -> dict:
    for step in job["steps"]:
        if step.get("name") == name:
            return step

    raise AssertionError(
        f"Step {name!r} not found in job. "
        f"Available steps: {step_names(job)}"
    )


def body_of(job: dict, name: str) -> str:
    return find_step(job, name).get("run", "") or ""


def all_run_bodies(document: dict) -> str:
    """Every shell script in the workflow, joined."""

    chunks: list[str] = []

    for job in document["jobs"].values():
        for step in job.get("steps", []):
            if step.get("run"):
                chunks.append(str(step["run"]))

    return "\n".join(chunks)


# ---------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------


def test_the_workflow_parses_and_declares_its_jobs():
    document = workflow()

    assert isinstance(document, dict)

    jobs = document["jobs"]

    assert set(jobs) == {"verify", "aggregate", "wiki", "deploy"}


def test_it_runs_on_a_push():
    """
    It can, now.

    It could not before: fanning a model out meant a push started
    hundreds of model calls, so the workflow was dispatch-only and a
    change could reach the default branch unvalidated. A deterministic
    pipeline is fast enough and cheap enough to gate every push, which
    is the point of taking the model out of it.
    """

    document = workflow()

    triggers = document.get("on", document.get(True))

    assert "push" in triggers
    assert "pull_request" in triggers


def test_every_job_has_a_timeout():
    """
    A hang should be reported.

    The old workflow's worker jobs had no bound at all, so a wedged
    model call held a runner for the six hours GitHub allows.
    """

    document = workflow()

    for name, job in document["jobs"].items():
        assert job.get("timeout-minutes"), name


# ---------------------------------------------------------------------
# CI does not call the model
# ---------------------------------------------------------------------


def test_ci_does_not_install_or_run_the_model():
    """
    The whole point of the change.

    Checked against the shell scripts as well as the step names,
    because the step could be renamed while still running the worker.
    """

    document = workflow()

    bodies = all_run_bodies(document)

    for banned in (
        "opencode",
        "npm install --global @opencode",
        "src.workers",
        "--model",
        "OPENCODE_EXECUTABLE",
        "api_key",
        "ANTHROPIC",
        "OPENAI",
    ):
        assert banned not in bodies, banned


def test_ci_never_installs_node():
    """
    Node existed only to install OpenCode.

    Left in place it would be a slow way to install nothing, and a
    reader would reasonably assume the workflow still needed it.
    """

    document = workflow()

    for job in document["jobs"].values():
        for step in job["steps"]:
            uses = str(step.get("uses", ""))

            assert "setup-node" not in uses, step.get("name")

    assert "actions/setup-node" not in all_run_bodies(document)


def test_no_job_fans_out_over_posts():
    """
    There is no matrix, because there is nothing to fan out.

    A matrix over posts is the shape the old architecture had, and it
    is also what would bring the 256-job limit back with it.
    """

    document = workflow()

    for name, job in document["jobs"].items():
        assert "strategy" not in job, name
        assert "matrix" not in job, name


def test_ci_does_not_enrich():
    document = workflow()

    bodies = all_run_bodies(document)

    for banned in (
        "src.pipeline",
        "--only enrich",
        "--force-enrich",
    ):
        assert banned not in bodies, banned


# ---------------------------------------------------------------------
# What CI does instead
# ---------------------------------------------------------------------


def test_ci_runs_the_tests_and_the_security_check():
    document = workflow()

    verify = document["jobs"]["verify"]

    assert "Run the test suite" in step_names(verify)

    assert "Verify no credential or session is tracked" in step_names(
        verify
    )


def test_the_secret_guard_names_the_real_paths():
    """
    The guard is only worth having if it names what would actually be
    committed. A pattern that matches nothing is a guard that passes
    forever.
    """

    document = workflow()

    body = body_of(
        document["jobs"]["verify"],
        "Verify no credential or session is tracked",
    )

    for pattern in (
        ".env",
        "chrome_session",
        "linkedin_saved_archive",
        "data/incoming/",
    ):
        assert pattern in body, pattern


def test_the_tracked_path_guard_cannot_report_a_clean_tree_wrongly():
    """
    The guard must be able to fail.

    It was written as ``printf ... | grep -q "$pattern"`` inside an
    ``if``. ``grep -q`` exits the moment it finds a match, which closes
    the pipe on ``printf``; under ``set -o pipefail`` the pipeline then
    reports printf's SIGPIPE status rather than grep's, so the condition
    is false and the guard passes on exactly the input it exists to
    catch.

    Whether it misfires depends on how far printf got before grep
    exited, so it passes on a handful of tracked paths and fails on a
    repository's worth. Measured against a 1,500-path listing -- close to
    this repository's -- a tracked ``.env`` was reported clean.

    Asserted structurally because the failure is a shell race that no
    unit test here can run, but the shape that causes it is knowable:
    a quiet grep at the end of a pipeline whose status is trusted.
    """

    document = workflow()

    body = body_of(
        document["jobs"]["verify"],
        "Verify no credential or session is tracked",
    )

    # Comments describe the shell and so may name the construct that is
    # forbidden; only the shell itself is judged.
    shell = "\n".join(
        line
        for line in body.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )

    assert "pipefail" in shell

    # The match is captured, and the capture is tested for emptiness.
    assert "match=$(printf" in shell
    assert '[ -n "$match" ]' in shell

    # The quiet form is what closes the pipe early.
    assert "grep -q" not in shell, (
        "grep -q exits on the first match and kills printf; capture the "
        "match instead"
    )


def test_the_tests_may_not_write_into_the_repository():
    document = workflow()

    body = body_of(
        document["jobs"]["verify"],
        "Confirm the working tree is unchanged by the tests",
    )

    assert "git status --porcelain" in body


def test_aggregation_reads_the_committed_results():
    """
    From the repository, not from artifacts.

    The committed results are what make CI deterministic: the same
    inputs produce the same knowledge base without a model call.
    """

    document = workflow()

    aggregate = document["jobs"]["aggregate"]

    body = body_of(aggregate, "Aggregate the committed results")

    assert "--input-dir data/results" in body
    assert "--expected-post-count" in body

    # And nothing is downloaded to aggregate.
    for step in aggregate["steps"]:
        assert "download-artifact" not in str(step.get("uses", ""))


def test_aggregation_fails_when_a_post_is_missing():
    """
    A partial knowledge base must not deploy.

    This is the check whose absence let run 37026765769 publish
    nothing at all while reporting only that some batches had failed.
    """

    document = workflow()

    body = body_of(
        document["jobs"]["aggregate"],
        "Verify the canonical knowledge base",
    )

    assert "posts_aggregated" in body
    assert "!= expected" in body
    assert "files_skipped" in body


def test_the_wiki_job_verifies_what_it_publishes():
    document = workflow()

    wiki = document["jobs"]["wiki"]

    names = step_names(wiki)

    assert "Verify the generated site" in names
    assert "Verify the site is reproducible" in names

    for page in (
        "site/index.html",
        "site/search.html",
        "site/topics.html",
        "site/questions.html",
        "site/assets/search-index.json",
    ):
        assert page in names or page in body_of(
            wiki, "Verify the generated site"
        ), page


def test_the_site_is_checked_for_local_paths():
    """
    A path that reaches a published page cannot be taken back.

    The old workflow checked that pages existed and nothing else.
    """

    document = workflow()

    body = body_of(
        document["jobs"]["wiki"], "Verify the generated site"
    )

    for pattern in ("C:\\\\Users", "AppData", "chrome_session"):
        assert pattern in body, pattern

    assert "--include='*.html'" in body


def test_two_builds_must_be_byte_identical():
    """
    A diff in the repository should mean something changed.

    Without this, a rebuild that differs only by a timestamp is
    indistinguishable from one that differs because the knowledge base
    did.
    """

    document = workflow()

    body = body_of(
        document["jobs"]["wiki"], "Verify the site is reproducible"
    )

    assert "diff -r" in body


def test_deployment_is_gated_on_the_default_branch():
    document = workflow()

    deploy = document["jobs"]["deploy"]

    condition = deploy.get("if", "")

    assert "pull_request" in condition
    assert "refs/heads/main" in condition


def test_deployment_has_only_the_permissions_pages_needs():
    """
    Nothing about the repository itself.

    The old workflow's worker jobs read the repository; the deploy job
    needs to write a Pages deployment and nothing else.
    """

    document = workflow()

    deploy = document["jobs"]["deploy"]

    assert deploy["permissions"] == {
        "pages": "write",
        "id-token": "write",
    }


def test_the_repository_is_never_written_to():
    """
    No job may push, commit or open a pull request.

    The only elevated permissions anywhere are the Pages and OIDC pair
    on the deploy job, which the official Pages deployment requires and
    which grants nothing about the repository itself.
    """

    document = workflow()

    bodies = all_run_bodies(document)

    for forbidden in (
        "git push",
        "git commit",
        "gh pr create",
        "git apply",
    ):
        assert forbidden not in bodies, forbidden


# ---------------------------------------------------------------------
# The local pipeline is the orchestrator
# ---------------------------------------------------------------------


def test_the_local_pipeline_still_exists_and_is_the_entry_point():
    """
    CI lost the enrichment; the local command has to be there to do it,
    or the two facts together leave nothing enriching anything.
    """

    from src.pipeline.cli import build_parser

    parser = build_parser()

    options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }

    assert "--jobs" in options
    assert "--attempts" in options
    assert "--force-enrich" in options
    assert "--only" in options


def test_the_worker_path_is_kept_but_no_longer_orchestrated():
    """
    The reusable per-post function stays.

    It is the same function the local orchestrator's retry loop is built
    on, so deleting it would have deleted the shared path rather than
    the GitHub-specific wrapper around it.
    """

    from src.workers.worker import process_enrichment_job

    assert callable(process_enrichment_job)

    # And it goes through the shared retry rather than one attempt.
    source = (
        REPO_ROOT / "src" / "workers" / "worker.py"
    ).read_text(encoding="utf-8")

    assert "Enricher" in source
    assert "attempts" in source


#: Workflows that legitimately install the model, and why.
#:
#: The remote control plane is a separate system: an issue is a command
#: channel and an agent acts on it in a runner. It is not the
#: enrichment orchestrator and it never was, and removing it is not what
#: this change is about.
MODEL_AWARE = {
    "opencode-agent.yml": (
        "the remote control plane drives an agent from an issue"
    ),
}


def test_only_the_control_plane_installs_the_model():
    """
    No workflow family quietly brings the model back.

    Checked across every workflow rather than only this one, because
    the model could be reintroduced through a new file that nothing
    else looks at. The cloud-worker smoke test that used to sit here
    existed to prove OpenCode worked in a runner; with enrichment local,
    nothing in CI calls the model, so it went.
    """

    offenders: list[str] = []

    for path in sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml")):
        if path.name in MODEL_AWARE:
            continue

        text = path.read_text(encoding="utf-8")

        for banned in (
            "@opencode/cli",
            "OPENCODE_EXECUTABLE",
            "--auto",
            "src.workers.cli",
        ):
            if banned in text:
                offenders.append(f"{path.name}: {banned}")

    assert offenders == []


def test_the_exceptions_are_named_and_still_exist():
    """
    An exception that names nothing is an exception nobody reads.

    Asserted rather than trusted, so a workflow being renamed cannot
    quietly become exempt.
    """

    for name in MODEL_AWARE:
        assert (
            REPO_ROOT / ".github" / "workflows" / name
        ).is_file(), name

    # And the control plane is not the enrichment path.
    text = (
        REPO_ROOT / ".github" / "workflows" / "opencode-agent.yml"
    ).read_text(encoding="utf-8")

    assert "src.workers.cli" not in text
    assert "--only enrich" not in text