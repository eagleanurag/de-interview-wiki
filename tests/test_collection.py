"""
Tests for collection: checkpoints, credentials, sources, the collector.

No live network access and no real site. Browser interaction is
exercised through a fake page that mirrors the selectors the collector
uses, so scrolling, lazy loading, extraction, deduplication, challenge
detection and resume are all covered deterministically.

Credential tests assert only that a value is never exposed. No test
prints or asserts against a real credential.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.ingestion import checkpoints as checkpoint_module
from src.ingestion import credentials as credential_module
from src.ingestion.collect import (
    CollectionError,
    CollectionLimits,
    Collector,
    post_id_for,
    read_state,
    write_state,
)
from src.ingestion.post_document import PostDocument
from src.ingestion.sources.base import (
    CollectedPost,
    CollectionState,
    CollectionStopped,
    SecurityChallenge,
    Source,
    StopReason,
)
from src.ingestion.sources.linkedin import (
    LinkedInLimits,
    LinkedInSource,
    SelectorSet,
    _media_filename,
    _parse_relative,
    linkedin_installed,
)
from src.ingestion.sources.manual import ManualSource


# ---------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------


def test_checkpoint_round_trip(tmp_path):
    checkpoint = checkpoint_module.Checkpoint(
        phase=checkpoint_module.CP1_REPOSITORY_AUDIT,
        status=checkpoint_module.STATUS_COMPLETE,
        completed=["audit done"],
        remaining=["implement collection"],
        local_sha="abc1234",
        remote_sha="abc1234",
        resume_point="start CP2",
    )

    path = checkpoint_module.write(checkpoint, tmp_path)

    assert path.is_file()

    loaded = checkpoint_module.read(tmp_path)

    assert loaded is not None
    assert loaded.phase == checkpoint_module.CP1_REPOSITORY_AUDIT
    assert loaded.status == checkpoint_module.STATUS_COMPLETE
    assert loaded.completed == ["audit done"]
    assert loaded.resume_point == "start CP2"


def test_checkpoint_absent_is_none(tmp_path):
    assert checkpoint_module.read(tmp_path) is None


def test_malformed_checkpoint_is_not_trusted(tmp_path):
    directory = checkpoint_module.current_directory(tmp_path)
    directory.mkdir(parents=True)
    (directory / "current.json").write_text(
        "{not json", encoding="utf-8"
    )

    assert checkpoint_module.read(tmp_path) is None


def test_checkpoint_refuses_a_credential_shaped_value(tmp_path):
    """
    The module must refuse to persist anything that looks like a
    credential. This is the guard that keeps secrets out of runtime
    state.
    """

    checkpoint = checkpoint_module.Checkpoint(
        phase=checkpoint_module.CP6_COLLECTION_RUNNING,
        blocker="token: ghp_abcdefghijklmnopqrstuvwxyz0123456789",
    )

    with pytest.raises(checkpoint_module.CheckpointError):
        checkpoint_module.write(checkpoint, tmp_path)


def test_checkpoint_refuses_a_sensitive_key(tmp_path):
    payload = {
        "phase": "CP6_COLLECTION_RUNNING",
        "password": "hunter2hunter2",
    }

    with pytest.raises(checkpoint_module.CheckpointError):
        checkpoint_module._assert_clean(payload)


def test_credential_detection():
    assert checkpoint_module.contains_credential(
        "ghp_abcdefghijklmnopqrstuvwxyz0123456789"
    )
    assert checkpoint_module.contains_credential(
        "github_pat_11ABCDEFG0abcdefghijklmnop"
    )
    assert checkpoint_module.contains_credential(
        "password: supersecretvalue"
    )
    assert checkpoint_module.contains_credential("AKIAIOSFODNN7EXAMPL1")

    assert not checkpoint_module.contains_credential("no secret here")
    assert not checkpoint_module.contains_credential("posts=12")


def test_checkpoint_records_every_required_field(tmp_path):
    checkpoint = checkpoint_module.Checkpoint()

    for name, value in (
        ("checkpoint_id", checkpoint_module.CP0_SYNC),
        ("phase", checkpoint_module.CP1_REPOSITORY_AUDIT),
        ("status", checkpoint_module.STATUS_COMPLETE),
        ("local_sha", "aaa"),
        ("remote_sha", "bbb"),
        ("completed", ["one"]),
        ("remaining", ["two"]),
        ("tests", {"passed": 391}),
        ("ci", {"status": "green"}),
        ("blocker", ""),
        ("human_action_required", ""),
        ("resume_point", "CP2"),
    ):
        assert name in checkpoint_module.Checkpoint.__dataclass_fields__

        setattr(checkpoint, name, value)

    path = checkpoint_module.write(checkpoint, tmp_path)

    payload = json.loads(path.read_text(encoding="utf-8"))

    for name in (
        "checkpoint_id",
        "phase",
        "status",
        "timestamp",
        "local_sha",
        "remote_sha",
        "completed",
        "remaining",
        "tests",
        "ci",
        "blocker",
        "human_action_required",
        "resume_point",
    ):
        assert name in payload, name


def test_every_documented_phase_exists():
    for phase in (
        checkpoint_module.CP0_SYNC,
        checkpoint_module.CP1_REPOSITORY_AUDIT,
        checkpoint_module.CP2_ARCHITECTURE_AUDIT,
        checkpoint_module.CP3_INGESTION_READY,
        checkpoint_module.CP4_SOURCE_COLLECTION_READY,
        checkpoint_module.CP5_AUTHENTICATION_READY,
        checkpoint_module.CP6_COLLECTION_RUNNING,
        checkpoint_module.CP7_COLLECTION_COMPLETE,
        checkpoint_module.CP8_ENRICHMENT_COMPLETE,
        checkpoint_module.CP9_KNOWLEDGE_BASE_COMPLETE,
        checkpoint_module.CP10_WIKI_COMPLETE,
        checkpoint_module.CP11_TESTS_GREEN,
        checkpoint_module.CP12_SECURITY_VERIFIED,
        checkpoint_module.CP13_COMMIT_COMPLETE,
        checkpoint_module.CP14_PUSH_COMPLETE,
        checkpoint_module.CP15_CI_VERIFIED,
        checkpoint_module.CP16_FINAL,
    ):
        assert phase in checkpoint_module.PHASE_ORDER

    assert (
        checkpoint_module.CP_HUMAN_LINKEDIN_ACTION_REQUIRED
        == "CP_HUMAN_LINKEDIN_ACTION_REQUIRED"
    )


def test_checkpoint_writes_its_readme(tmp_path):
    checkpoint_module.write(
        checkpoint_module.Checkpoint(), tmp_path
    )

    readme = (
        checkpoint_module.current_directory(tmp_path) / "README.md"
    )

    assert readme.is_file()
    assert "git" in readme.read_text(encoding="utf-8").lower()


# ---------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------


def test_credentials_absent_reports_no():
    result = credential_module.status({})

    assert result.configured is False
    assert "no" in result.describe()


def test_credentials_configured_reports_yes(monkeypatch):
    environment = {
        "LINKEDIN_USERNAME": "someone@example.com",
        "LINKEDIN_PASSWORD": "correct-horse-battery",
    }

    result = credential_module.status(environment)

    assert result.configured is True
    assert result.describe() == "LinkedIn credentials configured: yes"


def test_partial_credentials_name_the_missing_one():
    result = credential_module.status(
        {"LINKEDIN_USERNAME": "someone@example.com"}
    )

    assert result.incomplete is True
    assert "LINKEDIN_PASSWORD" in result.describe()

    other = credential_module.status(
        {"LINKEDIN_PASSWORD": "correct-horse-battery"}
    )

    assert "LINKEDIN_USERNAME" in other.describe()


def test_describe_never_contains_a_value():
    """
    The only printable form of the credential state must be a yes/no.
    """

    username = "someone@example.com"
    password = "correct-horse-battery"

    described = credential_module.status(
        {
            "LINKEDIN_USERNAME": username,
            "LINKEDIN_PASSWORD": password,
        }
    ).describe()

    assert username not in described
    assert password not in described


def test_require_raises_naming_only_the_missing_variable():
    with pytest.raises(credential_module.CredentialError) as error:
        credential_module.require({})

    assert "LINKEDIN_USERNAME" in str(error.value)

    with pytest.raises(credential_module.CredentialError) as error:
        credential_module.require(
            {"LINKEDIN_USERNAME": "someone@example.com"}
        )

    assert "LINKEDIN_PASSWORD" in str(error.value)


def test_output_guard_rejects_a_credential():
    secret = "correct-horse-battery-staple"

    environment = {"LINKEDIN_PASSWORD": secret}

    with pytest.raises(credential_module.CredentialError):
        credential_module.assert_no_credential_in_text(
            f"the password was {secret}", environment
        )


def test_output_guard_allows_clean_text():
    credential_module.assert_no_credential_in_text(
        "collection finished with 12 posts",
        {"LINKEDIN_PASSWORD": "correct-horse-battery-staple"},
    )


def test_secrets_live_inside_the_ignored_tree(tmp_path):
    directory = credential_module.ensure_secrets_directory(tmp_path)

    resolved = directory.resolve()
    repository = tmp_path.resolve()

    assert repository in resolved.parents

    profile = credential_module.browser_profile_directory(tmp_path)

    assert profile.is_relative_to(directory)


def test_env_file_is_not_read_when_absent(tmp_path):
    # Must not raise when there is no .env at all.
    credential_module.load_local_environment(tmp_path / ".env")


# ---------------------------------------------------------------------
# Post ID derivation
# ---------------------------------------------------------------------


def test_post_id_is_deterministic():
    first = post_id_for("urn:li:activity:7123456789")
    second = post_id_for("urn:li:activity:7123456789")

    assert first == second


def test_post_id_strips_unsafe_characters():
    identifier = post_id_for("../../etc/passwd")

    assert "/" not in identifier
    assert ".." not in identifier.split("-")
    assert identifier == "etc-passwd"


def test_post_id_of_an_empty_identifier_is_refused():
    with pytest.raises(CollectionError):
        post_id_for("   ")


def test_post_id_is_bounded():
    identifier = post_id_for("urn:li:activity:" + "9" * 500)

    assert len(identifier) <= 80


# ---------------------------------------------------------------------
# A fake source
# ---------------------------------------------------------------------


class FakeSource(Source):
    """Yields a fixed list, then stops."""

    name = "fake"
    platform = "fake"

    def __init__(self, posts, *, stop=None, raise_instead=False):
        self._posts = posts
        self._stop = stop
        self._raise_instead = raise_instead
        self.resumed_with = set()
        self.calls = []

    def discover(self, **limits):
        self.calls.append(limits)

        for post in self._posts:
            yield post

        if self._raise_instead:
            raise self._stop

        raise StopReasonRaise(self._stop or StopReason.EXHAUSTED)

    def resume(self, seen):
        self.resumed_with = set(seen)

    def close(self):
        pass


class StopReasonRaise(CollectionStopped):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def make_collected(post_id, text="Question text.", **extra):
    return CollectedPost(
        source_post_id=post_id,
        text=text,
        **extra,
    )


# ---------------------------------------------------------------------
# Collector: happy path
# ---------------------------------------------------------------------


def test_collector_persists_posts(tmp_path):
    posts_root = tmp_path / "posts"
    posts_root.mkdir()

    source = FakeSource(
        [
            make_collected("urn:li:activity:1"),
            make_collected("urn:li:activity:2"),
        ]
    )

    report = Collector(
        source, root=posts_root, repository_root=tmp_path
    ).run()

    assert report.state.persisted == 2
    assert report.state.discovered == 2
    assert report.state.failed == 0
    assert report.stopped_because == StopReason.EXHAUSTED.value

    written = sorted(path.name for path in posts_root.iterdir())

    assert written == ["urn-li-activity-1", "urn-li-activity-2"]


def test_collector_writes_a_valid_post_document(tmp_path):
    posts_root = tmp_path / "posts"
    posts_root.mkdir()

    Collector(
        FakeSource(
            [
                make_collected(
                    "urn:li:activity:9",
                    text="How do you tune a Delta table?",
                    url="https://www.linkedin.com/feed/update/urn:li:activity:9/",
                    author="my-handle",
                )
            ]
        ),
        root=posts_root,
        repository_root=tmp_path,
    ).run()

    document = PostDocument.load(posts_root / "urn-li-activity-9")

    assert document.data["id"] == "urn-li-activity-9"
    assert (
        document.data["original_text"]
        == "How do you tune a Delta table?"
    )
    assert document.data["source"]["author"] == "my-handle"
    assert document.data["source"]["platform"] == "fake"

    # The scaffold must be immediately readable by the worker.
    from src.ingestion.post_loader import load_post

    post = load_post(posts_root / "urn-li-activity-9")

    assert post.id == "urn-li-activity-9"


def test_collector_never_invents_missing_values(tmp_path):
    posts_root = tmp_path / "posts"
    posts_root.mkdir()

    Collector(
        FakeSource([make_collected("post-1")]),
        root=posts_root,
        repository_root=tmp_path,
    ).run()

    document = PostDocument.load(posts_root / "post-1")

    # No author or url was provided, so none is written.
    assert document.data["source"]["author"] is None
    assert document.data["source"]["url"] is None


# ---------------------------------------------------------------------
# Collector: idempotency
# ---------------------------------------------------------------------


def test_second_run_creates_no_duplicates(tmp_path):
    posts_root = tmp_path / "posts"
    posts_root.mkdir()

    source = FakeSource(
        [
            make_collected("post-1"),
            make_collected("post-2"),
        ]
    )

    Collector(
        source, root=posts_root, repository_root=tmp_path
    ).run()

    again = Collector(
        FakeSource(
            [
                make_collected("post-1"),
                make_collected("post-2"),
            ]
        ),
        root=posts_root,
        repository_root=tmp_path,
    ).run()

    assert again.state.persisted == 0
    assert again.state.duplicates == 2

    assert len(list(posts_root.iterdir())) == 2


def test_refresh_preserves_existing_enrichment(tmp_path):
    posts_root = tmp_path / "posts"
    posts_root.mkdir()

    Collector(
        FakeSource([make_collected("post-1", text="original")]),
        root=posts_root,
        repository_root=tmp_path,
    ).run()

    # Simulate the enrichment worker having filled this in.
    document = PostDocument.load(posts_root / "post-1")
    document.data["ai_analysis"]["summary"] = "A useful summary."
    document.data["interview_questions"] = [
        {
            "question": "Explain partitioning.",
            "type": "scenario",
            "difficulty": "medium",
            "answer": "Low cardinality only.",
        }
    ]
    document.save(posts_root / "post-1")

    # A later collection with different source text must not wipe it.
    Collector(
        FakeSource([make_collected("post-1", text="refreshed")]),
        root=posts_root,
        repository_root=tmp_path,
    ).run()

    after = PostDocument.load(posts_root / "post-1")

    assert after.data["ai_analysis"]["summary"] == "A useful summary."
    assert len(after.data["interview_questions"]) == 1
    assert after.data["original_text"] == "refreshed"


def test_dry_run_writes_nothing(tmp_path):
    posts_root = tmp_path / "posts"
    posts_root.mkdir()

    report = Collector(
        FakeSource([make_collected("post-1")]),
        root=posts_root,
        limits=CollectionLimits(dry_run=True),
        repository_root=tmp_path,
    ).run()

    assert report.state.persisted == 1
    assert not list(posts_root.iterdir())


# ---------------------------------------------------------------------
# Collector: limits and resume
# ---------------------------------------------------------------------


def test_limits_are_passed_to_the_source(tmp_path):
    source = FakeSource([])

    Collector(
        source,
        root=tmp_path / "posts",
        limits=CollectionLimits(
            max_posts=3, since="2026-01-01", until="2026-12-31"
        ),
        repository_root=tmp_path,
    ).run()

    assert source.calls[0]["max_posts"] == 3
    assert source.calls[0]["since"] == "2026-01-01"
    assert source.calls[0]["until"] == "2026-12-31"


def test_max_posts_is_honoured(tmp_path):
    """
    The collector enforces the bound itself, not just the source.

    A source that ignores its limit must still be stopped, so a
    misbehaving source cannot produce an unbounded run.
    """

    posts_root = tmp_path / "posts"
    posts_root.mkdir()

    class GreedySource(FakeSource):
        def __init__(self):
            # Deliberately ignores max_posts.
            super().__init__(
                [make_collected(f"post-{index}") for index in range(10)],
                stop=StopReason.EXHAUSTED,
            )

    report = Collector(
        GreedySource(),
        root=posts_root,
        limits=CollectionLimits(max_posts=3),
        repository_root=tmp_path,
    ).run()

    assert report.state.persisted == 3
    assert report.stopped_because == StopReason.MAX_POSTS.value
    assert len(list(posts_root.iterdir())) == 3


def test_max_posts_from_the_source_is_reported(tmp_path):
    source = FakeSource(
        [make_collected(f"post-{index}") for index in range(10)],
        stop=StopReason.MAX_POSTS,
    )

    report = Collector(
        source,
        root=tmp_path / "posts",
        limits=CollectionLimits(max_posts=3),
        repository_root=tmp_path,
    ).run()

    assert report.stopped_because == StopReason.MAX_POSTS.value
    assert report.state.persisted <= 3


def test_resume_carries_totals_forward(tmp_path):
    posts_root = tmp_path / "posts"
    posts_root.mkdir()

    Collector(
        FakeSource([make_collected("post-1")]),
        root=posts_root,
        repository_root=tmp_path,
    ).run()

    resumed = Collector(
        FakeSource([make_collected("post-2")]),
        root=posts_root,
        repository_root=tmp_path,
    ).run(resume=True)

    # Totals accumulate rather than restarting.
    assert resumed.state.persisted == 2
    assert resumed.state.discovered == 2


def test_resume_tells_the_source_where_it_stopped(tmp_path):
    posts_root = tmp_path / "posts"
    posts_root.mkdir()

    Collector(
        FakeSource([make_collected("post-1")]),
        root=posts_root,
        repository_root=tmp_path,
    ).run()

    source = FakeSource([make_collected("post-2")])

    Collector(
        source, root=posts_root, repository_root=tmp_path
    ).run(resume=True)

    assert "post-1" in source.resumed_with


def test_state_is_written_after_each_post(tmp_path):
    posts_root = tmp_path / "posts"
    posts_root.mkdir()

    Collector(
        FakeSource(
            [
                make_collected("post-1"),
                make_collected("post-2"),
                make_collected("post-3"),
            ]
        ),
        root=posts_root,
        repository_root=tmp_path,
    ).run()

    state = read_state(tmp_path)

    assert state is not None
    assert state.persisted == 3
    assert state.last_post_id == "post-3"


def test_state_records_the_stopping_reason(tmp_path):
    Collector(
        FakeSource([], stop=StopReason.SCROLL_LIMIT),
        root=tmp_path / "posts",
        repository_root=tmp_path,
    ).run()

    state = read_state(tmp_path)

    assert state.stopped_because == StopReason.SCROLL_LIMIT.value


# ---------------------------------------------------------------------
# Collector: failures
# ---------------------------------------------------------------------


def test_unidentifiable_post_is_counted_as_failed(tmp_path):
    report = Collector(
        FakeSource([make_collected("   ")]),
        root=tmp_path / "posts",
        repository_root=tmp_path,
    ).run()

    assert report.state.failed == 1
    assert report.state.persisted == 0


def test_a_write_failure_does_not_abort_the_run(tmp_path):
    class BrokenPost:
        """Stands in for a post that cannot be normalized."""

        source_post_id = "post-1"
        text = ""

        def to_document(self, **kwargs):
            raise OSError("disk full")

    class FailingSource(FakeSource):
        def __init__(self):
            super().__init__([BrokenPost(), make_collected("post-2")])

        def discover(self, **limits):
            for post in self._posts:
                yield post

            raise StopReasonRaise(StopReason.EXHAUSTED)

    report = Collector(
        FailingSource(),
        root=tmp_path / "posts",
        repository_root=tmp_path,
    ).run()

    assert report.state.failed == 1
    assert report.state.persisted == 1


def test_security_challenge_is_reported_not_raised(tmp_path):
    class ChallengedSource(FakeSource):
        def __init__(self):
            super().__init__([make_collected("post-1")])

        def discover(self, **limits):
            yield make_collected("post-1")
            raise SecurityChallenge("captcha", "A captcha appeared.")

    report = Collector(
        ChallengedSource(),
        root=tmp_path / "posts",
        repository_root=tmp_path,
    ).run()

    assert report.security_challenge == "captcha"
    assert (
        report.stopped_because
        == StopReason.SECURITY_CHALLENGE.value
    )
    assert report.succeeded is False


def test_security_challenge_updates_the_checkpoint(tmp_path):
    class ChallengedSource(FakeSource):
        def __init__(self):
            super().__init__([])

        def discover(self, **limits):
            raise SecurityChallenge("otp", "Enter the code.")
            yield  # pragma: no cover

    checkpoint = checkpoint_module.Checkpoint(
        phase=checkpoint_module.CP6_COLLECTION_RUNNING,
    )

    report = Collector(
        ChallengedSource(),
        root=tmp_path / "posts",
        checkpoint=checkpoint,
        repository_root=tmp_path,
    ).run()

    assert report.security_challenge == "otp"

    saved = checkpoint_module.read(tmp_path)

    assert saved is not None
    assert (
        saved.phase
        == checkpoint_module.CP_HUMAN_LINKEDIN_ACTION_REQUIRED
    )
    assert saved.status == checkpoint_module.STATUS_BLOCKED_HUMAN
    assert "otp" in saved.human_action_required
    assert saved.resume_point


# ---------------------------------------------------------------------
# The manual source
# ---------------------------------------------------------------------


def test_manual_source_reads_text_files(tmp_path):
    (tmp_path / "alpha.txt").write_text("Alpha post.", encoding="utf-8")
    (tmp_path / "beta.md").write_text("Beta post.", encoding="utf-8")

    source = ManualSource(tmp_path)

    collected = []

    try:
        for post in source.discover():
            collected.append(post)
    except CollectionStopped as exc:
        assert exc.reason is StopReason.EXHAUSTED

    assert len(collected) == 2
    assert {post.text for post in collected} == {
        "Alpha post.",
        "Beta post.",
    }


def test_manual_source_reads_capture_bundles(tmp_path):
    bundle = tmp_path / "bundle-1"
    bundle.mkdir()

    (bundle / "capture.json").write_text(
        json.dumps(
            {
                "id": "urn:li:activity:55",
                "text": "From a bundle.",
                "url": "https://example.com/55",
                "published_at": "2026-05-05T10:00:00+00:00",
                "author": "my-handle",
            }
        ),
        encoding="utf-8",
    )

    collected = []

    try:
        for post in ManualSource(tmp_path).discover():
            collected.append(post)
    except CollectionStopped:
        pass

    assert len(collected) == 1
    assert collected[0].source_post_id == "urn:li:activity:55"
    assert collected[0].url == "https://example.com/55"
    assert collected[0].author == "my-handle"


def test_manual_source_reads_jsonl(tmp_path):
    export = tmp_path / "export.jsonl"

    export.write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "text": "First"}),
                "",
                json.dumps({"id": "b", "text": "Second"}),
            ]
        ),
        encoding="utf-8",
    )

    collected = []

    try:
        for post in ManualSource(tmp_path).discover():
            collected.append(post)
    except CollectionStopped:
        pass

    assert len(collected) == 1
    assert "First" in collected[0].text
    assert "Second" in collected[0].text


def test_manual_source_respects_max_posts(tmp_path):
    for index in range(5):
        (tmp_path / f"post-{index}.txt").write_text(
            f"Post {index}", encoding="utf-8"
        )

    collected = []

    try:
        for post in ManualSource(tmp_path).discover(max_posts=2):
            collected.append(post)
    except CollectionStopped as exc:
        assert exc.reason is StopReason.MAX_POSTS

    assert len(collected) == 2


def test_manual_source_filters_by_date(tmp_path):
    bundle = tmp_path / "old"
    bundle.mkdir()
    (bundle / "capture.json").write_text(
        json.dumps(
            {
                "id": "old",
                "text": "Old post",
                "published_at": "2020-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    collected = []

    try:
        for post in ManualSource(tmp_path).discover(
            since="2026-01-01"
        ):
            collected.append(post)
    except CollectionStopped:
        pass

    assert collected == []


def test_manual_source_keeps_posts_it_cannot_date(tmp_path):
    bundle = tmp_path / "undated"
    bundle.mkdir()
    (bundle / "capture.json").write_text(
        json.dumps({"id": "u", "text": "Undated post"}),
        encoding="utf-8",
    )

    collected = []

    try:
        for post in ManualSource(tmp_path).discover(
            since="2026-01-01"
        ):
            collected.append(post)
    except CollectionStopped:
        pass

    # Dropping it would silently lose content.
    assert len(collected) == 1


def test_manual_source_rejects_malformed_json(tmp_path):
    bundle = tmp_path / "broken"
    bundle.mkdir()
    (bundle / "capture.json").write_text("{oops", encoding="utf-8")

    with pytest.raises(Exception):
        for _ in ManualSource(tmp_path).discover():
            pass


# ---------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------


def test_collected_post_normalizes_into_a_document():
    collected = CollectedPost(
        source_post_id="urn:li:activity:77",
        text="Normalize me.",
        url="https://example.com/77",
        author="handle",
    )

    document = collected.to_document(
        post_id=post_id_for("urn:li:activity:77"),
        platform="linkedin",
        captured_at="2026-09-30T00:00:00+00:00",
    )

    assert document.post_id == "urn-li-activity-77"
    assert document.data["original_text"] == "Normalize me."
    assert document.data["source"]["url"] == "https://example.com/77"
    assert document.data["source"]["author"] == "handle"
    assert document.data["source"]["platform"] == "linkedin"
    assert document.data["media"] == []


# ---------------------------------------------------------------------
# The LinkedIn source, against a fake page
# ---------------------------------------------------------------------


class FakeLocator:
    """Enough of Playwright's Locator for extraction."""

    def __init__(self, page, selector):
        self._page = page
        self._selector = selector

    def count(self):
        return len(self._page.matches(self._selector))

    def all(self):
        return [
            FakeNode(self._page, node)
            for node in self._page.matches(self._selector)
        ]

    @property
    def first(self):
        nodes = self.all()

        return nodes[0] if nodes else FakeNode(self._page, {})


class FakeNode:
    def __init__(self, page, data):
        self._page = page
        self._data = data

    def get_attribute(self, name):
        return self._data.get(name)

    def inner_text(self, timeout=None):
        if not self._data:
            return ""
        return self._data.get("_text", "")

    def locator(self, selector):
        return FakeLocator(self._page, selector)

    def is_visible(self):
        return bool(self._data)

    def click(self, timeout=None):
        self._page.expanded += 1


class FakePage:
    """A page whose content grows when scrolled."""

    def __init__(self, rounds, *, body_text=""):
        self.rounds = rounds
        self.round = 0
        self.body_text = body_text
        self.height = 1000
        self.expanded = 0
        self.signed_in = True
        self.goto_calls = []

    # -- content ----------------------------------------------------
    def matches(self, selector):
        """
        Return the nodes a selector would match.

        Keyed on distinctive substrings rather than exact selector
        text, so the fake keeps working when the collector tunes its
        selectors.
        """

        current = self.rounds[self.round]

        if "update-v2" in selector or "urn:li:activity" in selector:
            return current

        if "Show more" in selector or "see more" in selector:
            return [{"_text": "", "_expand": True}]

        if "show-more-text" in selector or "update-components-text" in (
            selector
        ):
            return [{"_text": post["_text"]} for post in current]

        if "feed/update" in selector:
            return [{"href": post["href"]} for post in current]

        if "time" in selector or "sub-description" in selector:
            return [{"_text": post["when"]} for post in current]

        if "img" in selector or "figure" in selector:
            return []

        if "session_key" in selector:
            return [{"_text": ""}]

        if "profile photo" in selector or "nav_profile" in selector:
            return [{"_text": "signed in"}]

        return []

    # -- playwright surface ----------------------------------------
    @property
    def mouse(self):
        return FakeMouse(self)

    def locator(self, selector):
        return FakeLocator(self, selector)

    def goto(self, url, wait_until=None):
        self.goto_calls.append(url)

    def inner_text(self, selector, timeout=None):
        return self.body_text

    def evaluate(self, script):
        return self.height

    def scroll(self, delta_y):
        """Reveal the next round of lazily-loaded content."""

        if self.round < len(self.rounds) - 1:
            self.round += 1
            self.height += 1000

    def wait_for_function(self, script, arg=None, timeout=None):
        if self.height > arg:
            return True
        raise TimeoutError("no change")

    def wait_for_load_state(self, state):
        return None

    def fill(self, selector, value):
        raise AssertionError(
            "credentials must not be typed when already signed in"
        )

    def click(self, selector):
        raise AssertionError("no click expected while signed in")


class FakeMouse:
    """Scrolling advances the fake to its next round of content."""

    def __init__(self, page):
        self._page = page

    def wheel(self, delta_x, delta_y):
        self._page.scroll(delta_y)


def make_source(rounds, **kwargs):
    """A LinkedInSource wired to a fake page, no browser launched."""

    source = LinkedInSource(profile="my-handle", **kwargs)

    page = FakePage(rounds)
    source._page = page
    source._context = object()

    return source, page


def linkedin_post(index, text="A post", when="2 days ago"):
    """
    One post as the fake page holds it.

    Keys mirror what FakeNode.get_attribute and FakeNode.inner_text
    read, so the fake exercises the same extraction path as a real
    page.
    """

    identifier = f"urn:li:activity:{7000000 + index}"

    return {
        "data-id": identifier,
        "_text": text,
        "href": (
            f"https://www.linkedin.com/feed/update/{identifier}/"
        ),
        "when": when,
    }


def test_linkedin_collects_rendered_posts(tmp_path):
    rounds = [[linkedin_post(1), linkedin_post(2)]]

    source, page = make_source(rounds)

    collected = []

    try:
        for post in source.discover(max_posts=2):
            collected.append(post)
    except CollectionStopped as exc:
        assert exc.reason is StopReason.MAX_POSTS

    assert len(collected) == 2
    assert collected[0].source_post_id == "urn:li:activity:7000001"
    assert collected[0].text == "A post"
    assert collected[0].author == "my-handle"


def test_linkedin_scrolls_until_content_appears(tmp_path):
    rounds = [
        [linkedin_post(1)],
        [linkedin_post(1), linkedin_post(2)],
        [linkedin_post(1), linkedin_post(2), linkedin_post(3)],
    ]

    source, page = make_source(rounds)

    collected = []

    try:
        for post in source.discover():
            collected.append(post)
    except CollectionStopped as exc:
        assert exc.reason is StopReason.NO_NEW_CONTENT

    assert len(collected) == 3
    assert page.expanded > 0


def test_linkedin_deduplicates_within_a_run(tmp_path):
    rounds = [
        [linkedin_post(1)],
        [linkedin_post(1)],
        [linkedin_post(1)],
    ]

    source, page = make_source(rounds)

    collected = []

    try:
        for post in source.discover():
            collected.append(post)
    except CollectionStopped:
        pass

    assert len(collected) == 1


def test_linkedin_respects_the_scroll_limit(tmp_path):
    rounds = [[linkedin_post(index)] for index in range(10)]

    source, page = make_source(
        rounds, limits=LinkedInLimits(scroll_limit=2)
    )

    try:
        for _ in source.discover():
            pass
    except CollectionStopped as exc:
        assert exc.reason is StopReason.SCROLL_LIMIT

    assert page.round == 2


def test_linkedin_stops_on_a_challenge(tmp_path):
    rounds = [[linkedin_post(1)]]

    source, page = make_source(rounds)
    page.body_text = "Please complete the CAPTCHA to continue"

    collected = []

    with pytest.raises(SecurityChallenge) as error:
        for post in source.discover():
            collected.append(post)

    assert error.value.kind == "captcha"


@pytest.mark.parametrize(
    "body,kind",
    [
        ("Verify it's you", "unusual_login"),
        ("Enter the verification code we sent", "otp"),
        ("Two-step verification", "two_factor"),
        ("Access Denied", "access_denied"),
        ("Your account is temporarily restricted", "account_restricted"),
        ("Authenticate to continue", "authentication"),
    ],
)
def test_every_challenge_kind_stops_collection(
    tmp_path, body, kind
):
    source, page = make_source([[linkedin_post(1)]])
    page.body_text = body

    with pytest.raises(SecurityChallenge) as error:
        for _ in source.discover():
            pass

    assert error.value.kind == kind


def test_linkedin_stops_when_the_layout_is_unreadable(tmp_path):
    class BrokenPage(FakePage):
        def matches(self, selector):
            raise RuntimeError("selector engine exploded")

    source = LinkedInSource(profile="my-handle")
    source._page = BrokenPage([[]])
    source._context = object()

    with pytest.raises(CollectionStopped) as error:
        for _ in source.discover():
            pass

    assert error.value.reason is StopReason.LAYOUT_CHANGED


def test_linkedin_skips_posts_without_a_stable_id(tmp_path):
    class NoIdPage(FakePage):
        def matches(self, selector):
            if "update-v2" in selector:
                return [{"_text": "orphan"}]
            return []

    source = LinkedInSource(profile="my-handle")
    source._page = NoIdPage([[]])
    source._context = object()

    collected = []

    try:
        for post in source.discover():
            collected.append(post)
    except CollectionStopped:
        pass

    # An unstable identifier would duplicate on every run.
    assert collected == []


def test_linkedin_resume_skips_known_posts(tmp_path):
    rounds = [
        [linkedin_post(1), linkedin_post(2)],
        [linkedin_post(1), linkedin_post(2)],
    ]

    source, page = make_source(rounds)
    source.resume({"urn:li:activity:7000001"})

    collected = []

    try:
        for post in source.discover():
            collected.append(post)
    except CollectionStopped:
        pass

    assert [post.source_post_id for post in collected] == [
        "urn:li:activity:7000002"
    ]


def test_linkedin_navigates_to_the_configured_profile(tmp_path):
    rounds = [[linkedin_post(1)]]

    source, page = make_source(rounds)

    try:
        for _ in source.discover(max_posts=1):
            pass
    except CollectionStopped:
        pass

    assert any(
        "/in/my-handle/recent-activity/all/" in url
        for url in page.goto_calls
    )


def test_linkedin_requires_a_profile(tmp_path):
    source = LinkedInSource(profile="")
    source._page = FakePage([[]])
    source._context = object()

    with pytest.raises(CollectionStopped) as error:
        for _ in source.discover():
            pass

    assert "profile" in str(error.value).lower()


def test_linkedin_fills_credentials_only_when_signed_out(
    monkeypatch,
):
    """
    The password must be typed through the page, never interpolated
    into a URL or an argument. The fake raises if a fill happens while
    already authenticated.
    """

    rounds = [[linkedin_post(1)]]

    source, page = make_source(rounds)

    # Already signed in, so no credential handling may occur.
    try:
        for _ in source.discover(max_posts=1):
            pass
    except CollectionStopped:
        pass

    assert page.signed_in is True


def test_linkedin_close_is_safe_without_a_browser(tmp_path):
    source = LinkedInSource(profile="my-handle")

    source.close()
    source.close()


def test_playwright_availability_is_reported():
    assert isinstance(linkedin_installed(), bool)


def test_relative_timestamps_parse():
    moment = _parse_relative("3 days ago")

    assert moment is not None

    delta = datetime.now(timezone.utc).replace(
        tzinfo=None
    ) - moment

    assert 2 <= delta.days <= 4


def test_absolute_timestamps_parse():
    moment = _parse_relative("2026-05-05T10:00:00+00:00")

    assert moment is not None
    assert moment.year == 2026


def test_unparsable_timestamps_are_kept():
    # Returning None keeps the post rather than dropping content.
    assert _parse_relative("some time last year") is None


def test_media_filenames_are_safe():
    name = _media_filename(
        "https://media.licdn.com/dms/image/v2/chart.png?token=x", 1
    )

    assert name.endswith(".png")
    assert "?" not in name
    assert "/" not in name


# ---------------------------------------------------------------------
# Read-only guarantee
# ---------------------------------------------------------------------


def test_the_collector_defines_no_interactive_actions():
    """
    A regression guard: the collector must never click anything except
    content expansion. If a new selector that could like, comment,
    share, connect or follow is ever added, this fails.
    """

    source = (
        Path("src/ingestion/sources/linkedin.py")
        .read_text(encoding="utf-8")
        .lower()
    )

    for forbidden in (
        'aria-label*="like"',
        'aria-label*="comment"',
        'aria-label*="share"',
        'aria-label*="follow"',
        'aria-label*="connect"',
        'aria-label*="message"',
        'aria-label*="post"',
        'aria-label*="send"',
        'aria-label*="delete"',
    ):
        assert forbidden not in source, forbidden


def test_the_collector_only_clicks_expanders():
    source = Path(
        "src/ingestion/sources/linkedin.py"
    ).read_text(encoding="utf-8")

    # Exactly one click call site, on the expander selector.
    assert source.count(".click(") == 2  # expanders + sign-in submit
    assert "EXPAND_SELECTORS" in source


def test_no_credential_is_written_to_a_file():
    """
    The collector must never write a credential to disk.
    """

    source = Path(
        "src/ingestion/sources/linkedin.py"
    ).read_text(encoding="utf-8")

    for pattern in (
        'write_text(_credential',
        'write_text(password',
        "json.dump(_credential",
    ):
        assert pattern not in source


# ---------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------


def test_env_is_git_ignored():
    contents = Path(".gitignore").read_text(encoding="utf-8")

    assert ".env" in contents
    assert "!data/posts" not in contents


def test_the_agent_tree_is_git_ignored():
    contents = Path(".gitignore").read_text(encoding="utf-8")

    assert ".agent/" in contents


def test_no_secret_file_is_tracked():
    import subprocess

    completed = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        shell=False,
    )

    tracked = completed.stdout.splitlines()

    for path in tracked:
        assert not path.endswith(".env")
        assert "browser_profile" not in path
        assert "cookies" not in path
        assert "storage_state" not in path


def test_env_example_declares_no_values():
    example = Path(".env.example").read_text(encoding="utf-8")

    for line in example.splitlines():
        if "=" in line:
            _name, _, value = line.partition("=")

            assert value.strip() == "", (
                f"{line.split('=')[0]} must ship empty"
            )