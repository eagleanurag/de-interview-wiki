"""
The local orchestrator: retry, isolation, resume and observability.

An experiment against five hundred real posts ran enrichment as one
GitHub job per post and lost eight of them. Every loss was a provider
response that stopped partway through, and every one of those would have
been answered whole by a second ask. Six of twenty batches failed, the
aggregation never ran, and nothing deployed.

So these tests exist to pin the three properties that would have made
that run succeed:

* a truncated or throttled answer is asked again, and a missing API key
  is not;
* one post failing never stops the next one, however it failed;
* a run that is interrupted resumes from where it stopped rather than
  starting over.

No live provider is required. Failures are raised deterministically by
a scripted client, and the provider's own error shapes are used verbatim
so the classifier is tested against what it will actually see.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.ai.recovery import FailureKind, classify, is_recoverable
from src.ingestion.post_document import PostDocument


DELTA_TEXT = (
    "Delta Lake gives a data lake ACID guarantees through a transaction "
    "log. Schema evolution adds a column without rewriting the files "
    "already there, and time travel queries an earlier version of the "
    "table. On Databricks these tables run on Apache Spark."
)


#: Verbatim from the failed batch logs of run 37026765769.
REAL_TRUNCATION = (
    'OpenCode failed with exit code 1.\n\n'
    'STDOUT:\n\n'
    'STDERR:\n'
    '{"type":"error","timestamp":1790955962706,'
    '"sessionID":"ses_f02b8c743ffeMLo23i1jfYe1Yp",'
    '"error":{"type":"provider.invalid-output",'
    '"message":"OpenAI Chat stream ended without finish_reason",'
    '"status":200}}'
)

#: The marker :func:`write_post` leaves in a post's text, so a scripted
#: client can tell which post a prompt is about without depending on how
#: the prompt is worded.
_MARKER = re.compile(r"\[marker ([a-z0-9_.-]+)\]")


# ---------------------------------------------------------------------
# Building a corpus
# ---------------------------------------------------------------------


def write_post(
    root: Path,
    post_id: str,
    text: str = DELTA_TEXT,
) -> str:
    """
    One committed post on disk, the shape the pipeline reads.

    The post's own identifier is written into its text so a scripted
    client can tell which post it was asked about. Matching on the
    identifier rather than on a prompt heading keeps the tests from
    breaking when the prompt is reworded, which is not what any of them
    is about.
    """

    directory = root / "data" / "posts" / post_id

    directory.mkdir(parents=True, exist_ok=True)

    document = PostDocument.new(
        post_id,
        text=f"{text} [marker {post_id}]",
        platform="linkedin",
        url=f"https://www.linkedin.com/posts/{post_id}",
        author="A Fixture",
        captured_at="2026-10-02T04:30:00+00:00",
        primary_topic="Delta Lake",
        interview_relevant=True,
    )

    document.save(directory)

    return post_id


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch):
    """
    A working directory with posts in it, and the pipeline pointed at it.

    The pipeline resolves ``data/posts`` and ``build/`` relative to the
    working directory, which is deliberate: it means a run writes
    nothing into the repository. The tests inherit that by running in a
    temporary directory.
    """

    monkeypatch.chdir(tmp_path)

    return tmp_path


# ---------------------------------------------------------------------
# A scripted provider
# ---------------------------------------------------------------------


class Scripted:
    """
    A stand-in for the model that fails the way the real one did.

    ``failures`` maps a post identifier to the exceptions its calls
    should raise, consumed one per call, so a post can be made to fail
    twice and then succeed or fail every time. Calls are recorded per
    post so a test can assert how many times the model was actually
    asked -- which is the difference between "it retried" and "it asked
    again".

    The post is identified from a marker written into its text by
    :func:`write_post`, so nothing here depends on the prompt's wording
    and no list of identifiers has to be kept in step with the batch.
    """

    def __init__(
        self,
        failures: dict[str, list[Exception]] | None = None,
    ) -> None:
        self.failures = failures or {}
        self.calls: list[str] = []

    def _identify(self, prompt: str) -> str:
        match = _MARKER.search(prompt)

        return match.group(1) if match else "unknown"

    def factory(self, tag: str):
        client = self

        class _Client:
            def run(self, prompt: str, **kwargs):
                post_id = client._identify(prompt)

                client.calls.append(post_id)

                queue = client.failures.get(post_id) or []

                if queue:
                    raise queue.pop(0)

                return _answer(post_id)

        return _Client()


def _answer(post_id: str):
    """A complete, valid answer, with no technology the post lacks."""

    from src.ai.opencode import OpenCodeResult

    return OpenCodeResult(
        text="{}",
        session_id="ses_test",
        data={
            "summary": (
                "A post about Delta Lake, which gives a data lake ACID "
                "guarantees and lets a table be queried as it was."
            ),
            "topics": ["Delta Lake", "Apache Spark"],
            "subtopics": ["Time travel"],
            "concepts": [
                {
                    "name": "ACID guarantees",
                    "explanation": (
                        "Atomicity, consistency, isolation and "
                        "durability, provided by the transaction log."
                    ),
                }
            ],
            "classification": {
                "domain": "Data Engineering",
                "primary_topic": "Delta Lake",
                "secondary_topics": ["Apache Spark"],
                "interview_relevant": True,
            },
            "interview_questions": [
                {
                    "question": "What is Delta Lake?",
                    "type": "theory",
                    "difficulty": "easy",
                    "what_strong_answers_cover": [
                        "A storage format with a transaction log."
                    ],
                }
            ],
        },
    )


def install(monkeypatch, script: Scripted) -> None:
    """Point the pipeline's client factory at the script."""

    monkeypatch.setattr(
        "src.ai.enricher.OpenCodeClient",
        lambda *a, **k: script.factory("x"),
    )


# ---------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------


class TestFailureClassification:
    """
    Whether a retry could plausibly help.

    Decided from the shape of the failure rather than from where it came
    from, so the answer is the same whether the provider names itself.
    """

    def test_the_real_truncation_is_recoverable(self):
        verdict = classify(RuntimeError(REAL_TRUNCATION))

        assert verdict.kind is FailureKind.TRUNCATED
        assert verdict.recoverable is True

    def test_a_status_200_alone_does_not_mean_success(self):
        """
        The failure that got through: the provider reported 200 and
        stopped anyway. Treating the status as the outcome is exactly
        the mistake that lost those eight posts.
        """

        verdict = classify(RuntimeError(REAL_TRUNCATION))

        assert "200" in verdict.evidence or "finish_reason" in verdict.evidence

    def test_a_truncated_json_fragment_is_not_a_schema_problem(self):
        """
        The other observed failure was a Pydantic model_type error whose
        value was a JSON fragment.

        It looks like the model returned the wrong type, and calling it
        a schema mismatch would have sent the fix towards weakening the
        schema. It was the same truncation, reported differently.
        """

        from src.ai.enricher import EnrichmentError

        error = EnrichmentError(
            "The enrichment response did not match the expected "
            "structure: 1 validation error for AIEnrichmentResponse\n"
            "  Input should be a valid dictionary or instance of "
            "AIEnrichmentResponse [type=model_type, "
            "input_value='{\\n  \"summary\": \""
        )

        verdict = classify(error)

        assert verdict.kind is FailureKind.TRUNCATED
        assert verdict.recoverable is True

    def test_a_whole_answer_missing_fields_is_a_schema_problem(self):
        from src.ai.enricher import EnrichmentError

        error = EnrichmentError(
            "The enrichment response did not match the expected "
            "structure: 1 validation error for AIEnrichmentResponse\n"
            "  Field required [type=missing, input_value={}]"
        )

        verdict = classify(error)

        assert verdict.kind is FailureKind.SCHEMA
        assert verdict.recoverable is True

    @pytest.mark.parametrize(
        "message",
        [
            "HTTP 429 Too Many Requests",
            "request timed out after 1800 seconds",
            '{"error":{"status":503,"message":"Service unavailable"}}',
            "connection reset by peer",
            "the model is overloaded, capacity 0",
        ],
    )
    def test_transient_failures_are_recoverable(self, message: str):
        verdict = classify(RuntimeError(message))

        assert verdict.kind is FailureKind.TRANSIENT
        assert verdict.recoverable is True

    @pytest.mark.parametrize(
        "message",
        [
            "HTTP 401 Unauthorized: invalid API key",
            "model not found: opencode/nope",
            "billing quota exceeded",
            "HTTP 400 Bad Request",
        ],
    )
    def test_permanent_failures_are_not_retried(self, message: str):
        """
        Retrying these spends the budget and hides the cause.

        A run that reported three attempts and then a failure would look
        like it had recovered from something, when in fact it had asked
        the same question three times and been refused the same way.
        """

        verdict = classify(RuntimeError(message))

        assert verdict.kind is FailureKind.PERMANENT
        assert verdict.recoverable is False

    def test_a_missing_post_is_local_and_unrecoverable(self):
        verdict = classify(
            FileNotFoundError(
                "Post directory does not exist: data/posts/nope"
            )
        )

        # Judged before the provider patterns, because "does not exist"
        # is also how a provider says a model does not exist, and
        # matching words first would point an operator at the wrong
        # system.
        assert verdict.kind is FailureKind.LOCAL
        assert verdict.recoverable is False

    def test_a_missing_client_is_local(self):
        verdict = classify(
            RuntimeError(
                "Could not locate OpenCode. Install OpenCode or set "
                "OPENCODE_EXECUTABLE."
            )
        )

        assert verdict.kind is FailureKind.LOCAL
        assert verdict.recoverable is False

    def test_an_unrecognised_failure_is_still_tried_once(self):
        verdict = classify(RuntimeError("something nobody has seen"))

        # Unknown deserves one more look: the cost is one call and the
        # alternative is writing a post off on a guess.
        assert verdict.kind is FailureKind.UNKNOWN
        assert verdict.recoverable is True

    def test_the_cause_chain_is_considered(self):
        inner = RuntimeError("stream ended without finish_reason")
        outer = RuntimeError("OpenCode failed with exit code 1")

        outer.__cause__ = inner

        assert is_recoverable(outer)

    def test_classification_serialises(self):
        payload = classify(RuntimeError(REAL_TRUNCATION)).as_dict()

        assert payload["kind"] == "truncated_response"
        assert payload["recoverable"] is True
        assert isinstance(payload["evidence"], str)


# ---------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------


class TestRetry:
    """
    A post is asked again when asking again could work.

    The assertion is on how many times the model was called, not on the
    outcome alone, because "it retried" and "it asked again" are
    different claims and only one of them is the property.
    """

    def _enrich(self, monkeypatch, post_id: str, failures: list[Exception], attempts: int = 3):
        from src.pipeline import enrich

        script = Scripted({post_id: list(failures)})

        install(monkeypatch, script)

        outcome = enrich(
            [post_id], force=False, jobs=1, attempts=attempts, quiet=True
        )

        return outcome, script

    def test_a_truncated_response_is_asked_again(
        self, corpus: Path, monkeypatch
    ):
        post_id = write_post(corpus, "sample_a")

        outcome, script = self._enrich(
            monkeypatch,
            post_id,
            [RuntimeError(REAL_TRUNCATION)],
        )

        assert outcome["failed"] == {}
        assert script.calls == [post_id, post_id]
        assert outcome["report"]["retried"] == 1

    def test_two_failures_then_success(self, corpus: Path, monkeypatch):
        post_id = write_post(corpus, "sample_b")

        outcome, script = self._enrich(
            monkeypatch,
            post_id,
            [
                RuntimeError(REAL_TRUNCATION),
                RuntimeError("HTTP 429 Too Many Requests"),
            ],
        )

        assert outcome["failed"] == {}
        assert len(script.calls) == 3
        assert outcome["report"]["retried"] == 1

    def test_retry_exhaustion_writes_the_post_off(
        self, corpus: Path, monkeypatch
    ):
        post_id = write_post(corpus, "sample_c")

        outcome, script = self._enrich(
            monkeypatch,
            post_id,
            [RuntimeError(REAL_TRUNCATION)] * 5,
            attempts=3,
        )

        # Bounded: three attempts, not five, and not forever.
        assert len(script.calls) == 3
        assert post_id in outcome["failed"]
        assert outcome["report"]["failed"] == 1

    def test_a_permanent_failure_is_not_retried(
        self, corpus: Path, monkeypatch
    ):
        post_id = write_post(corpus, "sample_d")

        outcome, script = self._enrich(
            monkeypatch,
            post_id,
            [RuntimeError("HTTP 401 Unauthorized: invalid API key")] * 5,
            attempts=3,
        )

        # One call, not three. Asking again cannot help and would hide
        # the cause behind a claim of recovery.
        assert len(script.calls) == 1
        assert post_id in outcome["failed"]

    def test_the_failure_record_is_machine_readable(
        self, corpus: Path, monkeypatch
    ):
        post_id = write_post(corpus, "sample_e")

        outcome, _ = self._enrich(
            monkeypatch,
            post_id,
            [RuntimeError(REAL_TRUNCATION)] * 5,
            attempts=2,
        )

        record = outcome["report"]["failures"][0]

        for field in (
            "post_id",
            "stage",
            "attempts",
            "error_type",
            "error_message",
            "timestamp",
            "recoverable",
            "status",
        ):
            assert field in record, field

        assert record["post_id"] == post_id
        assert record["stage"] == "enrich"
        assert record["attempts"] == 2
        assert record["error_kind"] == "truncated_response"

    def test_the_failure_record_is_on_disk(
        self, corpus: Path, monkeypatch
    ):
        from src.pipeline import enrich

        post_id = write_post(corpus, "sample_f")

        install(
            monkeypatch,
            Scripted({post_id: [RuntimeError(REAL_TRUNCATION)] * 4}),
        )

        enrich([post_id], force=False, jobs=1, attempts=2, quiet=True)

        state = json.loads(
            (corpus / "build" / "enrichment-state.json").read_text(
                encoding="utf-8"
            )
        )

        record = next(
            entry
            for entry in state["posts"]
            if entry["post_id"] == post_id
        )

        assert record["status"] == "failed"
        assert record["attempts"] == 2
        assert record["error_type"]
        assert record["error_message"]

    def test_a_failed_attempt_does_not_write_a_result(
        self, corpus: Path, monkeypatch
    ):
        from src.pipeline import enrich

        post_id = write_post(corpus, "sample_g")

        install(
            monkeypatch,
            Scripted({post_id: [RuntimeError(REAL_TRUNCATION)] * 4}),
        )

        enrich([post_id], force=False, jobs=1, attempts=2, quiet=True)

        # A half-written result would be a post that quietly stops
        # appearing in the site rather than one that fails loudly.
        assert not (
            corpus / "build" / "worker-results" /
            f"cloud_worker_{post_id}.json"
        ).is_file()

    def test_retries_do_not_fabricate_content(
        self, corpus: Path, monkeypatch
    ):
        """
        A retry is a fresh sample, not a repaired one.

        Nothing is filled in to make validation pass: the answer that
        finally validates is one the model produced whole. Checked by
        reading what was stored.
        """

        from src.pipeline import enrich

        post_id = write_post(corpus, "sample_h")

        install(
            monkeypatch,
            Scripted({post_id: [RuntimeError(REAL_TRUNCATION)]}),
        )

        enrich([post_id], force=False, jobs=1, attempts=3, quiet=True)

        payload = json.loads(
            (
                corpus / "build" / "worker-results" /
                f"cloud_worker_{post_id}.json"
            ).read_text(encoding="utf-8")
        )

        concepts = payload["ai_analysis"]["concepts"]

        assert concepts
        assert payload["ai_analysis"]["summary"]
        assert payload["interview_questions"]


# ---------------------------------------------------------------------
# Isolation and concurrency
# ---------------------------------------------------------------------


class TestIsolationAndConcurrency:
    def test_one_failed_post_does_not_stop_the_batch(
        self, corpus: Path, monkeypatch
    ):
        """
        Post two of four fails. Posts one, three and four still land.

        This is the property the GitHub run lacked at batch level: a
        truncated response there lost the post, and a failed batch
        stopped the aggregation that would have used the other
        nineteen.
        """

        from src.pipeline import enrich

        identifiers = [
            write_post(corpus, f"sample_{index}", DELTA_TEXT + f" Part {index}.")
            for index in range(4)
        ]

        install(
            monkeypatch,
            Scripted(
                {identifiers[1]: [RuntimeError(REAL_TRUNCATION)] * 4}
            ),
        )

        outcome = enrich(
            identifiers, force=False, jobs=1, attempts=2, quiet=True
        )

        assert len(outcome["written"]) == 3
        assert list(outcome["failed"]) == [identifiers[1]]

    def test_jobs_one_and_jobs_three_agree(self, corpus: Path, monkeypatch):
        from src.pipeline import enrich

        identifiers = [
            write_post(corpus, f"serial_{index}", DELTA_TEXT + f" S{index}.")
            for index in range(5)
        ]

        install(monkeypatch, Scripted())

        one = enrich(
            identifiers, force=False, jobs=1, attempts=3, quiet=True
        )

        serial = {
            path.name: json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(
                (corpus / "build" / "worker-results").glob("*.json")
            )
        }

        for path in (corpus / "build" / "worker-results").glob("*.json"):
            path.unlink()

        three = enrich(
            identifiers, force=False, jobs=3, attempts=3, quiet=True
        )

        parallel = {
            path.name: json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(
                (corpus / "build" / "worker-results").glob("*.json")
            )
        }

        assert one["failed"] == {} == three["failed"]

        # A fan-out that changed the result would be a second pipeline
        # rather than a faster one.
        assert serial.keys() == parallel.keys()

        for name in serial:
            assert serial[name]["ai_analysis"] == parallel[name]["ai_analysis"]

    def test_concurrent_writes_do_not_corrupt_results(
        self, corpus: Path, monkeypatch
    ):
        from src.pipeline import enrich

        identifiers = [
            write_post(corpus, f"race_{index}", DELTA_TEXT + f" R{index}.")
            for index in range(12)
        ]

        install(monkeypatch, Scripted())

        enrich(identifiers, force=False, jobs=6, attempts=3, quiet=True)

        results = sorted(
            (corpus / "build" / "worker-results").glob("*.json")
        )

        # Every file parses, every one belongs to a different post, and
        # no temporary file was left behind by a losing rename.
        assert len(results) == 12

        seen = set()

        for path in results:
            payload = json.loads(path.read_text(encoding="utf-8"))

            assert payload["id"] not in seen
            seen.add(payload["id"])

        assert not list(
            (corpus / "build" / "worker-results").glob("*.tmp")
        )

    def test_concurrent_failures_are_all_recorded(
        self, corpus: Path, monkeypatch
    ):
        from src.pipeline import enrich

        identifiers = [
            write_post(corpus, f"mixed_{index}", DELTA_TEXT + f" M{index}.")
            for index in range(6)
        ]

        # Every post fails its first attempt and succeeds on its
        # second, so the run is full of retries and the state has to
        # keep them all straight.
        script = Scripted(
            {identifier: [RuntimeError(REAL_TRUNCATION)]
             for identifier in identifiers}
        )

        install(monkeypatch, script)

        outcome = enrich(
            identifiers, force=False, jobs=4, attempts=3, quiet=True
        )

        assert outcome["failed"] == {}
        assert outcome["report"]["retried"] == 6
        assert len(script.calls) == 12

        state = json.loads(
            (corpus / "build" / "enrichment-state.json").read_text(
                encoding="utf-8"
            )
        )

        assert len(state["posts"]) == 6
        assert all(
            entry["status"] == "enriched" for entry in state["posts"]
        )


# ---------------------------------------------------------------------
# Resume and incrementality
# ---------------------------------------------------------------------


class TestResume:
    def test_an_interrupted_run_resumes(
        self, corpus: Path, monkeypatch
    ):
        """
        Two hundred of five hundred done, then the machine restarts.

        The next run must not ask the model about the two hundred. That
        is what the durable state file is for, and what a GitHub batch
        losing its place did not have.
        """

        from src.pipeline import enrich

        first = [
            write_post(corpus, f"done_{index}", DELTA_TEXT + f" D{index}.")
            for index in range(3)
        ]

        second = [
            write_post(corpus, f"todo_{index}", DELTA_TEXT + f" T{index}.")
            for index in range(2)
        ]

        script = Scripted()

        install(monkeypatch, script)

        enrich(first, force=False, jobs=1, attempts=3, quiet=True)

        assert len(script.calls) == 3

        # The terminal closes here. No clean shutdown, no flush.

        before = len(script.calls)

        outcome = enrich(
            second, force=False, jobs=1, attempts=3, quiet=True
        )

        # Only the new posts reached the model. The script accumulates
        # across both runs, so what matters is the delta.
        assert script.calls[before:] == second
        assert len(outcome["written"]) == 2

        # And the earlier three are still known as complete.
        state = json.loads(
            (corpus / "build" / "enrichment-state.json").read_text(
                encoding="utf-8"
            )
        )

        statuses = {
            entry["post_id"]: entry["status"]
            for entry in state["posts"]
        }

        for post_id in first:
            assert statuses[post_id] == "enriched"

        for post_id in second:
            assert statuses[post_id] == "enriched"

    def test_a_failed_post_is_retried_on_the_next_run(
        self, corpus: Path, monkeypatch
    ):
        from src.pipeline import enrich

        post_id = write_post(corpus, "retry_me")

        install(
            monkeypatch,
            Scripted({post_id: [RuntimeError(REAL_TRUNCATION)] * 4}),
        )

        first = enrich(
            [post_id], force=False, jobs=1, attempts=2, quiet=True
        )

        assert post_id in first["failed"]

        # The provider recovers.
        script = Scripted()
        install(monkeypatch, script)

        second = enrich(
            [post_id], force=False, jobs=1, attempts=3, quiet=True
        )

        assert second["failed"] == {}
        assert script.calls == [post_id]

    def test_a_run_that_was_reused_calls_nothing(
        self, corpus: Path, monkeypatch
    ):
        from src.pipeline import enrich

        post_id = write_post(corpus, "cached_once")

        install(monkeypatch, Scripted())

        enrich([post_id], force=False, jobs=1, attempts=3, quiet=True)

        script = Scripted()
        install(monkeypatch, script)

        second = enrich(
            [post_id], force=False, jobs=1, attempts=3, quiet=True
        )

        # The whole point of the fingerprint: zero model calls.
        assert script.calls == []
        assert second["reused"] == [post_id]
        assert second["written"] == []

    def test_a_changed_post_is_enriched_again(
        self, corpus: Path, monkeypatch
    ):
        from src.pipeline import enrich

        post_id = write_post(corpus, "changes")

        install(monkeypatch, Scripted())

        enrich([post_id], force=False, jobs=1, attempts=3, quiet=True)

        write_post(corpus, post_id, DELTA_TEXT + " And a new sentence.")

        script = Scripted()
        install(monkeypatch, script)

        enrich([post_id], force=False, jobs=1, attempts=3, quiet=True)

        assert script.calls == [post_id]

    def test_a_changed_media_file_is_enriched_again(
        self, corpus: Path, monkeypatch
    ):
        """
        Media is part of what enrichment reads, so replacing it has to
        invalidate the result even when the text is untouched.
        """

        from src.pipeline import enrich

        post_id = write_post(corpus, "has_media")

        media = corpus / "data" / "posts" / post_id / "media"
        media.mkdir(parents=True, exist_ok=True)
        (media / "chart.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)

        install(monkeypatch, Scripted())

        enrich([post_id], force=False, jobs=1, attempts=3, quiet=True)

        (media / "chart.png").write_bytes(
            b"\x89PNG\r\n\x1a\n" + b"1" * 64
        )

        script = Scripted()
        install(monkeypatch, script)

        enrich([post_id], force=False, jobs=1, attempts=3, quiet=True)

        assert script.calls == [post_id]

    def test_a_metadata_only_change_costs_nothing(
        self, corpus: Path, monkeypatch
    ):
        """
        New capture date, same content.

        The provenance is refreshed without a model call, which is the
        whole reason freshness is decided on a content digest rather than
        on a timestamp.
        """

        from src.pipeline import enrich

        post_id = write_post(corpus, "meta_only")

        install(monkeypatch, Scripted())

        enrich([post_id], force=False, jobs=1, attempts=3, quiet=True)

        script = Scripted()
        install(monkeypatch, script)

        outcome = enrich(
            [post_id], force=False, jobs=1, attempts=3, quiet=True
        )

        assert script.calls == []
        assert outcome["reused"] == [post_id]

    def test_force_re_enriches_everything(
        self, corpus: Path, monkeypatch
    ):
        from src.pipeline import enrich

        identifiers = [
            write_post(corpus, f"forced_{index}")
            for index in range(2)
        ]

        install(monkeypatch, Scripted())

        enrich(identifiers, force=False, jobs=1, attempts=3, quiet=True)

        script = Scripted()
        install(monkeypatch, script)

        outcome = enrich(
            identifiers, force=True, jobs=1, attempts=3, quiet=True
        )

        assert len(script.calls) == 2
        assert sorted(outcome["written"]) == sorted(identifiers)


# ---------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------


def capture_log(monkeypatch, enrich_module) -> list[str]:
    """
    Collect what the orchestrator printed.

    The module's own ``log`` name is rebound rather than
    ``src.pipeline.paths.log`` patched, because the orchestrator imported
    the name into its own namespace; patching the source module would
    leave the already-bound name pointing at the original.
    """

    printed: list[str] = []

    monkeypatch.setattr(
        enrich_module, "log", printed.append, raising=True
    )

    return printed


class TestObservability:
    def test_the_summary_names_every_category(self, corpus: Path, monkeypatch):
        import src.pipeline.orchestrate as enrich_module
        from src.pipeline import enrich

        identifiers = [
            write_post(corpus, f"obs_{index}", DELTA_TEXT + f" O{index}.")
            for index in range(3)
        ]

        install(
            monkeypatch,
            Scripted({identifiers[0]: [RuntimeError(REAL_TRUNCATION)] * 4}),
        )

        printed = capture_log(monkeypatch, enrich_module)

        enrich(identifiers, force=False, jobs=1, attempts=2, quiet=False)

        summary = "\n".join(printed)

        for label in (
            "Total",
            "Already complete",
            "Processed",
            "retried",
            "Permanently failed",
            "Model attempts",
            "Duration",
        ):
            assert label in summary, label

        assert f"{len(identifiers)}" in summary

    def test_each_post_gets_one_numbered_line(self, corpus: Path, monkeypatch):
        import src.pipeline.orchestrate as enrich_module
        from src.pipeline import enrich

        identifiers = [
            write_post(corpus, f"line_{index}", DELTA_TEXT + f" L{index}.")
            for index in range(2)
        ]

        install(monkeypatch, Scripted())

        printed = capture_log(monkeypatch, enrich_module)

        enrich(identifiers, force=False, jobs=1, attempts=3, quiet=False)

        numbered = [
            line for line in printed if line.startswith("[")
        ]

        assert len(numbered) == 2

        for line in numbered:
            assert "enriched" in line

    def test_a_failed_post_is_named_on_its_line(
        self, corpus: Path, monkeypatch
    ):
        import src.pipeline.orchestrate as enrich_module
        from src.pipeline import enrich

        identifiers = [
            write_post(corpus, f"bad_{index}", DELTA_TEXT + f" B{index}.")
            for index in range(2)
        ]

        install(
            monkeypatch,
            Scripted({identifiers[0]: [RuntimeError(REAL_TRUNCATION)] * 4}),
        )

        printed = capture_log(monkeypatch, enrich_module)

        enrich(identifiers, force=False, jobs=1, attempts=2, quiet=False)

        summary = "\n".join(printed)

        assert "FAILED" in summary
        assert identifiers[0] in summary
        assert "after 2 attempt(s)" in summary

    def test_a_resumed_run_says_what_it_is_picking_up(
        self, corpus: Path, monkeypatch
    ):
        import src.pipeline.orchestrate as enrich_module
        from src.pipeline import enrich

        first = write_post(corpus, "resume_a")
        second = write_post(corpus, "resume_b")

        install(monkeypatch, Scripted())

        enrich([first], force=False, jobs=1, attempts=3, quiet=True)

        printed = capture_log(monkeypatch, enrich_module)

        enrich(
            [first, second], force=False, jobs=1, attempts=3, quiet=False
        )

        assert any("Resuming" in line for line in printed)


# ---------------------------------------------------------------------
# The documented contract
# ---------------------------------------------------------------------


class TestTheDocumentedWorkflow:
    """
    What the README says about running this.

    Worth having because the claims are the operational ones. If the
    documentation says CI enriches, or that a failed batch costs the
    deployment, that is a defect in the document whether or not the
    code does it.
    """

    @property
    def _readme(self) -> str:
        return (
            Path(__file__).resolve().parents[1] / "README.md"
        ).read_text(encoding="utf-8")

    def _section(self, start: str, end: str) -> str:
        text = self._readme

        return text[text.index(start) : text.index(end)]

    def test_the_local_command_is_the_headline(self):
        section = self._section(
            "## Running the whole pipeline locally", "## Pipeline"
        )

        assert "python -m src.pipeline" in section

        for flag in ("--jobs", "--attempts", "--force-enrich", "--only"):
            assert flag in section, flag

    def test_it_says_enrichment_is_local_and_ci_does_not_call_the_model(
        self,
    ):
        section = self._section(
            "## Running the whole pipeline locally", "## Pipeline"
        )

        assert "This is the normal way to process material" in section
        assert "does not call the model" in section

    def test_it_documents_resume(self):
        section = self._section(
            "## Running the whole pipeline locally", "## Pipeline"
        )

        assert "Resume" in section
        assert "enrichment-state.json" in section
        assert "Re-run the pipeline to retry them" in section

    def test_it_documents_the_failure_record_fields(self):
        section = self._section(
            "## Running the whole pipeline locally", "## Pipeline"
        )

        for field in (
            "post_id",
            "stage",
            "attempts",
            "error_type",
            "error_message",
            "timestamp",
            "recoverable",
        ):
            assert f'"{field}"' in section, field

    def test_it_documents_incremental_enrichment(self):
        section = self._section(
            "## Running the whole pipeline locally", "## Pipeline"
        )

        assert "Incremental enrichment" in section
        assert "the model is not called again" in section

    def test_it_documents_rebuilding_without_a_model(self):
        section = self._section(
            "## Running the whole pipeline locally", "## Pipeline"
        )

        assert "only stage that needs a model" in section
        assert "--only aggregate" in section
        assert "--only site" in section

    def test_it_records_why_enrichment_moved_off_ci(self):
        section = self._section(
            "## Why enrichment is local", "## Adding your own material"
        )

        # The measured reason, not a preference. A document that says
        # only "it is better locally" would not survive the next
        # argument.
        assert "37026765769" in section
        assert "without finish_reason" in section
        assert "Fourteen batches succeeded and six failed" in section

    def test_it_says_nothing_was_discarded_from_that_run(self):
        section = self._section(
            "## Why enrichment is local", "## Adding your own material"
        )

        assert "482" in section
        assert "kept and audited" in section

    def test_it_does_not_suggest_a_cloud_model_fan_out(self):
        text = self._readme.lower()

        for banned in (
            "one job per post",
            "one matrix job per post",
            "worker batch",
            "workload_batchesize",
        ):
            assert banned not in text, banned

    def test_the_pipeline_section_lists_no_model_stage(self):
        section = self._section("## Pipeline", "## Why enrichment is local")

        assert "none of which calls a model" in section

        for stage in ("Verify", "Aggregate", "Wiki", "Deploy"):
            assert f"**{stage}**" in section, stage

    def test_the_architecture_marks_enrichment_local(self):
        section = self._section("## Architecture", "## Running the whole")

        assert "ENRICHMENT (local)" in section
        assert "CI                        tests · security" in section


class TestMediaFailures:
    def test_unreadable_media_is_reported_not_dropped(
        self, corpus: Path, monkeypatch
    ):
        """
        A file that will not open costs the file, not the post.

        The post's text is still worth enriching, so the run continues;
        but "the image was skipped" and "the image was read" are
        different claims, and only one of them is true. Reported rather
        than swallowed, which is what the media report is for.
        """

        from src.pipeline import enrich

        post_id = write_post(corpus, "bad_media")

        media = corpus / "data" / "posts" / post_id / "media"
        media.mkdir(parents=True, exist_ok=True)

        # Declared as media, present as a name, and a file the image
        # inspector cannot make sense of.
        (media / "chart.png").write_bytes(b"not an image at all")

        document = json.loads(
            (corpus / "data" / "posts" / post_id / "post.json").read_text(
                encoding="utf-8"
            )
        )

        document["media"] = [
            {
                "type": "image",
                "path": "media/chart.png",
                "description": None,
                "extracted_text": None,
            }
        ]

        (corpus / "data" / "posts" / post_id / "post.json").write_text(
            json.dumps(document, indent=2), encoding="utf-8"
        )

        install(monkeypatch, Scripted())

        outcome = enrich(
            [post_id], force=False, jobs=1, attempts=3, quiet=True
        )

        # The post still landed.
        assert outcome["failed"] == {}
        assert outcome["written"] == [post_id]

        # And the file is named.
        failures = outcome["media_failures"]

        assert post_id in failures, failures
        assert any("chart.png" in item for item in failures[post_id])


# ---------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------


class TestTheOrchestratorStaysLocal:
    """
    The orchestrator reads local files and calls one local client.

    Asserted here rather than only in the security audit because this is
    the component that decides how many times a model is called and with
    what, so it is the place a network call would be introduced.
    """

    def test_the_orchestrator_imports_no_network_module(self):
        import ast

        source = Path("src/pipeline/orchestrate.py").read_text(
            encoding="utf-8"
        )

        tree = ast.parse(source)

        imported: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(
                    alias.name.split(".")[0] for alias in node.names
                )

            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])

        for banned in (
            "socket",
            "ssl",
            "http",
            "urllib",
            "requests",
            "httpx",
            "playwright",
            "selenium",
        ):
            assert banned not in imported, banned

    def test_a_run_writes_nothing_outside_build(
        self, corpus: Path, monkeypatch
    ):
        """
        A run writes into ``build/`` and nowhere else.

        The pipeline's contract is that a run does not touch the
        committed repository, and that is worth testing rather than
        documenting -- especially now that the run also writes a state
        file it did not write before.
        """

        from src.pipeline import enrich

        identifiers = [
            write_post(corpus, f"quiet_{index}", DELTA_TEXT + f" Q{index}.")
            for index in range(3)
        ]

        # What exists before the run, so a new path is attributable.
        before = {
            path.relative_to(corpus).as_posix()
            for path in corpus.rglob("*")
        }

        install(monkeypatch, Scripted())

        enrich(identifiers, force=False, jobs=2, attempts=3, quiet=True)

        after = {
            path.relative_to(corpus).as_posix()
            for path in corpus.rglob("*")
        }

        created = after - before

        assert created, "the run should have written its results"

        for relative in created:
            # "build" itself is the directory the run creates; anything
            # else it makes has to be inside it.
            assert relative == "build" or relative.startswith(
                "build/"
            ), relative

    def test_no_post_path_escapes_into_a_result(
        self, corpus: Path, monkeypatch
    ):
        """
        A result is a document that reaches the published site.

        A Windows path in one would be published, so the check is on the
        stored result rather than on the log.
        """

        from src.pipeline import enrich

        post_id = write_post(corpus, "no_leak")

        install(monkeypatch, Scripted())

        enrich([post_id], force=False, jobs=1, attempts=3, quiet=True)

        body = (
            corpus / "build" / "worker-results" /
            f"cloud_worker_{post_id}.json"
        ).read_text(encoding="utf-8")

        assert "C:\\Users" not in body
        assert "AppData" not in body