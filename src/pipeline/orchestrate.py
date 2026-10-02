"""
Enrichment, as one orchestrator over many independent posts.

The shape is deliberately unchanged from what it was: one post is one
worker result, and one post failing costs that post. What changed is
that a post is no longer given exactly one chance. The GitHub worker run
lost eight posts to truncated provider responses, and every one of those
was an answer that had stopped partway through and would almost
certainly have arrived whole on a second draw.

Four properties are load-bearing:

* **One writer.** The run state belongs to this function on the
  orchestrating thread. Workers return outcomes and never touch shared
  state, so concurrency cannot corrupt it. That is a stronger guarantee
  than a lock around writes would be, because there are no writes to
  race -- and it is why the state file is written atomically anyway, so
  that a crash mid-write cannot leave a file that parses and is wrong.
* **Reuse before work.** A post whose content has not changed keeps the
  result it has and never reaches the model. Decided up front, in order,
  because it is cheap work that should not queue behind model calls.
* **A bounded retry that gives up for a reason.** Truncated and
  throttled responses are re-asked; a missing API key is not. An
  unbounded retry turns a provider outage into an infinite run, and a
  retry that cannot work hides a real fault behind a claim that
  something was recovered.
* **A record for everything that failed.** Stage, attempts, error type,
  message and whether the cause is recoverable, written to disk rather
  than to a log that has already scrolled away.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from src.ai.enricher import AIEnricher
from src.ai.recovery import classify
from src.enrichment.report import (
    RunReport,
    progress_line,
    resume_summary,
)
from src.enrichment.runner import (
    DEFAULT_ATTEMPTS,
    Attempt,
    Enricher,
    Outcome,
    cached,
)
from src.enrichment.state import RunState
from src.ingestion.post_loader import load_post, source_digest
from src.pipeline.freshness import (
    ENRICHER_VERSION,
    _refresh_provenance,
    _reusable,
)
from src.pipeline.paths import RESULTS_DIR, RUN_STATE, log
from src.processing.media_processor import MediaReport, process_media

#: How often the state is flushed. Every post would be a write per post
#: for a record only read on the next run, and every fifty means a crash
#: costs at most that much work.
STATE_FLUSH_EVERY = 25


@dataclass
class Settled:
    """One post, finished, with everything the caller needs to say so."""

    outcome: Outcome
    broken_media: list[str] = field(default_factory=list)


def enrich(
    identifiers: list[str],
    *,
    force: bool,
    jobs: int = 1,
    attempts: int = DEFAULT_ATTEMPTS,
    state_path: Path | None = None,
    quiet: bool = False,
) -> dict:
    """
    Enrich every post that needs it, into its own worker result.

    One post failing costs that post. The source is never modified and
    never deleted, so a failed post is simply retried on the next run,
    which is what makes the pipeline resumable.

    Enrichment is incremental. A post whose content has not changed and
    whose enrichment was produced by the current version keeps the
    result it has, because calling the model again would spend real
    time to arrive at the same answer. ``force`` re-enriches everything
    regardless, which is what a change to the prompt needs.

    ``jobs`` runs several posts at once. Each post is independent -- its
    own directory, its own result file, its own model call -- so this is
    a bound on how many run at once, not a second pipeline. One is the
    default because a bound nobody asked for is a surprise.

    ``attempts`` bounds the retry of a single post. Three by default:
    enough for the truncated responses that actually happened, and low
    enough that a genuinely broken provider fails the run rather than
    appearing to make progress.
    """

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    path = state_path or RUN_STATE

    state = RunState.load(path)
    state.begin_run(ENRICHER_VERSION)
    state.register(identifiers)

    report = RunReport(total=len(identifiers))
    positions = _positions(identifiers)

    started = time.monotonic()

    resume = resume_summary(state, len(identifiers))

    if resume and not quiet:
        log(resume)

    pending = _settle_cached(
        identifiers, state, report, positions, force, quiet
    )

    written: list[str] = []
    media_failures: dict[str, list[str]] = {}

    if pending:
        if not quiet:
            log(
                f"enriching {len(pending)} post(s), up to "
                f"{attempts} attempt(s) each, {jobs} at a time"
            )

        pool = _enricher_pool(attempts)

        written, media_failures = _run_pending(
            pool, pending, jobs, state, report, positions, path, quiet
        )

    report.seconds = time.monotonic() - started

    state.save(path)

    if not quiet:
        log(report.render())

    return _result(state, report, written, media_failures)


def _positions(identifiers: list[str]) -> dict[str, int]:
    """
    Each post's place in the batch.

    Fixed before any work starts, so a progress line can be numbered by
    the post's position even though the posts settle in whatever order
    the scheduler chose. The numbering is therefore stable between runs.
    """

    return {
        identifier: index + 1
        for index, identifier in enumerate(identifiers)
    }


def _settle_cached(
    identifiers: list[str],
    state: RunState,
    report: RunReport,
    positions: dict[str, int],
    force: bool,
    quiet: bool,
) -> list[tuple[str, Path]]:
    """
    Decide what needs the model, and settle what does not.

    Returns the posts still owed an attempt. A post is settled here
    rather than inside the pool because the check is cheap and putting
    it behind the pool would serialise no-cost work behind model calls,
    which is what makes a resumed run slow instead of instant.

    Also where a post that cannot even be loaded is written off: an
    unreadable post is a fact about this machine, not something a retry
    will change.
    """

    pending: list[tuple[str, Path]] = []

    for identifier in identifiers:
        directory = Path("data/posts") / identifier
        target = RESULTS_DIR / f"cloud_worker_{identifier}.json"

        try:
            post = load_post(directory)

            process_media(post, report=MediaReport())

            digest = source_digest(post, directory)

            post.enrichment = type(post.enrichment)(
                source_digest=digest,
                enricher_version=ENRICHER_VERSION,
            )

        except Exception as exc:  # noqa: BLE001
            _settle(
                state,
                report,
                _unloadable(identifier, exc),
                positions,
                quiet,
                path=None,
            )

            continue

        if target.is_file() and not force:
            existing = _reusable(target, digest)

            if existing is not None:
                # The analysis is still correct; the attribution around
                # it may not be, and a knowledge base that kept the
                # stale copy would publish a post that says less than
                # the source does. Refreshing costs no model call.
                _refresh_provenance(existing, post)

                _write_atomic(
                    target,
                    json.dumps(
                        existing, indent=2, ensure_ascii=False
                    ),
                )

                outcome = cached(identifier)

                _settle(
                    state,
                    report,
                    outcome,
                    positions,
                    quiet,
                    path=None,
                )

                continue

        pending.append((identifier, directory))

    return pending


def _run_pending(
    pool,
    pending: list[tuple[str, Path]],
    jobs: int,
    state: RunState,
    report: RunReport,
    positions: dict[str, int],
    state_path: Path,
    quiet: bool,
) -> tuple[list[str], dict[str, list[str]]]:
    """
    Run the posts, bounded, settling each as it finishes.

    Settled on completion rather than in batch, because a run over
    several hundred posts that says nothing for an hour is
    indistinguishable from a hung one. The index on the line is the
    post's fixed position in the batch, so the numbering is stable even
    though arrival is not.
    """

    def work(item: tuple[str, Path]) -> Settled:
        identifier, directory = item

        outcome, broken = _enrich_one(identifier, directory, pool)

        return Settled(outcome=outcome, broken_media=broken)

    written: list[str] = []
    media_failures: dict[str, list[str]] = {}

    def take(settled: Settled) -> None:
        outcome = settled.outcome

        _settle(
            state,
            report,
            outcome,
            positions,
            quiet,
            path=state_path,
        )

        if settled.broken_media:
            media_failures[outcome.post_id] = settled.broken_media

        if outcome.succeeded and not outcome.reused:
            written.append(outcome.post_id)

    if jobs > 1:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = [
                executor.submit(work, item) for item in pending
            ]

            for future in as_completed(futures):
                take(_result_of(future))

    else:
        for item in pending:
            take(work(item))

    return written, media_failures


def _result_of(future) -> Settled:
    """
    One finished future, as a settled post.

    A worker that raised rather than returned is still that post's
    problem rather than the run's, so the exception becomes a failure
    record and the next post carries on.
    """

    try:
        return future.result()

    except Exception as exc:  # noqa: BLE001
        return Settled(
            outcome=_unloadable("unknown", exc)
        )


def _settle(
    state: RunState,
    report: RunReport,
    outcome: Outcome,
    positions: dict[str, int],
    quiet: bool,
    path: Path | None,
) -> None:
    """
    Fold one post's outcome in, on the orchestrating thread.

    The only place run state is updated, which is what lets several
    workers run at once without a lock.
    """

    state.record_outcome(outcome)
    report.add(outcome)

    if not quiet:
        log(
            progress_line(
                positions.get(outcome.post_id, 0),
                report.total,
                outcome,
            )
        )

    if path is not None and report.processed % STATE_FLUSH_EVERY == 0:
        state.save(path)


def _enrich_one(
    identifier: str,
    directory: Path,
    pool,
) -> tuple[Outcome, list[str]]:
    """
    Enrich one post, re-asking what a re-ask can fix.

    Returns the outcome and the media that could not be read. A file
    that will not open is not a reason to lose the post -- the text is
    still worth enriching -- but it is reported rather than dropped,
    because "the image was skipped" and "the image was read" are
    different claims.

    The post is loaded once and the retry loop re-uses it, because a
    retry is a fresh model sample rather than a fresh post. The one
    thing the loop must not carry forward is a half-applied analysis,
    and :meth:`AIEnricher.enrich` raises before writing to the post
    when validation fails, so a failed attempt leaves the post as it
    was loaded.
    """

    enricher = pool()

    post = load_post(directory)

    media_report = MediaReport()

    process_media(post, report=media_report)

    broken = [item.path for item in media_report.failed]

    digest = source_digest(post, directory)

    post.enrichment = type(post.enrichment)(
        source_digest=digest,
        enricher_version=ENRICHER_VERSION,
    )

    outcome = enricher.enrich(post)

    if not outcome.succeeded:
        return outcome, broken

    payload = post.model_dump(mode="json")

    # The fingerprint is excluded from the model on purpose, so it is
    # written alongside. It is what lets a later run skip this post, and
    # it is build metadata rather than content, so the aggregator does
    # not read it.
    fingerprint: dict = {
        "source_digest": digest,
        "enricher_version": ENRICHER_VERSION,
    }

    if outcome.grounding:
        # Kept beside the fingerprint so what grounding removed can be
        # audited, and is not part of the knowledge a reader consumes.
        fingerprint["grounding"] = outcome.grounding

    payload["_enrichment"] = fingerprint

    target = RESULTS_DIR / f"cloud_worker_{identifier}.json"

    target.parent.mkdir(parents=True, exist_ok=True)

    _write_atomic(
        target, json.dumps(payload, indent=2, ensure_ascii=False)
    )

    return outcome, broken


def _write_atomic(target: Path, body: str) -> None:
    """
    Replace a result file without a reader ever seeing half of one.

    The knowledge base is built by reading these, and a truncated file
    there is not a post that fails loudly -- it is a post that quietly
    stops appearing in the site.

    Retried because Windows refuses a replace while anything holds the
    target open, which an indexer following a five-hundred-file write
    reliably does at least once.
    """

    temporary = target.with_suffix(target.suffix + ".tmp")

    for attempt in range(6):
        try:
            temporary.write_text(body, encoding="utf-8")
            temporary.replace(target)
            return

        except PermissionError:
            if attempt == 5:
                raise

            time.sleep(min(0.1 * (2**attempt), 2.0))


def _unloadable(identifier: str, exc: Exception) -> Outcome:
    """
    A post that failed before, or outside, the model.

    Classified rather than assumed: a post directory that has not
    appeared yet is worth one more run, and a permission error is not.
    """

    verdict = classify(exc)

    return Outcome(
        post_id=identifier,
        status="failed",
        attempts=[
            Attempt(
                number=1,
                outcome="failed",
                kind=verdict.kind.value,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        ],
    )


def _enricher_pool(attempts: int):
    """
    One retrying enricher per thread.

    The client is stateless, so sharing it would be fine. The enricher
    is not: it records the last grounding report so the pipeline can
    store what was removed, and two threads sharing one would let a post
    file another post's removals. One per thread costs a cheap object
    and removes the question entirely.
    """

    local = threading.local()

    def get() -> Enricher:
        if not hasattr(local, "enricher"):
            local.enricher = Enricher(
                factory=AIEnricher, attempts=attempts
            )

        return local.enricher

    return get


def _result(
    state: RunState,
    report: RunReport,
    written: list[str],
    media_failures: dict[str, list[str]],
) -> dict:
    """The same shape the pipeline returned before, plus the new detail."""

    return {
        "written": sorted(written),
        "reused": sorted(
            record.post_id
            for record in state.posts.values()
            if record.status == "cached"
        ),
        "failed": {
            record.post_id: record.error_message
            for record in state.failures()
        },
        "media_failures": dict(sorted(media_failures.items())),
        "report": report.as_dict(),
        "counts": state.counts(),
    }


__all__ = ["Settled", "enrich"]