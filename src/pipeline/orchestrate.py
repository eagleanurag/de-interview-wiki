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
from src.aggregation.consolidation import _question_key
from src.enrichment.state import RunState
from src.gemini.bridge import ocr_digest as digest_ocr
from src.gemini.bridge import inject as inject_ocr
from src.gemini.bridge import load_index
from src.ingestion.post_loader import load_post, source_digest
from src.models import InterviewQuestion, KnowledgePost
from src.pipeline.freshness import (
    ENRICHER_VERSION,
    OCR_PROCESSOR_VERSION,
    VISUAL_PROCESSOR_VERSION,
    _refresh_provenance,
    _reusable,
)
from src.pipeline.paths import RESULTS_DIR, RUN_STATE, log
from src.pipeline.visual import digest_for, inject
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

    #: Transcribed images the package described that this post does not
    #: carry, so were not attached. Reported rather than dropped: they are
    #: still in the import, and a reader told "3,047 transcriptions"
    #: should not have to guess how many reached a post.
    ocr_unmatched: int = 0


def enrich(
    identifiers: list[str],
    *,
    force: bool,
    jobs: int = 1,
    attempts: int = DEFAULT_ATTEMPTS,
    state_path: Path | None = None,
    quiet: bool = False,
    visual: dict | None = None,
    ocr=None,
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

    ``visual`` carries the visual stage's results, keyed by post
    identifier. Consulted twice: for freshness, so that slides which
    changed re-enrich their post, and for the source text, so that what
    a slide says is treated as source material by the same grounding
    check the post body goes through.

    ``ocr`` is the imported image-knowledge package's index. Absent, or
    empty, this is exactly the run CP12 left behind: no package has been
    imported, and a post the package says nothing about is untouched.
    Present, it contributes machine transcriptions as source material and
    its own freshness digest, so a corrected transcription re-enriches the
    posts it informs instead of being silently ignored.
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

    visual = visual or {}

    if ocr is None:
        # Loaded here rather than required from the caller, so that every
        # existing call site -- the CLI, the tests, the agent -- behaves
        # identically whether or not anyone thought about this.
        ocr = load_index()

    pending = _settle_cached(
        identifiers, state, report, positions, force, quiet, visual, ocr
    )

    written: list[str] = []
    media_failures: dict[str, list[str]] = {}

    #: Transcribed images with no post to attach to, keyed by post.
    #: Counted rather than dropped; see :func:`_enrich_one`.
    ocr_unmatched: dict[str, int] = {}

    if pending:
        if not quiet:
            log(
                f"enriching {len(pending)} post(s), up to "
                f"{attempts} attempt(s) each, {jobs} at a time"
            )

        pool = _enricher_pool(attempts)

        written, media_failures, ocr_unmatched = _run_pending(
            pool, pending, jobs, state, report, positions, path, quiet,
            visual, ocr,
        )

        if ocr_unmatched and not quiet:
            log(
                f"{sum(ocr_unmatched.values())} transcribed image(s) "
                f"across {len(ocr_unmatched)} post(s) are described by "
                "the package but not carried by the post, and were not "
                "attached. They remain in the import."
            )

    report.seconds = time.monotonic() - started

    state.save(path)

    if not quiet:
        log(report.render())

    return _result(
        state, report, written, media_failures, ocr_unmatched
    )


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
    visual: dict | None = None,
    ocr=None,
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

    ``visual`` holds the visual stage's results. A post whose slides
    were read, reordered or replaced is re-enriched even though its text
    and its file bytes are untouched, because what the enricher would be
    told about it has changed.
    """

    visual = visual or {}

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

        visual_digest = digest_for(post, visual.get(identifier))

        ocr_digest = ocr.digest_for(post) if ocr is not None else ""

        if target.is_file() and not force:
            existing = _reusable(
                target, digest, visual_digest, ocr_digest
            )

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
    visual: dict | None = None,
    ocr=None,
) -> tuple[list[str], dict[str, list[str]], dict[str, int]]:
    """
    Run the posts, bounded, settling each as it finishes.

    Settled on completion rather than in batch, because a run over
    several hundred posts that says nothing for an hour is
    indistinguishable from a hung one. The index on the line is the
    post's fixed position in the batch, so the numbering is stable even
    though arrival is not.

    ``ocr`` is shared across every worker thread. It holds one
    pre-built record index and a per-post cache of outcomes, and the
    outcome for a post is a pure function of its records, so two threads
    asking about the same post build equal objects and one of them
    discards it. Nothing in it is mutated after construction except that
    cache, and a duplicate key written twice is the same value twice.
    """

    visual = visual or {}

    def work(item: tuple[str, Path]) -> Settled:
        identifier, directory = item

        outcome, broken, unmatched = _enrich_one(
            identifier, directory, pool, visual.get(identifier), ocr
        )

        return Settled(
            outcome=outcome,
            broken_media=broken,
            ocr_unmatched=unmatched,
        )

    written: list[str] = []
    media_failures: dict[str, list[str]] = {}

    #: Transcribed images with no post to attach to, keyed by post.
    #: Counted rather than dropped; see :func:`_enrich_one`.
    ocr_unmatched: dict[str, int] = {}

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

        if settled.ocr_unmatched:
            ocr_unmatched[outcome.post_id] = settled.ocr_unmatched

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

    return written, media_failures, ocr_unmatched


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
    visual=None,
    ocr=None,
) -> tuple[Outcome, list[str], int]:
    """
    Enrich one post, re-asking what a re-ask can fix.

    Returns the outcome, the media that could not be read, and how many
    transcribed images the package described that this post does not
    carry. A file that will not open is not a reason to lose the post --
    the text is still worth enriching -- but it is reported rather than
    dropped, because "the image was skipped" and "the image was read" are
    different claims. The third figure is the same argument applied to
    images the archive holds and no post references: they are not
    attached, and the count says so rather than leaving a reader to
    assume every transcription reached a post.

    The post is loaded once and the retry loop re-uses it, because a
    retry is a fresh model sample rather than a fresh post. The one
    thing the loop must not carry forward is a half-applied analysis,
    and :meth:`AIEnricher.enrich` raises before writing to the post
    when validation fails, so a failed attempt leaves the post as it
    was loaded.

    ``ocr`` is the imported image-knowledge package's index. It is
    consulted *after* ``visual`` and only for text the vision stage did
    not produce, so a slide is never described twice from two sources and
    the better-attested reading wins. Both write to the same
    ``media.extracted_text`` field, which is why the order matters: the
    later injection overwrites, and it must be the fallback that lands
    last, not the other way round.
    """

    enricher = pool()

    post = load_post(directory)

    media_report = MediaReport()

    process_media(post, report=media_report)

    broken = [item.path for item in media_report.failed]

    digest = source_digest(post, directory)

    # Before the digest is taken, so that what the slides contribute to
    # the prompt is already on the post the enricher will see.
    if visual is not None:
        inject(post, visual)

    visual_digest = digest_for(post, visual)

    ocr_digest = ""

    unmatched = 0

    if ocr is not None:
        outcome_ocr = ocr.outcome_for(post)

        if outcome_ocr is not None:
            inject_ocr(post, outcome_ocr, ocr.records_for(post))

            derived = ocr.technologies_for(post)

            if derived:
                # On the post rather than passed to the aggregator
                # separately, so that the knowledge base and the site
                # read the same field and cannot disagree about which
                # posts cover a technology.
                post.ai_analysis.derived_technologies = derived

            _add_ocr_questions(post, ocr.questions_for(post))

            ocr_digest = digest_ocr(outcome_ocr)

            unmatched = outcome_ocr.unmatched

    post.enrichment = type(post.enrichment)(
        source_digest=digest,
        enricher_version=ENRICHER_VERSION,
        visual_digest=visual_digest,
        visual_processor_version=(
            VISUAL_PROCESSOR_VERSION if visual_digest else ""
        ),
        ocr_digest=ocr_digest,
        ocr_processor_version=(OCR_PROCESSOR_VERSION if ocr_digest else ""),
    )

    outcome = enricher.enrich(post)

    if not outcome.succeeded:
        return outcome, broken, unmatched

    payload = post.model_dump(mode="json")

    # The fingerprint is excluded from the model on purpose, so it is
    # written alongside. It is what lets a later run skip this post, and
    # it is build metadata rather than content, so the aggregator does
    # not read it.
    fingerprint: dict = {
        "source_digest": digest,
        "enricher_version": ENRICHER_VERSION,
        "visual_digest": visual_digest,
        "visual_processor_version": (
            VISUAL_PROCESSOR_VERSION if visual_digest else ""
        ),
        "ocr_digest": ocr_digest,
        "ocr_processor_version": (
            OCR_PROCESSOR_VERSION if ocr_digest else ""
        ),
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

    return outcome, broken, unmatched


def _add_ocr_questions(post: KnowledgePost, candidates) -> None:
    """
    Put the package's accepted questions onto the post.

    Added rather than replacing, and only where the post has no question
    saying the same thing. The existing dedup in the aggregator is the
    authority on whether two questions are the same question, and running
    it here would mean a second and weaker implementation of the same
    comparison running earlier and disagreeing with it.

    Every candidate carries a ``source_excerpt`` that is the slide's own
    transcription, and that is what gets recorded as the answer: it is
    the text the question was read from, attributed as such, rather than
    an answer this project composed and might have got wrong.
    """

    existing = {
        _question_key(question.question)
        for question in post.interview_questions
    }

    for candidate in candidates:
        if not candidate.question:
            continue

        key = _question_key(candidate.question)

        if key in existing:
            continue

        existing.add(key)

        post.interview_questions.append(
            InterviewQuestion(
                question=candidate.question,
                type="theory",
                difficulty="medium",
                answer=_source_answer(candidate),
                source_kind=candidate.source_kind.value,
                answer_source="source_excerpt",
                source_note=(
                    f"Read from the transcribed text of "
                    f"{candidate.filename}, slide "
                    f"{candidate.slide_number}."
                    if candidate.filename
                    else (
                        "Read from the transcribed text of a slide "
                        "image."
                    )
                ),
            )
        )


def _source_answer(candidate) -> str:
    """
    The answer a source-derived question gets.

    The slide's own transcription, introduced as an excerpt. Presenting
    OCR as an answer is the failure this avoids: the excerpt may not even
    answer the question, and a reader who cannot tell an excerpt from an
    answer will believe it does.
    """

    import re

    excerpt = re.sub(
        r"\s+", " ", candidate.source_excerpt or ""
    ).strip()

    if not excerpt:
        return (
            "No source text accompanies this question. It was read from "
            "a slide image, and the transcription was too damaged to "
            "recover a passage from it."
        )

    return (
        "Source excerpt, transcribed from the slide image rather than "
        f"written as an answer:\n\n{excerpt}"
    )


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
    ocr_unmatched: dict[str, int] | None = None,
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