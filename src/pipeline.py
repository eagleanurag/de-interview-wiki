"""
Run the whole pipeline locally, in order.

Discover → enrich → aggregate → generate the site → verify.

Each stage writes its own output and the next stage reads only that, so
a stage cannot quietly depend on something an earlier stage held in
memory. A failure in one post costs that post, not the run, because
enrichment is the only stage that touches untrusted model output.

Outputs go under ``build/``, which is git-ignored, so nothing here
writes into the committed repository.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

from src.aggregation.aggregator import aggregate_results
from src.ai.enricher import AIEnricher
from src.ingestion.importer import discover_posts, validate_posts
from src.ingestion.post_loader import load_post, source_digest
from src.processing.media_processor import MediaReport, process_media
from src.wiki.generator import generate_site

BUILD_ROOT = Path("build")
RESULTS_DIR = BUILD_ROOT / "worker-results"
KNOWLEDGE_BASE = BUILD_ROOT / "knowledge_base.json"
SITE_DIR = BUILD_ROOT / "site"

#: Bumped when the enrichment contract or the prompt changes. A result
#: produced under a different version is not reusable, because the
#: current version would answer differently about the same content.
#:
#: Version 3 added source grounding, so a result from version 2 may
#: contain a question about a technology the source never mentioned.
#: Those results are not reusable even though the underlying analysis is
#: largely right: the point of the check is that the knowledge base never
#: holds one, and a cached result would put it straight back.
ENRICHER_VERSION = "3"


class StageError(RuntimeError):
    """Raised when a stage cannot complete."""


def log(message: str) -> None:
    print(f"[pipeline] {message}", flush=True)


def discover() -> list[str]:
    """Every post the repository holds, validated first."""

    report = validate_posts()

    if not report.ok:
        problems = "; ".join(str(issue) for issue in report.issues)

        raise StageError(f"Posts do not validate: {problems}")

    warnings = [
        str(issue) for issue in report.issues if "warning" in str(issue).lower()
    ]

    if warnings:
        for warning in warnings:
            log(f"  warning: {warning}")

    identifiers = [
        summary.post_id for summary in discover_posts()
    ]

    log(f"discovered {len(identifiers)} post(s)")

    return identifiers


def enrich(
    identifiers: list[str],
    *,
    force: bool,
    jobs: int = 1,
) -> dict:
    """
    Enrich every post that needs it, into its own worker result.

    One post failing costs that post. The source is never modified and
    never deleted, so a failed post can simply be retried on the next
    run, which is what makes the pipeline resumable.

    Enrichment is incremental. A post whose content has not changed and
    whose enrichment was produced by the current version keeps the
    result it has, because calling the model again would spend real
    time to arrive at the same answer. ``force`` re-enriches
    everything regardless, which is what a change to the prompt needs.

    ``jobs`` runs several posts at once. Each post is already its own
    worker result and CI already runs one job per post, so the work is
    independent by design and this is the same fan-out applied locally
    rather than a new pipeline. It is off by default: a single post is
    the common case, and a bound nobody asked for is a surprise. Each
    worker gets its own enricher, because the enricher records the last
    grounding report and a shared one would let one post read another's.
    """

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pool = _enricher_pool()

    def one(identifier: str) -> dict:
        return _enrich_one(identifier, pool, force=force)

    if jobs > 1:
        from concurrent.futures import ThreadPoolExecutor

        # map yields in submission order, so the report and the log are
        # the same whether one post ran at a time or eight.
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            results = list(executor.map(one, identifiers))
    else:
        results = [one(identifier) for identifier in identifiers]

    written: list[str] = []
    reused: list[str] = []
    failed: dict[str, str] = {}
    media_failures: dict[str, list[str]] = {}

    for result in results:
        for line in result["lines"]:
            log(line)

        if result["failed"] is not None:
            failed[result["identifier"]] = result["failed"]
            continue

        if result["reused"]:
            reused.append(result["identifier"])
        else:
            written.append(result["identifier"])

        if result["broken"]:
            media_failures[result["identifier"]] = result["broken"]

    log(
        f"enrichment: {len(written)} written, {len(reused)} reused, "
        f"{len(failed)} failed"
    )

    if failed:
        # Reported rather than raised, so one bad post does not cost
        # the run. The caller can decide whether the failure matters.
        log(f"  {len(failed)} post(s) need a retry")

    return {
        "written": written,
        "reused": reused,
        "failed": failed,
        "media_failures": media_failures,
    }


def _enricher_pool():
    """
    One enricher per thread.

    The client is stateless, so sharing it would be fine. The enricher
    is not: it records the last grounding report so the pipeline can
    store what was removed, and two threads sharing one would let a post
    file another post's removals. Building one per thread costs a cheap
    object and removes the question.
    """
    import threading

    local = threading.local()

    def get():
        if not hasattr(local, "enricher"):
            local.enricher = AIEnricher()

        return local.enricher

    return get


def _enrich_one(identifier: str, enricher_for, *, force: bool) -> dict:
    """
    Enrich one post, or reuse the result already on file.

    Returns everything the caller needs to report, including the log
    lines, so that the lines are emitted in a fixed order however the
    work was scheduled.
    """

    directory = Path("data/posts") / identifier
    target = RESULTS_DIR / f"cloud_worker_{identifier}.json"

    started = time.monotonic()

    result: dict = {
        "identifier": identifier,
        "written": False,
        "reused": False,
        "failed": None,
        "broken": [],
        "lines": [],
    }

    enricher = enricher_for()

    try:
        post = load_post(directory)

        media_report = MediaReport()

        process_media(post, report=media_report)

        digest = source_digest(post)

        post.enrichment = type(post.enrichment)(
            source_digest=digest,
            enricher_version=ENRICHER_VERSION,
        )

        if target.is_file() and not force:
            existing = _reusable(target, digest)

            if existing is not None:
                # The fingerprint covers the content, so a cached
                # result is reused even after the post gains
                # provenance it did not have. The analysis is still
                # correct; the attribution around it is not, and a
                # knowledge base that kept the stale copy would
                # publish a post that says less than the source
                # does. Refreshing it costs no model call.
                _refresh_provenance(existing, post)

                target.write_text(
                    json.dumps(existing, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

                result["reused"] = True

                return result

        enriched = enricher.enrich(post)

        payload = enriched.model_dump(mode="json")

        # The fingerprint is excluded from the model on purpose, so
        # it is written alongside. It is what lets a later run skip
        # this post, and it is build metadata rather than content,
        # so the aggregator does not read it.
        fingerprint = {
            "source_digest": digest,
            "enricher_version": ENRICHER_VERSION,
        }

        grounding_report = getattr(enricher, "last_grounding", None)

        if grounding_report is not None and not grounding_report.clean:
            # Kept beside the fingerprint for the same reason: what
            # the pipeline removed is worth being able to audit, and
            # is not part of the knowledge a reader consumes.
            fingerprint["grounding"] = grounding_report.as_dict()

            result["lines"].append(
                f"  {identifier}: {grounding_report.summary()}"
            )

        payload["_enrichment"] = fingerprint

        target.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    except Exception as exc:  # noqa: BLE001
        result["failed"] = str(exc)
        result["lines"].append(f"  FAILED {identifier}: {exc}")

        return result

    result["written"] = True
    result["broken"] = [item.path for item in media_report.failed]

    elapsed = time.monotonic() - started

    result["lines"].append(
        f"  enriched {identifier} "
        f"({len(enriched.interview_questions)} question(s), "
        f"relevant={enriched.classification.interview_relevant}, "
        f"{elapsed:.1f}s)"
        + (
            f" media failed: {result['broken']}"
            if result["broken"]
            else ""
        )
    )

    return result


def _refresh_provenance(
    payload: dict,
    post: object,
) -> dict:
    """
    Copy the current post's attribution onto a reused worker result.

    Only the fields that say where the content came from and what it
    looks like are replaced. The analysis, the questions and the
    classification are left exactly as they were, because they are what
    the model was paid for and nothing about them has gone stale.

    A field that is absent from the result is removed rather than left,
    so a post that has stopped claiming to have been captured stops
    claiming it.
    """
    current = post.model_dump(mode="json")

    for key in ("source", "media"):
        value = current.get(key)

        if value is None:
            payload.pop(key, None)
        else:
            payload[key] = value

    if current.get("saved_item") is None:
        payload.pop("saved_item", None)
    else:
        payload["saved_item"] = current["saved_item"]

    payload["original_text"] = current.get("original_text", "")

    return payload


def _reusable(target: Path, digest: str) -> dict | None:
    """
    An existing worker result that still matches the current content.

    Read from the file rather than trusted from a stamp, because the
    stamp lives in the post and the result lives here, and the two can
    disagree if a run was interrupted between them.
    """

    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None

    recorded = payload.get("_enrichment") or {}

    if not isinstance(recorded, dict):
        return None

    if recorded.get("source_digest") != digest:
        return None

    if recorded.get("enricher_version") != ENRICHER_VERSION:
        return None

    return payload


def aggregate() -> Path:
    """Build the canonical knowledge base from the worker results."""

    if not any(RESULTS_DIR.glob("cloud_worker_*.json")):
        raise StageError(
            f"No worker results under {RESULTS_DIR}; nothing to aggregate."
        )

    log("aggregating")

    aggregate_results(
        input_directory=RESULTS_DIR,
        output_path=KNOWLEDGE_BASE,
    )

    payload = json.loads(KNOWLEDGE_BASE.read_text(encoding="utf-8"))
    stats = payload["stats"]

    log(
        f"  posts={stats['posts_aggregated']} "
        f"topics={stats.get('topics_consolidated', 0)} "
        f"concepts={stats.get('concepts_consolidated', 0)} "
        f"technologies={stats.get('technologies_consolidated', 0)} "
        f"questions={stats.get('questions_consolidated', 0)}"
    )

    return KNOWLEDGE_BASE


def build_site() -> Path:
    """Generate the static site from the canonical knowledge base."""

    if not KNOWLEDGE_BASE.is_file():
        raise StageError(
            f"{KNOWLEDGE_BASE} does not exist; aggregate first."
        )

    log("generating the site")

    if SITE_DIR.exists():
        # A stale file from a removed post would otherwise survive,
        # so the output is rebuilt from empty.
        shutil.rmtree(SITE_DIR)

    generate_site(input_path=KNOWLEDGE_BASE, output_dir=SITE_DIR)

    pages = list(SITE_DIR.rglob("*.html"))

    log(f"  {len(pages)} page(s) in {SITE_DIR}")

    return SITE_DIR


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the knowledge pipeline locally, end to end."
    )

    parser.add_argument(
        "--force-enrich",
        action="store_true",
        help=(
            "Re-enrich posts that already have a worker result. Without "
            "this an existing result is reused, so a change to the "
            "enrichment code would have no effect."
        ),
    )

    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help=(
            "Enrich this many posts at once. Each post is already its "
            "own worker result and CI already runs one job per post, so "
            "the work is independent; this applies the same fan-out "
            "locally. One is the default, because a bound nobody asked "
            "for is a surprise."
        ),
    )

    parser.add_argument(
        "--only",
        choices=["discover", "enrich", "aggregate", "site"],
        help="Run one stage instead of the whole pipeline.",
    )

    args = parser.parse_args()

    if args.jobs < 1:
        print(
            "PIPELINE_ERROR=--jobs must be at least 1",
            file=sys.stderr,
        )
        return 1

    try:
        if args.only == "discover":
            discover()
            return 0

        if args.only == "enrich":
            enrich(discover(), force=args.force_enrich, jobs=args.jobs)
            return 0

        if args.only == "aggregate":
            aggregate()
            return 0

        if args.only == "site":
            build_site()
            return 0

        identifiers = discover()
        enrich(identifiers, force=args.force_enrich, jobs=args.jobs)
        aggregate()
        build_site()

    except StageError as exc:
        print(f"PIPELINE_ERROR={exc}", file=sys.stderr)
        return 1

    log("done")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
