"""
Enriching one post, addressed by a job file.

This is the shape the GitHub worker workflow used: a job file names a
post directory and an output path, and something runs it. It stays,
because it is a perfectly good way to address one piece of work and
because a job file is a durable record of what was asked for.

What it no longer does is own the retry. It calls
:class:`~src.enrichment.runner.Enricher`, which is the same object the
local pipeline calls, so a post gets the same bounded retry whichever
way it was invoked. When this module had its own single attempt, a
truncated response was a lost post -- which is exactly what six of the
twenty batches in the GitHub run turned out to be.

The workflow no longer fans the model out; see the README. This module
is kept because "run one post from a job file" is useful on its own and
because deleting it would have deleted the only caller of nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.ai.enricher import AIEnricher
from src.enrichment.runner import DEFAULT_ATTEMPTS, Enricher
from src.ingestion.post_loader import load_post, source_digest
from src.processing.media_processor import process_media


class WorkerError(RuntimeError):
    """Raised when a worker job cannot be completed."""


def process_enrichment_job(
    post_directory: str | Path,
    output_path: str | Path,
    *,
    attempts: int = DEFAULT_ATTEMPTS,
) -> Path:
    """
    Execute one complete post-enrichment job.

    Deliberately independent of GitHub Actions or any particular cloud
    provider: this is one post and one output file. The retry lives in
    the shared runner rather than here, so a caller of this function and
    a caller of the local pipeline behave identically under a truncated
    provider response.
    """

    post_directory = Path(post_directory)
    output_path = Path(output_path)

    try:
        print(f"[worker] Loading post: {post_directory}")

        post = load_post(post_directory)

        print(f"[worker] Discovered {len(post.media)} media item(s)")
        print("[worker] Processing media...")

        process_media(post)

        digest = source_digest(post, post_directory)

        post.enrichment = type(post.enrichment)(
            source_digest=digest,
            enricher_version=_enricher_version(),
        )

        print(
            f"[worker] Running AI enrichment "
            f"(up to {attempts} attempt(s))..."
        )

        outcome = Enricher(
            factory=AIEnricher, attempts=attempts
        ).enrich(post)

        if not outcome.succeeded:
            last = outcome.attempts[-1]

            raise WorkerError(
                f"{outcome.post_id}: gave up after "
                f"{len(outcome.attempts)} attempt(s). "
                f"Last failure was {last.kind}: "
                f"{last.error_message[:300]}"
            )

        for attempt in outcome.attempts[:-1]:
            print(
                f"[worker] attempt {attempt.number} failed "
                f"({attempt.kind}), retried"
            )

        print(
            f"[worker] Enriched on attempt "
            f"{len(outcome.attempts)}"
        )

        payload = post.model_dump(mode="json")

        fingerprint: dict = {
            "source_digest": digest,
            "enricher_version": _enricher_version(),
        }

        if outcome.grounding:
            fingerprint["grounding"] = outcome.grounding

        payload["_enrichment"] = fingerprint

        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(f"[worker] Result written: {output_path}")

        return output_path

    except WorkerError:
        raise

    except Exception as exc:
        raise WorkerError(
            f"Enrichment job failed for {post_directory}: {exc}"
        ) from exc


def _enricher_version() -> str:
    """
    The current enricher contract, read from the pipeline.

    Imported lazily because the worker is the one place that must work
    without the pipeline's import graph being loaded, and a hard import
    would make that a lie.
    """

    from src.pipeline.freshness import ENRICHER_VERSION

    return ENRICHER_VERSION