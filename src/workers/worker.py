from __future__ import annotations

import json
from pathlib import Path

from src.ai.enricher import AIEnricher
from src.ingestion.post_loader import load_post
from src.processing.media_processor import process_media


class WorkerError(RuntimeError):
    """Raised when a worker job cannot be completed."""


def process_enrichment_job(
    post_directory: str | Path,
    output_path: str | Path,
) -> Path:
    """
    Execute one complete post-enrichment job.

    This is intentionally independent of GitHub Actions or any
    particular cloud provider. A cloud runner can invoke this same
    function later.
    """

    post_directory = Path(post_directory)
    output_path = Path(output_path)

    try:
        print(f"[worker] Loading post: {post_directory}")

        post = load_post(post_directory)

        print(
            f"[worker] Discovered {len(post.media)} media item(s)"
        )

        print("[worker] Processing media...")
        process_media(post)

        print("[worker] Running AI enrichment...")
        enricher = AIEnricher()
        enriched_post = enricher.enrich(post)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            json.dumps(
                enriched_post.model_dump(mode="json"),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print(f"[worker] Result written: {output_path}")

        return output_path

    except Exception as exc:
        raise WorkerError(
            f"Enrichment job failed for {post_directory}: {exc}"
        ) from exc