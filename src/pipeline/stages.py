"""
The stages either side of enrichment.

Discover, aggregate and site generation. None of them talks to a model:
each reads what an earlier stage wrote and nothing else, so a stage
cannot quietly depend on something an earlier one held in memory, and a
run can be repeated from any stage without repeating the ones before it.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.aggregation.aggregator import aggregate_results
from src.ingestion.importer import discover_posts, validate_posts
from src.pipeline.paths import (
    KNOWLEDGE_BASE,
    RESULTS_DIR,
    SITE_DIR,
    StageError,
    log,
)
from src.wiki.generator import generate_site


def discover() -> list[str]:
    """Every post the repository holds, validated first."""

    report = validate_posts()

    if not report.ok:
        problems = "; ".join(
            str(issue) for issue in report.issues
        )

        raise StageError(f"Posts do not validate: {problems}")

    warnings = [
        str(issue)
        for issue in report.issues
        if "warning" in str(issue).lower()
    ]

    for warning in warnings:
        log(f"  warning: {warning}")

    identifiers = [
        summary.post_id for summary in discover_posts()
    ]

    log(f"discovered {len(identifiers)} post(s)")

    return identifiers


def aggregate() -> Path:
    """
    Build the canonical knowledge base from the worker results.

    Only completed posts contribute, which is the honest reading of a
    partially-successful run: the site shows what was actually produced
    rather than a placeholder for what was not. The run state says which
    posts are missing so that gap is never silent.
    """

    if not any(RESULTS_DIR.glob("cloud_worker_*.json")):
        raise StageError(
            f"No worker results under {RESULTS_DIR}; "
            f"nothing to aggregate."
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
        # A stale file from a removed post would otherwise survive, so
        # the output is rebuilt from empty.
        shutil.rmtree(SITE_DIR)

    generate_site(input_path=KNOWLEDGE_BASE, output_dir=SITE_DIR)

    pages = list(SITE_DIR.rglob("*.html"))

    log(f"  {len(pages)} page(s) in {SITE_DIR}")

    return SITE_DIR


__all__ = ["aggregate", "build_site", "discover"]