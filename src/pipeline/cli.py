"""
The command line.

Thin on purpose. It parses, it calls the stages in order, and it turns
a stage failure into an exit code. Everything a stage decides lives in
the stage, so the behaviour a person gets from
``python -m src.pipeline`` is the behaviour the tests exercise.
"""

from __future__ import annotations

import argparse
import sys

from src.enrichment.runner import DEFAULT_ATTEMPTS
from src.pipeline.orchestrate import enrich
from src.pipeline.paths import StageError, log
from src.pipeline.stages import aggregate, build_site, discover


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline",
        description="Run the knowledge pipeline locally, end to end.",
        epilog=(
            "Enrichment runs locally against OpenCode and is retried "
            "per post; it is the primary way this project processes "
            "material. GitHub Actions validates and deploys."
        ),
    )

    parser.add_argument(
        "--force-enrich",
        action="store_true",
        help=(
            "Re-enrich posts that already have a worker result. "
            "Without this an existing result is reused, so a change to "
            "the enrichment code would have no effect."
        ),
    )

    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help=(
            "Enrich this many posts at once. Each post is independent "
            "-- its own directory, its own result file, its own model "
            "call -- so this is a bound on concurrency rather than a "
            "second pipeline. One is the default, because a bound "
            "nobody asked for is a surprise."
        ),
    )

    parser.add_argument(
        "--attempts",
        type=int,
        default=DEFAULT_ATTEMPTS,
        help=(
            "How many times to try one post before writing it off. "
            "Only failures a retry can fix are retried: a truncated or "
            "throttled response is asked again, a missing API key is "
            f"not. Default {DEFAULT_ATTEMPTS}."
        ),
    )

    parser.add_argument(
        "--only",
        choices=["discover", "enrich", "aggregate", "site"],
        help="Run one stage instead of the whole pipeline.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.jobs < 1:
        print(
            "PIPELINE_ERROR=--jobs must be at least 1",
            file=sys.stderr,
        )
        return 1

    if args.attempts < 1:
        print(
            "PIPELINE_ERROR=--attempts must be at least 1",
            file=sys.stderr,
        )
        return 1

    try:
        if args.only == "discover":
            discover()
            return 0

        if args.only == "enrich":
            enrich(
                discover(),
                force=args.force_enrich,
                jobs=args.jobs,
                attempts=args.attempts,
            )
            return 0

        if args.only == "aggregate":
            aggregate()
            return 0

        if args.only == "site":
            build_site()
            return 0

        enrich(
            discover(),
            force=args.force_enrich,
            jobs=args.jobs,
            attempts=args.attempts,
        )

        aggregate()
        build_site()

    except StageError as exc:
        print(f"PIPELINE_ERROR={exc}", file=sys.stderr)
        return 1

    log("done")

    return 0


__all__ = ["build_parser", "main"]