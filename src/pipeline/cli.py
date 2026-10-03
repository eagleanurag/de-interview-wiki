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
from src.pipeline.visual_stage import make_visual


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
        choices=[
            "discover",
            "visual-plan",
            "visual",
            "enrich",
            "aggregate",
            "site",
        ],
        help="Run one stage instead of the whole pipeline.",
    )

    parser.add_argument(
        "--archive",
        default=None,
        help=(
            "Root of the saved-posts archive. Read-only; this "
            "pipeline never writes to it. Defaults to "
            "$LINKEDIN_ARCHIVE_ROOT."
        ),
    )

    parser.add_argument(
        "--visual-jobs",
        type=int,
        default=1,
        help=(
            "Posts whose images to read at once. One by default: "
            "a post's slides are read in order, and raising this is "
            "worth measuring rather than assuming."
        ),
    )

    parser.add_argument(
        "--visual-batch",
        type=int,
        default=5,
        help=(
            "Slides per model call. Five costs about what one "
            "costs, so this bounds what a single failure loses "
            "rather than making the run faster."
        ),
    )

    parser.add_argument(
        "--visual-limit",
        type=int,
        default=None,
        help=(
            "Read images for at most this many posts, for "
            "measuring a change on part of the corpus."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Report what would be done and stop. Writes nothing: "
            "not the results, not the knowledge base, not the "
            "site, and not the visual cache."
        ),
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

    if args.visual_jobs < 1:
        print(
            "PIPELINE_ERROR=--visual-jobs must be at least 1",
            file=sys.stderr,
        )
        return 1

    if args.visual_batch < 1:
        print(
            "PIPELINE_ERROR=--visual-batch must be at least 1",
            file=sys.stderr,
        )
        return 1

    try:
        if args.only == "discover":
            discover()
            return 0

        if args.only == "visual-plan":
            return _visual_plan(args)

        if args.only == "visual":
            return _visual_only(args)

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

        if args.dry_run:
            _run_visual(args, discover())
            log("dry run: nothing was written")

            return 0

        visual = _run_visual(args, discover())

        enrich(
            discover(),
            force=args.force_enrich,
            jobs=args.jobs,
            attempts=args.attempts,
            visual=visual,
        )

        aggregate()
        build_site()

    except StageError as exc:
        print(f"PIPELINE_ERROR={exc}", file=sys.stderr)
        return 1

    log("done")

    return 0



# ---------------------------------------------------------------------
# Visual stages
# ---------------------------------------------------------------------


def _archive_root(args) -> str | None:
    """
    Where the archive is, or None when it has not been configured.

    Reported rather than guessed. A default path that does not exist
    turns a missing configuration into a confusing file-not-found deep
    inside the stage, and this stage reads a machine-specific directory
    that CI has no reason to have.
    """

    if args.archive:
        return args.archive

    import os

    return os.environ.get("LINKEDIN_ARCHIVE_ROOT")


def _visual_plan(args) -> int:
    """What reading the images would cost. Writes nothing."""

    root = _archive_root(args)

    if not root:
        print(
            "PIPELINE_ERROR=no archive configured; pass --archive or set "
            "LINKEDIN_ARCHIVE_ROOT",
            file=sys.stderr,
        )
        return 1

    runner = make_visual(root, batch_size=args.visual_batch)

    plan = runner.plan()

    if plan.archive_error:
        print(f"PIPELINE_ERROR={plan.archive_error}", file=sys.stderr)
        return 1

    log(plan.render())

    return 0


def _run_visual(args, identifiers: list[str]) -> dict:
    """
    Read the images of the posts that have any.

    Returns the results keyed by committed post identifier, which is what
    the enricher consults. An unconfigured archive is not an error: most
    posts have no images, and a run that cannot see the archive should
    enrich the corpus as it stands rather than refuse to.
    """

    root = _archive_root(args)

    if not root:
        log(
            "no archive configured; skipping the visual stage "
            "(pass --archive or set LINKEDIN_ARCHIVE_ROOT)"
        )
        return {}

    from pathlib import Path

    from src.ingestion.post_loader import load_post

    posts = []

    for identifier in identifiers:
        try:
            posts.append(load_post(Path("data/posts") / identifier))

        except Exception as exc:  # noqa: BLE001
            log(f"  could not load {identifier}: {exc}")

    runner = make_visual(
        root,
        batch_size=args.visual_batch,
        attempts=args.attempts,
    )

    run = runner.run(
        posts,
        jobs=args.visual_jobs,
        limit=args.visual_limit,
    )

    log(run.render())

    return run.outcomes


def _visual_only(args) -> int:
    """
    Run the visual stage and stop.

    Useful on its own: it fills the cache without spending a text
    enrichment call, so a later `--only enrich` reuses it.
    """

    root = _archive_root(args)

    if not root:
        print(
            "PIPELINE_ERROR=no archive configured; pass --archive or set "
            "LINKEDIN_ARCHIVE_ROOT",
            file=sys.stderr,
        )
        return 1

    if args.dry_run:
        return _visual_plan(args)

    _run_visual(args, discover())

    return 0


__all__ = ["build_parser", "main"]