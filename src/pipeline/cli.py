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
            "gemini-plan",
            "gemini-import",
            "visual-plan",
            "visual",
            "enrich",
            "aggregate",
            "site",
        ],
        help="Run one stage instead of the whole pipeline.",
    )

    parser.add_argument(
        "--gemini-package",
        default=None,
        help=(
            "Directory holding the Gemini-derived knowledge package. "
            "Read-only, like the archive. Defaults to "
            "$GEMINI_PACKAGE_ROOT. The package is machine transcription "
            "and derived data; it never becomes the post's own text."
        ),
    )

    parser.add_argument(
        "--gemini-output",
        default=None,
        help=(
            "Where an import writes its records. Default "
            "data/imported/gemini."
        ),
    )

    parser.add_argument(
        "--gemini-force",
        action="store_true",
        help=(
            "Re-import even when the package and archive are unchanged. "
            "Usually pointless: an unchanged package produces an "
            "identical import."
        ),
    )

    parser.add_argument(
        "--visual-gaps",
        action="store_true",
        help=(
            "Read only the slides a previous Gemini import could not "
            "transcribe. The vision processor stays the fallback rather "
            "than re-reading images that already have text."
        ),
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

        if args.only == "gemini-plan":
            return _gemini_plan(args)

        if args.only == "gemini-import":
            return _gemini_import(args)

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
# Gemini package
# ---------------------------------------------------------------------


def _package_root(args) -> str | None:
    """
    Where the Gemini package is, or None when it is not configured.

    Reported rather than guessed, for the same reason the archive root is:
    a default path that does not exist turns a missing configuration into
    a confusing error deep inside a parser, and CI has no reason to have
    somebody's Downloads folder.
    """

    if args.gemini_package:
        return args.gemini_package

    import os

    return os.environ.get("GEMINI_PACKAGE_ROOT")


def _gemini_paths(args) -> tuple[str, str] | None:
    """Both roots, or an error already printed."""

    package = _package_root(args)
    archive = _archive_root(args)

    missing = [
        name
        for name, value in (
            ("--gemini-package or GEMINI_PACKAGE_ROOT", package),
            ("--archive or LINKEDIN_ARCHIVE_ROOT", archive),
        )
        if not value
    ]

    if missing:
        print(
            "PIPELINE_ERROR=no source configured; pass " + " and ".join(missing),
            file=sys.stderr,
        )

        return None

    return package, archive


def _gemini_plan(args) -> int:
    """
    What importing the package would find. Writes nothing.

    Reads the package and the archive and stops. It is the only safe way
    to look at a package nobody has verified, because it answers "what
    does this claim, and does the archive agree" without leaving a
    trace in the repository.
    """

    from src.gemini import build

    resolved = _gemini_paths(args)

    if resolved is None:
        return 1

    package, archive = resolved

    try:
        result = build(package, archive)

    except Exception as exc:  # noqa: BLE001
        print(f"PIPELINE_ERROR={type(exc).__name__}: {exc}", file=sys.stderr)

        return 1

    log(result.describe())

    if result.unresolved_topic_groups:
        log("")
        log(
            f"Topic index references {len(result.unresolved_topic_groups)} "
            "group(s) with no image record; see cross_check.json after an "
            "import."
        )

    return 0


def _gemini_import(args) -> int:
    """
    Import the package, or report that nothing changed.

    Idempotent by digest: the same package and the same archive produce
    no second import, because there is nothing new to record and a
    second copy of every record would only make the repository larger
    and the knowledge base less trustworthy.
    """

    from src.gemini import DEFAULT_OUTPUT, run as run_import

    resolved = _gemini_paths(args)

    if resolved is None:
        return 1

    package, archive = resolved

    output = args.gemini_output or DEFAULT_OUTPUT

    try:
        result = run_import(
            package,
            archive,
            output,
            force=args.gemini_force,
        )

    except Exception as exc:  # noqa: BLE001
        print(f"PIPELINE_ERROR={type(exc).__name__}: {exc}", file=sys.stderr)

        return 1

    log(result.describe())

    if not result.wrote:
        log("")
        log(
            "nothing written: the package and the archive are both "
            "unchanged since the last import"
        )

        return 0

    actionable = [gap for gap in result.gaps if gap.actionable]

    log("")
    log(
        f"{len(actionable)} slide(s) a vision model could still read; run "
        "with --visual-gaps to spend calls on those alone."
    )

    from src.gemini.derived import claim_counts

    counts = claim_counts(output)

    log(
        f"Technology claims: {counts['claims_with_a_slide']} backed by a "
        f"named slide, {counts['claims_without_one']} attributed to a post "
        "with no slide that says so. The second kind is weaker and is "
        "kept only so nothing is lost."
    )

    return 0


def _gap_filenames() -> set[str] | None:
    """
    The slides an import left for the vision processor.

    ``None`` when there is no import to read, which means "no filter" and
    so a normal full CP12 run. Treating a missing import as an empty gap
    list would silently reduce a fallback run to reading nothing, and
    that is the one failure mode here worth engineering against.
    """

    from src.gemini import load_gaps

    gaps = load_gaps()

    if not gaps:
        return None

    return {gap.filename for gap in gaps if gap.actionable}


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

    only = _gap_filenames() if args.visual_gaps else None

    runner = make_visual(root, batch_size=args.visual_batch, only=only)

    plan = runner.plan()

    if plan.archive_error:
        print(f"PIPELINE_ERROR={plan.archive_error}", file=sys.stderr)
        return 1

    log(plan.render())

    if only is not None:
        log("")
        log(
            f"Restricted to {len(only)} slide(s) a previous import could "
            "not transcribe. Drop --visual-gaps to plan a full run."
        )

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

    only = _gap_filenames() if args.visual_gaps else None

    runner = make_visual(
        root,
        batch_size=args.visual_batch,
        attempts=args.attempts,
        only=only,
    )

    if only is not None:
        log(
            f"reading only the {len(only)} slide(s) a previous import "
            "could not transcribe"
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