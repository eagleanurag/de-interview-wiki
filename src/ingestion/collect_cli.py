#!/usr/bin/env python3
"""
Collection command line.

    python -m src.ingestion.collect doctor
    python -m src.ingestion.collect run --source manual \
        --bundle-root captures/ --max-posts 3
    python -m src.ingestion.collect run --source linkedin \
        --profile my-handle --max-posts 25 --headed
    python -m src.ingestion.collect_cli saved-items \
        --input data/incoming/saved-items/manifest.csv
    python -m src.ingestion.collect status
    python -m src.ingestion.collect reset

``doctor`` reports whether a source is usable without touching a live
site, and never prints a credential value.

``saved-items`` is the one command that needs no network and no
credentials at all: it reads a list the user exported from LinkedIn and
the captures the user placed beside it. Nothing is fetched to fill a
gap, and a saved link with no capture stays a saved link.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.ingestion import checkpoints as checkpoint_module
from src.ingestion import credentials as credential_module
from src.ingestion.collect import (
    CollectionLimits,
    CollectionReport,
    Collector,
    read_state,
    write_state,
)
from src.aggregation.consolidation import detect_technologies

# Imported as modules rather than through the package's re-exports, so
# ``archive_prepare`` is the module and not the function it also exposes.
import src.ingestion.linkedin_archive.archive as archive_module
import src.ingestion.linkedin_archive.prepare as archive_prepare
import src.ingestion.linkedin_archive.report as archive_report
from src.ingestion.post_document import PostDocument
from src.ingestion.post_loader import load_post
from src.ingestion.saved_items import diagnostics as diag
from src.ingestion.saved_items import intake
from src.ingestion.saved_items import plan as plan_module
from src.ingestion.saved_items.manifest import (
    ManifestUnreadable,
    SavedItemsManifest,
    manifest_path,
)
from src.ingestion.saved_items.model import SavedItem
from src.ingestion.saved_items.readers import (
    ManifestError,
    read_manifest,
)
from src.ingestion.saved_items.source import SavedItemsSource, reconcile
from src.ingestion.saved_items.urls import (
    SavedItemUrlError,
    normalize_linkedin_url,
)
from src.ingestion.sources.base import (
    CollectionState,
    CollectionStopped,
    StopReason,
)
from src.ingestion.sources.linkedin import (
    LinkedInLimits,
    LinkedInSource,
    browser_available,
    linkedin_installed,
    playwright_install_hint,
)
from src.ingestion.sources.manual import ManualSource
from src.ingestion.validation import (
    LEVEL_ERROR,
    describe_report,
    validate_post_directory,
)


DEFAULT_POSTS_DIRECTORY = Path("data") / "posts"

DEFAULT_SAVED_ITEMS_DIRECTORY = Path("data") / "incoming" / "saved-items"

#: Where a local LinkedIn archive is written out for import. A separate
#: directory from the saved-items drop zone, so the two are never
#: confused for one another and so a re-run of either does not disturb
#: the other's state.
DEFAULT_ARCHIVE_DIRECTORY = Path("data") / "incoming" / "linkedin-archive"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect authorized content into data/posts/."
        )
    )

    subcommands = parser.add_subparsers(dest="command")

    subcommands.add_parser(
        "doctor", help="Report what is configured, without credentials."
    )

    subcommands.add_parser(
        "status", help="Show the recorded collection state."
    )

    subcommands.add_parser(
        "reset", help="Clear the recorded collection state."
    )

    run = subcommands.add_parser(
        "run", help="Collect from a source, bounded."
    )

    run.add_argument(
        "--source",
        choices=["manual", "linkedin"],
        default="manual",
    )

    run.add_argument(
        "--posts-root",
        default=str(DEFAULT_POSTS_DIRECTORY),
    )

    run.add_argument(
        "--bundle-root",
        default="data/incoming",
        help=(
            "Directory of capture bundles, for the manual source. "
            "Each subdirectory becomes one post, with any images and "
            "PDFs beside its text. Loose .md, .txt and .jsonl files "
            "in the root are read directly."
        ),
    )

    run.add_argument(
        "--profile",
        default="",
        help="LinkedIn profile handle, for the linkedin source.",
    )

    run.add_argument("--max-posts", type=int, default=None)
    run.add_argument("--since", default=None)
    run.add_argument("--until", default=None)
    run.add_argument("--scroll-limit", type=int, default=None)

    run.add_argument(
        "--resume",
        action="store_true",
        help="Continue from the recorded state.",
    )

    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be imported without writing.",
    )

    run.add_argument(
        "--headed",
        action="store_true",
        help="Show the browser, which is needed for a human challenge.",
    )

    run.add_argument(
        "--json",
        action="store_true",
        help="Print the report as JSON.",
    )

    run.add_argument(
        "--login",
        action="store_true",
        help=(
            "Open a browser and pause so you can sign in yourself. "
            "Saves the session for later runs, which is how a human "
            "completes a challenge once instead of on every run."
        ),
    )

    saved = subcommands.add_parser(
        "saved-items",
        help=(
            "Import a list of saved LinkedIn items and any captures "
            "supplied beside it. Needs no credentials and no network."
        ),
    )

    saved.add_argument(
        "--input",
        action="append",
        default=[],
        dest="inputs",
        metavar="FILE",
        help=(
            "A saved-items list: .csv, .tsv, .txt, .json or .jsonl. "
            "May be given more than once. Columns are matched by name, "
            "so url/link/LinkedIn URL and saved date/date saved all "
            "work."
        ),
    )

    saved.add_argument(
        "--bundle-root",
        default=None,
        help=(
            "Where the capture bundles live. Defaults to the directory "
            "holding --input, or data/incoming/saved-items."
        ),
    )

    saved.add_argument(
        "--manifest-file",
        default=None,
        help=(
            "Where the saved-items state is kept between runs. Defaults "
            "to saved-items-manifest.json inside the bundle root. A "
            "manifest that cannot be read is a stop, not a warning, "
            "because re-importing it would duplicate every post."
        ),
    )

    saved.add_argument(
        "--posts-root",
        default=str(DEFAULT_POSTS_DIRECTORY),
    )

    saved.add_argument(
        "--report",
        default=None,
        metavar="FILE",
        help="Also write the report to this file.",
    )

    saved.add_argument(
        "--validate",
        action="store_true",
        help="Validate every post the run imported.",
    )

    saved.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Say exactly what would happen and change nothing. Nothing "
            "is written: not a post, not the manifest, not a report "
            "file."
        ),
    )

    saved.add_argument(
        "--plan",
        action="store_true",
        help=(
            "Print every item grouped by what would happen to it: NEW, "
            "CHANGED, UNCHANGED, DUPLICATE, MISSING_CAPTURE, INVALID "
            "or FAILED. Implies --dry-run."
        ),
    )

    saved.add_argument(
        "--adopt-orphans",
        action="store_true",
        help=(
            "Add a saved item for any capture that names a usable "
            "LinkedIn URL but has no manifest item yet. A capture with "
            "no URL is still reported, never adopted."
        ),
    )

    saved.add_argument(
        "--debug",
        action="store_true",
        help="Show the full traceback when something fails.",
    )

    saved.add_argument("--max-posts", type=int, default=None)
    saved.add_argument("--since", default=None)
    saved.add_argument("--until", default=None)

    saved.add_argument(
        "--json",
        action="store_true",
        help="Print the report as JSON.",
    )

    # ---------------------------------------------------------------
    # Building the list
    # ---------------------------------------------------------------

    init = subcommands.add_parser(
        "saved-items-init",
        help=(
            "Build or update the saved-items list from links, a "
            "spreadsheet, or a browser bookmark export."
        ),
    )

    init.add_argument(
        "--from",
        action="append",
        default=[],
        dest="sources",
        metavar="FILE",
        help=(
            "A file of links, a .csv/.tsv/.json/.jsonl list, or a "
            "browser bookmark export. May be given more than once."
        ),
    )

    init.add_argument(
        "--url",
        action="append",
        default=[],
        dest="urls",
        metavar="URL",
        help="A single saved link. May be given more than once.",
    )

    init.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help=(
            "Where to write the list. Defaults to manifest.csv in the "
            "saved-items directory."
        ),
    )

    init.add_argument(
        "--bundle-root",
        default=None,
        help=(
            "The saved-items directory. Defaults to "
            "data/incoming/saved-items."
        ),
    )

    init.add_argument(
        "--replace",
        action="store_true",
        help=(
            "Write only what was passed in, discarding the existing "
            "list. Captures and enrichment are never touched either "
            "way, so this only changes which links are known."
        ),
    )

    init.add_argument(
        "--json",
        action="store_true",
        help="Print the result as JSON.",
    )

    # ---------------------------------------------------------------
    # Status
    # ---------------------------------------------------------------

    status = subcommands.add_parser(
        "saved-items-status",
        help=(
            "Report what is saved, what has a capture, and what has "
            "already been imported. Reads only; changes nothing."
        ),
    )

    status.add_argument(
        "--bundle-root",
        default=None,
        help=(
            "The saved-items directory. Defaults to "
            "data/incoming/saved-items."
        ),
    )

    status.add_argument(
        "--posts-root",
        default=str(DEFAULT_POSTS_DIRECTORY),
    )

    status.add_argument(
        "--show-pending",
        action="store_true",
        help="List every item still waiting for a capture.",
    )

    status.add_argument(
        "--json",
        action="store_true",
        help="Print the report as JSON.",
    )

    # ---------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------

    check = subcommands.add_parser(
        "saved-items-validate",
        help=(
            "Check the whole saved-items inbox and report every problem, "
            "with the fix for each."
        ),
    )

    check.add_argument(
        "--input",
        action="append",
        default=[],
        dest="inputs",
        metavar="FILE",
        help="A list to read as well as the stored one.",
    )

    check.add_argument(
        "--bundle-root",
        default=None,
        help=(
            "The saved-items directory. Defaults to "
            "data/incoming/saved-items."
        ),
    )

    check.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors.",
    )

    check.add_argument(
        "--debug",
        action="store_true",
        help="Show the full traceback when something fails.",
    )

    check.add_argument(
        "--json",
        action="store_true",
        help="Print the report as JSON.",
    )

    # ---------------------------------------------------------------
    # A local LinkedIn archive
    # ---------------------------------------------------------------

    archive = subcommands.add_parser(
        "linkedin-archive",
        help=(
            "Read a local LinkedIn saved-post archive and write it out as "
            "a saved-items drop zone. Reads files you already have. Fetches "
            "nothing, opens no browser, and uses no credential."
        ),
    )

    archive.add_argument(
        "--input",
        default=None,
        metavar="DIR",
        help=(
            "The archive directory, the one holding posts_archive.json "
            "and media/. Required; there is no default, because a path "
            "this project guesses at is a path this project should not "
            "be reading."
        ),
    )

    archive.add_argument(
        "--out",
        default=None,
        metavar="DIR",
        help=(
            "Where to write the drop zone. Defaults to "
            "data/incoming/linkedin-archive."
        ),
    )

    archive.add_argument(
        "--report",
        default=None,
        metavar="FILE",
        help="Also write the import report to this file as JSON.",
    )

    archive.add_argument(
        "--include-duplicates",
        action="store_true",
        help=(
            "Write a capture folder for repeated content as well as for "
            "the record that keeps its place. Off by default: two "
            "captures of one post would become two posts."
        ),
    )

    archive.add_argument(
        "--import",
        action="store_true",
        dest="do_import",
        help=(
            "Import the prepared drop zone into data/posts/ as well, "
            "using the same saved-items importer as a hand-built list. "
            "Without this the drop zone is written and nothing else "
            "happens."
        ),
    )

    archive.add_argument(
        "--posts-root",
        default=str(DEFAULT_POSTS_DIRECTORY),
    )

    archive.add_argument(
        "--json",
        action="store_true",
        help="Print the report as JSON.",
    )

    archive.add_argument(
        "--debug",
        action="store_true",
        help="Show the full traceback when something fails.",
    )

    return parser


def run_linkedin_archive(
    args: argparse.Namespace,
    root: str | Path = ".",
) -> int:
    """
    Read a local LinkedIn archive and prepare it for import.

    Two stages, and the first is the only one that touches the archive.
    Reading produces a report of what is in it; preparing writes that
    content out as a saved-items drop zone, which is the form the
    existing importer already reads. Nothing is fetched, no browser is
    opened, no credential is read, and the archive is never written to.

    The archive is treated as read-only input throughout. Its own
    browser-session directory is named and refused rather than merely
    left alone, because a session is a credential in a different shape
    and a reader should not have to know that to avoid it.
    """

    if not args.input:
        print("This command needs the path to your archive.")
        print()
        print("  python -m src.ingestion.collect_cli linkedin-archive \\")
        print('      --input "C:\\path\\to\\linkedin_saved_archive"')
        print()
        print("That is the directory holding posts_archive.json and")
        print("media/. Nothing is downloaded and nothing is opened in a")
        print("browser; the command reads the files already on this")
        print("machine and writes them out for the saved-items importer.")
        return 1

    archive_root = Path(args.input).expanduser()

    destination = (
        Path(args.out).expanduser()
        if args.out
        else Path(DEFAULT_ARCHIVE_DIRECTORY)
    )

    try:
        archive = archive_module.read_archive(archive_root)

    except archive_module.ArchiveError as exc:
        if args.debug:
            raise

        print(f"Cannot read the archive: {exc}")
        return 1

    except Exception as exc:  # noqa: BLE001
        if args.debug:
            raise

        print(
            diag.diagnostic_from_exception(str(archive_root), exc).render()
        )
        return 1

    if not archive.records and not archive.failures:
        print(f"The archive at {archive_root} holds no post records.")
        return 1

    _report_archive_shape(archive)

    prepared = archive_prepare.prepare(
        archive,
        destination,
        include_duplicates=args.include_duplicates,
    )

    report = archive_report.build_report(archive, prepared=prepared)

    if args.report:
        written = archive_report.write_report(report, args.report)

        if not args.json:
            print(f"Report written to {written}")

    if args.json:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))

    else:
        print()
        print(report.render())

    imported = 0

    if args.do_import:
        imported = _import_prepared(
            destination, Path(args.posts_root), root
        )

    if args.json:
        return 0 if report.failed == 0 else 1

    if report.failed:
        print()
        print(
            f"{report.failed} record(s) could not be read or written. "
            "The rest are in the drop zone; re-run to retry them."
        )

    if not args.do_import:
        print()
        print("Nothing was imported yet. To import the prepared posts:")
        print(
            "  python -m src.ingestion.collect_cli saved-items "
            f"--bundle-root {destination}"
        )
        print()
        print("To do both in one step next time, add --import.")

    elif imported:
        print()
        print(
            f"Imported {imported} post(s) into {args.posts_root}. "
            "Next: python -m src.pipeline"
        )

    return 0 if report.failed == 0 else 1


def _report_archive_shape(archive) -> None:
    """
    Say what was found, before anything is written.

    Printed first because a wrong path should cost nothing. By the time
    a report is rendered a drop zone may exist, and a reader who finds
    out afterwards that they pointed at the wrong folder has already had
    to clean up.
    """

    media = archive.media

    print(f"Read {archive.total_seen} record(s) from {archive.path}")
    print(
        f"  valid {len(archive.records)}   "
        f"unreadable {len(archive.failures)}   "
        f"repeated content {len(archive.duplicates)}"
    )
    print(
        f"  media {len(media)} file(s), {media.total_bytes:,} bytes, "
        f"{len(media.unreadable)} unreadable"
    )

    if media.rejected:
        print(
            f"  {len(media.rejected)} file(s) resolved outside the media "
            "folder and were refused"
        )

    for directory in archive_module.FORBIDDEN_DIRECTORIES:
        if (archive.path / directory).is_dir():
            print(
                f"  {directory}/ is present and was not opened. It holds a "
                "browser session, and nothing here needs one."
            )


def _import_prepared(
    drop_zone: Path,
    posts_root: Path,
    root: str | Path,
) -> int:
    """
    Import a prepared drop zone through the existing saved-items path.

    Deliberately the same importer a hand-built list uses, rather than a
    second route into the posts directory. A second route would be a
    second thing whose behaviour differs, and the whole reason for
    writing the archive out as a drop zone was to avoid that.
    """
    manifest = drop_zone / archive_prepare.MANIFEST

    if not manifest.is_file():
        print(f"No manifest was written to {drop_zone}")
        return 0

    state = (
        drop_zone / "saved-items-manifest.json"
    )

    try:
        source = SavedItemsSource(drop_zone, manifest_file=state)

    except ManifestUnreadable as exc:
        print(f"Cannot read the drop zone's manifest: {exc}")
        return 0

    source.read_manifests([manifest])

    collector = Collector(
        source,
        root=posts_root,
        limits=CollectionLimits(),
        checkpoint=checkpoint_module.read(root),
        progress=(lambda message: print(f"  {message}")),
        repository_root=root,
    )

    collection = collector.run(resume=False)

    reconcile(source.manifest, posts_root=posts_root)

    # A count, not the identifiers. The caller prints this and the
    # summary is meant to be one line; returning the list made a
    # four-hundred-item import print four hundred ids.
    return len(collection.imported)


def print_doctor(root: str | Path = ".") -> int:
    """
    Report readiness. Never prints a credential value.
    """

    credential_module.load_local_environment(root / ".env")

    credential_status = credential_module.status()

    print("=== Collection doctor ===")
    print(f"LinkedIn credentials configured: "
          f"{'yes' if credential_status.configured else 'no'}")

    if credential_status.incomplete:
        missing = (
            "LINKEDIN_PASSWORD"
            if credential_status.username_configured
            else "LINKEDIN_USERNAME"
        )
        print(f"  missing: {missing}")

    print()
    print(f"Playwright: {'installed' if linkedin_installed() else 'not installed'}")

    if not linkedin_installed():
        print(f"  {playwright_install_hint()}")

    print(f"LinkedIn source: {browser_available()}")

    posts_root = root / DEFAULT_POSTS_DIRECTORY
    count = 0

    if posts_root.is_dir():
        count = sum(
            1
            for path in posts_root.iterdir()
            if path.is_dir() and (path / "post.json").is_file()
        )

    print(f"Posts directory: {posts_root} ({count} post(s))")

    state = read_state(root)

    print(
        "Collection state: "
        + (
            "not started"
            if state is None
            else (
                f"persisted={state.persisted} "
                f"duplicates={state.duplicates} "
                f"failed={state.failed} "
                f"stopped_because={state.stopped_because or 'unknown'}"
            )
        )
    )

    checkpoint = checkpoint_module.read(root)

    print(
        "Mission checkpoint: "
        + (
            "none"
            if checkpoint is None
            else f"{checkpoint.phase} ({checkpoint.status})"
        )
    )

    return 0


def print_status(root: str | Path = ".") -> int:
    """Show the recorded collection state as JSON."""

    state = read_state(root)

    print(
        json.dumps(
            state.to_dict() if state else {},
            indent=2,
            sort_keys=True,
        )
    )

    return 0


def reset_state(root: str | Path = ".") -> int:
    """Clear collection state so the next run starts fresh."""

    write_state(CollectionState(), root)

    print("Collection state cleared.")

    return 0


def saved_items_bundle_root(args: argparse.Namespace) -> Path:
    """
    Where the capture bundles are.

    An explicit ``--bundle-root`` wins. Otherwise the bundles are looked
    for beside the list that named them, because a user who exports a
    saved list drops both in the same folder.
    """
    if args.bundle_root:
        return Path(args.bundle_root)

    if getattr(args, "inputs", None):
        return Path(args.inputs[0]).expanduser().resolve().parent

    return Path(DEFAULT_SAVED_ITEMS_DIRECTORY)


# ---------------------------------------------------------------------
# Building the list
# ---------------------------------------------------------------------


def run_saved_items_init(
    args: argparse.Namespace,
    root: str | Path = ".",
) -> int:
    """
    Build or update the saved-items list.

    Reads whatever the user has and writes the one list the importer
    understands. Existing links are kept and enriched from the new input
    rather than replaced, so running this twice with different sources
    produces the union instead of the last one read.

    Only the list is written. A capture is the user's own material and an
    enrichment is real work that has already been paid for, so neither is
    touched by this command under any flag.
    """
    bundle_root = (
        Path(args.bundle_root)
        if args.bundle_root
        else Path(DEFAULT_SAVED_ITEMS_DIRECTORY)
    )

    output = (
        Path(args.output)
        if args.output
        else bundle_root / "manifest.csv"
    )

    incoming: list = []

    problems: list[str] = []

    for entry in args.sources or []:
        source = Path(entry).expanduser()

        try:
            read = intake.read_input(source)

        except ManifestError as exc:
            print(f"Could not read {source}: {exc}")
            return 1

        except Exception as exc:  # noqa: BLE001
            if args.debug:
                raise

            print(
                diag.diagnostic_from_exception(str(source), exc).render()
            )
            return 1

        incoming.extend(read.items)
        problems.extend(str(issue) for issue in read.issues)

    for entry in args.urls or []:
        try:
            incoming.append(
                SavedItem.from_url(normalize_linkedin_url(entry))
            )

        except SavedItemUrlError as exc:
            problems.append(f"--url {entry}: {exc}")

    if not incoming and not problems:
        print("Nothing to add.")
        print()
        print("Pass links or a file:")
        print()
        print("  python -m src.ingestion.collect_cli saved-items-init \\")
        print("      --url https://www.linkedin.com/posts/...")
        print()
        print("  python -m src.ingestion.collect_cli saved-items-init \\")
        print("      --from my-saved-posts.txt")
        print()
        print("  python -m src.ingestion.collect_cli saved-items-init \\")
        print("      --from bookmarks-export.html")
        return 1

    existing: list = []

    if output.is_file() and not args.replace:
        try:
            existing = read_manifest(output).items

        except ManifestError as exc:
            # The list on file is the user's own record of what they
            # saved. It is never overwritten because it could not be
            # read.
            print(f"Could not read the existing list {output}: {exc}")
            print()
            print("Nothing was written. Fix the file, or pass")
            print("--replace to write a new list from what you pass in.")
            return 1

    result = intake.merge_into(existing, incoming)

    items = incoming if args.replace else existing

    try:
        written = intake.write_list(output, items)

    except OSError as exc:
        print(f"Could not write {output}: {exc}")
        return 1

    if args.json:
        print(
            json.dumps(
                {
                    "list": str(written),
                    "items": len(items),
                    **result.as_dict(),
                },
                indent=2,
                sort_keys=True,
            )
        )

        return 0 if not problems else 1

    print(f"Wrote {len(items)} link(s) to {written}")
    print(f"  {result.line()}")

    if problems:
        print()
        print(f"Skipped {len(problems)} problem(s):")

        for problem in problems[:20]:
            print(f"  {problem}")

        if len(problems) > 20:
            print(f"  ... and {len(problems) - 20} more")

    print()
    print("Next:")
    print(f"  python -m src.ingestion.collect_cli saved-items-validate "
          f"--bundle-root {bundle_root}")
    print(f"  python -m src.ingestion.collect_cli saved-items "
          f"--input {written}")

    return 0 if not problems else 1


# ---------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------


def run_saved_items_status(
    args: argparse.Namespace,
    root: str | Path = ".",
) -> int:
    """
    Say where the saved list has got to.

    Every number comes from the manifest, the captures on disk and the
    posts that exist. Nothing is estimated, and a section with nothing in
    it says so rather than being omitted, because an absent number reads
    as a forgotten one.
    """
    bundle_root = (
        Path(args.bundle_root)
        if args.bundle_root
        else Path(DEFAULT_SAVED_ITEMS_DIRECTORY)
    )

    if not bundle_root.is_dir():
        print(f"The saved-items directory does not exist: {bundle_root}")
        print()
        print("Create it, or point at the one you use:")
        print(f"  mkdir {bundle_root}")
        return 1

    manifest_path = bundle_root / "saved-items-manifest.json"

    if not manifest_path.is_file():
        print(f"No saved-items manifest yet in {bundle_root}.")
        print()
        print("Build the list first:")
        print(f"  python -m src.ingestion.collect_cli saved-items-init "
              f"--bundle-root {bundle_root}")
        return 1

    try:
        manifest = SavedItemsManifest.load(manifest_path)

    except ManifestUnreadable as exc:
        print(str(exc))
        return 1

    plan = plan_module.build_plan(bundle_root, manifest=manifest)

    knowledge = _knowledge_counts(Path(args.posts_root))

    manifest_items = plan.manifest_items()
    with_capture = plan.with_capture()
    metadata_only = plan.metadata_only()
    importable = plan.importable()
    already = plan.already_imported()

    counts = plan.counts()
    changed = counts.get("CHANGED", 0)
    failed = counts.get("FAILED", 0) + counts.get("INVALID", 0)

    if args.json:
        print(
            json.dumps(
                {
                    "root": str(bundle_root),
                    "saved_items": plan.as_dict(),
                    "knowledge": knowledge,
                },
                indent=2,
                sort_keys=True,
            )
        )

        return 0

    print("Saved Items")
    print("-----------")
    print(f"{'Manifest items':<24}{manifest_items:>6}")
    print(f"{'With captures':<24}{with_capture:>6}")
    print(f"{'Metadata only':<24}{metadata_only:>6}")
    print(f"{'Ready to import':<24}{importable:>6}")
    print(f"{'Already imported':<24}{already:>6}")
    print(f"{'Changed':<24}{changed:>6}")
    print(f"{'Failed':<24}{failed:>6}")
    print(f"{'Pending':<24}{metadata_only:>6}")

    qualities = plan.by_quality()

    if any(qualities.values()):
        print()
        print("Capture quality")
        print("---------------")

        for label, value in qualities.items():
            if value:
                print(f"{label:<24}{value:>6}")

    if plan.content_types:
        print()
        print("Content types")
        print("-------------")

        for label, count in sorted(plan.content_types.items()):
            print(f"{label:<24}{count:>6}")

    print()
    print("Knowledge")
    print("---------")

    for label, value in knowledge.items():
        print(f"{label:<24}{value:>6}")

    if plan.orphans:
        print()
        print(f"Orphan captures: {len(plan.orphans)}")

        for folder in plan.orphans[:10]:
            print(f"  {folder}")

        if len(plan.orphans) > 10:
            print(f"  ... and {len(plan.orphans) - 10} more")

    if args.show_pending:
        pending = sorted(
            (
                item
                for item in manifest.items.values()
                if not item.has_content
            ),
            key=lambda entry: entry.canonical_url,
        )

        print()
        print(f"Pending ({len(pending)})")
        print("-" * (10 + len(str(len(pending)))))

        for item in pending[:50]:
            print(f"  {item.canonical_url}")
            print(f"    folder: captures/"
                  f"{item.source_id.replace(':', '-')}")

        if len(pending) > 50:
            print(f"  ... and {len(pending) - 50} more")

    return 0


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------


def run_saved_items_validate(
    args: argparse.Namespace,
    root: str | Path = ".",
) -> int:
    """
    Check the whole inbox and report every problem with its fix.

    Exits non-zero when there is a real error, and zero when there are
    only warnings. A command that failed on every warning would be a
    command users stop running, and the warnings here are mostly the
    expected state of a saved list: most of what a person saved has no
    capture, and that is a backlog rather than a fault.
    """
    bundle_root = (
        Path(args.bundle_root)
        if args.bundle_root
        else Path(DEFAULT_SAVED_ITEMS_DIRECTORY)
    )

    if not bundle_root.is_dir():
        print(f"The saved-items directory does not exist: {bundle_root}")
        return 1

    inputs = [Path(entry).expanduser() for entry in args.inputs or []]

    if not inputs:
        # The list the user put in the drop zone is the obvious thing to
        # check, and asking them to name it again would mean giving the
        # same path twice.
        inputs = _candidate_lists(bundle_root)

    try:
        plan = plan_module.build_plan(bundle_root, inputs=inputs)

    except Exception as exc:  # noqa: BLE001
        if args.debug:
            raise

        print(
            diag.diagnostic_from_exception(str(bundle_root), exc).render()
        )
        return 1

    if args.json:
        print(json.dumps(plan.as_dict(), indent=2, sort_keys=True))

        return _validation_exit(plan, strict=args.strict)

    counts = plan.counts()

    print("Saved Items Check")
    print("-----------------")
    print(f"{'Directory':<24}{bundle_root}")

    for outcome in plan_module.OUTCOMES:
        if counts.get(outcome):
            print(f"{outcome:<24}{counts[outcome]:>6}")

    print()
    print(f"{'Errors':<24}{len(plan.errors):>6}")
    print(f"{'Warnings':<24}{len(plan.warnings):>6}")

    if plan.problems:
        print()

        ordered = sorted(
            plan.problems,
            key=lambda item: (not item.is_error, item.code, item.location),
        )

        # Each problem is printed in full, with its fix, so only a
        # bounded number are shown. The counts above are exact and
        # ``--json`` returns the rest, so nothing is hidden from
        # anything that asks properly.
        for entry in ordered[: plan_module.Plan.MAX_LISTED]:
            print()
            print(entry.render())

        hidden = len(ordered) - plan_module.Plan.MAX_LISTED

        if hidden > 0:
            print()
            print(
                f"... and {hidden} more problem(s) not shown. "
                "Re-run with --json for all of them."
            )

    if not plan.problems:
        print()
        print("Nothing to fix.")

    return _validation_exit(plan, strict=args.strict)


def _validation_exit(plan, *, strict: bool) -> int:
    if plan.errors:
        return 1

    if strict and plan.warnings:
        return 1

    return 0


def _knowledge_counts(posts_root: Path) -> dict[str, int]:
    """
    What is actually in the knowledge base.

    Read from the posts on disk rather than from a stored total, so the
    number cannot drift from the thing it describes. Every value is a
    count of files that exist; nothing is carried over from a previous
    run.
    """
    counts = {
        "Imported posts": 0,
        "Interview relevant": 0,
        "Questions generated": 0,
        "Topics": 0,
        "Concepts": 0,
        "Technologies": 0,
    }

    if not posts_root.is_dir():
        return counts

    topics: set[str] = set()
    concepts: set[str] = set()
    technologies: set[str] = set()

    for directory in sorted(posts_root.iterdir()):
        if not directory.is_dir():
            continue

        path = directory / "post.json"

        if not path.is_file():
            continue

        try:
            post = load_post(directory)

        except Exception:  # noqa: BLE001
            # A post being written right now, or one that is not this
            # project's. Either way it is not a number to report.
            continue

        counts["Imported posts"] += 1

        if post.classification.interview_relevant:
            counts["Interview relevant"] += 1

        counts["Questions generated"] += len(post.interview_questions)

        topics.update(
            topic for topic in post.ai_analysis.topics if topic.strip()
        )
        concepts.update(
            concept
            for concept in post.ai_analysis.concepts
            if concept.strip()
        )

        # Read the same way consolidation reads it, so the number here
        # is the number the knowledge base holds rather than a second
        # opinion about what the posts mention.
        technologies.update(
            detect_technologies(post.original_text)
        )

    counts["Topics"] = len(topics)
    counts["Concepts"] = len(concepts)
    counts["Technologies"] = len(technologies)

    return counts


def run_saved_items(
    args: argparse.Namespace,
    root: str | Path = ".",
) -> int:
    """
    Import saved items and the captures supplied beside them.

    Reads a list the user exported, matches any capture bundles to the
    items they belong to, and hands the results to the same collector
    every other source uses, so a saved post is stored exactly where a
    collected one is.

    Nothing here reaches the network, and no credential is read. The
    report distinguishes an item that was imported from one that is only
    a link, because the difference decides whether there is anything in
    the knowledge base to read.

    With ``--dry-run`` nothing is written at all, and the run is decided
    from the same reading of the inbox that a real import would use, so
    the preview and the import cannot disagree.
    """
    dry_run = bool(args.dry_run or args.plan)

    bundle_root = saved_items_bundle_root(args)

    if not bundle_root.is_dir():
        print(f"The saved-items directory does not exist: {bundle_root}")
        print()
        print("Create it and drop your export in:")
        print(f"  {bundle_root}")
        return 1

    inputs = [
        Path(entry).expanduser()
        for entry in (args.inputs or [bundle_root])
    ]

    if not args.inputs and not any(
        path.is_file() for path in _candidate_lists(bundle_root)
    ):
        print(f"No saved-items list found in {bundle_root}.")
        print()
        print("Build one with saved-items-init, or write a .csv with a")
        print("url column and put it in that directory. Example:")
        print("  URL,Saved Date,Title,Notes")
        print("  https://www.linkedin.com/posts/... ,2026-01-02,,")
        return 1

    state_file = (
        Path(args.manifest_file)
        if args.manifest_file
        else manifest_path(bundle_root)
    )

    if dry_run:
        return _preview(args, bundle_root, state_file, inputs)

    try:
        source = SavedItemsSource(
            bundle_root,
            manifest_file=state_file,
        )

    except ManifestUnreadable as exc:
        # Stopping here is the point. Carrying on would import every
        # item a second time, because the record of what was already
        # imported is exactly what could not be read.
        print(f"Cannot read the saved-items manifest: {exc}")
        return 1

    if args.adopt_orphans:
        adopted = _adopt_orphans(source)

        for line in adopted:
            print(f"  {line}")

    source.read_manifests(inputs)

    limits = CollectionLimits(
        max_posts=args.max_posts,
        since=args.since,
        until=args.until,
        dry_run=args.dry_run,
    )

    collector = Collector(
        source,
        root=Path(args.posts_root),
        limits=limits,
        checkpoint=checkpoint_module.read(root),
        progress=(
            (lambda message: None)
            if args.json
            else (lambda message: print(f"  {message}"))
        ),
        repository_root=root,
    )

    collection = collector.run(resume=False)

    # The importer is the only thing that knows what was really stored,
    # so the manifest is brought up to date from the posts themselves
    # rather than from what the source intended.
    reconcile(source.manifest, posts_root=Path(args.posts_root))

    report = source.report()

    if args.json:
        print(
            json.dumps(
                {
                    "saved_items": report.as_dict(),
                    "collection": {
                        "stopped_because": collection.stopped_because,
                        "message": collection.message,
                        "imported": collection.imported,
                        "duplicates": collection.duplicates,
                        "failed": collection.failed,
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )

    else:
        print()
        print(report.render())

        if report.metadata_only:
            print()
            print(
                f"{report.metadata_only} item(s) are links only. Each one "
                "is a URL from your export with no body text behind it."
            )
            print()
            print("To capture one, create a folder named after it:")
            print(f"  {bundle_root / 'captures' / '<source_id>'}")

    if args.report:
        destination = Path(args.report)

        destination.parent.mkdir(parents=True, exist_ok=True)

        destination.write_text(
            report.render() + "\n", encoding="utf-8"
        )

        if not args.json:
            print()
            print(f"Report written to {destination}")

    problems: list[str] = []

    if args.validate:
        problems = validate_imported(
            source.manifest,
            posts_root=Path(args.posts_root),
        )

        if problems and not args.json:
            print()
            print(describe_report_summary(problems))

    succeeded = collection.succeeded and not problems

    if not succeeded and not args.json:
        print()
        print(
            f"Stopping reason: {collection.stopped_because or 'unknown'}"
        )

    return 0 if succeeded else 1


def _preview(
    args: argparse.Namespace,
    bundle_root: Path,
    state_file: Path,
    inputs: list[Path],
) -> int:
    """
    Say what an import would do, and write nothing.

    The plan is built against a copy of the manifest held in memory, so
    reading a list cannot record that it was read. That matters most for
    the list file itself and the manifest: those are the two things a
    user might reasonably expect a preview to leave alone, and a preview
    that quietly updated either would be worse than no preview.
    """
    try:
        manifest = SavedItemsManifest.load(state_file)

    except ManifestUnreadable as exc:
        print(f"Cannot read the saved-items manifest: {exc}")
        return 1

    plan = plan_module.build_plan(
        bundle_root,
        manifest=manifest,
        inputs=[path for path in inputs if path.is_file()],
    )

    if args.plan:
        if args.json:
            print(json.dumps(plan.as_dict(), indent=2, sort_keys=True))
        else:
            print(plan.render(title="Saved Items Plan (dry run)"))

            print()
            print("Nothing was written. Run without --dry-run to import.")

        return 0

    counts = plan.counts()

    if args.json:
        print(
            json.dumps(
                {"dry_run": True, **plan.as_dict()}, indent=2, sort_keys=True
            )
        )

        return 0

    print("Dry run")
    print("-------")
    print(f"{'Would import':<24}{counts['NEW']:>6}")
    print(f"{'Would re-import':<24}{counts['CHANGED']:>6}")
    print(f"{'Unchanged':<24}{counts['UNCHANGED']:>6}")
    print(f"{'Duplicate rows':<24}{counts['DUPLICATE']:>6}")
    print(f"{'Missing capture':<24}{counts['MISSING_CAPTURE']:>6}")
    print(f"{'Invalid':<24}{counts['INVALID']:>6}")
    print(f"{'Failed':<24}{counts['FAILED']:>6}")

    if plan.orphans:
        print(f"{'Orphan captures':<24}{len(plan.orphans):>6}")

    print()
    print("Nothing was written. Run without --dry-run to import.")

    return 0


def _adopt_orphans(source: SavedItemsSource) -> list[str]:
    """
    Add an item for a capture that names a URL but has no item yet.

    Only after every known item has had a chance to claim its own
    capture. Adopting first would hand a capture to a brand new item
    while the item it actually belongs to was still unmatched, which
    reports a capture the user never orphaned as one they did.

    Only when the URL is real. A capture with nothing usable in it is
    left alone and reported, because adopting it would mean inventing
    the link it belongs to, and a post attributed to the wrong saved item
    is worse than a capture waiting for its link.
    """
    index = source.bundle_index()

    for item in sorted(
        source.manifest.items.values(), key=lambda entry: entry.source_id
    ):
        if item.bundle:
            found = index.find_by_name(item.bundle)

            if found is not None:
                index.claim(found[0])

        index.find(item)

    adopted: list[str] = []

    for folder in index.unclaimed():
        association = index.associations.get(folder)

        if association is None or not association.url:
            continue

        try:
            item = SavedItem.from_url(
                normalize_linkedin_url(association.url)
            )

        except SavedItemUrlError:
            continue

        if source.manifest.get(item.source_id) is not None:
            # Already known under this link. It is not an orphan; it is
            # a capture the earlier matching pass could not attach, and
            # saying otherwise would send the user looking for a
            # problem that is somewhere else.
            continue

        source.manifest.upsert(item)

        adopted.append(f"Adopted {item.canonical_url} from {folder}")

    if adopted:
        source.manifest.save()

    return adopted


def _candidate_lists(bundle_root: Path) -> list[Path]:
    """List files in a drop zone that could be a saved-items list."""
    return [
        path
        for path in sorted(bundle_root.iterdir())
        if path.is_file()
        and path.suffix.lower() in {".csv", ".tsv", ".txt", ".json", ".jsonl"}
    ]


def validate_imported(
    manifest,
    *,
    posts_root: Path,
) -> list[str]:
    """
    Validate the posts this manifest accounts for.

    Only the posts belonging to a saved item are checked, so a problem
    elsewhere in ``data/posts`` is not reported as this run's problem.
    """
    messages: list[str] = []

    for item in sorted(manifest.items.values(), key=lambda entry: entry.post_id or ""):
        if not item.post_id:
            continue

        directory = posts_root / item.post_id

        if not directory.is_dir():
            messages.append(f"{item.post_id}: the post is not on disk")
            continue

        for issue in validate_post_directory(directory):
            if issue.level != LEVEL_ERROR:
                continue

            messages.append(f"{issue.post_id}: {issue.message}")

    return messages


def describe_report_summary(messages: list[str]) -> str:
    """Name the validation problems without hiding any of them."""
    if not messages:
        return "Validation: no errors."

    lines = [f"Validation: {len(messages)} error(s)"]

    lines.extend(f"  {message}" for message in messages)

    return "\n".join(lines)


def build_source(args: argparse.Namespace):

    if args.source == "manual":
        return ManualSource(
            args.bundle_root,
            platform="manual",
        )

    limits = LinkedInLimits(
        max_posts=args.max_posts,
        since=args.since,
        until=args.until,
        scroll_limit=(
            args.scroll_limit
            if args.scroll_limit is not None
            else 400
        ),
    )

    return LinkedInSource(
        profile=args.profile,
        headed=args.headed,
        limits=limits,
    )


def interactive_login(args: argparse.Namespace) -> int:
    """
    Sign in automatically, and hand over only if LinkedIn requires it.

    The normal path needs nothing from the user: the credentials in
    `.env` are filled and submitted. The browser is left open only if
    LinkedIn presents a challenge, which is completed by hand and then
    verified once, under a bound.

    Exits non-zero unless authentication was actually verified. A
    KeyboardInterrupt is reported as a failure, never as success.
    """

    from src.ingestion.sources.linkedin import (
        AUTH_HUMAN_WAIT_SECONDS,
        LinkedInSource,
    )

    if not credential_module.status().configured:
        print(
            "LinkedIn credentials configured: no\n"
            "Set LINKEDIN_USERNAME and LINKEDIN_PASSWORD in .env, "
            "then run again."
        )
        return 1

    print("LinkedIn credentials configured: yes")
    print("Opening a browser. Signing in automatically...")
    print()

    source = LinkedInSource(
        profile=args.profile,
        headed=True,
        limits=LinkedInLimits(max_posts=args.max_posts),
    )

    needs_human = False

    try:
        try:
            observation = source.open_for_manual_login()
        except CollectionStopped as exc:
            print(f"Could not complete sign-in: {exc}")
            return 1

        print(f"Sign-in state: {observation.state.value}")
        print(f"Detail: {observation.detail}")

        if observation.is_usable:
            saved = source.save_session()

            if saved is None:
                print("Signed in, but the session could not be saved.")
                return 1

            print("Signed in. Session saved inside the ignored tree.")
            print("Browser closing.")
            return 0

        if observation.state.value != "human_challenge":
            # Not authenticated, and nothing for the user to do about
            # it from inside the browser.
            print()
            print("Not authenticated. Nothing was saved.")
            return 1

        needs_human = True

        print()
        print("=" * 66)
        print("ACTION REQUIRED")
        print("=" * 66)
        print(f"LinkedIn is showing a {observation.challenge} challenge.")
        print()
        print("The browser window is open and holds a live session.")
        print("Your credentials have already been supplied.")
        print()
        print("DO:")
        print("  - complete the challenge in that browser window")
        print("  - complete any CAPTCHA, OTP or 2FA yourself")
        print()
        print("DO NOT:")
        print("  - share your credentials with me or anyone else")
        print("  - ask me to bypass or defeat the challenge")
        print()
        print("When the page shows you as signed in, press Enter here.")

        try:
            input()
        except (EOFError, KeyboardInterrupt):
            print()
            print("Interrupted before confirming. Nothing was saved.")
            return 1

        print()
        print("Verifying the session (bounded)...")

        verified = source.wait_for_manual_session(
            timeout_seconds=AUTH_HUMAN_WAIT_SECONDS
        )

        print(f"Verification state: {verified.state.value}")
        print(f"Detail: {verified.detail}")

        if not verified.is_usable:
            print()
            print("Still not authenticated. Nothing was saved.")
            return 1

        saved = source.save_session()

        if saved is None:
            print("Authenticated, but the session could not be saved.")
            return 1

        print()
        print("Authenticated and saved. Browser closing.")
        return 0

    except KeyboardInterrupt:
        print()
        print("Interrupted. Nothing was saved.")
        return 1
    finally:
        source.close()
        if needs_human:
            print("Browser closed. Rerun to use the saved session.")


def run_collection(
    args: argparse.Namespace,
    root: str | Path = ".",
) -> CollectionReport:
    """Run one bounded collection."""

    source = build_source(args)

    limits = CollectionLimits(
        max_posts=args.max_posts,
        since=args.since,
        until=args.until,
        scroll_limit=args.scroll_limit,
        dry_run=args.dry_run,
    )

    collector = Collector(
        source,
        root=Path(args.posts_root),
        limits=limits,
        checkpoint=checkpoint_module.read(root),
        progress=(
            (lambda message: None)
            if args.json
            else (lambda message: print(f"  {message}"))
        ),
        repository_root=root,
    )

    try:
        return collector.run(resume=args.resume)
    finally:
        source.close()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    root = Path.cwd()

    if args.command == "doctor":
        return print_doctor(root)

    if args.command == "status":
        return print_status(root)

    if args.command == "reset":
        return reset_state(root)

    if args.command == "saved-items":
        return run_saved_items(args, root)

    if args.command == "saved-items-init":
        return run_saved_items_init(args, root)

    if args.command == "saved-items-status":
        return run_saved_items_status(args, root)

    if args.command == "saved-items-validate":
        return run_saved_items_validate(args, root)

    if args.command == "linkedin-archive":
        return run_linkedin_archive(args, root)

    if args.command == "run":
        credential_module.load_local_environment(root / ".env")

        if args.source == "linkedin" and args.login:
            return interactive_login(args)

        if args.source == "linkedin" and not args.dry_run:
            credential_module.require()

        report = run_collection(args, root)

        if args.json:
            print(
                json.dumps(
                    {
                        "stopped_because": report.stopped_because,
                        "message": report.message,
                        "state": report.state.to_dict(),
                        "imported": report.imported,
                        "duplicates": report.duplicates,
                        "failed": report.failed,
                        "security_challenge": (
                            report.security_challenge
                        ),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print()
            print(f"Stopping reason: {report.stopped_because}")
            print(f"Summary: {report.message}")
            print(
                f"discovered={report.state.discovered} "
                f"persisted={report.state.persisted} "
                f"duplicates={report.state.duplicates} "
                f"failed={report.state.failed}"
            )

            if report.security_challenge:
                print()
                print("USER ACTION REQUIRED")
                print(
                    f"LinkedIn presented a "
                    f"{report.security_challenge} challenge."
                )
                print(
                    "Complete it in the browser window the collector "
                    "left open, then run:"
                )
                print("  python -m src.ingestion.collect run "
                      "--source linkedin --resume")
                return 2

        return 0 if report.succeeded else 1

    parser.print_help()

    return 1


if __name__ == "__main__":
    raise SystemExit(main())