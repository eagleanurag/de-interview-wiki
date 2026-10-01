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
from src.ingestion.post_document import PostDocument
from src.ingestion.saved_items.manifest import (
    ManifestUnreadable,
    manifest_path,
)
from src.ingestion.saved_items.source import SavedItemsSource, reconcile
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
        help="Report what would be imported without writing.",
    )

    saved.add_argument("--max-posts", type=int, default=None)
    saved.add_argument("--since", default=None)
    saved.add_argument("--until", default=None)

    saved.add_argument(
        "--json",
        action="store_true",
        help="Print the report as JSON.",
    )

    return parser


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

    if args.inputs:
        return Path(args.inputs[0]).expanduser().resolve().parent

    return Path(DEFAULT_SAVED_ITEMS_DIRECTORY)


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
    """
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
        print("Export your saved items, or write a .csv with a url")
        print("column, and put it in that directory. Example:")
        print("  URL,Saved Date,Title,Notes")
        print("  https://www.linkedin.com/posts/... ,2026-01-02,,")
        return 1

    state_file = (
        Path(args.manifest_file)
        if args.manifest_file
        else manifest_path(bundle_root)
    )

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
    if not args.dry_run:
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
            print(
                "To add one, create a directory named after it in:"
            )
            print(f"  {bundle_root}")
            print(
                "and put the text, HTML, PDF or a screenshot inside."
            )

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

    if args.validate and not args.dry_run:
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