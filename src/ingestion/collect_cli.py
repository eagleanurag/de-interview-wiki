#!/usr/bin/env python3
"""
Collection command line.

    python -m src.ingestion.collect doctor
    python -m src.ingestion.collect run --source manual \
        --bundle-root captures/ --max-posts 3
    python -m src.ingestion.collect run --source linkedin \
        --profile my-handle --max-posts 25 --headed
    python -m src.ingestion.collect status
    python -m src.ingestion.collect reset

``doctor`` reports whether a source is usable without touching a live
site, and never prints a credential value.
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


DEFAULT_POSTS_DIRECTORY = Path("data") / "posts"


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
        default="captures",
        help="Capture bundles, for the manual source.",
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


def build_source(args: argparse.Namespace):
    """Construct the requested source."""

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
    Let the user sign in by hand, once.

    The browser stays open and waits. Whatever the user completes,
    including a CAPTCHA or an OTP, is saved as a session so later runs
    reuse it. This is the supported path for any challenge, and it
    means the credentials are not typed by automation at all.
    """

    from src.ingestion.sources.linkedin import LinkedInSource

    print("Opening a browser for you to sign in.")
    print("Complete any challenge yourself, then return here.")
    print("Nothing is typed or solved automatically.")
    print()

    source = LinkedInSource(
        profile=args.profile,
        headed=True,
        limits=LinkedInLimits(max_posts=args.max_posts),
    )

    try:
        page = source.open_for_manual_login()
    except CollectionStopped as exc:
        print(f"Could not open the browser: {exc}")
        return 1

    print("Browser open. Sign in on the LinkedIn page.")
    print("Press Enter here once you are signed in...")

    try:
        input()
    except (EOFError, KeyboardInterrupt):
        print()

    if not source.wait_for_manual_session(timeout_seconds=600):
        print("Still not signed in. Nothing was saved.")
        return 1

    saved = source.save_session()

    source.close()

    if saved is None:
        print("Signed in, but the session could not be saved.")
        return 1

    print()
    print("Signed in. The session is saved inside the ignored")
    print(".agent tree and is never committed.")
    print("Run the collection again to use it.")

    return 0


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