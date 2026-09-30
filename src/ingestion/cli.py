"""
Command line access to the ingestion layer.

    python -m src.ingestion.cli new <post-id> --text-file notes.md
    python -m src.ingestion.cli add-media <post-id> shot.png notes.pdf
    python -m src.ingestion.cli import <post-id> ~/captures/databricks
    python -m src.ingestion.cli list
    python -m src.ingestion.cli validate

This is the supported way to add manually captured content. It never
reaches the network: everything it does is local file work under
``data/posts/``, which it refuses to leave.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.ingestion.errors import IngestionError
from src.ingestion.importer import (
    DEFAULT_POSTS_ROOT,
    PostImportResult,
    add_media,
    create_post,
    discover_posts,
    import_post,
    post_directory,
    posts_root,
    validate_post,
    validate_posts,
)
from src.ingestion.post_document import (
    DEFAULT_DOMAIN,
    DEFAULT_PLATFORM,
    normalize_post_id,
)
from src.ingestion.validation import (
    ValidationIssue,
    describe_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.ingestion.cli",
        description=(
            "Import manually captured interview content into "
            "data/posts/."
        ),
    )

    parser.add_argument(
        "--root",
        default=str(DEFAULT_POSTS_ROOT),
        help=(
            "Directory holding one subdirectory per post "
            f"(default: {DEFAULT_POSTS_ROOT})"
        ),
    )

    commands = parser.add_subparsers(dest="command")

    new = commands.add_parser(
        "new",
        help="Create a post from text you already have.",
    )
    _add_root(new)
    new.add_argument("post_id")
    _add_provenance(new)
    _add_classification(new)
    new.add_argument(
        "--text",
        default=None,
        help="Captured text, inline.",
    )
    new.add_argument(
        "--text-file",
        default=None,
        help="File holding the captured text.",
    )
    new.add_argument(
        "--allow-empty",
        action="store_true",
        help=(
            "Create a shell with no content yet. It stays flagged by "
            "`validate` until text or media is added."
        ),
    )
    new.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing post.json.",
    )

    media = commands.add_parser(
        "add-media",
        help="Attach media files to an existing post.",
    )
    _add_root(media)
    media.add_argument("post_id")
    media.add_argument(
        "sources",
        nargs="+",
        help="Files or directories to attach.",
    )
    media.add_argument(
        "--description",
        default="",
        help="Description recorded for every file added.",
    )
    media.add_argument(
        "--force",
        action="store_true",
        help=(
            "Replace a file that already exists with "
            "different content."
        ),
    )

    capture = commands.add_parser(
        "import",
        help=(
            "Import a capture bundle as a post, creating or "
            "updating it."
        ),
    )
    capture.add_argument("post_id")
    capture.add_argument(
        "bundle",
        help=(
            "Directory holding the capture, or a single text file. "
            "A bundle may contain its own post.json."
        ),
    )
    _add_provenance(capture)
    _add_classification(capture)
    capture.add_argument(
        "--text",
        default=None,
        help="Captured text, overriding any notes file in the bundle.",
    )
    capture.add_argument(
        "--text-file",
        default=None,
        help="File holding the captured text.",
    )
    capture.add_argument(
        "--description",
        default="",
        help="Description recorded for every media file added.",
    )
    capture.add_argument(
        "--force",
        action="store_true",
        help="Replace a media file that exists with different content.",
    )
    _add_root(capture)

    listing = commands.add_parser(
        "list",
        help="List the posts the pipeline will discover.",
    )
    _add_root(listing)

    validate = commands.add_parser(
        "validate",
        help="Check every discovered post. Exits non-zero on errors.",
    )
    _add_root(validate)
    validate.add_argument(
        "post_ids",
        nargs="*",
        help="Validate only these posts instead of the whole tree.",
    )

    return parser


def _add_root(parser: argparse.ArgumentParser) -> None:
    """
    Accept ``--root`` after the subcommand as well as before it.

    SUPPRESS matters: without it an unset value on the subparser would
    overwrite the top-level one, so `cli --root X validate` and
    `cli validate --root X` could disagree. Both spellings are natural
    to type, and the pipeline uses the second.
    """

    parser.add_argument(
        "--root",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )


def _add_provenance(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--platform",
        default=None,
        help=(
            "Where the post came from, recorded as provenance "
            f"(default: {DEFAULT_PLATFORM})"
        ),
    )
    parser.add_argument("--url", default=None, help="Source URL.")
    parser.add_argument(
        "--author",
        default=None,
        help="Who wrote or posted the content.",
    )
    parser.add_argument(
        "--captured-at",
        default=None,
        help="ISO-8601 capture time (default: now).",
    )


def _add_classification(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--domain",
        default=None,
        help=f"Domain (default: {DEFAULT_DOMAIN}).",
    )
    parser.add_argument(
        "--primary-topic",
        default=None,
        help="Primary topic, for example 'Databricks'.",
    )
    parser.add_argument(
        "--secondary-topic",
        action="append",
        default=[],
        help="Secondary topic. Repeatable.",
    )
    relevant = parser.add_mutually_exclusive_group()
    relevant.add_argument(
        "--interview-relevant",
        dest="interview_relevant",
        action="store_const",
        const=True,
        default=None,
        help="Mark the post as interview relevant.",
    )
    relevant.add_argument(
        "--not-interview-relevant",
        dest="interview_relevant",
        action="store_const",
        const=False,
        help="Mark the post as not interview relevant.",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    root = posts_root(args.root)

    try:
        if args.command == "new":
            return _new(args, root)

        if args.command == "add-media":
            return _add_media(args, root)

        if args.command == "import":
            return _import(args, root)

        if args.command == "list":
            return _list(root)

        return _validate(args, root)

    except IngestionError as exc:
        print(f"INGESTION_ERROR={exc}", file=sys.stderr)
        return 1


def _new(args, root: Path) -> int:
    identifier = normalize_post_id(args.post_id)

    text = args.text or ""

    if args.text_file:
        text = Path(args.text_file).read_text(
            encoding="utf-8-sig"
        ).strip()

    if not text and not args.text_file and not args.allow_empty:
        text = _stdin_text()

    create_post(
        identifier,
        text=text,
        root=root,
        platform=args.platform,
        url=args.url,
        author=args.author,
        captured_at=args.captured_at,
        domain=args.domain,
        primary_topic=args.primary_topic,
        secondary_topics=tuple(args.secondary_topic),
        interview_relevant=args.interview_relevant,
        overwrite=args.overwrite,
        allow_empty=args.allow_empty,
    )

    print(f"Created post: {_display(post_directory(identifier, root))}")

    return 0


def _add_media(args, root: Path) -> int:
    identifier = normalize_post_id(args.post_id)

    result = add_media(
        identifier,
        args.sources,
        root=root,
        description=args.description,
        force=args.force,
    )

    print(f"Post: {_display(result.directory)}")

    for added in result.added:
        print(f"  added   {added}")

    for skipped in result.skipped:
        print(f"  present {skipped} (unchanged)")

    for original, committed in result.renamed:
        print(f"  renamed {original!r} -> {committed}")

    if not result.changed and not result.skipped:
        print("  nothing to add")

    return 0


def _import(args, root: Path) -> int:
    identifier = normalize_post_id(args.post_id)

    result: PostImportResult = import_post(
        identifier,
        args.bundle,
        root=root,
        text=args.text,
        text_file=args.text_file,
        platform=args.platform,
        url=args.url,
        author=args.author,
        captured_at=args.captured_at,
        domain=args.domain,
        primary_topic=args.primary_topic,
        secondary_topics=tuple(args.secondary_topic),
        interview_relevant=args.interview_relevant,
        description=args.description,
        force=args.force,
    )

    verb = "Created" if result.created else "Updated"

    print(f"{verb} post: {_display(result.directory)}")

    if result.text_from:
        print(f"  text    {result.text_from}")

    for added in result.media_added:
        print(f"  media   {added}")

    if result.media:
        for skipped in result.media.skipped:
            print(f"  present {skipped} (unchanged)")

        for original, committed in result.media.renamed:
            print(f"  renamed {original!r} -> {committed}")

    for note in result.notes:
        print(f"  note    {note}")

    return 0


def _list(root: Path) -> int:
    posts = discover_posts(root)

    if not posts:
        print(f"No posts under {_display(root)}")
        return 0

    print(f"Posts under {_display(root)}:")

    for post in posts:
        if not post.has_post_file:
            print(f"  - {post.post_id} (no post.json)")
            continue

        print(
            f"  - {post.post_id}: "
            f"{post.text_length} characters of text, "
            f"{post.declared_media} declared / "
            f"{post.media_count} media file(s)"
        )

    return 0


def _validate(args, root: Path) -> int:
    if args.post_ids:
        issues: list[ValidationIssue] = []

        for post_id in args.post_ids:
            issues.extend(
                validate_post(post_id, root=root)
            )

        return _print_issues(issues)

    report = validate_posts(root=root)

    for issue in report.issues:
        _print_issue(issue)

    print(describe_report(report))

    return 0 if report.ok else 1


def _print_issue(issue: ValidationIssue) -> None:
    print(
        f"{issue.level.upper()}: {issue.post_id}: {issue.message}"
    )


def _print_issues(issues: list[ValidationIssue]) -> int:
    for issue in issues:
        _print_issue(issue)

    errors = sum(1 for issue in issues if issue.is_error)
    warnings = len(issues) - errors

    print(
        f"{len(issues)} finding(s): {errors} error(s), "
        f"{warnings} warning(s)"
    )

    return 0 if errors == 0 else 1


def _stdin_text() -> str:
    """
    Read piped text, so a capture can be pasted straight in.

    Nothing is read from an interactive terminal, and an unreadable
    stream is treated as empty rather than being allowed to interrupt
    the import.
    """

    stream = sys.stdin

    if stream is None or stream.isatty():
        return ""

    try:
        return stream.read().strip()
    except (OSError, ValueError):
        return ""


def _display(path: Path) -> str:
    """Show a path relative to the working directory when possible."""

    try:
        return path.resolve().relative_to(
            Path.cwd().resolve()
        ).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
