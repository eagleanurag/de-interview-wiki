#!/usr/bin/env python3
"""
Assemble continuation context for an ``issue_comment`` event.

Pulls the original task and the most recent comments from the issue,
truncates them, and writes a JSON file the preflight step consumes.
Fetching is bounded: only the newest page of comments is requested and
each is length-capped before being written.

    python -m src.agent.issue_context \\
      --issue 42 \\
      --out context.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from src.agent.events import (
    MAX_COMMENT_CHARACTERS,
    MAX_ISSUE_TITLE_CHARACTERS,
    MAX_TASK_CHARACTERS,
    build_prior_context,
    truncate,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch bounded issue context for continuation."
    )

    parser.add_argument("--issue", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--repository", default="")

    return parser


def _gh_json(
    arguments: list[str],
    repository: str,
) -> object:
    command = ["gh", *arguments]

    if repository:
        command += ["--repo", repository]

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        shell=False,
    )

    if completed.returncode != 0:
        print(
            f"CONTEXT_FETCH_FAILED={completed.stderr.strip()[:500]}",
            file=sys.stderr,
        )
        return None

    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None


def fetch(
    issue: str,
    *,
    limit: int = 20,
    repository: str = "",
) -> dict:
    """Fetch the issue title, body and comments, bounded."""

    issue_payload = _gh_json(
        [
            "issue",
            "view",
            str(issue),
            "--json",
            "title,body",
        ],
        repository,
    )

    comments_payload = _gh_json(
        [
            "issue",
            "view",
            str(issue),
            "--json",
            "comments",
            "--comments",
            str(max(1, limit)),
        ],
        repository,
    )

    title = ""
    body = ""

    if isinstance(issue_payload, dict):
        title = str(issue_payload.get("title") or "")
        body = str(issue_payload.get("body") or "")

    comments: list[str] = []

    if isinstance(comments_payload, dict):
        raw = comments_payload.get("comments")

        if isinstance(raw, list):
            for entry in raw:
                if isinstance(entry, dict):
                    text = str(entry.get("body") or "")

                    if text.strip():
                        comments.append(
                            truncate(text, MAX_COMMENT_CHARACTERS)
                        )

    return {
        "issue_title": truncate(
            title, MAX_ISSUE_TITLE_CHARACTERS
        ),
        "original_task": truncate(body, MAX_TASK_CHARACTERS),
        "prior_comments": comments,
        "prior_context": build_prior_context(comments),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    payload = fetch(
        args.issue,
        limit=args.limit,
        repository=args.repository,
    )

    Path(args.out).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    print(
        f"CONTEXT_COMMENTS={len(payload.get('prior_comments', []))}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
