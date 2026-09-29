#!/usr/bin/env python3
"""
Preflight: decide whether an event may run the agent, and emit the
resolved task as a JSON file for later steps.

Called by `.github/workflows/opencode-agent.yml`. Exits 0 when the
agent should run, and 78 when the event must be ignored, so the calling
step can short-circuit without failing the workflow.

    python -m src.agent.preflight \\
      --event-name issues \\
      --actor "${{ github.event.issue.user.login }}" \\
      --owner "${{ github.repository_owner }}" \\
      --title "${{ github.event.issue.title }}" \\
      --body-file issue-body.md \\
      --issue-number "${{ github.event.issue.number }}" \\
      --out trigger.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.agent.events import (
    Trigger,
    TriggerRejected,
    authorize_comment_event,
    authorize_dispatch_event,
    authorize_issue_event,
    build_prior_context,
)


EXIT_OK = 0
EXIT_ERROR = 1
EXIT_IGNORED = 78


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Authorize a GitHub event and write the resolved trigger."
        )
    )

    parser.add_argument(
        "--event-file",
        default="",
        help=(
            "JSON document describing the event. Preferred, because "
            "no untrusted value passes through the shell."
        ),
    )
    parser.add_argument("--event-name", default="")
    parser.add_argument("--actor", default="")
    parser.add_argument("--owner", default="")
    parser.add_argument("--issue-number", default="")
    parser.add_argument("--title", default="")
    parser.add_argument("--body-file", default="")
    parser.add_argument("--task", default="")
    parser.add_argument(
        "--context-file",
        default="",
        help=(
            "JSON file with prior_comments and original_task, used "
            "for continuation events."
        ),
    )
    parser.add_argument("--out", default="trigger.json")

    return parser


def _read_text(path: str) -> str:
    if not path:
        return ""

    file = Path(path)

    if not file.is_file():
        return ""

    return file.read_text(encoding="utf-8", errors="replace")


def _read_context(path: str) -> dict:
    if not path:
        return {}

    file = Path(path)

    if not file.is_file():
        return {}

    try:
        payload = json.loads(
            file.read_text(encoding="utf-8", errors="replace")
        )
    except json.JSONDecodeError:
        return {}

    return payload if isinstance(payload, dict) else {}


def _issue_number(value: str) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def load_event(path: str) -> dict:
    """
    Read the event document produced by the workflow.

    An unreadable or malformed document is an empty mapping, so the
    authorization functions deny rather than raise on missing data.
    """

    if not path:
        return {}

    file = Path(path)

    if not file.is_file():
        return {}

    try:
        payload = json.loads(
            file.read_text(encoding="utf-8", errors="replace")
        )
    except json.JSONDecodeError:
        return {}

    return payload if isinstance(payload, dict) else {}


def resolve(args: argparse.Namespace) -> Trigger:
    """
    Map the event onto the right authorization function.

    ``--event-file`` is authoritative when present. Individual flags
    exist for local testing and are only consulted when no event
    document was supplied.
    """

    context = _read_context(args.context_file)
    payload = load_event(args.event_file)

    if payload:
        event_name = str(payload.get("event") or "")
        actor = str(payload.get("actor") or "")
        title = str(payload.get("title") or "")
        task = str(payload.get("task") or "")
        number = _issue_number(payload.get("issue_number"))
        body = str(payload.get("body") or "")
        ref = str(payload.get("ref") or "")

        if event_name == "issues":
            return authorize_issue_event(
                actor=actor,
                owner=args.owner,
                title=title,
                body=body,
                issue_number=number,
            )

        if event_name == "issue_comment":
            return authorize_comment_event(
                actor=actor,
                owner=args.owner,
                body=body,
                issue_number=number,
                original_task=str(context.get("original_task", "")),
                issue_title=title,
                prior_comments=list(context.get("prior_comments", [])),
            )

        if event_name == "workflow_dispatch":
            trigger = authorize_dispatch_event(
                actor=actor,
                owner=args.owner,
                task=task,
                issue_number=number,
            )

            # Preserve the requested branch so the workflow can push
            # and validate on it.
            if ref:
                trigger.branch = ref

            return trigger

        raise TriggerRejected(
            f"unsupported event name: {event_name!r}"
        )

    body = _read_text(args.body_file)
    number = _issue_number(args.issue_number)

    if args.event_name == "issues":
        return authorize_issue_event(
            actor=args.actor,
            owner=args.owner,
            title=args.title,
            body=body,
            issue_number=number,
        )

    if args.event_name == "issue_comment":
        return authorize_comment_event(
            actor=args.actor,
            owner=args.owner,
            body=body,
            issue_number=number,
            original_task=str(context.get("original_task", "")),
            issue_title=str(context.get("issue_title", "")),
            prior_comments=list(context.get("prior_comments", [])),
        )

    if args.event_name == "workflow_dispatch":
        return authorize_dispatch_event(
            actor=args.actor,
            owner=args.owner,
            task=args.task,
            issue_number=number,
        )

    raise TriggerRejected(
        f"unsupported event name: {args.event_name}"
    )


def trigger_to_dict(trigger: Trigger) -> dict:
    return {
        "kind": trigger.kind,
        "task": trigger.task,
        "instruction": trigger.instruction,
        "issue_number": trigger.issue_number,
        "issue_title": trigger.issue_title,
        "actor": trigger.actor,
        "prior_context": trigger.prior_context,
        "branch": trigger.branch,
        "summary": trigger.summary,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        trigger = resolve(args)
    except TriggerRejected as exc:
        print(f"IGNORED: {exc.reason}")
        print(
            json.dumps(
                {
                    "authorized": False,
                    "reason": exc.reason,
                    "event": args.event_name,
                },
                indent=2,
            )
        )
        return EXIT_IGNORED
    except Exception as exc:  # noqa: BLE001
        print(f"PREFLIGHT_ERROR={exc}", file=sys.stderr)
        return EXIT_ERROR

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)

    payload = {"authorized": True, **trigger_to_dict(trigger)}

    output.write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    print(f"TRIGGER_KIND={trigger.kind}")
    print(f"ISSUE_NUMBER={trigger.issue_number or ''}")
    print(f"ACTOR={trigger.actor}")
    print(f"TASK_SUMMARY={trigger.summary}")

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
