"""
Diagnostics for the Saved Items inbox.

A user drops a folder in and it does not work. "ERROR" tells them
nothing they can act on, and a stack trace tells them less: the cause is
almost never a bug, it is a folder that does not say which saved item it
belongs to.

So every problem here is reported the same way: what was found, where it
was, why it could not be used, and the smallest change that would fix
it. The format is fixed so it can be read by a person and matched by a
test.

Nothing here inspects a file the pipeline would not otherwise read, and
no fix is ever applied. Reporting a problem is the whole job; changing
the user's material is not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.ingestion.saved_items.bundles import CONTAINER_NAMES
from src.ingestion.saved_items.capture import (
    CAPTURE_ID_KEYS,
    CAPTURE_URL_KEYS,
    capture_json_example,
    credential_fields,
)
from src.ingestion.saved_items.model import SavedItem
from src.ingestion.saved_items.urls import (
    SavedItemUrlError,
    normalize_linkedin_url,
)


#: How serious a problem is.
#:
#: An error means something is wrong that the user has to decide about.
#: A warning means something is worth knowing that does not stop the
#: import. Keeping them apart matters because a command that reports
#: both as failures is a command nobody trusts.
SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"

LEVEL_ERROR = SEVERITY_ERROR
LEVEL_WARNING = SEVERITY_WARNING


@dataclass
class Diagnostic:
    """
    One thing the user can act on.

    Carries the code so a test can assert on the cause without asserting
    on wording, and so a future reader of a report can look up what
    ``ORPHAN_CAPTURE`` means.
    """

    code: str
    severity: str
    location: str
    reason: str
    fix: str
    detail: dict = field(default_factory=dict)

    @property
    def is_error(self) -> bool:
        return self.severity == SEVERITY_ERROR

    def render(self) -> str:
        """
        The problem, the reason, and the fix.

        Labelled sections rather than a sentence, because the reader is
        looking for one of the three specifically and a paragraph makes
        them read all of it to find out which.
        """
        return "\n".join(
            [
                self.code,
                "",
                "Where:",
                f"  {self.location}",
                "",
                "Reason:",
                *[f"  {line}" for line in self.reason.splitlines()],
                "",
                "Fix:",
                *[f"  {line}" for line in self.fix.splitlines()],
            ]
        )

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "severity": self.severity,
            "location": self.location,
            "reason": self.reason,
            "fix": self.fix,
            **({"detail": self.detail} if self.detail else {}),
        }


#: The codes this module raises. Named so a report can be read without
#: guessing, and so a test can assert the cause rather than the prose.
UNREADABLE_MANIFEST = "MANIFEST_UNREADABLE"
INVALID_ROW = "INVALID_ROW"
MISSING_URL = "MISSING_URL"
INVALID_URL = "INVALID_URL"
NO_URL_COLUMN = "NO_URL_COLUMN"
ORPHAN_CAPTURE = "ORPHAN_CAPTURE"
UNCLAIMED_CAPTURE = "UNCLAIMED_CAPTURE"
MISSING_CAPTURE = "MISSING_CAPTURE"
CREDENTIAL_FIELD = "CREDENTIAL_FIELD"
MALFORMED_CAPTURE = "MALFORMED_CAPTURE"
UNSUPPORTED_FILE = "UNSUPPORTED_FILE"
UNREADABLE_FILE = "UNREADABLE_FILE"
EMPTY_CAPTURE = "EMPTY_CAPTURE"
ESCAPES_ROOT = "ESCAPES_ROOT"
DUPLICATE_ROW = "DUPLICATE_ROW"
CAPTURE_NOT_IN_MANIFEST = "CAPTURE_NOT_IN_MANIFEST"


#: Suffixes the pipeline understands, for the "unsupported file" advice.
SUPPORTED_SUFFIXES = (
    ".md", ".txt", ".json", ".jsonl", ".html", ".htm",
    ".pdf", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp",
)


def unsupported(path: Path) -> bool:
    """Whether a file is something the pipeline will not read."""
    return path.suffix.lower() not in SUPPORTED_SUFFIXES


def capture_json_example(url: str = "https://www.linkedin.com/posts/...") -> str:
    """A capture file the user can copy, containing nothing they must not."""
    return json.dumps(
        {"url": url}, indent=2
    )


def orphan_diagnostic(
    folder: str | Path,
    *,
    detected_url: str | None = None,
    detected_source_id: str | None = None,
) -> Diagnostic:
    """
    A capture nothing claims.

    Never guesses a URL. The advice names the two ways to attach a
    folder, and if a URL was actually found in the folder it is offered
    as the thing to put in the manifest rather than being used on its
    own.
    """
    location = str(folder)

    if detected_url:
        reason = (
            "This capture names a LinkedIn URL, but the saved-items "
            "manifest has no item for it yet."
        )

        fix = "\n".join(
            [
                "Add it to the manifest, then run the import again:",
                "",
                f"  {detected_url}",
                "",
                "Or add capture.json to the folder containing:",
                "",
                capture_json_example(detected_url),
            ]
        )

        return Diagnostic(
            code=ORPHAN_CAPTURE,
            severity=SEVERITY_WARNING,
            location=location,
            reason=reason,
            fix=fix,
            detail={
                "detected_url": detected_url,
                "detected_source_id": detected_source_id or "",
            },
        )

    reason = (
        "No source_id or LinkedIn URL was found in this folder, so it "
        "cannot be attached to a saved item. Nothing was imported from "
        "it and nothing was deleted."
    )

    fix = "\n".join(
        [
            "Add capture.json to this folder containing at least:",
            "",
            capture_json_example(),
            "",
            "Alternatively, rename the folder to the saved item's id.",
            "The 'saved-items-status' command prints the id for every "
            "item still waiting for a capture.",
        ]
    )

    return Diagnostic(
        code=ORPHAN_CAPTURE,
        severity=SEVERITY_WARNING,
        location=location,
        reason=reason,
        fix=fix,
        detail={"detected_source_id": detected_source_id or ""},
    )


def missing_capture_diagnostic(item: SavedItem) -> Diagnostic:
    """A saved item with no capture, which is a backlog rather than a fault."""
    return Diagnostic(
        code=MISSING_CAPTURE,
        severity=SEVERITY_WARNING,
        location=item.canonical_url,
        reason=(
            "This item is a saved link with no content behind it. It is "
            "kept as a link and stays pending; no post body was created "
            "for it, because there was nothing to create one from."
        ),
        fix="\n".join(
            [
                "To capture it, create a folder named after this item:",
                "",
                f"  captures/{item.source_id.replace(':', '-')}",
                "",
                "and put the post's text, saved page, PDF or a "
                "screenshot inside.",
            ]
        ),
        detail={"source_id": item.source_id},
    )


def credential_diagnostic(location: str, fields: list[str]) -> Diagnostic:
    """
    A capture or list carrying something that looks like a credential.

    The values are never echoed. Naming the field is enough to act on
    and does not put the value into a log, a report or a commit.
    """
    names = ", ".join(sorted(fields))

    return Diagnostic(
        code=CREDENTIAL_FIELD,
        severity=SEVERITY_WARNING,
        location=location,
        reason=(
            f"Field(s) named like a credential were present and were not "
            f"read: {names}. A saved-items list and a capture do not "
            "contain credentials, so nothing was stored."
        ),
        fix=(
            "Remove those fields from the file. Nothing in this project "
            "needs them, and a saved list that carries one is not "
            "something to keep."
        ),
        detail={"fields": sorted(fields)},
    )


def credential_diagnostic_for(
    location: str,
    payload: dict,
) -> Diagnostic | None:
    """A credential problem in a capture, or None when there is none."""
    fields = credential_fields(payload)

    if not fields:
        return None

    return credential_diagnostic(location, fields)


def invalid_url_diagnostic(location: str, reason: str) -> Diagnostic:
    return Diagnostic(
        code=INVALID_URL,
        severity=SEVERITY_ERROR,
        location=location,
        reason=reason,
        fix=(
            "Check the address. A saved post looks like:\n"
            "\n"
            "  https://www.linkedin.com/posts/...\n"
            "  https://www.linkedin.com/pulse/...\n"
            "  https://www.linkedin.com/feed/update/urn:li:activity:..."
        ),
    )


def malformed_capture_diagnostic(
    location: str,
    reason: str,
    *,
    field_name: str = "capture.json",
) -> Diagnostic:
    return Diagnostic(
        code=MALFORMED_CAPTURE,
        severity=SEVERITY_ERROR,
        location=location,
        reason=f"{field_name} could not be used: {reason}",
        fix="\n".join(
            [
                f"{field_name} must be a JSON object with at least a "
                "url, and nothing that names a credential:",
                "",
                capture_json_example(),
            ]
        ),
    )


def unsupported_file_diagnostic(path: str | Path) -> Diagnostic:
    name = Path(path).name

    return Diagnostic(
        code=UNSUPPORTED_FILE,
        severity=SEVERITY_WARNING,
        location=str(path),
        reason=(
            f"{name} is not a format this project reads, so it was "
            "left alone rather than guessed at."
        ),
        fix=(
            "Supported formats are: "
            + ", ".join(SUPPORTED_SUFFIXES)
            + ".\n"
            "Anything else stays where it is; it is not deleted."
        ),
        detail={"name": name},
    )


def escapes_root_diagnostic(location: str, root: str | Path) -> Diagnostic:
    return Diagnostic(
        code=ESCAPES_ROOT,
        severity=SEVERITY_ERROR,
        location=location,
        reason=(
            "This path resolves outside the capture root, so it was "
            "refused. A capture is user-supplied content and must not be "
            "able to name a file anywhere on the machine."
        ),
        fix=(
            f"Move the capture inside {root} and remove any symbolic "
            "link that points elsewhere."
        ),
    )


def empty_capture_diagnostic(location: str) -> Diagnostic:
    return Diagnostic(
        code=EMPTY_CAPTURE,
        severity=SEVERITY_WARNING,
        location=location,
        reason=(
            "Nothing in this folder could be read, so it was not "
            "treated as a capture. Nothing was deleted."
        ),
        fix=(
            "Put at least one readable file inside: text, a saved page, "
            "a PDF, or a screenshot. A file that will not open is "
            "reported separately rather than counted as a capture."
        ),
    )


def describe_containers() -> str:
    """The folder names treated as containers of captures."""
    return ", ".join(sorted(CONTAINER_NAMES))


def diagnostic_from_exception(
    location: str,
    exc: Exception,
) -> Diagnostic:
    """
    Turn an unexpected failure into something readable.

    Used where a failure is not one of the known shapes. The message is
    the exception's own, which is enough to act on, and no traceback is
    attached: a stack trace in a user-facing report hides the one line
    that says what to do.
    """
    from src.ingestion.saved_items.bundles import BundleError

    if isinstance(exc, BundleError):
        return Diagnostic(
            code=UNREADABLE_FILE,
            severity=SEVERITY_ERROR,
            location=location,
            reason=str(exc),
            fix=(
                "Move the capture inside the bundle root and remove any "
                "symbolic link that points elsewhere."
            ),
        )

    if isinstance(exc, SavedItemUrlError):
        return invalid_url_diagnostic(location, str(exc))

    return Diagnostic(
        code=UNREADABLE_FILE,
        severity=SEVERITY_ERROR,
        location=location,
        reason=f"{type(exc).__name__}: {exc}",
        fix=(
            "Re-run with --debug to see the full traceback. The capture "
            "was skipped; nothing was deleted."
        ),
    )


def capture_claim_from_json(payload: dict) -> tuple[str | None, str | None]:
    """
    The url and source id a capture file claims, if it claims either.

    Reports what was found and nothing more. A URL is only accepted once
    it normalizes, so a capture cannot attach itself to an item with a
    link that does not resolve to a real post.
    """
    url: str | None = None

    for key in CAPTURE_URL_KEYS:
        value = payload.get(key)

        if isinstance(value, str) and value.strip():
            try:
                url = normalize_linkedin_url(value).canonical
            except SavedItemUrlError:
                url = None

            if url:
                break

    source_id: str | None = None

    for key in CAPTURE_ID_KEYS:
        value = payload.get(key)

        if isinstance(value, str) and value.strip():
            source_id = value.strip()
            break

    return url, source_id


__all__ = [
    "CAPTURE_NOT_IN_MANIFEST",
    "CREDENTIAL_FIELD",
    "DUPLICATE_ROW",
    "EMPTY_CAPTURE",
    "ESCAPES_ROOT",
    "INVALID_ROW",
    "INVALID_URL",
    "LEVEL_ERROR",
    "LEVEL_WARNING",
    "MALFORMED_CAPTURE",
    "MISSING_CAPTURE",
    "MISSING_URL",
    "NO_URL_COLUMN",
    "ORPHAN_CAPTURE",
    "SEVERITY_ERROR",
    "SEVERITY_WARNING",
    "UNCLAIMED_CAPTURE",
    "UNREADABLE_FILE",
    "UNREADABLE_MANIFEST",
    "UNSUPPORTED_FILE",
    "Diagnostic",
    "capture_json_example",
    "capture_claim_from_json",
    "credential_diagnostic",
    "credential_diagnostic_for",
    "describe_containers",
    "diagnostic_from_exception",
    "empty_capture_diagnostic",
    "escapes_root_diagnostic",
    "invalid_url_diagnostic",
    "malformed_capture_diagnostic",
    "missing_capture_diagnostic",
    "orphan_diagnostic",
    "unsupported",
    "unsupported_file_diagnostic",
]
