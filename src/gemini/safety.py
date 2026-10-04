"""
What may not leave this machine.

Two classes of leak matter here and they are different problems.

**Credentials.** Handled by reusing
:func:`src.ingestion.credentials.assert_no_credential_in_text`, which is
already called on the way into committed output. Nothing in this package
reads a credential, needs one, or is allowed to, but the check is
applied to the payload anyway: the cost is one pass over a string, and
the cost of being wrong is somebody's password in a public repository.

**Local filesystem paths.** These are the quieter failure. The Gemini
package and the LinkedIn archive both live in one person's Downloads
folder, and the natural way to write an import record is to store where
a file was found. That path names a Windows user account, and a
published knowledge base should not. Every record in this package
therefore stores a *logical* path such as ``media/<filename>`` -- and this
module is the check that one never got in.

The patterns are specific enough to be useful rather than merely
loud. A bare ``C:\\`` would fire on any published SQL snippet, and a
bar ``.env`` would fire on any discussion of dotfiles, so both are
matched only in the shapes that actually leak: a drive root, a user
directory, a downloads folder, or one of this project's own scratch and
virtual-environment directories.
"""

from __future__ import annotations

import re
from typing import Any

from src.ingestion.credentials import assert_no_credential_in_text


class PublicOutputError(RuntimeError):
    """A payload was about to be published with something private in it."""


#: Shapes that only occur when a local machine's layout has leaked into
#: data that will be published.
LOCAL_PATH_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "a Windows user profile",
        re.compile(r"(?i)\b[A-Z]:\\\\?Users\\\\?[A-Za-z0-9._-]+"),
    ),
    (
        "a Windows drive root",
        re.compile(r"(?i)\b[A-Z]:\\\\{1,2}(?![\\/])"),
    ),
    (
        "a UNC path",
        re.compile(r"\\\\[A-Za-z0-9._-]+\\[^\\\s]+"),
    ),
    (
        "a downloads folder",
        re.compile(r"(?i)\b(?:Downloads|Document[s]?|Desktop)\\"),
    ),
    (
        "this project's virtual environment",
        re.compile(r"(?i)(?:^|[\\/\s\"'])\.venv[\\/\s\"']"),
    ),
    (
        "this project's scratch directory",
        re.compile(r"(?i)(?:^|[\\/\s\"'])\.agent[\\/\s\"']"),
    ),
)

#: Where the pattern was found, for a message a person can act on.
MAX_REPORTED = 8


def find_local_paths(text: str) -> list[str]:
    """
    Every local-path shape in the text, described rather than quoted.

    The description is returned instead of the matched text because this
    function's own findings may end up in an error message that gets
    committed, and a message saying ``a Windows user profile`` is
    actionable where a message repeating the profile is not.
    """

    if not text:
        return []

    return [
        label
        for label, pattern in LOCAL_PATH_PATTERNS
        if pattern.search(text)
    ]


def assert_public(
    payload: Any,
    *,
    where: str,
    environ: dict[str, str] | None = None,
) -> None:
    """
    Refuse a payload that is not safe to publish.

    Walks strings, keys and mapping values alike. Keys are checked
    because a dict keyed by a filename is a natural place for a path to
    hide, and a payload whose *values* are clean can still be useless if
    its keys are not.

    Raises rather than warns, because every call site here is on the way
    to writing a file that gets committed and published. A warning
    nobody reads is the same as no check.
    """

    problems: list[str] = []

    def walk(node: Any, trail: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(key, f"{trail}.<key>")

                walk(value, f"{trail}[{key!r}]")

            return

        if isinstance(node, (list, tuple, set)):
            for position, value in enumerate(node):
                walk(value, f"{trail}[{position}]")

            return

        if not isinstance(node, str):
            return

        found = find_local_paths(node)

        if found:
            problems.append(f"{trail}: {', '.join(found[:3])}")

        assert_no_credential_in_text(node, environ)

    walk(payload, where)

    if problems:
        shown = "\n  ".join(problems[:MAX_REPORTED])

        raise PublicOutputError(
            f"Refusing to write {where}: it contains local filesystem "
            f"paths.\n  {shown}"
        )


__all__ = [
    "LOCAL_PATH_PATTERNS",
    "PublicOutputError",
    "assert_public",
    "find_local_paths",
]