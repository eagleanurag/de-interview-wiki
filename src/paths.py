"""
Deciding whether a path from a file this project did not write stays
inside a root it is allowed to touch.

The archive's own metadata is data. Its paths are read on whatever
machine runs the pipeline, which is not necessarily the machine that
wrote them, and the two disagree about what a path means.

``pathlib`` on POSIX reads ``..\\outside.jpg`` as a *single filename*,
because a backslash is an ordinary character in a POSIX filename. So on
Linux ``root / "..\\\\outside.jpg"`` resolves to a file inside the root,
``relative_to`` succeeds, and nothing is refused -- while the identical
string on Windows walks out and is refused. ``C:/Windows/System32/cmd.exe``
has the same problem from the other side: it is absolute on Windows and
merely a relative name with a funny first component on POSIX.

That is not a theoretical difference. Three tests were written to
assert the safe behaviour and passed on the machine that wrote them,
and the same three failed on the Linux runner. The safe answer has to be
the same answer everywhere, so the decision is made here once, against
the stricter reading, rather than inherited from the platform.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


#: A Windows drive path in either separator style: ``C:\\x``, ``C:/x``.
#:
#: The separator is required, so an ordinary name that happens to
#: contain a colon -- ``A:notes.txt`` -- is not mistaken for one.
DRIVE_PREFIX = re.compile(r"^[A-Za-z]:[\\/]")

#: A UNC path or a Windows namespace path: ``\\\\?\\C:\\x``,
#: ``\\\\server\\share``, ``//server/share``.
#:
#: A leading ``//`` counts because POSIX reserves it and the path means
#: nothing portable either way.
NAMESPACE_PREFIX = re.compile(r"^(?:[\\/]{2})")


def candidate_path(candidate: str | Path) -> Path | None:
    """
    The candidate as a ``Path``, or ``None`` when it can never be inside
    a root.

    ``None`` rather than an exception, because the two callers already
    raise their own distinct types for the same condition -- an archive
    that cannot be trusted is a different failure from an asset that
    escaped -- and each words the reason in its own terms.

    Three shapes are refused outright on every platform:

    * a drive path, ``C:\\Windows\\System32\\cmd.exe``;
    * a UNC or namespace path, ``\\\\?\\C:\\x`` or ``//server/share/x``;
    * and, by the substitution below, anything that walks out with
      ``..`` written in either separator style.

    The substitution is the part worth arguing for: on POSIX a backslash
    is folded to a forward slash so ``..\\x`` traverses as it would on
    Windows. The cost is that a genuinely backslash-bearing POSIX
    filename would be read as a path. That trade is free here, because
    the archive's naming convention is ``{stem}_slide_{n}.jpg`` and
    ``{activity}_{id}.jpg`` -- neither can contain a backslash -- while
    the alternative is accepting a traversal string on the runner that
    publishes the site.
    """

    text = str(candidate)

    if DRIVE_PREFIX.match(text) or NAMESPACE_PREFIX.match(text):
        return None

    if os.sep != "\\":
        text = text.replace("\\", "/")

    return Path(text)