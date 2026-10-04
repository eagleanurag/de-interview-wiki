"""
Removing local filesystem paths from text that will be published.

Shared, and deliberately not owned by any one layer. Two callers need it
and neither is the right owner:

* the image-knowledge importer, because a transcription of a Windows
  command prompt contains a path;
* the wiki, because a captured post body is reproduced verbatim and
  would carry one to a published page.

It lives at the top level so ``src.gemini`` and ``src.wiki`` can both
reach it without either importing the other. A redaction rule that only
one of them can reach is a rule that protects one of them.

It does not own the decision about what to call the result. Keeping the
original and labelling the published copy is each caller's job, because
only the caller knows which field is being written.

One narrow problem, one narrow fix. OCR of a Windows command prompt comes
back as something like::

    C:\\Users\\Your Name\\AppData\\Local\\Programs\\Python\\Python36-32\\Scripts>pip install requests

That contains a local filesystem path, so it cannot be committed or
published as-is. But the *command* after the prompt is the entire
technical content of the slide, and a filter that removed the whole line
would throw away ``pip install requests`` along with the path.

So only the path component is replaced, and it is replaced with a marker
that says what it was rather than deleting it silently::

    <LOCAL_WINDOWS_PATH>>pip install requests

A reader can see that a path was there, and the command survives intact.

**What is deliberately not touched.** Measured against this package's own
transcriptions, the things most likely to be broken by an over-eager
filter are all preserved:

* ``spark-submit --master local[*]`` -- the word "local" is not a path.
* ``abfss://container@storage.dfs.core.windows.net/...``
* ``https://...``, ``s3://...``, ``mongodb://localhost:27017/``
* ``pip install requests``, ``python -m pip install pymongo``
* SQL, of any shape.

**How a Windows path is delimited.** The difficult part is that
``Your Name`` contains a space, so a path cannot stop at the first
whitespace without truncating ``C:\\Users\\Your``. A path therefore ends
at one of three things: the ``>`` of a command prompt, a run of two or
more spaces (a paragraph break), or the end of the line. All three are
visible structure rather than guesswork. The segment count is bounded so
a stray drive letter in prose cannot swallow the rest of a paragraph.

This function returns sanitised text and a count of what it removed. It
does not decide what to call the result.
"""

from __future__ import annotations

import re


#: What replaced each kind of removed path. Distinct markers, so a reader
#: -- and the redaction count -- can tell a Windows path from a UNC share
#: from an environment variable.
LOCAL_PATH_MARKER = "<LOCAL_WINDOWS_PATH>"
UNC_PATH_MARKER = "<UNC_PATH>"
ENV_PATH_MARKER = "<LOCAL_PATH_ENV_VAR>"
PROJECT_PATH_MARKER = "<LOCAL_PROJECT_PATH>"

#: A Windows drive path. The segment character class deliberately excludes
#: ``>``, which is what ends a cmd.exe prompt, and allows spaces so that
#: ``C:\Users\Your Name\...`` is not truncated at ``Your``.
#:
#: The separator is ``+`` rather than a fixed width because OCR frequently
#: doubles a backslash: the same slide reads ``c:\Users\Your Name>python``
#: on one line and ``c:\\Users\\Your Name>python`` on the next, and a
#: two-character separator left the surplus in the output.
#: One character of a path segment, with a guard.
#:
#: A space belongs *inside* a path in ``C:\Users\Your Name`` and
#: *between* two paths in ``copy C:\Users\Alice\a.txt D:\Users\Bob\b.txt``.
#: The two are indistinguishable by looking at the space alone, so each
#: character is required not to be the start of a drive specification.
#: That single guard resolves both measured cases correctly:
#:
#:     C:\Users\Your Name\...        space before "Name" -> next is "Na", fine
#:     ...\a.txt D:\Users\...        space before "D:"  -> blocked, so the
#:                                   first path ends at "a.txt"
#:
#: Getting this wrong is not cosmetic: the earlier version swallowed the
#: second drive letter and left ``:\Users\Bob\b.txt`` -- half a path --
#: in the published text.
#: OCR doubles, and sometimes quadruples, backslashes in a Windows path.
#:
#: Normalising them before matching is what makes the rest of this module
#: simple. It is why a drive path can anchor on a single ``X:`` and why
#: the UNC rule can run first without swallowing a doubled drive path's
#: separator pair -- which it did, leaving ``c<UNC_PATH>>python``.
#:
#: Only backslashes are collapsed. Forward slashes are left completely
#: alone, because collapsing ``https://`` to ``https:/`` would corrupt
#: every URL in the corpus -- and URLs are technical content this filter
#: must not touch.
_UNUSED_DOUBLED = re.compile(r"\\{2,}")


#: A run of backslashes anywhere, collapsed to one.
_ANY_SEPARATOR_RUN = re.compile(r"\\+")

#: A *token-initial* doubled backslash, which is a UNC prefix rather
#: than OCR noise.
#:
#: The negative lookbehind is what separates the two cases, and the colon
#: in it is load-bearing. ``\\server\share`` at the start of a token is a
#: share. ``c:\\Users`` is the same two characters immediately after a
#: drive colon, where they are a doubled separator and nothing else.
#: Without excluding ``:`` the UNC rule ate the drive path's own
#: separator and left ``c<UNC_PATH>>python`` behind.
_LEADING_UNC = re.compile(r"(?<![A-Za-z0-9:\\])\\\\(?=[^\\/:*?\"<>|\s])")

#: Placeholder standing in for a protected UNC prefix while the rest of
#: the text is normalised. A control character, so it cannot occur in a
#: transcription.
_UNC_GUARD = "\x00unc\x00"


def _normalise_separators(text: str) -> str:
    r"""
    Make OCR's doubled separators consistent before matching.

    Genuine UNC prefixes are set aside first, every remaining run of
    backslashes collapses to one, and the prefixes are restored. That
    ordering is what lets ``c:\\Users\\Your Name`` and ``\\server\share``
    both survive as the shapes they actually are, which in turn is what
    lets the drive and colon-less rules anchor on a single separator.

    Replacements are callables because ``re.sub`` interprets backslashes
    in a plain replacement string as escape sequences: ``"\\\\"`` there is
    *one* backslash, which silently halved an earlier version of this and
    made every UNC path vanish.
    """

    guarded = _LEADING_UNC.sub(_UNC_GUARD, text)

    collapsed = _ANY_SEPARATOR_RUN.sub(lambda _match: "\\", guarded)

    return collapsed.replace(_UNC_GUARD, "\\\\")


_PATH_CHAR = r"(?:(?![A-Za-z]:)[^\\/:*?\"<>|\r\n])"

_DRIVE_PATH = re.compile(
    r"(?i)(?<![A-Za-z0-9])\b[A-Z]:[\\/]+"
    rf"(?:{_PATH_CHAR}{{1,120}}[\\/]){{0,8}}"
    rf"{_PATH_CHAR}{{0,120}}"
)

#: A Windows user-profile path whose drive colon OCR dropped.
#:
#: Measured, this is a real and frequent variant rather than a
#: hypothetical one. Three lines of the same tutorial slide, read by the
#: same engine on the same page:
#:
#:     c:\Users\Your Name>python          colon present, matched above
#:     c\\Users\Your Name>python          colon present, backslash doubled
#:     c\\Users\Your Name>python          **colon gone**, backslash doubled
#:
#: Only the first is caught by a pattern anchored on ``X:``, which is why
#: the last one survived and then had its doubled backslash pair eaten by
#: the UNC rule, leaving a stray ``c`` in the output.
#:
#: Scoped to a leading ``Users`` segment, and that scope is the whole
#: point. A first attempt matched "a letter, a separator, several
#: segments", which is far too loose -- measured against this corpus it
#: rewrote ``n//2`` and ``m=m//10`` (integer division) into
#: ``<LOCAL_WINDOWS_PATH>/2``, turned ``I/O`` into a marker, and mangled
#: the date format ``'%d/%m/%Y'``. Every one of those is technical content
#: this filter has no business touching. Requiring the literal ``Users``
#: segment matches all three real occurrences and none of those.
_COLONLESS_PATH = re.compile(
    r"(?i)(?<![A-Za-z0-9])[A-Z][\\/]+Users[\\/]"
    rf"(?:{_PATH_CHAR}{{1,120}}[\\/]){{0,8}}"
    rf"{_PATH_CHAR}{{0,120}}"
)

#: A UNC path. Safe to run first once separators are normalised, because
#: a drive path then contains exactly one backslash after its colon and
#: cannot be mistaken for a doubled UNC prefix.
#:
#: Two guards, both added after measuring against this package:
#:
#: * The server name may not be whitespace. Without that, the OCR artefact
#:   ``\\ \`` in one transcription was redacted as a UNC path -- it is two
#:   stray backslashes, not a share.
#: * No drive letter may precede it. OCR renders ``c:\\Users\\Your Name``
#:   with the backslashes doubled, and the naive UNC pattern consumed the
#:   doubled pair and left a stray ``c`` in the output
#:   (``c<UNC_PATH>>python``) instead of the whole path.
_UNC_PATH = re.compile(
    r"\\\\[^\\/:*?\"<>|\s]+(?:[\\/][^\\/:*?\"<>|\r\n]+)+"
)

#: Environment variables whose *values* are local paths. The variable name
#: itself is not secret, but substituting it would put a real user
#: directory into the output on a machine where it is set.
_ENV_PATH = re.compile(
    r"(?i)%(?:USERPROFILE|LOCALAPPDATA|APPDATA|TEMP|TMP|HOMEPATH|HOMEDRIVE|USERDOMAIN)%"
    r"(?:[\\/][^\\/:*?\"<>|\r\n]*)*"
)

#: This project's own scratch, virtual-environment and archive directories.
#: Named rather than pattern-matched generically, because a generic
#: "dot-directory" rule would also redact ``.config`` or ``.git`` out of
#: legitimate technical text.
_PROJECT_PATH = re.compile(
    r"(?i)(?:"
    r"[A-Za-z]:[\\/]{0,2})?"
    r"(?:"
    r"\.venv|\.agent|chrome_session|linkedin_saved_archive"
    r"|LINKEDIN_KNOWLEDGE_ARCHIVE_PACKAGE"
    r")[\\/][^\\/:*?\"<>|\r\n]*(?:[\\/][^\\/:*?\"<>|\r\n]*)*"
)

#: Applied after a path match: stop it eating trailing prose, and drop the
#: space that sat between the path and the next word.
_TRAILING_SPACE = re.compile(r"[ \t]+$")


def _trim(text: str) -> str:
    """Tidy the seam left by a substitution, without touching the rest."""

    return _TRAILING_SPACE.sub("", text)


def redact(text: str | None) -> tuple[str, int]:
    """
    Replace local filesystem paths with explicit markers.

    Returns the sanitised text and how many substitutions were made, so
    the caller can record that it happened rather than presenting the
    result as untouched source text.
    """

    if not text:
        return "", 0

    body = _normalise_separators(str(text))
    count = 0

    for pattern, marker in (
        (_UNC_PATH, UNC_PATH_MARKER),
        (_DRIVE_PATH, LOCAL_PATH_MARKER),
        (_COLONLESS_PATH, LOCAL_PATH_MARKER),
        (_ENV_PATH, ENV_PATH_MARKER),
        (_PROJECT_PATH, PROJECT_PATH_MARKER),
    ):
        body, replaced = pattern.subn(marker, body)
        count += replaced

    if count:
        body = _trim(body)

    return body, count


def redact_text(text: str | None) -> str:
    """Sanitised text, for the common case where the count is not needed."""

    return redact(text)[0]


def summarise(count: int) -> str:
    """
    A sentence describing what was removed, for the record.

    Written as an observation rather than an apology, and never as
    "verbatim": a reader comparing it with the source should be able to
    see at a glance that something was replaced.
    """

    if count <= 0:
        return ""

    noun = "path" if count == 1 else "paths"

    return (
        f"{count} local filesystem {noun} in this transcription replaced "
        "with a marker; the surrounding commands and text are unchanged."
    )


__all__ = [
    "ENV_PATH_MARKER",
    "LOCAL_PATH_MARKER",
    "PROJECT_PATH_MARKER",
    "UNC_PATH_MARKER",
    "redact",
    "redact_text",
    "summarise",
]