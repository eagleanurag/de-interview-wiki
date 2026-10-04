"""
Containment decisions that must not depend on the platform.

Three tests in this repository assert that a path written on Windows is
refused. They passed on the machine that wrote them and failed on the
Linux runner, for the same reason: ``pathlib`` on POSIX reads
``..\\outside.jpg`` as one filename and ``C:/Windows/x`` as a relative
name, so both resolve *inside* the root and nothing is raised.

Running them on Windows therefore proves nothing, which is exactly how
the regression reached a commit. These tests check the decision instead
of the local platform's accident: they read what
:func:`src.paths.candidate_path` returns and judge it under POSIX
semantics explicitly, so the assertion holds on every machine.
"""

from __future__ import annotations

from pathlib import PurePosixPath

import pytest

from src.paths import candidate_path


#: A resolved root the POSIX simulation is judged against. Any absolute
#: path works; it only has to be consistent.
ROOT = PurePosixPath("/srv/media")


def _resolve_like_pathlib(path: PurePosixPath) -> PurePosixPath:
    """
    Collapse ``.`` and ``..`` the way ``Path.resolve()`` does.

    ``PurePosixPath`` keeps ``..`` in ``.parts``; ``resolve()`` removes
    it. Reproducing that here is what makes the simulation meaningful --
    without it every relative path looks like it stays inside.
    """

    parts: list[str] = []

    for part in path.parts:
        if part == ".":
            continue

        if part == "..":
            if parts and parts[-1] != "..":
                parts.pop()
            elif not path.is_absolute():
                parts.append("..")

            continue

        parts.append(part)

    return PurePosixPath(*parts) if parts else PurePosixPath("/")


def refuses_under_posix(candidate: str) -> bool:
    """What the real helper decides, judged the way Linux would judge it."""

    result = candidate_path(candidate)

    if result is None:
        return True

    # Read the result as a POSIX path even when running on Windows, so
    # the separator handling under test is POSIX's.
    posix = PurePosixPath(str(result).replace("\\", "/"))

    joined = posix if posix.is_absolute() else ROOT / posix

    try:
        _resolve_like_pathlib(joined).relative_to(ROOT)

    except ValueError:
        return True

    return False


ESCAPES = [
    "../outside.jpg",
    "..\\outside.jpg",
    "../../etc/passwd",
    "..\\..\\Windows\\win.ini",
    "C:/Windows/System32/drivers/etc/hosts",
    "C:\\Windows\\System32\\cmd.exe",
    "//server/share/secret.jpg",
    "\\\\server\\share\\secret.jpg",
    "/etc/passwd",
]


@pytest.mark.parametrize("candidate", ESCAPES)
def test_an_escape_is_refused_however_the_path_was_written(candidate):
    assert refuses_under_posix(candidate) is True, candidate


INSIDE = [
    "a.jpg",
    "activity_7326889752870170625_slide_0.jpg",
    "sub/dir/a.jpg",
    "A:notes.txt",
    "weird name (1).jpg",
    "....//x.jpg",
]


@pytest.mark.parametrize("candidate", INSIDE)
def test_a_real_archive_filename_is_still_accepted(candidate):
    """
    The other half, and the reason the fix is not simply "refuse more".

    Refusing every backslash, or every colon, would have made these fail
    -- and a containment rule that refuses legitimate files is not a
    safer rule, it is a broken one that gets worked around.
    """

    assert refuses_under_posix(candidate) is False, candidate


def test_a_drive_letter_needs_a_separator_to_be_a_drive():
    """
    ``A:notes.txt`` is a filename, not a drive.

    The separator is what separates the two, and requiring it is what
    stops the rule from refusing every name that happens to contain a
    colon -- of which there are more in a real archive than one would
    guess.
    """

    assert candidate_path("A:notes.txt") is not None
    assert candidate_path("C:/x") is None
    assert candidate_path("C:\\x") is None


def test_a_posix_absolute_path_is_left_for_resolution_to_judge():
    """
    Not refused outright, because it is a legitimate shape.

    ``/etc/passwd`` is absolute and outside the root, so resolution
    catches it -- which is the check that was already there and works on
    both platforms. Refusing it here as well would be belt and braces
    for no gain, and would refuse ``/srv/media/a.jpg``-shaped input that
    the caller may legitimately supply.
    """

    assert candidate_path("/etc/passwd") is not None
    assert candidate_path("/srv/media/a.jpg") is not None