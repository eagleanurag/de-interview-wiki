"""
Sort orders that are total.

Python randomises string hashing per process, so a set of labels
iterates in a different order in every run. Any sort whose key ties for
two distinct values leaves those two in set order, and two builds of the
same input produce different bytes.

This was invisible against seven posts, where no two labels differed
only in case. The existing determinism test could not catch it either:
it generates the site twice inside one process, where the hash seed has
not changed between the two runs. So this file checks the property
directly, and once across seeds in a subprocess.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Labels that fold to the same case-insensitive string. This is what
#: the model actually emits once a corpus runs to several hundred posts.
TIED = (
    "ADF activities",
    "ADF Activities",
    "ADF exception handling",
    "ADF Exception Handling",
    "ADF pipeline performance troubleshooting",
    "ADF Variables and Parameters",
    "ADF variables and parameters",
    "ADLS Gen2",
    "ADLS Gen2 and Data Lake Storage",
    "adls gen2",
)


def test_the_order_key_is_total():
    """Two labels must never produce the same key."""

    from src.wiki.analysis import label_order

    keys = [label_order(value) for value in TIED]

    assert len(set(keys)) == len(TIED), sorted(keys)


def test_case_variants_get_a_stable_order():
    """Ties on casefold are broken the same way every time."""

    from src.wiki.analysis import label_order

    upper = label_order("ADF Activities")
    lower = label_order("ADF activities")

    assert upper != lower

    assert sorted(["ADF Activities", "ADF activities"], key=label_order) == (
        sorted(["ADF activities", "ADF Activities"], key=label_order)
    )


def test_case_still_decides_the_reading_order():
    """Case-insensitive first, because that is how a reader reads."""

    from src.wiki.analysis import label_order

    assert label_order("apple") < label_order("Banana")
    assert label_order("Apple") < label_order("banana")


def test_ordering_survives_a_hash_seed_change():
    """
    The property, across three interpreter seeds.

    In a subprocess because the seed is fixed per process, and that
    fixed seed is the reason a same-process determinism test passes
    straight over this bug.
    """

    script = (
        "from src.wiki.analysis import label_order;"
        f"labels={TIED!r};"
        "print('|'.join(sorted(set(labels), key=label_order)))"
    )

    outputs: set[str] = set()

    for seed in ("0", "1", "12345"):
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env={
                "PYTHONHASHSEED": seed,
                "PATH": "",
                "SYSTEMROOT": "",
            },
        )

        assert result.returncode == 0, result.stderr[-400:]

        outputs.add(result.stdout.strip())

    assert len(outputs) == 1, sorted(outputs)


def test_casefold_alone_really_was_unstable_across_seeds():
    """
    What the bug looked like, demonstrated rather than asserted about.

    Sorting on casefold alone leaves the tied pairs in set order, and set
    order over strings follows a hash the interpreter salts per process.
    So the same labels sort differently in different runs of the same
    command -- which is how two builds of one knowledge base produced a
    different search index.

    Checked here so the reason for ``label_order``'s second key stays
    visible, rather than leaving a future reader to simplify it away.
    """

    script = (
        f"labels={TIED!r};"
        "print('|'.join(sorted(set(labels), key=lambda v: v.casefold())))"
    )

    outputs: set[str] = set()

    for seed in ("0", "1", "2", "3", "4", "5"):
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env={
                "PYTHONHASHSEED": seed,
                "PATH": "",
                "SYSTEMROOT": "",
            },
        )

        assert result.returncode == 0, result.stderr[-400:]

        outputs.add(result.stdout.strip())

    assert len(outputs) > 1, (
        "casefold alone gave one order across six seeds, so this "
        "fixture no longer demonstrates a tie; the fixture needs "
        "labels that differ only in case"
    )


def test_the_label_order_survives_every_seed():
    """The same demonstration, for the key that is actually used."""

    script = (
        "from src.wiki.analysis import label_order;"
        f"labels={TIED!r};"
        "print('|'.join(sorted(set(labels), key=label_order)))"
    )

    outputs: set[str] = set()

    for seed in ("0", "1", "2", "3", "4", "5"):
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env={
                "PYTHONHASHSEED": seed,
                "PATH": "",
                "SYSTEMROOT": "",
            },
        )

        assert result.returncode == 0, result.stderr[-400:]

        outputs.add(result.stdout.strip())

    assert len(outputs) == 1, sorted(outputs)