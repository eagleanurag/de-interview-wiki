"""
Where the pipeline reads and writes, and how it says things.

Split out so that the orchestrator, the stages and the command line can
all reach the same paths and the same log without importing each other.
The alternative was one large module, which reads fine until something
needs the freshness rules without wanting the command line.
"""

from __future__ import annotations

from pathlib import Path


#: Everything the pipeline produces goes here, and this is git-ignored,
#: so no stage of a run writes into the committed repository. The
#: committed inputs are the posts; the outputs are derived.
BUILD_ROOT = Path("build")

#: One file per post, which is what makes the work independent: a post
#: is enriched, stored and aggregated without reference to any other.
RESULTS_DIR = BUILD_ROOT / "worker-results"

#: The canonical knowledge base: one authoritative JSON that the site is
#: generated from and that anything downstream should read instead of
#: re-deriving.
KNOWLEDGE_BASE = BUILD_ROOT / "knowledge_base.json"

SITE_DIR = BUILD_ROOT / "site"

#: The orchestrator's record of its own run: what settled, what is owed,
#: and why anything failed. Sits beside the results it describes rather
#: than in the repository, because it describes a run and not the project.
RUN_STATE = BUILD_ROOT / "enrichment-state.json"


class StageError(RuntimeError):
    """Raised when a stage cannot complete."""


def log(message: str) -> None:
    """One line, flushed, so a long run shows progress as it happens."""

    print(f"[pipeline] {message}", flush=True)


__all__ = [
    "BUILD_ROOT",
    "KNOWLEDGE_BASE",
    "RESULTS_DIR",
    "RUN_STATE",
    "SITE_DIR",
    "StageError",
    "log",
]