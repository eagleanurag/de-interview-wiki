"""
Enriching posts, once, for every caller.

The GitHub worker experiment established that this project's enrichment
is not reliable when each post gets exactly one attempt: six of twenty
batches failed, and every failure was a truncated provider response that
a second draw would almost certainly have fixed. The fix is not a
different model or a looser schema. It is a bounded retry around the
call that already exists, plus a record of what happened.

This package holds that, and nothing else. The knowledge base, the
grounding rules, the fingerprints and the post models all live where
they already lived.
"""

from __future__ import annotations

from src.enrichment.report import (
    RunReport,
    progress_line,
    resume_summary,
)
from src.enrichment.runner import (
    DEFAULT_ATTEMPTS,
    Attempt,
    Enricher,
    Outcome,
    cached,
)
from src.enrichment.state import (
    PostRecord,
    RunState,
    STATE_VERSION,
)

__all__ = [
    "Attempt",
    "DEFAULT_ATTEMPTS",
    "Enricher",
    "Outcome",
    "PostRecord",
    "RunReport",
    "RunState",
    "STATE_VERSION",
    "cached",
    "progress_line",
    "resume_summary",
]