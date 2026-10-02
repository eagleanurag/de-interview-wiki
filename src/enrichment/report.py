"""
What a run did, on one screen.

A five-hundred-post run that printed one line per model response is a
run nobody reads, and a run that prints nothing is a run nobody trusts.
So each post gets exactly one line as it settles, carrying only the
outcome and the count that changed, and the detail goes to the state
file and the result files where it can be read properly.

The summary at the end names every category the task cares about
separately, because "489 of 490" and "490 of 490" are different
outcomes and a reader should not have to subtract to find out which
happened.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.enrichment.runner import Outcome
from src.enrichment.state import RunState


#: What one settled post is called on its line. Short, because there
#: may be five hundred of them.
CACHED = "cached"
ENRICHED = "enriched"
RETRIED = "retried"
FAILED = "FAILED"
SKIPPED = "skipped"


def label_for(outcome: Outcome) -> str:
    """One word, or two when a retry is the interesting part."""

    if not outcome.succeeded:
        return FAILED

    if outcome.retry_count:
        return RETRIED

    return CACHED if outcome.reused else ENRICHED


@dataclass
class RunReport:
    """
    The counts for one run, and the clock.

    Holds the totals rather than recomputing them at the end, so the
    summary is about the run that happened and not about whatever the
    state file happens to say afterwards.
    """

    total: int = 0
    cached: int = 0
    processed: int = 0
    retried: int = 0
    failed: int = 0
    skipped: int = 0
    attempts: int = 0
    seconds: float = 0.0
    failures: list[dict] = field(default_factory=list)

    @property
    def successful(self) -> int:
        return self.cached + self.processed - self.retried + self.retried

    def add(self, outcome: Outcome) -> None:
        """Fold one settled post into the totals."""

        label = label_for(outcome)

        if label == CACHED:
            self.cached += 1

        elif label == RETRIED:
            self.retried += 1
            self.processed += 1

        elif label == ENRICHED:
            self.processed += 1

        elif label == FAILED:
            self.failed += 1

            record = outcome.failure_record()

            if record is not None:
                self.failures.append(record)

        elif label == SKIPPED:
            self.skipped += 1

        self.attempts += len(outcome.attempts)

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "cached": self.cached,
            "processed": self.processed,
            "retried": self.retried,
            "failed": self.failed,
            "skipped": self.skipped,
            "attempts": self.attempts,
            "seconds": round(self.seconds, 1),
            "failures": self.failures,
        }

    def render(self) -> str:
        """The end-of-run summary."""

        lines: list[str] = []
        lines.append("")
        lines.append("Enrichment")
        lines.append("----------")
        lines.append(f"{'Total':<26}{self.total:>8}")
        lines.append(f"{'Already complete':<26}{self.cached:>8}")
        lines.append(f"{'Processed':<26}{self.processed:>8}")
        lines.append(f"{'  of which retried':<26}{self.retried:>8}")
        lines.append(f"{'Permanently failed':<26}{self.failed:>8}")
        lines.append(f"{'Skipped':<26}{self.skipped:>8}")
        lines.append(f"{'Model attempts':<26}{self.attempts:>8}")
        lines.append("")
        lines.append(
            f"Duration {self.seconds / 60:.1f} min "
            f"({self.seconds:.0f}s)"
        )

        if self.failures:
            lines.append("")
            lines.append("Failed posts")
            lines.append("------------")
            lines.append(
                "  Every one of these has a record in the run state "
                "file."
            )
            lines.append("")

            for entry in self.failures[:25]:
                lines.append(
                    f"  {entry['post_id']}"
                )
                lines.append(
                    f"    {entry['error_type']}: "
                    f"{entry['error_message'][:90]}"
                )
                lines.append(
                    f"    after {entry['attempts']} attempt(s), "
                    f"kind={entry['error_kind']}"
                )

            if len(self.failures) > 25:
                lines.append(
                    f"  ... and {len(self.failures) - 25} more"
                )

            lines.append("")
            lines.append(
                "Re-run to retry them: nothing that succeeded is "
                "re-done."
            )

        return "\n".join(lines)


def progress_line(
    index: int,
    total: int,
    outcome: Outcome,
) -> str:
    """
    One line per settled post.

    ``[042/490] retried (2 attempts, 71.3s) urn-li-archive-...``

    The attempt count is on the line because "retried" without it is a
    claim, and the elapsed time because a run that is all retries looks
    identical to a run that is not without it.
    """

    label = label_for(outcome)

    detail = ""

    if outcome.retry_count:
        detail = (
            f" ({len(outcome.attempts)} attempts, "
            f"{outcome.seconds:.1f}s)"
        )

    elif outcome.seconds >= 1:
        detail = f" ({outcome.seconds:.1f}s)"

    return (
        f"[{index:>4}/{total}] {label}{detail} {outcome.post_id}"
    )


def resume_summary(state: RunState, total: int) -> str | None:
    """
    What a resumed run is picking up, when there is something to say.

    Silent on a first run, and silent when everything is already done --
    in both cases there is nothing a reader needs.
    """

    counts = state.counts()

    if not state.runs:
        return None

    if counts["enriched"] + counts["cached"] == 0 and counts["failed"] == 0:
        return None

    if counts["enriched"] + counts["cached"] == total and counts["failed"] == 0:
        return None

    return (
        f"Resuming: {counts['enriched'] + counts['cached']} already "
        f"complete, {counts['pending']} pending, "
        f"{counts['failed']} previously failed"
    )


def elapsed(seconds: float) -> str:
    """A duration, in the unit that reads best at that size."""

    if seconds < 90:
        return f"{seconds:.0f}s"

    if seconds < 5400:
        return f"{seconds / 60:.1f} min"

    return f"{seconds / 3600:.1f} h"


__all__ = [
    "RunReport",
    "elapsed",
    "label_for",
    "progress_line",
    "resume_summary",
]