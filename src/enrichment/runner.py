"""
The one path a post is enriched by.

Both the local pipeline and the job-file worker call :func:`enrich_with_retry`,
so retry, failure classification and the repair strategy exist once. That
is the point of this package: the GitHub run failed six batches because
each post had exactly one attempt and a truncated response was therefore
a lost post, and a fix applied to only one of the two callers would have
left the other able to lose posts the same way.

The retry is bounded and gives up for a reason. A truncated or
rate-limited response gets a fresh sample, because the next sample is a
different sample. A missing API key does not, because the third attempt
fails identically to the first and reporting "recovered" would be a lie.

Nothing here relaxes a schema. When an answer arrives whole but does
not satisfy the model, the model is asked again and told what was wrong,
because inventing the missing explanation to satisfy validation would
put content in the knowledge base that no source and no model ever
produced.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from src.ai.enricher import AIEnricher, EnrichmentError
from src.ai.recovery import Classification, FailureKind, classify
from src.models import KnowledgePost


#: How many times a post is attempted before it is written off.
#:
#: Three, because the failures seen in practice are truncated responses
#: and the first retry has historically been enough. Bounded on purpose:
#: an unbounded retry turns a provider outage into an infinite run.
DEFAULT_ATTEMPTS = 3

#: How long to wait before the second attempt, doubling after that.
#:
#: Long enough that a provider which is briefly saturated has recovered,
#: short enough that a run over hundreds of posts is not spent waiting.
BASE_BACKOFF_SECONDS = 2.0
MAX_BACKOFF_SECONDS = 30.0


@dataclass
class Attempt:
    """One try at one post, recorded whatever it ended as."""

    number: int
    outcome: str
    kind: str = ""
    error_type: str = ""
    error_message: str = ""
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return {
            "attempt": self.number,
            "outcome": self.outcome,
            "kind": self.kind,
            "error_type": self.error_type,
            "error_message": self.error_message[:600],
            "seconds": round(self.seconds, 2),
        }


@dataclass
class Outcome:
    """
    What happened to one post.

    ``status`` is one of ``enriched``, ``cached``, ``failed`` or
    ``skipped``, and it is the only field the summary counts. The rest is
    there so a failure can be diagnosed without re-running anything.
    """

    post_id: str
    status: str
    attempts: list[Attempt] = field(default_factory=list)
    grounding: dict | None = None
    seconds: float = 0.0
    reused: bool = False
    retry_count: int = 0

    @property
    def succeeded(self) -> bool:
        return self.status in {"enriched", "cached"}

    def failure_record(self) -> dict | None:
        """
        A machine-readable record of a permanent failure.

        Written for every post that never produced a result, so the set
        of missing posts is a fact on disk rather than a number in a log
        that has already scrolled away.
        """

        if self.succeeded:
            return None

        last = self.attempts[-1] if self.attempts else None

        return {
            "post_id": self.post_id,
            "stage": "enrich",
            "status": self.status,
            "attempts": len(self.attempts),
            "retry_count": self.retry_count,
            "error_type": last.error_type if last else "unknown",
            "error_kind": last.kind if last else FailureKind.UNKNOWN.value,
            "error_message": (last.error_message if last else "")[:600],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "recoverable": bool(
                last and classify_text(last.error_message).recoverable
            ),
            "attempt_log": [attempt.as_dict() for attempt in self.attempts],
        }


def classify_text(message: str) -> Classification:
    """Classify a stored error message, for a record read back later."""

    return classify(RuntimeError(message))


@dataclass
class Enricher:
    """
    An enricher plus the retry policy applied to it.

    ``factory`` rather than an instance so each worker thread gets its
    own. The client is stateless, but the enricher records the last
    grounding report, and a shared one would let one post file another
    post's removals.
    """

    factory: Callable[[], AIEnricher]
    attempts: int = DEFAULT_ATTEMPTS
    backoff: float = BASE_BACKOFF_SECONDS
    sleep: Callable[[float], None] = time.sleep

    def _new(self) -> AIEnricher:
        return self.factory()

    def enrich(self, post: KnowledgePost) -> Outcome:
        """
        Enrich one post, retrying only what a retry can fix.

        Each attempt gets a fresh enricher and therefore a fresh sample,
        so a retry is a genuinely different draw rather than the same
        truncated answer parsed again.
        """

        started = time.monotonic()

        outcome = Outcome(post_id=post.id, status="failed")

        for number in range(1, max(1, self.attempts) + 1):
            attempt_started = time.monotonic()

            enricher = self._new()

            try:
                enricher.enrich(post)

            except Exception as exc:  # noqa: BLE001
                verdict = classify(exc)

                outcome.attempts.append(
                    Attempt(
                        number=number,
                        outcome="failed",
                        kind=verdict.kind.value,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        seconds=time.monotonic() - attempt_started,
                    )
                )

                if not verdict.recoverable:
                    break

                if number >= max(1, self.attempts):
                    break

                outcome.retry_count += 1

                self.sleep(
                    min(
                        self.backoff * (2 ** (number - 1)),
                        MAX_BACKOFF_SECONDS,
                    )
                )

                continue

            report = getattr(enricher, "last_grounding", None)

            outcome.attempts.append(
                Attempt(
                    number=number,
                    outcome="ok",
                    seconds=time.monotonic() - attempt_started,
                )
            )

            outcome.status = "enriched"

            if report is not None:
                outcome.grounding = report.as_dict()

            break

        outcome.seconds = time.monotonic() - started

        return outcome


def cached(post_id: str) -> Outcome:
    """A post that already had a current result and cost nothing."""

    return Outcome(post_id=post_id, status="cached", reused=True)


__all__ = [
    "Attempt",
    "DEFAULT_ATTEMPTS",
    "Enricher",
    "Outcome",
    "cached",
    "classify_text",
]