"""
Decide whether an enrichment failure is worth trying again.

The distinction that matters is not what went wrong but whether a fresh
attempt could plausibly succeed. A truncated response almost certainly
will, because the next sample is a different sample. A missing API key
certainly will not, and retrying it three times just spends the time and
hides the cause.

Nothing here names a provider. The signals are the ones every
structured-output service emits -- a status, an error kind, a message
shape -- so a new provider is a new row in a table rather than new
branches through the pipeline. That was the mistake worth avoiding: the
GitHub worker run failed on provider behaviour, and the correct fix
belongs next to the client, not spread through the code that calls it.

The two failure kinds observed against five hundred real posts were both
the same thing underneath. One was reported as a provider error:

    {"type": "provider.invalid-output",
     "message": "OpenAI Chat stream ended without finish_reason",
     "status": 200}

and the other arrived as a successful exit carrying a JSON fragment that
no parser could complete:

    Input should be a valid dictionary or instance of
    AIEnrichmentResponse [type=model_type, input_value='{\\n  "summary": "']

So the response schema was never the problem in either case. Classifying
the second as a schema failure and "fixing" the model would have been
the wrong repair; re-asking is the right one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class FailureKind(str, Enum):
    """
    What sort of failure this is, in terms of what to do about it.

    Named for the response rather than for a stage, because the
    orchestrator's decision only depends on that.
    """

    #: The provider returned part of an answer and stopped.
    TRUNCATED = "truncated_response"

    #: The provider refused or failed in a way that clears on its own.
    TRANSIENT = "transient_provider_error"

    #: The provider will refuse again exactly the same way.
    PERMANENT = "permanent_provider_error"

    #: The answer arrived whole and did not satisfy the schema.
    SCHEMA = "schema_mismatch"

    #: Something local broke: no post, unreadable media, no client.
    LOCAL = "local_error"

    #: Not something this module recognises.
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Classification:
    """
    One failure, judged.

    ``recoverable`` is the answer the orchestrator acts on. ``evidence``
    is the text that decided it, kept so a report can explain the
    judgement instead of asserting it.
    """

    kind: FailureKind
    recoverable: bool
    evidence: str

    def as_dict(self) -> dict:
        return {
            "kind": self.kind.value,
            "recoverable": self.recoverable,
            "evidence": self.evidence[:400],
        }


#: Substrings that mean the provider stopped early. Matched against the
#: whole rendered failure, because the client flattens the provider's
#: JSON error event into the exception message.
_TRUNCATION = (
    "without finish_reason",
    "invalid-output",
    "invalid_output",
    "unexpected eof",
    "premature end",
    "stream ended",
    "stream closed",
    "truncated",
    "incomplete json",
    "expecting ',' delimiter",
    "unterminated string",
    "expecting property name",
)

#: Substrings that mean the provider is busy, throttling or briefly
#: broken, and that a later attempt clears.
_TRANSIENT = (
    "timeout",
    "timed out",
    "rate limit",
    "rate_limit",
    "too many requests",
    "429",
    "500",
    "502",
    "503",
    "504",
    "529",
    "overloaded",
    "capacity",
    "service unavailable",
    "connection reset",
    "connection refused",
    "connection aborted",
    "eof occurred",
    "temporarily unavailable",
    "try again",
)

#: Substrings that mean the provider will refuse the same way again.
#: Retrying these only spends the budget and hides the cause behind a
#: claim that something was recovered.
_PERMANENT = (
    "401",
    "403",
    "unauthorized",
    "unauthenticated",
    "forbidden",
    "invalid api key",
    "invalid_api_key",
    "incorrect api key",
    "no api key",
    "api key not",
    "model not found",
    "does not exist",
    "unknown model",
    "no such model",
    "bad request",
    "400",
    "unsupported",
    "not supported",
    "billing",
    "quota exceeded",
    "payment required",
)

#: A pydantic model_type error whose input looks like the start of a
#: JSON object. The value is truncated to a few characters by
#: pydantic's own message, so only the opening is visible, and that is
#: enough: a value that opens like JSON and did not parse is a
#: truncated body, not a schema disagreement.
_JSON_FRAGMENT = re.compile(
    r"input_value='?\s*\{\s*\\?n?\s*\\?\"",
    re.IGNORECASE,
)

_ERROR_EVENT = re.compile(r'"type"\s*:\s*"([^"]+)"')
_STATUS = re.compile(r'"status"\s*:\s*(\d{3})')
_MESSAGE = re.compile(r'"message"\s*:\s*"([^"]{0,300})"')


def _text_of(exc: BaseException) -> str:
    """
    Everything known about a failure, flattened for matching.

    The exception type and its whole chain, because the client raises
    one error type for a transport failure and for a provider-reported
    one, and the useful signal is in the message of either.
    """

    parts: list[str] = [type(exc).__name__, str(exc)]

    cause = exc.__cause__

    seen = 0

    while cause is not None and seen < 4:
        parts.append(f"{type(cause).__name__}: {cause}")
        cause = cause.__cause__
        seen += 1

    return "\n".join(parts)


def classify(exc: BaseException) -> Classification:
    """
    Judge one failure.

    Ordered from most specific to least, because a single message can
    carry several signals -- a truncated stream usually also reports a
    200, and a rate-limit message may mention a retry -- and the first
    match is the one that describes the cause rather than the symptom.
    """

    text = _text_of(exc)
    lowered = text.lower()

    def evidence(pattern: str) -> str:
        for line in text.splitlines():
            if pattern in line.lower():
                return line.strip()

        return text.strip()[:200]

    # Local problems are judged before the provider patterns, because
    # they are local whatever they say. A missing post directory reads
    # "does not exist", which is also how a provider says a model does
    # not exist, and matching the words first would file a post we never
    # loaded as a provider refusal -- unrecoverable, and pointing an
    # operator at the wrong system.
    if isinstance(
        exc, (FileNotFoundError, IsADirectoryError, NotADirectoryError)
    ) or "post directory does not exist" in lowered:
        return Classification(
            FailureKind.LOCAL, False, evidence("does not exist")
        )

    if isinstance(exc, PermissionError) and "open" in lowered:
        return Classification(
            FailureKind.LOCAL, False, evidence("denied")
        )

    if "could not locate opencode" in lowered or "opencode_executable" in lowered:
        # Nobody is going to install it mid-run. A person has to.
        return Classification(
            FailureKind.LOCAL, False, evidence("could not locate")
        )

    # A pydantic model_type failure whose input opened like JSON is the
    # same truncation as an invalid-output event, so it is judged as one
    # before the generic schema check gets a chance to call it a schema
    # problem. Both were observed against the real corpus, and both mean
    # the body stopped early.
    if _JSON_FRAGMENT.search(text):
        return Classification(
            FailureKind.TRUNCATED, True, evidence("input_value")
        )

    for pattern in _TRUNCATION:
        if pattern in lowered:
            return Classification(
                FailureKind.TRUNCATED, True, evidence(pattern)
            )

    for pattern in _PERMANENT:
        if pattern in lowered:
            return Classification(
                FailureKind.PERMANENT, False, evidence(pattern)
            )

    for pattern in _TRANSIENT:
        if pattern in lowered:
            return Classification(
                FailureKind.TRANSIENT, True, evidence(pattern)
            )

    if "validation error" in lowered or "did not match the expected structure" in lowered:
        return Classification(
            FailureKind.SCHEMA, True, evidence("validation error")
        )

    message = _MESSAGE.search(text)
    event = _ERROR_EVENT.search(text)
    status = _STATUS.search(text)

    detail = " ".join(
        part
        for part in (
            f"type={event.group(1)}" if event else "",
            f"status={status.group(1)}" if status else "",
            f"message={message.group(1)}" if message else "",
        )
        if part
    )

    return Classification(
        FailureKind.UNKNOWN, True, detail or text.strip()[:200]
    )


def is_recoverable(exc: BaseException) -> bool:
    """Whether a fresh attempt is worth making."""

    return classify(exc).recoverable


__all__ = [
    "Classification",
    "FailureKind",
    "classify",
    "is_recoverable",
]