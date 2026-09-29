"""
Redaction helpers.

The agent may print command output that incidentally contains a token,
and CI logs can carry a masked or partially echoed secret. Anything
that becomes an issue comment or a job summary goes through
`redact` first, so a credential cannot leak into a durable, widely
readable place.
"""

from __future__ import annotations

import re


REDACTED = "[REDACTED]"

# Patterns for common credential shapes.
_PATTERNS: tuple[re.Pattern[str], ...] = (
    # GitHub tokens: ghp_, gho_, ghu_, ghs_, ghr_ and the fine-grained
    # github_pat_ prefix.
    re.compile(
        r"\b(?:gh[pousr]_[A-Za-z0-9]{16,}"
        r"|github_pat_[A-Za-z0-9_]{20,})\b"
    ),
    # GitHub Actions OIDC / app and webhook tokens.
    re.compile(r"\bgh[osur]_[A-Za-z0-9]{16,}\b"),
    # Slack, Stripe and OpenAI style prefixed keys.
    re.compile(
        r"\b(?:sk|pk|rk|xox[baprs])[-_]"
        r"[A-Za-z0-9_-]{16,}\b"
    ),
    # AWS access key ids.
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    # Google API keys: "AIza" plus a run of URL-safe base64. The
    # documented length is 35; a range is used so a slightly different
    # key length is still caught rather than slipping through.
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,45}\b"),
    # Bearer tokens, before the generic assignment rule below.
    # "Authorization: Bearer <token>" must lose the token, not just
    # the word "Bearer", which the assignment rule alone would do.
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{12,}"),
    # Generic key=value / "key": "value" assignments whose key name
    # looks secret. Deliberately last: it is the broadest rule and
    # would otherwise swallow a more specific pattern's prefix.
    #
    # The leading lookahead stops an earlier substitution from being
    # re-matched, which would otherwise leave a stray bracket behind.
    re.compile(
        r"(?i)\b((?:[A-Za-z0-9_]*"
        r"(?:token|secret|password|passwd|api[_-]?key|access[_-]?key"
        r"|private[_-]?key|credential|auth)"
        r"[A-Za-z0-9_]*))"
        r"(\s*[:=]\s*)"
        r"(?!\[REDACTED\])"
        r"(\"[^\"\n]{4,}\"|'[^'\n]{4,}'|[^\s,;}\]]{4,})"
    ),
    # PEM private key blocks.
    re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----"
        r"[\s\S]*?"
        r"-----END [A-Z ]*PRIVATE KEY-----"
    ),
)

_ASSIGNMENT_GROUP = 3


def redact(
    text: str | None,
    *,
    secrets: tuple[str, ...] = (),
) -> str:
    """
    Remove credential-shaped content from text.

    Known secret values are replaced literally first, because they may
    not match any shape-based pattern. Longest values are replaced
    first so a short secret cannot partially mask a longer one.
    """

    if not text:
        return ""

    result = text

    for secret in sorted(
        {value for value in secrets if value and len(value) >= 8},
        key=len,
        reverse=True,
    ):
        result = result.replace(secret, REDACTED)

    for pattern in _PATTERNS:
        if pattern.groups >= _ASSIGNMENT_GROUP:
            result = pattern.sub(
                lambda match: (
                    f"{match.group(1)}{match.group(2)}{REDACTED}"
                ),
                result,
            )
        elif pattern.groups == 2:
            result = pattern.sub(
                lambda match: (
                    f"{match.group(1)}{match.group(2)}{REDACTED}"
                ),
                result,
            )
        elif pattern.groups == 1:
            result = pattern.sub(
                lambda match: f"{match.group(1)}{REDACTED}",
                result,
            )
        else:
            result = pattern.sub(REDACTED, result)

    return result


def truncate_for_comment(
    text: str | None,
    limit: int = 6000,
) -> str:
    """
    Bound text destined for an issue comment.

    Keeps the head and the tail, because the tail of a failure log is
    usually where the actual error is. The middle is what gets cut.
    """

    cleaned = (text or "").strip()

    if len(cleaned) <= limit:
        return cleaned

    head = limit // 3
    tail = limit - head - 40

    omitted = len(cleaned) - head - tail

    return (
        cleaned[:head]
        + f"\n\n... [{omitted} characters omitted] ...\n\n"
        + cleaned[-tail:]
    )
