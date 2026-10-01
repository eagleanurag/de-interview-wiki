"""
Bounded authentication state machine.

Replaces the previous approach, which polled ``Locator.count()`` in a
loop. That had three defects:

* every Playwright call inherited the context's default timeout, so a
  single call could block for minutes while the page was navigating;
* the poll budget counted iterations rather than elapsed time, so a
  slow call multiplied the total wait without bound;
* each poll made dozens of round trips, so the loop appeared to hang.

The fix is one observation per poll instead of dozens, using a single
JavaScript evaluation that returns everything needed to classify the
state. Every call is bounded by an explicit deadline, and an
indeterminate result becomes ``UNKNOWN`` rather than a hang or a false
success.

No state here solves, evades or works around a challenge. A challenge
is reported to the human and the machine stops.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class AuthState(str, Enum):
    """Every state the authentication machine can report."""

    #: The sign-in form is on screen.
    LOGIN_FORM = "login_form"

    #: Credentials were supplied and the form submitted.
    LOGIN_SUBMITTED = "login_submitted"

    #: The session is authenticated. Verified, never assumed.
    AUTHENTICATED = "authenticated"

    #: A challenge a human must complete. Never automated.
    HUMAN_CHALLENGE = "human_challenge"

    #: Credentials were rejected.
    AUTHENTICATION_FAILED = "authentication_failed"

    #: The browser could not be navigated.
    NAVIGATION_FAILED = "navigation_failed"

    #: The browser, page or session ended underneath us.
    BROWSER_UNAVAILABLE = "browser_unavailable"

    #: The state could not be determined.
    UNKNOWN = "unknown"

    @property
    def is_terminal_success(self) -> bool:
        return self is AuthState.AUTHENTICATED


#: Challenge markers, checked against the visible page text.
#: Biased toward stopping: a false positive costs the user one
#: confirmation, a false negative would mean working around a
#: challenge.
CHALLENGE_MARKERS: tuple[tuple[str, str], ...] = (
    ("captcha", "captcha"),
    ("recaptcha", "recaptcha"),
    ("hcaptcha", "hcaptcha"),
    ("not a robot", "captcha"),
    ("verify it's you", "suspicious_login"),
    ("verify it is you", "suspicious_login"),
    ("unusual activity", "suspicious_login"),
    ("security verification", "security_verification"),
    ("identity verification", "identity_verification"),
    ("verify your identity", "identity_verification"),
    ("confirm your identity", "identity_verification"),
    ("two-step verification", "two_factor"),
    ("two factor authentication", "two_factor"),
    ("verification code", "otp"),
    ("one-time code", "otp"),
    ("enter the code", "otp"),
    ("authenticate to continue", "authentication"),
    ("please sign in to continue", "authentication"),
    ("sign in to confirm", "authentication"),
    ("access denied", "access_denied"),
    ("account restricted", "account_restricted"),
    ("temporarily restricted", "account_restricted"),
    ("too many attempts", "rate_limited"),
    ("try again later", "rate_limited"),
)

#: Visible-page markers of a rejected sign-in.
FAILURE_MARKERS: tuple[str, ...] = (
    "password is incorrect",
    "incorrect password",
    "wrong password",
    "please try again",
    "check your credentials",
)

#: A page that is neither a form nor a session.
UNKNOWN_MARKERS: tuple[str, ...] = (
    "something went wrong",
    "unexpected error",
    "page not found",
)


# One round trip. Returns everything needed to classify the page, so
# polling is a single cheap call instead of dozens of locator calls.
PAGE_PROBE_JS = r"""
() => {
  const visible = (el) => !!el && (
    el.offsetWidth > 0 || el.offsetHeight > 0 ||
    el.getClientRects().length > 0
  );
  const all = (selector) => Array.from(
    document.querySelectorAll(selector)
  ).filter(visible);
  const text = (nodes) => nodes
    .map((n) => (n.innerText || n.textContent || "").trim())
    .filter(Boolean)
    .slice(0, 3);

  const username = all(
    'input[autocomplete="username"], input#session_key, ' +
    'input[name="session_key"], input[type="email"]'
  );
  const password = all(
    'input[autocomplete="current-password"], input#password, ' +
    'input[type="password"]'
  );
  const sessionMarkers = all(
    'a[data-tracking-control-name="nav_profile"], ' +
    'img[alt*="profile photo"], button[aria-label*="profile"], ' +
    'a[href*="/in/"][aria-label*="profile"]'
  );
  const feed = all(
    '[data-id^="urn:li:activity"], div.feed-shared-update-v2, ' +
    'main [data-id^="urn:li:"]'
  );
  const alerts = text(
    all('[role="alert"], .error-form__message, #error, ' +
        '.authwall-error, div[data-id*="error"]')
  );

  const body = document.body ? (document.body.innerText || "") : "";

  return {
    username_count: username.length,
    password_count: password.length,
    session_markers: sessionMarkers.length,
    feed_nodes: feed.length,
    alerts: alerts,
    body_sample: body.slice(0, 6000)
  };
}
"""

# Waits on the page itself rather than polling from Python. Bounded by
# an explicit timeout, so it cannot outlive its budget.
PAGE_SETTLED_JS = r"""
() => {
  const hasForm = (
    document.querySelector('input[autocomplete="username"], ' +
      'input#session_key, input[name="session_key"]') !== null
  );
  const hasSession = (
    document.querySelector('a[data-tracking-control-name="nav_profile"], ' +
      'img[alt*="profile photo"]') !== null
  );
  const hasFeed = (
    document.querySelector('[data-id^="urn:li:activity"], ' +
      'div.feed-shared-update-v2') !== null
  );
  const hasAlert = (
    document.querySelector('[role="alert"], .error-form__message, #error')
    !== null
  );
  const body = (document.body ? (document.body.innerText || "") : "")
    .toLowerCase();

  return hasForm || hasSession || hasFeed || hasAlert ||
    body.includes("captcha") || body.includes("verification code") ||
    body.includes("two-step");
}
"""


@dataclass(frozen=True)
class AuthObservation:
    """What one bounded probe saw."""

    state: AuthState
    detail: str = ""
    challenge: str = ""
    url: str = ""
    alerts: tuple[str, ...] = ()

    @property
    def needs_human(self) -> bool:
        return self.state is AuthState.HUMAN_CHALLENGE

    @property
    def is_usable(self) -> bool:
        return self.state.is_terminal_success


@dataclass
class AuthJournal:
    """
    Every transition, for diagnosis and for the run log.

    Held in memory only. Nothing here is ever written to disk, so no
    transition can leak a credential through a checkpoint.
    """

    states: list[str] = field(default_factory=list)
    details: list[str] = field(default_factory=list)

    def record(self, observation: AuthObservation) -> AuthObservation:
        self.states.append(observation.state.value)
        self.details.append(observation.detail)

        return observation

    @property
    def path(self) -> str:
        return " -> ".join(self.states)

    def as_dict(self) -> dict:
        return {"path": self.path, "details": list(self.details)}


def classify(
    probe: dict,
    *,
    url: str = "",
    last_action: str = "",
) -> AuthObservation:
    """
    Turn one probe into a state.

    Order matters. A challenge outranks a login form, because a
    challenge page often still renders form elements, and a session
    outranks a challenge marker only when the markers are absent.
    """

    if not isinstance(probe, dict):
        return AuthObservation(
            state=AuthState.UNKNOWN,
            detail="The page could not be inspected.",
            url=url,
        )

    body = str(probe.get("body_sample") or "").lower()
    alerts = tuple(
        str(item) for item in (probe.get("alerts") or []) if str(item)
    )

    # A challenge is checked first and stops everything else.
    haystack = " ".join([body, " ".join(alerts).lower()])

    for marker, kind in CHALLENGE_MARKERS:
        if marker in haystack:
            return AuthObservation(
                state=AuthState.HUMAN_CHALLENGE,
                detail=(
                    "LinkedIn is showing a "
                    f"{kind.replace('_', ' ')}. Complete it "
                    "yourself; it will not be solved here."
                ),
                challenge=kind,
                url=url,
                alerts=alerts,
            )

    has_form = (
        int(probe.get("username_count") or 0) > 0
        and int(probe.get("password_count") or 0) > 0
    )
    has_session = int(probe.get("session_markers") or 0) > 0
    has_feed = int(probe.get("feed_nodes") or 0) > 0

    # An authenticated session: a session marker, or a feed with no
    # sign-in form. Both are positive evidence, not an inference.
    if has_session or (has_feed and not has_form):
        return AuthObservation(
            state=AuthState.AUTHENTICATED,
            detail="Session markers are present on the page.",
            url=url,
            alerts=alerts,
        )

    if has_form:
        if last_action == "submit":
            # The form came back after submitting, which is what a
            # rejected sign-in looks like.
            if any(marker in haystack for marker in FAILURE_MARKERS):
                return AuthObservation(
                    state=AuthState.AUTHENTICATION_FAILED,
                    detail="LinkedIn rejected the credentials.",
                    url=url,
                    alerts=alerts,
                )

            return AuthObservation(
                state=AuthState.LOGIN_FORM,
                detail=(
                    "The sign-in form is still present after "
                    "submitting."
                ),
                url=url,
                alerts=alerts,
            )

        return AuthObservation(
            state=AuthState.LOGIN_FORM,
            detail="The sign-in form is available.",
            url=url,
            alerts=alerts,
        )

    if any(marker in haystack for marker in FAILURE_MARKERS):
        return AuthObservation(
            state=AuthState.AUTHENTICATION_FAILED,
            detail="LinkedIn reported a sign-in failure.",
            url=url,
            alerts=alerts,
        )

    if any(marker in haystack for marker in UNKNOWN_MARKERS):
        return AuthObservation(
            state=AuthState.UNKNOWN,
            detail="The page reported an unexpected state.",
            url=url,
            alerts=alerts,
        )

    return AuthObservation(
        state=AuthState.UNKNOWN,
        detail=(
            "Neither a sign-in form nor session markers were found."
        ),
        url=url,
        alerts=alerts,
    )


class Deadline:
    """
    A wall-clock budget shared by several browser operations.

    The previous implementation counted poll iterations, so a slow call
    silently multiplied the total wait. This measures elapsed time, so
    the total is bounded no matter how slow any individual call is.
    """

    def __init__(self, total_seconds: float) -> None:
        self.total_seconds = float(total_seconds)
        self._start = time.monotonic()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start

    @property
    def remaining(self) -> float:
        return max(0.0, self.total_seconds - self.elapsed)

    @property
    def expired(self) -> bool:
        return self.remaining <= 0.0

    def slice_ms(self, *, cap_ms: int, default_ms: int) -> int:
        """
        Milliseconds left, capped so no single call dominates.

        Always returns at least a small positive value while the budget
        lasts, so the driver never receives a zero timeout, which it
        treats as "no timeout".
        """

        if self.expired:
            return 0

        return int(max(250, min(cap_ms, self.remaining * 1000)))

    def default_ms(self) -> int:
        return int(max(250, self.remaining * 1000))


def probe(page, *, timeout_ms: int) -> dict | None:
    """
    One bounded probe of the page.

    A single ``evaluate`` call, rather than many locator calls, so a
    poll is one round trip and cannot accumulate waits. Returns None
    when the probe could not complete, which the caller turns into an
    unknown state rather than a hang.
    """

    try:
        page.set_default_timeout(timeout_ms)
    except Exception:  # noqa: BLE001
        pass

    try:
        result = page.evaluate(PAGE_PROBE_JS)
    except Exception:  # noqa: BLE001
        return None

    return result if isinstance(result, dict) else None


def settle(page, *, timeout_ms: int) -> bool:
    """
    Wait for the page to reach a decision, bounded.

    Uses a single ``wait_for_function`` rather than a Python-side
    polling loop, so the driver owns the wait and the budget is
    enforced once.
    """

    try:
        page.set_default_timeout(timeout_ms)
    except Exception:  # noqa: BLE001
        pass

    try:
        page.wait_for_function(PAGE_SETTLED_JS, timeout=timeout_ms)
    except Exception:  # noqa: BLE001
        return False

    return True


def current_url(page) -> str:
    """The page URL, or empty when the page has gone."""

    try:
        return str(page.url or "")
    except Exception:  # noqa: BLE001
        return ""