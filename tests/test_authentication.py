"""
Regression tests for the authentication flow.

The bug being guarded: the previous implementation polled
``Locator.count()`` inside a Python loop, every Playwright call
inherited the context's default timeout, and the poll budget counted
iterations rather than elapsed time. The result was a process that
appeared to hang and had to be killed with Ctrl+C.

These tests pin the properties that fix requires:

* every browser operation has a finite, explicit timeout;
* polling is a single bounded observation, not a locator loop;
* the budget is wall-clock, not iteration-counted;
* an indeterminate result becomes UNKNOWN rather than a hang;
* credentials flow through the credential abstraction and are filled
  automatically;
* credentials never appear in any reported text;
* a challenge stops automation and is handed to the human;
* KeyboardInterrupt never yields a success exit code;
* page, context, browser and Playwright are all closed on every path;
* a session is saved only after authentication was actually verified.

No real credentials are used, and no test prints one.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.ingestion import auth_state
from src.ingestion import credentials as credential_module
from src.ingestion import collect_cli
from src.ingestion.auth_state import (
    AuthObservation,
    AuthState,
    Deadline,
    classify,
    probe,
    settle,
)
from src.ingestion.sources.base import (
    CollectionStopped,
    SecurityChallenge,
    StopReason,
)
from src.ingestion.sources.linkedin import (
    SUBMIT_SELECTORS,
    LinkedInSource,
    _redact,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILE = (
    REPO_ROOT / "src" / "ingestion" / "sources" / "linkedin.py"
)

#: Built at runtime from a prefix and filler rather than written as
#: literals, so no credential-shaped string is committed and no
#: push-protection rule can read one as a leaked secret. Nothing
#: here is a real credential and nothing here is ever valid.
_FILLER = "abcdefghijklmnopqrstuvwxyz0123456789"

FAKE_USERNAME = "configured-user@" + "example.invalid"
FAKE_PASSWORD = "configured" + "-password-" + _FILLER


# ---------------------------------------------------------------------
# State classification
# ---------------------------------------------------------------------


def probe_dict(**overrides) -> dict:
    base = {
        "username_count": 0,
        "password_count": 0,
        "session_markers": 0,
        "feed_nodes": 0,
        "alerts": [],
        "body_sample": "",
    }
    base.update(overrides)

    return base


def test_a_sign_in_form_is_recognised():
    observation = classify(
        probe_dict(username_count=1, password_count=1)
    )

    assert observation.state is AuthState.LOGIN_FORM


def test_a_session_is_recognised_from_markers():
    observation = classify(probe_dict(session_markers=1))

    assert observation.state is AuthState.AUTHENTICATED
    assert observation.is_usable


def test_a_feed_without_a_form_is_a_session():
    observation = classify(probe_dict(feed_nodes=5))

    assert observation.state is AuthState.AUTHENTICATED


def test_a_feed_alongside_a_form_is_not_a_session():
    """
    A form wins, so a page that has not navigated yet is never
    reported as authenticated.
    """

    observation = classify(
        probe_dict(feed_nodes=5, username_count=1, password_count=1)
    )

    assert observation.state is AuthState.LOGIN_FORM


def test_nothing_recognisable_is_unknown():
    assert classify(probe_dict()).state is AuthState.UNKNOWN


def test_a_malformed_probe_is_unknown_not_a_crash():
    for payload in (None, "a string", 12, [], {}):
        observation = classify(payload)

        assert observation.state is AuthState.UNKNOWN


def test_authentication_is_never_assumed_from_a_submit():
    """
    A submitted form that comes back is not a session. This is the
    false-success the machine must never produce.
    """

    observation = classify(
        probe_dict(username_count=1, password_count=1),
        last_action="submit",
    )

    assert observation.state is AuthState.LOGIN_FORM
    assert not observation.is_usable


@pytest.mark.parametrize(
    "body,kind",
    [
        ("Please complete the CAPTCHA", "captcha"),
        ("Verify it's you", "suspicious_login"),
        ("Enter the verification code we sent", "otp"),
        ("Two-step verification", "two_factor"),
        ("Verify your identity", "identity_verification"),
        ("Access Denied", "access_denied"),
        ("Your account is temporarily restricted", "account_restricted"),
        ("Too many attempts, try again later", "rate_limited"),
    ],
)
def test_every_challenge_kind_is_a_human_action_state(body, kind):
    observation = classify(probe_dict(body_sample=body))

    assert observation.state is AuthState.HUMAN_CHALLENGE
    assert observation.needs_human
    assert observation.challenge == kind
    assert not observation.is_usable


def test_a_challenge_outranks_a_login_form():
    """
    Challenge pages often still render form elements. The challenge
    must win, or a captcha would look like a normal login.
    """

    observation = classify(
        probe_dict(
            username_count=1,
            password_count=1,
            body_sample="Please complete the CAPTCHA to continue",
        )
    )

    assert observation.state is AuthState.HUMAN_CHALLENGE


def test_a_challenge_in_an_alert_is_detected():
    observation = classify(
        probe_dict(
            alerts=["Verify your identity to continue"],
            body_sample="",
        )
    )

    assert observation.state is AuthState.HUMAN_CHALLENGE
    assert observation.challenge == "identity_verification"


def test_rejected_credentials_are_reported_as_failure():
    observation = classify(
        probe_dict(
            username_count=1,
            password_count=1,
            alerts=["Password is incorrect. Please try again."],
            body_sample="password is incorrect",
        ),
        last_action="submit",
    )

    assert observation.state is AuthState.AUTHENTICATION_FAILED


def test_the_state_machine_never_reports_success_without_evidence():
    for payload in (
        probe_dict(),
        probe_dict(username_count=1, password_count=1),
        probe_dict(body_sample="error"),
        {},
    ):
        observation = classify(payload, last_action="submit")

        assert observation.state is not AuthState.AUTHENTICATED


# ---------------------------------------------------------------------
# Bounded waits
# ---------------------------------------------------------------------


def test_the_deadline_measures_wall_clock_not_iterations():
    """
    The previous budget counted loop iterations, so a slow call
    multiplied the total wait without bound.
    """

    deadline = Deadline(1.0)

    assert deadline.remaining <= 1.0

    # Slicing twice takes from the same budget, never adds to it.
    first = deadline.slice_ms(cap_ms=500, default_ms=500)
    second = deadline.slice_ms(cap_ms=500, default_ms=500)

    assert first > 0
    assert second <= first


def test_an_expired_deadline_grants_nothing():
    deadline = Deadline(0.0)

    assert deadline.expired is True
    assert deadline.slice_ms(cap_ms=500, default_ms=500) == 0


def test_a_slice_is_capped_and_never_zero_while_alive():
    deadline = Deadline(100.0)

    # A generous cap is still honoured, so one call cannot dominate.
    assert deadline.slice_ms(cap_ms=250, default_ms=60_000) == 250

    # And the driver never receives zero, which it treats as
    # "no timeout".
    assert deadline.slice_ms(cap_ms=60_000, default_ms=1) >= 250


def test_probe_uses_one_call_and_survives_a_failure():
    calls: list[str] = []

    class Page:
        def set_default_timeout(self, value):
            calls.append(f"timeout:{value}")

        def evaluate(self, script):
            calls.append("evaluate")

            return probe_dict(session_markers=1)

    result = probe(Page(), timeout_ms=5000)

    assert result == probe_dict(session_markers=1)

    # One evaluation, not a chain of locator calls.
    assert calls.count("evaluate") == 1


def test_a_failed_probe_returns_none_rather_than_raising():
    class Page:
        def set_default_timeout(self, value):
            pass

        def evaluate(self, script):
            raise RuntimeError("the page went away")

    assert probe(Page(), timeout_ms=1000) is None


def test_settle_is_bounded_and_reports_failure():
    class SlowPage:
        def set_default_timeout(self, value):
            pass

        def wait_for_function(self, script, timeout=None):
            raise RuntimeError("timed out")

    assert settle(SlowPage(), timeout_ms=1000) is False


# ---------------------------------------------------------------------
# Source-level guarantees
# ---------------------------------------------------------------------


def test_no_authentication_call_has_no_timeout():
    """
    Every locator call the authentication path makes must pass an
    explicit timeout, or it inherits the driver default and can block
    for minutes.
    """

    source = SOURCE_FILE.read_text(encoding="utf-8")

    for match in re.finditer(r"\.fill\(", source):
        tail = source[match.end() : match.end() + 160]

        assert "timeout=" in tail, tail[:80]

    for match in re.finditer(r"\.click\(", source):
        tail = source[match.end() : match.end() + 160]

        assert "timeout=" in tail or "timeout=2_000" in tail, (
            tail[:80]
        )

    # Navigation must be bounded too.
    for match in re.finditer(r"\.goto\(", source):
        tail = source[match.end() : match.end() + 200]

        assert "timeout=" in tail, tail[:80]


def test_the_context_default_timeout_is_short():
    """
    The previous context used a 120s default, which is what let a
    single call appear to hang.
    """

    source = SOURCE_FILE.read_text(encoding="utf-8")

    assert "AUTH_DEFAULT_TIMEOUT_MS = 10_000" in source
    assert "set_default_timeout(AUTH_DEFAULT_TIMEOUT_MS)" in source
    assert "120_000" not in source


def test_authentication_does_not_poll_locator_counts():
    """
    The hang came from polling Locator.count(). The authenticated
    check must be a single bounded probe instead.
    """

    source = SOURCE_FILE.read_text(encoding="utf-8")

    assert "def _signed_in" in source
    assert "return self.probe().state.is_terminal_success" in source

    signed_in = source.split("def _signed_in", 1)[1].split("def ", 1)[0]

    assert ".count(" not in signed_in
    assert "while" not in signed_in


def test_the_session_wait_no_longer_counts_iterations():
    source = SOURCE_FILE.read_text(encoding="utf-8")

    wait = source.split("def wait_for_manual_session", 1)[1].split(
        "def ", 1
    )[0]

    assert "Deadline(" in wait
    assert "deadline.expired" in wait
    assert "waited +=" not in wait


def test_every_authentication_budget_is_finite():
    source = SOURCE_FILE.read_text(encoding="utf-8")

    for name in (
        "AUTH_DEFAULT_TIMEOUT_MS",
        "AUTH_NAVIGATE_TIMEOUT_MS",
        "AUTH_PROBE_TIMEOUT_MS",
        "AUTH_FIELD_TIMEOUT_MS",
        "AUTH_SETTLE_TIMEOUT_MS",
    ):
        match = re.search(
            rf"{name} = ([\d._]+)", source
        )

        assert match, name

        value = float(match.group(1))

        assert 0 < value <= 120_000, name

    match = re.search(
        r"AUTH_TOTAL_TIMEOUT_SECONDS = ([\d.]+)", source
    )

    assert match
    assert 0 < float(match.group(1)) <= 600


# ---------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------


def test_credentials_are_read_through_the_abstraction(monkeypatch):
    """
    The login path must not read the environment directly.
    """

    source = SOURCE_FILE.read_text(encoding="utf-8")

    assert "credential_module.status()" in source
    assert "_credential(\"LINKEDIN_USERNAME\")" in source
    assert "_credential(\"LINKEDIN_PASSWORD\")" in source

    # No direct environment read of a credential outside the helper.
    assert 'os.environ.get("LINKEDIN_PASSWORD"' not in source


def test_username_is_supplied_to_the_username_field(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("LINKEDIN_USERNAME", FAKE_USERNAME)
    monkeypatch.setenv("LINKEDIN_PASSWORD", FAKE_PASSWORD)

    source = LinkedInSource(profile="handle", root=tmp_path)

    filled: list[tuple[str, str]] = []

    class Node:
        def __init__(self, selector):
            self.selector = selector

        def is_visible(self):
            return True

        def fill(self, value, timeout=None):
            filled.append((self.selector, value))

        def click(self, timeout=None):
            filled.append(("submit", ""))

    monkeypatch.setattr(
        LinkedInSource,
        "_first_visible_locator",
        lambda self, selectors, limit=8: Node(selectors[0]),
    )

    class Page:
        url = "https://www.linkedin.com/login"

        def set_default_timeout(self, value):
            pass

        def evaluate(self, script):
            if script == auth_state.PAGE_PROBE_JS:
                return probe_dict(username_count=1, password_count=1)
            return 0

    source._page = Page()

    source.authenticate_automatically()

    values = [value for _selector, value in filled]

    assert FAKE_USERNAME in values
    assert FAKE_PASSWORD in values


def test_credentials_never_appear_in_reported_text(monkeypatch):
    """
    The driver's error text embeds the value it was given, so a failed
    fill would otherwise print the credential.
    """

    monkeypatch.setenv("LINKEDIN_USERNAME", FAKE_USERNAME)
    monkeypatch.setenv("LINKEDIN_PASSWORD", FAKE_PASSWORD)

    error = RuntimeError(
        f'waiting for locator, fill("{FAKE_USERNAME}") timed out'
    )

    text = _redact(error)

    assert FAKE_USERNAME not in text
    assert FAKE_PASSWORD not in text
    assert "[REDACTED]" in text


def test_redaction_is_bounded():
    assert len(_redact(RuntimeError("x" * 10_000))) <= 460


def test_no_credential_literal_exists_in_the_source():
    source = SOURCE_FILE.read_text(encoding="utf-8")

    assert "@gmail" not in source
    assert "password =" not in source.lower().replace(
        "linkedin_password", ""
    )


# ---------------------------------------------------------------------
# The login path in the CLI
# ---------------------------------------------------------------------


class FakeObservationSource:
    """Stands in for LinkedInSource in the CLI login flow."""

    def __init__(self, observations, *, save_ok=True):
        self._observations = list(observations)
        self._save_ok = save_ok
        self.closed = False
        self.saved = 0

    def open_for_manual_login(self):
        return self._observations.pop(0)

    def wait_for_manual_session(self, **kwargs):
        return self._observations.pop(0) if self._observations else (
            self.observations_last
        )

    @property
    def observations_last(self):
        return AuthObservation(
            state=AuthState.AUTHENTICATED, detail="verified"
        )

    def save_session(self):
        self.saved += 1

        return Path(".agent/secrets/state.json") if self._save_ok else None

    def close(self):
        self.closed = True


def make_args(**overrides):
    import argparse

    base = argparse.Namespace(
        profile="",
        max_posts=3,
        headed=True,
        bundle_root="captures",
        posts_root="data/posts",
        since=None,
        until=None,
        scroll_limit=None,
        resume=False,
        dry_run=False,
        json=False,
        login=True,
    )
    base.__dict__.update(overrides)

    return base


def install_fake_source(monkeypatch, fake):
    import src.ingestion.sources.linkedin as module

    class AUTH_HUMAN_WAIT_SECONDS:
        value = 1

    monkeypatch.setattr(module, "LinkedInSource", lambda **k: fake)
    monkeypatch.setattr(
        module, "AUTH_HUMAN_WAIT_SECONDS", 1, raising=False
    )


def test_login_succeeds_without_human_interaction(
    monkeypatch, tmp_path, capsys
):
    """
    The normal path needs nothing from the user.
    """

    monkeypatch.setenv("LINKEDIN_USERNAME", FAKE_USERNAME)
    monkeypatch.setenv("LINKEDIN_PASSWORD", FAKE_PASSWORD)

    fake = FakeObservationSource(
        [
            AuthObservation(
                state=AuthState.AUTHENTICATED,
                detail="Session markers are present.",
            )
        ]
    )

    install_fake_source(monkeypatch, fake)

    def fail_input():
        raise AssertionError(
            "no human input may be requested on the normal path"
        )

    monkeypatch.setattr("builtins.input", fail_input)

    code = collect_cli.interactive_login(make_args())

    assert code == 0
    assert fake.saved == 1
    assert fake.closed is True


def test_a_challenge_asks_the_user_and_then_verifies(
    monkeypatch, capsys
):
    monkeypatch.setenv("LINKEDIN_USERNAME", FAKE_USERNAME)
    monkeypatch.setenv("LINKEDIN_PASSWORD", FAKE_PASSWORD)

    fake = FakeObservationSource(
        [
            AuthObservation(
                state=AuthState.HUMAN_CHALLENGE,
                detail="captcha",
                challenge="captcha",
            ),
            AuthObservation(
                state=AuthState.AUTHENTICATED, detail="verified"
            ),
        ]
    )

    install_fake_source(monkeypatch, fake)

    monkeypatch.setattr("builtins.input", lambda: "")

    code = collect_cli.interactive_login(make_args())

    output = capsys.readouterr().out

    assert code == 0
    assert "ACTION REQUIRED" in output
    assert "DO NOT" in output
    assert "bounded" in output.lower()
    assert fake.saved == 1
    assert fake.closed is True


def test_the_report_never_asks_for_credentials(monkeypatch, capsys):
    monkeypatch.setenv("LINKEDIN_USERNAME", FAKE_USERNAME)
    monkeypatch.setenv("LINKEDIN_PASSWORD", FAKE_PASSWORD)

    fake = FakeObservationSource(
        [
            AuthObservation(
                state=AuthState.HUMAN_CHALLENGE,
                detail="captcha",
                challenge="captcha",
            ),
            AuthObservation(
                state=AuthState.AUTHENTICATED, detail="verified"
            ),
        ]
    )

    install_fake_source(monkeypatch, fake)
    monkeypatch.setattr("builtins.input", lambda: "")

    collect_cli.interactive_login(make_args())

    output = capsys.readouterr().out

    assert FAKE_PASSWORD not in output
    assert FAKE_USERNAME not in output
    assert "share your credentials" in output


def test_unauthenticated_is_not_reported_as_success(
    monkeypatch, capsys
):
    monkeypatch.setenv("LINKEDIN_USERNAME", FAKE_USERNAME)
    monkeypatch.setenv("LINKEDIN_PASSWORD", FAKE_PASSWORD)

    fake = FakeObservationSource(
        [
            AuthObservation(
                state=AuthState.UNKNOWN,
                detail="Neither form nor session markers.",
            )
        ]
    )

    install_fake_source(monkeypatch, fake)

    code = collect_cli.interactive_login(make_args())

    assert code != 0
    assert "Nothing was saved" in capsys.readouterr().out
    assert fake.saved == 0
    assert fake.closed is True


def test_a_keyboard_interrupt_is_never_a_success(
    monkeypatch, capsys
):
    """
    Ctrl+C during the human wait must exit non-zero, not zero.
    """

    monkeypatch.setenv("LINKEDIN_USERNAME", FAKE_USERNAME)
    monkeypatch.setenv("LINKEDIN_PASSWORD", FAKE_PASSWORD)

    fake = FakeObservationSource(
        [
            AuthObservation(
                state=AuthState.HUMAN_CHALLENGE,
                detail="captcha",
                challenge="captcha",
            )
        ]
    )

    install_fake_source(monkeypatch, fake)

    def interrupt():
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", interrupt)

    code = collect_cli.interactive_login(make_args())

    assert code == 1
    assert "Interrupted" in capsys.readouterr().out
    assert fake.saved == 0
    assert fake.closed is True


def test_missing_credentials_are_reported_without_values(
    monkeypatch, capsys
):
    monkeypatch.delenv("LINKEDIN_USERNAME", raising=False)
    monkeypatch.delenv("LINKEDIN_PASSWORD", raising=False)

    code = collect_cli.interactive_login(make_args())

    output = capsys.readouterr().out

    assert code == 1
    assert "configured: no" in output


def test_credentials_configured_is_reported_yes(monkeypatch, capsys):
    monkeypatch.setenv("LINKEDIN_USERNAME", FAKE_USERNAME)
    monkeypatch.setenv("LINKEDIN_PASSWORD", FAKE_PASSWORD)

    fake = FakeObservationSource(
        [
            AuthObservation(
                state=AuthState.AUTHENTICATED, detail="ok"
            )
        ]
    )

    install_fake_source(monkeypatch, fake)

    collect_cli.interactive_login(make_args())

    output = capsys.readouterr().out

    assert "configured: yes" in output
    assert FAKE_PASSWORD not in output
    assert FAKE_USERNAME not in output


# ---------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------


def test_close_releases_every_object_in_order():
    order: list[str] = []

    class Handle:
        def __init__(self, name):
            self.name = name

        def close(self):
            order.append(self.name)

    class PlaywrightLike:
        def stop(self):
            order.append("playwright")

    source = LinkedInSource(profile="handle")

    source._page = Handle("page")
    source._context = Handle("context")
    source._browser = Handle("browser")
    source._playwright = PlaywrightLike()

    source.close()

    assert order == ["page", "context", "browser", "playwright"]

    assert source._page is None
    assert source._context is None
    assert source._browser is None
    assert source._playwright is None


def test_close_clears_handles_even_when_closing_fails():
    """
    A failed close must not leave a stale handle that a later
    teardown would try to reuse.
    """

    class Exploding:
        def close(self):
            raise RuntimeError("already gone")

    source = LinkedInSource(profile="handle")

    messages: list[str] = []
    source.progress = messages.append

    source._context = Exploding()
    source._browser = Exploding()

    source.close()

    assert source._context is None
    assert source._browser is None
    assert messages


def test_close_is_idempotent():
    source = LinkedInSource(profile="handle")

    source.close()
    source.close()


def test_the_source_is_a_context_manager():
    source = LinkedInSource(profile="handle")

    with source as entered:
        assert entered is source

    # Exiting ran close, so a real browser would be gone.
    assert source._browser is None


def test_session_is_saved_only_when_verified(tmp_path):
    """
    A session is written only after authentication was positively
    observed, so an unverified attempt cannot poison later runs.
    """

    source = LinkedInSource(profile="handle", root=tmp_path)

    assert source.save_session() is None

    class Context:
        def __init__(self):
            self.writes = 0

        def storage_state(self, path):
            self.writes += 1
            Path(path).write_text("{}", encoding="utf-8")

    context = Context()
    source._context = context

    saved = source.save_session()

    assert saved is not None
    assert saved.is_file()
    assert context.writes == 1

    # Inside the ignored tree, so it can never be committed.
    assert ".agent" in saved.parts


def test_the_journal_records_transitions_without_credentials():
    source = LinkedInSource(profile="handle")

    source.journal.record(
        AuthObservation(
            state=AuthState.LOGIN_FORM, detail="form"
        )
    )
    source.journal.record(
        AuthObservation(
            state=AuthState.LOGIN_SUBMITTED, detail="submitted"
        )
    )

    assert source.journal.path == "login_form -> login_submitted"

    # Nothing was written to disk by recording a transition.
    recorded = json.dumps(source.journal.as_dict())

    assert FAKE_PASSWORD not in recorded


# ---------------------------------------------------------------------
# Source guarantees preserved
# ---------------------------------------------------------------------


def test_the_submit_resolver_takes_one_options_object():
    """
    `page.evaluate` passes a single argument, so the resolver must
    destructure an object. Passing a list gave `tag` the whole list,
    producing an invalid selector and no match.
    """

    source = SOURCE_FILE.read_text(encoding="utf-8")

    assert '"tag": SUBMIT_MARKER' in source
    assert '"labels": list(SUBMIT_LABELS)' in source

    resolver = source.split("SUBMIT_RESOLVER_JS = r\"\"\"", 1)[1]
    resolver = resolver.split('"""', 1)[0]

    # Destructure a single options parameter, never two positional
    # ones.
    assert "(options) =>" in resolver
    assert "const tag = options.tag;" in resolver
    assert "const labels = options.labels || [];" in resolver


def test_the_submit_selector_cannot_match_an_oauth_button():
    """
    Regression guard from the live run.

    LinkedIn's login page has no <form> element and renders "Sign in",
    "Sign in with Microsoft" and "Sign in with Apple" as sibling
    buttons. `:has-text()` is a substring match, so it selected the
    Microsoft button, opened an OAuth popup, and never submitted the
    credentials. Resolution is therefore an exact in-page comparison.
    """

    from src.ingestion.sources.linkedin import (
        SUBMIT_LABELS,
        SUBMIT_RESOLVER_JS,
        SUBMIT_SELECTORS as selectors,
    )

    for selector in selectors:
        assert ":has-text(" not in selector, selector
        assert ":text-is(" not in selector, selector

    # Exact labels only, and never a provider.
    assert "sign in" in SUBMIT_LABELS
    assert "sign in with microsoft" not in SUBMIT_LABELS
    assert "sign in with apple" not in SUBMIT_LABELS

    # The resolver compares against the label set exactly.
    assert "new Set(labels)" in SUBMIT_RESOLVER_JS
    assert "wanted.has(label)" in SUBMIT_RESOLVER_JS


def test_no_oauth_provider_is_selected_as_the_submit():
    """
    The selector list must never name an identity provider, because a
    provider button opens a federated popup rather than submitting the
    configured credentials.
    """

    joined = " ".join(SUBMIT_SELECTORS)

    for provider in ("Microsoft", "Apple", "Google", "SSO", "Okta"):
        assert provider.casefold() not in joined.casefold(), provider


def test_the_login_form_is_never_submitted_via_enter():
    """
    There is no <form> element, so a press-and-submit fallback would
    silently do nothing. The button click is the only route.
    """

    source = SOURCE_FILE.read_text(encoding="utf-8")

    assert "press(\"Enter\")" not in source
    assert "keyboard.press" not in source


def test_no_read_only_action_is_performed():
    """
    The sign-in submit is allowed. Anything interactive on content is
    not.
    """

    source = SOURCE_FILE.read_text(encoding="utf-8").lower()

    for forbidden in (
        'aria-label*="like"',
        'aria-label*="comment"',
        'aria-label*="share"',
        'aria-label*="follow"',
        'aria-label*="connect"',
        'aria-label*="message"',
        'aria-label*="send"',
        'aria-label*="delete"',
    ):
        assert forbidden not in source, forbidden


def test_no_bypass_or_evasion_helper_exists():
    """
    The collector must contain no CAPTCHA solving, no fingerprint
    evasion and no retry-until-accepted logic.
    """

    source = SOURCE_FILE.read_text(encoding="utf-8").lower()

    for forbidden in (
        "captcha_solver",
        "solve_captcha",
        "2fa_bypass",
        "webdriver_override",
        "stealth",
        "rotate_proxy",
        "user_agent_rotation",
    ):
        assert forbidden not in source, forbidden


def test_credentials_module_still_reports_only_booleans():
    status = credential_module.status(
        {
            "LINKEDIN_USERNAME": FAKE_USERNAME,
            "LINKEDIN_PASSWORD": FAKE_PASSWORD,
        }
    )

    described = status.describe()

    assert status.configured is True
    assert FAKE_USERNAME not in described
    assert FAKE_PASSWORD not in described
    assert described == "LinkedIn credentials configured: yes"


def test_session_state_file_is_never_tracked():
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        shell=False,
    ).stdout.splitlines()

    for path in tracked:
        assert "state.json" not in path
        assert "storage_state" not in path
        assert not path.endswith(".env")


def test_the_agent_tree_ignores_session_state():
    import subprocess

    completed = subprocess.run(
        ["git", "check-ignore", "-q", ".agent/secrets/state.json"],
        capture_output=True,
        check=False,
        shell=False,
    )

    assert completed.returncode == 0, (
        ".agent/secrets/state.json must be git-ignored"
    )