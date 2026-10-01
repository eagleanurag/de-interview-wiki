"""
An authorized LinkedIn collector.

Scope and safety
----------------
This collects the user's *own* accessible posts through a browser the
user authorizes. It is deliberately read-only:

* it never likes, comments, shares, reposts, connects, follows,
  messages, posts, edits, deletes, or changes any setting;
* it only navigates and expands content;
* the only interactive elements it clicks are navigation controls and
  "show more" expanders, which are required to reach the content.

It never defeats a security challenge. A CAPTCHA, MFA, OTP, account
restriction or explicit access denial stops collection and asks the
human, because that is the correct and only appropriate response.

Credentials are read from the environment at the moment of use. They
are never placed in a URL, a command-line argument, a log line, or any
file the collector writes.

Everything the browser touches lives under ``.agent/``, which is
git-ignored, so no profile, cookie jar, or storage state can be
committed.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from src.ingestion import credentials as credential_module
from src.ingestion.sources.base import (
    CollectedPost,
    CollectionStopped,
    SecurityChallenge,
    StopReason,
    Source,
)


PROFILE_URL = "https://www.linkedin.com"

FEED_SELECTOR = 'div.feed-shared-update-v2, [data-id^="urn:li:activity"]'

POST_ID_PATTERN = re.compile(r"urn:li:(activity|share|ugcPost):(\d+)")

# Text that indicates the account is being challenged or blocked.
CHALLENGE_MARKERS = (
    ("captcha", "captcha"),
    ("recaptcha", "recaptcha"),
    ("hcaptcha", "hcaptcha"),
    ("verify it is you", "unusual_login"),
    ("verify it's you", "unusual_login"),
    ("unusual activity", "unusual_login"),
    ("security verification", "security_verification"),
    ("identity verification", "identity_verification"),
    ("verify your identity", "identity_verification"),
    ("two-step verification", "two_factor"),
    ("two factor", "two_factor"),
    ("enter the code", "otp"),
    ("verification code", "otp"),
    ("one-time code", "otp"),
    ("authenticate to continue", "authentication"),
    ("please sign in to continue", "authentication"),
    ("sign in to confirm", "authentication"),
    ("access denied", "access_denied"),
    ("account restricted", "account_restricted"),
    ("temporarily restricted", "account_restricted"),
)

# Controls that may be clicked. Anything else is left alone.
EXPAND_SELECTORS = (
    'button[aria-label*="Show more"]',
    'button[aria-label*="see more"]',
    'button[class*="feed-shared-inline-show-more"]',
)

SCROLL_SETTLE_TIMEOUT_MS = 1500
DEFAULT_SCROLL_LIMIT = 400
DEFAULT_IDLE_ROUNDS = 3

# Bounded so a single post cannot pull an unbounded number of images.
MAX_MEDIA_PER_POST = 4


def linkedin_installed() -> bool:
    """Whether Playwright is importable."""

    try:
        import playwright  # noqa: F401
    except ImportError:
        return False

    return True


def playwright_install_hint() -> str:
    return (
        "The LinkedIn source needs Playwright:\n"
        "  pip install playwright\n"
        "  python -m playwright install chromium"
    )


@dataclass
class LinkedInLimits:
    """Bounds for one collection run."""

    max_posts: int | None = None
    since: str | None = None
    until: str | None = None
    scroll_limit: int = DEFAULT_SCROLL_LIMIT
    idle_rounds: int = DEFAULT_IDLE_ROUNDS


# LinkedIn rebuilt its sign-in page. The username and password inputs
# no longer carry `session_key`/`password` ids or a named form, and the
# ids are regenerated per render. Semantic attributes are stable, so
# they are what the collector keys on. Several alternatives are listed
# because the page renders more than one copy (a sign-in panel and a
# join panel); the first visible match is used.
USERNAME_SELECTORS = (
    'input[autocomplete="username"]',
    'input[type="email"]',
    'input#session_key',
    'input[name="session_key"]',
)

PASSWORD_SELECTORS = (
    'input[autocomplete="current-password"]',
    'input[type="password"]',
    'input#password',
)

SUBMIT_SELECTORS = (
    'button[type="submit"]',
    'button:has-text("Sign in")',
    'button:has-text("sign in")',
    'input[type="submit"]',
)


@dataclass
class SelectorSet:
    """
    CSS selectors used for extraction.

    Kept in one place so a layout change is a data change, and so the
    layout can be pointed at a local fixture during tests without
    touching a real site.
    """

    feed: str = FEED_SELECTOR
    text: str = (
        'div.feed-shared-inline-show-more-text, '
        'div.update-components-text, '
        'span.feed-shared-inline-show-more-text'
    )
    permalink: str = (
        'a[href*="/feed/update/"], a[data-tracking-control-name*="share"]'
    )
    timestamp: str = (
        "time, span[data-aria-label*='ago'], "
        "span.feed-shared-actor__sub-description"
    )
    media: str = (
        'figure img, div.feed-shared-image img, '
        'img[data-delayed-url], img[src*="media.licdn.com"]'
    )
    expand: str = ", ".join(EXPAND_SELECTORS)
    profile_link: str = 'a.app-navigation__link[href*="/in/"]'
    activity_tab: str = (
        'a[href*="/in/"][href*="/recent-activity/all/"], '
        'a[data-id*="profile Activity"]'
    )


class LinkedInSource(Source):
    """
    Collects the user's own posts from their LinkedIn activity feed.

    Playwright is imported lazily, so this module can be imported, and
    the rest of the suite run, without a browser installed.
    """

    name = "linkedin"
    platform = "linkedin"

    def __init__(
        self,
        *,
        profile: str | None = None,
        headed: bool = False,
        root: str | Path = ".",
        selectors: SelectorSet | None = None,
        limits: LinkedInLimits | None = None,
        env_file: str | Path = ".env",
    ) -> None:
        self.profile = (profile or "").strip()
        self.headed = headed
        self.root = Path(root)
        self.selectors = selectors or SelectorSet()
        self.limits = limits or LinkedInLimits()

        credential_module.load_local_environment(env_file)

        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._seen: set[str] = set()

        #: The profile actually read, resolved from the session when
        #: no handle was configured. Recorded so a run can be audited
        #: without exposing credentials.
        self.resolved_profile: str = ""

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    def start(self) -> None:
        """
        Launch the browser and authenticate if needed.

        Raises SecurityChallenge rather than attempting to get past a
        challenge.
        """

        if not linkedin_installed():
            raise CollectionStopped(
                StopReason.FAILED, playwright_install_hint()
            )

        from playwright.sync_api import sync_playwright

        credential_module.require()

        profile_directory = credential_module.browser_profile_directory(
            self.root
        )

        self._playwright = sync_playwright().start()

        try:
            self._browser = self._playwright.chromium.launch(
                headless=not self.headed
            )
        except Exception as exc:  # noqa: BLE001
            self.close()
            raise CollectionStopped(
                StopReason.FAILED,
                f"Could not launch the browser: {exc}",
            ) from exc

        self._context = self._browser.new_context(
            storage_state=(
                str(profile_directory / "state.json")
                if (profile_directory / "state.json").is_file()
                else None
            )
        )

        # Defaults that reduce automation fingerprints without evading
        # any access control.
        self._context.set_default_timeout(30_000)
        self._context.set_extra_http_headers(
            {"Accept-Language": "en-US,en;q=0.9"}
        )

        self._page = self._context.new_page()

        self._ensure_authenticated()

        # Persist the session so a later run does not need to sign in
        # again. This is what makes a resume possible after a human
        # completes a challenge by hand.
        self.save_session()

    def save_session(self) -> Path | None:
        """
        Store the authenticated session inside the ignored tree.

        The file holds cookies and is treated as a credential: it lives
        under ``.agent/secrets/``, is never committed, and its path is
        never logged.
        """

        if self._context is None:
            return None

        directory = credential_module.ensure_secrets_directory(
            self.root
        )

        target = directory / "state.json"

        try:
            self._context.storage_state(path=str(target))
        except Exception as exc:  # noqa: BLE001
            self.progress(f"Could not save the session: {_redact(exc)}")
            return None

        credential_module.restrict_permissions(target)

        return target

    def close(self) -> None:
        """Close the browser. Safe to call repeatedly."""

        for attribute in ("_context", "_browser", "_playwright"):
            handle = getattr(self, attribute, None)

            if handle is None:
                continue

            closer = getattr(handle, "close", None) or getattr(
                handle, "stop", None
            )

            if closer is None:
                continue

            try:
                closer()
            except Exception:  # noqa: BLE001
                # Teardown must never mask the real outcome.
                pass

            setattr(self, attribute, None)

    # -----------------------------------------------------------------
    # Authentication
    # -----------------------------------------------------------------

    def _ensure_authenticated(self) -> None:
        """
        Reach the profile feed, signing in only if required.

        Any challenge found here stops collection and asks the human.
        """

        self._require_page().goto(
            f"{PROFILE_URL}/feed/", wait_until="domcontentloaded"
        )

        self._assert_no_challenge("sign in")

        if self._signed_in():
            return

        self._sign_in()

    def _signed_in(self) -> bool:
        """Whether the session is authenticated."""

        page = self._require_page()

        try:
            if self._first_visible(USERNAME_SELECTORS):
                return False

            for selector in (
                'img[alt*="profile photo"]',
                'button[aria-label*="profile"]',
                'a[data-tracking-control-name="nav_profile"]',
            ):
                if page.locator(selector).count():
                    return True
        except Exception:  # noqa: BLE001
            return False

        return False

    def _first_visible(self, selectors: tuple[str, ...]) -> str | None:
        """
        The first selector with a visible match.

        Used for detection only, where the selector string is enough.
        """

        page = self._require_page()

        for selector in selectors:
            try:
                located = page.locator(selector)
                count = located.count()

                for index in range(min(count, 4)):
                    if located.nth(index).is_visible():
                        return selector
            except Exception:  # noqa: BLE001
                continue

        return None

    def _first_visible_locator(
        self,
        selectors: tuple[str, ...],
        limit: int = 6,
    ):
        """
        The first *visible element* matching any selector.

        The element itself is returned, not its selector. The sign-in
        page renders several overlapping panels, and re-resolving a
        selector afterwards picks the first match, which is one of the
        hidden ones.
        """

        page = self._require_page()

        for selector in selectors:
            try:
                located = page.locator(selector)
                count = located.count()

                for index in range(min(count, limit)):
                    element = located.nth(index)

                    if element.is_visible():
                        return element
            except Exception:  # noqa: BLE001
                continue

        return None

    def open_for_manual_login(self):
        """
        Launch a browser and hand it to the user. No automation past
        this point.

        Used by the ``--login`` path so a human completes any challenge
        themselves, once, rather than on every run.
        """

        if not linkedin_installed():
            raise CollectionStopped(
                StopReason.FAILED, playwright_install_hint()
            )

        from playwright.sync_api import sync_playwright

        self._playwright = sync_playwright().start()

        try:
            self._browser = self._playwright.chromium.launch(
                headless=False
            )
        except Exception as exc:  # noqa: BLE001
            self.close()
            raise CollectionStopped(
                StopReason.FAILED,
                f"Could not launch the browser: {_redact(exc)}",
            ) from exc

        self._context = self._browser.new_context()
        self._context.set_default_timeout(120_000)

        self._page = self._context.new_page()

        page = self._require_page()

        page.goto(
            f"{PROFILE_URL}/login", wait_until="domcontentloaded"
        )

        return page

    def wait_for_manual_session(
        self,
        *,
        timeout_seconds: int = 600,
        interval_seconds: int = 5,
    ) -> bool:
        """
        Poll until the user has signed in by hand.

        Only asks whether the session is authenticated. It never
        inspects or completes a challenge.
        """

        page = self._require_page()

        waited = 0

        while waited <= timeout_seconds:
            try:
                if self._signed_in():
                    return True
            except Exception:  # noqa: BLE001
                pass

            try:
                page.wait_for_timeout(interval_seconds * 1000)
            except Exception:  # noqa: BLE001
                pass

            waited += interval_seconds

        return False

    def _require_page(self):
        """The page, or a clear error if the browser never started."""

        if self._page is None:
            raise CollectionStopped(
                StopReason.FAILED,
                "The browser is not running. Call start() first.",
            )

        return self._page

    def _sign_in(self) -> None:
        """
        Sign in with locally configured credentials.

        Credentials are typed through Playwright's ``fill`` rather than
        being interpolated into a selector, a URL, or a log line. The
        password never appears anywhere else in the process.
        """

        page = self._require_page()

        state = credential_module.status()

        if not state.configured:
            raise CollectionStopped(
                StopReason.FAILED,
                "LinkedIn credentials are not configured.",
            )

        username_field = self._first_visible_locator(USERNAME_SELECTORS)
        password_field = self._first_visible_locator(PASSWORD_SELECTORS)

        if username_field is None or password_field is None:
            raise CollectionStopped(
                StopReason.LAYOUT_CHANGED,
                "Could not find the sign-in form. LinkedIn's login "
                "layout appears to have changed; the collector's "
                "selectors need updating.",
            )

        submit = self._first_visible_locator(SUBMIT_SELECTORS)

        if submit is None:
            raise CollectionStopped(
                StopReason.LAYOUT_CHANGED,
                "Could not find the sign-in button.",
            )

        try:
            # The located element is used directly rather than its
            # selector. Re-resolving the selector would pick the first
            # match, which on this page is a hidden panel.
            username_field.fill(_credential("LINKEDIN_USERNAME"))
            password_field.fill(_credential("LINKEDIN_PASSWORD"))

            submit.click()

            page.wait_for_load_state("domcontentloaded")
        except Exception as exc:  # noqa: BLE001
            # The driver's error text can echo the value that was
            # typed, so it is redacted before being reported or
            # printed. This is the difference between a failed sign-in
            # and a leaked username.
            raise CollectionStopped(
                StopReason.FAILED,
                "Sign-in did not complete: "
                + _redact(exc)
                + ". If the form was found but not editable, the "
                "page layout may have changed.",
            ) from None

        self._assert_no_challenge("sign in")

        if not self._signed_in():
            # Either the credentials are wrong or the account needs a
            # challenge. Either way this is the human's call, and the
            # distinction is not guessable from the page.
            raise SecurityChallenge(
                "authentication",
                "Sign-in did not succeed. The credentials may be "
                "incorrect, or the account may require an additional "
                "verification step. Complete any challenge in the open "
                "browser window, then resume.",
            )

    # -----------------------------------------------------------------
    # Challenge detection
    # -----------------------------------------------------------------

    def _assert_no_challenge(self, stage: str) -> None:
        """
        Stop if the page shows a security challenge.

        Detection is deliberately broad and biased toward stopping. A
        false positive costs the user one confirmation; a false
        negative would mean bypassing a challenge, which is not an
        acceptable trade.
        """

        self._require_page()

        try:
            body = (
                self._page.inner_text("body", timeout=5_000) or ""
            ).lower()
        except Exception:  # noqa: BLE001
            return

        for marker, kind in CHALLENGE_MARKERS:
            if marker in body:
                raise SecurityChallenge(
                    kind,
                    f"LinkedIn presented a {kind} challenge during "
                    f"{stage}. Complete it in the open browser window, "
                    f"then resume.",
                )

    # -----------------------------------------------------------------
    # Discovery
    # -----------------------------------------------------------------

    def discover(self, **limits: object) -> Iterator[CollectedPost]:
        """
        Walk the activity feed and yield posts.

        Stops on a limit, on exhaustion, on a challenge, or when the
        page stops yielding anything new.
        """

        if self._page is None:
            self.start()

        self._require_page()

        effective = self.limits

        if limits.get("max_posts") is not None:
            effective = replace(
                effective,
                max_posts=int(limits["max_posts"]),
            )

        if limits.get("since") is not None:
            effective = replace(
                effective, since=str(limits["since"])
            )

        if limits.get("until") is not None:
            effective = replace(
                effective, until=str(limits["until"])
            )

        if limits.get("scroll_limit") is not None:
            effective = replace(
                effective, scroll_limit=int(limits["scroll_limit"])
            )

        self._go_to_activity()

        produced = 0
        idle_rounds = 0
        scrolls = 0
        seen: set[str] = set(self._seen)

        while True:
            self._assert_no_challenge("collection")

            posts = self._extract_current()

            new_this_round = 0

            for collected in posts:
                if collected.source_post_id in seen:
                    continue

                seen.add(collected.source_post_id)
                self._seen.add(collected.source_post_id)
                new_this_round += 1

                if (
                    effective.max_posts is not None
                    and produced >= effective.max_posts
                ):
                    raise CollectionStopped(
                        StopReason.MAX_POSTS,
                        f"Reached the configured limit of "
                        f"{effective.max_posts} posts.",
                    )

                if not self._within_window(collected.published_at, effective):
                    continue

                produced += 1

                yield collected

            if new_this_round:
                idle_rounds = 0
            else:
                idle_rounds += 1

            if (
                effective.max_posts is not None
                and produced >= effective.max_posts
            ):
                raise CollectionStopped(
                    StopReason.MAX_POSTS,
                    f"Reached the configured limit of "
                    f"{effective.max_posts} posts.",
                )

            if idle_rounds >= effective.idle_rounds:
                raise CollectionStopped(
                    StopReason.NO_NEW_CONTENT,
                    f"No new posts after {idle_rounds} passes with no "
                    f"change.",
                )

            if scrolls >= effective.scroll_limit:
                raise CollectionStopped(
                    StopReason.SCROLL_LIMIT,
                    f"Reached the configured scroll limit of "
                    f"{effective.scroll_limit}.",
                )

            scrolls += 1
            self._scroll_once()

    # -----------------------------------------------------------------
    # Navigation and extraction
    # -----------------------------------------------------------------

    def _go_to_activity(self) -> None:
        """
        Open the authenticated user's own activity feed.

        When no profile handle was configured, the slug is resolved
        from the signed-in session via ``/in/me/``, which redirects to
        the real profile. That avoids making the user look up their own
        public handle, and it guarantees the feed belongs to the
        authenticated account rather than someone else's.
        """

        page = self._require_page()

        slug = self._resolve_slug()

        if not slug:
            raise CollectionStopped(
                StopReason.FAILED,
                "Could not resolve the authenticated profile. Pass "
                "--profile with your LinkedIn handle.",
            )

        self.resolved_profile = slug

        page.goto(
            f"{PROFILE_URL}/in/{slug}/recent-activity/all/",
            wait_until="domcontentloaded",
        )

        self._assert_no_challenge("navigation")

    def _resolve_slug(self) -> str:
        """
        Determine which profile to read.

        A configured handle wins. Otherwise the session's own profile
        is resolved, so the collector only ever reads the account the
        user actually authenticated as.
        """

        configured = self.profile.strip().strip("/")

        if configured:
            if configured.startswith("in/"):
                configured = configured[3:]

            return configured

        self._require_page()

        try:
            self._page.goto(
                f"{PROFILE_URL}/in/me/", wait_until="domcontentloaded"
            )
        except Exception as exc:  # noqa: BLE001
            raise CollectionStopped(
                StopReason.FAILED,
                f"Could not resolve the signed-in profile: {exc}",
            ) from exc

        self._assert_no_challenge("profile resolution")

        url = ""

        try:
            url = self._page.url or ""
        except Exception:  # noqa: BLE001
            url = ""

        match = re.search(
            r"linkedin\.com/in/([^/?#]+)", url
        )

        if match and match.group(1) not in {"me", ""}:
            return match.group(1)

        # Some responses do not change the URL, so fall back to the
        # profile link rendered in the navigation.
        try:
            link = self._page.locator(
                self.selectors.profile_link
            ).first

            href = link.get_attribute("href") or ""
        except Exception:  # noqa: BLE001
            return ""

        match = re.search(r"/in/([^/?#]+)", href)

        return match.group(1) if match else ""

    def _expand_truncated_posts(self) -> None:
        """
        Click "show more" so truncated text becomes extractable.

        These are content-expansion controls, required for ordinary
        collection. Nothing else is clicked.
        """

        self._require_page()

        try:
            buttons = self._page.locator(
                self.selectors.expand
            ).all()
        except Exception:  # noqa: BLE001
            return

        for button in buttons[:20]:
            try:
                if button.is_visible():
                    button.click(timeout=2_000)
            except Exception:  # noqa: BLE001
                # A single stubborn expander must not end the run.
                continue

    def _scroll_once(self) -> None:
        """
        Scroll and wait for an observable change.

        Waits on page height rather than a fixed sleep, so a slow feed
        is handled without guessing at timings.
        """

        page = self._require_page()

        before = self._page_height()

        self._expand_truncated_posts()

        try:
            page.mouse.wheel(0, 4000)
        except Exception:  # noqa: BLE001
            return

        # Bounded wait for the height to change, which is the signal
        # that content actually loaded.
        try:
            page.wait_for_function(
                "previous => document.body.scrollHeight > previous",
                arg=before,
                timeout=SCROLL_SETTLE_TIMEOUT_MS,
            )
        except Exception:  # noqa: BLE001
            # No change within the bound. The caller's idle-round check
            # decides whether that means the end of the feed.
            pass

    def _page_height(self) -> int:
        self._require_page()

        try:
            return int(
                self._page.evaluate(
                    "() => document.body.scrollHeight"
                )
                or 0
            )
        except Exception:  # noqa: BLE001
            return 0

    def _extract_current(self) -> list[CollectedPost]:
        """Extract every post currently rendered."""

        self._require_page()

        try:
            nodes = self._page.locator(
                self.selectors.feed
            ).all()
        except Exception as exc:  # noqa: BLE001
            raise CollectionStopped(
                StopReason.LAYOUT_CHANGED,
                f"Could not read posts from the page: {exc}",
            ) from exc

        extracted: list[CollectedPost] = []

        for node in nodes:
            collected = self._extract_one(node)

            if collected is not None:
                extracted.append(collected)

        return extracted

    def _extract_one(self, node: Any) -> CollectedPost | None:
        """Extract one post node, or None when it has no stable ID."""

        identifier = self._identifier(node)

        if not identifier:
            return None

        text = self._first_text(node, self.selectors.text)
        url = self._first_attribute(node, self.selectors.permalink, "href")
        published = self._first_text(node, self.selectors.timestamp)
        media = self._media_urls(node)

        return CollectedPost(
            source_post_id=identifier,
            text=text,
            url=url,
            published_at=published,
            author=self.profile.strip("/").split("/")[-1] or None,
            media=media,
            extra={"collected_at": _now()},
        )

    def _identifier(self, node: Any) -> str:
        """
        Derive a stable ID for a post.

        LinkedIn's activity URN is stable, so it is preferred. When a
        layout change hides it, the permalink carries the same numeric
        identifier. If neither is available the post is skipped rather
        than given a fabricated ID, because an unstable ID would
        create a duplicate on every run.
        """

        raw = None

        try:
            raw = node.get_attribute("data-id") or node.get_attribute(
                "data-urn"
            )
        except Exception:  # noqa: BLE001
            raw = None

        if raw:
            match = POST_ID_PATTERN.search(raw)

            if match:
                return match.group(0)

        href = self._first_attribute(
            node, self.selectors.permalink, "href"
        )

        if href:
            match = POST_ID_PATTERN.search(href)

            if match:
                return match.group(0)

        return ""

    def _first_text(self, node: Any, selector: str) -> str:
        try:
            located = node.locator(selector)
            count = located.count()

            if count:
                value = located.first.inner_text(timeout=2_000)

                if value and value.strip():
                    return value.strip()
        except Exception:  # noqa: BLE001
            return ""

        try:
            value = node.inner_text(timeout=2_000)

            return value.strip() if value else ""
        except Exception:  # noqa: BLE001
            return ""

    def _first_attribute(
        self,
        node: Any,
        selector: str,
        attribute: str,
    ) -> str | None:
        try:
            located = node.locator(selector)
            count = located.count()

            if count:
                value = located.first.get_attribute(attribute)

                if value and value.strip():
                    return value.strip()
        except Exception:  # noqa: BLE001
            return None

        return None

    def _media_urls(self, node: Any) -> list[str]:
        """
        Media URLs referenced by a post.

        Collected as URLs rather than downloaded inline, so a fetch
        failure can never corrupt a post that was already extracted.
        The collector downloads them separately, under its own limits.
        """

        try:
            images = node.locator(self.selectors.media).all()
        except Exception:  # noqa: BLE001
            return []

        found: list[str] = []

        for image in images[:MAX_MEDIA_PER_POST]:
            for attribute in ("data-delayed-url", "src"):
                try:
                    url = image.get_attribute(attribute)
                except Exception:  # noqa: BLE001
                    url = None

                if url and url.strip().startswith("http"):
                    found.append(url.strip())
                    break

        # Preserve order while removing repeats.
        return list(dict.fromkeys(found))

    def _within_window(
        self,
        published: str | None,
        limits: LinkedInLimits,
    ) -> bool:
        if not published:
            return True

        if limits.since is None and limits.until is None:
            return True

        moment = _parse_relative(published)

        if moment is None:
            return True

        lower = _parse_day(limits.since) if limits.since else None
        upper = _parse_day(limits.until) if limits.until else None

        if lower and moment < lower:
            return False

        if upper and moment > upper:
            return False

        return True

    def resume(self, seen: set[str]) -> None:
        """Skip posts already persisted in an earlier run."""

        self._seen = set(seen)


def _redact(error: object, limit: int = 400) -> str:
    """
    Summarize an error without anything sensitive in it.

    Playwright's messages embed the argument that was passed, so a
    failed ``fill`` would otherwise print the credential. Every
    configured credential is removed, and the result is bounded so a
    verbose driver trace cannot flood the log either.
    """

    text = str(error)

    for name in credential_module.CREDENTIAL_ENVIRONMENT_VARIABLES:
        import os

        value = os.environ.get(name, "")

        if value and len(value) >= 4:
            text = text.replace(value, "[REDACTED]")

    text = " ".join(text.split())

    if len(text) <= limit:
        return text

    return text[:limit] + " [truncated]"


def _credential(name: str) -> str:
    """
    Read a credential at the moment it is needed.

    Kept in one function so the value is never held anywhere longer
    than a single call and never travels through a URL or an argument.
    """

    import os

    value = os.environ.get(name, "")

    if not value:
        credential_module.require()

        raise CollectionStopped(
            StopReason.FAILED,
            f"{name} is not set.",
        )

    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _media_filename(url: str, index: int) -> str:
    """A stable local file name for a media URL."""

    match = re.search(
        r"/([^/]+\.(?:png|jpe?g|gif|webp|pdf))", url, re.IGNORECASE
    )

    if match:
        candidate = re.sub(r"[^A-Za-z0-9._-]", "_", match.group(1))

        if candidate.lower().endswith(".pdf"):
            return candidate

        return candidate or f"image_{index}.png"

    return f"image_{index}.png"


_RELATIVE_PATTERN = re.compile(
    r"(\d+)\s*(second|minute|hour|day|week|month|year)s?\s+ago",
    re.IGNORECASE,
)


def _parse_relative(value: str) -> datetime | None:
    """
    Parse a relative timestamp such as ``3 days ago``.

    LinkedIn renders relative time in the feed. An absolute ISO
    timestamp is also accepted. Anything else returns None so the post
    is kept rather than dropped.
    """

    text = value.strip()

    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        pass

    match = _RELATIVE_PATTERN.search(text)

    if not match:
        return None

    amount = int(match.group(1))
    unit = match.group(2).lower()

    from datetime import timedelta

    deltas = {
        "second": timedelta(seconds=amount),
        "minute": timedelta(minutes=amount),
        "hour": timedelta(hours=amount),
        "day": timedelta(days=amount),
        "week": timedelta(weeks=amount),
        "month": timedelta(days=30 * amount),
        "year": timedelta(days=365 * amount),
    }

    delta = deltas.get(unit)

    if delta is None:
        return None

    return datetime.now(timezone.utc).replace(
        tzinfo=None
    ) - delta


def _parse_day(value: str | None) -> datetime | None:
    if not value:
        return None

    text = value.strip()

    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass

    try:
        return datetime.strptime(text[:10], "%Y-%m-%d")
    except ValueError:
        return None


def browser_available() -> str:
    """Human-readable availability, safe to print."""

    if not linkedin_installed():
        return "Playwright not installed"

    if not shutil.which("python"):
        return "python not on PATH"

    return "ready"