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

import hashlib
import re
import shutil
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from src.ingestion import credentials as credential_module
from src.ingestion.auth_state import (
    AuthJournal,
    AuthObservation,
    AuthState,
    Deadline,
    classify,
    current_url as auth_current_url,
    probe as auth_probe,
    settle,
)
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

# Controls that may be clicked. Anything else is left alone.
EXPAND_SELECTORS = (
    'button[aria-label*="Show more"]',
    'button[aria-label*="see more"]',
    'button[class*="feed-shared-inline-show-more"]',
)

SCROLL_SETTLE_TIMEOUT_MS = 1500

#: Pixels advanced per scroll. A large step reaches the end of a
#: feed in fewer round trips, and a short container makes a small
#: step waste iterations on content already rendered.
SCROLL_STEP_PIXELS = 2400
DEFAULT_SCROLL_LIMIT = 400
DEFAULT_IDLE_ROUNDS = 3

# Bounded so a single post cannot pull an unbounded number of images.
MAX_MEDIA_PER_POST = 4

# Authentication budgets, all explicit and finite.
#
# The previous implementation inherited a 120s context timeout and
# polled in a Python loop, so one slow call multiplied the total wait
# and the process appeared to hang. Nothing here is unbounded.
AUTH_DEFAULT_TIMEOUT_MS = 10_000
AUTH_NAVIGATE_TIMEOUT_MS = 30_000
AUTH_PROBE_TIMEOUT_MS = 8_000
AUTH_FIELD_TIMEOUT_MS = 8_000
AUTH_SETTLE_TIMEOUT_MS = 25_000
AUTH_TOTAL_TIMEOUT_SECONDS = 90.0

# How long the human may take before verification is attempted anyway.
AUTH_HUMAN_WAIT_SECONDS = 900

#: Profile resolution gets its own budget rather than borrowing the
#: sign-in one, because a slow first paint of the feed is expected
#: and must not eat the time available to detect a challenge.
#: What to say when no profile can be resolved.
#:
#: The overwhelmingly common cause is a saved session that is no longer
#: signed in, so the message says that and names the recovery, rather
#: than reporting a missing --profile that would send the reader
#: looking in the wrong place.
_UNRESOLVED_PROFILE_MESSAGE = (
    "The signed-in profile could not be resolved. The saved session is "
    "most likely no longer signed in. Run with --login to sign in again "
    "and save a fresh session."
)

PROFILE_RESOLVE_TIMEOUT_SECONDS = 45.0
PROFILE_RESOLVE_SETTLE_MS = 6_000


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

    # How long to keep looking for the signed-in profile. A limit
    # rather than a constant, so a caller that is waiting on a human
    # can shorten it and a test does not have to pay for it.
    resolve_timeout_seconds: float = (
        PROFILE_RESOLVE_TIMEOUT_SECONDS
    )


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

# Submit selectors.
#
# Two findings from the live page shaped this list. LinkedIn's login is
# a React application with no <form> element, so pressing Enter is not
# a fallback. And `:has-text()` is a substring match, so "Sign in with
# Microsoft" and "Sign in with Apple" both match "Sign in" and the
# first would win, opening an OAuth popup instead of submitting the
# credentials. Exact matching is therefore mandatory, and
# `_resolve_submit` resolves the button inside the page where an exact
# comparison is reliable, because Playwright's `:text-is()` engine
# returned no match for a button whose textContent was exactly
# "Sign in".
SUBMIT_SELECTORS = (
    'button[type="submit"]',
    'input[type="submit"]',
    'button[data-oc-submit="1"]',
)

#: Attribute used to mark the resolved submit button.
SUBMIT_MARKER = "data-oc-submit"

#: Labels accepted as the credentials submit, compared exactly after
#: whitespace is collapsed. A federated provider button is never in
#: this set, so it can never be selected.
SUBMIT_LABELS = ("sign in", "sign in with password", "log in")

# Resolves the signed-in profile from the links the page renders.
#
# `/in/me/` does not redirect once authenticated, so the handle has to
# come from a real profile link. Only a bare `/in/<handle>/` counts:
# the navigation also renders `/in/<handle>/edit/...` and
# `/in/<handle>/overlay/...`, which are settings routes, not the
# profile.
PROFILE_RESOLVER_JS = r"""
() => {
  const anchors = Array.from(
    document.querySelectorAll('a[href*="/in/"]')
  );

  for (const anchor of anchors) {
    const href = anchor.getAttribute('href') || '';

    const match = href.match(
      /^https?:\/\/(?:[a-z]{2,3}\.)?linkedin\.com\/in\/([A-Za-z0-9._%-]+)\/?$/
    );

    if (!match) continue;

    const handle = match[1];

    if (!handle || handle === 'me') continue;

    return handle;
  }

  return '';
}
"""

# Extraction runs inside the page and returns structured data.
#
# LinkedIn's activity feed is a React application whose class names are
# hashed per build, so a selector chain against those classes matches
# nothing. The only stable handles are the permalink URNs, which is
# also the best identifier: a permalink never changes for a post.
#
# One bounded call returns every field, so extraction costs a single
# round trip instead of a locator call per field per post.
EXTRACT_JS = r"""
(options) => {
  const permalinkSelector = options.permalink;
  const profileBase = options.profileBase;
  const chrome = new Set(options.chrome || []);

  const visible = (el) => !!el && (
    el.offsetWidth > 0 || el.offsetHeight > 0 ||
    el.getClientRects().length > 0
  );

  // How many different posts an element holds. Anchors, not URNs: a
  // single post links to itself several times.
  const distinct_permalinks = (root) => {
    const urns = new Set();

    for (const link of root.querySelectorAll(permalinkSelector)) {
      const href = link.getAttribute('href') || '';
      const urn = (href.match(/urn:li:[A-Za-z]+:\d+/) || [])[0];

      if (urn) urns.add(urn);
    }

    return urns.size;
  };
  const clean = (value) => (value || '').replace(/ /g, ' ').trim();

  // "Jan 15, 2025", "15 Jan 2025", "Jan 15, 2025 - Edited",
  // "Jan 15, 2025 at 4:15 PM", and relative forms like "2 days ago".
  // Month-first comes first because that is how LinkedIn renders it;
  // the day-first and year-first orders are kept for older caches and
  // other locales.
  const MONTHS = 'jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec';

  const DATE_LINE = new RegExp(
    '^(?:' +
      '(?:mon|tue|wed|thu|fri|sat|sun)[a-z]*,?\\s+\\d{1,2}(?:,?\\s+\\d{4})?' +
      '|' + '(?:' + MONTHS + ')[a-z]*\\.?\\s+\\d{1,2}(?:,?\\s+\\d{4})?' +
      '|' + '\\d{1,2}\\s+(?:' + MONTHS + ')[a-z]*\\.?(?:,?\\s+\\d{4})?' +
      '|' + '\\d{4}\\s+(?:' + MONTHS + ')[a-z]*\\s+\\d{1,2}' +
    ')' +
    '(?:\\s*(\\u2022|\\u00b7)?\\s*(?:edited|at\\s+\\d{1,2}(:\\d{2})?\\s*(am|pm)?))?$' +
    '|' +
    '^(?:\\d+\\s+)?(?:second|minute|hour|day|week|month|year)s?\\s+ago$',
    'i'
  );

  // " - Edited", " \u2022 Edited", " at 4:15 PM", " \u00b7 Edited".
  const EDITION_SUFFIX =
    /\s*(?:\u2022|\u00b7|-)?\s*(?:edited|at\s+\d{1,2}(:\d{2})?\s*(?:am|pm)?)\s*$/i;

  const COUNTER_LINE =
    /^\d+(?:[.,]\d+)?\s*[km]?\s*(reaction|comment|repost)s?$/i;

  // Interface text, not post content.
  // `lineAbove` is the line that precedes this one, because the rule
  // below depends on position rather than on content alone.
  const isChrome = (line, lineAbove) => {
    const text = clean(line).toLowerCase();

    if (!text) return true;
    if (chrome.has(text)) return true;
    if (COUNTER_LINE.test(text)) return true;
    if (/^\u00b7?\s*(you|following|public|private)?$/.test(text)) {
      return true;
    }
    if (/^\d+\/\d+$/.test(text)) return true;

    // A bare figure directly below a counter is that same tally
    // rendered a second time in the accessible text. Only that
    // position counts, so a post whose body is genuinely "15" keeps
    // it.
    if (/^\d{1,3}$/.test(text)) {
      const above = clean(lineAbove || '').toLowerCase();
      if (COUNTER_LINE.test(above)) return true;
    }

    if (text.endsWith(' more') && text.length < 24) return true;

    return false;
  };

  const anchors = Array.from(
    document.querySelectorAll(permalinkSelector)
  );

  const seen = new Set();
  const results = [];

  for (const anchor of anchors) {
    const href = anchor.getAttribute('href') || '';
    const urn = (href.match(/urn:li:[A-Za-z]+:\d+/) || [])[0];

    // A post with no permalink has no stable identity, so it is
    // skipped rather than given a fabricated ID that would duplicate
    // on every run.
    if (!urn || seen.has(urn)) continue;

    // Climb to the post container: the nearest ancestor that holds
    // this post and no other.
    //
    // Distinct permalinks are counted, not anchors. A post renders its
    // permalink more than once (a timestamp and an "ago" label both
    // link to it), so counting anchors rejects the very posts that
    // duplicate their own link, and they were being dropped silently.
    let element = anchor;
    let container = null;

    for (let depth = 0; depth < 12; depth += 1) {
      element = element.parentElement;
      if (!element) break;

      const inner = distinct_permalinks(element);
      const text = clean(element.innerText);

      if (inner === 1 && text.length > 20) {
        container = element;
        break;
      }
    }

    if (!container) continue;

    seen.add(urn);

    const lines = clean(container.innerText)
      .split('\n')
      .map((line) => clean(line))
      .filter((line) => line.length > 0);

    // The header ends at the date line; the body follows it.
    let headerEnd = -1;

    for (let index = 0; index < lines.length; index += 1) {
      if (DATE_LINE.test(lines[index])) {
        headerEnd = index;
        break;
      }
    }

    // The rendered date line can carry an annotation, as in
    // "Jan 15, 2025 - Edited" or "Jan 15, 2025 at 4:15 PM". Only the
    // date belongs in the timestamp, so the annotation is dropped.
    let published = '';

    if (headerEnd >= 0) {
      published = clean(lines[headerEnd])
        .replace(EDITION_SUFFIX, '')
        .trim();
    }

    let body = headerEnd >= 0 ? lines.slice(headerEnd + 1) : lines;

    // Strip trailing interface chrome. `pop()` is never assigned back:
    // it returns the removed element, which would turn the array into
    // a string on the next iteration.
    while (body.length > 0) {
      const last = body[body.length - 1];
      const above = body.length >= 2 ? body[body.length - 2] : '';

      if (!isChrome(last, above)) break;

      body.pop();
    }

    const text = body.join('\n').trim();

    // The author is the profile link inside the container.
    let author = '';

    const authorLink = container.querySelector(
      'a[href*="/in/"]'
    );

    if (authorLink) {
      const href = authorLink.getAttribute('href') || '';
      const match = href.match(/\/in\/([A-Za-z0-9._%-]+)/);

      if (match) author = match[1];
    }

    // Post media only. A post renders the author's avatar and the
    // actor headline image inside its container, and both are served
    // from media.licdn.com, so the path identifies what an image
    // actually is rather than trusting its host.
    const AVATAR = [
      'profile-displayphoto',
      'profile-displ',
      'profile_photo',
      'faces/',
      'person_'
    ];

    const media = Array.from(container.querySelectorAll('img'))
      .map((img) => {
        const src =
          img.getAttribute('data-delayed-url') ||
          img.getAttribute('src') ||
          '';
        return src.split('?')[0];
      })
      .filter((src) => {
        if (src.indexOf('media.licdn.com') === -1) return false;

        const lowered = src.toLowerCase();

        for (const marker of AVATAR) {
          if (lowered.indexOf(marker) !== -1) return false;
        }

        // Tracking pixels and placeholders are not post media.
        if (lowered.indexOf('/dms/image/') === -1) return false;

        return true;
      })
      .filter((src, index, all) => all.indexOf(src) === index)
      .slice(0, 4);

    results.push({
      urn: urn,
      url: href.startsWith('http') ? href : profileBase + href,
      text: text,
      published: published,
      author: author,
      media: media
    });
  }

  return results;
}
"""

# Interface text that surrounds a post but is not part of it.
CHROME_LINES = (
    "feed post",
    "like",
    "comment",
    "repost",
    "send",
    "follow",
    "show all",
    "show more",
    "… more",
    "... more",
    "more",
    "view all",
    "see all",
    "share",
    "not interested",
    "hide this post",
    "save",
    "subscribe",
)

# Resolves the credentials submit button inside the page and tags it,
# so the click targets exactly one known element instead of relying on
# a text engine that does not match this page.
SUBMIT_RESOLVER_JS = r"""
(options) => {
  const tag = options.tag;
  const labels = options.labels || [];

  const visible = (el) => !!el && (
    el.offsetWidth > 0 || el.offsetHeight > 0 ||
    el.getClientRects().length > 0
  );
  const clean = (value) => (value || '').replace(/\s+/g, ' ').trim();

  document.querySelectorAll('[' + tag + ']').forEach(
    (el) => el.removeAttribute(tag)
  );

  const wanted = new Set(labels);

  const candidates = Array.from(
    document.querySelectorAll(
      'button, input[type="submit"], [role="button"]'
    )
  );

  for (const el of candidates) {
    if (!visible(el)) continue;

    const label = clean(
      el.textContent || el.value || ''
    ).toLowerCase();

    // Exact membership only, so a provider button cannot match.
    if (!wanted.has(label)) continue;

    el.setAttribute(tag, '1');

    return label;
  }

  return '';
}
"""


PERMALINK_SELECTOR = 'a[href*="/feed/update/urn:li:"]'


def article_post_id(slug: str) -> str:
    """
    A short, stable identifier for an article.

    Article slugs run to a hundred characters, and a post identifier is
    capped well below that because it also becomes a directory name, a
    job id and a URL segment. The slug is truncated for legibility and a
    digest of the whole slug is appended, so two articles whose slugs
    share a prefix cannot collide and the identifier never changes for
    a given slug.
    """

    digest = hashlib.sha256(slug.encode("utf-8")).hexdigest()[:10]

    readable = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in slug.lower()
    ).strip("-")

    readable = re.sub(r"-{2,}", "-", readable)[:36].strip("-")

    if not readable:
        readable = "article"

    return f"urn:li:article:{readable}-{digest}"


#: Articles live under ``/pulse/<slug>`` rather than under a feed
#: permalink, so they are identified separately. The listing already
#: carries each article's title and opening text, which is the
#: authorized content; the article page itself refuses to render
#: without an in-app navigation, so nothing is invented to stand in for
#: a body that did not load.
ARTICLE_LINK_SELECTOR = 'a[href*="/pulse/"]'

#: Extracts the article cards from the listing.
#:
#: The card is the nearest ancestor whose text is substantial but still
#: smaller than the whole list, which is where a further climb would
#: swallow the next article.
EXTRACT_ARTICLES_JS = r"""
(options) => {
  const clean = (value) => (value || '').replace(/\s+/g, ' ').trim();

  // The reading time is a reading time, not content.
  const isReadingTime = (text) =>
    /^\s*\d+\s*(?:min|mins|minute|minutes)\s+read\s*$/i.test(text) ||
    /^\s*\d+\s*(?:sec|secs|hour|hours)\s+read\s*$/i.test(text);

  const anchors = Array.from(
    document.querySelectorAll(options.link)
  );

  const results = [];
  const seen = new Set();

  for (const anchor of anchors) {
    const href = anchor.getAttribute('href') || '';

    const slug = (href.match(/\/pulse\/([^/?#]+)/) || [])[1];

    if (!slug || seen.has(slug)) continue;
    seen.add(slug);

    // Everything belonging to the article lives inside its own link:
    // a paragraph holding the title, a paragraph holding the opening
    // text, and a reading time. Walking outward to find a card
    // instead swallowed the profile header and the whole site
    // footer, so the link is the boundary.
    const blocks = Array.from(anchor.querySelectorAll('p'))
      .map((el) => clean(el.innerText))
      .filter((text) => text.length > 0 && !isReadingTime(text));

    if (blocks.length === 0) continue;

    // The title is the opening block; the excerpt is the longest,
    // because a listing truncates the body but never the heading.
    const title = blocks[0].slice(0, 200);
    const excerpt = blocks.reduce(
      (longest, block) =>
        block.length > longest.length ? block : longest,
      ''
    );

    const parts = [title];

    if (excerpt && excerpt !== title) parts.push(excerpt);

    const text = parts.join('\n\n');

    // Without a body there is nothing to learn from the article.
    if (text.length < 60) continue;

    const media = Array.from(anchor.querySelectorAll('img'))
      .map((img) => {
        const src =
          img.getAttribute('data-delayed-url') ||
          img.getAttribute('src') ||
          '';
        return src.split('?')[0];
      })
      .filter((src) => {
        if (src.indexOf('media.licdn.com') === -1) return false;

        const lowered = src.toLowerCase();

        return !lowered.includes('profile-displayphoto') &&
               !lowered.includes('profile-displ') &&
               lowered.includes('/dms/image/');
      })
      .filter((src, index, all) => all.indexOf(src) === index)
      .slice(0, 4);

    results.push({
      slug: slug,
      url: href.startsWith('http') ? href : options.base + href,
      title: title,
      text: text,
      author: options.handle || '',
      media: media
    });
  }

  return results;
}
"""

# Identifies the element that actually scrolls, and reports how far
# along it is.
#
# LinkedIn renders the activity feed inside a scrolling container, not
# into the document: the document height is pinned to the viewport
# while the container holds several thousand pixels of posts. Scrolling
# the mouse or reading document.body.scrollHeight therefore does
# nothing, which is why collection stopped at three posts.
#
# The container is found structurally, by walking up from a permalink
# to the nearest element that can scroll, rather than by naming an id
# or a class, because those change with every build.
SCROLL_STATE_JS = r"""
(permalink) => {
  const anchor = document.querySelector(permalink);

  const scrolls = (el) => {
if (!el) return false;
if (el.scrollHeight <= el.clientHeight + 20) return false;
return ['auto', 'scroll'].includes(getComputedStyle(el).overflowY);
  };

  // Prefer the nearest scrolling ancestor of a post, because that is
  // the container the feed actually lives in.
  let container = null;

  let node = anchor ? anchor.parentElement : null;

  while (node) {
if (scrolls(node)) {
  container = node;
  break;
}
node = node.parentElement;
  }

  // Fall back to the largest scrolling element on the page, which
  // covers a feed that renders before a post is present.
  if (!container) {
let best = null;

for (const el of document.querySelectorAll('*')) {
  if (!scrolls(el)) continue;

  if (!best || el.scrollHeight > best.scrollHeight) {
    best = el;
  }
}

container = best;
  }

  if (!container) {
// No inner scroller, so the document itself is the scroller.
const root = document.scrollingElement || document.documentElement;

return {
  inWindow: true,
  scrollTop: window.scrollY,
  scrollHeight: root.scrollHeight,
  clientHeight: window.innerHeight,
  atBottom: true
};
  }

  const atBottom =
container.scrollTop + container.clientHeight >=
container.scrollHeight - 8;

  return {
inWindow: false,
scrollTop: container.scrollTop,
scrollHeight: container.scrollHeight,
clientHeight: container.clientHeight,
atBottom: atBottom
  };
}
"""

# Advances the container by one page and reports the new position.
# Assigning scrollTop is used rather than a synthetic wheel event
# because it works headless and cannot land on the wrong element.
SCROLL_ADVANCE_JS = r"""
(options) => {
  const step = options.step;
  const selector = options.permalink;

  const scrolls = (el) => {
if (!el) return false;
if (el.scrollHeight <= el.clientHeight + 20) return false;
return ['auto', 'scroll'].includes(getComputedStyle(el).overflowY);
  };

  const anchor = document.querySelector(selector);

  let container = null;
  let node = anchor ? anchor.parentElement : null;

  while (node) {
if (scrolls(node)) { container = node; break; }
node = node.parentElement;
  }

  if (!container) {
window.scrollBy(0, step);

return {
  moved: window.scrollY > 0,
  atBottom:
    window.innerHeight + window.scrollY >=
    document.documentElement.scrollHeight - 8
};
  }

  const before = container.scrollTop;

  container.scrollTop = before + step;

  // A container that refuses to move is the end of the feed, and the
  // caller's idle-round check decides what that means.
  const moved = container.scrollTop > before;

  const atBottom =
container.scrollTop + container.clientHeight >=
container.scrollHeight - 8;

  return { moved: moved, atBottom: atBottom };
}
"""


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
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.profile = (profile or "").strip()
        self.headed = headed
        self.root = Path(root)
        self.selectors = selectors or SelectorSet()
        self.limits = limits or LinkedInLimits()
        self.progress = progress or (lambda message: None)

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

        #: Every authentication transition, for diagnosis. In memory
        #: only, so nothing here can reach a checkpoint.
        self.journal = AuthJournal()

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    def launch(self, *, headed: bool = False) -> None:
        """
        Create the browser, context and page.

        Split out from :meth:`start` so the login path and the
        collection path share one lifecycle, and so teardown is always
        the same set of objects in the same order.
        """

        from playwright.sync_api import sync_playwright

        self._playwright = sync_playwright().start()

        try:
            self._browser = self._playwright.chromium.launch(
                headless=not headed
            )
        except Exception as exc:  # noqa: BLE001
            self.close()
            raise CollectionStopped(
                StopReason.FAILED,
                f"Could not launch the browser: {_redact(exc)}",
            ) from exc

        state_file = (
            credential_module.browser_profile_directory(self.root)
            / "state.json"
        )

        self._context = self._browser.new_context(
            storage_state=(
                str(state_file) if state_file.is_file() else None
            )
        )

        # A short default so no call can inherit an unbounded wait.
        self._context.set_default_timeout(AUTH_DEFAULT_TIMEOUT_MS)

        try:
            self._context.set_extra_http_headers(
                {"Accept-Language": "en-US,en;q=0.9"}
            )
        except Exception:  # noqa: BLE001
            pass

        self._page = self._context.new_page()

    def start(self) -> None:
        """
        Launch the browser and authenticate.

        Raises SecurityChallenge rather than attempting to get past a
        challenge.
        """

        if not linkedin_installed():
            raise CollectionStopped(
                StopReason.FAILED, playwright_install_hint()
            )

        credential_module.require()

        self.launch(headed=self.headed)

        self._ensure_authenticated()

        # Persist the session so a later run does not sign in again,
        # which is what makes a resume possible after a human has
        # completed a challenge by hand.
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
        """
        Close page, context, browser and Playwright, in order.

        Runs on success, failure, unknown state and KeyboardInterrupt,
        because every caller uses it inside a finally block. Teardown
        failures are collected rather than raised, so cleanup can never
        mask the real outcome, and are reported instead.
        """

        problems: list[str] = []

        for attribute in ("_page", "_context", "_browser", "_playwright"):
            handle = getattr(self, attribute, None)

            if handle is None:
                continue

            # Cleared first, so a failed close cannot leave a stale
            # handle behind for a later attempt to reuse.
            setattr(self, attribute, None)

            closer = getattr(handle, "close", None) or getattr(
                handle, "stop", None
            )

            if closer is None:
                continue

            try:
                closer()
            except Exception as exc:  # noqa: BLE001
                problems.append(f"{attribute}: {_redact(exc, 120)}")

        if problems:
            self.progress(
                "Cleanup reported: " + "; ".join(problems)
            )

    def __enter__(self) -> "LinkedInSource":
        return self

    def __exit__(self, *exception: object) -> None:
        self.close()

    # -----------------------------------------------------------------
    # Authentication
    # -----------------------------------------------------------------

    def _ensure_authenticated(self) -> None:
        """
        Reach an authenticated session, signing in only if required.

        Credentials come from the local environment, so the ordinary
        path is fully automatic. A challenge stops the run and asks the
        human; it is never worked around.
        """

        self.navigate(f"{PROFILE_URL}/feed/")

        observation = self.probe()

        if observation.is_usable:
            return

        if observation.needs_human:
            self.journal.record(observation)
            raise SecurityChallenge(
                observation.challenge or "security_challenge",
                observation.detail,
            )

        if observation.state is AuthState.BROWSER_UNAVAILABLE:
            raise CollectionStopped(
                StopReason.FAILED,
                "The browser became unavailable during sign-in.",
            )

        self.journal.record(observation)

        self.authenticate_automatically()

    # -----------------------------------------------------------------
    # The authentication state machine
    # -----------------------------------------------------------------

    def authenticate_automatically(self) -> AuthObservation:
        """
        Fill the configured credentials and submit the normal form.

        Nothing is typed unless a form was positively observed, and the
        whole sequence shares one wall-clock budget, so it cannot hang.
        """

        status = credential_module.status()

        if not status.configured:
            raise CollectionStopped(
                StopReason.FAILED,
                "LinkedIn credentials are not configured.",
            )

        observation = self.probe()

        if observation.is_usable:
            return self.journal.record(observation)

        if observation.needs_human:
            return self.journal.record(observation)

        if observation.state is not AuthState.LOGIN_FORM:
            # A form we cannot see is a layout change, not a reason to
            # retry blindly.
            return self.journal.record(
                observation
                if observation.state is not AuthState.UNKNOWN
                else AuthObservation(
                    state=AuthState.UNKNOWN,
                    detail=(
                        "No sign-in form was found. LinkedIn's login "
                        "layout may have changed."
                    ),
                    url=observation.url,
                )
            )

        deadline = Deadline(AUTH_TOTAL_TIMEOUT_SECONDS)

        username_field = self._first_visible_locator(
            USERNAME_SELECTORS, limit=8
        )
        password_field = self._first_visible_locator(
            PASSWORD_SELECTORS, limit=8
        )

        if username_field is None or password_field is None:
            return self.journal.record(
                AuthObservation(
                    state=AuthState.UNKNOWN,
                    detail=(
                        "The sign-in form was observed but its fields "
                        "could not be located."
                    ),
                    url=observation.url,
                )
            )

        # Values are read one at a time, immediately before use, and
        # never stored on the instance or written anywhere.
        try:
            username_field.fill(
                _credential("LINKEDIN_USERNAME"),
                timeout=deadline.slice_ms(
                    cap_ms=AUTH_FIELD_TIMEOUT_MS, default_ms=10_000
                ),
            )
            password_field.fill(
                _credential("LINKEDIN_PASSWORD"),
                timeout=deadline.slice_ms(
                    cap_ms=AUTH_FIELD_TIMEOUT_MS, default_ms=10_000
                ),
            )
        except Exception as exc:  # noqa: BLE001
            # The driver's error text embeds what was typed, so it is
            # redacted before being surfaced.
            return self.journal.record(
                AuthObservation(
                    state=AuthState.UNKNOWN,
                    detail=(
                        "Could not fill the sign-in form: "
                        f"{_redact(exc)}"
                    ),
                    url=self.page_url(),
                )
            )

        submit = self._resolve_submit(
            deadline=deadline
        )

        if submit is None:
            return self.journal.record(
                AuthObservation(
                    state=AuthState.UNKNOWN,
                    detail=(
                        "The credentials submit button could not be "
                        "located. LinkedIn's login layout may have "
                        "changed."
                    ),
                    url=self.page_url(),
                )
            )

        try:
            submit.click(
                timeout=deadline.slice_ms(
                    cap_ms=AUTH_FIELD_TIMEOUT_MS, default_ms=10_000
                )
            )
        except Exception as exc:  # noqa: BLE001
            return self.journal.record(
                AuthObservation(
                    state=AuthState.UNKNOWN,
                    detail=(
                        f"Could not submit the sign-in form: "
                        f"{_redact(exc)}"
                    ),
                    url=self.page_url(),
                )
            )

        self.journal.record(
            AuthObservation(
                state=AuthState.LOGIN_SUBMITTED,
                detail="The sign-in form was submitted.",
                url=self.page_url(),
            )
        )

        return self.wait_for_outcome(deadline)

    def wait_for_outcome(
        self,
        deadline: Deadline,
        *,
        last_action: str = "submit",
    ) -> AuthObservation:
        """
        Wait for the submitted form to resolve. Bounded.

        One bounded driver wait followed by bounded probes, rather than
        a Python loop of locator calls.
        """

        remaining_ms = int(deadline.remaining * 1000)

        if remaining_ms > 0:
            settle(
                self._require_page(),
                timeout_ms=min(remaining_ms, AUTH_SETTLE_TIMEOUT_MS),
            )

        observation = self.probe(last_action=last_action)

        if observation.state is AuthState.UNKNOWN:
            # One more bounded pass, so a page that was still
            # rendering gets a chance to be classified. It cannot
            # extend the budget.
            if not deadline.expired:
                settle(
                    self._require_page(),
                    timeout_ms=deadline.slice_ms(
                        cap_ms=AUTH_SETTLE_TIMEOUT_MS, default_ms=5_000
                    ),
                )

                observation = self.probe(last_action=last_action)

        return self.journal.record(observation)

    def probe(
        self,
        *,
        last_action: str = "",
    ) -> AuthObservation:
        """
        Observe the page once and classify it.

        A single bounded evaluation. Never raises, so a wedged page
        becomes ``UNKNOWN`` instead of hanging the run.
        """

        page = self._page

        if page is None:
            return AuthObservation(
                state=AuthState.BROWSER_UNAVAILABLE,
                detail="The browser is not running.",
            )

        result = auth_probe(
            page, timeout_ms=AUTH_PROBE_TIMEOUT_MS
        )

        if result is None:
            return AuthObservation(
                state=AuthState.UNKNOWN,
                detail="The page could not be inspected.",
                url=self.page_url(),
            )

        return classify(
            result,
            url=self.page_url(),
            last_action=last_action,
        )

    def navigate(self, url: str) -> bool:
        """Navigate with an explicit, bounded timeout."""

        page = self._require_page()

        try:
            page.set_default_timeout(AUTH_NAVIGATE_TIMEOUT_MS)
            page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=AUTH_NAVIGATE_TIMEOUT_MS,
            )
        except Exception as exc:  # noqa: BLE001
            self.progress(f"Could not open {url}: {_redact(exc)}")
            return False

        return True

    def page_url(self) -> str:
        """The current URL, or empty when the page has gone."""

        return auth_current_url(self._page) if self._page else ""

    def _signed_in(self) -> bool:
        """
        Whether the session is authenticated.

        One bounded probe, not a chain of locator calls. This is what
        the previous implementation polled, and polling it was what
        appeared to hang.
        """

        return self.probe().state.is_terminal_success

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

    def _resolve_submit(
        self,
        *,
        deadline: Deadline,
    ):
        """
        Find and return the credentials submit button.

        Resolution happens inside the page with an exact label
        comparison, because Playwright's text engine matched nothing on
        this page even though the button's text was exactly
        "Sign in". An exact comparison also guarantees a federated
        provider button, whose label merely contains "Sign in", is
        never chosen.
        """

        page = self._require_page()

        budget = deadline.slice_ms(
            cap_ms=AUTH_PROBE_TIMEOUT_MS, default_ms=8_000
        )

        try:
            page.set_default_timeout(budget)

            matched = page.evaluate(
                SUBMIT_RESOLVER_JS,
                {
                    "tag": SUBMIT_MARKER,
                    "labels": list(SUBMIT_LABELS),
                },
            )
        except Exception:  # noqa: BLE001
            return None

        if not matched:
            return None

        try:
            return page.locator(
                f'[{SUBMIT_MARKER}="1"]'
            ).first
        except Exception:  # noqa: BLE001
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
        Launch a headed browser and sign in with the local credentials.

        Credentials are filled automatically. The browser is left open
        so that if LinkedIn presents a challenge, the human can complete
        it by hand. Nothing about a challenge is solved here.
        """

        if not linkedin_installed():
            raise CollectionStopped(
                StopReason.FAILED, playwright_install_hint()
            )

        self.launch(headed=True)

        page = self._require_page()

        self.navigate(f"{PROFILE_URL}/login")

        observation = self.probe()

        if observation.is_usable:
            return observation

        if observation.state is AuthState.LOGIN_FORM:
            observation = self.authenticate_automatically()

        return observation

    def wait_for_manual_session(
        self,
        *,
        timeout_seconds: int = 900,
        interval_seconds: int = 5,
    ) -> AuthObservation:
        """
        Wait for the human to complete any challenge, then verify once.

        The wait itself may be long, because a person is involved. The
        *verification* after they return is bounded, so pressing Enter
        can never lead to an unbounded browser wait.

        Reports UNKNOWN rather than hanging if the browser disappears
        while the user is working.
        """

        deadline = Deadline(timeout_seconds)

        while not deadline.expired:
            observation = self.probe()

            if observation.is_usable:
                return observation

            # The browser or page closed: nothing can be verified, so
            # report it rather than waiting out the budget.
            if observation.state is AuthState.BROWSER_UNAVAILABLE:
                return observation

            try:
                self._require_page().wait_for_timeout(
                    interval_seconds * 1000
                )
            except Exception:  # noqa: BLE001
                return self.probe()

        return self.probe()

    def _require_page(self):
        """The page, or a clear error if the browser never started."""

        if self._page is None:
            raise CollectionStopped(
                StopReason.FAILED,
                "The browser is not running. Call start() first.",
            )

        return self._page

    # -----------------------------------------------------------------
    # Challenge detection
    # -----------------------------------------------------------------

    def _assert_no_challenge(self, stage: str) -> None:
        """
        Stop if the page shows a security challenge.

        Uses the same bounded probe as the state machine, so it cannot
        hang and cannot disagree with what the machine concluded.
        """

        observation = self.probe()

        if observation.needs_human:
            kind = observation.challenge.replace("_", " ")

            raise SecurityChallenge(
                observation.challenge or "security_challenge",
                f"LinkedIn presented a {kind} challenge during "
                f"{stage}. Complete it in the open browser window, "
                f"then resume. {observation.detail}",
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

            # The container reporting its end is a stronger signal than
            # an idle round, because a feed can hold thousands of
            # pixels of already-loaded content that yields nothing new.
            # Checked after the idle rounds so a transient stall at the
            # end of a page is still retried.
            if idle_rounds and self._feed_exhausted():
                raise CollectionStopped(
                    StopReason.EXHAUSTED,
                    "Reached the end of the available activity.",
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

    def _go_to_articles(self) -> None:
        """
        Open the authored-articles tab of the same profile.

        Articles are the profile's own long-form content and are part
        of the same authorized source as the posts, but they are a
        different content kind: they carry a slug rather than a feed
        permalink, so they are collected and identified separately.
        """

        slug = self._resolve_slug()

        if not slug:
            raise CollectionStopped(
                StopReason.FAILED,
                _UNRESOLVED_PROFILE_MESSAGE,
            )

        self.navigate(
            f"{PROFILE_URL}/in/{slug}/recent-activity/articles/"
        )

        self._assert_no_challenge("article listing")

    def _extract_articles(self) -> list[CollectedPost]:
        """
        Extract every article currently rendered in the listing.

        One bounded in-page call, keyed on the article slug. A slug is
        stable and unique, so an article keeps the same identifier
        across runs and cannot be collected twice.
        """

        page = self._require_page()

        try:
            page.set_default_timeout(AUTH_PROBE_TIMEOUT_MS)

            payload = page.evaluate(
                EXTRACT_ARTICLES_JS,
                {
                    "link": ARTICLE_LINK_SELECTOR,
                    "base": PROFILE_URL,
                    "handle": self.resolved_profile,
                },
            )
        except Exception as exc:  # noqa: BLE001
            raise CollectionStopped(
                StopReason.LAYOUT_CHANGED,
                f"Could not read articles from the page: {_redact(exc)}",
            ) from exc

        if not isinstance(payload, list):
            raise CollectionStopped(
                StopReason.LAYOUT_CHANGED,
                "The article listing returned an unexpected result.",
            )

        extracted: list[CollectedPost] = []

        for entry in payload:
            if not isinstance(entry, dict):
                continue

            slug = str(entry.get("slug") or "").strip()

            if not slug:
                # Without a slug there is no stable identity, so a
                # generated one would duplicate on every run.
                continue

            text = str(entry.get("text") or "").strip()

            if not text:
                continue

            title = str(entry.get("title") or "").strip()

            # The title is already part of the card text, so it is not
            # prepended again. It is carried separately for the wiki to
            # use as a heading.
            extracted.append(
                CollectedPost(
                    source_post_id=article_post_id(slug),
                    text=text,
                    url=str(entry.get("url") or "").strip() or None,
                    # The listing renders no publication date, so
                    # none is claimed rather than one being inferred
                    # from the article slug.
                    published_at=None,
                    author=(
                        str(entry.get("author") or "").strip() or None
                    ),
                    media_urls=[
                        str(item) for item in (entry.get("media") or [])
                    ],
                    extra={
                        "collected_at": _now(),
                        "kind": "article",
                        "title": title or None,
                    },
                )
            )

        return extracted

    def discover_articles(
        self, **limits: object
    ) -> Iterator[CollectedPost]:
        """
        Yield the profile's authored articles.

        Bounded by the same limit and idle rules as the post walk, so a
        long article list cannot produce an unbounded run.
        """

        if self._page is None:
            self.start()

        self._require_page()

        effective = self.limits

        if limits.get("max_posts") is not None:
            effective = replace(
                effective, max_posts=int(limits["max_posts"])
            )

        self._go_to_articles()

        produced = 0
        idle_rounds = 0
        scrolls = 0
        seen: set[str] = set(self._seen)

        while True:
            self._assert_no_challenge("article collection")

            new_this_round = 0

            for collected in self._extract_articles():
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
                        f"{effective.max_posts} articles.",
                    )

                produced += 1

                yield collected

            idle_rounds = 0 if new_this_round else idle_rounds + 1

            if (
                effective.max_posts is not None
                and produced >= effective.max_posts
            ):
                raise CollectionStopped(
                    StopReason.MAX_POSTS,
                    f"Reached the configured limit of "
                    f"{effective.max_posts} articles.",
                )

            if idle_rounds >= effective.idle_rounds:
                raise CollectionStopped(
                    StopReason.NO_NEW_CONTENT,
                    f"No new articles after {idle_rounds} passes with "
                    f"no change.",
                )

            if idle_rounds and self._feed_exhausted():
                raise CollectionStopped(
                    StopReason.EXHAUSTED,
                    "Reached the end of the available articles.",
                )

            if scrolls >= effective.scroll_limit:
                raise CollectionStopped(
                    StopReason.SCROLL_LIMIT,
                    f"Reached the configured scroll limit of "
                    f"{effective.scroll_limit}.",
                )

            scrolls += 1
            self._scroll_once()

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

        # Bounded, and through the same helper as the sign-in
        # navigation, so no navigation can block indefinitely.
        self.navigate(
            f"{PROFILE_URL}/in/{slug}/recent-activity/all/"
        )

        self._assert_no_challenge("navigation")

    def _resolve_slug(self) -> str:
        """
        Determine which profile to read.

        A configured handle wins. Otherwise the session's own profile
        is resolved from the links the page renders, so the collector
        only ever reads the account the user actually authenticated as
        rather than someone else's.

        ``/in/me/`` is not used: it does not redirect once
        authenticated, so there is nothing to follow. The handle comes
        from a rendered link instead, which is not on the page
        immediately, so the read is retried under a bound rather than
        trusted on the first attempt.
        """

        configured = self.profile.strip().strip("/")

        if configured:
            if configured.startswith("in/"):
                configured = configured[3:]

            return configured

        # A handle resolved earlier in this run is still the right
        # answer, and reusing it avoids a second round trip.
        if self.resolved_profile:
            return self.resolved_profile

        page = self._require_page()

        budget = Deadline(self.limits.resolve_timeout_seconds)

        while not budget.expired:
            if not self.navigate(f"{PROFILE_URL}/feed/"):
                break

            handle = ""

            try:
                page.set_default_timeout(AUTH_PROBE_TIMEOUT_MS)

                handle = str(
                    page.evaluate(PROFILE_RESOLVER_JS) or ""
                ).strip()
            except Exception:  # noqa: BLE001
                handle = ""

            if handle:
                self.resolved_profile = handle

                return handle

            # The navigation shell renders before the profile link
            # does, so one bounded settle before asking again.
            settle(
                page,
                timeout_ms=budget.slice_ms(
                    cap_ms=PROFILE_RESOLVE_SETTLE_MS, default_ms=5_000
                ),
            )

        return ""

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

    def _scroll_state(self) -> dict:
        """
        Where the feed is scrolled to, and whether it is exhausted.

        Reads the container rather than the document, because LinkedIn
        renders the feed into an inner scroller: the document height is
        pinned to the viewport, so reading it reported a page that was
        always already "the same height" and never advanced.
        """

        page = self._require_page()

        try:
            page.set_default_timeout(AUTH_PROBE_TIMEOUT_MS)

            state = page.evaluate(
                SCROLL_STATE_JS, PERMALINK_SELECTOR
            )
        except Exception:  # noqa: BLE001
            return {}

        return state if isinstance(state, dict) else {}

    def _scroll_once(self) -> None:
        """
        Advance the feed and wait for it to load more.

        The container is scrolled by assigning its scroll position
        rather than by synthesising a wheel event, because an
        assignment is deterministic, works headless, and cannot land on
        the wrong element when the pointer position is unknown.

        Waits on the container's height rather than sleeping, so a slow
        feed is handled without guessing at timings.
        """

        page = self._require_page()

        before = self._scroll_state()

        self._expand_truncated_posts()

        try:
            page.set_default_timeout(AUTH_PROBE_TIMEOUT_MS)

            page.evaluate(
                SCROLL_ADVANCE_JS,
                {
                    "step": SCROLL_STEP_PIXELS,
                    "permalink": PERMALINK_SELECTOR,
                },
            )
        except Exception:  # noqa: BLE001
            return

        target = int(before.get("scrollHeight") or 0)

        if target <= 0:
            return

        # Bounded wait for the container to grow, which is the signal
        # that more content actually loaded.
        try:
            page.wait_for_function(
                """
                options => {
                  const scrolls = (el) => {
                    if (!el) return false;
                    if (el.scrollHeight <= el.clientHeight + 20) return false;
                    return [
                      'auto',
                      'scroll'
                    ].includes(getComputedStyle(el).overflowY);
                  };

                  const anchor = document.querySelector(options.permalink);

                  let node = anchor ? anchor.parentElement : null;

                  while (node) {
                    if (scrolls(node)) {
                      return node.scrollHeight > options.target;
                    }
                    node = node.parentElement;
                  }

                  return (
                    document.documentElement.scrollHeight >
                    options.target
                  );
                }
                """,
                arg={
                    "target": target,
                    "permalink": PERMALINK_SELECTOR,
                },
                timeout=SCROLL_SETTLE_TIMEOUT_MS,
            )
        except Exception:  # noqa: BLE001
            # No growth within the bound. The caller's idle-round check
            # decides whether that means the end of the feed.
            pass

    def _page_height(self) -> int:
        """
        How much content the feed holds.

        Read from the scroll container, falling back to the document,
        so a page that scrolls in the window is still measured.
        """

        state = self._scroll_state()

        height = state.get("scrollHeight")

        if isinstance(height, int) and height > 0:
            return height

        self._require_page()

        try:
            return int(
                self._page.evaluate(
                    "() => document.documentElement.scrollHeight"
                )
                or 0
            )
        except Exception:  # noqa: BLE001
            return 0

    def _feed_exhausted(self) -> bool:
        """Whether the feed has reached its end."""

        return bool(self._scroll_state().get("atBottom"))

    def _extract_current(self) -> list[CollectedPost]:
        """
        Extract every post currently rendered.

        One bounded in-page call. Posts with no permalink URN are
        skipped rather than given a fabricated identifier, because an
        unstable identifier would create a duplicate on every run.
        """

        page = self._require_page()

        try:
            page.set_default_timeout(AUTH_PROBE_TIMEOUT_MS)

            payload = page.evaluate(
                EXTRACT_JS,
                {
                    "permalink": PERMALINK_SELECTOR,
                    "profileBase": PROFILE_URL,
                    "chrome": list(CHROME_LINES),
                },
            )
        except Exception as exc:  # noqa: BLE001
            raise CollectionStopped(
                StopReason.LAYOUT_CHANGED,
                f"Could not read posts from the page: {_redact(exc)}",
            ) from exc

        if not isinstance(payload, list):
            raise CollectionStopped(
                StopReason.LAYOUT_CHANGED,
                "The page returned an unexpected extraction result.",
            )

        extracted: list[CollectedPost] = []

        for entry in payload:
            if not isinstance(entry, dict):
                continue

            urn = str(entry.get("urn") or "").strip()

            if not urn:
                continue

            text = str(entry.get("text") or "").strip()

            if not text:
                # A post with no extractable body carries no interview
                # knowledge, so it is skipped rather than stored as an
                # empty shell.
                continue

            extracted.append(
                CollectedPost(
                    source_post_id=urn,
                    text=text,
                    url=str(entry.get("url") or "").strip() or None,
                    published_at=(
                        str(entry.get("published") or "").strip()
                        or None
                    ),
                    author=(
                        str(entry.get("author") or "").strip() or None
                    ),
                    media_urls=[
                        str(item)
                        for item in (entry.get("media") or [])
                    ],
                    extra={"collected_at": _now()},
                )
            )

        return extracted

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