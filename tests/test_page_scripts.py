"""
Tests for the JavaScript that runs inside the page.

The LinkedIn collector does most of its reading in the page rather than
through a chain of locator calls: one evaluation returns every field of
every post, one resolves the submit button, one resolves the profile.
That code is the part most likely to break silently, because a change
to the real page shows up as zero posts rather than an exception.

So it is tested against a real DOM. A static HTML fixture stands in for
the activity feed, and the same constants the collector sends are
executed against it. These assertions are about behaviour, so a change
to LinkedIn's markup is caught here rather than on a live run.

Skipped when Chromium is not installed, because a browser test must
not be the reason the suite fails on a fresh checkout.
"""

from __future__ import annotations

import json

import pytest

from src.ingestion.sources.linkedin import (
    CHROME_LINES,
    EXTRACT_JS,
    PERMALINK_SELECTOR,
    PROFILE_RESOLVER_JS,
    PROFILE_URL,
    SCROLL_ADVANCE_JS,
    SCROLL_STATE_JS,
    SUBMIT_LABELS,
    SUBMIT_MARKER,
    SUBMIT_RESOLVER_JS,
)

pytestmark = pytest.mark.skipif(
    pytest.importorskip("playwright.sync_api", reason="no Playwright")
    is None,
    reason="Playwright is not installed",
)


@pytest.fixture(scope="module")
def browser():
    """One browser for the module. Launching is the slow part."""

    from playwright.sync_api import sync_playwright

    try:
        with sync_playwright() as engine:
            try:
                instance = engine.chromium.launch()
            except Exception as exc:  # noqa: BLE001
                pytest.skip(f"Chromium is unavailable: {exc}")

            yield instance

            instance.close()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Playwright is unavailable: {exc}")


def render(browser, html: str):
    """A page holding the given markup, with nothing navigated."""

    context = browser.new_context()
    page = context.new_page()

    page.set_content(html, wait_until="domcontentloaded")

    return context, page


def extract(page):
    return page.evaluate(
        EXTRACT_JS,
        {
            "permalink": PERMALINK_SELECTOR,
            "profileBase": PROFILE_URL,
            "chrome": list(CHROME_LINES),
        },
    )


# ---------------------------------------------------------------------
# Post extraction
# ---------------------------------------------------------------------


# The fixture uses block-level elements, because that is what the real
# activity feed renders. inline elements collapse into one another in
# innerText, so an inline fixture would test markup LinkedIn does not
# produce rather than the extractor.
FEED = """
<main>
  <div class="post">
    <div>
      <a href="https://www.linkedin.com/feed/update/urn:li:activity:1/">
        <img alt="View author profile"
             src="https://media.licdn.com/dms/image/v2/A/profile-displayphoto-shrink_100_100/x">
      </a>
    </div>
    <div><a href="https://www.linkedin.com/in/someone">Anurag Pandey</a></div>
    <div>Jan 15, 2025</div>
    <div><p>Window functions let you compute across a partition without
         collapsing rows.</p></div>
    <div>
      <img alt="View image"
           src="https://media.licdn.com/dms/image/v2/B/feedshare-shrink_480/x">
    </div>
    <div><span>15 reactions</span></div>
    <div><span>15</span></div>
    <div><span>4 comments</span></div>
    <div><span>4 comments</span></div>
    <div><button>Like</button></div>
    <div><button>Comment</button></div>
    <div><button>Repost</button></div>
    <div><button>Send</button></div>
  </div>
</main>
"""


def test_a_post_is_extracted_with_its_body(browser):
    _context, page = render(browser, FEED)

    posts = extract(page)

    assert len(posts) == 1
    assert posts[0]["urn"] == "urn:li:activity:1"


def test_the_post_body_excludes_the_header_and_the_interface(browser):
    """
    The header carries the author, the headline and the date; the
    trailer carries the reaction tally and the action buttons. None of
    it is post content, so none of it may reach the knowledge base.
    """

    _context, page = render(browser, FEED)

    body = extract(page)[0]["text"]

    assert "Window functions" in body

    # Header.
    assert "Anurag Pandey" not in body
    assert "Jan 15" not in body

    # Trailer.
    for chrome in ("reactions", "comments", "Like", "Comment",
                   "Repost", "Send"):
        assert chrome not in body, chrome


def test_the_bare_counter_after_a_tally_is_dropped(browser):
    """
    LinkedIn renders the reaction tally twice: "15 reactions" and then
    a bare "15", which is how the count reads out as accessible text.
    The duplicate is not content.
    """

    html = """
    <main><div>
      <div><a href="https://www.linkedin.com/feed/update/urn:li:activity:13/">
        permalink</a></div>
      <div>Jan 4, 2026</div>
      <div>Idempotent writes make a pipeline safe to re-run.</div>
      <div><span>15 reactions</span></div>
      <div><span>15</span></div>
    </div></main>
    """

    _context, page = render(browser, html)

    body = extract(page)[0]["text"]

    assert "Idempotent writes" in body
    assert not any(line.strip() == "15" for line in body.splitlines())
    assert "reactions" not in body


def test_a_number_that_is_itself_the_body_is_kept(browser):
    """
    The bare-figure rule is scoped to the position directly after a
    counter. A post whose body is genuinely a number keeps it.
    """

    html = """
    <main><div>
      <div><a href="https://www.linkedin.com/feed/update/urn:li:activity:2/">
        permalink</a></div>
      <div>Feb 3, 2026</div>
      <div>15</div>
      <div><button>Like</button></div>
    </div></main>
    """

    _context, page = render(browser, html)

    assert extract(page)[0]["text"].strip() == "15"


def test_the_author_is_read_from_the_profile_link(browser):
    _context, page = render(browser, FEED)

    assert extract(page)[0]["author"] == "someone"


def test_the_permalink_is_kept_as_the_source_url(browser):
    _context, page = render(browser, FEED)

    url = extract(page)[0]["url"]

    assert url.endswith("/feed/update/urn:li:activity:1/")


def test_the_published_date_is_captured(browser):
    _context, page = render(browser, FEED)

    assert extract(page)[0]["published"] == "Jan 15, 2025"


@pytest.mark.parametrize(
    "line,expected",
    [
        ("Jan 15, 2025", "Jan 15, 2025"),
        ("15 Jan 2025", "15 Jan 2025"),
        ("January 15, 2025", "January 15, 2025"),
        ("Jan 15", "Jan 15"),
        ("Jan 15, 2025 \u2022 Edited", "Jan 15, 2025"),
        ("2 days ago", "2 days ago"),
        ("3 weeks ago", "3 weeks ago"),
        ("1 year ago", "1 year ago"),
        ("5 minutes ago", "5 minutes ago"),
    ],
)
def test_the_date_header_is_recognised_in_every_rendered_form(
    browser, line, expected
):
    """
    LinkedIn renders the same date several ways depending on age and
    locale. A form the detector misses leaves the whole header in the
    body, which is how the first live run stored an author and a date
    as if they were the post.
    """

    html = f"""
    <main><div>
      <div><a href="https://www.linkedin.com/feed/update/urn:li:activity:3/">
        permalink</a></div>
      <div>Anurag Pandey</div>
      <div>{line}</div>
      <div>The actual post body about delta lakehouse compaction.</div>
    </div></main>
    """

    _context, page = render(browser, html)

    post = extract(page)[0]

    assert post["published"] == expected
    assert "delta lakehouse" in post["text"]
    assert "Anurag Pandey" not in post["text"]


def test_media_excludes_the_author_avatar(browser):
    """
    The author's avatar is served from media.licdn.com and sits inside
    the post container, so host alone does not identify post media.
    """

    _context, page = render(browser, FEED)

    media = extract(page)[0]["media"]

    assert len(media) == 1
    assert "profile-displayphoto" not in media[0]
    assert "feedshare" in media[0]


def test_media_query_strings_are_dropped(browser):
    """Query strings carry tracking tokens, which have no use here."""

    html = """
    <main><div>
      <div><a href="https://www.linkedin.com/feed/update/urn:li:activity:4/">
        permalink</a></div>
      <div>Body text long enough to be recognised as a post.</div>
      <img src="https://media.licdn.com/dms/image/v2/Z/feedshare-shrink_480/x?e=abc&amp;w=1">
    </div></main>
    """

    _context, page = render(browser, html)

    media = extract(page)[0]["media"]

    assert media == [
        "https://media.licdn.com/dms/image/v2/Z/feedshare-shrink_480/x"
    ]


def test_a_delayed_image_is_still_media(browser):
    """Images that have not loaded expose data-delayed-url instead."""

    html = """
    <main><div>
      <div><a href="https://www.linkedin.com/feed/update/urn:li:activity:5/">
        permalink</a></div>
      <div>Body text long enough to be recognised as a post.</div>
      <img data-delayed-url="https://media.licdn.com/dms/image/v2/Q/feedshare-shrink_480/y">
    </div></main>
    """

    _context, page = render(browser, html)

    assert extract(page)[0]["media"] == [
        "https://media.licdn.com/dms/image/v2/Q/feedshare-shrink_480/y"
    ]


def test_a_post_without_a_permalink_is_not_invented(browser):
    """
    A permalink is the only stable identity a post has. Without one,
    a generated identifier would change per run and duplicate the post
    on every collection.
    """

    html = """
    <main><div>
      <div>Body text long enough to be recognised as a post, with no
         permalink anywhere near it.</div>
    </div></main>
    """

    _context, page = render(browser, html)

    assert extract(page) == []


def test_a_permalink_repeated_across_posts_is_counted_once(browser):
    """
    The same URN appears in more than one container on a real feed. It
    is the same post, so it must be collected once.
    """

    html = """
    <main>
      <div>
        <div><a href="https://www.linkedin.com/feed/update/urn:li:activity:6/">a</a></div>
        <div>Jan 1, 2026</div>
        <div>First rendering of a post that also appears below.</div>
      </div>
      <div>
        <div><a href="https://www.linkedin.com/feed/update/urn:li:activity:6/">b</a></div>
        <div>Jan 1, 2026</div>
        <div>Second rendering of the very same post.</div>
      </div>
    </main>
    """

    _context, page = render(browser, html)

    posts = extract(page)

    assert len(posts) == 1
    assert posts[0]["urn"] == "urn:li:activity:6"


def test_two_posts_are_extracted_separately(browser):
    """
    Posts are siblings, not nested. Climbing to the nearest ancestor
    holding exactly one permalink is what keeps them apart.
    """

    html = """
    <main>
      <div>
        <div><a href="https://www.linkedin.com/feed/update/urn:li:activity:7/">a</a></div>
        <div>Jan 1, 2026</div>
        <div>First post body, comfortably longer than the minimum.</div>
      </div>
      <div>
        <div><a href="https://www.linkedin.com/feed/update/urn:li:activity:8/">b</a></div>
        <div>Jan 2, 2026</div>
        <div>Second post body, comfortably longer than the minimum.</div>
      </div>
    </main>
    """

    _context, page = render(browser, html)

    posts = extract(page)

    assert len(posts) == 2
    assert {p["urn"] for p in posts} == {
        "urn:li:activity:7",
        "urn:li:activity:8",
    }

    # And neither borrowed the other's text.
    assert "First post body" in posts[0]["text"]
    assert "Second post body" not in posts[0]["text"]


def test_a_post_with_no_body_is_skipped_by_the_collector(browser):
    """
    Extraction reports it, and the collector drops it. A post with no
    text carries no interview knowledge, so storing an empty shell
    would only add noise to the wiki.
    """

    from src.ingestion.sources.base import CollectedPost

    html = """
    <main><div>
      <div><a href="https://www.linkedin.com/feed/update/urn:li:activity:9/">a</a></div>
      <div>Jan 1, 2026</div>
      <div><button>Like</button></div>
      <div><button>Comment</button></div>
    </div></main>
    """

    _context, page = render(browser, html)

    entries = extract(page)

    # Nothing to keep.
    assert all(not e["text"].strip() for e in entries)

    kept = [
        CollectedPost(source_post_id=e["urn"], text=e["text"].strip())
        for e in entries
        if e["text"].strip()
    ]

    assert kept == []


def test_a_reposted_post_keeps_its_own_identity(browser):
    """
    A share and its origin are different posts. Both carry a distinct
    URN, so both are collected rather than one being merged away.
    """

    html = """
    <main>
      <div>
        <div><a href="https://www.linkedin.com/feed/update/urn:li:share:10/">a</a></div>
        <div>May 31, 2016</div>
        <div>A Simple Pick and Place Robot By The Team</div>
      </div>
      <div>
        <div><a href="https://www.linkedin.com/feed/update/urn:li:ugcPost:11/">b</a></div>
        <div>Aug 2, 2022</div>
        <div>Moments from the Infosys X UNLEASH event.</div>
      </div>
    </main>
    """

    _context, page = render(browser, html)

    urns = {p["urn"] for p in extract(page)}

    assert urns == {"urn:li:share:10", "urn:li:ugcPost:11"}


def test_multiline_bodies_keep_their_line_breaks(browser):
    """Paragraph structure carries meaning in prose."""

    html = """
    <main><div>
      <div><a href="https://www.linkedin.com/feed/update/urn:li:activity:12/">a</a></div>
      <div>Mar 1, 2026</div>
      <div>First paragraph of the post.</div>
      <div>Second paragraph of the post.</div>
    </div></main>
    """

    _context, page = render(browser, html)

    text = extract(page)[0]["text"]

    assert "First paragraph" in text
    assert "Second paragraph" in text
    assert len(text.splitlines()) >= 2


def test_an_empty_page_yields_no_posts(browser):
    """A legitimately empty feed is not an error."""

    _context, page = render(browser, "<main></main>")

    assert extract(page) == []


# ---------------------------------------------------------------------
# Submit resolution
# ---------------------------------------------------------------------


LOGIN = """
<main>
  <button type="button">Sign in with Microsoft</button>
  <button type="button">Sign in with Apple</button>
  <button type="button" aria-label="Show password"></button>
  <button type="button">Sign in</button>
</main>
"""


def resolve_submit(page):
    return page.evaluate(
        SUBMIT_RESOLVER_JS,
        {"tag": SUBMIT_MARKER, "labels": list(SUBMIT_LABELS)},
    )


def test_the_credentials_button_is_selected_not_a_provider(browser):
    """
    The bug this fixes: `:has-text("Sign in")` matches "Sign in with
    Microsoft" as well, and the provider button opens a federated popup
    instead of submitting the configured credentials.
    """

    _context, page = render(browser, LOGIN)

    assert resolve_submit(page) == "sign in"

    marked = page.locator(f'[{SUBMIT_MARKER}="1"]')

    assert marked.count() == 1
    assert marked.inner_text().strip() == "Sign in"


def test_no_provider_label_is_ever_accepted(browser):
    for provider in ("Microsoft", "Apple", "Google"):
        html = f'<main><button type="button">Sign in with {provider}</button></main>'

        _context, page = render(browser, html)

        # Nothing is tagged, so no click can land on a provider.
        assert resolve_submit(page) == ""
        assert page.locator(f'[{SUBMIT_MARKER}="1"]').count() == 0


def test_resolving_twice_does_not_leave_two_marked_buttons(browser):
    """The resolver clears its own marker before each search."""

    _context, page = render(browser, LOGIN)

    resolve_submit(page)
    resolve_submit(page)

    assert page.locator(f'[{SUBMIT_MARKER}="1"]').count() == 1


def test_a_page_with_no_submit_button_reports_nothing(browser):
    _context, page = render(browser, "<main><p>No form here.</p></main>")

    assert resolve_submit(page) == ""


def test_a_provider_only_page_is_not_a_usable_login(browser):
    """The realistic failure: only federated buttons are present."""

    html = """
    <main>
      <button type="button">Sign in with Microsoft</button>
      <button type="button">Sign in with Apple</button>
    </main>
    """

    _context, page = render(browser, html)

    assert resolve_submit(page) == ""


# ---------------------------------------------------------------------
# Profile resolution
# ---------------------------------------------------------------------


def test_the_signed_in_profile_is_read_from_a_bare_link(browser):
    """
    `/in/me/` does not redirect once authenticated, so the handle has
    to come from a rendered link.
    """

    html = """
    <nav>
      <a href="https://www.linkedin.com/in/anurag">Anurag Pandey</a>
    </nav>
    """

    _context, page = render(browser, html)

    assert page.evaluate(PROFILE_RESOLVER_JS) == "anurag"


def test_a_settings_link_is_not_mistaken_for_the_profile(browser):
    """
    The navigation also renders `/in/<handle>/edit/...` and
    `/in/<handle>/overlay/...`. Those are settings routes; resolving one
    would point the collector at a page that is not the profile.
    """

    html = """
    <nav>
      <a href="https://www.linkedin.com/in/anurag/edit/intro/">Edit</a>
      <a href="https://www.linkedin.com/in/anurag/overlay/contact-info/">Contact</a>
    </nav>
    """

    _context, page = render(browser, html)

    assert page.evaluate(PROFILE_RESOLVER_JS) == ""


def test_a_bare_link_is_preferred_over_a_settings_link(browser):
    html = """
    <nav>
      <a href="https://www.linkedin.com/in/anurag/edit/secondary-language/">x</a>
      <a href="https://www.linkedin.com/in/anurag/">Anurag Pandey</a>
    </nav>
    """

    _context, page = render(browser, html)

    assert page.evaluate(PROFILE_RESOLVER_JS) == "anurag"


def test_the_me_placeholder_is_not_returned_as_a_handle(browser):
    """Returning "me" would produce a URL that resolves to nobody."""

    html = '<nav><a href="https://www.linkedin.com/in/me/">Me</a></nav>'

    _context, page = render(browser, html)

    assert page.evaluate(PROFILE_RESOLVER_JS) == ""


def test_a_page_with_no_profile_link_reports_nothing(browser):
    _context, page = render(browser, "<main><p>Signed out.</p></main>")

    assert page.evaluate(PROFILE_RESOLVER_JS) == ""


def test_resolution_never_returns_a_credentials_bearing_string(browser):
    """The handle is public; nothing here may carry a credential."""

    html = '<nav><a href="https://www.linkedin.com/in/anurag/">x</a></nav>'

    _context, page = render(browser, html)

    handle = page.evaluate(PROFILE_RESOLVER_JS)

    assert "@" not in handle
    assert len(handle) < 64


# ---------------------------------------------------------------------
# The probe
# ---------------------------------------------------------------------


def probe_dict(**overrides):
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


def test_the_probe_sees_the_login_form_it_was_built_for(browser):
    """
    The probe and the login flow must agree about what a form looks
    like, or the machine fills credentials it cannot see.
    """

    from src.ingestion.auth_state import PAGE_PROBE_JS, classify

    html = """
    <main>
      <input autocomplete="username">
      <input autocomplete="current-password">
      <button type="button">Sign in</button>
    </main>
    """

    _context, page = render(browser, html)

    result = page.evaluate(PAGE_PROBE_JS)

    assert result["username_count"] == 1
    assert result["password_count"] == 1
    assert result["session_markers"] == 0
    assert classify(result).state.value == "login_form"


def test_the_probe_sees_a_signed_in_session(browser):
    from src.ingestion.auth_state import PAGE_PROBE_JS, classify

    html = """
    <nav>
      <img alt="Anurag Pandey's profile photo">
    </nav>
    """

    _context, page = render(browser, html)

    result = page.evaluate(PAGE_PROBE_JS)

    assert result["session_markers"] == 1
    assert classify(result).state.value == "authenticated"


def test_the_probe_surfaces_a_challenge_it_must_not_work_around(
    browser,
):
    from src.ingestion.auth_state import PAGE_PROBE_JS, classify

    html = '<main><p>Please complete the CAPTCHA to continue.</p></main>'

    _context, page = render(browser, html)

    result = page.evaluate(PAGE_PROBE_JS)

    assert classify(result).state.value == "human_challenge"
    assert classify(result).challenge == "captcha"


def test_the_probe_and_the_extractor_agree_on_an_activity_feed(browser):
    """
    The state machine calls the page authenticated because a feed is
    present, and the extractor reads that same feed. If they disagreed,
    collection would start against a page with nothing on it.
    """

    from src.ingestion.auth_state import PAGE_PROBE_JS, classify

    _context, page = render(browser, FEED)

    assert classify(page.evaluate(PAGE_PROBE_JS)).state.value == (
        "authenticated"
    )
    assert len(extract(page)) == 1


def test_the_extract_contract_is_a_list_of_dicts(browser):
    """The collector indexes the result, so its shape is a contract."""

    _context, page = render(browser, FEED)

    payload = json.dumps(extract(page))

    assert isinstance(json.loads(payload), list)

    for entry in extract(page):
        assert isinstance(entry, dict)
        assert set(entry) >= {
            "urn",
            "url",
            "text",
            "published",
            "author",
            "media",
        }

# ---------------------------------------------------------------------
# Scroll container
# ---------------------------------------------------------------------
#
# The bug these cover: the collector scrolled the mouse and measured
# document.body.scrollHeight, but LinkedIn renders the feed into an
# inner scrolling container. The document height was pinned to the
# viewport, so every poll saw "no change", collection stopped after the
# three posts already on screen, and nothing reported an error.


# A container that genuinely overflows, which is what makes it the
# scroller. A container whose content fits its window is correctly not
# treated as one, so the fixture has to hold more than it can show.
FEED_MARKUP = """
<main id="workspace" style="height:200px;overflow-y:scroll">
  <div class="post">
    <div>
      <a href="https://www.linkedin.com/feed/update/urn:li:activity:1/">
        permalink</a>
    </div>
    <div>Jan 15, 2025</div>
    <div>Window functions compute across a partition.</div>
  </div>
  <div style="height:3000px">the rest of the feed</div>
</main>
"""


def scroll_state(page):
    return page.evaluate(SCROLL_STATE_JS, PERMALINK_SELECTOR)


def test_the_scroll_state_finds_the_inner_container(browser):
    """
    The document is 720px tall in a 720px viewport, so it is not the
    scroller. The container holding the feed is.
    """

    _context, page = render(browser, FEED_MARKUP)

    state = scroll_state(page)

    assert state["inWindow"] is False
    assert state["clientHeight"] == 200
    # The container is taller than its window, which is what makes it
    # the thing worth scrolling.
    assert state["scrollHeight"] > state["clientHeight"]
    assert state["atBottom"] is False


def test_the_scroll_state_falls_back_to_the_window(browser):
    """A page that scrolls the window has no inner container."""

    _context, page = render(browser, "<main>Short.</main>")

    state = scroll_state(page)

    assert state["inWindow"] is True


def test_advancing_scrolls_the_container_not_the_window(browser):
    """
    The container's position moves and the window's does not. A wheel
    event aimed at the window would have changed nothing.
    """

    _context, page = render(browser, FEED_MARKUP)

    before = scroll_state(page)

    result = page.evaluate(
        SCROLL_ADVANCE_JS,
        {"step": 400, "permalink": PERMALINK_SELECTOR},
    )

    after = scroll_state(page)

    assert result["moved"] is True
    assert after["scrollTop"] > before["scrollTop"]


def test_advancing_reports_the_end_of_the_feed(browser):
    """A container that has reached its end reports atBottom."""

    _context, page = render(browser, FEED_MARKUP)

    state = scroll_state(page)
    assert state["atBottom"] is False

    page.evaluate(
        SCROLL_ADVANCE_JS,
        {"step": 100000, "permalink": PERMALINK_SELECTOR},
    )

    assert scroll_state(page)["atBottom"] is True


def test_a_container_that_cannot_move_reports_no_movement(browser):
    """
    The end of a feed is the container refusing to scroll further. The
    caller turns that into a stop rather than looping forever.
    """

    _context, page = render(browser, FEED_MARKUP)

    for _ in range(4):
        page.evaluate(
            SCROLL_ADVANCE_JS,
            {"step": 100000, "permalink": PERMALINK_SELECTOR},
        )

    result = page.evaluate(
        SCROLL_ADVANCE_JS,
        {"step": 100000, "permalink": PERMALINK_SELECTOR},
    )

    # It reports being at the bottom even though the position cannot
    # advance any further.
    assert result["atBottom"] is True


def test_the_scroll_state_survives_a_page_without_posts(browser):
    """An empty feed must not raise; it simply has nothing to scroll."""

    _context, page = render(browser, "<main>Nothing here.</main>")

    state = scroll_state(page)

    assert isinstance(state, dict)
    assert "atBottom" in state


def test_the_scroll_state_is_bounded_by_the_caller(browser):
    """
    Every scroll read is a single evaluation, so a poll cannot
    accumulate the per-call waits that made the original loop appear
    to hang.
    """

    calls: list[str] = []

    _context, page = render(browser, FEED_MARKUP)

    original = page.evaluate

    def counting(expression, arg=None):
        calls.append(expression)
        return original(expression, arg)

    page.evaluate = counting

    scroll_state(page)

    assert calls.count(SCROLL_STATE_JS) == 1
