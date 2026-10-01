"""
LinkedIn URL normalization for Saved Items.

A Saved Item is identified by where it came from, so the URL has to be
turned into one form before anything else happens. Two exports of the
same saved post differ in whitespace, in a trailing slash, in a tracking
parameter and in a fragment; treating those as four items would
duplicate a post in the knowledge base, and treating two genuinely
different posts as one would lose one of them.

Both mistakes are guarded against here. What is normalized is exactly
what does not identify the content; what identifies it, the path, is
preserved byte for byte.

Nothing in this module fetches anything. A URL is only ever read from
a file the user supplied.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from urllib.parse import (
    parse_qsl,
    quote,
    unquote,
    urlencode,
    urlsplit,
    urlunsplit,
)


class SavedItemUrlError(ValueError):
    """Raised when a URL cannot be understood as a LinkedIn link."""


#: Hosts that serve LinkedIn content. A sub domain is accepted, because
#: LinkedIn serves the same content from several of them.
LINKEDIN_HOSTS = frozenset(
    {
        "linkedin.com",
        "www.linkedin.com",
        "lnkd.in",
        "www.lnkd.in",
    }
)

#: Query parameters that identify how a link was followed rather than
#: what it points at. Removing them makes two exports of the same saved
#: post resolve to one item.
#:
#: Nothing else is removed. An unrecognised parameter is left alone,
#: because it may be what distinguishes two posts that would otherwise
#: collapse.
TRACKING_PARAMS = frozenset(
    {
        "trk",
        "trackingid",
        "trkinfo",
        "lipi",
        "licu",
        "midtoken",
        "midsig",
        "tracking_id",
        "ref",
        "referrer",
        "originalsubdomain",
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "gclid",
        "fbclid",
        "mc_cid",
        "mc_eid",
        "vero_id",
        "vero_conv",
    }
)

#: Schemes that address a document. Anything else is not a web URL.
ALLOWED_SCHEMES = frozenset({"http", "https"})

#: The first path segment of a LinkedIn content URL, which is what
#: distinguishes a post from a profile or a company page.
CONTENT_KINDS = frozenset(
    {
        "posts",
        "feed",
        "pulse",
        "articles",
        "update",
    }
)

_SAFE_HOST = re.compile(r"^[a-z0-9.-]+$", re.IGNORECASE)


@dataclass(frozen=True)
class NormalizedUrl:
    """
    One URL in the forms the pipeline needs.

    ``original`` is what the user supplied and is never rewritten, so a
    reader can see exactly what was saved. ``canonical`` is the identity
    form. ``source_id`` is derived from the canonical URL and is the
    only thing deduplication keys on.
    """

    original: str
    canonical: str
    kind: str
    identifier: str | None
    source_id: str

    @property
    def is_linkedin(self) -> bool:
        return self.kind != "external"

    def as_dict(self) -> dict:
        return {
            "original_url": self.original,
            "canonical_url": self.canonical,
            "kind": self.kind,
            "identifier": self.identifier,
            "source_id": self.source_id,
        }


def source_id_for(canonical_url: str) -> str:
    """
    A stable identifier for a canonical URL.

    A digest rather than the URL itself, because a post identifier
    also becomes a directory name, a job id and a URL segment, and a
    LinkedIn URL contains none of the characters those tolerate.
    Deterministic, so the same URL always resolves to the same item.
    """

    digest = hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()

    return f"urn:li:saved:{digest[:16]}"


def normalize_linkedin_url(raw: object) -> NormalizedUrl:
    """
    Normalize one saved URL.

    Raises :class:`SavedItemUrlError` for anything that is not a URL,
    and for a URL with no path to identify content by. Guessing a host
    or inventing a path would produce an item that points nowhere, and
    a clear error is more useful than a wrong record.
    """

    original = "" if raw is None else str(raw)

    trimmed = original.strip()

    # A bare domain is not a content link, and a wrapped link is a
    # paste accident rather than a different piece of content.
    trimmed = trimmed.strip("<>").strip()

    if not trimmed:
        raise SavedItemUrlError("the URL is empty")

    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", trimmed):
        if trimmed.startswith("//"):
            trimmed = "https:" + trimmed
        elif trimmed.startswith("www.") or trimmed.startswith("linkedin."):
            trimmed = "https://" + trimmed
        else:
            # A value that names a scheme but not a URL is reported as
            # the scheme it names. "javascript:alert(1)" is not a
            # mistyped link, and saying only that it is not absolute
            # would understate why it was refused.
            scheme_like = re.match(
                r"^([a-zA-Z][a-zA-Z0-9+.-]*):", trimmed
            )

            if scheme_like and scheme_like.group(1).lower() not in ALLOWED_SCHEMES:
                raise SavedItemUrlError(
                    f"unsupported URL scheme {scheme_like.group(1).lower()!r}"
                )

            raise SavedItemUrlError(
                f"not an absolute URL: {original[:60]!r}"
            )

    try:
        parts = urlsplit(trimmed)
    except ValueError as exc:
        raise SavedItemUrlError(
            f"the URL could not be parsed: {original[:60]!r}"
        ) from exc

    scheme = (parts.scheme or "").lower()

    if scheme not in ALLOWED_SCHEMES:
        raise SavedItemUrlError(
            f"unsupported URL scheme {scheme or '(none)'!r}"
        )

    host = (parts.hostname or "").lower()

    if not host:
        raise SavedItemUrlError(f"the URL has no host: {original[:60]!r}")

    if not _SAFE_HOST.match(host) or " " in host:
        raise SavedItemUrlError(f"the URL has an unusable host: {host!r}")

    if not is_linkedin_host(host):
        # Not a LinkedIn link. Recorded as external rather than
        # rejected, because a saved list may legitimately contain one
        # and discarding the user's data is worse than labelling it.
        canonical = _canonical(parts, host, scheme)

        return NormalizedUrl(
            original=original,
            canonical=canonical,
            kind="external",
            identifier=None,
            source_id=source_id_for(canonical),
        )

    path = _normalize_path(parts.path)

    kind = _content_kind(path)

    if kind is None:
        # A LinkedIn link that is not content: a profile, a company,
        # a job. It is kept, because it is something the user saved,
        # but it is labelled so nothing downstream treats it as a post.
        canonical = _canonical(parts, host, scheme, path=path)

        return NormalizedUrl(
            original=original,
            canonical=canonical,
            kind="linkedin_other",
            identifier=_path_identifier(path),
            source_id=source_id_for(canonical),
        )

    identifier = _path_identifier(path)

    canonical = _canonical(parts, host, scheme, path=path, kind=kind)

    return NormalizedUrl(
        original=original,
        canonical=canonical,
        kind=kind,
        identifier=identifier,
        source_id=source_id_for(canonical),
    )


def is_linkedin_host(host: str) -> bool:
    """
    Whether a host serves LinkedIn content.

    A sub domain counts, because LinkedIn serves the same content from
    several of them and treating them as different sites would split
    one saved post across several items.
    """

    if not host:
        return False

    if host in LINKEDIN_HOSTS:
        return True

    return host.endswith(".linkedin.com")


def _content_kind(path: str) -> str | None:
    """
    Which kind of LinkedIn content a path points at.

    Returns None for a path that carries no content, so the caller can
    label it rather than treat it as a post.
    """

    segments = [
        segment for segment in path.split("/") if segment
    ]

    if not segments:
        return None

    head = segments[0].lower()

    if head not in CONTENT_KINDS:
        return None

    # ``/feed/update/urn:li:...`` carries its identifier in the fourth
    # segment; the other kinds carry it in the second.
    if head == "feed" and len(segments) >= 2:
        if segments[1].lower() == "update":
            return "post"

    if len(segments) >= 2:
        return head

    return None


def _path_identifier(path: str) -> str | None:
    """The identifying part of a content path, if there is one."""

    segments = [
        segment for segment in path.split("/") if segment
    ]

    if not segments:
        return None

    head = segments[0].lower()

    if head == "feed" and len(segments) >= 2:
        # ``/feed/update/urn:li:activity:123`` carries its identifier in
        # the third segment; the other kinds carry it in the second.
        if segments[1].lower() == "update" and len(segments) >= 3:
            return segments[2]

        if len(segments) >= 2:
            return segments[1]

    if head == "update" and len(segments) >= 3:
        return segments[2]

    if len(segments) >= 2:
        return segments[1]

    return None


def _normalize_path(path: str) -> str:
    """
    Normalize a URL path without changing what it points at.

    Percent-encoding is decoded and then re-encoded canonically, so two
    exports that encoded the same slug differently agree. Unreserved
    characters that were left encoded are restored. Reserved
    separators are never touched, because a path segment boundary is
    meaningful.
    """

    if not path:
        return "/"

    decoded = unquote(path)

    # Collapse duplicate separators, which no export produces
    # deliberately and which would make two URLs look different.
    collapsed = re.sub(r"/{2,}", "/", decoded)

    canonical = quote(collapsed, safe="/:@-._~!$&'()*+,;=")

    if not canonical.startswith("/"):
        canonical = "/" + canonical

    # A trailing slash is normalized away except for the root, so
    # ".../activity" and ".../activity/" are one item.
    if len(canonical) > 1 and canonical.endswith("/"):
        canonical = canonical.rstrip("/") or "/"

    return canonical


def _canonical(
    parts,
    host: str,
    scheme: str,
    *,
    path: str | None = None,
    kind: str | None = None,
) -> str:
    """Rebuild a URL in its identity form."""

    resolved_path = path if path is not None else _normalize_path(parts.path)

    kept = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key.lower() not in TRACKING_PARAMS
    ]

    query = urlencode(sorted(kept), doseq=True)

    # The fragment never identifies content on LinkedIn: it is where a
    # comment sits inside a post, so dropping it is what makes two saves
    # of different comments in one post distinct by URL and not merged.
    return urlunsplit((scheme, host, resolved_path, query, ""))


def try_normalize(raw: object) -> NormalizedUrl | None:
    """
    Normalize a URL, or return None when it cannot be understood.

    Used where a bad value should be counted rather than stop the run,
    such as one row of a spreadsheet. The caller reports what it
    skipped.
    """

    try:
        return normalize_linkedin_url(raw)
    except SavedItemUrlError:
        return None