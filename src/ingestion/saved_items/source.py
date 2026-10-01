"""
The Saved Items source.

This is the whole of the project's contact with a saved list: it reads
files the user exported, reads bundles the user assembled beside them,
and produces posts. It has no browser, opens no session and fetches
nothing. That is the point of the phase — the knowledge pipeline does
not depend on how LinkedIn's web interface happens to be laid out, so
it keeps working when the interface changes, and the project never has
to retrieve a saved post to know what the user saved.

An item with no content behind it is not turned into a post. A link and
a date is what an export of a saved list contains, and manufacturing a
body for it would put text into the knowledge base that nobody wrote.
Such an item stays pending in the manifest until content arrives, and
the report says how many are waiting.

Because the item's identity is derived from its canonical URL, running
this twice over the same export produces the same posts rather than
two copies of each, and only content that actually changed is re-read.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from src.ingestion.saved_items.bundles import (
    BundleAssociation,
    BundleError,
    CapturedContent,
    association_for,
    content_digest,
    discover_bundles,
    read_bundle,
)
from src.ingestion.saved_items.manifest import (
    SavedItemsManifest,
    SavedItemsReport,
    manifest_path,
)
from src.ingestion.saved_items.model import (
    CaptureMethod,
    SavedItem,
    SavedItemState,
)
from src.ingestion.saved_items.readers import (
    ManifestError,
    ManifestRead,
    read_manifest,
)
from src.ingestion.sources.base import (
    CollectedPost,
    CollectionStopped,
    Source,
    StopReason,
)
from src.ingestion.sources.manual import parse_day


#: How a bundle was matched to an item, in the order the match is
#: tried. A name is cheapest and least ambiguous, so it is tried first,
#: and which one matched is recorded on the post: it is the difference
#: between "found by its own name" and "guessed from a link inside it".
MATCH_BY_NAME = "directory name"
MATCH_BY_ID = "source id in capture.json"
MATCH_BY_URL = "canonical url in capture.json"
MATCH_BY_MANIFEST = "bundle named in the manifest"
MATCH_NONE = "no bundle"


class SavedItemsSource(Source):
    """
    Saved Items as a first-class source.

    Implements the same contract as every other source, so the collector
    keeps ownership of persistence, deduplication and limits, and a
    saved post is stored exactly where a collected one is.
    """

    name = "saved_items"

    #: A saved post is a LinkedIn post, so it is stored under the same
    #: platform. What differs is how it arrived, which is recorded on
    #: each post as its capture method.
    platform = "linkedin"

    def __init__(
        self,
        bundle_root: str | Path,
        *,
        manifest_file: str | Path | None = None,
        manifest: SavedItemsManifest | None = None,
    ) -> None:
        self.bundle_root = Path(bundle_root)

        self.manifest = manifest or SavedItemsManifest.load(
            Path(manifest_file)
            if manifest_file is not None
            else manifest_path(self.bundle_root)
        )

        #: Problems found while reading, carried into the report rather
        #: than raised, because one unreadable row must not cost the
        #: rest of the run.
        self.issues: list[str] = []

        self._bundles: list[Path] | None = None
        self._claimed: set[str] = set()
        self._content: dict[str, CapturedContent] = {}
        self._match: dict[str, str] = {}

        self._new = 0
        self._duplicates = 0
        self._read = 0
        self._changed = 0
        self._unchanged = 0
        self._captured = 0

    # -----------------------------------------------------------------
    # Manifests
    # -----------------------------------------------------------------

    def read_manifests(self, paths: list[str | Path]) -> int:
        """
        Read one or more saved-list files into the manifest.

        Returns how many items were newly discovered. Reading the same
        list twice adds nothing, because a repeat collapses onto the
        record already stored rather than becoming a second item.
        """
        discovered = 0

        for entry in paths:
            try:
                read = read_manifest(entry)

            except ManifestError as exc:
                self.issues.append(f"manifest: {exc}")
                continue

            for issue in read.issues:
                self.issues.append(str(issue))

            self._read += len(read.items)

            discovered += self._absorb_manifest(read)

        self.manifest.save()

        return discovered

    def _absorb_manifest(self, read: ManifestRead) -> int:
        """Record every item in a list, counting new ones and repeats."""
        new = 0

        for item in read.items:
            if self.manifest.upsert(item) == "new":
                new += 1
                self._new += 1

            else:
                self._duplicates += 1

        return new

    # -----------------------------------------------------------------
    # Discovery
    # -----------------------------------------------------------------

    def discover(self, **limits: object):
        """
        Yield one collected post per Saved Item that has content.

        Items with no content are not yielded and are not an error. They
        remain pending, and the report counts them, because the next
        thing that happens to them is that the user supplies a capture.
        """
        max_posts = limits.get("max_posts")
        since = limits.get("since")
        until = limits.get("until")

        if not self.bundle_root.is_dir():
            raise CollectionStopped(
                StopReason.FAILED,
                f"Saved items directory does not exist: {self.bundle_root}",
            )

        produced = 0

        for source_id, item in self._ordered():
            if isinstance(max_posts, int) and produced >= max_posts:
                raise CollectionStopped(
                    StopReason.MAX_POSTS,
                    f"Reached the configured limit of {max_posts}.",
                )

            content = self._absorb(item)

            if content is None:
                continue

            collected = self._to_post(item, content)

            if collected is None:
                continue

            if not _within_window(collected.published_at, since, until):
                continue

            yield collected

            produced += 1

            if isinstance(max_posts, int) and produced >= max_posts:
                raise CollectionStopped(
                    StopReason.MAX_POSTS,
                    f"Reached the configured limit of {max_posts}.",
                )

        self.manifest.save()

        raise CollectionStopped(StopReason.EXHAUSTED)

    def _ordered(self) -> list[tuple[str, SavedItem]]:
        """
        Every known item, new ones first.

        A new item is read before a known one so a first run makes
        progress on the backlog before it re-checks work already done.
        Within each group the order is by source id, so two runs over
        the same input do the same thing in the same sequence.
        """
        items = list(self.manifest.items.items())

        def by_id(pair: tuple[str, SavedItem]) -> str:
            return pair[0]

        fresh = sorted(
            (
                pair
                for pair in items
                if pair[1].state is SavedItemState.PENDING
            ),
            key=by_id,
        )

        known = sorted(
            (
                pair
                for pair in items
                if pair[1].state is not SavedItemState.PENDING
            ),
            key=by_id,
        )

        return fresh + known

    # -----------------------------------------------------------------
    # Bundles
    # -----------------------------------------------------------------

    def _absorb(self, item: SavedItem) -> CapturedContent | None:
        """
        Read the bundle belonging to an item, if there is one.

        Returns the content, or None when the item has none. A bundle
        that cannot be read is recorded as a failure against the item
        rather than raised, so one bad capture does not cost the run.
        """
        bundle = self._find_bundle(item)

        if bundle is None:
            if not item.capture_notes:
                # Recorded once, so a report can tell "nothing supplied
                # yet" apart from "supplied and unreadable".
                item.capture_notes = [
                    f"{MATCH_NONE}; nothing has been supplied yet"
                ]

            self._content.pop(item.source_id, None)

            return None

        try:
            content = read_bundle(bundle, root=self.bundle_root)

        except BundleError as exc:
            self.issues.append(f"{item.canonical_url}: {exc}")

            item.state = SavedItemState.FAILED
            item.failure_reason = str(exc)
            item.capture_notes = [str(exc)]

            return None

        for note in content.unreadable:
            self.issues.append(f"{item.canonical_url}: {note}")

        digest = content_digest(content)

        if not digest:
            # The bundle held nothing readable. That is a real answer
            # and recording it as a capture would be a false one.
            item.capture_notes = list(content.notes) or [
                "the bundle held nothing readable"
            ]
            item.content_path = None
            item.content_digest = None
            item.media_paths = []

            self._content.pop(item.source_id, None)

            if item.state.rank >= SavedItemState.CAPTURED.rank:
                # Content that was readable is not any more, so the
                # analysis on file no longer describes it.
                item.state = SavedItemState.PENDING
                item.post_id = None

            return None

        if digest == item.content_digest:
            self._unchanged += 1

            self._content[item.source_id] = content

            if item.state is not SavedItemState.ENRICHED:
                item.state = SavedItemState.CAPTURED

            return content

        if item.content_digest:
            self._changed += 1

        self._captured += 1

        self._apply(item, content, digest, bundle)

        return content

    def _apply(
        self,
        item: SavedItem,
        content: CapturedContent,
        digest: str,
        bundle: str,
    ) -> None:
        """Record a capture on an item, keeping the user's metadata."""
        stale_post = item.post_id

        item.content_path = bundle
        item.content_digest = digest
        item.media_paths = [path.name for path in content.media]
        item.capture_notes = list(content.notes)

        if content.title and not item.title:
            item.title = content.title

        if content.author and not item.author:
            item.author = content.author

        if content.published_at and not item.notes:
            # The page's own date is a publication date, not a save date,
            # so it is kept apart rather than merged into one.
            item.notes = f"published: {content.published_at}"

        item.capture_method = (
            CaptureMethod.USER_PROVIDED
            if content.has_text
            else CaptureMethod.USER_SAVED_PAGE
        )

        item.state = SavedItemState.CAPTURED
        item.failure_reason = None

        if stale_post:
            # The stored post holds content this bundle no longer has.
            # The collector rewrites it and the content fingerprint
            # invalidates the analysis, so nothing stale survives.
            item.post_id = None

        self._content[item.source_id] = content

    def _find_bundle(self, item: SavedItem) -> str | None:
        """
        The bundle belonging to an item, as a root-relative name.

        A bundle is claimed by at most one item, so two items cannot end
        up sharing the same capture and being given the same body.
        """
        if item.bundle:
            candidate = item.bundle

            if (self.bundle_root / candidate).exists():
                self._match[item.source_id] = MATCH_BY_MANIFEST

                return candidate

        for bundle in self._bundle_list():
            key = str(bundle)

            if key in self._claimed:
                continue

            association = self._associate(bundle)

            reason = self._match_reason(item, association)

            if reason is not None:
                self._claimed.add(key)
                self._match[item.source_id] = reason

                return key

        return None

    def _bundle_list(self) -> list[Path]:
        if self._bundles is None:
            try:
                self._bundles = discover_bundles(self.bundle_root)

            except BundleError as exc:
                self.issues.append(str(exc))
                self._bundles = []

        return self._bundles

    def _associate(self, bundle: Path) -> BundleAssociation | None:
        try:
            return association_for(bundle, root=self.bundle_root)

        except (BundleError, OSError) as exc:
            self.issues.append(f"{bundle.name}: {exc}")

            return None

    def _match_reason(
        self,
        item: SavedItem,
        association: BundleAssociation | None,
    ) -> str | None:
        """How a bundle claims an item, or None if it does not."""
        if association is None:
            return None

        if association.url and association.url == item.canonical_url:
            return MATCH_BY_URL

        keys = {item.source_id, _safe(item.source_id)}

        if item.identifier:
            keys.add(item.identifier)
            keys.add(_safe(item.identifier))

        if item.post_id:
            keys.add(item.post_id)
            keys.add(_safe(item.post_id))

        claims = set(association.claim_keys)

        if keys & claims:
            return (
                MATCH_BY_ID
                if association.source_id == item.source_id
                else MATCH_BY_NAME
            )

        return None

    # -----------------------------------------------------------------
    # Posts
    # -----------------------------------------------------------------

    def _to_post(
        self,
        item: SavedItem,
        content: CapturedContent,
    ) -> CollectedPost | None:
        """Turn a captured item into a post, or None if it has no body."""
        text = self._body(item, content)

        if not text.strip():
            self.issues.append(
                f"{item.canonical_url}: the capture produced no text"
            )
            return None

        return CollectedPost(
            source_post_id=item.source_id,
            text=text,
            url=item.canonical_url,
            published_at=item.saved_date,
            author=item.author,
            media=list(content.media),
            capture_method=item.capture_method.value,
            provenance={
                "saved_item_id": item.source_id,
                "saved_date": item.saved_date,
                "canonical_url": item.canonical_url,
                "url_kind": item.kind,
                "capture_state": SavedItemState.CAPTURED.value,
                "capture_match": self._match.get(
                    item.source_id, MATCH_BY_MANIFEST
                ),
                "capture_notes": list(item.capture_notes),
                "saved_notes": item.notes,
                "metadata_only": False,
            },
        )

    def _body(
        self,
        item: SavedItem,
        content: CapturedContent,
    ) -> str:
        """
        The text of a captured item.

        The capture's own text is used when there is any. When there is
        none, the body states what the item is and where it came from and
        says plainly that no body text was supplied. It never invents
        one, and it never claims the post was collected from LinkedIn.
        """
        if content.has_text:
            return content.text

        lines = [
            f"Saved item from LinkedIn: "
            f"{item.title or item.canonical_url}"
        ]

        if item.notes:
            lines.extend(["", item.notes])

        lines.extend(
            [
                "",
                "No body text was supplied with this capture. The link "
                "and save date come from the user's saved-items export; "
                "the material listed below is what the user attached.",
            ]
        )

        if content.media:
            lines.append("")
            lines.append(
                "Attached: "
                + ", ".join(path.name for path in content.media)
            )

        for note in item.capture_notes:
            if note:
                lines.append(f"- {note}")

        return "\n".join(lines)

    # -----------------------------------------------------------------
    # Reporting
    # -----------------------------------------------------------------

    def report(self) -> SavedItemsReport:
        """
        Count the run.

        Two different questions, kept apart because they have different
        answers. ``discovered``, ``new`` and ``duplicates`` describe
        what reading the list files found this run, and the first is
        exactly the sum of the other two. The rest describe the state of
        every known item, because a list of links is still outstanding
        work long after the run that read it.
        """
        report = self.manifest.report(
            discovered=self._read,
            new=self._new,
            duplicates=self._duplicates,
            issues=self.issues,
        )

        report.changed = self._changed
        report.unchanged = self._unchanged

        return report

    @property
    def captured(self) -> int:
        """How many items had content read for them."""
        return self._captured


_SAFE = str.maketrans(
    {character: "-" for character in ":\\/?*|\"<>\n\r\t"}
)


def reconcile(
    manifest: SavedItemsManifest,
    *,
    posts_root: str | Path,
) -> int:
    """
    Update the manifest from what is actually in ``data/posts``.

    The importer is the only thing that can say an item was imported,
    and it says so by writing a post. So rather than trusting the source
    to have succeeded, this reads the posts back and records what is
    really there. An item whose post is missing stays unimported, and
    an item whose post carries an analysis is recorded as enriched
    because the analysis is on disk, not because a run intended to write
    one.

    Returns how many items changed state.
    """
    from src.ingestion.post_document import PostDocument

    root = Path(posts_root)

    if not root.is_dir():
        return 0

    by_item: dict[str, tuple[str, bool]] = {}

    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or directory.name.startswith("."):
            continue

        try:
            document = PostDocument.load(directory)

        except Exception:  # noqa: BLE001
            # A post this manifest knows nothing about, or one that is
            # being written right now. Neither is this manifest's
            # business, and guessing would be worse than skipping.
            continue

        item_id = document.provenance().get("saved_item_id")

        if not isinstance(item_id, str) or not item_id.strip():
            continue

        analysis = document.data.get("ai_analysis")

        enriched = isinstance(analysis, dict) and any(
            analysis.get(key)
            for key in ("summary", "topics", "subtopics", "concepts")
        )

        # Read in sorted order and last one wins, so two runs agree.
        by_item[item_id.strip()] = (document.post_id, enriched)

    changed = 0

    for item in list(manifest.items.values()):
        found = by_item.get(item.source_id)

        if found is None:
            # The post is gone, so whatever the manifest claimed about
            # it no longer holds and it is worth trying again.
            if item.post_id or item.state.rank >= SavedItemState.IMPORTED.rank:
                item.post_id = None
                item.state = SavedItemState.PENDING
                item.touch()
                changed += 1

            continue

        post_id, enriched = found

        target = (
            SavedItemState.ENRICHED if enriched else SavedItemState.IMPORTED
        )

        if item.post_id != post_id or item.state is not target:
            item.post_id = post_id
            item.state = target
            item.failure_reason = None
            item.touch()
            changed += 1

    if changed:
        manifest.save()

    return changed


def _safe(value: str) -> str:
    return value.translate(_SAFE).strip("-")


def _within_window(
    published_at: str | None,
    since: object,
    until: object,
) -> bool:
    """Whether a moment falls inside a date window."""
    if since is None and until is None:
        return True

    if not published_at:
        # Nothing to compare. Excluding it would silently drop every
        # item whose export carried no date, which is most of them.
        return True

    try:
        moment = _as_date(published_at)

    except ValueError:
        return True

    for limit, keep_earlier in ((since, True), (until, False)):
        if limit is None:
            continue

        try:
            boundary = _as_date(str(limit))

        except ValueError:
            continue

        if keep_earlier and moment < boundary:
            return False

        if not keep_earlier and moment > boundary:
            return False

    return True


def _as_date(value: str) -> date:
    try:
        return parse_day(value)

    except (ValueError, TypeError):
        pass

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()

    except ValueError as exc:
        raise ValueError(f"{value!r} is not a date") from exc
