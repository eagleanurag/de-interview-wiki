"""
The visual stage, as the pipeline runs it.

Between the archive and the enricher. Its whole job is to turn a post's
images into text the existing enrichment already knows how to consume,
and then get out of the way.

That it needs no change to the enricher is the point of the design.
``AIEnricher._source_text`` already appends ``media.extracted_text`` to
the post body before building the prompt, and the grounding check reads
the same string. So writing what a slide says into that field makes the
slide eligible source material for questions, topics and concepts
without a single change to the prompt, the models or the grounding
rules. A technology on a slide is adopted only if the slide's own words
are in the grounding source, which is exactly the guarantee CP9 built
for the post body.

Two decisions worth stating plainly.

**The committed post is not modified.** ``post.json`` is the author's
words plus the files attached to them, and this stage reads images and
adds an interpretation. Writing the interpretation back would make the
source depend on a model's output, and the source digest with it. So
the visual text lives on the in-memory post during enrichment, and in
the worker result afterwards -- which is a derived artifact, and the
right place for a derived claim.

**One writer.** Concurrent posts share the store on disk, but each post
writes only its own analyses and no two posts write the same file,
because a file is named by the content digest it describes. Identical
pictures in different posts resolve to one stored analysis by content,
which is the deduplication working, not a race.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from src.models import KnowledgePost, MediaItem
from src.pipeline.paths import log
from src.visual.archive import (
    Archive,
    ArchiveError,
    assets_for_post,
    load_archive,
    resolve_asset_path,
)
from src.visual.dedupe import asset_digest, mark_duplicates, redundant_assets
from src.visual.models import (
    PROCESSOR_CONFIGURATION,
    PROCESSOR_VERSION,
    AssetRole,
    PostVisual,
    ProcessingState,
)
from src.visual.plan import build_plan
from src.visual.processor import build_default_processor
from src.visual.stage import PostOutcome, VisualStage, usable_assets
from src.visual.store import VisualStore, store_root


def archive_post_id(media_path: str) -> str | None:
    """
    The archive post a stored media file belongs to.

    Taken from the filename rather than from a lookup table, because the
    filename is the only link the committed posts carry. The committed
    identifier is a hash of the archive identifier, which is
    deliberate -- it keeps the archive's own ids out of the public site
    -- so the reverse has to come from the file name instead.
    """

    from src.visual.assets import SLIDE_NAME

    stem = Path(str(media_path)).stem

    match = SLIDE_NAME.match(stem)

    if match:
        return match.group("stem")

    from src.visual.assets import DOCUMENT_NAME

    document = DOCUMENT_NAME.match(stem)

    if document:
        return document.group("stem")

    return stem or None


@dataclass
class VisualRun:
    """What the stage did, across every post it touched."""

    posts: int = 0
    posts_processed: int = 0
    posts_skipped: int = 0
    posts_failed: int = 0
    slides_read: int = 0
    slides_reused: int = 0
    calls: int = 0
    retries: int = 0
    failed_assets: int = 0
    duplicates_reused: int = 0
    seconds: float = 0.0
    outcomes: dict[str, PostOutcome] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "posts": self.posts,
            "posts_processed": self.posts_processed,
            "posts_skipped": self.posts_skipped,
            "posts_failed": self.posts_failed,
            "slides_read": self.slides_read,
            "slides_reused": self.slides_reused,
            "duplicates_reused": self.duplicates_reused,
            "calls": self.calls,
            "retries": self.retries,
            "failed_assets": self.failed_assets,
            "seconds": round(self.seconds, 1),
            "failures": self.failures[:50],
        }

    def render(self) -> str:
        lines = [
            "",
            "Visual stage",
            "------------",
            f"{'Posts considered':<34}{self.posts:>8}",
            f"{'Posts processed':<34}{self.posts_processed:>8}",
            f"{'Posts already understood':<34}{self.posts_skipped:>8}",
            f"{'Slides read':<34}{self.slides_read:>8}",
            f"{'Slides reused':<34}{self.slides_reused:>8}",
            f"{'  of which duplicate content':<34}{self.duplicates_reused:>8}",
            f"{'Model calls':<34}{self.calls:>8}",
            f"{'Retries':<34}{self.retries:>8}",
            f"{'Assets that could not be read':<34}{self.failed_assets:>8}",
            f"{'Posts failed':<34}{self.posts_failed:>8}",
            "",
            f"Duration {self.seconds / 60:.1f} min ({self.seconds:.0f}s)",
        ]

        if self.failures:
            lines.append("")
            lines.append("Unreadable assets")
            lines.append("------------------")

            for entry in self.failures[:20]:
                lines.append(f"  {entry}")

            if len(self.failures) > 20:
                lines.append(f"  ... and {len(self.failures) - 20} more")

        return "\n".join(lines)


class VisualRunner:
    """
    Runs the visual stage over posts, with bounded concurrency.

    Concurrency is over *posts*, never over slides within one post. A
    post's slides are read in order and synthesised together, and
    interleaving two posts' batches would make the order of a post's own
    slides depend on the scheduler.
    """

    def __init__(
        self,
        archive_root: str | Path,
        *,
        store: VisualStore | None = None,
        processor=None,
        batch_size: int = 5,
        attempts: int = 3,
        version: str = PROCESSOR_VERSION,
        configuration: str = PROCESSOR_CONFIGURATION,
        progress=None,
    ) -> None:
        self.archive_root = Path(archive_root)
        self.store = store or VisualStore(store_root())
        self.batch_size = batch_size
        self.attempts = attempts
        self.version = version
        self.configuration = configuration
        self.progress = progress or log
        self._processor = processor
        self._archive: Archive | None = None
        self._lock = threading.Lock()

    @property
    def archive(self) -> Archive:
        if self._archive is None:
            self._archive = load_archive(self.archive_root)

        return self._archive

    def processor(self):
        """The processor, built once and shared across threads."""

        if self._processor is None:
            self._processor = build_default_processor()

        return self._processor

    # -- discovery ---------------------------------------------------

    def posts_with_media(self, posts: list[KnowledgePost]) -> dict[str, str]:
        """
        Map committed post identifiers to archive identifiers.

        From the media each committed post already references. A post
        with no media has no images and nothing to do here, which is
        most of them.
        """

        mapping: dict[str, str] = {}

        for post in posts:
            for item in post.media:
                archive_id = archive_post_id(item.path)

                if archive_id and archive_id in self.archive.posts:
                    mapping[post.id] = archive_id

                    break

        return mapping

    def plan(self, limit: int | None = None):
        """The plan, without building a processor or spending anything."""

        return build_plan(
            self.archive_root,
            store=self.store,
            version=self.version,
            configuration=self.configuration,
            batch_size=self.batch_size,
            limit=limit,
        )

    # -- execution ---------------------------------------------------

    def run(
        self,
        posts: list[KnowledgePost],
        *,
        jobs: int = 1,
        limit: int | None = None,
    ) -> VisualRun:
        """
        Understand the images of every post that has any.

        ``limit`` bounds how many posts are processed, which is how a
        run is made to be small while the code stays the code that will
        eventually do all of it.
        """

        started = time.monotonic()

        run = VisualRun()

        try:
            mapping = self.posts_with_media(posts)

        except ArchiveError as exc:
            run.failures.append(f"archive unavailable: {exc}")

            run.seconds = time.monotonic() - started

            return run

        run.posts = len(mapping)

        if limit is not None:
            mapping = dict(list(mapping.items())[:limit])
            run.posts = len(mapping)

        if not mapping:
            run.seconds = time.monotonic() - started

            return run

        processor = self.processor()

        if not processor.available():
            reason = getattr(processor, "unavailable_reason", None) or (
                getattr(processor, "last_error", "")
                or "no visual processor is available"
            )

            run.failures.append(f"processor unavailable: {reason}")
            run.seconds = time.monotonic() - started

            return run

        stage = VisualStage(
            processor=processor,
            store=self.store,
            version=self.version,
            configuration=self.configuration,
            batch_size=self.batch_size,
            attempts=self.attempts,
            progress=self.progress,
        )

        def work(post_id: str, archive_id: str) -> PostOutcome | None:
            try:
                assets = assets_for_post(self.archive, archive_id)

            except Exception as exc:  # noqa: BLE001
                self.progress(f"[{post_id}] discovery failed: {exc}")

                return None

            if not assets:
                return None

            return stage.run_post(
                archive_id,
                assets,
                lambda asset: resolve_asset_path(self.archive, asset),
            )

        def collect(post_id: str, outcome: PostOutcome) -> None:
            with self._lock:
                # Keyed by the committed post identifier, which is
                # what the caller holds, rather than the archive's
                # own. The two differ deliberately: the committed
                # identifier is a hash, which keeps the archive's
                # ids off the published site.
                run.outcomes[post_id] = outcome
                run.posts_processed += 1
                run.slides_read += outcome.processed
                run.slides_reused += outcome.reused
                run.calls += outcome.calls
                run.retries += outcome.retries
                run.failed_assets += outcome.failures

                for entry in outcome.visual.failures:
                    run.failures.append(f"{post_id}/{entry}")

        if jobs > 1:
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                futures = {
                    executor.submit(work, post_id, archive_id): post_id
                    for post_id, archive_id in mapping.items()
                }

                for future in as_completed(futures):
                    post_id = futures[future]

                    try:
                        outcome = future.result()

                    except Exception as exc:  # noqa: BLE001
                        # One post failing is one post. The rest of the
                        # batch continues, and the reason is kept.
                        with self._lock:
                            run.posts_failed += 1
                            run.failures.append(
                                f"{post_id}: {type(exc).__name__}: {exc}"
                            )

                        continue

                    if outcome is None:
                        run.posts_skipped += 1

                        continue

                    collect(post_id, outcome)

        else:
            for post_id, archive_id in mapping.items():
                outcome = work(post_id, archive_id)

                if outcome is None:
                    run.posts_skipped += 1

                    continue

                collect(post_id, outcome)

        run.seconds = time.monotonic() - started

        return run


def inject(post: KnowledgePost, outcome: PostOutcome) -> KnowledgePost:
    """
    Put what was read into the post's media, in memory.

    ``extracted_text`` is what the existing enricher and grounding check
    already read, so this is the whole integration. ``sequence``,
    ``role``, ``sha256`` and ``extraction_method`` are recorded beside
    it so a reader can walk a claim back to the file it came from.

    Media already in the post that the archive did not contribute keeps
    whatever it had: this stage adds to the post, it does not replace it.

    Assets that failed keep their entry and gain a note. A gap is
    visible, which is the point.
    """

    by_filename = {
        Path(item.path).name: item for item in post.media
    }

    analyses = {
        Path(analysis.source_path).name: analysis
        for analysis in outcome.visual.analyses
    }

    for asset in outcome.visual.assets:
        filename = asset.filename

        item = by_filename.get(filename)

        if item is None:
            item = MediaItem(
                type=(
                    "pdf"
                    if asset.media_type == "pdf"
                    else "image"
                    if asset.media_type == "image"
                    else "other"
                ),
                path=asset.path,
            )

            post.media.append(item)
            by_filename[filename] = item

        item.sequence = asset.sequence
        item.role = asset.role.value
        item.sha256 = asset.sha256 or None

        if asset.state is ProcessingState.FAILED:
            item.extraction_method = None
            item.description = (
                f"{asset.format or 'unreadable'} image, "
                f"{asset.width or '?'}x{asset.height or '?'} pixels. "
                f"Could not be read: {asset.note}"
            )

            continue

        analysis = analyses.get(filename)

        if analysis is None:
            item.extraction_method = None
            item.description = (
                f"{asset.format} image, "
                f"{asset.width}x{asset.height} pixels"
            )

            continue

        text = analysis.as_source_text()

        item.extracted_text = text or None
        item.extraction_method = analysis.processor

        parts = [
            f"{asset.format} image, {asset.width}x{asset.height} pixels"
        ]

        if asset.role is AssetRole.SLIDE:
            parts.append(f"slide {asset.sequence + 1}")

        if asset.role is AssetRole.THUMBNAIL:
            parts.append("low-resolution preview, full slide also present")

        if analysis.confidence:
            parts.append(
                f"read with confidence {analysis.confidence:.2f}"
            )

        item.description = ". ".join(parts)

    # Ordered so a post's media reads in slide order, which is what the
    # prompt and the wiki both assume.
    post.media.sort(
        key=lambda entry: (entry.sequence, Path(entry.path).name)
    )

    return post


def digest_for(post: KnowledgePost, outcome: PostOutcome | None) -> str:
    """
    The visual fingerprint to record and compare against.

    Empty for a post with no images, so the overwhelming majority of
    posts keep the exact freshness behaviour they had before this stage
    existed.
    """

    if outcome is None or not outcome.visual.assets:
        return ""

    marked = list(outcome.visual.assets)
    mark_duplicates(marked)
    redundant_assets(marked)

    return asset_digest(
        marked,
        processor_version=PROCESSOR_VERSION,
        configuration=PROCESSOR_CONFIGURATION,
    )


def visual_for(posts: list[KnowledgePost]) -> dict[str, PostOutcome]:
    """The stage's results, keyed by committed post identifier."""

    mapping: dict[str, PostOutcome] = {}

    for post in posts:
        archive_id = None

        for item in post.media:
            candidate = archive_post_id(item.path)

            if candidate:
                archive_id = candidate
                break

        if archive_id:
            mapping[post.id] = archive_id

    return mapping


def useful_assets(outcome: PostOutcome) -> list:
    """The assets that carried knowledge, for reporting."""

    return usable_assets(outcome.visual.assets)


__all__ = [
    "VisualRun",
    "VisualRunner",
    "archive_post_id",
    "digest_for",
    "inject",
    "useful_assets",
    "visual_for",
]
