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
from src.visual.assets import activity_for
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
from src.visual.stage import inject as inject_media
from src.visual.store import VisualStore, store_root


def archive_post_id(media_path: str) -> str | None:
    """
    The archive post a stored media file belongs to.

    Taken from the filename rather than from a lookup table, because the
    filename is the only link the committed posts carry. The committed
    identifier is a hash of the archive identifier, which is
    deliberate -- it keeps the archive's own ids out of the public site
    -- so the reverse has to come from the file name instead.

    The implementation lives in :mod:`src.visual.assets`, which owns the
    two filename patterns it is built from. It was defined here, and that
    put the orchestration layer in the import path of every consumer that
    only needed to map a media path to an activity -- including the
    imported-package bridge, which this same package's orchestrator
    imports. Importing ``src.pipeline.visual`` executes
    ``src/pipeline/__init__.py``, and that imports the orchestrator back,
    so a module importing the bridge partway through its own
    initialisation could not reach a name defined further down this file.

    Still exported from here because callers already import it from here,
    including :mod:`src.pipeline.visual_stage`, and churning those to fix
    an import cycle would be a poor trade.
    """

    return activity_for(media_path)


@dataclass
class VisualRun:
    """What the stage did, across every post it touched."""

    posts: int = 0
    posts_processed: int = 0
    posts_skipped: int = 0
    posts_failed: int = 0
    slides_read: int = 0
    slides_reused: int = 0
    #: Assets deliberately not read because another tool already
    #: transcribed them. Reported so a run restricted to the gaps is
    #: never mistaken for a run that looked at everything.
    slides_filtered: int = 0
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
            "slides_filtered": self.slides_filtered,
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
            f"{'Slides already transcribed elsewhere':<34}{self.slides_filtered:>8}",
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
        only: set[str] | None = None,
    ) -> None:
        self.archive_root = Path(archive_root)
        self.store = store or VisualStore(store_root())
        self.batch_size = batch_size
        self.attempts = attempts
        self.version = version
        self.configuration = configuration
        self.progress = progress or log
        # Restricts this run to a named set of files, used when another
        # tool has already transcribed most of a corpus and this stage
        # is only the fallback for what that tool could not read. The
        # default is ``None``, meaning every asset, so nothing about the
        # existing behaviour changes when the option is not used.
        self.only = only
        self._processor = processor
        self._archive: Archive | None = None
        self._lock = threading.Lock()

    @property
    def gaps_skipped(self) -> int:
        """Assets left out by the ``only`` filter, for reporting."""

        return getattr(self, "_gaps_skipped", 0)

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
            only=self.only,
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

            if self.only is not None:
                kept = [
                    asset for asset in assets if asset.filename in self.only
                ]

                if len(kept) != len(assets):
                    # Counted rather than discarded silently: a reader
                    # of the run report needs to know slides were passed
                    # over on purpose and how many.
                    with self._lock:
                        self._gaps_skipped = (
                            getattr(self, "_gaps_skipped", 0)
                            + (len(assets) - len(kept))
                        )

                assets = kept

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

        run.slides_filtered = getattr(self, "_gaps_skipped", 0)

        return run


#: Re-exported from :mod:`src.visual.stage`, where the implementation
#: lives. It is a pure function of a ``PostOutcome`` and a post, needs
#: nothing from the pipeline, and belongs beside the type it consumes.
#:
#: It was defined here, which made the imported-package bridge depend on
#: ``src.pipeline`` in order to reuse it -- and importing
#: ``src.pipeline.visual`` executes ``src/pipeline/__init__.py``, which
#: imports the orchestrator, which imports that same bridge. A source
#: layer reaching into the orchestration layer for a helper is the
#: inversion that produced the cycle, so the helper moved down to where
#: its inputs live rather than the caller being made to work around it.
#:
#: A plain alias rather than a wrapper, so the two names are the same
#: object: a wrapper would be a second signature to keep in step and a
#: second place for the behaviour to drift.
inject = inject_media


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
