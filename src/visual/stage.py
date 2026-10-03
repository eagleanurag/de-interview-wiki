"""
Turning a post's images into knowledge, one slide at a time.

This is the part that has to survive a real archive. The properties it
exists to hold:

**A slide is the unit of work and of storage.** A post with 262 slides
is processed in bounded batches, and every batch is written to the store
as it lands. A run that dies at slide 200 keeps 199 results, and the
next run starts at 201.

**One unreadable slide does not cost the post.** A corrupt file, a
refused path or a batch the provider mangled produces an explicit
failure record naming the asset. Nothing is inferred to fill the gap,
because a fabricated reading of slide 47 is worse than an acknowledged
gap at 47.

**Reuse is decided without asking anyone.** An analysis is reused when
the asset's content digest, the processor version and the processor
configuration all match what is stored. That means the second run of
this pipeline costs no model calls at all, and it means a re-run is
reproducible rather than merely repeated.

**Batches are bounded in count and in size.** Five by default, which the
measurements in the plan justify: five slides cost about what one costs,
so batching buys a smaller failure unit rather than speed, and there is
no reason to push it further.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from src.visual.assets import order_assets
from src.visual.dedupe import (
    asset_digest,
    mark_duplicates,
    redundant_assets,
)
from src.visual.models import (
    PROCESSOR_CONFIGURATION,
    PROCESSOR_VERSION,
    AssetRole,
    PostVisual,
    ProcessingState,
    VisualAnalysis,
    VisualAsset,
)
from src.visual.processor import (
    MAX_ASSET_BYTES,
    MAX_BATCH_ASSETS,
    failure_note,
)
from src.visual.store import VisualStore


@dataclass
class PostOutcome:
    """What one post's visual stage produced."""

    post_id: str
    visual: PostVisual
    #: Assets newly read, and assets taken from the store.
    processed: int = 0
    reused: int = 0
    #: Model calls made, which is fewer than the assets processed
    #: whenever a batch succeeded.
    calls: int = 0
    retries: int = 0
    failures: int = 0
    seconds: float = 0.0

    @property
    def digest(self) -> str:
        return self.visual.source_digest

    def summary(self) -> str:
        coverage = self.visual.coverage()

        return (
            f"{self.post_id}: {self.processed} read, {self.reused} reused, "
            f"{self.calls} call(s), {self.failures} failed, "
            f"{coverage['with_text']}/{coverage['slides']} slides with text"
        )


def usable_assets(assets: list[VisualAsset]) -> list[VisualAsset]:
    """
    The assets worth asking about, in order.

    Excluded, each for a stated reason: files that could not be read,
    duplicates whose content is already covered by another asset, and
    previews whose full slide is present. A preview is kept when it is
    the only image a post has, because then it is not a preview of
    anything -- it is the evidence.
    """

    marked = list(assets)

    mark_duplicates(marked)

    redundant = {asset.filename for asset in redundant_assets(marked)}

    usable = [
        asset
        for asset in marked
        if asset.state is not ProcessingState.FAILED
        and asset.filename not in redundant
        and asset.byte_size <= MAX_ASSET_BYTES
    ]

    return order_assets(usable)


def _chunks(items: list, size: int) -> list[list]:
    """Split into batches of at most ``size``, never fewer than one."""

    size = max(1, min(size, MAX_BATCH_ASSETS))

    return [items[index : index + size] for index in range(0, len(items), size)]


class VisualStage:
    """
    Runs the visual stage over posts.

    Holds the store and the processor, and is the only thing that writes
    analyses. Concurrency lives in the caller, which knows about the
    posts; this class is deliberately unaware of it, so that a post's
    slides are always processed in order even when posts run at once.
    """

    def __init__(
        self,
        processor,
        store: VisualStore,
        *,
        version: str = PROCESSOR_VERSION,
        configuration: str = PROCESSOR_CONFIGURATION,
        batch_size: int = 5,
        attempts: int = 3,
        progress=None,
    ) -> None:
        self.processor = processor
        self.store = store
        self.version = version
        self.configuration = configuration
        self.batch_size = batch_size
        self.attempts = max(1, attempts)
        self.progress = progress or (lambda message: None)

    # -- planning ----------------------------------------------------

    def plan_post(self, post_id: str, assets: list[VisualAsset]) -> dict:
        """
        What this post would cost, before anything is spent.

        Returned rather than acted on so the number can be read before
        the money goes. ``todo`` is the count of assets that would reach
        a model, which is the figure worth checking against a budget.
        """

        usable = usable_assets(assets)

        todo = 0
        reusable = 0

        for asset in usable:
            stored = self.store.load(
                asset.sha256, self.version, self.configuration
            )

            if stored is None:
                todo += 1

            else:
                reusable += 1

        digest = asset_digest(
            assets,
            processor_version=self.version,
            configuration=self.configuration,
        )

        batches = len(_chunks(usable, self.batch_size))

        return {
            "post_id": post_id,
            "assets": len(assets),
            "slides": sum(1 for a in assets if a.role is AssetRole.SLIDE),
            "thumbnails": sum(
                1 for a in assets if a.role is AssetRole.THUMBNAIL
            ),
            "duplicates": sum(1 for a in assets if a.duplicate_of),
            "failed": sum(
                1 for a in assets if a.state is ProcessingState.FAILED
            ),
            "usable": len(usable),
            "todo": todo,
            "reusable": reusable,
            "batches": batches,
            "calls_if_unchanged": batches if todo else 0,
            "digest": digest,
        }

    # -- execution ---------------------------------------------------

    def run_post(
        self,
        post_id: str,
        assets: list[VisualAsset],
        resolve,
    ) -> PostOutcome:
        """
        Understand one post's images, reusing whatever is already known.

        ``resolve`` maps an asset to its file on disk. Passed in rather
        than imported so the stage can be tested against fixtures with
        no archive at all, and so that path containment is enforced by
        the archive reader that owns the root.
        """

        started = time.monotonic()

        marked = list(assets)
        mark_duplicates(marked)

        usable = usable_assets(marked)

        digest = asset_digest(
            marked,
            processor_version=self.version,
            configuration=self.configuration,
        )

        outcome = PostOutcome(post_id=post_id, visual=PostVisual(post_id=post_id))
        outcome.visual.assets = order_assets(marked)
        outcome.visual.source_digest = digest
        outcome.visual.processor_version = self.version
        outcome.visual.processor_configuration = self.configuration

        analyses: list[VisualAnalysis] = []
        failures: list[str] = []

        for asset in order_assets(marked):
            if asset.state is ProcessingState.FAILED:
                failures.append(f"{asset.filename}: {asset.note}")

                outcome.failures += 1

                asset.state = ProcessingState.FAILED

                continue

            if asset.duplicate_of or asset.filename not in {
                item.filename for item in usable
            }:
                # Kept in the asset list with its relationship intact, and
                # given the primary's analysis below rather than a call
                # of its own.
                asset.state = ProcessingState.SKIPPED

        pending = [
            asset for asset in usable if self._needs_work(asset)
        ]

        reusable = [asset for asset in usable if asset not in pending]

        for asset in reusable:
            stored = self.store.load(
                asset.sha256, self.version, self.configuration
            )

            if stored is None:
                continue

            stored = stored.model_copy(deep=True)
            stored.reused = True
            stored.sequence = asset.sequence
            stored.source_path = asset.path

            analyses.append(stored)
            outcome.reused += 1
            asset.state = ProcessingState.REUSED

        by_filename = {asset.filename: asset for asset in usable}

        batches = _chunks(pending, self.batch_size)

        for index, batch in enumerate(batches):
            paths: list[Path] = []
            batch_assets: list[VisualAsset] = []

            for asset in batch:
                try:
                    path = resolve(asset)

                except Exception as exc:  # noqa: BLE001
                    failures.append(f"{asset.filename}: {exc}")
                    outcome.failures += 1
                    asset.state = ProcessingState.FAILED

                    continue

                paths.append(path)
                batch_assets.append(asset)

            if not paths:
                continue

            self.progress(
                f"[{post_id}] batch {index + 1}/{len(batches)}: "
                f"{len(paths)} slide(s)"
            )

            produced, attempts_used = self._analyse_batch(
                paths, batch_assets, post_id
            )

            outcome.calls += 1 if produced else 0
            outcome.retries += max(0, attempts_used - 1)

            if not produced:
                for asset in batch_assets:
                    failures.append(
                        f"{asset.filename}: unreadable after "
                        f"{attempts_used} attempt(s)"
                    )
                    outcome.failures += 1
                    asset.state = ProcessingState.FAILED

                continue

            for analysis in produced:
                analyses.append(analysis)
                outcome.processed += 1

                asset = by_filename.get(Path(analysis.source_path).name)

                if asset is not None:
                    asset.state = ProcessingState.PROCESSED

        # A duplicate takes its group's analysis rather than its own
        # call. The record keeps both the asset and the fact that it was
        # not analysed separately.
        by_hash: dict[str, VisualAnalysis] = {}

        for analysis in analyses:
            by_hash.setdefault(analysis.source_asset_hash, analysis)

        for asset in order_assets(marked):
            if asset.duplicate_of and asset.sha256 in by_hash:
                twin = by_hash[asset.sha256].model_copy(deep=True)
                twin.sequence = asset.sequence
                twin.source_path = asset.path

                analyses.append(twin)
                outcome.reused += 1

        # An asset that was skipped because it is redundant still needs
        # its content recorded, or the next run would decide again.
        outcome.visual.analyses = analyses
        outcome.visual.failures = failures
        outcome.visual.calls = outcome.calls
        outcome.visual.reused = outcome.reused
        outcome.seconds = time.monotonic() - started

        return outcome

    def _needs_work(self, asset: VisualAsset) -> bool:
        """Whether this asset's content has not been read yet."""

        return (
            self.store.load(asset.sha256, self.version, self.configuration)
            is None
        )

    def _analyse_batch(
        self,
        paths: list[Path],
        assets: list[VisualAsset],
        post_id: str,
    ) -> tuple[list[VisualAnalysis], int]:
        """
        One batch, retried while retrying could help.

        Uses the same classification as the enricher, so a truncated
        response is recognised as the recoverable thing it is. Anything
        judged permanent -- a missing API key, an unknown model -- is not
        retried, because asking again cannot change the answer and the
        retry would only hide the cause.
        """

        from src.ai.recovery import is_recoverable

        by_name = {asset.filename: asset for asset in assets}

        setter = getattr(self.processor, "set_assets", None)

        last_error: Exception | None = None

        for attempt in range(1, self.attempts + 1):
            if callable(setter):
                setter(by_name)

            try:
                produced = self.processor.analyse(paths)

            except Exception as exc:  # noqa: BLE001
                last_error = exc

                self.progress(
                    f"[{post_id}] attempt {attempt}/{self.attempts} "
                    f"failed: {failure_note(exc)}"
                )

                if not is_recoverable(exc):
                    break

                if attempt < self.attempts:
                    # Bounded and doubling, matching the enricher so
                    # that a provider struggling under both stages is
                    # given room in the same way.
                    time.sleep(min(2.0 * (2 ** (attempt - 1)), 20.0))

                continue

            if not produced:
                last_error = ValueError(
                    "the processor returned nothing for a batch of "
                    f"{len(paths)} image(s)"
                )

                break

            for analysis in produced:
                self.store.save(
                    analysis, self.version, self.configuration
                )

            return produced, attempt

        self.progress(
            f"[{post_id}] batch abandoned after {self.attempts} "
            f"attempt(s): {failure_note(last_error) if last_error else 'unknown'}"
        )

        return [], self.attempts


__all__ = ["PostOutcome", "VisualStage", "usable_assets"]
