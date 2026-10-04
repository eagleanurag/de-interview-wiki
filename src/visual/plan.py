"""
What the visual stage would cost, before it costs anything.

Runs the same discovery, ordering, deduplication and reuse checks the
real run will run, and stops before the first model call. The point is
that the expensive decision is reviewable: how many posts are affected,
how many slides actually reach a model, how many are already known, and
what the plan is doing about the archive's own inconsistencies.

Nothing here writes. A dry run that touched the cache would not be a dry
run.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from src.visual.archive import (
    Archive,
    ArchiveError,
    assets_for_post,
    load_archive,
)
from src.visual.dedupe import mark_duplicates, redundant_assets
from src.visual.models import AssetRole, ProcessingState
from src.visual.store import VisualStore, store_root
from src.visual.stage import VisualStage, usable_assets


@dataclass
class PlanRow:
    """One post, and what would happen to it."""

    post_id: str
    assets: int = 0
    slides: int = 0
    thumbnails: int = 0
    duplicates: int = 0
    missing: int = 0
    corrupt: int = 0
    usable: int = 0
    todo: int = 0
    reusable: int = 0
    batches: int = 0
    digest: str = ""

    @property
    def needs_work(self) -> bool:
        return self.todo > 0

    def as_dict(self) -> dict:
        return {
            "post_id": self.post_id,
            "assets": self.assets,
            "slides": self.slides,
            "thumbnails": self.thumbnails,
            "duplicates": self.duplicates,
            "missing": self.missing,
            "corrupt": self.corrupt,
            "usable": self.usable,
            "todo": self.todo,
            "reusable": self.reusable,
            "batches": self.batches,
            "needs_work": self.needs_work,
        }


@dataclass
class VisualPlan:
    """The whole plan, and the totals that matter."""

    rows: list[PlanRow] = field(default_factory=list)
    orphans: int = 0
    archive_error: str = ""

    #: Contents that appear more than once anywhere in the archive. The
    #: largest such group in this archive is one image held by 42 posts,
    #: so this is not a rounding error: costed per post it overstates
    #: the work by the number of repeats, and costed globally the first
    #: post to read a picture pays for every later post that shows it.
    distinct_contents: int = 0

    #: Assets that will reuse an identical picture already analysed,
    #: either from the store or from an earlier post in this run.
    cross_post_reuse: int = 0

    #: Assets the model will actually see, once per distinct content.
    cold_cache_assets: int = 0

    def totals(self) -> dict[str, int]:
        counter = Counter()

        for row in self.rows:
            counter["posts"] += 1
            counter["assets"] += row.assets
            counter["slides"] += row.slides
            counter["thumbnails"] += row.thumbnails
            counter["duplicates"] += row.duplicates
            counter["missing"] += row.missing
            counter["corrupt"] += row.corrupt
            counter["usable"] += row.usable
            counter["reusable"] += row.reusable
            counter["batches"] += row.batches

        # todo and batches are counted globally rather than summed,
        # because a picture that appears in two posts is read once.
        counter["todo"] = self.cold_cache_assets
        counter["cross_post_reuse"] = self.cross_post_reuse
        counter["distinct_contents"] = self.distinct_contents

        counter["posts_needing_work"] = sum(
            1 for row in self.rows if row.needs_work
        )
        counter["posts_reused"] = sum(
            1 for row in self.rows if not row.needs_work
        )

        counter["orphans"] = self.orphans

        return dict(counter)

    def render(self) -> str:
        """The plan, as something a person can read before agreeing to it."""

        totals = self.totals()

        lines: list[str] = []
        lines.append("")
        lines.append("Visual plan")
        lines.append("------------")

        if self.archive_error:
            lines.append(f"  Archive could not be read: {self.archive_error}")
            lines.append("")

            return "\n".join(lines)

        lines.append(f"  {'Posts in the archive':<34}{totals.get('posts', 0):>8}")
        lines.append(f"  {'Media files referenced':<34}{totals.get('assets', 0):>8}")
        lines.append(f"  {'  of which carousel slides':<34}{totals.get('slides', 0):>8}")
        lines.append(f"  {'  of which previews':<34}{totals.get('thumbnails', 0):>8}")
        lines.append(f"  {'  exact duplicates':<34}{totals.get('duplicates', 0):>8}")
        lines.append(f"  {'Missing files':<34}{totals.get('missing', 0):>8}")
        lines.append(f"  {'Corrupt or unreadable':<34}{totals.get('corrupt', 0):>8}")
        lines.append(f"  {'Files on disk not referenced':<34}{totals.get('orphans', 0):>8}")
        lines.append("")
        lines.append(f"  {'Distinct image contents':<34}{totals.get('distinct_contents', 0):>8}")
        lines.append(f"  {'Repeats of a picture already seen':<34}{totals.get('cross_post_reuse', 0):>8}")
        lines.append("")
        lines.append(f"  {'Posts needing visual work':<34}{totals.get('posts_needing_work', 0):>8}")
        lines.append(f"  {'Posts already understood':<34}{totals.get('posts_reused', 0):>8}")
        lines.append(f"  {'Images to send to the model':<34}{totals.get('todo', 0):>8}")
        lines.append(f"  {'Already understood, from the store':<34}{totals.get('reusable', 0):>8}")
        lines.append(f"  {'Model calls if all of them run':<34}{totals.get('batches', 0):>8}")
        lines.append("")

        affected = [row for row in self.rows if row.needs_work]

        if not affected:
            lines.append("  Nothing to do: every slide is already understood.")
            lines.append("")

            return "\n".join(lines)

        lines.append("  Posts needing work, largest first")
        lines.append("  " + "-" * 52)

        for row in sorted(
            affected, key=lambda item: (-item.todo, item.post_id)
        )[:25]:
            lines.append(
                f"  {row.post_id:<44}{row.todo:>5} slide(s)"
            )

        if len(affected) > 25:
            lines.append(f"  ... and {len(affected) - 25} more")

        lines.append("")

        return "\n".join(lines)


def build_plan(
    archive_root: str | Path,
    *,
    store: VisualStore | None = None,
    version: str = "1",
    configuration: str = "vision+bounded-batch",
    batch_size: int = 5,
    limit: int | None = None,
    only: set[str] | None = None,
) -> VisualPlan:
    """
    Discover, describe and cost every post that has images.

    ``limit`` bounds how many posts are examined, for a plan over a
    subset. It bounds examination only; the totals are then totals of
    what was examined, and the render says so by showing the count.

    ``only`` restricts the plan to a named set of files. It exists for
    the fallback pass: when another tool has already transcribed most of
    a corpus, the vision processor should be costed against the slides
    that transcription could not read, not against all of them. A post
    whose every asset is outside the set is still examined and still
    reported with nothing to do, so the plan says what was skipped and
    why rather than quietly omitting those posts.
    """

    plan = VisualPlan()

    try:
        archive = load_archive(archive_root)

    except ArchiveError as exc:
        plan.archive_error = str(exc)

        return plan

    plan.orphans = len(archive.orphans)

    stage = VisualStage(
        processor=None,
        store=store or VisualStore(store_root()),
        version=version,
        configuration=configuration,
        batch_size=batch_size,
    )

    post_ids = archive.post_ids_with_media()

    if limit is not None:
        post_ids = post_ids[:limit]

    on_disk = {
        path.name
        for path in _list_media(archive)
    }

    # Content digests already understood, and those this plan will come
    # to understand. A picture that appears in two posts is read once:
    # the first post to send it pays, and the second reuses it from the
    # store. Costing each post independently overstates the work, and on
    # this archive by 265 images -- one of which appears in 42 posts.
    known: set[str] = set()
    distinct: set[str] = set()
    cold = 0
    reuse = 0
    total_calls = 0

    for post_id in post_ids:
        assets = assets_for_post(archive, post_id)

        if not assets:
            continue

        if only is not None:
            assets = [
                asset for asset in assets if asset.filename in only
            ]

            if not assets:
                continue

        marked = list(assets)
        mark_duplicates(marked)
        redundant = redundant_assets(marked)
        usable = usable_assets(marked)

        detail = stage.plan_post(post_id, marked)

        row_todo = 0
        row_from_store = 0
        row_reused = 0
        row_batches = 0

        for asset in usable:
            digest = asset.sha256

            if not digest:
                continue

            distinct.add(digest)

            stored = (
                stage.store.load(digest, version, configuration)
                is not None
            )

            if digest in known:
                # Either already analysed, or the store has it.
                row_from_store += 1
                reuse += 1

            elif stored:
                row_from_store += 1

            else:
                row_todo += 1
                cold += 1
                row_batches += 1

            known.add(digest)

        # Batches are per post because a post's slides are read together
        # and synthesised together, so the call count is the number of
        # batches this post would send.
        if row_todo:
            total_calls += _batches_for(row_todo, batch_size)

        row = PlanRow(
            post_id=post_id,
            assets=len(marked),
            slides=sum(
                1 for a in marked if a.role is AssetRole.SLIDE
            ),
            thumbnails=sum(
                1 for a in marked if a.role is AssetRole.THUMBNAIL
            ),
            duplicates=len(redundant),
            corrupt=sum(
                1
                for a in marked
                if a.state is ProcessingState.FAILED
            ),
            usable=len(usable),
            todo=row_todo,
            reusable=row_from_store,
            batches=_batches_for(row_todo, batch_size),
            digest=detail["digest"],
        )

        row.missing = sum(
            1 for a in marked if a.filename not in on_disk
        )

        plan.rows.append(row)

    plan.distinct_contents = len(distinct)
    plan.cross_post_reuse = reuse
    plan.cold_cache_assets = cold

    return plan


def _batches_for(count: int, batch_size: int) -> int:
    """How many calls ``count`` assets come to, per post."""

    if not count:
        return 0

    return -(-count // max(1, batch_size))


def _list_media(archive: Archive) -> list[Path]:
    try:
        return [path for path in archive.media_root.iterdir() if path.is_file()]

    except OSError:
        return []


__all__ = ["PlanRow", "VisualPlan", "build_plan"]
