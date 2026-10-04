"""
Wiring the visual stage into the orchestrator.

The orchestrator gains three things and changes almost nothing:

* ``visual`` is an optional runner. Absent, the pipeline behaves exactly
  as CP11 left it, which is what lets the visual stage be developed and
  tested without the rest of the system moving.
* ``_visual_for`` decides, per post, what the visual fingerprint is, and
  the cached-result check consults it. A post whose slides changed is
  re-enriched even though its text did not.
* ``_enrich_one`` injects the visual text into the in-memory post before
  the enricher sees it, and records the visual fingerprint beside the
  source one.

The commit message for CP11 said the results are what the site is built
from. That stays true: visual text lands in the worker result, which is
the derived artifact, and ``post.json`` keeps holding only what the
author wrote and the files they attached.
"""

from __future__ import annotations

from src.pipeline.visual import (
    VisualRunner,
    archive_post_id,
    digest_for,
    inject,
)
from src.visual.archive import assets_for_post, resolve_asset_path
from src.visual.models import (
    PROCESSOR_CONFIGURATION,
    PROCESSOR_VERSION,
)
from src.visual.processor import build_default_processor
from src.visual.stage import VisualStage
from src.visual.store import VisualStore, store_root


def make_visual(
    archive_root: str,
    *,
    batch_size: int = 5,
    attempts: int = 3,
    only: set[str] | None = None,
) -> VisualRunner:
    """
    A runner over the archive, with its own processor and store.

    ``only`` narrows the run to a named set of files. It is how the
    fallback pass works: another tool has already transcribed most of
    the corpus, so the vision processor is pointed at the slides that
    transcription could not read rather than at all of them. Left unset
    -- the normal case -- every asset is considered and nothing about
    CP12's behaviour changes.
    """

    return VisualRunner(
        archive_root,
        store=VisualStore(store_root()),
        processor=build_default_processor(),
        batch_size=batch_size,
        attempts=attempts,
        only=only,
    )


__all__ = [
    "PROCESSOR_CONFIGURATION",
    "PROCESSOR_VERSION",
    "VisualRunner",
    "VisualStage",
    "archive_post_id",
    "assets_for_post",
    "digest_for",
    "inject",
    "make_visual",
    "resolve_asset_path",
]
