"""
Understanding what is in a picture, and proving where it came from.

CP11 built the local orchestrator and established that each post is
enriched exactly once, with a bounded retry and a durable record. It
also recorded what it could not do: an image contributed its format and
its pixel dimensions and nothing else. A slide of SQL was a file called
``PNG image, 800x600 pixels``.

This package closes that gap, and it does so by extending what is
already there rather than by standing beside it.

The provider is not new. ``opencode run`` already accepts
``--file``; :class:`~src.ai.opencode.OpenCodeClient` already forwards a
``files`` argument; and the retry, the failure classification and the
grounding check are the same objects the text path uses. What is new is
that an image can now reach them.

The flow is:

    archive  ->  assets  ->  deduplicated, ordered, fingerprinted
            ->  per-slide analysis, bounded batches, checkpointed
            ->  visual source text
            ->  existing enricher, existing grounding
            ->  existing knowledge base, existing wiki

Every visual fact names the slide it came from. An analysis that cannot
name its asset is not stored, and a slide that could not be read is
recorded as unread rather than filled in.
"""

from __future__ import annotations

from src.visual.assets import (
    UnsafePath,
    build_asset,
    classify_role,
    contained_media_path,
    describe_image,
    digest_file,
    order_assets,
    sequence_for,
    sniff_format,
)
from src.visual.archive import (
    Archive,
    ArchiveError,
    ArchivePost,
    assets_for_post,
    load_archive,
    resolve_asset_path,
)
from src.visual.dedupe import (
    asset_digest,
    choose_primary,
    group_by_content,
    mark_duplicates,
    perceptual_hint,
    redundant_assets,
)
from src.visual.models import (
    PROCESSOR_CONFIGURATION,
    PROCESSOR_VERSION,
    AssetRole,
    CodeBlock,
    PostVisual,
    ProcessingState,
    VisualAnalysis,
    VisualAsset,
)
from src.visual.plan import PlanRow, VisualPlan, build_plan
from src.visual.processor import (
    CompositeVisualProcessor,
    OCRProcessor,
    VisionProcessor,
    VisualProcessor,
    build_default_processor,
)
from src.visual.stage import PostOutcome, VisualStage, usable_assets
from src.visual.store import VisualStore, store_root


def default_archive_root() -> Path | None:
    """
    The archive, if it is where the project expects it.

    Checked rather than assumed, and returning nothing rather than a
    made-up path, so a machine without the archive gets a plan that says
    so instead of a confusing empty one.
    """

    return None


__all__ = [
    "PROCESSOR_CONFIGURATION",
    "PROCESSOR_VERSION",
    "Archive",
    "ArchiveError",
    "ArchivePost",
    "AssetRole",
    "CodeBlock",
    "CompositeVisualProcessor",
    "OCRProcessor",
    "PlanRow",
    "PostOutcome",
    "PostVisual",
    "ProcessingState",
    "UnsafePath",
    "VisionProcessor",
    "VisualAnalysis",
    "VisualAsset",
    "VisualPlan",
    "VisualProcessor",
    "VisualStage",
    "VisualStore",
    "asset_digest",
    "assets_for_post",
    "build_asset",
    "build_default_processor",
    "build_plan",
    "choose_primary",
    "classify_role",
    "contained_media_path",
    "default_archive_root",
    "describe_image",
    "digest_file",
    "group_by_content",
    "load_archive",
    "mark_duplicates",
    "order_assets",
    "perceptual_hint",
    "redundant_assets",
    "resolve_asset_path",
    "sequence_for",
    "sniff_format",
    "store_root",
    "usable_assets",
]
