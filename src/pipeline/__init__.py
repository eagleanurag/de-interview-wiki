"""
The local knowledge pipeline, and the primary AI orchestrator.

Discover → enrich → aggregate → generate the site.

Each stage writes its own output and the next stage reads only that, so
a stage cannot quietly depend on something an earlier stage held in
memory. A failure in one post costs that post, not the run, because
enrichment is the only stage that touches untrusted model output.

Enrichment is the primary way this project processes material. It runs
locally against OpenCode, retries per post, and records what it did.
GitHub Actions validates the repository and deploys the site; it does
not fan the model out, which an experiment against five hundred real
posts showed to be the least reliable part of the system.

Outputs go under ``build/``, which is git-ignored, so nothing here
writes into the committed repository.
"""

from __future__ import annotations

from src.pipeline.orchestrate import Settled, enrich
from src.pipeline.freshness import (
    ENRICHER_VERSION,
    _refresh_provenance,
    _reusable,
)
from src.pipeline.paths import (
    BUILD_ROOT,
    KNOWLEDGE_BASE,
    RESULTS_DIR,
    RUN_STATE,
    SITE_DIR,
    StageError,
    log,
)
from src.pipeline.stages import aggregate, build_site, discover

__all__ = [
    "BUILD_ROOT",
    "ENRICHER_VERSION",
    "KNOWLEDGE_BASE",
    "RESULTS_DIR",
    "RUN_STATE",
    "SITE_DIR",
    "Settled",
    "StageError",
    "_refresh_provenance",
    "_reusable",
    "aggregate",
    "build_site",
    "discover",
    "enrich",
    "log",
]