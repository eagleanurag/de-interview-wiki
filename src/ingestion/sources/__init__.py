"""
Source abstraction for the ingestion layer.

The downstream pipeline must not care where a post came from. Every
source implements the same three-step contract:

    SOURCE
      -> discover()   find content the source makes accessible
      -> normalize()  turn each item into a PostDocument
      -> persist()    hand documents to the importer

Sources are explicitly authorized. Nothing in this package
authenticates on its own initiative, and no source bypasses an access
control: a source that meets a security challenge stops and asks for a
human.
"""

from __future__ import annotations

from src.ingestion.sources.base import (
    CollectedPost,
    CollectionStopped,
    CollectionState,
    SecurityChallenge,
    Source,
    StopReason,
)
from src.ingestion.sources.manual import ManualSource
from src.ingestion.sources.linkedin import (
    LinkedInSource,
    linkedin_installed,
)


__all__ = [
    "CollectedPost",
    "CollectionState",
    "CollectionStopped",
    "LinkedInSource",
    "ManualSource",
    "SecurityChallenge",
    "Source",
    "StopReason",
    "linkedin_installed",
]