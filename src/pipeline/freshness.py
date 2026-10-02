"""
Whether an existing enrichment result still describes its post.

Two decisions live here and nothing else does:

* **is this result reusable?** Its recorded fingerprint is compared with
  the post's current content digest, and the enricher version has to
  match as well. A result from an older contract is not reusable even
  when the analysis is largely right, because the whole point of the
  change was that the knowledge base must never hold something the
  current rules would have removed.
* **what has gone stale in a reused result?** Only the attribution --
  where the content came from and what files came with it. The analysis,
  the questions and the classification are exactly what the model was
  paid for and nothing about them has gone out of date, so replacing
  them would spend real time to arrive at the same answer.

A field the post has stopped claiming is removed rather than left, so a
post that has stopped claiming to have been captured stops claiming it.
"""

from __future__ import annotations

import json
from pathlib import Path


#: Bumped when the enrichment contract or the prompt changes.
#:
#: Version 3 added source grounding, so a result from version 2 may
#: contain a question about a technology the source never mentioned.
#: Those results are not reusable even though the underlying analysis is
#: largely right: the point of the check is that the knowledge base never
#: holds one, and a cached result would put it straight back.
ENRICHER_VERSION = "3"


def _reusable(target: Path, digest: str) -> dict | None:
    """
    An existing worker result that still matches the current content.

    Read from the file rather than trusted from a stamp, because the
    stamp lives in the post and the result lives here, and the two can
    disagree if a run was interrupted between them.
    """

    try:
        payload = json.loads(target.read_text(encoding="utf-8"))

    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None

    recorded = payload.get("_enrichment") or {}

    if not isinstance(recorded, dict):
        return None

    if recorded.get("source_digest") != digest:
        return None

    if recorded.get("enricher_version") != ENRICHER_VERSION:
        return None

    return payload


def _refresh_provenance(
    payload: dict,
    post: object,
) -> dict:
    """
    Copy the current post's attribution onto a reused worker result.

    Only the fields that say where the content came from and what it
    looks like are replaced. The analysis, the questions and the
    classification are left exactly as they were, because they are what
    the model was paid for and nothing about them has gone stale.
    """

    current = post.model_dump(mode="json")

    for key in ("source", "media"):
        value = current.get(key)

        if value is None:
            payload.pop(key, None)
        else:
            payload[key] = value

    if current.get("saved_item") is None:
        payload.pop("saved_item", None)
    else:
        payload["saved_item"] = current["saved_item"]

    payload["original_text"] = current.get("original_text", "")

    return payload


__all__ = ["ENRICHER_VERSION", "_refresh_provenance", "_reusable"]