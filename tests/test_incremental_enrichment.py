"""
Tests for incremental enrichment.

Re-enriching everything on every run is the obvious implementation and
the wrong one: it calls a model once per post on every invocation, so
the cost of the pipeline grows with how often it is run rather than
with how much changed.

These cover the three cases that matter: an unchanged post is reused, a
post whose content changed is enriched again, and a change to the
enrichment contract invalidates every result because the current
version would answer differently.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.ingestion.post_document import PostDocument
from src.ingestion.post_loader import load_post, source_digest
from src.models import (
    AIAnalysis,
    Classification,
    EnrichmentFingerprint,
    KnowledgePost,
    SourceInfo,
)


def make_post(text: str = "Spark partitioning prunes files.") -> KnowledgePost:
    return KnowledgePost(
        id="sample_x",
        source=SourceInfo(
            platform="manual",
            captured_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
        original_text=text,
    )


# ---------------------------------------------------------------------
# The digest
# ---------------------------------------------------------------------


def test_the_digest_is_stable_for_unchanged_content():
    post = make_post()

    assert source_digest(post) == source_digest(post)


def test_the_digest_changes_with_the_text():
    before = source_digest(make_post("First version."))
    after = source_digest(make_post("Second version."))

    assert before != after


def test_the_digest_ignores_whitespace_at_the_edges():
    """
    Trailing whitespace is not a content change. Re-enriching over it
    would spend a model call to get the same answer.
    """

    assert source_digest(make_post("Body.")) == source_digest(
        make_post("  Body.  \n\n")
    )


def test_the_digest_changes_with_extracted_media_text():
    """
    A document that now yields different text is new evidence, and the
    enrichment that described the old text no longer describes this
    post.
    """

    post = make_post()
    post.media.append(
        __import__("src.models", fromlist=["MediaItem"]).MediaItem(
            type="pdf",
            path="media/notes.pdf",
            extracted_text="First extraction.",
        )
    )

    before = source_digest(post)

    post.media[0].extracted_text = "A different extraction."

    assert source_digest(post) != before


def test_the_digest_does_not_depend_on_media_order():
    from src.models import MediaItem

    first = make_post()
    first.media = [
        MediaItem(type="pdf", path="media/a.pdf", extracted_text="A"),
        MediaItem(type="pdf", path="media/b.pdf", extracted_text="B"),
    ]

    second = make_post()
    second.media = [
        MediaItem(type="pdf", path="media/b.pdf", extracted_text="B"),
        MediaItem(type="pdf", path="media/a.pdf", extracted_text="A"),
    ]

    assert source_digest(first) == source_digest(second)


def test_the_digest_separates_differently_named_files():
    """
    Two files with identical text are not the same evidence, because
    they are different documents.
    """

    from src.models import MediaItem

    first = make_post()
    first.media = [
        MediaItem(type="pdf", path="media/a.pdf", extracted_text="Same.")
    ]

    second = make_post()
    second.media = [
        MediaItem(type="pdf", path="media/b.pdf", extracted_text="Same.")
    ]

    assert source_digest(first) != source_digest(second)


# ---------------------------------------------------------------------
# The fingerprint on a document
# ---------------------------------------------------------------------


def test_a_new_post_always_needs_enrichment():
    document = PostDocument.new("x", text="Body")

    assert document.needs_enrichment(
        source_digest="abc", enricher_version="1"
    )


def test_an_unchanged_post_does_not_need_enrichment():
    document = PostDocument.new("x", text="Body")

    document.mark_enriched(source_digest="abc", enricher_version="1")

    assert not document.needs_enrichment(
        source_digest="abc", enricher_version="1"
    )


def test_changed_content_needs_enrichment():
    document = PostDocument.new("x", text="Body")

    document.mark_enriched(source_digest="abc", enricher_version="1")

    assert document.needs_enrichment(
        source_digest="xyz", enricher_version="1"
    )


def test_a_new_enricher_version_needs_enrichment():
    """
    An analysis produced under an older contract does not describe what
    the current one would say, so it is not reusable.
    """

    document = PostDocument.new("x", text="Body")

    document.mark_enriched(source_digest="abc", enricher_version="1")

    assert document.needs_enrichment(
        source_digest="abc", enricher_version="2"
    )


def test_the_fingerprint_records_when_enrichment_happened():
    document = PostDocument.new("x", text="Body")

    document.mark_enriched(
        source_digest="abc", enricher_version="1", enriched_at="2026-01-01"
    )

    fingerprint = document.enrichment_fingerprint()

    assert fingerprint["source_digest"] == "abc"
    assert fingerprint["enricher_version"] == "1"
    assert fingerprint["enriched_at"] == "2026-01-01"


def test_a_document_without_a_fingerprint_reports_nothing():
    """
    A document written before the fingerprint existed has none, and must
    not fail when asked for one.
    """

    document = PostDocument.new("x", text="Body")
    document.data.pop("enrichment", None)

    assert document.enrichment_fingerprint() == {}
    assert document.needs_enrichment(
        source_digest="abc", enricher_version="1"
    )


# ---------------------------------------------------------------------
# The fingerprint through the loader
# ---------------------------------------------------------------------


def test_the_fingerprint_survives_a_load(tmp_path):
    directory = tmp_path / "post"
    directory.mkdir()

    document = PostDocument.new("post", text="Body")
    document.mark_enriched(
        source_digest="abc", enricher_version="1", enriched_at="2026-01-01"
    )
    document.save(directory)

    loaded = load_post(directory)

    assert loaded.enrichment.source_digest == "abc"
    assert loaded.enrichment.enricher_version == "1"


def test_the_fingerprint_is_not_published(tmp_path):
    """
    It records how this pipeline processed a post, not what the post
    says, so it must not reach the canonical knowledge base.
    """

    post = make_post()
    post.enrichment = EnrichmentFingerprint(
        source_digest="abc", enricher_version="1"
    )

    payload = post.model_dump(mode="json")

    assert "enrichment" not in payload
    assert "directory" not in payload


# ---------------------------------------------------------------------
# Reuse through the pipeline
# ---------------------------------------------------------------------


def test_a_matching_result_is_reused():
    from src.pipeline import ENRICHER_VERSION, _reusable

    target = Path("build/worker-results/cloud_worker_sample_001.json")
    target.parent.mkdir(parents=True, exist_ok=True)

    target.write_text(
        json.dumps(
            {
                "id": "sample_001",
                "_enrichment": {
                    "source_digest": "abc",
                    "enricher_version": ENRICHER_VERSION,
                },
            }
        ),
        encoding="utf-8",
    )

    assert _reusable(target, "abc") is not None


def test_a_result_for_changed_content_is_not_reused():
    from src.pipeline import ENRICHER_VERSION, _reusable

    target = Path("build/worker-results/cloud_worker_sample_002.json")
    target.parent.mkdir(parents=True, exist_ok=True)

    target.write_text(
        json.dumps(
            {
                "id": "sample_002",
                "_enrichment": {
                    "source_digest": "abc",
                    "enricher_version": ENRICHER_VERSION,
                },
            }
        ),
        encoding="utf-8",
    )

    assert _reusable(target, "changed") is None


def test_a_result_from_an_older_version_is_not_reused():
    from src.pipeline import ENRICHER_VERSION, _reusable

    target = Path("build/worker-results/cloud_worker_sample_003.json")
    target.parent.mkdir(parents=True, exist_ok=True)

    target.write_text(
        json.dumps(
            {
                "id": "sample_003",
                "_enrichment": {
                    "source_digest": "abc",
                    "enricher_version": "ancient",
                },
            }
        ),
        encoding="utf-8",
    )

    assert _reusable(target, "abc") is None


def test_an_unreadable_result_is_not_reused():
    """
    A truncated result from an interrupted run must be redone rather
    than trusted.
    """

    from src.pipeline import _reusable

    target = Path("build/worker-results/cloud_worker_broken.json")
    target.parent.mkdir(parents=True, exist_ok=True)

    target.write_text("{ truncated", encoding="utf-8")

    assert _reusable(target, "abc") is None


def test_a_result_with_no_fingerprint_is_not_reused():
    """
    A result written before fingerprints existed cannot be shown to
    match the current content, so it is redone.
    """

    from src.pipeline import _reusable

    target = Path("build/worker-results/cloud_worker_legacy.json")
    target.parent.mkdir(parents=True, exist_ok=True)

    target.write_text(
        json.dumps({"id": "legacy"}), encoding="utf-8"
    )

    assert _reusable(target, "abc") is None