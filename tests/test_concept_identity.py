"""
Concept identity between the knowledge base and the site.

Found against a 490-post corpus: the site re-derived concepts from the
posts on their own grouping rule instead of the one the aggregator used
to decide what a concept is. One concept whose two posts spelled it
differently became two pages, and each page listed only the post that
spelled it that way -- so a reader following a concept saw fewer sources
than the knowledge base attributed to it.

These build a small corpus where two posts mean the same concept and
assert the two never disagree about it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.test_wiki_generator import make_post


@pytest.fixture
def corpus(tmp_path: Path):
    """A canonical file where two posts spell one concept two ways."""

    posts = [
        make_post(
            "urn:li:activity:1",
            concepts=["Delta Lake", "time travel"],
        ),
        make_post(
            "urn:li:activity:2",
            # Same concept, different spelling.
            concepts=["delta lake", "Time Travel"],
        ),
    ]

    source = tmp_path / "kb"
    source.mkdir()

    target = source / "knowledge_base.json"

    target.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "generated_at": "2026-10-02T00:00:00+00:00",
                "stats": {"posts_aggregated": len(posts)},
                "posts": posts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return target, tmp_path / "site"


def _build(canonical_file: Path, output: Path) -> Path:
    from src.wiki.generator import generate_site

    generate_site(input_path=canonical_file, output_dir=output)

    return output


def _concepts(site: Path) -> list[str]:
    directory = site / "concepts"

    return sorted(path.stem for path in directory.glob("*.html"))


def test_two_spellings_of_one_concept_make_one_page(corpus):
    canonical_file, output = corpus

    site = _build(canonical_file, output)

    pages = _concepts(site)

    # One concept, one page. Two spellings are one concept, because that
    # is what the knowledge base says and a page is not a second
    # concept.
    assert len([p for p in pages if "delta" in p]) == 1, pages
    assert len([p for p in pages if "travel" in p]) == 1, pages


def test_that_page_lists_both_posts(corpus):
    canonical_file, output = corpus

    site = _build(canonical_file, output)

    page = next(
        (site / "concepts").glob("*delta*")
    ).read_text(encoding="utf-8")

    # Both sources, because both posts are about it. A page listing one
    # would be a page that hides half its own provenance.
    assert "urn-li-activity-1" in page or "activity_1" in page
    assert "urn-li-activity-2" in page or "activity_2" in page


def test_the_site_and_the_knowledge_base_agree_how_many(corpus):
    from src.aggregation.aggregator import aggregate_results

    canonical_file, output = corpus

    # Aggregate the same posts the way the pipeline does, and compare.
    staged = output.parent / "worker-results"
    staged.mkdir()

    for post in json.loads(canonical_file.read_text(encoding="utf-8"))[
        "posts"
    ]:
        # The aggregator globs cloud_worker_*.json, so only the prefix
        # matters. A post id contains colons, which Windows will not
        # accept in a file name.
        safe = post["id"].replace(":", "-")

        (staged / f"cloud_worker_{safe}.json").write_text(
            json.dumps(post), encoding="utf-8"
        )

    aggregate_results(
        input_directory=staged,
        output_path=output.parent / "kb.json",
    )

    kb = json.loads(
        (output.parent / "kb.json").read_text(encoding="utf-8")
    )

    site = _build(canonical_file, output)

    assert len(_concepts(site)) == kb["stats"]["concepts_consolidated"]


def test_a_long_concept_is_reachable_by_its_stored_slug(corpus):
    canonical_file, output = corpus

    long_name = (
        "Azure Data Factory data flows, incremental loading, "
        "triggers, and scheduling for a medallion pipeline"
    )

    payload = json.loads(canonical_file.read_text(encoding="utf-8"))

    for post in payload["posts"]:
        post["ai_analysis"]["concepts"] = [long_name]

    canonical_file.write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    site = _build(canonical_file, output)

    from src.aggregation.consolidation import _slug

    # The knowledge base records this slug; the site must have written
    # a page at exactly that address, or a consumer following the
    # knowledge base arrives nowhere.
    assert (site / "concepts" / f"{_slug(long_name)}.html").is_file()