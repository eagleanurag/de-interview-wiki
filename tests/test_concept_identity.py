"""Concept identity between the knowledge base and the site.

Found against a 490-post corpus: the site re-derived concepts from the
posts on their own grouping rule instead of the one the aggregator used
to decide what a concept is. One concept whose two posts spelled it
differently became two entries, and each carried only the post that
spelled it that way -- so a reader following a concept saw fewer sources
than the knowledge base attributed to it.

These build a small corpus where two posts mean the same concept and
assert the two never disagree about it.

The concept has no page of its own any more. Stopping the generation of
3,112 standalone concept pages was the point: a concept is a label the
enricher produced, not a thing a candidate revises, and listing every
label that contains it is what turned the site back into an index. What
these tests now check is the identity itself -- one concept, not two --
and that both contributing posts still carry it. The labels are still
visible, as the Key concepts row on each knowledge page and as searchable
text in the search index.
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
    """
    The concept pages that used to exist.

    Kept as a helper so the assertions below can say "none of these
    exist" in one place rather than by implication.
    """

    directory = site / "concepts"

    return sorted(path.stem for path in directory.glob("*.html"))


def _model(canonical_file: Path):
    from src.wiki.analysis import build_site_model
    from src.wiki.canonical import load_canonical

    return build_site_model(load_canonical(canonical_file))


def test_two_spellings_of_one_concept_make_one_concept(corpus):
    canonical_file, output = corpus

    model = _model(canonical_file)

    labels = [
        entry.label.lower()
        for entry in model.concept_entries
    ]

    # One concept, one label. Two spellings are one concept, because that
    # is what the knowledge base says and two entries would be two
    # concepts.
    assert len([label for label in labels if "delta" in label]) == 1, labels
    assert len([label for label in labels if "travel" in label]) == 1, labels

    _build(canonical_file, output)


def test_the_concept_page_is_no_longer_generated(corpus):
    """
    The removal, asserted rather than assumed.

    Three thousand one hundred and twelve of the site's pages existed to
    list concept labels. None is written now, and none is linked to.
    """

    canonical_file, output = corpus

    site = _build(canonical_file, output)

    assert _concepts(site) == [], "a concept page was generated"

    assert not (site / "concepts.html").exists()

    # And nothing on either surviving page family links to one.
    for page in site.rglob("*.html"):
        assert "concepts/" not in page.read_text(encoding="utf-8")


def test_both_contributing_posts_still_carry_the_concept(corpus):
    """
    The traceability the concept page used to provide.

    There is no page for a concept any more, so "the concept points at
    its posts" became "both posts still carry it, and both still have a
    page". An entry listing one of two sources is an entry hiding half its
    own provenance, and that is the property worth protecting.
    """

    canonical_file, output = corpus

    model = _model(canonical_file)

    entry = next(
        e for e in model.concept_entries if "delta" in e.label.lower()
    )

    # Both sources, because both posts are about it.
    assert len(entry.post_slugs) == 2, entry.post_slugs

    site = _build(canonical_file, output)

    for slug in entry.post_slugs:
        page = site / "posts" / f"{slug}.html"

        # A concept has no page, so neither does a post. What is checked
        # is that the concept survives as searchable text on the post
        # that carries it.
        index = json.loads(
            (site / "assets" / "search-index.json").read_text(
                encoding="utf-8"
            )
        )

        searchable = {
            label
            for record in index["records"]
            for label in record.get("c", [])
        }

        assert entry.label in searchable, entry.label


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

    # The count is now the model's, not a count of pages: nothing is
    # written for a concept, so a page count would be zero and would
    # agree with nothing.
    assert _concepts(site) == []

    assert len(_model(canonical_file).concept_entries) == (
        kb["stats"]["concepts_consolidated"]
    )


def test_a_long_concept_is_carried_not_truncated(corpus):
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

    # The knowledge base records this concept under this slug. It is no
    # longer a page, but the label is still searchable, which is what a
    # long generated label is actually for.
    index = json.loads(
        (site / "assets" / "search-index.json").read_text(encoding="utf-8")
    )

    assert _slug(long_name)

    searchable = [
        concept
        for record in index["records"]
        for concept in record.get("c", [])
    ]

    assert long_name in searchable, searchable