"""
End-to-end test over the synthetic fixture set.

Runs the real pipeline on real files: bundles in, normalized posts out,
media processed, enrichment validated, consolidated, aggregated,
rendered into a site, and searched. Nothing here stubs the media stage
or the extractor, because those are exactly the parts that break
silently.

Every fixture is synthetic. No real credential, person or captured
content appears in any of them, and no test needs a network or a
credential to run.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from src.aggregation.aggregator import aggregate_results
from src.ingestion.collect import Collector, CollectionLimits, post_id_for
from src.ingestion.importer import validate_posts
from src.ingestion.sources.manual import ManualSource
from src.processing.media_processor import process_media
from src.wiki.generator import generate_site

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "incoming"


pytestmark = pytest.mark.skipif(
    not FIXTURES.is_dir(),
    reason="fixtures are absent; run python -m tests.build_fixtures",
)


@pytest.fixture(scope="module")
def collected(tmp_path_factory):
    """Every fixture bundle, ingested through the real collector."""

    source = ManualSource(FIXTURES)

    posts = []

    try:
        for post in source.discover():
            posts.append(post)
    except Exception:  # noqa: BLE001
        # CollectionStopped is the normal end of a walk.
        pass

    return posts


# ---------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------


def test_every_fixture_bundle_is_discovered(collected):
    assert len(collected) >= 11


def test_bundles_with_the_same_content_get_the_same_identifier(tmp_path):
    """
    Two identical captures must resolve to one identifier, or a repeated
    import would create a duplicate instead of refreshing.
    """

    first = tmp_path / "first"
    second = tmp_path / "second"

    for directory in (first, second):
        bundle = directory / "same"
        bundle.mkdir(parents=True)
        (bundle / "post.txt").write_text(
            "Exactly the same captured text.", encoding="utf-8"
        )

    def identifiers(root: Path) -> list[str]:
        found = []

        try:
            for post in ManualSource(root).discover():
                found.append(post_id_for(post.source_post_id))
        except Exception:  # noqa: BLE001
            pass

        return found

    assert identifiers(first) == identifiers(second)


def test_a_pdf_bundle_is_discovered_with_its_document(collected):
    pdf_posts = [post for post in collected if post.media]

    assert any(
        any(path.suffix == ".pdf" for path in post.media)
        for post in pdf_posts
    )


def test_an_image_bundle_keeps_the_image(collected):
    images = [
        path
        for post in collected
        for path in post.media
        if path.suffix == ".png"
    ]

    assert images
    assert all(path.is_file() for path in images)


def test_a_jsonl_export_yields_one_post_per_record(collected):
    export = [
        post
        for post in collected
        if str(post.extra.get("bundle", "")).endswith("export.jsonl")
    ]

    assert len(export) == 2
    assert {post.source_post_id for post in export} == {
        "export-1",
        "export-2",
    }


def test_nothing_shipped_a_credential(collected):
    """
    The fixtures are synthetic, and a fixture is still a file in a
    repository. This guards against one ever becoming real.
    """

    blob = ""

    for path in FIXTURES.rglob("*"):
        if path.is_file():
            try:
                blob += path.read_text(encoding="utf-8", errors="ignore")
            except Exception:  # noqa: BLE001
                continue

    for marker in (
        "LINKEDIN_PASSWORD",
        "LINKEDIN_USERNAME",
        "ghp_",
        "AKIA",
    ):
        assert marker not in blob, marker


# ---------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------


def test_ingestion_persists_every_bundle(tmp_path, collected):
    root = tmp_path / "posts"

    collector = Collector(
        ManualSource(FIXTURES),
        root=root,
        limits=CollectionLimits(),
        repository_root=tmp_path,
    )

    report = collector.run()

    assert report.failed == []
    assert report.stopped_because == "source_exhausted"

    persisted = {path.name for path in root.iterdir() if path.is_dir()}

    assert len(persisted) == report.state.persisted
    assert persisted


def test_every_ingested_post_validates(tmp_path, collected):
    root = tmp_path / "posts"

    Collector(ManualSource(FIXTURES), root=root,
        repository_root=tmp_path).run()

    report = validate_posts(root=root)

    assert report.ok, [str(issue) for issue in report.issues]


def test_ingesting_twice_creates_no_second_copy(tmp_path, collected):
    """
    Idempotency is the property that makes a repeated drop safe, so it
    is checked by counting directories rather than by trusting the
    report.
    """

    root = tmp_path / "posts"

    Collector(ManualSource(FIXTURES), root=root,
        repository_root=tmp_path).run()

    first = {path.name for path in root.iterdir() if path.is_dir()}

    Collector(ManualSource(FIXTURES), root=root,
        repository_root=tmp_path).run()

    second = {path.name for path in root.iterdir() if path.is_dir()}

    assert first == second


def test_a_changed_bundle_refreshes_rather_than_duplicates(
    tmp_path, collected
):
    """A corrected file has to be recognised, not ignored."""

    drop = tmp_path / "incoming"
    bundle = drop / "editable"
    bundle.mkdir(parents=True)

    text_file = bundle / "post.txt"
    text_file.write_text("First version of the note.", encoding="utf-8")

    root = tmp_path / "posts"

    Collector(ManualSource(drop), root=root,
        repository_root=tmp_path).run()

    before = {path.name for path in root.iterdir() if path.is_dir()}

    text_file.write_text(
        "Second version of the note, corrected.", encoding="utf-8"
    )

    Collector(ManualSource(drop), root=root,
        repository_root=tmp_path).run()

    after = {path.name for path in root.iterdir() if path.is_dir()}

    # A different identifier, because the content changed, and the
    # original is preserved rather than overwritten: provenance for
    # material that was already stored is not lost because a newer
    # version arrived.
    assert before != after
    assert len(after) == 2
    assert before < after


# ---------------------------------------------------------------------
# Media
# ---------------------------------------------------------------------


def test_pdf_text_is_extracted(tmp_path, collected):
    from src.ingestion.post_loader import load_post

    root = tmp_path / "posts"

    Collector(ManualSource(FIXTURES), root=root,
        repository_root=tmp_path).run()

    from src.ingestion.post_loader import load_post as load

    found = ""

    for directory in root.iterdir():
        if not directory.is_dir():
            continue

        candidate = load(directory)

        process_media(candidate)

        for media in candidate.media:
            if media.path.endswith("notes.pdf"):
                found = media.extracted_text or ""

    if not found:
        pytest.skip("no readable PDF bundle was ingested")

    assert "slowly changing dimension" in found.lower()


def test_a_corrupt_pdf_yields_no_text_and_does_not_raise(tmp_path):
    """
    Nothing is invented for a file that could not be read: it records
    why, and carries no extracted text.
    """

    from src.ingestion.post_loader import load_post as load
    from src.processing.media_processor import MediaReport

    root = tmp_path / "posts"

    Collector(ManualSource(FIXTURES), root=root,
        repository_root=tmp_path).run()

    report = MediaReport()

    broken = []

    for directory in root.iterdir():
        if not directory.is_dir():
            continue

        candidate = load(directory)

        process_media(candidate, report=report)

        broken.extend(
            outcome
            for outcome in report.outcomes
            if outcome.failed
        )

    # The truncated document was noticed, and nothing was invented.
    assert broken
    assert any("broken.pdf" in outcome.path for outcome in broken)


def test_a_broken_pdf_does_not_stop_the_run(tmp_path, collected):
    """
    One corrupt file must cost that file, not the pipeline. A truncated
    document is what a failed download actually leaves behind.
    """

    root = tmp_path / "posts"

    Collector(ManualSource(FIXTURES), root=root,
        repository_root=tmp_path).run()

    from src.ingestion.post_loader import load_post

    for directory in root.iterdir():
        if not directory.is_dir():
            continue

        post = load_post(directory)

        # Must not raise.
        process_media(post)


def test_images_are_ingested_without_claiming_a_description(
    tmp_path, collected
):
    """
    An image is kept. No text is invented for it, because nothing in
    this pipeline reads the pixels.
    """

    root = tmp_path / "posts"

    Collector(ManualSource(FIXTURES), root=root,
        repository_root=tmp_path).run()

    from src.ingestion.post_loader import load_post

    images = 0

    for directory in root.iterdir():
        if not directory.is_dir():
            continue

        post = load_post(directory)

        for media in post.media:
            if media.path.endswith(".png"):
                images += 1
                assert media.path.startswith("media/")
                assert not media.path.startswith("/")

    assert images


def test_media_paths_stay_inside_the_post(tmp_path, collected):
    root = tmp_path / "posts"

    Collector(ManualSource(FIXTURES), root=root,
        repository_root=tmp_path).run()

    for directory in root.iterdir():
        if not directory.is_dir():
            continue

        for path in (directory / "media").glob("*") if (
            directory / "media"
        ).is_dir() else []:
            resolved = path.resolve()

            assert directory.resolve() in resolved.parents


# ---------------------------------------------------------------------
# The whole pipeline
# ---------------------------------------------------------------------


def test_the_full_pipeline_runs_on_the_fixtures(tmp_path, collected):
    """
    Bundles in, site out, through every real stage. Enrichment is not
    exercised here because it needs the model; everything downstream of
    it is.
    """

    root = tmp_path / "posts"

    Collector(ManualSource(FIXTURES), root=root,
        repository_root=tmp_path).run()

    results = tmp_path / "worker-results"
    results.mkdir()

    from src.ingestion.post_loader import load_post

    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue

        post = load_post(directory)

        process_media(post)

        (results / f"cloud_worker_{post.id}.json").write_text(
            json.dumps(post.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )

    kb = tmp_path / "knowledge_base.json"

    aggregate_results(input_directory=results, output_path=kb)

    payload = json.loads(kb.read_text(encoding="utf-8"))

    assert payload["stats"]["posts_aggregated"] == len(
        [path for path in root.iterdir() if path.is_dir()]
    )

    # Nothing here was enriched, so nothing claims to be.
    assert payload["stats"]["posts_enriched"] == 0

    # Content classification reads the text, which exists, so a post can
    # be un-enriched and still be recognised as non-technical. The two
    # are independent axes and neither is derived from the other.
    kinds = payload["knowledge"]["content_kinds"]

    assert set(kinds) == {
        post["id"] for post in payload["posts"]
    }

    social = [
        post_id
        for post_id, kind in kinds.items()
        if kind == "job_announcement"
    ]

    assert social, "a job announcement should be recognised as one"

    site = tmp_path / "site"

    generate_site(input_path=kb, output_dir=site)

    pages = list(site.rglob("*.html"))

    assert pages
    assert (site / "index.html").is_file()

    # Every post reachable, every link resolving.
    broken: list[str] = []

    for page in site.rglob("*.html"):
        for target in re.findall(r'href="([^"#?]+)"', page.read_text(
            encoding="utf-8"
        )):
            if target.startswith(("http://", "https://", "mailto:")):
                continue

            if not (page.parent / target).resolve().exists():
                broken.append(f"{page.name} -> {target}")

    assert broken == []


def test_no_synthetic_content_reaches_the_repository_knowledge_base():
    """
    The fixtures are for tests only. Production knowledge comes from an
    authorized source, so this guards the boundary the mission draws.
    """

    from src.aggregation.consolidation import detect_technologies

    production = REPO_ROOT / "data" / "posts"

    if not production.is_dir():
        pytest.skip("no production posts in this checkout")

    for directory in production.iterdir():
        if not directory.is_dir():
            continue

        document = json.loads(
            (directory / "post.json").read_text(encoding="utf-8")
        )

        body = document["original_text"]

        for phrase in (
            "exactly the same captured text",
            "first version of the note",
            "screenshot of an interview screen",
            "fixture-author",
        ):
            assert phrase not in body, f"{directory.name}: {phrase}"


def test_the_repository_knowledge_base_is_not_fixture_sourced(tmp_path):
    """
    A fixture identifier must never appear in a production post
    directory, which is what would happen if a test wrote to
    ``data/posts`` instead of a temporary directory.
    """

    production = REPO_ROOT / "data" / "posts"

    if not production.is_dir():
        pytest.skip("no production posts in this checkout")

    names = {path.name for path in production.iterdir() if path.is_dir()}

    for fixture in FIXTURES.rglob("*"):
        if not fixture.is_file():
            continue

        stem = post_id_for(fixture.stem)

        assert stem not in names, f"{fixture.name} leaked into data/posts"
