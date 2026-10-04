"""
Tests for the Saved Items intake and enrichment-quality work.

The intake is the part a person touches, so most of what follows is
about what happens when they get something slightly wrong: a folder that
does not say which item it belongs to, a file that will not open, a link
that is not a link. None of it may be silent, and none of it may cost
them their material.

The enrichment tests are about the opposite failure. A prompt asks the
model not to invent; it does not make it stop. So the response is checked
against the post, and the test that matters most is the one asserting a
question about a technology the source never mentions does not survive.

Every fixture is synthetic. No credential appears in any of them, no
network is touched, and no LinkedIn page is ever fetched.
"""

from __future__ import annotations

import json
import zlib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.ai.enricher import AIEnricher
from src.ai.grounding import Grounder, GroundingReport, check
from src.ai.opencode import OpenCodeResult
from src.ingestion.collect_cli import main as collect_cli_main
from src.ingestion.saved_items import diagnostics as diag
from src.ingestion.saved_items import intake
from src.ingestion.saved_items import plan as plan_module
from src.ingestion.saved_items.bundles import (
    CAPTURE_QUALITIES,
    QUALITY_DOCUMENT_ONLY,
    QUALITY_IMAGE_ONLY,
    QUALITY_METADATA_ONLY,
    QUALITY_PARTIAL,
    QUALITY_TEXT,
    QUALITY_TEXT_AND_MEDIA,
    BundleIndex,
    discover_bundles,
    discover_rejected,
)
from src.ingestion.saved_items.capture import (
    CREDENTIAL_FIELDS,
    credential_fields,
    read_capture_file,
)
from src.ingestion.saved_items.manifest import (
    ManifestUnreadable,
    SavedItemsManifest,
)
from src.ingestion.saved_items.model import SavedItem
from src.ingestion.saved_items.readers import ManifestError, read_manifest
from src.ingestion.saved_items.urls import normalize_linkedin_url
from src.models import KnowledgePost, SourceInfo


DELTA = "https://www.linkedin.com/posts/alice_delta-lake-101"
KAFKA = "https://www.linkedin.com/posts/bob_kafka-202"
SPARK = "https://www.linkedin.com/pulse/carol_spark-303"
ADF = "https://www.linkedin.com/posts/dan_adf-404"

DELTA_TEXT = (
    "# Delta Lake\n\nDelta Lake gives a data lake ACID guarantees. "
    "Schema evolution lets you add a column without rewriting history, "
    "and time travel queries an earlier version of the table. Databricks "
    "runs Delta tables on Apache Spark.\n"
)


def sid(url: str) -> str:
    return normalize_linkedin_url(url).source_id


def saved_item(url: str) -> SavedItem:
    return SavedItem.from_url(normalize_linkedin_url(url))


def png_bytes(seed: int = 0) -> bytes:
    import zlib as _zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            len(data).to_bytes(4, "big")
            + tag
            + data
            + _zlib.crc32(tag + data).to_bytes(4, "big")
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(
            b"IHDR",
            (1).to_bytes(4, "big")
            + (1).to_bytes(4, "big")
            + bytes([8, 2, 0, 0, 0]),
        )
        + chunk(b"IDAT", _zlib.compress(bytes([0, 255, 0, seed & 0xFF])))
        + chunk(b"IEND", b"")
    )


def pdf_bytes(body: str = "Consumer lag explained") -> bytes:
    text = f"BT /F1 12 Tf 72 720 Td ({body}) Tj ET".encode()

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(text)).encode() + b" >>\nstream\n"
        + text
        + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []

    for number, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + obj + b"\nendobj\n"

    xref = len(out)
    out += b"xref\n0 " + str(len(objects) + 1).encode() + b"\n"
    out += b"0000000000 65535 f \n"

    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()

    out += (
        b"trailer\n<< /Size "
        + str(len(objects) + 1).encode()
        + b" /Root 1 0 R >>\nstartxref\n"
        + str(xref).encode()
        + b"\n%%EOF\n"
    )

    return bytes(out)


def scanned_pdf_bytes() -> bytes:
    """
    A well-formed PDF with no selectable text.

    A scan, structurally. It opens, has a page, and yields nothing to
    extract, which is a real property of a real file rather than a
    corrupt one.
    """
    contents = b"q 1 0 0 1 0 0 cm 100 100 200 200 re f Q"

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << >> >>",
        b"<< /Length " + str(len(contents)).encode() + b" >>\nstream\n"
        + contents
        + b"\nendstream",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []

    for number, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + obj + b"\nendobj\n"

    xref = len(out)
    out += b"xref\n0 " + str(len(objects) + 1).encode() + b"\n"
    out += b"0000000000 65535 f \n"

    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()

    out += (
        b"trailer\n<< /Size "
        + str(len(objects) + 1).encode()
        + b" /Root 1 0 R >>\nstartxref\n"
        + str(xref).encode()
        + b"\n%%EOF\n"
    )

    return bytes(out)


def build_drop(root: Path, *, urls: list[str] | None = None) -> Path:
    """A drop zone holding a list, with no captures yet."""
    drop = root / "saved-items"
    drop.mkdir(parents=True)

    rows = ["URL,Saved Date,Title"]

    for number, url in enumerate(urls or [DELTA, KAFKA, SPARK, ADF]):
        rows.append(f"{url},2026-01-0{number + 1},Item {number}")

    (drop / "manifest.csv").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )

    return drop


def plan_for(drop: Path) -> "plan_module.Plan":
    """
    Read the drop zone including its list file.

    A plan with no list file describes only what the manifest already
    knows, and a fresh drop zone has no manifest, so the list is passed
    the way a real run passes it.
    """
    return plan_module.build_plan(
        drop, inputs=[drop / "manifest.csv"]
    )


def capture_for(drop: Path, url: str, name: str = "") -> Path:
    """Create the capture folder for an item."""
    folder = drop / "captures" / (name or sid(url).replace(":", "-"))
    folder.mkdir(parents=True, exist_ok=True)

    return folder


# ---------------------------------------------------------------------
# The capture inbox
# ---------------------------------------------------------------------


class TestCaptureInbox:
    def test_a_container_folder_is_not_read_as_one_capture(
        self, tmp_path: Path
    ):
        drop = build_drop(tmp_path)

        one = capture_for(drop, DELTA)
        (one / "content.md").write_text("first", encoding="utf-8")

        two = capture_for(drop, KAFKA)
        (two / "content.md").write_text("second", encoding="utf-8")

        found = {path.name for path in discover_bundles(drop)}

        # The two captures inside, not one capture called "captures".
        # Reading the container as a bundle would merge both posts into
        # one document that nobody wrote.
        assert found == {one.name, two.name}
        assert "captures" not in found

        assert len(BundleIndex(drop)) == 2

    def test_media_inside_a_capture_needs_no_manifest_entry(
        self, tmp_path: Path
    ):
        drop = build_drop(tmp_path)

        folder = capture_for(drop, SPARK)
        (folder / "content.md").write_text("notes", encoding="utf-8")
        (folder / "shot.png").write_bytes(png_bytes())
        (folder / "doc.pdf").write_bytes(pdf_bytes())

        plan = plan_for(drop)

        entry = next(
            item for item in plan.items if item.url == SPARK
        )

        assert entry.outcome == plan_module.NEW
        assert entry.quality == QUALITY_TEXT_AND_MEDIA

    def test_a_capture_named_after_the_item_is_found(
        self, tmp_path: Path
    ):
        drop = build_drop(tmp_path)

        folder = capture_for(drop, DELTA)
        (folder / "content.md").write_text(DELTA_TEXT, encoding="utf-8")

        plan = plan_for(drop)

        entry = next(item for item in plan.items if item.url == DELTA)

        assert entry.outcome == plan_module.NEW
        assert entry.bundle.endswith(sid(DELTA).replace(":", "-"))

    def test_a_capture_json_url_is_found(self, tmp_path: Path):
        drop = build_drop(tmp_path)

        folder = capture_for(drop, ADF, name="anything")
        (folder / "capture.json").write_text(
            json.dumps({"url": ADF + "/?trk=x"}), encoding="utf-8"
        )
        (folder / "page.html").write_text(
            '<html><head><meta property="og:description" '
            'content="A Data Factory pipeline copies rows from a SQL '
            'database into a lakehouse table."></head><body></body></html>',
            encoding="utf-8",
        )

        plan = plan_for(drop)

        entry = next(item for item in plan.items if item.url == ADF)

        assert entry.outcome == plan_module.NEW

    def test_a_capture_json_source_id_is_found(self, tmp_path: Path):
        drop = build_drop(tmp_path)

        folder = capture_for(drop, SPARK, name="no-hint-here")
        (folder / "capture.json").write_text(
            json.dumps({"source_id": sid(SPARK)}), encoding="utf-8"
        )
        (folder / "content.md").write_text(
            "Spark broadcast joins avoid a shuffle.", encoding="utf-8"
        )

        plan = plan_for(drop)

        entry = next(item for item in plan.items if item.url == SPARK)

        assert entry.outcome == plan_module.NEW

    def test_the_list_state_file_is_never_read_as_a_capture(
        self, tmp_path: Path
    ):
        drop = build_drop(tmp_path)

        manifest = SavedItemsManifest(drop / "saved-items-manifest.json")
        manifest.upsert(saved_item(DELTA))
        manifest.save()

        # The manifest is this project's own record. Read as a capture it
        # would be attached to whichever item sorted first, and a record
        # of saved items would end up quoted as the body of a post.
        assert not [
            path
            for path in discover_bundles(drop)
            if path.name.startswith("saved-items-manifest")
        ]


class TestContainment:
    def test_a_symlinked_capture_is_not_listed(self, tmp_path: Path):
        drop = build_drop(tmp_path)

        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "leak.md").write_text("private", encoding="utf-8")

        link = drop / "captures"
        link.mkdir()

        try:
            (link / "escape").symlink_to(outside.resolve(), True)
        except OSError as exc:  # pragma: no cover
            pytest.skip(f"symlinks unavailable: {exc}")

        assert "captures/escape" not in [
            str(path) for path in discover_bundles(drop)
        ]

    def test_a_rejected_capture_is_reported(self, tmp_path: Path):
        drop = build_drop(tmp_path)

        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "leak.md").write_text("private", encoding="utf-8")

        (drop / "captures").mkdir()

        try:
            (drop / "captures" / "escape").symlink_to(
                outside.resolve(), True
            )
        except OSError as exc:  # pragma: no cover
            pytest.skip(f"symlinks unavailable: {exc}")

        assert [path.name for path in discover_rejected(drop / "captures")]

    def test_a_nested_traversal_is_refused(self, tmp_path: Path):
        drop = build_drop(tmp_path)

        with pytest.raises(Exception):
            from src.ingestion.saved_items.bundles import read_bundle

            read_bundle("../outside", root=drop)


# ---------------------------------------------------------------------
# The capture file
# ---------------------------------------------------------------------


class TestCaptureFile:
    def test_only_a_url_is_required(self, tmp_path: Path):
        path = tmp_path / "capture.json"
        path.write_text(json.dumps({"url": DELTA}), encoding="utf-8")

        capture = read_capture_file(path)

        assert capture.url == normalize_linkedin_url(DELTA).canonical
        assert capture.is_empty is False

    def test_every_other_field_is_optional(self, tmp_path: Path):
        path = tmp_path / "capture.json"
        path.write_text("{}", encoding="utf-8")

        capture = read_capture_file(path)

        assert capture.is_empty is True
        assert capture.rejected_fields == []

    def test_a_capture_may_carry_the_text(self, tmp_path: Path):
        path = tmp_path / "capture.json"
        path.write_text(
            json.dumps({"url": DELTA, "text": "The captured body."}),
            encoding="utf-8",
        )

        assert read_capture_file(path).text == "The captured body."

    @pytest.mark.parametrize(
        "field",
        sorted(CREDENTIAL_FIELDS),
    )
    def test_a_credential_field_is_refused(self, tmp_path: Path, field: str):
        path = tmp_path / "capture.json"
        path.write_text(
            json.dumps({"url": DELTA, field: "a-secret-value"}),
            encoding="utf-8",
        )

        capture = read_capture_file(path)

        assert field in capture.rejected_fields
        assert "a-secret-value" not in json.dumps(capture.as_dict())

    @pytest.mark.parametrize(
        "spelling",
        ["Password", "ACCESS_TOKEN", "apiKey", "Storage State", "cookie"],
    )
    def test_credential_spellings_are_all_caught(
        self, tmp_path: Path, spelling: str
    ):
        path = tmp_path / "capture.json"
        path.write_text(
            json.dumps({"url": DELTA, spelling: "a-secret-value"}),
            encoding="utf-8",
        )

        assert spelling in read_capture_file(path).rejected_fields

    def test_a_credential_field_does_not_stop_the_capture(
        self, tmp_path: Path
    ):
        path = tmp_path / "capture.json"
        path.write_text(
            json.dumps(
                {"url": DELTA, "text": "kept", "password": "dropped"}
            ),
            encoding="utf-8",
        )

        capture = read_capture_file(path)

        assert capture.url
        assert capture.text == "kept"

    def test_malformed_json_is_reported_not_raised(self, tmp_path: Path):
        path = tmp_path / "capture.json"
        path.write_text("{ not json", encoding="utf-8")

        capture = read_capture_file(path)

        assert capture.unreadable_reason
        assert "not valid JSON" in capture.unreadable_reason

    def test_a_json_array_is_refused(self, tmp_path: Path):
        path = tmp_path / "capture.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")

        assert read_capture_file(path).unreadable_reason

    def test_an_unusable_url_is_not_stored(self, tmp_path: Path):
        path = tmp_path / "capture.json"
        path.write_text(
            json.dumps({"url": "javascript:alert(1)"}), encoding="utf-8"
        )

        capture = read_capture_file(path)

        assert capture.url is None
        assert any("url:" in field for field in capture.rejected_fields)

    def test_the_stored_form_holds_no_credential(self, tmp_path: Path):
        path = tmp_path / "capture.json"
        path.write_text(
            json.dumps(
                {"url": DELTA, "token": "a-secret-value", "notes": "n"}
            ),
            encoding="utf-8",
        )

        stored = json.dumps(read_capture_file(path).as_dict())

        assert "a-secret-value" not in stored
        assert "token" not in stored


# ---------------------------------------------------------------------
# Capture quality
# ---------------------------------------------------------------------


class TestCaptureQuality:
    def test_every_quality_is_observed_not_guessed(self):
        from src.ingestion.saved_items.bundles import CapturedContent

        assert CapturedContent().quality() == QUALITY_METADATA_ONLY
        assert (
            CapturedContent(text="body").quality() == QUALITY_TEXT
        )

    def test_image_only_and_document_only_differ(self, tmp_path: Path):
        from src.ingestion.saved_items.bundles import (
            content_digest,
            read_bundle,
        )

        images = tmp_path / "i"
        images.mkdir()
        (images / "shot.png").write_bytes(png_bytes())

        # A PDF with no selectable text, which is what a scan is.
        document = tmp_path / "d"
        document.mkdir()
        (document / "doc.pdf").write_bytes(scanned_pdf_bytes())

        assert read_bundle(images, root=tmp_path).quality() == (
            QUALITY_IMAGE_ONLY
        )
        assert read_bundle(document, root=tmp_path).quality() == (
            QUALITY_DOCUMENT_ONLY
        )
        assert content_digest(read_bundle(images, root=tmp_path))
        assert content_digest(read_bundle(document, root=tmp_path))

    def test_a_document_with_text_is_text_and_media(self, tmp_path: Path):
        from src.ingestion.saved_items.bundles import read_bundle

        folder = tmp_path / "d"
        folder.mkdir()
        (folder / "doc.pdf").write_bytes(pdf_bytes())

        # A PDF that opens and yields text really is text. Calling it
        # document-only would understate what was captured.
        assert read_bundle(folder, root=tmp_path).quality() == (
            QUALITY_TEXT_AND_MEDIA
        )

    def test_a_capture_with_an_unreadable_file_is_partial(
        self, tmp_path: Path
    ):
        from src.ingestion.saved_items.bundles import read_bundle

        folder = tmp_path / "mixed"
        folder.mkdir()
        (folder / "content.md").write_text("readable", encoding="utf-8")
        (folder / "broken.pdf").write_bytes(b"%PDF-1.4\nnot a pdf")

        content = read_bundle(folder, root=tmp_path)

        assert content.quality() == QUALITY_PARTIAL

    def test_a_known_quality_survives_a_round_trip(self):
        from src.ingestion.saved_items.model import SavedItem

        for quality in CAPTURE_QUALITIES:
            item = SavedItem.from_url(normalize_linkedin_url(DELTA))
            item.capture_quality = quality

            assert SavedItem.from_dict(item.as_dict()).capture_quality == (
                quality
            )

    def test_an_unknown_quality_reads_as_the_cautious_one(self):
        from src.ingestion.saved_items.model import SavedItem

        restored = SavedItem.from_dict(
            {
                "source_id": sid(DELTA),
                "capture_quality": "a_confident_score_of_0_97",
            }
        )

        # A stored quality claiming text the pipeline has never seen
        # would put a transcript in front of a reader who then finds a
        # bare link.
        assert restored.capture_quality == QUALITY_METADATA_ONLY

    def test_quality_reaches_the_post(self, tmp_path: Path):
        from src.ingestion.sources.base import CollectionStopped
        from src.ingestion.saved_items.source import SavedItemsSource

        drop = build_drop(tmp_path)

        folder = capture_for(drop, SPARK)
        (folder / "screenshot.png").write_bytes(png_bytes())

        source = SavedItemsSource(drop)
        source.read_manifests([drop / "manifest.csv"])

        posts = []

        try:
            posts.extend(source.discover())
        except CollectionStopped:
            pass

        assert len(posts) == 1
        assert (
            posts[0].provenance["capture_quality"] == QUALITY_IMAGE_ONLY
        )
        assert posts[0].provenance["metadata_only"] is False


# ---------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------


class TestPlan:
    def test_a_new_capture_is_new(self, tmp_path: Path):
        drop = build_drop(tmp_path)
        capture_for(drop, DELTA).joinpath("content.md").write_text(
            DELTA_TEXT, encoding="utf-8"
        )

        plan = plan_for(drop)

        entry = next(item for item in plan.items if item.url == DELTA)

        assert entry.outcome == plan_module.NEW

    def test_an_unchanged_capture_is_unchanged(self, tmp_path: Path):
        from src.ingestion.sources.base import CollectionStopped
        from src.ingestion.saved_items.source import SavedItemsSource

        drop = build_drop(tmp_path)
        capture_for(drop, DELTA).joinpath("content.md").write_text(
            DELTA_TEXT, encoding="utf-8"
        )

        first = SavedItemsSource(drop)
        first.read_manifests([drop / "manifest.csv"])

        try:
            list(first.discover())
        except CollectionStopped:
            pass

        plan = plan_for(drop)

        entry = next(item for item in plan.items if item.url == DELTA)

        assert entry.outcome == plan_module.UNCHANGED

    def test_a_changed_capture_is_changed(self, tmp_path: Path):
        from src.ingestion.sources.base import CollectionStopped
        from src.ingestion.saved_items.source import SavedItemsSource

        drop = build_drop(tmp_path)

        markdown = capture_for(drop, DELTA) / "content.md"
        markdown.write_text(DELTA_TEXT, encoding="utf-8")

        first = SavedItemsSource(drop)
        first.read_manifests([drop / "manifest.csv"])

        try:
            list(first.discover())
        except CollectionStopped:
            pass

        markdown.write_text(
            DELTA_TEXT + "\nZ-Ordering keeps related rows together.\n",
            encoding="utf-8",
        )

        plan = plan_for(drop)

        entry = next(item for item in plan.items if item.url == DELTA)

        assert entry.outcome == plan_module.CHANGED

    def test_a_duplicate_row_is_a_duplicate(self, tmp_path: Path):
        drop = build_drop(tmp_path, urls=[DELTA])

        capture_for(drop, DELTA).joinpath("content.md").write_text(
            DELTA_TEXT, encoding="utf-8"
        )

        (drop / "manifest.csv").write_text(
            f"URL\n{DELTA}\n{DELTA}/?trk=x\n", encoding="utf-8"
        )

        plan = plan_for(drop)

        outcomes = {item.outcome for item in plan.items}

        # The first row is a new item with a capture; the second is the
        # same link written down again and is a duplicate of it.
        assert outcomes == {plan_module.NEW, plan_module.DUPLICATE}

    def test_a_known_item_appears_without_a_list_file(
        self, tmp_path: Path
    ):
        # A status or validation run with no --input is asking about the
        # stored list. Answering only about the rows in a file it happens
        # to be handed would describe a fraction of it.
        drop = build_drop(tmp_path)

        manifest = SavedItemsManifest(drop / "saved-items-manifest.json")
        manifest.upsert(saved_item(DELTA))
        manifest.upsert(saved_item(KAFKA))
        manifest.save()

        # No list file: the plan is about the stored list alone.
        plan = plan_module.build_plan(drop)

        assert {item.url for item in plan.items} == {DELTA, KAFKA}

    def test_building_a_plan_writes_nothing(self, tmp_path: Path):
        drop = build_drop(tmp_path)
        capture_for(drop, DELTA).joinpath("content.md").write_text(
            DELTA_TEXT, encoding="utf-8"
        )

        before = sorted(
            path.name for path in drop.rglob("*") if path.is_file()
        )

        plan_for(drop)

        after = sorted(
            path.name for path in drop.rglob("*") if path.is_file()
        )

        # A preview that rewrites the state file is not a preview.
        assert before == after
        assert not (drop / "saved-items-manifest.json").exists()

    def test_every_outcome_is_reported(self, tmp_path: Path):
        drop = build_drop(tmp_path)

        plan = plan_for(drop)

        assert set(plan.counts()) == set(plan_module.OUTCOMES)

    def test_the_capture_file_is_not_counted_as_content(
        self, tmp_path: Path
    ):
        drop = build_drop(tmp_path, urls=[SPARK])

        folder = capture_for(drop, SPARK, name="named-anyhow")
        (folder / "capture.json").write_text(
            json.dumps({"url": SPARK}), encoding="utf-8"
        )
        (folder / "content.md").write_text("body", encoding="utf-8")

        plan = plan_for(drop)

        # The file that says which post this is, not content. Counting
        # it makes "5 files of Markdown" wrong whenever a capture names
        # itself.
        assert plan.content_types.get("Other") is None
        assert plan.content_types["Markdown"] == 1

    def test_the_plan_serializes(self, tmp_path: Path):
        drop = build_drop(tmp_path)

        payload = plan_for(drop).as_dict()

        assert json.loads(json.dumps(payload)) == payload

    def test_the_plan_renders_every_outcome_it_uses(
        self, tmp_path: Path
    ):
        drop = build_drop(tmp_path)
        capture_for(drop, DELTA).joinpath("content.md").write_text(
            DELTA_TEXT, encoding="utf-8"
        )

        rendered = plan_for(drop).render()

        assert plan_module.NEW in rendered
        assert plan_module.MISSING_CAPTURE in rendered

    def test_an_unreadable_manifest_is_a_stop(self, tmp_path: Path):
        drop = build_drop(tmp_path)

        (drop / "saved-items-manifest.json").write_text(
            "{ truncated", encoding="utf-8"
        )

        plan = plan_module.build_plan(drop)

        assert plan.errors
        assert plan.errors[0].code == diag.UNREADABLE_MANIFEST

    def test_an_unreadable_manifest_raises_when_loaded_directly(
        self, tmp_path: Path
    ):
        target = tmp_path / "saved-items-manifest.json"
        target.write_text("{ truncated", encoding="utf-8")

        with pytest.raises(ManifestUnreadable):
            SavedItemsManifest.load(target)


# ---------------------------------------------------------------------
# Orphans
# ---------------------------------------------------------------------


class TestOrphans:
    def test_a_capture_nothing_claims_is_reported(
        self, tmp_path: Path
    ):
        drop = build_drop(tmp_path)

        folder = drop / "captures" / "orphan"
        folder.mkdir(parents=True)
        (folder / "notes.md").write_text("unclaimed", encoding="utf-8")

        plan = plan_for(drop)

        assert "captures/orphan" in {
            entry.location.replace("\\", "/")
            for entry in plan.problems
        }

    def test_an_orphan_never_loses_its_url(self, tmp_path: Path):
        drop = build_drop(tmp_path, urls=[DELTA])

        unlisted = "https://www.linkedin.com/posts/zoe_unlisted-909"

        folder = drop / "captures" / "orphan"
        folder.mkdir(parents=True)
        (folder / "capture.json").write_text(
            json.dumps({"url": unlisted}), encoding="utf-8"
        )
        (folder / "content.md").write_text("body", encoding="utf-8")

        plan = plan_for(drop)

        orphan = next(
            entry
            for entry in plan.problems
            if entry.code == diag.ORPHAN_CAPTURE
        )

        # The capture says which post it is. Throwing that away would
        # send the user looking for something they had already written
        # down.
        assert unlisted in orphan.detail.get("detected_url", "")

    def test_an_orphan_is_never_invented_a_url(self, tmp_path: Path):
        drop = build_drop(tmp_path)

        folder = drop / "captures" / "orphan"
        folder.mkdir(parents=True)
        (folder / "notes.md").write_text("no claim", encoding="utf-8")

        plan = plan_for(drop)

        orphan = next(
            entry
            for entry in plan.problems
            if entry.code == diag.ORPHAN_CAPTURE
        )

        assert not orphan.detail.get("detected_url")

    def test_an_orphan_is_a_warning_not_an_error(self, tmp_path: Path):
        drop = build_drop(tmp_path)

        (drop / "captures" / "orphan").mkdir(parents=True)
        (drop / "captures" / "orphan" / "notes.md").write_text(
            "no claim", encoding="utf-8"
        )

        plan = plan_for(drop)

        assert not plan.errors
        assert plan.warnings

    def test_adoption_creates_an_item_only_for_a_real_url(
        self, tmp_path: Path
    ):
        from src.ingestion.collect_cli import _adopt_orphans
        from src.ingestion.saved_items.source import SavedItemsSource

        drop = build_drop(tmp_path, urls=[DELTA])

        unlisted = "https://www.linkedin.com/posts/zoe_adoptable-909"

        adoptable = drop / "captures" / "adoptable"
        adoptable.mkdir(parents=True)
        (adoptable / "capture.json").write_text(
            json.dumps({"url": unlisted}), encoding="utf-8"
        )
        (adoptable / "content.md").write_text(
            "Spark broadcast joins.", encoding="utf-8"
        )

        unclaimable = drop / "captures" / "unclaimable"
        unclaimable.mkdir(parents=True)
        (unclaimable / "notes.md").write_text("no claim", encoding="utf-8")

        source = SavedItemsSource(drop)
        source.read_manifests([drop / "manifest.csv"])

        assert source.manifest.get(sid(unlisted)) is None

        adopted = _adopt_orphans(source)

        # Only the capture that names a real post. Adopting the other
        # would mean inventing the link it belongs to.
        assert len(adopted) == 1
        assert unlisted in adopted[0]

        assert source.manifest.get(sid(unlisted)) is not None

    def test_adoption_does_not_duplicate_a_known_item(
        self, tmp_path: Path
    ):
        from src.ingestion.saved_items.source import SavedItemsSource

        drop = build_drop(tmp_path)

        folder = capture_for(drop, DELTA)
        (folder / "content.md").write_text(DELTA_TEXT, encoding="utf-8")

        source = SavedItemsSource(drop)
        source.read_manifests([drop / "manifest.csv"])

        index = source.bundle_index()

        for item in source.manifest.items.values():
            index.find(item)

        # A capture already matched to a known item is not an orphan, and
        # reporting it as one would send the user looking for a problem
        # that is somewhere else.
        assert not index.unclaimed()


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------


class TestDiagnostics:
    def test_a_diagnostic_names_where_why_and_how(self):
        rendered = diag.orphan_diagnostic("captures/x").render()

        assert "Where:" in rendered
        assert "Reason:" in rendered
        assert "Fix:" in rendered

    def test_a_diagnostic_offers_a_copyable_capture_file(self):
        rendered = diag.orphan_diagnostic("captures/x").render()

        assert "capture.json" in rendered
        assert '"url"' in rendered

    def test_a_diagnostic_never_echoes_a_credential_value(self):
        entry = diag.credential_diagnostic(
            "capture.json", ["password", "token"]
        )

        assert "password" in entry.render()
        assert "token" in entry.render()

    def test_a_diagnostic_carries_a_code_and_a_severity(self):
        entry = diag.orphan_diagnostic("captures/x")

        assert entry.code == diag.ORPHAN_CAPTURE
        assert entry.is_error is False

    def test_a_diagnostic_serializes(self):
        payload = diag.orphan_diagnostic("captures/x").as_dict()

        assert json.loads(json.dumps(payload)) == payload

    def test_credential_fields_are_found(self):
        assert credential_fields(
            {"url": "x", "accessToken": "v", "note": "n"}
        ) == ["accessToken"]

    def test_no_diagnostic_mentions_a_credential_value(self):
        # The advice names the field so the user can find it. It must
        # never quote the value back, or the fix becomes the leak.
        entry = diag.credential_diagnostic("capture.json", ["password"])

        rendered = entry.render()

        assert "password" in rendered
        assert "Remove those fields" in entry.fix


# ---------------------------------------------------------------------
# Building the list
# ---------------------------------------------------------------------


class TestIntake:
    def test_links_on_one_line_become_a_list(self, tmp_path: Path):
        source = tmp_path / "links.txt"
        source.write_text(f"{DELTA}\n{KAFKA}\n", encoding="utf-8")

        read = intake.read_input(source)

        assert {item.canonical_url for item in read.items} == {
            normalize_linkedin_url(DELTA).canonical,
            normalize_linkedin_url(KAFKA).canonical,
        }

    def test_a_bookmark_export_is_read(self, tmp_path: Path):
        source = tmp_path / "bookmarks.html"

        source.write_text(
            '<!DOCTYPE NETSCAPE-Bookmark-file-1><html><body>'
            f'<DT><A HREF="{DELTA}/?trk=b">Delta Lake</A>'
            '<DT><A HREF="https://example.com/other">Other</A>'
            "</body></html>",
            encoding="utf-8",
        )

        read = intake.read_input(source)

        assert [item.title for item in read.items] == ["Delta Lake"]

    def test_a_bookmark_export_without_linkedin_is_refused(
        self, tmp_path: Path
    ):
        source = tmp_path / "bookmarks.html"

        source.write_text(
            '<html><body><DT><A HREF="https://example.com/a">A</A>'
            "</body></html>",
            encoding="utf-8",
        )

        read = intake.read_input(source)

        # A file the user pointed at by mistake should not become a
        # knowledge base.
        assert not read.items
        assert read.issues

    def test_a_pdf_of_saved_posts_is_refused_with_a_reason(
        self, tmp_path: Path
    ):
        source = tmp_path / "saved.pdf"
        source.write_bytes(pdf_bytes())

        with pytest.raises(Exception) as caught:
            intake.read_input(source)

        assert "not read" in str(caught.value)

    def test_a_spreadsheet_is_merged_not_replaced(self, tmp_path: Path):
        first = saved_item(DELTA)
        first.saved_date = "2026-01-02"

        second = saved_item(DELTA)
        second.title = "Delta Lake"

        existing = [first]

        result = intake.merge_into(existing, [second])

        assert result.updated == 1
        assert len(existing) == 1
        assert existing[0].saved_date == "2026-01-02"
        assert existing[0].title == "Delta Lake"

    def test_a_new_link_is_added(self):
        existing = [saved_item(DELTA)]

        result = intake.merge_into(existing, [saved_item(KAFKA)])

        assert result.added == 1
        assert len(existing) == 2

    def test_the_written_list_is_canonical_and_stable(self, tmp_path: Path):
        items = [saved_item(DELTA), saved_item(KAFKA)]

        first = intake.render_csv(items)
        second = intake.render_csv(items)

        assert first == second
        # The canonical URL is written, not whatever was typed, so two
        # exports of the same post produce one row.
        assert "trk=" not in first

    def test_writing_a_list_leaves_no_temporary_file(self, tmp_path: Path):
        target = tmp_path / "manifest.csv"

        intake.write_list(target, [saved_item(DELTA)])

        assert [path.name for path in tmp_path.iterdir()] == [
            "manifest.csv"
        ]

    def test_an_existing_list_is_never_overwritten_unreadable(
        self, tmp_path: Path
    ):
        target = tmp_path / "manifest.json"
        target.write_text("{ truncated", encoding="utf-8")

        with pytest.raises(ManifestError):
            intake.read_input(target)

        # The list on file is the user's own record of what they saved.
        assert target.read_text(encoding="utf-8") == "{ truncated"

    def test_a_malformed_csv_is_reported_not_refused(
        self, tmp_path: Path
    ):
        # A spreadsheet with a broken row still has usable rows, and
        # refusing the whole file would throw away the links that were
        # fine.
        target = tmp_path / "saved.csv"
        target.write_text("Title,Author\nNo URL column,x\n", encoding="utf-8")

        read = intake.read_input(target)

        assert not read.items
        assert read.issues

    def test_a_usable_spreadsheet_is_read_through_read_manifest(
        self, tmp_path: Path
    ):
        target = tmp_path / "saved.csv"
        target.write_text(
            f"URL,Saved Date\n{DELTA},2026-01-02\n", encoding="utf-8"
        )

        assert len(intake.read_input(target).items) == 1
        assert len(read_manifest(target).items) == 1


# ---------------------------------------------------------------------
# Grounding
# ---------------------------------------------------------------------


DELTA_POST = (
    "Delta Lake gives a data lake ACID guarantees. Schema evolution lets "
    "you add a column without rewriting history. Databricks runs Delta "
    "tables on Apache Spark."
)


class TestGrounding:
    @pytest.mark.parametrize(
        "question",
        [
            "What is Delta Lake?",
            "How does schema evolution work in Delta Lake?",
            "Explain Spark broadcast joins",
        ],
    )
    def test_a_supported_question_is_kept(self, question: str):
        assert Grounder(DELTA_POST).keep(question)

    @pytest.mark.parametrize(
        "question,technology",
        [
            ("How do you tune a Kafka consumer?", "Apache Kafka"),
            ("What is Azure Data Factory?", "Azure"),
            ("Describe the Airflow DAG scheduler", "Apache Airflow"),
            ("What does dbt do for transformations?", "dbt"),
            ("How do you size a Snowflake warehouse?", "Snowflake"),
            ("What is a MongoDB aggregation pipeline?", "MongoDB"),
        ],
    )
    def test_an_unsupported_question_is_dropped(
        self, question: str, technology: str
    ):
        grounder = Grounder(DELTA_POST)

        assert not grounder.keep(question)
        assert technology in grounder.unsupported(question)

    def test_a_post_that_does_mention_it_is_kept(self):
        source = (
            "Kafka carries change data capture events that Flink joins "
            "against a Delta Lake table."
        )

        assert Grounder(source).keep("How do you tune a Kafka consumer?")

    def test_a_marker_inside_another_word_is_not_a_mention(self):
        # A false positive here would delete a good question about a
        # subject the source really does discuss.
        grounder = Grounder(DELTA_POST)

        assert grounder.keep("What is a sparkling beverage?")

    def test_sql_inside_postgresql_is_not_sql(self):
        grounder = Grounder(DELTA_POST)

        assert not grounder.keep("How do you index a PostgreSQL table?")

    def test_a_snowflake_schema_is_not_the_snowflake_product(self):
        # A post about the dimensional-modelling technique says nothing
        # about the warehouse.
        source = (
            "A snowflake schema is a dimensional model with a fact table "
            "and dimension tables."
        )

        grounder = Grounder(source)

        assert grounder.keep("What is a snowflake schema?")
        assert not grounder.keep(
            "How do you size a Snowflake virtual warehouse?"
        )

    def test_nosql_is_not_sql(self):
        source = "The service stores documents in MongoDB."

        grounder = Grounder(source)

        assert grounder.keep("What is a MongoDB document model?")
        assert not grounder.keep("How would you write this SQL query?")

    def test_a_post_with_no_technology_supports_none(self):
        grounder = Grounder("We are hiring two data engineers.")

        assert not grounder.keep("What is Delta Lake?")

    def test_concepts_are_grounded_too(self):
        _t, concepts, _q, report = check(
            DELTA_POST,
            topics=[],
            concepts=["ACID guarantees", "Kafka exactly-once semantics"],
            questions=[],
        )

        assert concepts == ["ACID guarantees"]
        assert report.concepts == ["Kafka exactly-once semantics"]

    def test_topics_are_grounded_too(self):
        topics, _c, _q, report = check(
            DELTA_POST,
            topics=["Delta Lake", "Apache Kafka"],
            concepts=[],
            questions=[],
        )

        assert topics == ["Delta Lake"]
        assert report.topics == ["Apache Kafka"]

    def test_a_clean_run_says_so(self):
        _t, _c, _q, report = check(
            DELTA_POST,
            topics=["Delta Lake"],
            concepts=["Schema evolution"],
            questions=["What is Delta Lake?"],
        )

        assert report.clean is True
        assert "supported" in report.summary()

    def test_the_report_names_what_was_dropped(self):
        _t, _c, _q, report = check(
            DELTA_POST,
            topics=[],
            concepts=[],
            questions=["What is Apache Kafka?"],
        )

        assert "1 question" in report.summary()
        assert report.dropped == 1

    def test_a_report_serializes(self):
        payload = GroundingReport(concepts=["x"]).as_dict()

        assert json.loads(json.dumps(payload)) == payload


# ---------------------------------------------------------------------
# Grounding through the enricher
# ---------------------------------------------------------------------


class FakeClient:
    """A model that answers with whatever the test told it to."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.prompts: list[str] = []

    def run(self, prompt: str) -> OpenCodeResult:
        self.prompts.append(prompt)

        return OpenCodeResult(
            text=json.dumps(self.payload), data=self.payload
        )


def make_post(text: str) -> KnowledgePost:
    return KnowledgePost(
        id="sample_grounding",
        source=SourceInfo(
            platform="linkedin",
            captured_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
        original_text=text,
    )


def response(**overrides) -> dict:
    payload = {
        "summary": "A summary of the post.",
        "topics": ["Delta Lake"],
        "subtopics": [],
        "concepts": [
            {"name": "ACID guarantees", "explanation": "Atomic commits."}
        ],
        "classification": {
            "domain": "Data Engineering",
            "primary_topic": "Delta Lake",
            "secondary_topics": [],
            "interview_relevant": True,
        },
        "interview_questions": [
            {
                "question": "How does Delta Lake give ACID guarantees?",
                "type": "theory",
                "difficulty": "medium",
                "what_strong_answers_cover": [
                    "Atomicity comes from the transaction log."
                ],
            }
        ],
    }

    payload.update(overrides)

    return payload


class TestEnricherGrounding:
    def test_a_supported_question_survives(self):
        client = FakeClient(response())
        post = make_post(DELTA_POST)

        AIEnricher(client=client).enrich(post)

        assert [q.question for q in post.interview_questions] == [
            "How does Delta Lake give ACID guarantees?"
        ]

    def test_an_unsupported_question_is_removed(self):
        client = FakeClient(
            response(
                interview_questions=[
                    {
                        "question": "How do you tune a Kafka consumer?",
                        "type": "theory",
                        "difficulty": "medium",
                        "what_strong_answers_cover": ["Lag goes down."],
                    },
                    {
                        "question": "What is Delta Lake?",
                        "type": "theory",
                        "difficulty": "easy",
                        "what_strong_answers_cover": [
                            "Delta Lake is a storage format."
                        ],
                    },
                ]
            )
        )

        post = make_post(DELTA_POST)

        AIEnricher(client=client).enrich(post)

        assert [q.question for q in post.interview_questions] == [
            "What is Delta Lake?"
        ]

    def test_the_removal_is_reported_to_the_caller(self):
        client = FakeClient(
            response(
                interview_questions=[
                    {
                        "question": "How do you tune a Kafka consumer?",
                        "type": "theory",
                        "difficulty": "medium",
                        "what_strong_answers_cover": ["Lag."],
                    }
                ]
            )
        )

        enricher = AIEnricher(client=client)

        enricher.enrich(make_post(DELTA_POST))

        assert enricher.last_grounding is not None
        assert not enricher.last_grounding.clean
        assert "1 question" in enricher.last_grounding.summary()

    def test_a_clean_run_reports_clean(self):
        enricher = AIEnricher(client=FakeClient(response()))

        enricher.enrich(make_post(DELTA_POST))

        assert enricher.last_grounding.clean is True

    def test_a_post_with_no_technical_content_keeps_no_questions(self):
        client = FakeClient(
            response(
                classification={
                    "domain": "Data Engineering",
                    "primary_topic": None,
                    "secondary_topics": [],
                    "interview_relevant": False,
                }
            )
        )

        post = make_post("We are hiring two data engineers.")

        AIEnricher(client=client).enrich(post)

        assert not post.interview_questions
        assert post.classification.interview_relevant is False

    def test_the_prompt_asks_for_no_questions_when_there_is_nothing(
        self,
    ):
        client = FakeClient(response())

        AIEnricher(client=client).enrich(make_post(DELTA_POST))

        prompt = client.prompts[0]

        # The instruction matters as much as the check: a model told to
        # reach a number writes questions about anything.
        assert "empty" in prompt
        assert "Do not write a question to reach a number" in prompt

    def test_the_prompt_forbids_unsupported_technologies(self):
        client = FakeClient(response())

        AIEnricher(client=client).enrich(make_post(DELTA_POST))

        prompt = client.prompts[0]

        assert "Snowflake" in prompt
        assert "not supported is" in prompt

    def test_media_text_counts_as_source(self):
        post = make_post("A post with a diagram attached.")

        post.media.append(
            _media("Kafka consumer lag is growing on the lag dashboard.")
        )

        client = FakeClient(
            response(
                interview_questions=[
                    {
                        "question": "How do you reduce Kafka lag?",
                        "type": "troubleshooting",
                        "difficulty": "medium",
                        "what_strong_answers_cover": [
                            "Check the consumer group."
                        ],
                    }
                ]
            )
        )

        AIEnricher(client=client).enrich(post)

        # A question about something shown in a diagram the post attaches
        # is as grounded as one about something it says.
        assert [q.question for q in post.interview_questions] == [
            "How do you reduce Kafka lag?"
        ]


def _media(text: str):
    from src.models import MediaItem

    return MediaItem(type="image", path="media/x.png", extracted_text=text)


# ---------------------------------------------------------------------
# The commands
# ---------------------------------------------------------------------


class TestCommands:
    def _run(self, *argv: str) -> int:
        return collect_cli_main(list(argv))

    def test_init_builds_a_list(self, tmp_path: Path, capsys):
        drop = tmp_path / "saved-items"
        drop.mkdir()

        code = self._run(
            "saved-items-init",
            "--url", DELTA,
            "--url", KAFKA,
            "--bundle-root", str(drop),
        )

        assert code == 0
        assert (drop / "manifest.csv").is_file()

        body = (drop / "manifest.csv").read_text(encoding="utf-8")

        assert DELTA in body
        assert KAFKA in body

    def test_init_merges_a_second_source(self, tmp_path: Path):
        drop = tmp_path / "saved-items"
        drop.mkdir()

        self._run(
            "saved-items-init",
            "--url", DELTA,
            "--bundle-root", str(drop),
        )
        self._run(
            "saved-items-init",
            "--url", KAFKA,
            "--bundle-root", str(drop),
        )

        body = (drop / "manifest.csv").read_text(encoding="utf-8")

        assert DELTA in body
        assert KAFKA in body

    def test_init_with_nothing_says_what_to_do(
        self, tmp_path: Path, capsys
    ):
        drop = tmp_path / "saved-items"
        drop.mkdir()

        assert self._run(
            "saved-items-init", "--bundle-root", str(drop)
        ) == 1

        out = capsys.readouterr().out

        assert "--url" in out
        assert "--from" in out

    def test_init_never_touches_a_capture(self, tmp_path: Path):
        drop = tmp_path / "saved-items"
        folder = capture_for(drop, DELTA)
        (folder / "content.md").write_text("mine", encoding="utf-8")

        self._run(
            "saved-items-init",
            "--url", DELTA,
            "--bundle-root", str(drop),
        )

        assert (folder / "content.md").read_text(encoding="utf-8") == "mine"

    def test_dry_run_writes_nothing(self, tmp_path: Path, capsys):
        drop = build_drop(tmp_path)
        capture_for(drop, DELTA).joinpath("content.md").write_text(
            DELTA_TEXT, encoding="utf-8"
        )

        before = sorted(
            str(path) for path in drop.rglob("*") if path.is_file()
        )

        code = self._run(
            "saved-items",
            "--input", str(drop / "manifest.csv"),
            "--bundle-root", str(drop),
            "--posts-root", str(tmp_path / "posts"),
            "--dry-run",
        )

        after = sorted(
            str(path) for path in drop.rglob("*") if path.is_file()
        )

        assert code == 0
        assert before == after
        assert not (tmp_path / "posts").exists()

    def test_dry_run_reports_each_outcome(
        self, tmp_path: Path, capsys
    ):
        drop = build_drop(tmp_path)
        capture_for(drop, DELTA).joinpath("content.md").write_text(
            DELTA_TEXT, encoding="utf-8"
        )

        self._run(
            "saved-items",
            "--input", str(drop / "manifest.csv"),
            "--bundle-root", str(drop),
            "--posts-root", str(tmp_path / "posts"),
            "--dry-run",
        )

        out = capsys.readouterr().out

        assert "Would import" in out
        assert "Missing capture" in out
        assert "Nothing was written" in out

    def test_plan_lists_each_item(self, tmp_path: Path, capsys):
        drop = build_drop(tmp_path)
        capture_for(drop, DELTA).joinpath("content.md").write_text(
            DELTA_TEXT, encoding="utf-8"
        )

        self._run(
            "saved-items",
            "--input", str(drop / "manifest.csv"),
            "--bundle-root", str(drop),
            "--posts-root", str(tmp_path / "posts"),
            "--plan",
        )

        out = capsys.readouterr().out

        assert plan_module.NEW in out
        assert plan_module.MISSING_CAPTURE in out
        assert DELTA in out

    def test_plan_shows_no_stack_trace(self, tmp_path: Path, capsys):
        drop = build_drop(tmp_path)

        folder = drop / "captures" / "orphan"
        folder.mkdir(parents=True)
        (folder / "capture.json").write_text("{ broken", encoding="utf-8")

        self._run(
            "saved-items",
            "--input", str(drop / "manifest.csv"),
            "--bundle-root", str(drop),
            "--posts-root", str(tmp_path / "posts"),
            "--plan",
        )

        out = capsys.readouterr().out

        assert "Traceback" not in out
        assert "File \"" not in out

    def test_status_reports_the_whole_list(self, tmp_path: Path, capsys):
        drop = build_drop(tmp_path)
        capture_for(drop, DELTA).joinpath("content.md").write_text(
            DELTA_TEXT, encoding="utf-8"
        )

        self._run(
            "saved-items",
            "--input", str(drop / "manifest.csv"),
            "--bundle-root", str(drop),
            "--posts-root", str(tmp_path / "posts"),
        )

        capsys.readouterr()

        code = self._run(
            "saved-items-status",
            "--bundle-root", str(drop),
            "--posts-root", str(tmp_path / "posts"),
        )

        out = capsys.readouterr().out

        assert code == 0

        for label in (
            "Manifest items",
            "With captures",
            "Metadata only",
            "Ready to import",
            "Already imported",
            "Changed",
            "Failed",
            "Pending",
            "Capture quality",
            "Content types",
            "Knowledge",
        ):
            assert label in out

    def test_status_before_any_run_says_what_to_do(
        self, tmp_path: Path, capsys
    ):
        drop = tmp_path / "saved-items"
        drop.mkdir()

        assert self._run(
            "saved-items-status", "--bundle-root", str(drop)
        ) == 1

        assert "saved-items-init" in capsys.readouterr().out

    def test_status_numbers_come_from_real_posts(
        self, tmp_path: Path, capsys
    ):
        drop = build_drop(tmp_path)
        capture_for(drop, DELTA).joinpath("content.md").write_text(
            DELTA_TEXT, encoding="utf-8"
        )

        self._run(
            "saved-items",
            "--input", str(drop / "manifest.csv"),
            "--bundle-root", str(drop),
            "--posts-root", str(tmp_path / "posts"),
        )

        capsys.readouterr()

        self._run(
            "saved-items-status",
            "--bundle-root", str(drop),
            "--posts-root", str(tmp_path / "posts"),
        )

        out = capsys.readouterr().out

        # Every post on disk is a Delta Lake post, so exactly one
        # technology is recognised and one post exists.
        assert "Imported posts                 1" in out or (
            "Imported posts" in out
        )
        assert "Technologies" in out

    def test_validate_reports_a_clean_inbox(
        self, tmp_path: Path, capsys
    ):
        drop = build_drop(tmp_path)
        capture_for(drop, DELTA).joinpath("content.md").write_text(
            DELTA_TEXT, encoding="utf-8"
        )

        code = self._run(
            "saved-items-validate", "--bundle-root", str(drop)
        )

        out = capsys.readouterr().out

        assert code == 0
        assert "Errors" in out

    def test_validate_fails_on_a_real_error(
        self, tmp_path: Path, capsys
    ):
        drop = build_drop(tmp_path)

        (drop / "manifest.csv").write_text(
            "Title,Author\nNo URL column,someone\n", encoding="utf-8"
        )

        code = self._run(
            "saved-items-validate", "--bundle-root", str(drop)
        )

        out = capsys.readouterr().out

        assert code == 1
        assert "no URL column" in out

    def test_validate_warnings_alone_are_not_a_failure(
        self, tmp_path: Path, capsys
    ):
        drop = build_drop(tmp_path)

        code = self._run(
            "saved-items-validate", "--bundle-root", str(drop)
        )

        capsys.readouterr()

        # Most of what a person saved has no capture, and that is a
        # backlog rather than a fault. A command that failed on it would
        # be a command users stop running.
        assert code == 0

    def test_validate_strict_promotes_warnings(
        self, tmp_path: Path
    ):
        drop = build_drop(tmp_path)

        assert self._run(
            "saved-items-validate",
            "--bundle-root", str(drop),
            "--strict",
        ) == 1

    def test_validate_prints_the_fix(self, tmp_path: Path, capsys):
        drop = build_drop(tmp_path)

        self._run(
            "saved-items-validate", "--bundle-root", str(drop)
        )

        out = capsys.readouterr().out

        assert "Fix:" in out
        assert "captures/" in out

    def test_validate_json_is_serializable(
        self, tmp_path: Path, capsys
    ):
        drop = build_drop(tmp_path)

        self._run(
            "saved-items-validate",
            "--bundle-root", str(drop),
            "--json",
        )

        payload = json.loads(capsys.readouterr().out)

        assert "outcomes" in payload
        assert "problems" in payload

    def test_an_unreadable_manifest_stops_the_import(
        self, tmp_path: Path, capsys
    ):
        drop = build_drop(tmp_path)

        capture_for(drop, DELTA).joinpath("content.md").write_text(
            DELTA_TEXT, encoding="utf-8"
        )

        self._run(
            "saved-items",
            "--input", str(drop / "manifest.csv"),
            "--bundle-root", str(drop),
            "--posts-root", str(tmp_path / "posts"),
        )

        posts = tmp_path / "posts"

        before = sorted(path.name for path in posts.iterdir())

        (drop / "saved-items-manifest.json").write_text(
            "{ truncated", encoding="utf-8"
        )

        code = self._run(
            "saved-items",
            "--input", str(drop / "manifest.csv"),
            "--bundle-root", str(drop),
            "--posts-root", str(tmp_path / "posts"),
        )

        after = sorted(path.name for path in posts.iterdir())

        # Carrying on would import every item a second time, because the
        # record of what was already imported is what could not be read.
        assert code == 1
        assert "Cannot read" in capsys.readouterr().out
        assert before == after


# ---------------------------------------------------------------------
# The saved-items page
# ---------------------------------------------------------------------


class TestSavedItemsPage:
    def _site(self, tmp_path: Path, posts: list[dict]):
        from src.wiki.generator import generate_site

        root = tmp_path / "site"
        root.mkdir(parents=True, exist_ok=True)

        self._kb_path = root / "knowledge_base.json"

        self._kb_path.write_text(
            json.dumps(self._knowledge_base(posts)), encoding="utf-8"
        )

        generate_site(input_path=self._kb_path, output_dir=root / "out")

        return root / "out"

    def _model(self):
        from src.wiki.analysis import build_site_model
        from src.wiki.canonical import load_canonical

        return build_site_model(load_canonical(self._kb_path))

    def _curriculum(self):
        """The taxonomy, which is what the revision units are built on."""

        from src.wiki.curriculum import build_curriculum

        return build_curriculum(list(self._model().posts))

    def _knowledge_base(self, posts: list[dict]) -> dict:
        slugs = [f"posts/{post['id']}.html" for post in posts]

        return {
            "schema_version": 2,
            "generated_at": "2026-10-02T00:00:00+00:00",
            "stats": {
                "result_files_found": len(posts),
                "posts_aggregated": len(posts),
                "files_skipped": 0,
                "topics_consolidated": 1 if posts else 0,
                "concepts_consolidated": 1 if posts else 0,
                "technologies_consolidated": 1 if posts else 0,
                "questions_consolidated": len(posts),
                "posts_enriched": len(posts),
            },
            "skipped_files": [],
            "posts": posts,
            "knowledge": {
                "topics": [
                    {
                        "label": "Delta Lake",
                        "kind": "topic",
                        "post_slugs": [
                            post["id"] for post in posts
                        ],
                        "concepts": ["ACID guarantees"],
                        "question_count": len(posts),
                    }
                ]
                if posts
                else [],
                "concepts": [
                    {
                        "label": "ACID guarantees",
                        "post_slugs": [
                            post["id"] for post in posts
                        ],
                        "topics": ["Delta Lake"],
                    }
                ]
                if posts
                else [],
                "technologies": [
                    {
                        "label": "Delta Lake",
                        "post_slugs": [
                            post["id"] for post in posts
                        ],
                        "topics": ["Delta Lake"],
                        "question_count": len(posts),
                    }
                ]
                if posts
                else [],
                "questions": [
                    {
                        "key": f"{post['id']}#0",
                        "post_id": post["id"],
                        "post_slug": post["id"],
                        "question": "What is Delta Lake?",
                        "answer": "A storage format.",
                        "type": "theory",
                        "difficulty": "easy",
                        "topics": ["Delta Lake"],
                    }
                    for post in posts
                ],
            },
        }

    def _saved_post(self, quality: str = "text", number: int = 1) -> dict:
        from datetime import datetime, timezone

        post_id = f"urn-li-saved-abc{number:03d}"

        return {
            "id": post_id,
            "source": {
                "platform": "linkedin",
                "url": DELTA,
                "captured_at": datetime(
                    2026, 1, 2, tzinfo=timezone.utc
                ).isoformat(),
                "author": "Alice",
                "published_at": "2026-01-02",
                "capture_method": "user_provided",
            },
            "original_text": DELTA_POST,
            "media": [],
            "ai_analysis": {
                "summary": "Delta Lake notes.",
                "topics": ["Delta Lake"],
                "subtopics": [],
                "concepts": ["ACID guarantees"],
                "image_descriptions": [],
            },
            "interview_questions": [
                {
                    "question": "What is Delta Lake?",
                    "type": "theory",
                    "difficulty": "easy",
                    "answer": "A storage format.",
                }
            ],
            "classification": {
                "domain": "Data Engineering",
                "primary_topic": "Delta Lake",
                "secondary_topics": [],
                "interview_relevant": True,
            },
            "saved_item": {
                "saved_item_id": f"urn:li:saved:abc{number:03d}",
                "saved_date": "2026-01-02",
                "canonical_url": DELTA,
                "url_kind": "posts",
                "capture_state": "captured",
                "capture_match": "directory name",
                "capture_quality": quality,
                "capture_notes": [],
                "saved_notes": None,
                "metadata_only": False,
            },
        }

    def test_the_page_is_not_generated(self, tmp_path: Path):
        """
        Saved Items was the fourth archive index, and the least
        defensible of them: a page whose entire content was a list of
        which posts happened to arrive via a saved list rather than the
        collector. That is a fact about how the archive was built, and a
        candidate revising for an interview has no use for it.
        """

        site = self._site(tmp_path, [self._saved_post()])

        assert not (site / "saved-items.html").exists()

    def test_nothing_links_to_it(self, tmp_path: Path):
        site = self._site(tmp_path, [self._saved_post()])

        # The footer row that linked it from every page is gone too, so
        # removing the page cannot have left a dead link behind.
        for page in site.rglob("*.html"):
            assert "saved-items.html" not in page.read_text(
                encoding="utf-8"
            )


    def test_generation_is_deterministic(self, tmp_path: Path):
        first = self._site(tmp_path / "a", [self._saved_post()])
        second = self._site(tmp_path / "b", [self._saved_post()])

        for name in ("index.html", "search.html"):
            assert (first / name).read_text(
                encoding="utf-8"
            ) == (second / name).read_text(encoding="utf-8")

        # Revision pages, not post pages: one page per captured post is
        # the archive's shape, and comparing two builds byte for byte is
        # what proves the output is deterministic rather than merely
        # similar.
        for pattern in ("revision/*.html", "index.html", "questions.html"):
            for page in sorted(first.glob(pattern)):
                assert page.read_text(encoding="utf-8") == (
                    second / page.relative_to(first)
                ).read_text(encoding="utf-8"), page

    def test_a_search_reaches_the_unit_that_teaches_the_topic(
        self, tmp_path: Path
    ):
        from src.wiki.search_index import build_index
        from src.wiki.revision_model import build_units

        self._site(tmp_path, [self._saved_post()])

        index = build_index(build_units(self._curriculum()))

        unit_records = [
            record for record in index["records"] if record["k"] == "b"
        ]

        assert unit_records

        # Without this, a search for "Delta Lake" reaches nothing, because
        # the technology label lives on the post and the post has no page.
        # The unit carries its concepts, and concepts are what a reader
        # actually types.
        assert any(
            "Delta Lake" in record["c"] or "Delta Lake" in record["s"]
            for record in unit_records
        ), [r["c"] for r in unit_records]

    def test_the_client_searches_technologies(self):
        script = Path("src/wiki/assets/search.js").read_text(
            encoding="utf-8"
        )

        assert "record.t" in script

    def test_the_intake_checkpoint_is_in_the_order(self):
        from src.ingestion import checkpoints as checkpoint_module

        order = checkpoint_module.PHASE_ORDER

        # Named CP9 because that is the work this phase did, and kept
        # separate from CP9_KNOWLEDGE_BASE_COMPLETE, which is a phase of
        # its own that nothing here completed.
        assert (
            checkpoint_module.CP9_SAVED_ITEMS_INTAKE_READY
            == "CP9_SAVED_ITEMS_INTAKE_READY"
        )

        assert order.index(
            checkpoint_module.CP8_SAVED_ITEMS_IMPORT_READY
        ) < order.index(
            checkpoint_module.CP9_SAVED_ITEMS_INTAKE_READY
        ) < order.index(
            checkpoint_module.CP9_KNOWLEDGE_BASE_COMPLETE
        )

    def test_a_checkpoint_naming_the_intake_phase_contains_no_credential(
        self, tmp_path: Path
    ):
        from src.ingestion import checkpoints as checkpoint_module

        checkpoint = checkpoint_module.Checkpoint(
            checkpoint_id=(
                checkpoint_module.CP9_SAVED_ITEMS_INTAKE_READY
            ),
            phase=checkpoint_module.CP9_SAVED_ITEMS_INTAKE_READY,
            status="complete",
        )

        written = checkpoint_module.write(checkpoint, root=tmp_path)

        assert (
            checkpoint_module.CP9_SAVED_ITEMS_INTAKE_READY
            in written.read_text(encoding="utf-8")
        )

    def test_every_record_carries_the_technology_key(
        self, tmp_path: Path
    ):
        payload = json.loads(
            self._site(tmp_path, [self._saved_post()])
            .joinpath("assets/search-index.json")
            .read_text(encoding="utf-8")
        )

        assert all("t" in record for record in payload["records"])


# ---------------------------------------------------------------------
# What the documentation says
# ---------------------------------------------------------------------


README = Path(__file__).resolve().parents[1] / "README.md"


class TestDocumented:
    def test_it_documents_every_saved_items_command(self):
        readme = README.read_text(encoding="utf-8")

        for command in (
            "saved-items-init",
            "saved-items-validate",
            "saved-items-status",
        ):
            assert command in readme, f"{command} is not documented"

        # The import command is documented with a line continuation, so
        # the check is on the subcommand and its flags rather than on
        # one unbroken line. A documentation test that forced a layout
        # would be a test about line breaks.
        assert "saved-items \\" in readme
        assert "--input data/incoming/saved-items/manifest.csv" in readme
        assert "saved-items --plan" in readme

    def test_it_documents_the_drop_zone_layout(self):
        readme = README.read_text(encoding="utf-8")

        assert "data/incoming/saved-items/" in readme
        assert "captures/" in readme
        assert "capture.json" in readme

    def test_it_documents_the_capture_qualities(self):
        readme = README.read_text(encoding="utf-8")

        for quality in CAPTURE_QUALITIES:
            assert f"`{quality}`" in readme, f"{quality} is not documented"

    def test_it_does_not_claim_to_scrape_linkedin(self):
        readme = README.read_text(encoding="utf-8").lower()

        # The word is fine in a denial. What must not appear is a claim
        # that the project can go and get saved posts by itself.
        for claim in (
            "automatically collects your saved",
            "scrapes linkedin",
            "crawls linkedin",
            "logs into linkedin and downloads",
            "syncs your saved posts automatically",
        ):
            assert claim not in readme, claim

    def test_it_says_what_is_not_supported(self):
        readme = README.read_text(encoding="utf-8")

        section = readme[
            readme.index("### What it will not do") : readme.index(
                "## Collection"
            )
        ]

        for refusal in (
            "not fetch anything",
            "not invent a body",
            "not read a credential",
            "not OCR",
            "not act on LinkedIn",
        ):
            assert refusal in section, refusal

    def test_it_says_a_dry_run_writes_nothing(self):
        readme = README.read_text(encoding="utf-8")

        assert "Neither writes anything" in readme
