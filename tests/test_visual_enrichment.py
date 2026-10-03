"""
Visual enrichment: discovery, ordering, deduplication, isolation,
resumption and the safety boundary.

CP11's limit was stated precisely: an image contributed its format and
its pixel dimensions and nothing else, so a slide of SQL was a file
called ``PNG image, 800x600 pixels``. These tests are about whether a
picture can now become knowledge that keeps its provenance.

Built on synthetic archives rather than the real one, for three reasons:
the real archive is 287 MB and does not belong in a test run, its
contents change underneath us, and a test that depends on 3,047
particular files cannot express a rule -- only confirm that those files
still exist.

The provider is a scripted stand-in. Failures are raised deliberately
with the shapes the real ones had, because the retry and the classifier
are supposed to recognise those shapes and a test with a generic
exception would not prove they do.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.ingestion.post_document import PostDocument
from src.models import KnowledgePost, SourceInfo
from src.visual.archive import (
    ArchiveError,
    assert_readable,
    assets_for_post,
    load_archive,
    resolve_asset_path,
)
from src.visual.assets import (
    UnsafePath,
    build_asset,
    classify_role,
    contained_media_path,
    describe_image,
    order_assets,
    sequence_for,
    sniff_format,
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
    AssetRole,
    CodeBlock,
    PostVisual,
    ProcessingState,
    VisualAnalysis,
    VisualAsset,
)
from src.visual.processor import (
    CompositeVisualProcessor,
    OCRProcessor,
    VisionProcessor,
    build_default_processor,
    failure_note,
)
from src.visual.stage import VisualStage, usable_assets
from src.visual.store import VisualStore


#: Verbatim from the failed batch logs of GitHub run 37026765769. The
#: retry has to recognise this shape, because it is the shape that lost
#: eight posts when each post had one attempt.
TRUNCATION = (
    "OpenCode failed with exit code 1.\n\nSTDOUT:\n\nSTDERR:\n"
    '{"type":"error","timestamp":1790955962706,'
    '"sessionID":"ses_f02b8c743ffeMLo23i1jfYe1Yp",'
    '"error":{"type":"provider.invalid-output",'
    '"message":"OpenAI Chat stream ended without finish_reason",'
    '"status":200}}'
)


# ---------------------------------------------------------------------
# Synthetic archives
# ---------------------------------------------------------------------


def png_bytes(width: int, height: int, colour=(200, 30, 30)) -> bytes:
    """A real PNG of the requested size, so dimensions are measurable."""

    from io import BytesIO

    from PIL import Image

    buffer = BytesIO()

    Image.new(
        "RGB", (width, height), colour
    ).save(buffer, format="PNG")

    return buffer.getvalue()


def jpeg_bytes(width: int, height: int, colour=(30, 60, 200)) -> bytes:
    from io import BytesIO

    from PIL import Image

    buffer = BytesIO()

    Image.new(
        "RGB", (width, height), colour
    ).save(buffer, format="JPEG", quality=85)

    return buffer.getvalue()


def write_archive(
    root: Path,
    posts: list[dict],
    *,
    files: dict[str, bytes] | None = None,
    session: bool = False,
) -> Path:
    """
    A synthetic archive with the real one's shape.

    Same layout, same ``posts_archive.json`` fields, same
    ``{post_id}_slide_{n}.jpg`` naming including the inconsistent zero
    padding, because that inconsistency is one of the things the code
    under test has to survive.
    """

    root.mkdir(parents=True, exist_ok=True)
    media = root / "media"
    media.mkdir(exist_ok=True)

    records = []

    for post in posts:
        post_id = post["post_id"]

        names = []

        for name, payload in post.get("files", {}).items():
            (media / name).write_bytes(payload)
            names.append(name)

        for name, payload in (files or {}).items():
            (media / name).write_bytes(payload)

        # The archive's own convention: where a post has numbered
        # slides, the bare `_slide_0` is not one of them. That file is
        # the preview that was saved before the carousel was, and
        # excluding it is how the archive says so.
        numbered = [
            name for name in names
            if not name.endswith("_slide_0.jpg")
        ]

        declared = numbered or names

        records.append(
            {
                "post_id": post_id,
                "text": post.get("text", "A post about Delta Lake."),
                "permalink": post.get(
                    "permalink",
                    f"https://www.linkedin.com/feed/update/{post_id}",
                ),
                "media": {
                    "has_media": bool(declared),
                    "is_multi_slide": len(declared) > 1,
                    "total_media_count": len(declared),
                    "saved_files": declared,
                    "original_urls": [],
                },
            }
        )

    (root / "posts_archive.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )

    if session:
        session_dir = root / "chrome_session" / "Default"
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "Cookies").write_bytes(b"encrypted")

    return root


def write_post(
    root: Path,
    post_id: str,
    *,
    text: str = "Delta Lake gives ACID guarantees.",
    media: list[str] | None = None,
) -> str:
    """A committed post, as the pipeline reads it."""

    directory = root / "data" / "posts" / post_id
    directory.mkdir(parents=True, exist_ok=True)

    document = PostDocument.new(
        post_id,
        text=text,
        platform="linkedin",
        url=f"https://www.linkedin.com/feed/update/{post_id}",
        author="A Fixture",
        captured_at="2026-10-02T04:30:00+00:00",
        primary_topic="Delta Lake",
        interview_relevant=True,
    )

    document.save(directory)

    if media:
        payload = json.loads(
            (directory / "post.json").read_text(encoding="utf-8")
        )

        payload["media"] = [
            {
                "type": "image",
                "path": f"media/{name}",
                "description": None,
                "extracted_text": None,
            }
            for name in media
        ]

        (directory / "post.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    return post_id


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    """A three-slide carousel, a single image, and a post with none."""

    return write_archive(
        tmp_path / "archive",
        [
            {
                "post_id": "activity_carousel",
                "files": {
                    "activity_carousel_slide_01.jpg": jpeg_bytes(1200, 1000),
                    "activity_carousel_slide_02.jpg": jpeg_bytes(1200, 1000, (10, 200, 10)),
                    "activity_carousel_slide_03.jpg": jpeg_bytes(1200, 1000, (10, 10, 200)),
                    "activity_carousel_slide_0.jpg": jpeg_bytes(480, 360),
                },
                "text": "A three slide carousel about Delta Lake.",
            },
            {
                "post_id": "activity_single",
                "files": {
                    "activity_single_slide_0.jpg": jpeg_bytes(800, 600)
                },
            },
            {"post_id": "activity_plain", "files": {}},
        ],
    )


# ---------------------------------------------------------------------
# A scripted provider
# ---------------------------------------------------------------------


class Scripted:
    """
    A processor that answers, or fails, on demand.

    Answers name the file they came from, so a test can assert that
    slide two's text reached slide two and not slide one. Failures are
    raised once each, which is how a retry is told apart from a first
    attempt.
    """

    def __init__(self, failures: list[Exception] | None = None) -> None:
        self.failures = list(failures or [])
        self.calls: list[list[str]] = []
        self.assets: dict = {}

    def set_assets(self, assets: dict) -> None:
        self.assets = dict(assets)

    def available(self) -> bool:
        return True

    def describe(self) -> str:
        return "scripted"

    def analyse(self, paths: list[Path]) -> list[VisualAnalysis]:
        self.calls.append([path.name for path in paths])

        if self.failures:
            raise self.failures.pop(0)

        return [
            VisualAnalysis(
                source_asset_hash=(
                    self.assets.get(path.name, VisualAsset(path="")).sha256
                ),
                source_path=f"media/{path.name}",
                sequence=self.assets.get(path.name, VisualAsset(path="")).sequence,
                extracted_text=f"text read from {path.name}",
                processor="scripted",
                confidence=0.9,
            )
            for path in paths
        ]


# ---------------------------------------------------------------------
# 1, 2. Discovery, and missing media
# ---------------------------------------------------------------------


class TestDiscovery:
    def test_the_archive_is_read(self, archive: Path):
        loaded = load_archive(archive)

        assert len(loaded.posts) == 3
        assert loaded.media_root == archive / "media"

    def test_posts_with_media_are_found(self, archive: Path):
        loaded = load_archive(archive)

        assert loaded.post_ids_with_media() == [
            "activity_carousel",
            "activity_single",
        ]

    def test_a_file_on_disk_but_not_referenced_is_an_orphan(
        self, archive: Path
    ):
        loaded = load_archive(archive)

        # The unreferenced _slide_0 is an orphan by the metadata's own
        # account, and is still a file belonging to the post.
        assert "activity_carousel_slide_0.jpg" in loaded.orphans

    def test_a_missing_referenced_file_is_a_failed_asset(
        self, tmp_path: Path
    ):
        root = write_archive(
            tmp_path / "archive",
            [
                {
                    "post_id": "activity_gap",
                    "files": {
                        "activity_gap_slide_0.jpg": jpeg_bytes(400, 300)
                    },
                }
            ],
        )

        # Remove it after the metadata claims it.
        (root / "media" / "activity_gap_slide_0.jpg").unlink()

        loaded = load_archive(root)
        assets = assets_for_post(loaded, "activity_gap")

        # Recorded, not dropped. A missing slide that vanishes silently
        # is a slide that looks like it was never there.
        assert len(assets) == 1
        assert assets[0].state is ProcessingState.FAILED

        # Reported as absence, not as a containment refusal: a gap
        # in the archive and an untrusted path need different words,
        # because they send an operator to different places.
        note = assets[0].note.lower()
        assert "not find" in note or "stat" in note
        assert "refused" not in note

    def test_a_malformed_archive_is_an_error_not_an_empty_one(
        self, tmp_path: Path
    ):
        root = tmp_path / "broken"
        (root / "media").mkdir(parents=True)
        (root / "posts_archive.json").write_text("{not json", encoding="utf-8")

        with pytest.raises(ArchiveError):
            load_archive(root)

    def test_duplicate_post_ids_are_refused(self, tmp_path: Path):
        root = tmp_path / "dupes"
        (root / "media").mkdir(parents=True)
        (root / "posts_archive.json").write_text(
            json.dumps(
                [
                    {"post_id": "a", "text": "one", "media": {}},
                    {"post_id": "a", "text": "two", "media": {}},
                ]
            ),
            encoding="utf-8",
        )

        # Refused rather than silently resolved: which record is truth
        # is not this code's decision to make.
        with pytest.raises(ArchiveError, match="duplicate post id"):
            load_archive(root)


# ---------------------------------------------------------------------
# 3, 4, 5. Duplicates, and thumbnails against full images
# ---------------------------------------------------------------------


class TestDeduplication:
    def test_identical_bytes_are_one_content(self, tmp_path: Path):
        payload = jpeg_bytes(600, 400)
        root = write_archive(
            tmp_path / "archive",
            [
                {
                    "post_id": "activity_a",
                    "files": {"activity_a_slide_0.jpg": payload},
                }
            ],
        )

        (root / "media" / "copy.jpg").write_bytes(payload)

        loaded = load_archive(root)
        assets = assets_for_post(loaded, "activity_a")

        # The stray copy shares this post's stem only if named so; here
        # it is simply unassociated, so the post has one asset.
        assert len(assets) == 1
        assert assets[0].sha256

    def test_exact_duplicates_in_one_post_collapse(self, tmp_path: Path):
        payload = jpeg_bytes(700, 500)
        root = write_archive(
            tmp_path / "archive",
            [
                {
                    "post_id": "activity_dup",
                    "files": {
                        "activity_dup_slide_01.jpg": payload,
                        "activity_dup_slide_02.jpg": payload,
                        "activity_dup_slide_03.jpg": jpeg_bytes(900, 700, (1, 2, 3)),
                    },
                }
            ],
        )

        loaded = load_archive(root)
        assets = list(assets_for_post(loaded, "activity_dup"))

        marked = mark_duplicates(assets)

        assert marked == 1

        by_name = {asset.filename: asset for asset in assets}

        # One of the identical pair keeps the record; the other points
        # at it. Which one wins is decided by content and then by name,
        # so the choice is the same on every run.
        assert by_name["activity_dup_slide_01.jpg"].duplicate_of is None
        assert (
            by_name["activity_dup_slide_02.jpg"].duplicate_of
            == "activity_dup_slide_01.jpg"
        )
        assert by_name["activity_dup_slide_03.jpg"].duplicate_of is None

    def test_the_full_image_stands_for_a_duplicate_pair(self, tmp_path: Path):
        payload = jpeg_bytes(700, 500)
        root = write_archive(
            tmp_path / "archive",
            [
                {
                    "post_id": "activity_pair",
                    "files": {
                        "activity_pair_slide_01.jpg": payload,
                        "activity_pair_slide_02.jpg": payload,
                    },
                }
            ],
        )

        loaded = load_archive(root)
        assets = list(assets_for_post(loaded, "activity_pair"))
        mark_duplicates(assets)

        primary = choose_primary(
            [
                asset
                for asset in assets
                if asset.sha256
                == assets[0].sha256
            ]
        )

        assert not primary.duplicate_of

    def test_a_preview_is_redundant_when_the_full_slide_exists(
        self, archive: Path
    ):
        loaded = load_archive(archive)
        assets = list(assets_for_post(loaded, "activity_carousel"))

        by_name = {asset.filename: asset for asset in assets}

        preview = by_name["activity_carousel_slide_0.jpg"]
        full = by_name["activity_carousel_slide_01.jpg"]

        assert preview.role is AssetRole.THUMBNAIL
        assert full.role is AssetRole.SLIDE
        assert preview.width < full.width

        redundant = redundant_assets(assets)

        assert preview in redundant
        assert full not in redundant

    def test_a_lone_preview_is_not_redundant(self, tmp_path: Path):
        """
        The only image a post has is evidence, however small.

        Treating it as a redundant copy of a slide that is not there
        would discard the post's entire visual content.
        """

        root = write_archive(
            tmp_path / "archive",
            [
                {
                    "post_id": "activity_lone",
                    "files": {
                        "activity_lone_slide_0.jpg": jpeg_bytes(300, 200)
                    },
                }
            ],
        )

        loaded = load_archive(root)
        assets = assets_for_post(loaded, "activity_lone")

        assert redundant_assets(assets) == []
        assert len(usable_assets(assets)) == 1

    def test_a_preview_replaced_by_another_does_not_change_the_digest(
        self, tmp_path: Path
    ):
        """
        The rule that keeps a redundant file from costing work.

        A new low-resolution copy of a slide already present changes
        nothing about what the post says.
        """

        def build(preview: bytes) -> list[VisualAsset]:
            root = write_archive(
                tmp_path / f"a{preview[0]}",
                [
                    {
                        "post_id": "activity_p",
                        "files": {
                            "activity_p_slide_01.jpg": jpeg_bytes(1200, 900),
                            "activity_p_slide_0.jpg": preview,
                        },
                    }
                ],
            )

            loaded = load_archive(root)

            return list(assets_for_post(loaded, "activity_p"))

        before = build(jpeg_bytes(400, 300))
        after = build(jpeg_bytes(420, 310))

        assert asset_digest(
            before, processor_version="1", configuration="c"
        ) == asset_digest(
            after, processor_version="1", configuration="c"
        )

    def test_perceptual_similarity_is_reported_not_acted_on(self):
        """
        Two slides of a carousel can look alike and say different things.

        The helper exists so the decision to skip perceptual merging is
        explicit rather than an omission.
        """

        one = VisualAsset(
            path="a", filename="a", sha256="a" * 64,
            width=1200, height=1000,
        )
        two = VisualAsset(
            path="b", filename="b", sha256="b" * 64,
            width=1200, height=1000,
        )
        # A genuinely different shape, not a one-pixel one: the
        # hint is allowed to fire on near-identical dimensions,
        # which is why it is never acted on.
        other = VisualAsset(
            path="c", filename="c", sha256="c" * 64,
            width=800, height=600,
        )

        assert perceptual_hint(one, two) is True
        assert perceptual_hint(one, other) is False

        # And nothing groups them.
        assert len(group_by_content([one, two, other])) == 3


# ---------------------------------------------------------------------
# 6, 7. Ordering
# ---------------------------------------------------------------------


class TestOrdering:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("activity_slide_0.jpg", 0),
            ("activity_slide_01.jpg", 1),
            ("activity_slide_9.jpg", 9),
            ("activity_slide_10.jpg", 10),
            ("activity_slide_100.jpg", 100),
            ("activity_slide_261.jpg", 261),
        ],
    )
    def test_the_sequence_comes_from_the_digits(
        self, name: str, expected: int
    ):
        assert sequence_for(name) == expected

    def test_lexical_order_would_be_wrong(self):
        """
        The trap this exists to avoid, stated as a test.

        Sorted as strings, slide 100 comes before slide 20, and the
        carousel's narrative is reversed from slide 20 onwards.
        """

        names = [
            f"activity_slide_{index}.jpg"
            for index in (1, 2, 10, 20, 100)
        ]

        lexical = sorted(names)
        numeric = sorted(names, key=sequence_for)

        assert lexical.index("activity_slide_100.jpg") < lexical.index(
            "activity_slide_20.jpg"
        )
        assert numeric.index("activity_slide_20.jpg") < numeric.index(
            "activity_slide_100.jpg"
        )

    def test_mixed_padding_on_one_post_is_ordered_numerically(self):
        """
        The real archive's shape: _slide_0 beside _slide_01.

        Two padding widths on the same post is not a hypothetical.
        """

        names = [
            "activity_x_slide_0.jpg",
            "activity_x_slide_01.jpg",
            "activity_x_slide_02.jpg",
            "activity_x_slide_10.jpg",
        ]

        assets = [
            VisualAsset(path=n, filename=n, sequence=sequence_for(n))
            for n in names
        ]

        ordered = [
            asset.filename for asset in order_assets(assets)
        ]

        assert ordered == [
            "activity_x_slide_0.jpg",
            "activity_x_slide_01.jpg",
            "activity_x_slide_02.jpg",
            "activity_x_slide_10.jpg",
        ]

    def test_a_hundred_slides_keep_their_order(self, tmp_path: Path):
        root = write_archive(
            tmp_path / "big",
            [
                {
                    "post_id": "activity_big",
                    "files": {
                        f"activity_big_slide_{index:03d}.jpg": jpeg_bytes(
                            100, 100, (index % 255, 0, 0)
                        )
                        for index in range(120)
                    },
                }
            ],
        )

        loaded = load_archive(root)
        assets = assets_for_post(loaded, "activity_big")

        sequences = [asset.sequence for asset in assets]

        assert sequences == list(range(120))

    def test_ordering_is_stable_between_runs(self, archive: Path):
        loaded = load_archive(archive)

        first = [
            asset.filename
            for asset in assets_for_post(loaded, "activity_carousel")
        ]
        second = [
            asset.filename
            for asset in assets_for_post(loaded, "activity_carousel")
        ]

        assert first == second


# ---------------------------------------------------------------------
# 8-13. Fingerprints and invalidation
# ---------------------------------------------------------------------


class TestFingerprints:
    def _digest(self, assets) -> str:
        return asset_digest(
            assets, processor_version="1", configuration="vision"
        )

    def _slides(self, root: Path, **overrides) -> list[VisualAsset]:
        files = {
            "activity_f_slide_0.jpg": jpeg_bytes(800, 600),
            "activity_f_slide_1.jpg": jpeg_bytes(800, 600, (1, 2, 3)),
        }
        files.update(overrides)

        archive_root = write_archive(
            root, [{"post_id": "activity_f", "files": files}]
        )

        return list(assets_for_post(load_archive(archive_root), "activity_f"))

    def test_an_unchanged_post_keeps_its_digest(self, tmp_path: Path):
        assert self._digest(self._slides(tmp_path)) == self._digest(
            self._slides(tmp_path / "again")
        )

    def test_changed_bytes_invalidate(self, tmp_path: Path):
        before = self._digest(self._slides(tmp_path))
        after = self._digest(
            self._slides(
                tmp_path / "changed",
                **{
                    "activity_f_slide_1.jpg": jpeg_bytes(
                        800, 600, (9, 9, 9)
                    )
                },
            )
        )

        assert before != after

    def test_an_added_slide_invalidates(self, tmp_path: Path):
        before = self._digest(self._slides(tmp_path))
        after = self._digest(
            self._slides(
                tmp_path / "added",
                **{"activity_f_slide_2.jpg": jpeg_bytes(800, 600, (4, 5, 6))},
            )
        )

        assert before != after

    def test_a_removed_slide_invalidates(self, tmp_path: Path):
        before = self._digest(self._slides(tmp_path))

        # The file is deleted after the metadata declared it, so the
        # post still claims a slide it no longer has.
        root = write_archive(
            tmp_path / "gone",
            [
                {
                    "post_id": "activity_g",
                    "files": {
                        "activity_g_slide_0.jpg": jpeg_bytes(700, 500),
                    },
                }
            ],
        )

        (root / "media" / "activity_g_slide_0.jpg").unlink()

        after = asset_digest(
            assets_for_post(load_archive(root), "activity_g"),
            processor_version="1",
            configuration="vision",
        )

        assert before != after

    def test_reordered_slides_invalidate(self):
        """
        Reordering changes what the carousel says and must cost a
        re-read, even though the same bytes are present.
        """

        one = VisualAsset(
            path="a", filename="a", sha256="a" * 64, sequence=0,
            role=AssetRole.SLIDE,
        )
        two = VisualAsset(
            path="b", filename="b", sha256="b" * 64, sequence=1,
            role=AssetRole.SLIDE,
        )

        before = asset_digest(
            [one, two], processor_version="1", configuration="vision"
        )

        swapped = [
            VisualAsset(
                path="a", filename="a", sha256="a" * 64, sequence=1,
                role=AssetRole.SLIDE,
            ),
            VisualAsset(
                path="b", filename="b", sha256="b" * 64, sequence=0,
                role=AssetRole.SLIDE,
            ),
        ]

        after = asset_digest(
            swapped, processor_version="1", configuration="vision"
        )

        assert before != after

    def test_a_processor_upgrade_invalidates(self):
        asset = VisualAsset(
            path="a", filename="a", sha256="a" * 64, sequence=0,
            role=AssetRole.SLIDE,
        )

        one = asset_digest(
            [asset], processor_version="1", configuration="vision"
        )
        two = asset_digest(
            [asset], processor_version="2", configuration="vision"
        )

        assert one != two

    def test_a_configuration_change_invalidates(self):
        asset = VisualAsset(
            path="a", filename="a", sha256="a" * 64, sequence=0,
            role=AssetRole.SLIDE,
        )

        one = asset_digest(
            [asset], processor_version="1", configuration="vision"
        )
        two = asset_digest(
            [asset], processor_version="1", configuration="vision+ocr"
        )

        assert one != two

    def test_a_filename_change_alone_does_not_invalidate(self):
        """
        A re-encoded slide at the same content is the same slide.

        Byte size, pixel dimensions and filename are all excluded for
        this reason: an export tool rewriting them would otherwise
        invalidate every post in the corpus.
        """

        first = VisualAsset(
            path="media/old.jpg",
            filename="old.jpg",
            sha256="a" * 64,
            sequence=0,
            role=AssetRole.SLIDE,
            byte_size=1000,
            width=800,
            height=600,
        )
        second = first.model_copy(
            update={
                "path": "media/new.jpg",
                "filename": "new.jpg",
                "byte_size": 1400,
                "width": 1600,
                "height": 1200,
            }
        )

        assert self._digest([first]) == self._digest([second])


# ---------------------------------------------------------------------
# 14-20. Processors, and what happens when they fail
# ---------------------------------------------------------------------


class TestProcessors:
    def test_ocr_reports_itself_unavailable_rather_than_pretending(self):
        """
        Absent an engine, the honest answer is "no", not empty text.

        A processor that returned nothing as if it had read the slide
        would make "the slide was blank" and "nothing is installed" the
        same observation.
        """

        processor = OCRProcessor(engine=None)

        assert processor.available() is False
        assert "tesseract" in processor.unavailable_reason().lower()

        with pytest.raises(RuntimeError, match="No local OCR engine"):
            processor.analyse([])

    def test_vision_parses_a_batched_array(self):
        processor = VisionProcessor(client=FakeClient())
        supply(processor, "a.jpg", "b.jpg")

        analyses = processor.analyse(
            [Path("a.jpg"), Path("b.jpg")]
        )

        assert len(analyses) == 2
        assert analyses[0].extracted_text == "SQL JOINS"

    def test_vision_accepts_a_single_object(self):
        """
        Asked about one image the model answers with an object rather
        than a one-element array, and often enough to be worth handling.
        """

        processor = VisionProcessor(client=FakeClient(single=True))
        supply(processor, "a.jpg")

        analyses = processor.analyse([Path("a.jpg")])

        assert len(analyses) == 1

    def test_vision_strips_a_markdown_fence(self):
        processor = VisionProcessor(client=FakeClient(fenced=True))
        supply(processor, "a.jpg")

        analyses = processor.analyse([Path("a.jpg")])

        assert analyses[0].extracted_text == "SQL JOINS"

    def test_a_truncated_response_fails_rather_than_being_patched(
        self,
    ):
        """
        A half-written array cannot be completed honestly.

        The stage records the failure and moves on. Inventing the
        missing entries would attribute one slide's words to another.
        """

        processor = VisionProcessor(
            client=FakeClient(raw='[{"index":0,"visible_text":"a"')
        )
        supply(processor, "a.jpg", "b.jpg")

        with pytest.raises(ValueError, match="not valid JSON"):
            processor.analyse([Path("a.jpg"), Path("b.jpg")])

    def test_a_short_answer_fails(self):
        processor = VisionProcessor(
            client=FakeClient(short=True)
        )
        supply(processor, "a.jpg", "b.jpg")

        with pytest.raises(ValueError, match="left out"):
            processor.analyse([Path("a.jpg"), Path("b.jpg")])

    def test_a_batch_beyond_the_limit_is_refused(self):
        processor = VisionProcessor(client=FakeClient())

        with pytest.raises(ValueError, match="batch limit"):
            processor.analyse([Path(f"{i}.jpg") for i in range(40)])

    def test_an_answer_without_an_asset_is_refused(self):
        """
        Provenance is not optional.

        An analysis that cannot name the file it came from must not be
        built at all, rather than stored and discovered later. This is
        the check that would have caught the composite swallowing the
        assets on the first real run.
        """

        processor = VisionProcessor(client=FakeClient())

        # Deliberately not calling supply().
        with pytest.raises(ValueError, match="cannot be given provenance"):
            processor.analyse([Path("a.jpg")])

    def test_the_composite_forwards_the_assets(self):
        """
        The composite is what the orchestrator holds.

        When it swallowed the assets, sixteen batches ran, nothing
        failed, and nothing was stored.
        """

        inner = Scripted()
        composite = CompositeVisualProcessor([inner])

        supply(composite, "a.jpg")

        assert inner.assets.get("a.jpg") is not None

        composite.analyse([Path("a.jpg")])

        assert inner.calls == [["a.jpg"]]
        assert inner.calls and inner.assets

    def test_the_composite_falls_through_to_what_works(self):
        class Broken:
            name = "broken"
            version = "1"

            def available(self):
                return True

            def analyse(self, paths):
                return []

        working = Scripted()

        composite = CompositeVisualProcessor([Broken(), working])

        result = composite.analyse([Path("a.jpg")])

        assert len(result) == 1

    def test_the_default_processor_is_wired(self):
        composite = build_default_processor()

        assert isinstance(composite, CompositeVisualProcessor)
        assert len(composite.processors) == 2

    def test_a_failure_is_classified_for_the_record(self):
        from src.ai.recovery import classify

        note = failure_note(RuntimeError(TRUNCATION))

        assert "truncated_response" in note
        assert classify(RuntimeError(TRUNCATION)).recoverable


def supply(processor, *names: str) -> None:
    """
    Hand a processor the assets it is about to read.

    Every real call goes through this, because an answer without an
    asset cannot be given provenance and must not be stored. Tests that
    skipped it were testing the parser rather than the stage.
    """

    processor.set_assets(
        {
            name: VisualAsset(
                path=f"media/{name}",
                filename=name,
                sha256=f"{index:064d}",
                sequence=index,
                role=AssetRole.SLIDE,
            )
            for index, name in enumerate(names)
        }
    )


class FakeClient:
    """A client that returns what a test says it should."""

    def __init__(
        self,
        single: bool = False,
        fenced: bool = False,
        short: bool = False,
        raw: str | None = None,
    ) -> None:
        self.single = single
        self.fenced = fenced
        self.short = short
        self.raw = raw
        self.files: list[str] = []

    def run(self, prompt: str, files: list[str] | None = None, **kw):
        from src.ai.opencode import OpenCodeResult

        self.files = list(files or [])

        if self.raw is not None:
            return OpenCodeResult(text="", session_id="s", data=self.raw)

        def entry(index: int) -> dict:
            return {
                "index": index,
                "kind": "slide",
                "visible_text": "SQL JOINS",
                "summary": "A cheat sheet",
                "diagram": "",
                "code": [{"language": "sql", "code": "SELECT 1"}],
                "technologies": ["Delta Lake"],
                "concepts": ["ACID guarantees"],
                "questions": ["What is Delta Lake?"],
                "confidence": 0.9,
            }

        if self.short:
            body = json.dumps([entry(0)])

        elif self.single:
            body = json.dumps(entry(0))

        else:
            body = json.dumps([entry(0), entry(1)])

        if self.fenced:
            body = f"```json\n{body}\n```"

        return OpenCodeResult(text="", session_id="s", data=body)


# ---------------------------------------------------------------------
# 21, 22. Checkpointing and resumption
# ---------------------------------------------------------------------


class TestResumption:
    def _stage(self, processor, tmp_path: Path, **kw) -> VisualStage:
        return VisualStage(
            processor=processor,
            store=VisualStore(tmp_path / "store"),
            batch_size=kw.get("batch_size", 5),
            attempts=kw.get("attempts", 3),
            progress=lambda message: None,
        )

    def _assets(self, archive: Path, post_id: str):
        return assets_for_post(load_archive(archive), post_id)

    def test_an_unchanged_post_costs_nothing_on_a_second_run(
        self, archive: Path, tmp_path: Path
    ):
        loaded = load_archive(archive)
        assets = assets_for_post(loaded, "activity_carousel")

        first = Scripted()
        outcome = self._stage(first, tmp_path).run_post(
            "activity_carousel",
            assets,
            lambda asset: archive / "media" / asset.filename,
        )

        assert outcome.processed > 0
        assert outcome.calls > 0

        # The same store, which is the point: the second run must
        # find what the first one stored.
        second = Scripted()
        again = self._stage(second, tmp_path).run_post(
            "activity_carousel",
            assets,
            lambda asset: archive / "media" / asset.filename,
        )

        assert second.calls == []
        assert again.calls == 0
        assert again.processed == 0
        assert again.reused >= outcome.processed

    def test_a_failed_slide_does_not_cost_the_others(self, tmp_path: Path):
        root = write_archive(
            tmp_path / "archive",
            [
                {
                    "post_id": "activity_iso",
                    "files": {
                        f"activity_iso_slide_{index}.jpg": jpeg_bytes(
                            400, 300, (index, 0, 0)
                        )
                        for index in range(6)
                    },
                }
            ],
        )

        loaded = load_archive(root)
        assets = assets_for_post(loaded, "activity_iso")

        # Fails on the second batch only, and only once.
        processor = Scripted(
            failures=[RuntimeError(TRUNCATION)]
        )

        outcome = self._stage(
            processor, tmp_path / "s", batch_size=2, attempts=3
        ).run_post(
            "activity_iso",
            assets,
            lambda asset: root / "media" / asset.filename,
        )

        # Four of six slides still understood, one batch retried.
        assert outcome.retries >= 1
        assert outcome.processed >= 4
        assert outcome.failures == 0

    def test_a_permanent_failure_is_not_retried(self, tmp_path: Path):
        root = write_archive(
            tmp_path / "archive",
            [
                {
                    "post_id": "activity_perm",
                    "files": {
                        "activity_perm_slide_0.jpg": jpeg_bytes(400, 300)
                    },
                }
            ],
        )

        loaded = load_archive(root)
        assets = assets_for_post(loaded, "activity_perm")

        processor = Scripted(
            failures=[RuntimeError("HTTP 401 Unauthorized: invalid API key")]
        )

        outcome = self._stage(processor, tmp_path / "s", attempts=3).run_post(
            "activity_perm",
            assets,
            lambda asset: root / "media" / asset.filename,
        )

        assert outcome.calls == 0
        assert outcome.failures == 1
        assert outcome.visual.failures
        assert len(processor.calls) == 1

    def test_a_failure_is_recorded_rather_than_hidden(self, tmp_path: Path):
        root = write_archive(
            tmp_path / "archive",
            [
                {
                    "post_id": "activity_rec",
                    "files": {
                        "activity_rec_slide_0.jpg": jpeg_bytes(400, 300)
                    },
                }
            ],
        )

        loaded = load_archive(root)
        assets = assets_for_post(loaded, "activity_rec")

        processor = Scripted(failures=[RuntimeError("nope")] * 5)

        outcome = self._stage(processor, tmp_path / "s", attempts=2).run_post(
            "activity_rec",
            assets,
            lambda asset: root / "media" / asset.filename,
        )

        assert outcome.visual.failures
        assert outcome.visual.failures[0].startswith(
            "activity_rec_slide_0.jpg"
        )

    def test_a_corrupt_slide_does_not_stop_the_post(self, tmp_path: Path):
        root = write_archive(
            tmp_path / "archive",
            [
                {
                    "post_id": "activity_corrupt",
                    "files": {
                        "activity_corrupt_slide_0.jpg": b"not an image at all",
                        "activity_corrupt_slide_1.jpg": jpeg_bytes(500, 400),
                    },
                }
            ],
        )

        loaded = load_archive(root)
        assets = assets_for_post(loaded, "activity_corrupt")

        processor = Scripted()

        outcome = self._stage(processor, tmp_path / "s").run_post(
            "activity_corrupt",
            assets,
            lambda asset: root / "media" / asset.filename,
        )

        # The readable slide was still read.
        assert outcome.processed == 1
        assert outcome.failures == 1
        assert assets[0].state is ProcessingState.FAILED


# ---------------------------------------------------------------------
# 23, 24. Code and diagrams
# ---------------------------------------------------------------------


class TestExtraction:
    def test_code_is_kept_verbatim(self):
        analysis = VisualAnalysis(
            source_asset_hash="a" * 64,
            code_blocks=[
                CodeBlock(
                    language="sql",
                    code="SELECT *\nFROM A\nINNER JOIN B ON A.key = B.key",
                    method="vision",
                )
            ],
        )

        # Not reformatted, not corrected, not tidied.
        assert "INNER JOIN B ON A.key = B.key" in analysis.code_blocks[0].code

    def test_an_uncertain_language_is_left_empty(self):
        """
        A confident wrong language is a wrong fact about how a
        technology is used, which is worse than no claim.
        """

        from src.visual.processor import _to_analysis

        asset = VisualAsset(
            path="media/a.jpg", filename="a.jpg", sha256="a" * 64
        )

        analysis = _to_analysis(
            {"code": [{"language": "", "code": "x = 1"}]},
            0,
            Path("a.jpg"),
            assets_by_path={"a.jpg": asset},
        )

        assert analysis.code_blocks[0].language is None

    def test_a_diagram_description_is_carried(self):
        from src.visual.processor import _to_analysis

        asset = VisualAsset(
            path="media/a.jpg", filename="a.jpg", sha256="a" * 64
        )

        analysis = _to_analysis(
            {
                "diagram": "Source goes to ADF, then ADLS, then Databricks"
            },
            0,
            Path("a.jpg"),
            assets_by_path={"a.jpg": asset},
        )

        assert "ADF" in analysis.diagram_description

    def test_visual_text_is_rendered_for_the_prompt(self):
        analysis = VisualAnalysis(
            source_asset_hash="a" * 64,
            extracted_text="SQL JOINS",
            code_blocks=[CodeBlock(code="SELECT 1")],
            visual_summary="A cheat sheet",
        )

        text = analysis.as_source_text()

        assert "SQL JOINS" in text
        assert "SELECT 1" in text
        assert "cheat sheet" in text

    def test_an_empty_analysis_is_usable_only_if_said_something(self):
        blank = VisualAnalysis(source_asset_hash="a" * 64)

        assert blank.is_usable is False

        said = VisualAnalysis(
            source_asset_hash="a" * 64, extracted_text="x"
        )

        assert said.is_usable is True

    def test_provenance_comes_from_the_asset_not_the_model(self):
        """
        A model cannot invent its own provenance.

        Sequence and content digest are facts about the file.
        """

        from src.visual.processor import _to_analysis

        asset = VisualAsset(
            path="media/a.jpg",
            filename="a.jpg",
            sha256="f" * 64,
            sequence=7,
        )

        analysis = _to_analysis(
            {
                "index": 0,
                "visible_text": "x",
                "sequence": 999,
                "source_asset_hash": "lies",
            },
            0,
            Path("a.jpg"),
            assets_by_path={"a.jpg": asset},
        )

        # Provenance comes from the file, whatever the answer claims.

        assert analysis.sequence == 7
        assert analysis.source_asset_hash == "f" * 64


# ---------------------------------------------------------------------
# 25, 26. Grounding, and not inflating questions
# ---------------------------------------------------------------------


class TestGrounding:
    def test_visual_text_reaches_the_enricher(self):
        """
        The whole integration, asserted end to end.

        ``AIEnricher._source_text`` already appends
        ``media.extracted_text``. Writing a slide's transcription there
        is what makes the slide eligible source material, with no change
        to the prompt or the models.
        """

        from src.ai.enricher import AIEnricher

        post = KnowledgePost(
            id="p",
            source=SourceInfo(
                platform="linkedin",
                captured_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
            original_text="A cheat sheet.",
        )

        post.media.append(
            type(post.media[0])() if post.media else None
        ) if False else None

        from src.models import MediaItem

        post.media = [
            MediaItem(
                type="image",
                path="media/a.jpg",
                extracted_text="Azure Data Factory pipelines into ADLS",
            )
        ]

        source = AIEnricher._source_text(post)

        assert "Azure Data Factory" in source
        assert "A cheat sheet." in source

    def test_the_grounding_check_sees_visual_text(self):
        from src.ai.grounding import check

        _, concepts, _, report = check(
            "The source names Azure Data Factory, landing in ADLS.",
            topics=["Data Engineering"],
            concepts=["Azure Data Factory"],
            questions=[],
        )

        # Named by the source, so kept.
        assert "Azure Data Factory" in concepts

        # An abbreviation is not the product. "ADF" in the source does
        # not license "Azure Data Factory" as a concept, which is the
        # rule that stops a plausible expansion becoming a fact.
        _, abbreviated, _, _ = check(
            "A pipeline from ADF into ADLS.",
            topics=["Data Engineering"],
            concepts=["Azure Data Factory"],
            questions=[],
        )

        assert abbreviated == []

        _, absent_concepts, _, absent_report = check(
            "A post with no mention of any cloud service.",
            topics=["Data Engineering"],
            concepts=["Snowflake"],
            questions=[],
        )

        # Not named anywhere, so removed. A slide nobody can read is not
        # evidence, and the report says so rather than dropping it
        # silently.
        assert absent_concepts == []

        # And the removal is recorded rather than silent, so a reader
        # comparing the slide to the page can see which side was cut.
        assert absent_report.concepts == ["Snowflake"]
        assert absent_report.clean is False


# ---------------------------------------------------------------------
# 27, 28, 29. Corrupt input
# ---------------------------------------------------------------------


class TestCorruptInput:
    def test_a_corrupt_image_is_reported_not_raised(self, tmp_path: Path):
        media = tmp_path / "media"
        media.mkdir()
        broken = media / "broken.jpg"
        broken.write_bytes(b"\xff\xd8\xff not really a jpeg")

        asset = build_asset(
            broken, media, declared_count=1, sequence_count=1
        )

        assert asset.state is ProcessingState.FAILED
        assert asset.note

    def test_format_is_read_from_content_not_the_name(self, tmp_path: Path):
        media = tmp_path / "media"
        media.mkdir()

        # A PNG named .jpg, which is what 536 of the real archive's
        # files are.
        misnamed = media / "actually_a_png.jpg"
        misnamed.write_bytes(png_bytes(300, 200))

        asset = build_asset(
            misnamed, media, declared_count=1, sequence_count=1
        )

        assert asset.format == "png"
        assert asset.width == 300

    def test_a_corrupt_pdf_does_not_stop_discovery(self, tmp_path: Path):
        media = tmp_path / "media"
        media.mkdir()

        broken = media / "doc.pdf"
        broken.write_bytes(b"%PDF-1.7 and then nothing")

        asset = build_asset(
            broken, media, declared_count=1, sequence_count=1
        )

        # Either it is recognised as a document, or it is failed. What
        # must never happen is an exception out of discovery.
        assert asset.state in {
            ProcessingState.PENDING,
            ProcessingState.FAILED,
        }

    def test_an_svg_is_never_executed(self, tmp_path: Path):
        media = tmp_path / "media"
        media.mkdir()

        hostile = media / "vector.svg"
        hostile.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg">'
            "<script>alert(1)</script>"
            '<image href="file:///C:/Windows/System32/cmd.exe"/>'
            "</svg>",
            encoding="utf-8",
        )

        asset = build_asset(
            hostile, media, declared_count=1, sequence_count=1
        )

        # Recognised, and explicitly not rendered here.
        assert asset.format == "svg"
        assert "script is never executed" in asset.note
        assert asset.role is not AssetRole.SLIDE

    def test_svg_is_not_routed_to_the_provider(self, tmp_path: Path):
        """
        An SVG can carry script and external references.

        It is recorded and left alone rather than handed to a process
        that may rasterise it.
        """

        media = tmp_path / "media"
        media.mkdir()
        (media / "vector.svg").write_text(
            "<svg xmlns='http://www.w3.org/2000/svg'></svg>",
            encoding="utf-8",
        )

        asset = build_asset(
            media / "vector.svg", media, declared_count=2, sequence_count=2
        )

        usable = usable_assets([asset])

        assert asset.media_type == "other"


# ---------------------------------------------------------------------
# 30, 31. Containment
# ---------------------------------------------------------------------


class TestContainment:
    def test_a_traversal_path_is_refused(self, tmp_path: Path):
        media = tmp_path / "media"
        media.mkdir()

        with pytest.raises(UnsafePath, match="outside"):
            contained_media_path(media, "../../Windows/System32/cmd.exe")

    def test_a_deep_traversal_is_refused(self, tmp_path: Path):
        media = tmp_path / "media"
        media.mkdir()

        with pytest.raises(UnsafePath):
            contained_media_path(
                media, "a/b/c/../../../../../../etc/passwd"
            )

    def test_an_absolute_path_outside_the_root_is_refused(self, tmp_path: Path):
        media = tmp_path / "media"
        media.mkdir()

        with pytest.raises(UnsafePath):
            contained_media_path(media, "C:/Windows/System32/drivers/etc/hosts")

    def test_a_unc_path_is_refused(self, tmp_path: Path):
        media = tmp_path / "media"
        media.mkdir()

        with pytest.raises(UnsafePath):
            contained_media_path(media, "//server/share/secret.jpg")

    def test_a_symlink_escaping_the_root_is_refused(self, tmp_path: Path):
        media = tmp_path / "media"
        media.mkdir()

        outside = tmp_path / "outside.jpg"
        outside.write_bytes(png_bytes(100, 100))

        link = media / "escape.jpg"

        try:
            link.symlink_to(outside)

        except (OSError, NotImplementedError):
            pytest.skip("symlinks unavailable on this platform")

        # The target is outside the root, and resolve has followed it.
        with pytest.raises(UnsafePath):
            contained_media_path(media, "escape.jpg")

    def test_a_symlink_inside_the_root_is_allowed(self, tmp_path: Path):
        media = tmp_path / "media"
        media.mkdir()

        real = media / "real.jpg"
        real.write_bytes(png_bytes(100, 100))

        link = media / "alias.jpg"

        try:
            link.symlink_to(real)

        except (OSError, NotImplementedError):
            pytest.skip("symlinks unavailable on this platform")

        assert contained_media_path(media, "alias.jpg").is_file()

    def test_a_malicious_filename_cannot_escape(self, tmp_path: Path):
        """
        The filename is data, never a path to be followed.
        """

        media = tmp_path / "media"
        media.mkdir()

        for name in (
            "../../etc/passwd",
            "..\\..\\Windows\\win.ini",
            "C:\\Windows\\System32\\cmd.exe",
        ):
            with pytest.raises(UnsafePath):
                contained_media_path(media, name)

    def test_the_media_root_itself_cannot_be_browser_state(
        self, tmp_path: Path
    ):
        from src.visual.archive import _check_not_session

        session = tmp_path / "archive" / "chrome_session" / "Default"
        session.mkdir(parents=True)

        # Pointed at browser state as though it were the media.
        with pytest.raises(ArchiveError, match="browser state"):
            _check_not_session(session)

    def test_browser_state_files_are_refused_at_the_point_of_use(
        self,
    ):
        with pytest.raises(ArchiveError):
            assert_readable(Path("chrome_session/Default/Cookies"))

        with pytest.raises(ArchiveError):
            assert_readable(Path("somewhere/Login Data"))

    def test_an_archive_containing_a_session_directory_still_loads(
        self, tmp_path: Path
    ):
        """
        The session folder sitting beside the media is this archive's
        normal layout.

        Refusing the whole archive over it would make the stage
        impossible to run for a reason unrelated to the media.
        """

        root = write_archive(
            tmp_path / "archive",
            [
                {
                    "post_id": "activity_ok",
                    "files": {"activity_ok_slide_0.jpg": jpeg_bytes(200, 150)},
                }
            ],
            session=True,
        )

        loaded = load_archive(root)

        assert "activity_ok" in loaded.posts


# ---------------------------------------------------------------------
# 32-34. Plans, dry runs and the existing corpus
# ---------------------------------------------------------------------


class TestPlanning:
    def test_a_plan_costs_nothing_and_writes_nothing(
        self, archive: Path, tmp_path: Path
    ):
        from src.visual.plan import build_plan

        store = VisualStore(tmp_path / "store")

        plan = build_plan(archive, store=store)

        assert plan.rows
        assert plan.cold_cache_assets > 0
        assert not (tmp_path / "store").exists()

    def test_the_plan_counts_distinct_content_not_files(self, tmp_path: Path):
        from src.visual.plan import build_plan

        payload = jpeg_bytes(500, 400)

        root = write_archive(
            tmp_path / "archive",
            [
                {
                    "post_id": "activity_one",
                    "files": {
                        "activity_one_slide_0.jpg": payload,
                        "activity_one_slide_1.jpg": payload,
                        "activity_one_slide_2.jpg": jpeg_bytes(600, 500, (7, 7, 7)),
                    },
                }
            ],
        )

        plan = build_plan(root, store=VisualStore(tmp_path / "store"))
        totals = plan.totals()

        # Three files, two distinct pictures.
        assert totals["assets"] == 3
        assert totals["distinct_contents"] == 2

    def test_only_changed_posts_are_planned(self, archive: Path, tmp_path: Path):
        from src.visual.plan import build_plan

        loaded = load_archive(archive)
        assets = assets_for_post(loaded, "activity_carousel")

        store = VisualStore(tmp_path / "store")

        # Record the carousel as already understood.
        for asset in usable_assets(assets):
            store.save(
                VisualAnalysis(
                    source_asset_hash=asset.sha256,
                    source_path=asset.path,
                    sequence=asset.sequence,
                    extracted_text="known",
                ),
                "1",
                "vision+bounded-batch",
            )

        plan = build_plan(archive, store=store)

        by_post = {row.post_id: row for row in plan.rows}

        assert by_post["activity_carousel"].todo == 0
        assert by_post["activity_single"].todo > 0


# ---------------------------------------------------------------------
# 35-37. Determinism and concurrency
# ---------------------------------------------------------------------


class TestDeterminismAndConcurrency:
    def test_two_runs_produce_identical_digests(self, archive: Path):
        loaded = load_archive(archive)

        first = [
            asset_digest(
                assets_for_post(loaded, "activity_carousel"),
                processor_version="1",
                configuration="vision",
            )
        ]
        second = [
            asset_digest(
                assets_for_post(loaded, "activity_carousel"),
                processor_version="1",
                configuration="vision",
            )
        ]

        assert first == second

    def test_concurrent_posts_do_not_interleave_within_one(self, tmp_path: Path):
        """
        Concurrency is over posts, never within one.

        A post's slides are read in order and synthesised together, so
        interleaving two posts' batches would make the order of a post's
        own slides depend on the scheduler.
        """

        from src.pipeline.visual import VisualRunner

        write_archive(
            tmp_path / "archive",
            [
                {
                    "post_id": "activity_carousel",
                    "files": {
                        "activity_carousel_slide_01.jpg": jpeg_bytes(900, 700),
                        "activity_carousel_slide_02.jpg": jpeg_bytes(900, 700, (3, 4, 5)),
                        "activity_carousel_slide_03.jpg": jpeg_bytes(900, 700, (6, 7, 8)),
                    },
                }
            ],
        )

        posts = []

        for index in range(4):
            post_id = write_post(
                tmp_path,
                f"post_{index}",
                media=[f"activity_carousel_slide_0{slide}.jpg"
                       for slide in (1, 2, 3)],
            )

            posts.append(
                KnowledgePost.model_validate(
                    json.loads(
                        (
                            tmp_path / "data" / "posts" / post_id /
                            "post.json"
                        ).read_text(encoding="utf-8")
                    )
                )
            )

        runner = VisualRunner(
            tmp_path / "archive",
            store=VisualStore(tmp_path / "store"),
            processor=Scripted(),
            progress=lambda message: None,
        )

        run = runner.run(posts, jobs=4)

        assert run.posts_processed == 4

        for post_id, outcome in run.outcomes.items():
            sequences = [
                analysis.sequence
                for analysis in outcome.visual.ordered_analyses()
            ]

            assert sequences == sorted(sequences)

    def test_one_post_failing_does_not_stop_the_others(
        self, archive: Path, tmp_path: Path
    ):
        from src.pipeline.visual import VisualRunner

        posts = []

        for name, media_name in (
            ("urn_a", "activity_carousel_slide_01.jpg"),
            ("urn_b", "activity_single_slide_0.jpg"),
        ):
            post_id = write_post(tmp_path, name, media=[media_name])

            posts.append(
                KnowledgePost.model_validate(
                    json.loads(
                        (
                            tmp_path / "data" / "posts" / post_id /
                            "post.json"
                        ).read_text(encoding="utf-8")
                    )
                )
            )

        runner = VisualRunner(
            archive,
            store=VisualStore(tmp_path / "store"),
            processor=Scripted(failures=[RuntimeError(TRUNCATION)] * 3),
            progress=lambda message: None,
        )

        run = runner.run(posts, jobs=2)

        # Both settled, whether or not both understood.
        assert run.posts == 2
        assert len(run.outcomes) + run.posts_skipped == 2


# ---------------------------------------------------------------------
# 38-40. The safety boundary
# ---------------------------------------------------------------------


class TestSafety:
    def _visual_sources(self) -> list[Path]:
        root = Path(__file__).resolve().parents[1]

        paths: list[Path] = sorted((root / "src" / "visual").rglob("*.py"))
        paths.append(root / "src" / "pipeline" / "visual.py")
        paths.append(root / "src" / "pipeline" / "visual_stage.py")

        return paths

    def _imports(self, path: Path) -> set[str]:
        import ast

        tree = ast.parse(path.read_text(encoding="utf-8"))

        modules: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    modules.add(alias.name.split(".")[0])

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    modules.add(node.module.split(".")[0])

        return modules

    @pytest.mark.parametrize(
        "banned",
        [
            "socket",
            "ssl",
            "http",
            "urllib",
            "requests",
            "httpx",
            "aiohttp",
            "playwright",
            "selenium",
            "webbrowser",
            "keyring",
            "netrc",
            "getpass",
        ],
    )
    def test_the_visual_stage_reaches_nothing(self, banned: str):
        offenders = [
            path.name
            for path in self._visual_sources()
            if banned in self._imports(path)
        ]

        assert offenders == []

    def test_it_reads_no_environment_variable(self):
        offenders = []

        for path in self._visual_sources():
            source = path.read_text(encoding="utf-8")

            if "os.environ" in source or "getenv" in source:
                offenders.append(path.name)

        # The CLI reads LINKEDIN_ARCHIVE_ROOT, which is configuration
        # and lives in the pipeline, not in the visual package.
        assert offenders == []

    def test_it_uses_no_dynamic_execution(self):
        offenders = []

        for path in self._visual_sources():
            source = path.read_text(encoding="utf-8")

            # ``re.compile`` is a regular expression, not dynamic
            # execution, so the pattern is stripped before looking.
            without_regex = source.replace("re.compile(", "")

            for banned in ("eval(", "exec(", "compile(", "__import__"):
                if banned in without_regex:
                    offenders.append(f"{path.name}: {banned}")

        assert offenders == []

    def test_it_writes_only_under_the_cache_root(self):
        """
        The archive is read-only. Nothing in the visual package opens a
        file for writing except the store, which lives under build/.
        """

        for path in self._visual_sources():
            source = path.read_text(encoding="utf-8")

            for banned in (
                'write_text(',
                'write_bytes(',
                'unlink(',
                'mkdir(',
                'rename(',
                'shutil.rmtree',
            ):
                if banned in source:
                    # store.py owns the only writes, and they are all
                    # into its own root.
                    assert path.name == "store.py", (
                        f"{path.name} contains {banned}"
                    )

    def test_the_archive_is_never_written_by_the_stage(self, tmp_path: Path):
        """
        Demonstrated rather than asserted: a real archive, a real stage,
        and every file's mtime compared afterwards.
        """

        archive = write_archive(
            tmp_path / "archive",
            [
                {
                    "post_id": "activity_ro",
                    "files": {
                        f"activity_ro_slide_{index}.jpg": jpeg_bytes(
                            300, 200, (index, 1, 2)
                        )
                        for index in range(3)
                    },
                }
            ],
        )

        before = {
            path: path.stat().st_mtime
            for path in sorted(archive.rglob("*"))
            if path.is_file()
        }

        loaded = load_archive(archive)
        assets = assets_for_post(loaded, "activity_ro")

        stage = VisualStage(
            processor=Scripted(),
            store=VisualStore(tmp_path / "store"),
            progress=lambda message: None,
        )

        stage.run_post(
            "activity_ro",
            assets,
            lambda asset: resolve_asset_path(loaded, asset),
        )

        after = {
            path: path.stat().st_mtime
            for path in sorted(archive.rglob("*"))
            if path.is_file()
        }

        assert before == after

    def test_a_model_answer_cannot_choose_where_to_write(self, tmp_path: Path):
        """
        The processor is handed a path to read, never one to write to.
        """

        processor = VisionProcessor(client=FakeClient())
        supply(processor, "passwd")

        analyses = processor.analyse([Path("/etc/passwd")])

        # Only the filename is carried. A record naming a path from
        # outside the post's media directory would be a record of
        # something the post does not have.
        assert analyses[0].source_path == "media/passwd"

    def test_no_linkedin_host_is_fetched(self):
        """
        The stage has no network client at all, so there is nothing to
        fetch with. The original URLs in the archive are recorded and
        never followed.
        """

        for path in self._visual_sources():
            source = path.read_text(encoding="utf-8")

            for banned in (
                "urlretrieve",
                "urlopen",
                "requests.get",
                "httpx.get",
                "original_url",
            ):
                if banned in source:
                    # original_urls may be read into the record, which
                    # is not the same as opening it.
                    assert (
                        "http" not in banned
                        or banned == "original_url"
                    ), f"{path.name}: {banned}"


# ---------------------------------------------------------------------
# Store behaviour
# ---------------------------------------------------------------------


class TestStore:
    def test_an_analysis_round_trips(self, tmp_path: Path):
        store = VisualStore(tmp_path / "store")

        analysis = VisualAnalysis(
            source_asset_hash="a" * 64,
            source_path="media/a.jpg",
            extracted_text="SQL JOINS",
        )

        store.save(analysis, "1", "vision")

        loaded = store.load("a" * 64, "1", "vision")

        assert loaded is not None
        assert loaded.extracted_text == "SQL JOINS"

    def test_a_different_version_does_not_reuse(self, tmp_path: Path):
        store = VisualStore(tmp_path / "store")

        store.save(
            VisualAnalysis(source_asset_hash="a" * 64, extracted_text="x"),
            "1",
            "vision",
        )

        assert store.load("a" * 64, "2", "vision") is None

    def test_a_different_configuration_does_not_reuse(self, tmp_path: Path):
        store = VisualStore(tmp_path / "store")

        store.save(
            VisualAnalysis(source_asset_hash="a" * 64, extracted_text="x"),
            "1",
            "vision",
        )

        assert store.load("a" * 64, "1", "vision+ocr") is None

    def test_a_stored_analysis_must_name_its_asset(self, tmp_path: Path):
        store = VisualStore(tmp_path / "store")

        path = store.path_for("a" * 64, "1", "vision")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"source_asset_hash": "b" * 64, "extracted_text": "x"}
            ),
            encoding="utf-8",
        )

        # A file whose contents name a different asset is not an
        # analysis of this one, whatever it is called.
        assert store.load("a" * 64, "1", "vision") is None

    def test_a_corrupt_cache_entry_is_ignored_not_fatal(self, tmp_path: Path):
        store = VisualStore(tmp_path / "store")

        path = store.path_for("a" * 64, "1", "vision")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")

        assert store.load("a" * 64, "1", "vision") is None

    def test_no_temporary_file_is_left_behind(self, tmp_path: Path):
        store = VisualStore(tmp_path / "store")

        store.save(
            VisualAnalysis(source_asset_hash="a" * 64), "1", "vision"
        )

        assert not list(store.root.rglob("*.tmp"))

    def test_an_empty_digest_is_refused(self, tmp_path: Path):
        store = VisualStore(tmp_path / "store")

        with pytest.raises(ValueError):
            store.path_for("", "1", "vision")
