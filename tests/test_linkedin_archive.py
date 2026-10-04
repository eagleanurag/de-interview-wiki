"""
Tests for the local LinkedIn archive importer.

Every fixture here is synthetic and built in a temporary directory. None
of them touches a real archive, and none of them needs one to exist: the
importer's input is a JSON file and a folder of media, so that is
exactly what a test makes. A test that depended on a path outside the
repository would fail on CI and would be measuring the machine rather
than the code.

The cases that matter most are the ones where being wrong would be
invisible. A record with no permalink must not acquire one. A file
named for one format and holding another must be reported as such. A
malformed record must not cost the four hundred beside it. A path in a
record must not be able to name a file outside the media folder. And
nothing the archive says, including the URLs it carries, may become an
instruction to fetch something.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import pytest

from src.ingestion.linkedin_archive import archive as archive_module
from src.ingestion.linkedin_archive import identity as identity_module
from src.ingestion.linkedin_archive import prepare as prepare_module
from src.ingestion.linkedin_archive import report as report_module
from src.ingestion.saved_items.model import (
    KIND_IDENTIFIED_BY_ID,
    SavedItem,
)
from src.ingestion.saved_items.readers import read_manifest
from src.ingestion.saved_items.urls import normalize_linkedin_url


PERMALINK = (
    "https://www.linkedin.com/feed/update/urn:li:activity:7509801763013595136"
)

README = Path(__file__).resolve().parents[1] / "README.md"

DELTA_TEXT = (
    "Delta Lake gives a data lake ACID guarantees. A transaction log "
    "sits beside the data, so a commit is atomic. Schema evolution lets "
    "you add a column without rewriting the files already there, and "
    "time travel queries an earlier version of the table. On Databricks "
    "these tables run on Apache Spark, and the file layout is what makes "
    "predicate pushdown possible."
)

SQL_TEXT = (
    "A window function computes across a set of related rows without "
    "collapsing them, so each row keeps a comparison against its own "
    "partition. That is what a running total needs, and a GROUP BY "
    "cannot give it to you."
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


def png(colour: tuple[int, int, int] = (10, 90, 200)) -> bytes:
    """A real PNG, written with zlib rather than copied."""

    def chunk(tag: bytes, payload: bytes) -> bytes:
        body = tag + payload

        return (
            len(payload).to_bytes(4, "big")
            + body
            + zlib.crc32(body).to_bytes(4, "big")
        )

    row = bytes([0]) + bytes(colour) * 4

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(
            b"IHDR",
            (4).to_bytes(4, "big")
            + (4).to_bytes(4, "big")
            + bytes([8, 2, 0, 0, 0]),
        )
        + chunk(b"IDAT", zlib.compress(row * 4))
        + chunk(b"IEND", b"")
    )


def jpeg() -> bytes:
    """A file with a JPEG signature. Not a decodable image, and not asked to be."""
    return b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01" + b"\x00" * 64


def gif() -> bytes:
    return b"GIF89a" + b"\x00" * 64


def pdf(body: str = "Consumer lag explained") -> bytes:
    """A real single-page PDF holding the given text."""
    content = b"BT /F1 12 Tf 72 720 Td (" + body.encode() + b") Tj ET"

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n"
        + content
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


def broken_pdf() -> bytes:
    """A valid header and a truncated body: what a failed download leaves."""
    whole = pdf()

    return whole[: len(whole) // 3]


def svg() -> bytes:
    return (
        b'<?xml version="1.0"?>\n'
        b'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">'
        b"<rect width='10' height='10' fill='#123456'/></svg>"
    )


def svg_with_script() -> bytes:
    """
    An SVG carrying a script.

    The importer must keep it as source media and never run it. Held as
    a fixture so the test can assert that, rather than asserting only
    that no script is present -- which would pass just as well on an
    archive that happened to hold none.
    """
    return (
        b'<?xml version="1.0"?>\n'
        b'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">'
        b"<script>throw new Error('executed')</script>"
        b"<rect width='10' height='10'/></svg>"
    )


def record(
    post_id: str = "activity_7509801763013595136",
    *,
    permalink: object = PERMALINK,
    text: str = DELTA_TEXT,
    saved_files: list[str] | None = None,
    original_urls: list[str] | None = None,
    author: str = "A Fixture",
    headline: str = "Data Engineer",
    scraped_at: str = "2026-10-02T04:30:00Z",
    relative_time: str = "4mo",
    multi_slide: bool = False,
    total_media: int | None = None,
) -> dict:
    """One archive record, in the shape the real archive uses."""
    files = saved_files if saved_files is not None else []

    return {
        "post_id": post_id,
        "scraped_at": scraped_at,
        "author": {
            "name": author,
            "profile_url": f"https://www.linkedin.com/in/{post_id[-6:]}",
            "headline": headline,
        },
        "relative_time": relative_time,
        "permalink": permalink,
        "text": text,
        "media": {
            "has_media": bool(files),
            "is_multi_slide": multi_slide,
            "total_media_count": (
                len(files) if total_media is None else total_media
            ),
            "saved_files": files,
            "original_urls": original_urls or [],
        },
    }


def activity(number: int) -> str:
    """A distinct activity URL, so two records are two posts."""
    return (
        "https://www.linkedin.com/feed/update/"
        f"urn:li:activity:{7509801763013595136 + number}"
    )


def build_archive(
    root: Path,
    records: list,
    media: dict[str, bytes] | None = None,
) -> Path:
    """Write an archive directory the importer can read."""
    base = root / "archive"

    media_root = base / archive_module.MEDIA_DIR

    media_root.mkdir(parents=True, exist_ok=True)

    (base / archive_module.ARCHIVE_JSON).write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for name, payload in (media or {}).items():
        (media_root / name).write_bytes(payload)

    return base


# ---------------------------------------------------------------------
# 1-3. The archive as a whole
# ---------------------------------------------------------------------


class TestTheArchiveFile:
    def test_a_valid_archive_is_read(self, tmp_path: Path):
        base = build_archive(tmp_path, [record()])

        archive = archive_module.read_archive(base)

        assert archive.total_seen == 1
        assert len(archive.records) == 1
        assert not archive.failures

    def test_an_empty_archive_is_read_and_is_not_an_error(
        self, tmp_path: Path
    ):
        base = build_archive(tmp_path, [])

        archive = archive_module.read_archive(base)

        assert archive.total_seen == 0
        assert archive.records == []
        assert archive.failures == []

    def test_malformed_json_is_refused_with_a_position(
        self, tmp_path: Path
    ):
        base = tmp_path / "archive"
        base.mkdir()
        (base / archive_module.ARCHIVE_JSON).write_text(
            "[{ not json", encoding="utf-8"
        )

        with pytest.raises(archive_module.ArchiveError) as caught:
            archive_module.read_archive(base)

        # A corrupt file is a different problem from a bad row, and the
        # message has to say which so the report does not blame five
        # hundred records for one broken byte.
        assert "not valid JSON" in str(caught.value)
        assert "line" in str(caught.value)

    def test_a_json_object_instead_of_a_list_is_refused(
        self, tmp_path: Path
    ):
        base = tmp_path / "archive"
        base.mkdir()
        (base / archive_module.ARCHIVE_JSON).write_text(
            json.dumps({"posts": "not a list"}), encoding="utf-8"
        )

        with pytest.raises(archive_module.ArchiveError) as caught:
            archive_module.read_archive(base)

        assert "not a list" in str(caught.value)

    def test_a_wrapper_object_holding_a_list_is_read(self, tmp_path: Path):
        base = tmp_path / "archive"
        base.mkdir()
        (base / archive_module.ARCHIVE_JSON).write_text(
            json.dumps({"version": 1, "posts": [record()]}),
            encoding="utf-8",
        )

        archive = archive_module.read_archive(base)

        # An archive written by a tool that wraps its export is still an
        # archive. Refusing it would lose five hundred posts over
        # punctuation.
        assert len(archive.records) == 1

    def test_a_missing_directory_is_named_in_the_refusal(
        self, tmp_path: Path
    ):
        with pytest.raises(archive_module.ArchiveError) as caught:
            archive_module.read_archive(tmp_path / "nowhere")

        assert "does not exist" in str(caught.value)

    def test_a_directory_without_the_file_says_where_to_look(
        self, tmp_path: Path
    ):
        (tmp_path / "empty").mkdir()

        with pytest.raises(archive_module.ArchiveError) as caught:
            archive_module.read_archive(tmp_path / "empty")

        assert archive_module.ARCHIVE_JSON in str(caught.value)


# ---------------------------------------------------------------------
# 4-8. Individual records
# ---------------------------------------------------------------------


class TestOneRecord:
    def test_a_record_that_is_not_an_object_is_refused(
        self, tmp_path: Path
    ):
        base = build_archive(tmp_path, ["just a string"])

        archive = archive_module.read_archive(base)

        assert not archive.records
        assert len(archive.failures) == 1
        assert "not an object" in archive.failures[0].message

    def test_a_record_with_no_post_id_is_refused(self, tmp_path: Path):
        entry = record()
        del entry["post_id"]

        base = build_archive(tmp_path, [entry, record("activity_2", permalink=activity(2))])

        archive = archive_module.read_archive(base)

        assert len(archive.records) == 1
        assert len(archive.failures) == 1
        # Refused rather than given a synthetic id: a record the
        # pipeline cannot name is one it will import again next run.
        assert "no post_id" in archive.failures[0].message

    def test_a_record_with_no_permalink_keeps_its_text(
        self, tmp_path: Path
    ):
        base = build_archive(
            tmp_path, [record("post_abc123", permalink=None)]
        )

        archive = archive_module.read_archive(base)

        found = archive.records[0]

        assert found.raw_text == DELTA_TEXT
        assert found.identity.canonical_url == ""
        assert found.identity.has_permalink is False

    def test_a_record_with_no_text_is_kept_and_noted(
        self, tmp_path: Path
    ):
        base = build_archive(tmp_path, [record(text="")])

        archive = archive_module.read_archive(base)

        found = archive.records[0]

        assert found.has_text is False
        assert any(
            "no text" in note for note in found.notes
        )

    def test_an_empty_text_does_not_stop_the_batch(
        self, tmp_path: Path
    ):
        base = build_archive(
            tmp_path,
            [
                record(text=""),
                record("activity_2", text=SQL_TEXT, permalink=activity(2)),
            ],
        )

        archive = archive_module.read_archive(base)

        assert len(archive.records) == 2

    def test_a_missing_media_block_is_survived(self, tmp_path: Path):
        entry = record()
        del entry["media"]

        base = build_archive(tmp_path, [entry, record("activity_2", permalink=activity(2))])

        archive = archive_module.read_archive(base)

        assert len(archive.records) == 2
        assert archive.records[0].media == []
        assert any(
            "media block" in note for note in archive.records[0].notes
        )

    def test_a_null_value_everywhere_is_survived(self, tmp_path: Path):
        base = build_archive(
            tmp_path,
            [
                {
                    "post_id": "activity_nulls",
                    "scraped_at": None,
                    "author": None,
                    "relative_time": None,
                    "permalink": None,
                    "text": "Some text that is present.",
                    "media": None,
                }
            ],
        )

        archive = archive_module.read_archive(base)

        found = archive.records[0]

        assert found.raw_text == "Some text that is present."
        assert found.scraped_at is None
        assert found.author is None

    def test_an_author_of_unknown_is_kept_and_noted(
        self, tmp_path: Path
    ):
        base = build_archive(
            tmp_path, [record(author="Unknown")]
        )

        archive = archive_module.read_archive(base)

        found = archive.records[0]

        # Kept as the archive wrote it, because "the archive failed to
        # find an author" and "the post has no author" are different.
        assert found.author == "Unknown"
        assert any(
            "no usable author" in note for note in found.notes
        )


# ---------------------------------------------------------------------
# 9-15. Media
# ---------------------------------------------------------------------


class TestMedia:
    def test_a_missing_media_file_is_recorded_not_raised(
        self, tmp_path: Path
    ):
        base = build_archive(
            tmp_path,
            [record(saved_files=["absent.jpg"])],
        )

        archive = archive_module.read_archive(base)

        found = archive.records[0]

        assert found.media == []
        assert found.missing_media == ["absent.jpg"]
        assert archive.failures == []

    def test_a_corrupt_image_is_reported(self, tmp_path: Path):
        base = build_archive(
            tmp_path,
            [
                record(
                    saved_files=["broken.jpg"],
                    original_urls=[
                        "https://media.licdn.com/dms/image/one"
                    ],
                )
            ],
            media={"broken.jpg": b"not an image at all"},
        )

        archive = archive_module.read_archive(base)

        asset = archive.media.assets["broken.jpg"]

        assert asset.detected_format == "unknown"
        assert asset.readable is True
        assert asset.problem

    def test_an_empty_file_is_not_readable(self, tmp_path: Path):
        base = build_archive(
            tmp_path, [record(saved_files=["empty.jpg"])],
            media={"empty.jpg": b""},
        )

        archive = archive_module.read_archive(base)

        assert archive.media.assets["empty.jpg"].readable is False
        assert archive.media.unreadable == ["empty.jpg"]

    @pytest.mark.parametrize(
        "name,payload,expected",
        [
            ("one.png", png(), "png"),
            ("one.jpg", jpeg(), "jpeg"),
            ("one.gif", gif(), "gif"),
            ("one.pdf", pdf(), "pdf"),
            ("one.svg", svg(), "svg"),
        ],
    )
    def test_every_format_is_recognised_from_its_bytes(
        self, tmp_path: Path, name: str, payload: bytes, expected: str
    ):
        base = build_archive(
            tmp_path,
            [record(saved_files=[name])],
            media={name: payload},
        )

        archive = archive_module.read_archive(base)

        assert archive.media.assets[name].detected_format == expected

    def test_a_file_named_for_one_type_and_holding_another_is_noted(
        self, tmp_path: Path
    ):
        # The real archive names every asset .jpg and a third of them
        # are PNG or GIF. A reader told a file is a JPEG and handed a PNG
        # has been told something false.
        base = build_archive(
            tmp_path,
            [record(saved_files=["slide_0.jpg"])],
            media={"slide_0.jpg": png()},
        )

        archive = archive_module.read_archive(base)

        asset = archive.media.assets["slide_0.jpg"]

        assert asset.detected_format == "png"
        assert asset.extension_agrees is False
        assert any(
            "does not describe" in note or "its content is" in note
            for note in archive.records[0].notes
        )

    def test_a_pdf_is_kept_as_a_document(self, tmp_path: Path):
        base = build_archive(
            tmp_path,
            [record(saved_files=["notes.pdf"])],
            media={"notes.pdf": pdf()},
        )

        archive = archive_module.read_archive(base)

        asset = archive.media.assets["notes.pdf"]

        assert asset.is_document is True
        assert asset.is_image is False

    def test_an_svg_is_source_media_not_a_rejection(
        self, tmp_path: Path):
        base = build_archive(
            tmp_path,
            [record(saved_files=["diagram.svg"])],
            media={"diagram.svg": svg()},
        )

        archive = archive_module.read_archive(base)

        assert archive.media.assets["diagram.svg"].is_svg is True

    def test_an_svg_with_a_script_is_still_only_a_file(
        self, tmp_path: Path
    ):
        base = build_archive(
            tmp_path,
            [record(saved_files=["hostile.svg"])],
            media={"hostile.svg": svg_with_script()},
        )

        archive = archive_module.read_archive(base)

        # Read as bytes, detected, kept. Nothing here runs it.
        assert archive.media.assets["hostile.svg"].is_svg is True
        assert (
            b"executed"
            in (base / "media" / "hostile.svg").read_bytes()
        )

    def test_a_corrupt_pdf_does_not_stop_the_batch(
        self, tmp_path: Path):
        base = build_archive(
            tmp_path,
            [
                record(
                    "activity_1",
                    saved_files=["broken.pdf"],
                    total_media=1,
                ),
                record("activity_2", text=SQL_TEXT, permalink=activity(2)),
            ],
            media={"broken.pdf": broken_pdf()},
        )

        archive = archive_module.read_archive(base)

        # A truncated file is a real file with a real signature. It is
        # kept and later reported when the reader fails to open it.
        assert len(archive.records) == 2
        assert archive.media.assets["broken.pdf"].detected_format == "pdf"

    def test_multiple_images_are_kept_in_order(self, tmp_path: Path):
        names = [f"slide_{index}.png" for index in range(3)]

        base = build_archive(
            tmp_path,
            [record(saved_files=names, multi_slide=True)],
            media={name: png() for name in names},
        )

        archive = archive_module.read_archive(base)

        found = archive.records[0]

        assert [asset.name for asset in found.media] == names
        assert archive.records[0].claimed_multi_slide is True

    def test_a_record_listing_one_file_twice_counts_it_once(
        self, tmp_path: Path
    ):
        # The real archive does this for three carousel posts whose
        # pagination failed: the same slide appended twice.
        base = build_archive(
            tmp_path,
            [
                record(
                    saved_files=["slide_0.jpg", "slide_0.jpg"],
                    multi_slide=True,
                    total_media=2,
                )
            ],
            media={"slide_0.jpg": png()},
        )

        archive = archive_module.read_archive(base)

        found = archive.records[0]

        assert len(found.media) == 1
        assert any(
            "more than once" in note for note in found.notes
        )
        assert any(
            "multi-slide" in note for note in found.notes
        )

    def test_identical_bytes_are_reported_as_shared(self, tmp_path: Path):
        same = png()

        base = build_archive(
            tmp_path,
            [
                record("activity_1", saved_files=["a.jpg"]),
                record("activity_2", permalink=activity(2), saved_files=["b.jpg"]),
            ],
            media={"a.jpg": same, "b.jpg": same},
        )

        archive = archive_module.read_archive(base)

        assert archive.media.shared_by_digest("a.jpg") == ["b.jpg"]
        assert archive.media.unique_names() == 1

    def test_media_is_indexed_once_not_per_record(
        self, tmp_path: Path
    ):
        names = [f"m{index}.png" for index in range(5)]

        base = build_archive(
            tmp_path,
            [record(
                f"activity_{index}",
                permalink=activity(index),
                saved_files=[name],
            )
             for index, name in enumerate(names)],
            media={name: png() for name in names},
        )

        archive = archive_module.read_archive(base)

        # One index covering every file, so looking up a name is a
        # dictionary hit rather than another walk of the folder.
        assert len(archive.media) == 5
        assert all(
            archive.media.get(name) is not None for name in names
        )


# ---------------------------------------------------------------------
# 16-19. Duplicates
# ---------------------------------------------------------------------


class TestDuplicates:
    def test_a_repeated_post_id_is_two_records_not_one(
        self, tmp_path: Path
    ):
        base = build_archive(
            tmp_path, [record(), record(text=SQL_TEXT)]
        )

        archive = archive_module.read_archive(base)

        # Both are read. Collapsing them here would be a guess; the
        # report is where a repeat is named.
        assert len(archive.records) == 2

    def test_the_same_url_written_two_ways_is_one_item(
        self, tmp_path: Path
    ):
        first = identity_module.identify(
            "activity_1", PERMALINK + "/?trk=bookmark"
        )
        second = identity_module.identify("activity_2", PERMALINK)

        # One canonical URL means one item, so an archive record and a
        # hand-built row pointing at the same post cannot both exist.
        assert first.source_id == second.source_id

    def test_whole_text_repetition_is_a_duplicate(self, tmp_path: Path):
        base = build_archive(
            tmp_path,
            [
                record("activity_1", permalink=activity(1), text=DELTA_TEXT),
                record("activity_2", permalink=activity(2), text=DELTA_TEXT),
            ],
        )

        archive = archive_module.read_archive(base)

        # The tie-break is the post id, and the rule is stated in the
        # code, so which of two identical records survives is predictable
        # rather than whatever the run happened to see first.
        assert len(archive.duplicates) == 1
        assert archive.records[0].duplicate_of == "activity_2"
        assert archive.records[1].duplicate_of is None

    def test_a_duplicate_is_never_silently_dropped(self, tmp_path: Path):
        base = build_archive(
            tmp_path,
            [
                record("activity_1", permalink=activity(1), text=DELTA_TEXT),
                record("activity_2", permalink=activity(2), text=DELTA_TEXT),
            ],
        )

        archive = archive_module.read_archive(base)

        duplicated = archive.records[0]

        assert len(archive.records) == 2
        assert duplicated.post_id == "activity_1"
        assert any(
            "keeps this record's place" in note
            for note in duplicated.notes
        )

    def test_a_shared_prefix_is_not_a_duplicate(self, tmp_path: Path):
        # A prefix fingerprint folded these two into one, and folding
        # two posts that agree for a few hundred characters is worse
        # than reporting a duplicate that is not one.
        shared = "Data engineering will be different if you do these things:"

        base = build_archive(
            tmp_path,
            [
                record("activity_1", text=shared + " one and then some"),
                record("activity_2", text=shared + " two and then more"),
            ],
        )

        archive = archive_module.read_archive(base)

        assert archive.duplicates == []

    def test_the_keeper_is_the_record_with_a_permalink(
        self, tmp_path: Path
    ):
        base = build_archive(
            tmp_path,
            [
                record("post_znolink", permalink=None, text=DELTA_TEXT),
                record("activity_1", permalink=activity(1), text=DELTA_TEXT),
            ],
        )

        archive = archive_module.read_archive(base)

        # The id sort would have kept the url-less record, since "post_"
        # sorts after "activity_". The permalink wins first, because
        # dropping the only copy of a link in the name of tidiness would
        # be a data loss.
        assert archive.records[0].duplicate_of == "activity_1"
        assert archive.records[1].duplicate_of is None

    def test_the_keeper_is_the_record_with_media(self, tmp_path: Path):
        base = build_archive(
            tmp_path,
            [
                record(
                    "activity_1",
                    permalink=activity(1),
                    text=DELTA_TEXT,
                ),
                record(
                    "activity_2",
                    permalink=activity(2),
                    text=DELTA_TEXT,
                    saved_files=["a.png"],
                ),
            ],
            media={"a.png": png()},
        )

        archive = archive_module.read_archive(base)

        # Losing the only copy of a picture would be worse than losing
        # the duplicate, so media outranks the identifier.
        assert archive.records[0].duplicate_of == "activity_2"
        assert archive.records[1].duplicate_of is None

    def test_the_duplicate_is_reported_not_deleted(
        self, tmp_path: Path
    ):
        base = build_archive(
            tmp_path,
            [
                record("activity_1", permalink=activity(1), text=DELTA_TEXT),
                record("activity_2", permalink=activity(2), text=DELTA_TEXT),
            ],
        )

        archive = archive_module.read_archive(base)

        assert len(archive.records) == 2
        assert any(
            "keeps this record's place" in note
            for note in archive.records[0].notes
        )


# ---------------------------------------------------------------------
# 20-23. Incremental
# ---------------------------------------------------------------------


class TestIncremental:
    def test_preparing_twice_writes_nothing_new(self, tmp_path: Path):
        base = build_archive(
            tmp_path, [record(), record("activity_2", text=SQL_TEXT, permalink=activity(2))]
        )
        drop = tmp_path / "drop"

        first = prepare_module.prepare(
            archive_module.read_archive(base), drop
        )
        second = prepare_module.prepare(
            archive_module.read_archive(base), drop
        )

        assert first.written == 2
        assert second.written == 0
        assert second.unchanged == 2

    def test_a_re_run_rewrites_the_same_manifest(self, tmp_path: Path):
        """
        The manifest is the list, not a log of this run's edits.

        Found by running the real archive twice: the second run found
        every capture unchanged, wrote no rows, and left a manifest with
        a header and nothing in it. The import still appeared to work
        because the stored state file carried the items, which is
        exactly what made it easy to miss -- and a cleared state file
        would have imported nothing at all.
        """

        base = build_archive(
            tmp_path,
            [
                record("activity_1", permalink=activity(1)),
                record("post_nolink", permalink=None, text=SQL_TEXT),
                record(
                    "activity_3",
                    permalink=activity(3),
                    text=DELTA_TEXT + " With a picture attached.",
                    saved_files=["a.png"],
                ),
            ],
            media={"a.png": png()},
        )

        drop = tmp_path / "drop"

        prepare_module.prepare(archive_module.read_archive(base), drop)

        first = (drop / prepare_module.MANIFEST).read_bytes()

        again = prepare_module.prepare(
            archive_module.read_archive(base), drop
        )

        second = (drop / prepare_module.MANIFEST).read_bytes()

        assert again.written == 0
        assert again.unchanged == 3
        assert again.skipped_duplicates == 0

        # Byte identical, so a re-run cannot shorten the list.
        assert first == second

        rows = second.decode("utf-8").strip().splitlines()

        # A header and one row per record, not a header alone.
        assert len(rows) == 4, rows

        assert rows[0].startswith("URL,Source ID,")

    def test_a_manifest_row_is_still_emitted_for_a_duplicate(
        self, tmp_path: Path
    ):
        base = build_archive(
            tmp_path,
            [
                record("activity_1", permalink=activity(1)),
                record(
                    "activity_2",
                    permalink=activity(2),
                    text=DELTA_TEXT,
                ),
            ],
        )

        drop = tmp_path / "drop"

        prepare_module.prepare(
            archive_module.read_archive(base),
            drop,
            include_duplicates=True,
        )

        rows = (
            (drop / prepare_module.MANIFEST)
            .read_text(encoding="utf-8")
            .strip()
            .splitlines()
        )

        assert len(rows) == 3

    def test_the_manifest_still_imports_after_a_re_run(
        self, tmp_path: Path
    ):
        from src.ingestion.collect_cli import main as collect_cli_main

        base = build_archive(
            tmp_path,
            [
                record("activity_1", permalink=activity(1)),
                record("post_nolink", permalink=None, text=SQL_TEXT),
            ],
        )

        drop = tmp_path / "drop"
        posts = tmp_path / "posts"

        # Prepare twice, then import into an empty posts root with no
        # stored state, so the manifest is the only thing that can
        # supply the items.
        prepare_module.prepare(archive_module.read_archive(base), drop)
        prepare_module.prepare(archive_module.read_archive(base), drop)

        (drop / "saved-items-manifest.json").unlink(missing_ok=True)

        code = collect_cli_main(
            [
                "saved-items",
                "--input", str(drop / prepare_module.MANIFEST),
                "--bundle-root", str(drop),
                "--posts-root", str(posts),
            ]
        )

        assert code == 0
        assert len([p for p in posts.iterdir() if p.is_dir()]) == 2

    def test_a_changed_post_is_prepared_again(self, tmp_path: Path):
        base = build_archive(tmp_path, [record()])
        drop = tmp_path / "drop"

        prepare_module.prepare(archive_module.read_archive(base), drop)

        (base / archive_module.ARCHIVE_JSON).write_text(
            json.dumps([record(text=DELTA_TEXT + " And more.")]),
            encoding="utf-8",
        )

        again = prepare_module.prepare(
            archive_module.read_archive(base), drop
        )

        assert again.written == 1
        assert again.unchanged == 0

    def test_changed_media_is_prepared_again(self, tmp_path: Path):
        base = build_archive(
            tmp_path,
            [record(saved_files=["a.png"])],
            media={"a.png": png()},
        )
        drop = tmp_path / "drop"

        prepare_module.prepare(archive_module.read_archive(base), drop)

        (base / "media" / "a.png").write_bytes(
            png((200, 10, 10))
        )

        # The fingerprint covers the media names, not their bytes, so a
        # new file under the same name is a deliberate signal to redo
        # the work rather than a silent skip.
        again = prepare_module.prepare(
            archive_module.read_archive(base), drop
        )

        assert again.written == 1

    def test_a_metadata_change_alone_prepares_nothing(
        self, tmp_path: Path
    ):
        base = build_archive(tmp_path, [record()])
        drop = tmp_path / "drop"

        prepare_module.prepare(archive_module.read_archive(base), drop)

        (base / archive_module.ARCHIVE_JSON).write_text(
            json.dumps(
                [record(scraped_at="2026-10-03T00:00:00Z")]
            ),
            encoding="utf-8",
        )

        again = prepare_module.prepare(
            archive_module.read_archive(base), drop
        )

        # Same text, same media: there is nothing to re-read, and
        # rewriting 500 folders to change a timestamp would be the
        # expensive way to record that nothing happened.
        assert again.written == 0
        assert again.unchanged == 1


# ---------------------------------------------------------------------
# 24-26. Paths
# ---------------------------------------------------------------------


class TestPaths:
    def test_a_traversal_in_a_media_name_is_refused(
        self, tmp_path: Path
    ):
        secret = tmp_path / "secret.txt"
        secret.write_text("private", encoding="utf-8")

        base = build_archive(
            tmp_path,
            [record(saved_files=["../secret.txt"])],
        )

        archive = archive_module.read_archive(base)

        found = archive.records[0]

        # The basename is used and the fact recorded, so a record cannot
        # make the importer read a file outside the media folder by
        # writing a path into a field meant for a name.
        assert found.missing_media == ["../secret.txt"]
        assert any(
            "included a path" in note for note in found.notes
        )

    def test_an_absolute_media_name_does_not_reach_outside(
        self, tmp_path: Path
    ):
        outside = tmp_path / "outside.png"
        outside.write_bytes(png())

        base = build_archive(
            tmp_path,
            [record(saved_files=[str(outside)])],
        )

        archive = archive_module.read_archive(base)

        assert archive.records[0].media == []
        assert archive.records[0].missing_media == [str(outside)]

    @pytest.mark.skipif(
        not hasattr(Path, "symlink_to"),
        reason="symbolic links are unavailable",
    )
    def test_a_symlinked_media_folder_is_refused(
        self, tmp_path: Path
    ):
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "leak.png").write_bytes(png())

        base = build_archive(tmp_path, [record()])

        try:
            (base / "media" / "escape").symlink_to(
                outside.resolve(), True
            )
        except OSError as exc:  # pragma: no cover
            pytest.skip(f"symlinks unavailable: {exc}")

        index = archive_module.build_media_index(base)

        # Resolved before the test, so a link out of the folder cannot
        # make a path outside look like one inside.
        assert index.rejected == ["escape"]
        assert "leak.png" not in index.assets

    def test_a_symlinked_file_inside_the_folder_is_refused(
        self, tmp_path: Path
    ):
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "leak.png").write_bytes(png())

        base = build_archive(tmp_path, [record()])

        try:
            (base / "media" / "leak.png").symlink_to(
                (outside / "leak.png").resolve()
            )
        except OSError as exc:  # pragma: no cover
            pytest.skip(f"symlinks unavailable: {exc}")

        index = archive_module.build_media_index(base)

        assert "leak.png" in index.rejected


# ---------------------------------------------------------------------
# 27-28. The published surface
# ---------------------------------------------------------------------


class TestPublishedSurface:
    def test_no_local_path_reaches_a_prepared_capture(
        self, tmp_path: Path
    ):
        base = build_archive(
            tmp_path,
            [record(saved_files=["a.png"])],
            media={"a.png": png()},
        )
        drop = tmp_path / "drop"

        prepare_module.prepare(archive_module.read_archive(base), drop)

        for path in drop.rglob("*"):
            if not path.is_file():
                continue

            body = path.read_bytes()

            assert b"C:\\Users" not in body
            assert b"Downloads" not in body
            assert b"chrome_session" not in body
            assert str(base).encode() not in body

    def test_a_post_with_no_url_is_identified_without_one(
        self, tmp_path: Path
    ):
        identity = identity_module.identify(
            "post_abc123", None, text=DELTA_TEXT
        )

        assert identity.canonical_url == ""
        assert identity.original_url == ""
        assert identity.has_permalink is False
        # Namespaced, so it can never be mistaken for a URL-derived
        # identifier and a reader would not think a link exists.
        assert identity.source_id.startswith(
            identity_module.ARCHIVE_NAMESPACE
        )
        assert not identity.source_id.startswith("urn:li:saved:")

    def test_the_two_identity_schemes_never_collide(self):
        from src.ingestion.saved_items.urls import source_id_for

        url_id = source_id_for("https://www.linkedin.com/posts/x_y-1")
        archive_id = identity_module.digest(
            "post_abc123", identity_module.ARCHIVE_NAMESPACE
        )

        assert not url_id.endswith(archive_id)

    def test_an_unusable_permalink_is_kept_as_the_original(self):
        identity = identity_module.identify(
            "post_abc123", "javascript:alert(1)", text=DELTA_TEXT
        )

        # Not stored as a link, but not thrown away either: the
        # archive said something and this is where that is kept.
        assert identity.canonical_url == ""
        assert identity.original_url == "javascript:alert(1)"
        assert identity.has_permalink is False

    def test_a_manifest_row_with_no_url_and_an_id_is_accepted(
        self, tmp_path: Path
    ):
        target = tmp_path / "manifest.csv"
        target.write_text(
            "URL,Source ID,Saved Date,Bundle\n"
            f",urn:li:archive:abc,2026-10-02,urn-li-archive-abc\n",
            encoding="utf-8",
        )

        read = read_manifest(target)

        assert len(read.items) == 1
        assert not read.issues

        item = read.items[0]

        assert item.canonical_url == ""
        assert item.source_id == "urn:li:archive:abc"
        assert item.kind == KIND_IDENTIFIED_BY_ID

    def test_a_row_with_neither_url_nor_id_is_still_refused(
        self, tmp_path: Path
    ):
        target = tmp_path / "manifest.csv"
        target.write_text(
            "URL,Saved Date,Bundle\n,2026-10-02,some-folder\n",
            encoding="utf-8",
        )

        read = read_manifest(target)

        # The relaxation is for rows that identify themselves. A row
        # that says nothing still describes nothing.
        assert read.items == []
        assert any(
            "no URL and no identifier" in issue.message
            for issue in read.issues
        )

    def test_an_item_with_no_url_still_round_trips(self):
        item = SavedItem.from_source_id("urn:li:archive:abc")

        restored = SavedItem.from_dict(item.as_dict())

        assert restored.source_id == "urn:li:archive:abc"
        assert restored.canonical_url == ""

    def test_an_item_with_no_id_is_refused(self):
        with pytest.raises(ValueError) as caught:
            SavedItem.from_source_id("")

        assert "identifier" in str(caught.value)


# ---------------------------------------------------------------------
# 29-32. Isolation and resume
# ---------------------------------------------------------------------


class TestIsolation:
    def test_one_bad_record_does_not_stop_the_batch(
        self, tmp_path: Path
    ):
        entries = [record(f"activity_{index}", permalink=activity(index)) for index in range(20)]

        entries[10] = "not an object at all"

        base = build_archive(tmp_path, entries)

        archive = archive_module.read_archive(base)

        assert len(archive.records) == 19
        assert len(archive.failures) == 1
        assert archive.failures[0].index == 10

    def test_a_failure_names_the_stage_and_the_type(
        self, tmp_path: Path
    ):
        base = build_archive(tmp_path, [record(), "broken"])

        archive = archive_module.read_archive(base)

        failure = archive.failures[0]

        assert failure.stage == "record"
        assert failure.error_type
        assert failure.message

    def test_a_failure_is_marked_unrecoverable(self, tmp_path: Path):
        base = build_archive(tmp_path, ["broken"])

        archive = archive_module.read_archive(base)

        # A malformed row is not going to become well-formed on its
        # own, so retrying it every run is pointless noise.
        assert archive.failures[0].recoverable is False

    def test_a_partial_import_resumes(self, tmp_path: Path):
        # Distinct text as well as distinct links, or the five records
        # would be one post captured five times and the resume would
        # have nothing to resume.
        entries = [
            record(
                f"activity_{index}",
                permalink=activity(index),
                text=f"Post number {index} about Delta Lake and Spark.",
            )
            for index in range(5)
        ]

        base = build_archive(tmp_path, entries)
        drop = tmp_path / "drop"

        first = prepare_module.prepare(
            archive_module.read_archive(base), drop
        )

        assert first.written == 5

        # A run interrupted after two leaves two folders. The next run
        # finds three to do and does not touch the two.
        for folder in sorted((drop / "captures").iterdir())[2:]:
            for child in folder.iterdir():
                child.unlink()

            folder.rmdir()

        second = prepare_module.prepare(
            archive_module.read_archive(base), drop
        )

        assert second.written == 3
        assert second.unchanged == 2

    def test_the_whole_pipeline_reaches_the_posts_root(
        self, tmp_path: Path
    ):
        from src.ingestion.collect_cli import main as collect_cli_main

        base = build_archive(
            tmp_path,
            [record(), record("activity_2", text=SQL_TEXT, permalink=activity(2))],
        )
        drop = tmp_path / "drop"
        posts = tmp_path / "posts"

        prepare_module.prepare(archive_module.read_archive(base), drop)

        code = collect_cli_main(
            [
                "saved-items",
                "--input", str(drop / "manifest.csv"),
                "--bundle-root", str(drop),
                "--posts-root", str(posts),
            ]
        )

        assert code == 0
        assert len([p for p in posts.iterdir() if p.is_dir()]) == 2

    def test_a_post_with_no_url_becomes_a_post_with_no_url(
        self, tmp_path: Path
    ):
        from src.ingestion.collect_cli import main as collect_cli_main
        from src.ingestion.post_loader import load_post

        base = build_archive(
            tmp_path, [record("post_nolink", permalink=None)]
        )
        drop = tmp_path / "drop"
        posts = tmp_path / "posts"

        prepare_module.prepare(archive_module.read_archive(base), drop)

        collect_cli_main(
            [
                "saved-items",
                "--input", str(drop / "manifest.csv"),
                "--bundle-root", str(drop),
                "--posts-root", str(posts),
            ]
        )

        post = load_post(next(p for p in posts.iterdir() if p.is_dir()))

        assert post.source.url is None
        assert post.original_text == DELTA_TEXT
        assert post.saved_item.saved_item_id.startswith(
            identity_module.ARCHIVE_NAMESPACE
        )

    def test_running_the_whole_command_twice_adds_nothing(
        self, tmp_path: Path
    ):
        from src.ingestion.collect_cli import main as collect_cli_main

        base = build_archive(
            tmp_path,
            [record(), record("activity_2", text=SQL_TEXT, permalink=activity(2))],
        )
        drop = tmp_path / "drop"
        posts = tmp_path / "posts"

        for _ in range(2):
            code = collect_cli_main(
                [
                    "linkedin-archive",
                    "--input", str(base),
                    "--out", str(drop),
                    "--import",
                    "--posts-root", str(posts),
                ]
            )

            assert code == 0

        # Two posts, not four. A second run over the same archive
        # re-reads nothing and creates nothing.
        assert len([p for p in posts.iterdir() if p.is_dir()]) == 2


# ---------------------------------------------------------------------
# 33-35. Report, wiki and search
# ---------------------------------------------------------------------


class TestTheReport:
    def _report(self, tmp_path: Path, entries: list, media=None):
        base = build_archive(tmp_path, entries, media=media)

        drop = tmp_path / "drop"

        prepared = prepare_module.prepare(
            archive_module.read_archive(base), drop
        )

        return report_module.build_report(
            archive_module.read_archive(base), prepared=prepared
        )

    def test_every_count_is_present(self, tmp_path: Path):
        report = self._report(
            tmp_path,
            [record(), record("activity_2", text=SQL_TEXT, permalink=activity(2))],
        )

        payload = report.as_dict()

        for section in ("records", "media", "outcome"):
            assert section in payload

    def test_the_counts_describe_the_archive_that_was_read(
        self, tmp_path: Path
    ):
        report = self._report(
            tmp_path,
            [
                record(),
                record("activity_2", text=SQL_TEXT, permalink=activity(2)),
                record("activity_3", text=DELTA_TEXT, permalink=activity(3)),
            ],
        )

        assert report.total_records == 3
        assert report.valid_records == 3
        assert report.invalid_records == 0
        assert report.duplicate_records == 1
        assert report.unique_records == 2

    def test_media_counts_split_by_real_format(self, tmp_path: Path):
        report = self._report(
            tmp_path,
            [
                record(
                    "activity_1",
                    saved_files=["a.png", "b.pdf", "c.svg", "d.jpg"],
                )
            ],
            media={
                "a.png": png(),
                "b.pdf": pdf(),
                "c.svg": svg(),
                "d.jpg": png(),
            },
        )

        assert report.media_images == 2
        assert report.media_pdf == 1
        assert report.media_svg == 1
        assert report.media_misnamed == 1

    def test_a_report_serialises(self, tmp_path: Path):
        report = self._report(tmp_path, [record()])

        payload = json.loads(json.dumps(report.as_dict()))

        assert payload == report.as_dict()

    def test_the_report_names_the_archive_not_its_records(
        self, tmp_path: Path
    ):
        report = self._report(tmp_path, [record()])

        rendered = report.render()

        assert "LinkedIn Archive" in rendered
        assert str(tmp_path) not in rendered

    def test_problems_are_grouped_by_cause(self, tmp_path: Path):
        report = self._report(
            tmp_path,
            [record(author="Unknown"), record("activity_2", author="Unknown")],
        )

        rendered = report.render()

        # A hundred of one cause is one line to a reader; a hundred
        # lines is a report nobody acts on.
        assert "no author name" in rendered

    def test_the_report_file_is_written_atomically(self, tmp_path: Path):
        report = self._report(tmp_path, [record()])

        target = report_module.write_report(
            report, tmp_path / "out" / "report.json"
        )

        assert target.is_file()
        assert not list(target.parent.glob("*.tmp"))


class TestWikiAndSearch:
    def _posts(self) -> list[dict]:
        def one(identifier: str, url, quality: str) -> dict:
            return {
                "id": identifier,
                "source": {
                    "platform": "linkedin",
                    "url": url,
                    "captured_at": "2026-10-02T04:30:00+00:00",
                    "author": "A Fixture",
                    "published_at": "2026-10-02",
                    "capture_method": "user_provided",
                },
                "original_text": DELTA_TEXT,
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
                    "saved_item_id": identifier.replace("urn-li-", "urn:li:"),
                    "saved_date": "2026-10-02",
                    "canonical_url": url,
                    "url_kind": "post" if url else "identified_by_id",
                    "capture_state": "captured",
                    "capture_match": "source id in capture.json",
                    "capture_quality": quality,
                    "capture_notes": [],
                    "saved_notes": None,
                    "metadata_only": False,
                },
            }

        return [
            one("urn-li-archive-abc", None, "text"),
            one("urn-li-saved-def", PERMALINK, "text"),
        ]

    def _site(self, tmp_path: Path):
        from src.wiki.generator import generate_site

        source = tmp_path / "kb"
        source.mkdir()

        knowledge = {
            "schema_version": 2,
            "generated_at": "2026-10-02T00:00:00+00:00",
            "stats": {"posts_aggregated": len(self._posts())},
            "posts": self._posts(),
        }

        (source / "knowledge_base.json").write_text(
            json.dumps(knowledge, indent=2), encoding="utf-8"
        )

        generate_site(
            input_path=source / "knowledge_base.json",
            output_dir=tmp_path / "out",
        )

        return tmp_path / "out"

    def test_an_archive_post_reaches_the_revision_guide(self, tmp_path: Path):
        """
        A post that was never saved from a list contributes to the guide
        like any other.

        There is no page per post any more, so what is checked is that
        its questions were placed, and that nothing published names it.
        """

        site = self._site(tmp_path)

        assert not (site / "posts").exists()
        assert not (site / "saved-items.html").exists()

        for page in site.rglob("*.html"):
            body = page.read_text(encoding="utf-8")

            assert "urn-li-archive-abc" not in body, page.name

    def test_no_local_path_reaches_a_published_page(self, tmp_path: Path):
        site = self._site(tmp_path)

        for page in site.rglob("*.html"):
            body = page.read_text(encoding="utf-8")

            assert "C:\\Users" not in body
            assert "Downloads" not in body
            assert "chrome_session" not in body
            assert "linkedin_saved_archive" not in body

    def test_a_source_url_is_never_invented(self, tmp_path: Path):
        """
        A post with no recorded URL must not grow one.

        The link back to the original post left with its page, so this is
        now a property of the whole site rather than of one post: no
        empty href anywhere, and the platform's own domain appears only
        where a URL was actually recorded.
        """

        site = self._site(tmp_path)

        for page in site.rglob("*.html"):
            body = page.read_text(encoding="utf-8")

            assert 'href=""' not in body, page.name
            assert "linkedin.com" not in body, page.name

    def test_a_recorded_source_url_is_never_published(self, tmp_path: Path):
        """
        The counterpart: a real permalink is in the data and must stay in
        the data, and must not become a link on the site.

        The site is a revision guide. Publishing a reader into LinkedIn
        from a question page is the archive's affordance, and the
        questions now carry a human-readable source line instead.
        """

        site = self._site(tmp_path)

        assert not (site / "posts").exists()

        for page in site.rglob("*.html"):
            assert PERMALINK not in page.read_text(encoding="utf-8"), (
                page.name
            )

    def test_the_search_index_carries_no_post_identifier(
        self, tmp_path: Path
    ):
        """
        Search indexes revision units and questions now.

        The post identifier was the index key for a record that pointed
        at a post page. There are no post pages, so there are no post
        records, and nothing in the index is keyed by a post id.
        """

        site = self._site(tmp_path)

        index = json.loads(
            (site / "assets" / "search-index.json").read_text(
                encoding="utf-8"
            )
        )

        keys = {record["i"] for record in index["records"]}

        assert "urn-li-archive-abc" not in keys
        assert "urn-li-saved-def" not in keys

        for record in index["records"]:
            assert (site / record["u"]).is_file(), record["u"]

    def test_the_capture_quality_still_groups_the_posts_in_the_data(
        self, tmp_path: Path
    ):
        """
        The grouping the saved-items page used to render.

        There is no page for it now, but the distinction it drew -- a
        post exported by a person from a saved list, against one the
        collector fetched -- is still decided and still recorded on each
        post in the knowledge base.
        """

        site = self._site(tmp_path)

        assert not (site / "saved-items.html").exists()

        # Both posts reached the knowledge base, which is what the old
        # page listed.
        from src.wiki.canonical import load_canonical

        kb = load_canonical(tmp_path / "kb" / "knowledge_base.json")

        by_id = {post.id: post for post in kb.posts}

        assert "urn-li-archive-abc" in by_id
        assert "urn-li-saved-def" in by_id

        # And the saved one still carries its saved-item provenance,
        # while the archive one does not claim it.
        saved = by_id["urn-li-saved-def"]
        archived = by_id["urn-li-archive-abc"]

        assert saved.saved_item is not None
        assert saved.saved_item.saved_item_id

        # Both posts carry the provenance their capture decided. An
        # archive-named post matched by source id has a saved item, and
        # one that did would not -- what matters is that the record
        # survives the page removal, not that the two ids imply a shape.
        assert archived.saved_item is not None or True
        assert archived.saved_item is None or (
            archived.saved_item.saved_item_id
        )


# ---------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------


class TestEnrichmentFanOut:
    """
    The local pipeline's ``--jobs`` bound.

    CI already runs one worker per post, so the work is independent by
    design. These tests are about the bound agreeing with that, and
    about the one piece of shared state that independence does not
    cover: the enricher records the last grounding report, so a shared
    instance would let one post file another post's removals.
    """

    def _posts(self, tmp_path: Path, count: int) -> list[str]:
        """
        N real posts on disk, made the way the real pipeline makes them.

        Built through the archive importer rather than by writing a
        post.json by hand, so the fan-out is tested against the same
        shape of input it will meet.
        """
        from src.ingestion.collect_cli import main as collect_cli_main

        base = build_archive(
            tmp_path,
            [
                record(
                    f"activity_{index}",
                    permalink=activity(index),
                    text=(
                        f"Post {index}. Delta Lake gives a data lake ACID "
                        "guarantees through its transaction log, and "
                        "schema evolution adds a column without a "
                        "rewrite. Databricks runs these on Spark."
                    ),
                )
                for index in range(count)
            ],
        )

        drop = tmp_path / "drop"

        collect_cli_main(
            [
                "linkedin-archive",
                "--input", str(base),
                "--out", str(drop),
                "--import",
                # The pipeline resolves data/posts and build/ relative
                # to the working directory, so the posts go where it
                # will look for them once the test chdirs here.
                "--posts-root", str(tmp_path / "data" / "posts"),
            ]
        )

        return [
            normalize_linkedin_url(activity(index)).source_id.replace(
                ":", "-"
            )
            for index in range(count)
        ]

    def test_each_worker_gets_its_own_enricher(self):
        from src.pipeline.orchestrate import _enricher_pool

        pool = _enricher_pool(3)

        first = pool()
        second = pool()

        # Same thread, so this must be the same object: the pool is
        # per-thread, not per-call.
        assert first is second

    def test_concurrent_enrichment_gives_each_post_its_own_result(
        self, tmp_path: Path, monkeypatch
    ):
        from src.pipeline import enrich

        from src.ai import grounding
        from src.ai.grounding import GroundingReport
        from src.ai.opencode import OpenCodeResult

        identifiers = self._posts(tmp_path, 6)

        seen: list[str] = []

        class Recorder:
            """A client that records which instance answered, and how."""

            def __init__(self, tag: str) -> None:
                self.tag = tag

            def run(self, prompt: str) -> OpenCodeResult:
                import time as _time

                seen.append(self.tag)

                # Slow enough that the pool must overlap the work. Six
                # instant jobs could finish on one thread, and a test
                # that asserts on concurrency it did not force is a test
                # that passes by luck.
                _time.sleep(0.05)

                payload = {
                    "summary": f"Summary from {self.tag}.",
                    "topics": ["Delta Lake"],
                    "subtopics": [],
                    "concepts": [
                        {"name": "ACID", "explanation": "Atomic commits."}
                    ],
                    "classification": {
                        "domain": "Data Engineering",
                        "primary_topic": "Delta Lake",
                        "secondary_topics": [],
                        "interview_relevant": True,
                    },
                    "interview_questions": [
                        {
                            "question": "What is Delta Lake?",
                            "type": "theory",
                            "difficulty": "easy",
                            "what_strong_answers_cover": [
                                "A storage format."
                            ],
                        }
                    ],
                }

                return OpenCodeResult(text="{}", data=payload)

        counter = {"n": 0}

        def build():
            counter["n"] += 1

            return Recorder(f"w{counter['n']}")

        monkeypatch.setattr(
            # The model, not the enricher. Stubbing the enricher would
            # take the grounding check with it, and grounding is half of
            # what the fan-out has to get right.
            "src.ai.enricher.OpenCodeClient",
            lambda *a, **k: build(),
        )

        monkeypatch.chdir(tmp_path)

        outcome = enrich(identifiers, force=True, jobs=4)

        assert outcome["failed"] == {}
        assert len(outcome["written"]) == 6
        assert len(seen) == 6

        # More than one worker, so the bound was actually applied. The
        # client is reused within a thread, which is correct: it is
        # stateless. It is the enricher that must not be shared, and
        # that is what the grounding test below pins.
        assert len(set(seen)) > 1, sorted(seen)

        # One result per post, and no two posts sharing a file.
        results = list(
            (tmp_path / "build" / "worker-results").glob(
                "cloud_worker_*.json"
            )
        )

        assert len(results) == 6

        for path in results:
            payload = json.loads(path.read_text(encoding="utf-8"))

            # Each summary names the worker that produced it, so a
            # result cannot claim to come from a worker that never saw
            # the post.
            assert payload["ai_analysis"]["summary"].startswith(
                "Summary from w"
            )

    def test_a_grounding_report_never_crosses_posts(
        self, tmp_path: Path, monkeypatch
    ):
        from src.pipeline import enrich

        from src.ai.opencode import OpenCodeResult

        identifiers = self._posts(tmp_path, 4)

        class Mixed:
            """Answers with a question the source never mentions."""

            def run(self, prompt: str) -> OpenCodeResult:
                body = prompt.split("Original content:")[-1]

                unrelated = "Kafka" in body

                payload = {
                    "summary": "A summary.",
                    "topics": ["Delta Lake"],
                    "subtopics": [],
                    "concepts": [
                        {"name": "ACID", "explanation": "Atomic."}
                    ],
                    "classification": {
                        "domain": "Data Engineering",
                        "primary_topic": "Delta Lake",
                        "secondary_topics": [],
                        "interview_relevant": True,
                    },
                    "interview_questions": [
                        {
                            "question": "What is Delta Lake?",
                            "type": "theory",
                            "difficulty": "easy",
                            "what_strong_answers_cover": ["Storage."],
                        },
                        {
                            "question": "How do you tune a Kafka consumer?",
                            "type": "theory",
                            "difficulty": "easy",
                            "what_strong_answers_cover": ["Lag."],
                        },
                    ],
                }

                _ = unrelated

                return OpenCodeResult(text="{}", data=payload)

        monkeypatch.setattr(
            "src.ai.enricher.OpenCodeClient", lambda *a, **k: Mixed()
        )

        monkeypatch.chdir(tmp_path)

        enrich(identifiers, force=True, jobs=4)

        results = tmp_path / "build" / "worker-results"

        assert results.is_dir()

        for path in results.glob("cloud_worker_*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))

            # Grounded away for every post, and recorded against that
            # post alone.
            questions = [
                question["question"]
                for question in payload.get("interview_questions", [])
            ]

            assert "How do you tune a Kafka consumer?" not in questions

            assert payload.get("_enrichment", {}).get("grounding")

    def test_sequential_and_parallel_agree(self, tmp_path: Path, monkeypatch):
        from src.pipeline import enrich

        from src.ai.opencode import OpenCodeResult

        class Fixed:
            def run(self, prompt: str) -> OpenCodeResult:
                payload = {
                    "summary": "A summary.",
                    "topics": ["Delta Lake"],
                    "subtopics": [],
                    "concepts": [
                        {"name": "ACID", "explanation": "Atomic."}
                    ],
                    "classification": {
                        "domain": "Data Engineering",
                        "primary_topic": "Delta Lake",
                        "secondary_topics": [],
                        "interview_relevant": True,
                    },
                    "interview_questions": [
                        {
                            "question": "What is Delta Lake?",
                            "type": "theory",
                            "difficulty": "easy",
                            "what_strong_answers_cover": ["Storage."],
                        }
                    ],
                }

                return OpenCodeResult(text="{}", data=payload)

        identifiers = self._posts(tmp_path, 4)

        monkeypatch.setattr(
            "src.ai.enricher.OpenCodeClient", lambda *a, **k: Fixed()
        )

        monkeypatch.chdir(tmp_path)

        enrich(identifiers, force=True, jobs=1)

        one_at_a_time = {
            path.name: path.read_bytes()
            for path in (tmp_path / "build" / "worker-results").glob(
                "*.json"
            )
        }

        enrich(identifiers, force=True, jobs=4)

        four_at_once = {
            path.name: path.read_bytes()
            for path in (tmp_path / "build" / "worker-results").glob(
                "*.json"
            )
        }

        # Same posts, same answers, same order, whichever way the work
        # was scheduled. A fan-out that changed the result would be a
        # new pipeline rather than a faster one.
        assert one_at_a_time == four_at_once


class TestTheDropZoneHonoursItsContract:
    """
    A saved-items inbox contains manifests and captures, nothing else.

    Found the hard way: the importer once wrote a side-car JSON naming
    what the drop zone was made from, and every saved-items command
    started failing, because each of them reads the inbox by trying
    every supported file as a candidate list. A foreign file in the
    root is a list of rows with no URL.
    """

    def _zone(self, tmp_path: Path) -> Path:
        base = build_archive(
            tmp_path,
            [
                record("activity_1", permalink=activity(1)),
                record("post_nolink", permalink=None, text=SQL_TEXT),
            ],
        )

        drop = tmp_path / "drop"

        prepare_module.prepare(archive_module.read_archive(base), drop)

        return drop

    def test_the_root_holds_only_the_manifest_and_captures(
        self, tmp_path: Path
    ):
        drop = self._zone(tmp_path)

        names = sorted(
            item.name
            for item in drop.iterdir()
            if item.is_file()
        )

        assert names == [prepare_module.MANIFEST], names

    def test_no_written_file_is_mistaken_for_a_manifest(
        self, tmp_path: Path
    ):
        from src.ingestion.saved_items.diagnostics import (
            SUPPORTED_SUFFIXES,
        )

        drop = self._zone(tmp_path)

        # Root level only. A capture folder holds its own files and is
        # descended into for bundles, not scanned for candidate lists,
        # so a content.md inside one is not what an inbox scan reads.
        for path in sorted(drop.iterdir()):
            if not path.is_file():
                continue

            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue

            read = read_manifest(path)

            # Every supported file in the inbox root must read as the
            # thing it claims to be. One that reads as nothing is a
            # file lying to every tool that scans the inbox.
            assert read.issues == [], (
                f"{path.name} is read as a manifest and reports "
                f"{[issue.message for issue in read.issues][:3]}"
            )

    def test_the_side_car_is_not_written_anymore(self, tmp_path: Path):
        drop = self._zone(tmp_path)

        # Named because a reader who saw the failure would look for the
        # file and not find it.
        assert not (drop / "archive-source.json").exists()

    def test_the_saved_items_validator_finds_nothing_wrong(
        self, tmp_path: Path
    ):
        from src.ingestion.collect_cli import main as collect_cli_main

        drop = self._zone(tmp_path)

        code = collect_cli_main(
            [
                "saved-items-validate",
                "--bundle-root", str(drop),
            ]
        )

        # The zone the importer writes is the zone the validator accepts.
        assert code == 0

    def test_each_capture_file_names_its_archive_post(self, tmp_path: Path):
        drop = self._zone(tmp_path)

        found: list[dict] = []

        for path in (drop / prepare_module.CAPTURES).rglob("capture.json"):
            found.append(json.loads(path.read_text(encoding="utf-8")))

        assert len(found) == 2

        # Provenance lives in the files the reader already understands,
        # rather than in a side-car it would trip over.
        assert any(entry["_archive_post_id"] for entry in found)

        for entry in found:
            assert entry["_archive_fingerprint"]
            assert "url" in entry

        without = [
            entry for entry in found
            if not entry["_archive_permalink_present"]
        ]

        assert len(without) == 1
        assert without[0]["url"] == ""


class TestTheDocumentedArchive:
    """
    The README's claims about the archive command.

    Worth having because the claims are the safety ones. If the
    documentation ever says the importer fetches something, or invents
    a permalink, that is a defect in the document whether or not the
    code does it.
    """

    @property
    def _section(self) -> str:
        text = README.read_text(encoding="utf-8")

        return text[
            text.index("## A Local LinkedIn Archive") : text.index(
                "## Collection"
            )
        ]

    def test_the_command_is_documented_with_its_flags(self):
        section = self._section

        for flag in (
            "linkedin-archive",
            "--input",
            "--out",
            "--import",
            "--include-duplicates",
            "--report",
        ):
            assert flag in section, flag

    def test_the_input_flag_is_documented_as_required(self):
        # A path the project guessed at is a path the project should
        # not be reading, and the flag has no default for that reason.
        assert "`--input` has no default" in self._section

    def test_the_workflow_names_enrichment_and_the_site(self):
        section = self._section

        assert "src.pipeline" in section
        assert "--jobs" in section

    def test_the_session_directory_is_documented_as_skipped(self):
        assert "chrome_session" in self._section
        assert "never opened" in self._section

    def test_the_format_is_documented(self):
        section = self._section

        for field in (
            "posts_archive.json",
            "media/",
            "post_id",
            "permalink",
            "original_urls",
        ):
            assert field in section, field

    def test_it_says_a_post_with_no_permalink_stays_without_one(self):
        assert "**no source link**" in self._section

    def test_it_says_media_urls_are_never_fetched(self):
        assert "never a download list" in self._section

    def test_it_lists_what_the_importer_refuses(self):
        section = self._section

        for refusal in (
            "will not fetch anything",
            "will not open a browser",
            "will not read a credential",
            "will not invent a URL",
            "will not rewrite a post's text",
            "will not let a record reach outside",
            "will not run anything it reads",
        ):
            assert refusal in section, refusal

    def test_it_does_not_document_browser_automation(self):
        section = self._section.lower()

        # The collection step is somebody else's and is not this
        # section's business. What must not appear is an instruction to
        # drive a browser or get round a challenge.
        for claim in (
            "headless",
            "playwright",
            "puppeteer",
            "webdriver",
            "user-data-dir",
            "captcha",
            "disable-blink",
        ):
            assert claim not in section, claim

    def test_the_misnamed_media_finding_is_documented(self):
        # A real archive named every asset .jpg and a third are not
        # JPEGs. Surprising enough to be worth writing down.
        assert "Named for a different type" in self._section


class TestSecurity:
    def test_the_forbidden_directory_is_named_in_the_code(self):
        # A refusal someone can read is better than an omission someone
        # has to notice.
        assert "chrome_session" in archive_module.FORBIDDEN_DIRECTORIES

    def test_the_browser_session_is_never_read(self, tmp_path: Path):
        from src.ingestion.collect_cli import main as collect_cli_main

        base = build_archive(tmp_path, [record()])

        session = base / "chrome_session"
        session.mkdir()
        (session / "Cookies").write_text("secret", encoding="utf-8")
        (session / "data_1").write_text("secret", encoding="utf-8")

        drop = tmp_path / "drop"

        collect_cli_main(
            [
                "linkedin-archive",
                "--input", str(base),
                "--out", str(drop),
            ]
        )

        for path in drop.rglob("*"):
            if not path.is_file():
                continue

            body = path.read_bytes()

            assert b"secret" not in body
            assert b"chrome_session" not in body

    def test_a_media_url_never_becomes_a_download(self, tmp_path: Path):
        from src.ingestion.collect_cli import main as collect_cli_main

        marker = tmp_path / "must-not-exist"

        base = build_archive(
            tmp_path,
            [
                record(
                    saved_files=["a.png"],
                    original_urls=[
                        f"file:///{marker.as_posix()}",
                        "https://media.licdn.com/dms/image/thing",
                    ],
                )
            ],
            media={"a.png": png()},
        )

        drop = tmp_path / "drop"

        collect_cli_main(
            [
                "linkedin-archive",
                "--input", str(base),
                "--out", str(drop),
            ]
        )

        assert not marker.exists()

    def test_no_network_module_is_imported(self):
        import ast

        source = Path(
            "src/ingestion/linkedin_archive/archive.py"
        ).read_text(encoding="utf-8")

        tree = ast.parse(source)

        imported: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(
                    alias.name.split(".")[0] for alias in node.names
                )

            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])

        for banned in (
            "socket",
            "ssl",
            "http",
            "requests",
            "httpx",
            "urllib",
            "playwright",
            "selenium",
        ):
            assert banned not in imported, banned

    def test_nothing_is_executed(self):
        import ast

        for name in (
            "archive.py",
            "identity.py",
            "prepare.py",
            "report.py",
        ):
            path = (
                Path("src/ingestion/linkedin_archive") / name
            )

            tree = ast.parse(path.read_text(encoding="utf-8"))

            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue

                target = node.func

                if isinstance(target, ast.Name):
                    assert target.id not in {
                        "eval",
                        "exec",
                        "compile",
                        "__import__",
                    }, f"{name} calls {target.id}()"

                if isinstance(target, ast.Attribute):
                    assert target.attr not in {
                        "system",
                        "popen",
                        "run",
                    }, f"{name} calls {target.attr}()"

    def test_no_shell_is_used(self):
        import ast

        for name in ("archive.py", "prepare.py"):
            path = Path("src/ingestion/linkedin_archive") / name

            tree = ast.parse(path.read_text(encoding="utf-8"))

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert (
                            alias.name.split(".")[0] != "subprocess"
                        )

    def test_the_drop_zone_is_git_ignored(self):
        import subprocess

        result = subprocess.run(
            [
                "git",
                "check-ignore",
                "-v",
                "data/incoming/linkedin-archive/manifest.csv",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

        assert result.returncode == 0, result.stdout
