"""
Tests for Saved Items ingestion.

A saved list is the least reliable input the pipeline accepts: it is
whatever tool the user happened to export it with, and it describes
links rather than content. These cover what the reader tolerates, what
it refuses, and the rule that matters most — a link with nothing behind
it stays a link.

Every fixture is synthetic. No credential appears in any of them, and
none is needed to run any of these tests.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.ingestion.saved_items.bundles import (
    BundleError,
    CapturedContent,
    association_for,
    content_digest,
    discover_bundles,
    read_bundle,
)
from src.ingestion.saved_items.manifest import (
    MANIFEST_FILE,
    ManifestUnreadable,
    SavedItemsManifest,
    manifest_path,
)
from src.ingestion.saved_items.model import (
    CaptureMethod,
    SavedItem,
    SavedItemState,
)
from src.ingestion.saved_items.readers import (
    ManifestError,
    read_manifest,
)
from src.ingestion.saved_items.urls import (
    SavedItemUrlError,
    normalize_linkedin_url,
    source_id_for,
)


POST_A = "https://www.linkedin.com/posts/alice_delta-lake-101"
POST_B = "https://www.linkedin.com/posts/bob_kafka-partitioning-202"
ARTICLE = "https://www.linkedin.com/pulse/carol-spark-skew-303"
FEED = "https://www.linkedin.com/feed/update/urn:li:activity:7285158797206056960/"

# Credential-shaped fixtures are assembled at runtime from a prefix and a
# filler alphabet, for the reason the rest of the suite does it: writing
# complete literals here would make GitHub push protection treat this
# file as containing real credentials and block the push.
_FILLER = "abcdefghijklmnopqrstuvwxyz0123456789"

FAKE_PASSWORD = "pw" + _FILLER
FAKE_TOKEN = "urn" + ":" + "li" + ":" + "share" + ":" + _FILLER
FAKE_COOKIE = "sessionid" + "=" + _FILLER
FAKE_API_KEY = "sk" + "-live-" + _FILLER


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


def png_bytes(seed: int = 0) -> bytes:
    """A minimal but real PNG, so image handling has a true file."""
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            len(data).to_bytes(4, "big")
            + tag
            + data
            + zlib.crc32(tag + data).to_bytes(4, "big")
        )

    body = bytes([0x00, 0xFF, 0x00, seed & 0xFF])

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(
            b"IHDR",
            (1).to_bytes(4, "big")
            + (1).to_bytes(4, "big")
            + bytes([8, 2, 0, 0, 0]),
        )
        + chunk(b"IDAT", zlib.compress(body))
        + chunk(b"IEND", b"")
    )


def pdf_bytes(body: str = "Delta Lake interview notes") -> bytes:
    """A minimal single-page PDF with extractable text."""
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

    xref_at = len(out)

    out += b"xref\n0 " + str(len(objects) + 1).encode() + b"\n"
    out += b"0000000000 65535 f \n"

    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()

    out += (
        b"trailer\n<< /Size "
        + str(len(objects) + 1).encode()
        + b" /Root 1 0 R >>\nstartxref\n"
        + str(xref_at).encode()
        + b"\n%%EOF\n"
    )

    return bytes(out)


def item_for(url: str) -> SavedItem:
    return SavedItem.from_url(normalize_linkedin_url(url))


def write_csv(path: Path, rows: list[str], header: str = "URL") -> Path:
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")

    return path


# ---------------------------------------------------------------------
# URL normalization
# ---------------------------------------------------------------------


class TestUrlNormalization:
    def test_the_same_url_always_gives_the_same_id(self):
        first = normalize_linkedin_url(POST_A)
        second = normalize_linkedin_url(POST_A)

        assert first.source_id == second.source_id
        assert first.canonical == second.canonical

    def test_a_trailing_slash_is_the_same_post(self):
        with_slash = normalize_linkedin_url(POST_A + "/")
        without = normalize_linkedin_url(POST_A)

        assert with_slash.source_id == without.source_id

    def test_a_fragment_is_the_same_post(self):
        # A fragment is where a comment sits inside a post, so it never
        # identifies the post itself.
        with_fragment = normalize_linkedin_url(POST_A + "#comment-99")
        plain = normalize_linkedin_url(POST_A)

        assert with_fragment.source_id == plain.source_id

    @pytest.mark.parametrize(
        "tracking",
        ["?trk=abc", "?utm_source=newsletter", "?lipi=urn%3Ali%3Athing"],
    )
    def test_tracking_parameters_are_dropped(self, tracking: str):
        assert (
            normalize_linkedin_url(POST_A + tracking).source_id
            == normalize_linkedin_url(POST_A).source_id
        )

    def test_a_meaningful_query_parameter_is_kept(self):
        # An unrecognised parameter may be the only thing telling two
        # posts apart, so it is not removed.
        first = normalize_linkedin_url(POST_A + "?activity=1")
        second = normalize_linkedin_url(POST_A + "?activity=2")

        assert first.source_id != second.source_id

    def test_surrounding_whitespace_is_ignored(self):
        padded = normalize_linkedin_url(f"  \t{POST_A}  \n")

        assert padded.source_id == normalize_linkedin_url(POST_A).source_id

    def test_angle_brackets_are_ignored(self):
        wrapped = normalize_linkedin_url(f"<{POST_A}>")

        assert wrapped.source_id == normalize_linkedin_url(POST_A).source_id

    def test_equivalent_encodings_agree(self):
        # A slug written with an escape for an unreserved character is
        # the same slug as one written plainly.
        escaped = normalize_linkedin_url(
            "https://www.linkedin.com/posts/alice_sp%61rk-skew-303"
        )
        plain = normalize_linkedin_url(
            "https://www.linkedin.com/posts/alice_spark-skew-303"
        )

        assert escaped.source_id == plain.source_id

    def test_a_duplicate_separator_is_collapsed(self):
        doubled = normalize_linkedin_url(POST_A.replace("/posts/", "//posts//"))

        assert doubled.source_id == normalize_linkedin_url(POST_A).source_id

    def test_different_posts_keep_different_ids(self):
        assert (
            normalize_linkedin_url(POST_A).source_id
            != normalize_linkedin_url(POST_B).source_id
        )

    def test_a_post_and_an_article_are_different(self):
        assert (
            normalize_linkedin_url(POST_A).source_id
            != normalize_linkedin_url(ARTICLE).source_id
        )

    def test_a_feed_update_is_recognised_as_content(self):
        result = normalize_linkedin_url(FEED)

        assert result.kind == "post"
        assert result.identifier == "urn:li:activity:7285158797206056960"

    def test_a_profile_is_kept_but_not_called_a_post(self):
        result = normalize_linkedin_url("https://www.linkedin.com/in/someone/")

        assert result.kind == "linkedin_other"
        assert result.is_linkedin is True

    def test_an_external_link_is_kept_and_labelled(self):
        result = normalize_linkedin_url("https://example.com/article")

        assert result.kind == "external"
        assert result.is_linkedin is False

    def test_an_external_item_is_labelled_as_external(self):
        item = SavedItem.from_url(
            normalize_linkedin_url("https://example.com/article")
        )

        assert item.is_external is True
        assert item.is_linkedin_content is False

    def test_a_subdomain_is_the_same_site(self):
        result = normalize_linkedin_url(
            "https://uk.linkedin.com/posts/alice_delta-lake-101"
        )

        assert result.is_linkedin is True

    def test_the_original_url_is_preserved(self):
        messy = f"  {POST_A}/?trk=x#comment-1  "

        assert normalize_linkedin_url(messy).original == messy

    def test_the_source_id_is_derived_from_the_canonical_url(self):
        result = normalize_linkedin_url(POST_A + "/?trk=x")

        assert result.source_id == source_id_for(result.canonical)

    def test_a_host_case_difference_is_the_same_url(self):
        assert (
            normalize_linkedin_url(
                "https://WWW.LinkedIn.COM/posts/alice_delta-lake-101"
            ).source_id
            == normalize_linkedin_url(POST_A).source_id
        )


class TestUrlValidation:
    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "   ",
            "not a url",
            "example.com/no-scheme",
            "javascript:alert(1)",
            "file:///etc/passwd",
            "ftp://linkedin.com/posts/x",
        ],
    )
    def test_an_unusable_url_is_rejected(self, bad: str):
        with pytest.raises(SavedItemUrlError):
            normalize_linkedin_url(bad)

    def test_a_rejection_names_the_problem(self):
        with pytest.raises(SavedItemUrlError) as caught:
            normalize_linkedin_url("javascript:alert(1)")

        assert "scheme" in str(caught.value)

    def test_a_bare_host_is_made_absolute(self):
        result = normalize_linkedin_url("www.linkedin.com/posts/alice_delta-lake-101")

        assert result.canonical.startswith("https://")

    def test_a_protocol_relative_url_is_accepted(self):
        result = normalize_linkedin_url("//www.linkedin.com/posts/alice_delta-lake-101")

        assert result.canonical.startswith("https://")

    def test_the_error_does_not_echo_a_whole_url(self):
        long_url = POST_A + "?" + "x=1&" * 200

        with pytest.raises(SavedItemUrlError) as caught:
            normalize_linkedin_url("ftp://" + long_url)

        assert len(str(caught.value)) < 120


# ---------------------------------------------------------------------
# Manifest readers
# ---------------------------------------------------------------------


class TestReaders:
    def test_csv_columns_are_matched_by_name(self, tmp_path: Path):
        path = write_csv(
            tmp_path / "saved.csv",
            [f"{POST_A},2026-01-02,Delta Lake,Alice,schema evolution"],
            header="URL,Saved Date,Title,Author,Notes",
        )

        read = read_manifest(path)

        assert len(read.items) == 1

        item = read.items[0]

        assert item.saved_date == "2026-01-02"
        assert item.title == "Delta Lake"
        assert item.author == "Alice"
        assert item.notes == "schema evolution"

    @pytest.mark.parametrize(
        "header",
        [
            "url,Saved Date",
            "URL,saved date",
            "link,date saved",
            "LinkedIn URL,Saved On",
            "Permalink,Date",
            "Post URL,Created At",
        ],
    )
    def test_reasonable_header_variations_all_work(
        self, tmp_path: Path, header: str
    ):
        name, date_header = header.split(",")

        path = write_csv(
            tmp_path / "saved.csv", [f"{POST_A},2026-01-02"], header=header
        )

        read = read_manifest(path)

        assert read.items[0].source_id == normalize_linkedin_url(POST_A).source_id
        assert read.items[0].saved_date == "2026-01-02"
        assert name and date_header

    def test_tsv_is_read(self, tmp_path: Path):
        path = tmp_path / "saved.tsv"

        path.write_text(
            "url\tsaved date\ttitle\n"
            f"{POST_A}\t2026-03-04\tKafka partitioning\n",
            encoding="utf-8",
        )

        read = read_manifest(path)

        assert read.source_format == "tsv"
        assert read.items[0].title == "Kafka partitioning"

    def test_txt_is_read_one_link_per_line(self, tmp_path: Path):
        path = tmp_path / "saved.txt"

        path.write_text(
            f"# my saved list\n"
            f"{POST_A} - read this later\n"
            f"{POST_B}.\n"
            f"a line with no link at all\n",
            encoding="utf-8",
        )

        read = read_manifest(path)

        assert read.source_format == "txt"
        assert len(read.items) == 2
        assert read.items[0].notes == "read this later"
        assert any("no URL" in str(issue) for issue in read.issues)

    def test_a_bare_host_in_a_plain_list_is_reported_not_guessed(
        self, tmp_path: Path
    ):
        # A plain list is read as a list of full links. Treating any
        # bare token as a URL would be guessing at what the user meant,
        # so the line is reported and skipped instead.
        path = tmp_path / "saved.txt"

        path.write_text(
            "www.linkedin.com/posts/alice_delta-lake-101\n", encoding="utf-8"
        )

        read = read_manifest(path)

        assert not read.items
        assert any("no URL" in str(issue) for issue in read.issues)

    def test_a_bare_host_is_still_accepted_in_a_column(self, tmp_path: Path):
        # A CSV cell naming a host is a link, because a column exists to
        # hold one, so the missing scheme is filled in.
        path = tmp_path / "saved.csv"

        path.write_text(
            "URL\nwww.linkedin.com/posts/alice_delta-lake-101\n",
            encoding="utf-8",
        )

        read = read_manifest(path)

        assert len(read.items) == 1
        assert read.items[0].canonical_url.startswith("https://")

    def test_json_list_is_read(self, tmp_path: Path):
        path = tmp_path / "saved.json"

        path.write_text(
            json.dumps(
                [
                    {
                        "url": POST_A,
                        "savedDate": "2026-01-02",
                        "title": "Delta Lake",
                    }
                ]
            ),
            encoding="utf-8",
        )

        read = read_manifest(path)

        assert read.items[0].saved_date == "2026-01-02"
        assert read.items[0].title == "Delta Lake"

    def test_a_wrapped_json_object_is_read(self, tmp_path: Path):
        path = tmp_path / "saved.json"

        path.write_text(
            json.dumps({"items": [{"url": POST_A}]}), encoding="utf-8"
        )

        assert len(read_manifest(path).items) == 1

    def test_jsonl_is_read_one_record_per_line(self, tmp_path: Path):
        path = tmp_path / "saved.jsonl"

        path.write_text(
            "\n".join(
                [
                    json.dumps({"url": POST_A, "saved_date": "2026-01-02"}),
                    "",
                    json.dumps({"url": POST_B}),
                ]
            ),
            encoding="utf-8",
        )

        read = read_manifest(path)

        assert read.source_format == "jsonl"
        assert len(read.items) == 2

    def test_a_broken_jsonl_line_is_reported_not_fatal(self, tmp_path: Path):
        path = tmp_path / "saved.jsonl"

        path.write_text(
            "\n".join(
                [
                    json.dumps({"url": POST_A}),
                    "{ not json",
                    json.dumps({"url": POST_B}),
                ]
            ),
            encoding="utf-8",
        )

        read = read_manifest(path)

        assert len(read.items) == 2
        assert any("not valid JSON" in str(issue) for issue in read.issues)

    def test_invalid_json_is_an_error_naming_the_file(self, tmp_path: Path):
        path = tmp_path / "saved.json"

        path.write_text("{ not json", encoding="utf-8")

        with pytest.raises(ManifestError) as caught:
            read_manifest(path)

        assert "saved.json" in str(caught.value)

    def test_a_missing_file_is_an_error(self, tmp_path: Path):
        with pytest.raises(ManifestError):
            read_manifest(tmp_path / "absent.csv")

    def test_a_row_without_a_url_is_reported(self, tmp_path: Path):
        path = write_csv(
            tmp_path / "saved.csv",
            [f"{POST_A},2026-01-02", ",2026-01-03"],
            header="URL,Saved Date",
        )

        read = read_manifest(path)

        assert len(read.items) == 1
        assert any("no URL" in str(issue) for issue in read.issues)

    def test_an_invalid_url_is_reported_with_its_line(self, tmp_path: Path):
        path = write_csv(
            tmp_path / "saved.csv",
            [f"{POST_A},2026-01-02", "not-a-url,2026-01-03"],
            header="URL,Saved Date",
        )

        read = read_manifest(path)

        assert len(read.items) == 1
        assert any("line 3" in str(issue) for issue in read.issues)

    def test_a_file_with_no_url_column_is_reported(self, tmp_path: Path):
        path = tmp_path / "saved.csv"

        path.write_text("Title,Author\nDelta Lake,Alice\n", encoding="utf-8")

        read = read_manifest(path)

        assert not read.items
        assert any("no URL column" in str(issue) for issue in read.issues)

    def test_a_byte_order_mark_does_not_hide_the_first_column(
        self, tmp_path: Path
    ):
        path = tmp_path / "saved.csv"

        path.write_text(
            "﻿URL,Saved Date\n" f"{POST_A},2026-01-02\n",
            encoding="utf-8",
        )

        read = read_manifest(path)

        assert len(read.items) == 1

    def test_duplicate_urls_become_several_items_for_dedup_to_handle(
        self, tmp_path: Path
    ):
        path = write_csv(
            tmp_path / "saved.csv",
            [
                f"{POST_A},2026-01-02",
                f"{POST_A}/?trk=x,2026-01-02",
                POST_B,
            ],
            header="URL,Saved Date",
        )

        read = read_manifest(path)

        assert len(read.items) == 3

        # Two of them normalize to one id, which is what dedup keys on.
        ids = {item.source_id for item in read.items}

        assert len(ids) == 2

    def test_a_metadata_only_item_is_kept_as_metadata(self, tmp_path: Path):
        path = write_csv(
            tmp_path / "saved.csv",
            [f"{POST_B},2026-01-02"],
            header="URL,Saved Date",
        )

        item = read_manifest(path).items[0]

        assert item.is_metadata_only is True
        assert item.has_content is False
        assert item.state is SavedItemState.PENDING

    def test_a_bundle_column_names_the_capture(self, tmp_path: Path):
        path = tmp_path / "saved.csv"

        path.write_text(
            f"URL,Bundle\n{POST_A},urn-li-saved-abc123\n", encoding="utf-8"
        )

        assert read_manifest(path).items[0].bundle == "urn-li-saved-abc123"

    def test_a_camel_case_json_key_is_understood(self, tmp_path: Path):
        path = tmp_path / "saved.json"

        path.write_text(
            json.dumps({"linkedInUrl": POST_A, "dateSaved": "2026-05-06"}),
            encoding="utf-8",
        )

        item = read_manifest(path).items[0]

        assert item.source_id == normalize_linkedin_url(POST_A).source_id
        assert item.saved_date == "2026-05-06"


class TestCredentialExclusion:
    @pytest.mark.parametrize(
        "header,value",
        [
            ("Password", FAKE_PASSWORD),
            ("Token", FAKE_TOKEN),
            ("Cookie", FAKE_COOKIE),
            ("access token", FAKE_API_KEY),
            ("API Key", FAKE_API_KEY),
            ("storage_state", "{}"),
        ],
    )
    def test_a_credential_column_is_never_read(
        self, tmp_path: Path, header: str, value: str
    ):
        path = tmp_path / "saved.csv"

        path.write_text(
            f"URL,Notes,{header}\n{POST_A},a real note,{value}\n",
            encoding="utf-8",
        )

        read = read_manifest(path)

        item = read.items[0]

        assert item.notes == "a real note"
        assert value not in json.dumps(item.as_dict())
        assert any("credential" in str(issue) for issue in read.issues)

    def test_a_credential_field_in_json_is_not_read(self, tmp_path: Path):
        path = tmp_path / "saved.json"

        path.write_text(
            json.dumps(
                {
                    "url": POST_A,
                    "title": "Delta Lake",
                    "password": FAKE_PASSWORD,
                }
            ),
            encoding="utf-8",
        )

        item = read_manifest(path).items[0]

        assert FAKE_PASSWORD not in json.dumps(item.as_dict())
        assert item.title == "Delta Lake"

    def test_no_saved_item_field_could_hold_a_credential(self):
        stored = item_for(POST_A).as_dict()

        for name in stored:
            assert not any(
                word in name
                for word in ("password", "token", "cookie", "secret", "session")
            )


# ---------------------------------------------------------------------
# The item
# ---------------------------------------------------------------------


class TestSavedItem:
    def test_a_fresh_item_is_pending(self):
        item = item_for(POST_A)

        assert item.state is SavedItemState.PENDING
        assert item.post_id is None
        assert item.created_at
        assert item.updated_at

    def test_merging_keeps_the_stored_metadata(self):
        stored = item_for(POST_A)
        stored.saved_date = "2026-01-02"
        stored.title = "Delta Lake"

        fresh = item_for(POST_A)
        fresh.title = "A different title from a later export"
        fresh.notes = "a later note"

        stored.merge(fresh)

        assert stored.title == "Delta Lake"
        assert stored.notes == "a later note"

    def test_merging_a_new_date_into_an_empty_one_fills_it(self):
        stored = item_for(POST_A)
        fresh = item_for(POST_A)
        fresh.saved_date = "2026-02-03"

        stored.merge(fresh)

        assert stored.saved_date == "2026-02-03"

    def test_changed_content_invalidates_the_capture(self):
        stored = item_for(POST_A)
        stored.content_digest = "old"
        stored.state = SavedItemState.IMPORTED

        fresh = item_for(POST_A)
        fresh.content_digest = "new"
        fresh.state = SavedItemState.CAPTURED

        stored.merge(fresh)

        assert stored.content_digest == "new"
        assert stored.state is SavedItemState.CAPTURED

    def test_a_round_trip_preserves_the_record(self):
        item = item_for(POST_A)
        item.saved_date = "2026-01-02"
        item.title = "Delta Lake"
        item.content_digest = "abc"
        item.state = SavedItemState.CAPTURED
        item.capture_method = CaptureMethod.USER_PROVIDED

        restored = SavedItem.from_dict(item.as_dict())

        assert restored.as_dict() == item.as_dict()

    def test_an_unknown_state_is_read_as_pending(self):
        # Acting on a state this version does not understand would be
        # worse than redoing the work.
        restored = SavedItem.from_dict(
            {
                "source_id": "urn:li:saved:abc",
                "state": "some_future_state",
            }
        )

        assert restored.state is SavedItemState.PENDING

    def test_states_are_ordered_by_how_far_they_got(self):
        assert (
            SavedItemState.IMPORTED.rank > SavedItemState.CAPTURED.rank
            > SavedItemState.PENDING.rank
        )

    def test_only_enriched_and_failed_are_terminal(self):
        assert SavedItemState.ENRICHED.is_terminal
        assert SavedItemState.FAILED.is_terminal
        assert not SavedItemState.PENDING.is_terminal
        assert not SavedItemState.CAPTURED.is_terminal
        assert not SavedItemState.IMPORTED.is_terminal


# ---------------------------------------------------------------------
# Bundles
# ---------------------------------------------------------------------


class TestBundles:
    def test_a_markdown_bundle_is_read(self, tmp_path: Path):
        bundle = tmp_path / item_for(POST_A).source_id.replace(":", "-")
        bundle.mkdir()

        (bundle / "content.md").write_text(
            "# Delta Lake\n\nACID gives schema evolution.\n", encoding="utf-8"
        )

        content = read_bundle(bundle, root=tmp_path)

        assert "ACID gives schema evolution." in content.text
        assert content.has_text is True

    def test_a_pdf_bundle_yields_its_text_and_keeps_the_file(
        self, tmp_path: Path
    ):
        bundle = tmp_path / "notes"
        bundle.mkdir()

        (bundle / "document.pdf").write_bytes(pdf_bytes("Consumer lag explained"))

        content = read_bundle(bundle, root=tmp_path)

        assert "Consumer lag explained" in content.text
        assert [path.name for path in content.media] == ["document.pdf"]

    def test_an_image_is_kept_and_contributes_no_invented_text(
        self, tmp_path: Path
    ):
        bundle = tmp_path / "shot"
        bundle.mkdir()

        (bundle / "screenshot.png").write_bytes(png_bytes())

        content = read_bundle(bundle, root=tmp_path)

        assert [path.name for path in content.media] == ["screenshot.png"]
        assert content.text == ""
        # The absence of text is stated, not glossed over.
        assert any("no OCR" in note for note in content.notes)

    def test_an_html_page_yields_its_declared_text(self, tmp_path: Path):
        bundle = tmp_path / "page"
        bundle.mkdir()

        (bundle / "page.html").write_text(
            "<html><head>"
            '<meta property="og:description" content="A long tail join is '
            'where skew shows up, and salting the key is the fix.">'
            "</head><body><nav>Home Jobs Sign in</nav></body></html>",
            encoding="utf-8",
        )

        content = read_bundle(bundle, root=tmp_path)

        assert content.text.startswith("A long tail join")
        # The navigation chrome is not mistaken for the post.
        assert "Sign in" not in content.text

    def test_html_metadata_is_preserved(self, tmp_path: Path):
        bundle = tmp_path / "page"
        bundle.mkdir()

        (bundle / "page.html").write_text(
            "<html><head>"
            '<meta property="og:title" content="Spark skew joins">'
            '<meta name="author" content="Carol">'
            '<meta property="article:published_time" '
            'content="2026-02-14T09:00:00Z">'
            '<meta property="og:description" content="Salting the key on '
            'the hot dimension is the standard fix for skew.">'
            "</head><body></body></html>",
            encoding="utf-8",
        )

        content = read_bundle(bundle, root=tmp_path)

        assert content.title == "Spark skew joins"
        assert content.author == "Carol"
        assert content.published_at == "2026-02-14T09:00:00Z"

    def test_a_page_with_nothing_in_it_is_reported_not_invented(
        self, tmp_path: Path
    ):
        bundle = tmp_path / "blank"
        bundle.mkdir()

        (bundle / "page.html").write_text(
            "<html><head></head><body></body></html>", encoding="utf-8"
        )

        content = read_bundle(bundle, root=tmp_path)

        assert content.text == ""
        assert any(
            "nothing was fetched" in note for note in content.notes
        )

    def test_a_page_never_produces_engagement_counts(self, tmp_path: Path):
        bundle = tmp_path / "page"
        bundle.mkdir()

        (bundle / "page.html").write_text(
            '<html><body><div class="engagement">1,234 reactions</div>'
            "<p>Some actual post text that is long enough to count.</p>"
            "</body></html>",
            encoding="utf-8",
        )

        content = read_bundle(bundle, root=tmp_path)

        # The page's own text is kept exactly as written, because it is
        # the user's material and rewriting it would change what they
        # saved. What must not happen is a count being lifted out of it
        # and recorded as a fact about the post, so no field of the
        # capture claims to hold one.
        for name in CapturedContent.__dataclass_fields__:
            for word in ("reaction", "engagement", "like", "view", "count"):
                assert word not in name

        assert content.title is None
        assert content.author is None
        assert content.published_at is None

    def test_a_corrupt_pdf_is_reported_and_produces_nothing(
        self, tmp_path: Path
    ):
        bundle = tmp_path / "broken"
        bundle.mkdir()

        (bundle / "document.pdf").write_bytes(b"%PDF-1.4\nnot a pdf body")

        content = read_bundle(bundle, root=tmp_path)

        assert content.text == ""
        assert content.media == []
        assert any("document.pdf" in note for note in content.unreadable)

    def test_a_corrupt_image_is_reported(self, tmp_path: Path):
        bundle = tmp_path / "broken"
        bundle.mkdir()

        (bundle / "shot.png").write_bytes(b"NOT AN IMAGE AT ALL")

        content = read_bundle(bundle, root=tmp_path)

        assert content.media == []
        assert any("shot.png" in note for note in content.unreadable)

    def test_an_empty_file_is_reported(self, tmp_path: Path):
        bundle = tmp_path / "empty"
        bundle.mkdir()

        (bundle / "shot.png").write_bytes(b"")

        content = read_bundle(bundle, root=tmp_path)

        assert any("empty" in note for note in content.unreadable)

    def test_a_capture_failure_does_not_leak_a_local_path(
        self, tmp_path: Path
    ):
        bundle = tmp_path / "broken"
        bundle.mkdir()

        (bundle / "document.pdf").write_bytes(b"%PDF-1.4\nnot a pdf body")

        content = read_bundle(bundle, root=tmp_path)

        joined = " ".join(content.unreadable)

        assert str(tmp_path) not in joined
        assert os.path.expanduser("~") not in joined

    def test_an_empty_bundle_digests_to_nothing(self, tmp_path: Path):
        bundle = tmp_path / "nothing"
        bundle.mkdir()

        (bundle / "page.html").write_text("<html></html>", encoding="utf-8")

        # A truthy digest is what marks an item as having content, so a
        # bundle of unreadable files must not count as a capture.
        assert content_digest(read_bundle(bundle, root=tmp_path)) == ""

    def test_a_real_capture_digests_to_something(self, tmp_path: Path):
        bundle = tmp_path / "real"
        bundle.mkdir()

        (bundle / "content.md").write_text("something", encoding="utf-8")

        assert content_digest(read_bundle(bundle, root=tmp_path))


class TestContentDigest:
    def test_unchanged_content_digests_the_same(self, tmp_path: Path):
        bundle = tmp_path / "b"
        bundle.mkdir()
        (bundle / "content.md").write_text("stable text", encoding="utf-8")

        first = content_digest(read_bundle(bundle, root=tmp_path))
        second = content_digest(read_bundle(bundle, root=tmp_path))

        assert first == second

    def test_changed_text_digests_differently(self, tmp_path: Path):
        bundle = tmp_path / "b"
        bundle.mkdir()

        (bundle / "content.md").write_text("before", encoding="utf-8")
        before = content_digest(read_bundle(bundle, root=tmp_path))

        (bundle / "content.md").write_text("after", encoding="utf-8")
        after = content_digest(read_bundle(bundle, root=tmp_path))

        assert before != after

    def test_a_replaced_image_is_a_change(self, tmp_path: Path):
        # Hashed by content, not by file name, or a swapped screenshot
        # would read as unchanged and the stale analysis would survive.
        bundle = tmp_path / "b"
        bundle.mkdir()

        (bundle / "screenshot.png").write_bytes(png_bytes(seed=1))
        before = content_digest(read_bundle(bundle, root=tmp_path))

        (bundle / "screenshot.png").write_bytes(png_bytes(seed=2))
        after = content_digest(read_bundle(bundle, root=tmp_path))

        assert before != after


class TestBundleContainment:
    def test_a_parent_traversal_is_refused(self, tmp_path: Path):
        root = tmp_path / "drop"
        root.mkdir()

        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "leak.md").write_text("private", encoding="utf-8")

        with pytest.raises(BundleError) as caught:
            read_bundle("../outside", root=root)

        assert "escapes" in str(caught.value)

    def test_an_absolute_path_outside_the_root_is_refused(
        self, tmp_path: Path
    ):
        root = tmp_path / "drop"
        root.mkdir()

        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "leak.md").write_text("private", encoding="utf-8")

        with pytest.raises(BundleError):
            read_bundle(outside, root=root)

    def test_a_windows_style_traversal_is_refused(self, tmp_path: Path):
        root = tmp_path / "drop"
        root.mkdir()

        (tmp_path / "outside").mkdir()

        with pytest.raises(BundleError):
            read_bundle("..\\outside", root=root)

    def test_a_symlinked_bundle_is_refused(self, tmp_path: Path):
        root = tmp_path / "drop"
        root.mkdir()

        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "leak.md").write_text("private", encoding="utf-8")

        link = root / "escape"

        try:
            link.symlink_to(outside.resolve(), target_is_directory=True)
        except OSError as exc:  # pragma: no cover - needs privileges
            pytest.skip(f"symlinks unavailable here: {exc}")

        with pytest.raises(BundleError) as caught:
            read_bundle("escape", root=root)

        assert "escapes" in str(caught.value)

    def test_a_symlinked_file_inside_a_bundle_is_not_read(
        self, tmp_path: Path
    ):
        root = tmp_path / "drop"
        root.mkdir()

        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "leak.md").write_text("private material", encoding="utf-8")

        bundle = root / "legitimate"
        bundle.mkdir()
        (bundle / "content.md").write_text("real content", encoding="utf-8")

        try:
            (bundle / "escape.md").symlink_to((outside / "leak.md").resolve())
        except OSError as exc:  # pragma: no cover - needs privileges
            pytest.skip(f"symlinks unavailable here: {exc}")

        content = read_bundle("legitimate", root=root)

        assert "real content" in content.text
        assert "private material" not in content.text
        assert any("outside" in note for note in content.unreadable)

    def test_a_bundle_inside_the_root_is_read(self, tmp_path: Path):
        root = tmp_path / "drop"
        root.mkdir()

        nested = root / "one" / "two"
        nested.mkdir(parents=True)
        (nested / "content.md").write_text("deep", encoding="utf-8")

        content = read_bundle("one/two", root=root)

        assert content.text == "deep"


class TestBundleDiscovery:
    def test_directories_are_bundles(self, tmp_path: Path):
        (tmp_path / "one").mkdir()
        (tmp_path / "one" / "content.md").write_text("a", encoding="utf-8")

        (tmp_path / "two").mkdir()
        (tmp_path / "two" / "shot.png").write_bytes(png_bytes())

        assert {path.name for path in discover_bundles(tmp_path)} == {
            "one",
            "two",
        }

    def test_a_loose_file_is_a_bundle(self, tmp_path: Path):
        (tmp_path / "page.html").write_text("<html></html>", encoding="utf-8")

        assert [path.name for path in discover_bundles(tmp_path)] == [
            "page.html"
        ]

    def test_a_list_file_is_not_a_capture(self, tmp_path: Path):
        (tmp_path / "manifest.csv").write_text("URL\n" + POST_A, encoding="utf-8")

        assert discover_bundles(tmp_path) == []

    def test_an_empty_directory_is_not_a_bundle(self, tmp_path: Path):
        (tmp_path / "empty").mkdir()

        assert discover_bundles(tmp_path) == []

    def test_a_missing_root_is_an_error(self, tmp_path: Path):
        with pytest.raises(BundleError):
            discover_bundles(tmp_path / "absent")

    def test_discovered_paths_resolve_against_the_root(self, tmp_path: Path):
        (tmp_path / "one").mkdir()
        (tmp_path / "one" / "content.md").write_text("a", encoding="utf-8")

        found = discover_bundles(tmp_path)

        assert read_bundle(found[0], root=tmp_path).text == "a"


class TestBundleAssociation:
    def test_a_directory_named_after_the_source_id_claims_the_item(
        self, tmp_path: Path
    ):
        item = item_for(POST_A)
        name = item.source_id.replace(":", "-")

        (tmp_path / name).mkdir()

        association = association_for(name, root=tmp_path)

        assert name in association.claim_keys

    def test_a_url_inside_a_capture_claims_the_item(self, tmp_path: Path):
        (tmp_path / "page").mkdir()

        (tmp_path / "page" / "capture.json").write_text(
            json.dumps({"url": POST_A + "/?trk=x#comment-1"}), encoding="utf-8"
        )

        association = association_for("page", root=tmp_path)

        assert association.url == normalize_linkedin_url(POST_A).canonical

    def test_a_source_id_inside_a_capture_claims_the_item(
        self, tmp_path: Path
    ):
        item = item_for(POST_A)

        (tmp_path / "anything").mkdir()

        (tmp_path / "anything" / "capture.json").write_text(
            json.dumps({"source_id": item.source_id}), encoding="utf-8"
        )

        association = association_for("anything", root=tmp_path)

        assert association.source_id == item.source_id

    def test_a_bundle_claiming_nothing(self, tmp_path: Path):
        (tmp_path / "unrelated").mkdir()
        (tmp_path / "unrelated" / "content.md").write_text("x", encoding="utf-8")

        association = association_for("unrelated", root=tmp_path)

        assert association.url is None
        assert association.source_id is None

    def test_an_unusable_url_is_not_a_claim(self, tmp_path: Path):
        (tmp_path / "page").mkdir()

        (tmp_path / "page" / "capture.json").write_text(
            json.dumps({"url": "javascript:alert(1)"}), encoding="utf-8"
        )

        association = association_for("page", root=tmp_path)

        assert association.url is None
        assert association.reason

    def test_a_capture_may_carry_the_text_itself(self, tmp_path: Path):
        (tmp_path / "b").mkdir()

        (tmp_path / "b" / "capture.json").write_text(
            json.dumps(
                {
                    "url": POST_A,
                    "text": "The captured body of the post.",
                    "title": "Delta Lake",
                }
            ),
            encoding="utf-8",
        )

        content = read_bundle("b", root=tmp_path)

        assert content.text == "The captured body of the post."
        assert content.title == "Delta Lake"


# ---------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------


class TestManifest:
    def test_a_new_item_is_reported_as_new(self, tmp_path: Path):
        manifest = SavedItemsManifest(manifest_path(tmp_path))

        assert manifest.upsert(item_for(POST_A)) == "new"

    def test_the_same_item_twice_is_a_duplicate(self, tmp_path: Path):
        manifest = SavedItemsManifest(manifest_path(tmp_path))

        manifest.upsert(item_for(POST_A))
        manifest.upsert(item_for(POST_A + "/?trk=x"))

        assert len(manifest) == 1

    def test_different_items_stay_separate(self, tmp_path: Path):
        manifest = SavedItemsManifest(manifest_path(tmp_path))

        manifest.upsert(item_for(POST_A))
        manifest.upsert(item_for(POST_B))

        assert len(manifest) == 2

    def test_the_manifest_survives_a_reload(self, tmp_path: Path):
        path = manifest_path(tmp_path)

        manifest = SavedItemsManifest(path)
        manifest.upsert(item_for(POST_A))
        manifest.save()

        assert len(SavedItemsManifest.load(path)) == 1

    def test_saving_leaves_no_temporary_file_behind(self, tmp_path: Path):
        path = manifest_path(tmp_path)

        manifest = SavedItemsManifest(path)
        manifest.upsert(item_for(POST_A))
        manifest.save()

        assert sorted(p.name for p in tmp_path.iterdir()) == [MANIFEST_FILE]

    def test_a_half_written_manifest_is_never_observable(
        self, tmp_path: Path
    ):
        # The target is replaced by a move, so a reader sees either the
        # old file or the new one and never a partial one.
        path = manifest_path(tmp_path)

        manifest = SavedItemsManifest(path)
        manifest.upsert(item_for(POST_A))
        manifest.save()

        before = path.read_text(encoding="utf-8")

        manifest.upsert(item_for(POST_B))
        manifest.save()

        assert path.read_text(encoding="utf-8") != before
        assert json.loads(path.read_text(encoding="utf-8"))

    def test_an_unreadable_manifest_is_a_stop_naming_the_cause(
        self, tmp_path: Path
    ):
        path = manifest_path(tmp_path)

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{ not json", encoding="utf-8")

        with pytest.raises(ManifestUnreadable) as caught:
            SavedItemsManifest.load(path)

        assert MANIFEST_FILE in str(caught.value)

    def test_a_manifest_from_another_version_is_refused(
        self, tmp_path: Path
    ):
        path = manifest_path(tmp_path)

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"manifest_version": 999, "items": []}), encoding="utf-8"
        )

        with pytest.raises(ManifestUnreadable) as caught:
            SavedItemsManifest.load(path)

        assert "version" in str(caught.value)

    def test_the_manifest_holds_no_credential(self, tmp_path: Path):
        path = manifest_path(tmp_path)

        manifest = SavedItemsManifest(path)
        manifest.upsert(item_for(POST_A))
        manifest.save()

        assert "password" not in path.read_text(encoding="utf-8").lower()

    def test_marking_imported_records_the_post(self, tmp_path: Path):
        manifest = SavedItemsManifest(manifest_path(tmp_path))

        item = item_for(POST_A)
        manifest.upsert(item)
        manifest.mark_imported(item.source_id, "urn-li-saved-abc")

        stored = manifest.get(item.source_id)

        assert stored.post_id == "urn-li-saved-abc"
        assert stored.state is SavedItemState.IMPORTED

    def test_marking_enriched_advances_the_state(self, tmp_path: Path):
        manifest = SavedItemsManifest(manifest_path(tmp_path))

        item = item_for(POST_A)
        manifest.upsert(item)
        manifest.mark_enriched(item.source_id, "urn-li-saved-abc")

        assert manifest.get(item.source_id).state is SavedItemState.ENRICHED

    def test_a_failure_reason_is_recorded(self, tmp_path: Path):
        manifest = SavedItemsManifest(manifest_path(tmp_path))

        item = item_for(POST_A)
        manifest.upsert(item)
        manifest.mark_failed(item.source_id, "the PDF could not be opened")

        stored = manifest.get(item.source_id)

        assert stored.state is SavedItemState.FAILED
        assert stored.failure_reason == "the PDF could not be opened"

    def test_a_later_import_clears_an_earlier_failure(
        self, tmp_path: Path
    ):
        manifest = SavedItemsManifest(manifest_path(tmp_path))

        item = item_for(POST_A)
        manifest.upsert(item)
        manifest.mark_failed(item.source_id, "corrupt")
        manifest.mark_imported(item.source_id, "urn-li-saved-abc")

        assert manifest.get(item.source_id).failure_reason is None

    def test_outstanding_items_exclude_terminal_ones(
        self, tmp_path: Path
    ):
        manifest = SavedItemsManifest(manifest_path(tmp_path))

        pending = item_for(POST_A)
        done = item_for(POST_B)

        manifest.upsert(pending)
        manifest.upsert(done)
        manifest.mark_enriched(done.source_id, "urn-li-saved-b")

        assert [entry.source_id for entry in manifest.outstanding()] == [
            pending.source_id
        ]


class TestReport:
    def _manifest(self, tmp_path: Path) -> SavedItemsManifest:
        manifest = SavedItemsManifest(manifest_path(tmp_path))

        captured = item_for(POST_A)
        captured.content_digest = "abc"
        captured.state = SavedItemState.IMPORTED

        manifest.upsert(item_for(POST_B))
        manifest.upsert(captured)

        return manifest

    def test_the_counts_add_up(self, tmp_path: Path):
        manifest = self._manifest(tmp_path)

        report = manifest.report(discovered=5, new=4, duplicates=1)

        assert report.discovered == report.new + report.duplicates

    def test_metadata_only_items_are_counted_separately(
        self, tmp_path: Path
    ):
        report = self._manifest(tmp_path).report()

        assert report.with_content == 1
        assert report.metadata_only == 1

    def test_pending_is_the_backlog_not_a_failure(self, tmp_path: Path):
        report = self._manifest(tmp_path).report()

        assert report.pending == 1
        assert report.failed == 0

    def test_the_report_names_every_row(self, tmp_path: Path):
        rendered = self._manifest(tmp_path).report().render()

        for label in (
            "Discovered",
            "New",
            "Duplicates",
            "With content",
            "Metadata only",
            "Imported",
            "Enriched",
            "Failed",
            "Pending",
        ):
            assert label in rendered

    def test_issues_are_listed_with_their_reason(self, tmp_path: Path):
        report = self._manifest(tmp_path).report(
            issues=["manifest.csv line 4: not an absolute URL"]
        )

        assert "line 4" in report.render()
        assert len(report.as_dict()["issues"]) == 1

    def test_the_report_is_serializable(self, tmp_path: Path):
        payload = self._manifest(tmp_path).report().as_dict()

        assert json.loads(json.dumps(payload)) == payload

    def test_the_report_holds_no_credential(self, tmp_path: Path):
        report = self._manifest(tmp_path).report()

        assert "password" not in json.dumps(report.as_dict()).lower()


def test_captured_content_describes_what_it_found():
    assert CapturedContent().describe() == "no content"
    assert "text" in CapturedContent(text="hello").describe()
