"""
Tests for the manual source.

The manual source is how authorized material actually enters the
project, so what it accepts and what it refuses is the boundary that
matters. These cover the bundle shapes a user would realistically drop
in, the determinism that makes a repeated import safe, and the
containment that stops a bundle naming a file anywhere on the machine.

Every fixture here is synthetic. No real credential appears in any of
them, and none is needed to run these tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ingestion.errors import InvalidPostError
from src.ingestion.sources.base import CollectionStopped
from src.ingestion.sources.manual import (
    ManualSource,
    content_digest,
)


def collect(source: ManualSource) -> list:
    posts = []

    try:
        for post in source.discover():
            posts.append(post)
    except CollectionStopped:
        pass

    return posts


def bundle(root: Path, name: str) -> Path:
    path = root / name
    path.mkdir(parents=True, exist_ok=True)

    return path


def png_bytes() -> bytes:
    """A minimal but real PNG, so media handling has a true file."""

    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
        b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
        b"\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )


def pdf_bytes() -> bytes:
    """A minimal single-page PDF with extractable text."""

    body = (
        b"BT /F1 12 Tf 72 720 Td "
        b"(Delta Lake compaction interview notes) Tj ET"
    )

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(body)).encode() + b" >>\nstream\n"
        + body + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")

    offsets = []

    for number, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + obj + b"\nendobj\n"

    xref_at = len(out)

    out += b"xref\n0 " + str(len(objects) + 1).encode() + b"\n"
    out += b"0000000000 65535 f \n"

    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()

    out += (
        b"trailer\n<< /Size " + str(len(objects) + 1).encode()
        + b" /Root 1 0 R >>\nstartxref\n"
        + str(xref_at).encode() + b"\n%%EOF\n"
    )

    return bytes(out)


# ---------------------------------------------------------------------
# Bundle shapes
# ---------------------------------------------------------------------


def test_a_plain_text_file_is_one_post(tmp_path):
    (tmp_path / "note.txt").write_text(
        "Spark partitioning interview notes.", encoding="utf-8"
    )

    collected = collect(ManualSource(tmp_path))

    assert len(collected) == 1
    assert collected[0].text == "Spark partitioning interview notes."


def test_a_markdown_file_is_one_post(tmp_path):
    (tmp_path / "guide.md").write_text(
        "# Z-Ordering\n\nZ-Ordering improves pruning.",
        encoding="utf-8",
    )

    collected = collect(ManualSource(tmp_path))

    assert len(collected) == 1
    assert "# Z-Ordering" in collected[0].text


def test_a_json_capture_declares_its_own_identity(tmp_path):
    directory = bundle(tmp_path, "capture-1")

    (directory / "capture.json").write_text(
        json.dumps(
            {
                "id": "urn:li:activity:99",
                "text": "From a capture.",
                "url": "https://example.com/99",
                "author": "my-handle",
                "published_at": "2026-05-05T10:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    collected = collect(ManualSource(tmp_path))

    assert len(collected) == 1
    assert collected[0].source_post_id == "urn:li:activity:99"
    assert collected[0].url == "https://example.com/99"
    assert collected[0].author == "my-handle"


def test_a_capture_only_directory_is_still_a_bundle(tmp_path):
    """
    A capture with no other files is still material the user supplied.
    Dropping it would lose the capture, because the capture file on its
    own is not treated as a post.
    """

    directory = bundle(tmp_path, "capture-only")

    (directory / "capture.json").write_text(
        json.dumps({"id": "c1", "text": "Only a capture."}),
        encoding="utf-8",
    )

    collected = collect(ManualSource(tmp_path))

    assert len(collected) == 1
    assert collected[0].source_post_id == "c1"


def test_a_jsonl_export_is_one_post_per_line(tmp_path):
    (tmp_path / "export.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "text": "First"}),
                "",
                json.dumps({"id": "b", "text": "Second"}),
            ]
        ),
        encoding="utf-8",
    )

    collected = collect(ManualSource(tmp_path))

    assert [post.source_post_id for post in collected] == ["a", "b"]
    assert [post.text for post in collected] == ["First", "Second"]


def test_a_malformed_jsonl_line_is_reported_not_fatal(tmp_path):
    """
    One bad record must cost one record, not the whole export. A
    truncated export is a normal thing for a user to produce.
    """

    (tmp_path / "export.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "text": "Good"}),
                "{not json",
                json.dumps({"id": "b", "text": "Also good"}),
            ]
        ),
        encoding="utf-8",
    )

    collected = collect(ManualSource(tmp_path))

    assert len(collected) == 2
    assert collected[0].extra["jsonl_problems"]


def test_a_jsonl_with_no_usable_record_is_rejected(tmp_path):
    (tmp_path / "export.jsonl").write_text(
        "\n".join(["{not json", "also not json"]), encoding="utf-8"
    )

    with pytest.raises(InvalidPostError):
        collect(ManualSource(tmp_path))


def test_a_directory_becomes_one_post_with_its_media(tmp_path):
    """The shape a user actually drops in: text plus images."""

    directory = bundle(tmp_path, "source-001")

    (directory / "content.md").write_text(
        "Delta Lake compaction interview notes.", encoding="utf-8"
    )
    (directory / "diagram.png").write_bytes(png_bytes())
    (directory / "excerpt.png").write_bytes(png_bytes())

    collected = collect(ManualSource(tmp_path))

    assert len(collected) == 1
    assert "Delta Lake" in collected[0].text
    assert len(collected[0].media) == 2
    assert [path.name for path in collected[0].media] == [
        "diagram.png",
        "excerpt.png",
    ]


def test_a_pdf_only_directory_is_a_post_with_media_and_no_text(tmp_path):
    """
    A document carries its content, and the media stage is what turns it
    into text. The post exists with its provenance and its file, so it
    is reachable rather than invisible.
    """

    directory = bundle(tmp_path, "document-only")
    (directory / "notes.pdf").write_bytes(pdf_bytes())

    collected = collect(ManualSource(tmp_path))

    assert len(collected) == 1
    assert collected[0].text == ""
    assert [path.name for path in collected[0].media] == ["notes.pdf"]


def test_a_capture_names_its_own_media(tmp_path):
    directory = bundle(tmp_path, "declared")

    (directory / "capture.json").write_text(
        json.dumps(
            {
                "id": "decl-1",
                "text": "Declared media.",
                "media": ["shot.png"],
            }
        ),
        encoding="utf-8",
    )
    (directory / "shot.png").write_bytes(png_bytes())

    collected = collect(ManualSource(tmp_path))

    assert len(collected) == 1
    assert [path.name for path in collected[0].media] == ["shot.png"]


def test_several_bundles_become_several_posts(tmp_path):
    for index in range(3):
        directory = bundle(tmp_path, f"source-{index:03d}")
        (directory / "content.md").write_text(
            f"Bundle {index} content.", encoding="utf-8"
        )

    collected = collect(ManualSource(tmp_path))

    assert len(collected) == 3
    assert {post.extra["bundle_name"] for post in collected} == {
        "source-000",
        "source-001",
        "source-002",
    }


def test_nested_directories_are_each_a_bundle(tmp_path):
    outer = bundle(tmp_path, "outer")
    inner = bundle(outer, "inner")

    (outer / "a.md").write_text("Outer note.", encoding="utf-8")
    (inner / "b.md").write_text("Inner note.", encoding="utf-8")

    collected = collect(ManualSource(tmp_path))

    assert sorted(post.text for post in collected) == [
        "Inner note.",
        "Outer note.",
    ]


# ---------------------------------------------------------------------
# Determinism and idempotency
# ---------------------------------------------------------------------


def test_the_same_bundle_yields_the_same_identifier(tmp_path):
    directory = bundle(tmp_path, "stable")
    (directory / "content.md").write_text("Stable text.", encoding="utf-8")

    first = collect(ManualSource(tmp_path))
    second = collect(ManualSource(tmp_path))

    assert first[0].source_post_id == second[0].source_post_id


def test_editing_a_bundle_changes_its_identifier(tmp_path):
    """
    A changed bundle has to be recognisable as changed, or a corrected
    file would be silently ignored as a duplicate.
    """

    directory = bundle(tmp_path, "edited")
    (directory / "content.md").write_text("First version.", encoding="utf-8")

    before = collect(ManualSource(tmp_path))

    (directory / "content.md").write_text(
        "Second version, corrected.", encoding="utf-8"
    )

    after = collect(ManualSource(tmp_path))

    assert before[0].source_post_id != after[0].source_post_id


def test_adding_media_changes_the_identifier(tmp_path):
    directory = bundle(tmp_path, "media-changed")
    (directory / "content.md").write_text("Same text.", encoding="utf-8")

    before = collect(ManualSource(tmp_path))

    (directory / "extra.png").write_bytes(png_bytes())

    after = collect(ManualSource(tmp_path))

    assert before[0].source_post_id != after[0].source_post_id


def test_media_order_does_not_change_the_identifier(tmp_path):
    directory = bundle(tmp_path, "order")

    for name in ("a.md", "b.md"):
        (directory / name).write_text("text", encoding="utf-8")

    first = content_digest("body", ["b.png", "a.png"])
    second = content_digest("body", ["a.png", "b.png"])

    assert first == second


def test_an_anonymous_bundle_still_gets_a_stable_identifier(tmp_path):
    """A file with no useful name must not fall back to the same id."""

    (tmp_path / "content.md").write_text(
        "Unnamed but real content.", encoding="utf-8"
    )

    collected = collect(ManualSource(tmp_path))

    identifier = collected[0].source_post_id

    # Derived from the content, so it is readable and stable even when
    # the file name says nothing useful.
    assert identifier.startswith("unnamed-but-real-content")
    assert len(identifier) > 8


def test_two_different_anonymous_bundles_do_not_collide(tmp_path):
    (tmp_path / "a.md").write_text("First thing.", encoding="utf-8")
    (tmp_path / "b.md").write_text("Second thing.", encoding="utf-8")

    collected = collect(ManualSource(tmp_path))

    identifiers = {post.source_post_id for post in collected}

    assert len(identifiers) == len(collected) == 2


def test_the_content_digest_is_recorded(tmp_path):
    directory = bundle(tmp_path, "recorded")
    (directory / "content.md").write_text("Digest me.", encoding="utf-8")

    collected = collect(ManualSource(tmp_path))

    assert len(collected[0].extra["content_digest"]) == 64


def test_a_capture_without_an_id_is_identified_by_content(tmp_path):
    directory = bundle(tmp_path, "anonymous-capture")

    (directory / "capture.json").write_text(
        json.dumps({"text": "No declared identity."}), encoding="utf-8"
    )

    collected = collect(ManualSource(tmp_path))

    assert collected[0].source_post_id
    assert collected[0].text == "No declared identity."


# ---------------------------------------------------------------------
# Containment
# ---------------------------------------------------------------------


def test_a_capture_cannot_name_a_file_outside_its_bundle(tmp_path):
    """
    A capture is user-supplied content, so it must not be able to name
    a file anywhere on the machine.
    """

    outside = tmp_path / "outside.txt"
    outside.write_text("Secret file outside any bundle.", encoding="utf-8")

    directory = bundle(tmp_path, "escaping")

    (directory / "capture.json").write_text(
        json.dumps(
            {"id": "esc-1", "text": "Body", "media": ["../outside.txt"]}
        ),
        encoding="utf-8",
    )

    collected = collect(ManualSource(tmp_path))

    assert collected[0].media == []


def test_a_capture_cannot_use_an_absolute_path(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("Elsewhere.", encoding="utf-8")

    directory = bundle(tmp_path, "absolute")

    (directory / "capture.json").write_text(
        json.dumps(
            {"id": "abs-1", "text": "Body", "media": [str(outside)]}
        ),
        encoding="utf-8",
    )

    collected = collect(ManualSource(tmp_path))

    assert collected[0].media == []


def test_a_capture_naming_a_missing_file_yields_no_media(tmp_path):
    directory = bundle(tmp_path, "missing")

    (directory / "capture.json").write_text(
        json.dumps(
            {"id": "miss-1", "text": "Body", "media": ["nope.png"]}
        ),
        encoding="utf-8",
    )

    collected = collect(ManualSource(tmp_path))

    assert collected[0].media == []


def test_a_capture_that_is_not_an_object_is_rejected(tmp_path):
    directory = bundle(tmp_path, "not-an-object")

    (directory / "capture.json").write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(InvalidPostError):
        collect(ManualSource(tmp_path))


def test_malformed_json_is_rejected_clearly(tmp_path):
    directory = bundle(tmp_path, "broken")

    (directory / "capture.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(InvalidPostError):
        collect(ManualSource(tmp_path))


# ---------------------------------------------------------------------
# What is not a bundle
# ---------------------------------------------------------------------


def test_unrelated_files_are_ignored(tmp_path):
    (tmp_path / "notes.py").write_text("x = 1", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("a: 1", encoding="utf-8")
    (tmp_path / "scratch.tmp").write_text("noise", encoding="utf-8")

    assert collect(ManualSource(tmp_path)) == []


def test_an_empty_directory_yields_nothing(tmp_path):
    tmp_path.mkdir(exist_ok=True)

    assert collect(ManualSource(tmp_path)) == []


def test_an_empty_text_file_is_not_a_post(tmp_path):
    """
    A post with no text and no media is nothing, and storing it would
    publish an empty page.
    """

    (tmp_path / "empty.md").write_text("   \n\n  ", encoding="utf-8")

    assert collect(ManualSource(tmp_path)) == []


def test_a_capture_claims_the_text_beside_it(tmp_path):
    """
    A capture declares what a post is, so a stray note next to it is
    not a second post.
    """

    directory = bundle(tmp_path, "claimed")

    (directory / "capture.json").write_text(
        json.dumps({"id": "cl-1", "text": "The real content."}),
        encoding="utf-8",
    )
    (directory / "scratch.md").write_text(
        "My working notes.", encoding="utf-8"
    )

    collected = collect(ManualSource(tmp_path))

    assert len(collected) == 1
    assert collected[0].text == "The real content."


# ---------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------


def test_max_posts_is_honoured(tmp_path):
    for index in range(5):
        directory = bundle(tmp_path, f"b{index}")
        (directory / "content.md").write_text(
            f"Post {index}.", encoding="utf-8"
        )

    yielded = []

    with pytest.raises(CollectionStopped) as error:
        for post in ManualSource(tmp_path).discover(max_posts=2):
            yielded.append(post)

    assert len(yielded) == 2
    assert error.value.reason.value == "max_posts"


def test_a_date_window_filters_posts(tmp_path):
    old = bundle(tmp_path, "old")
    (old / "capture.json").write_text(
        json.dumps({"id": "old-1", "text": "Old.", "published_at": "2020-01-01"}),
        encoding="utf-8",
    )

    new = bundle(tmp_path, "new")
    (new / "capture.json").write_text(
        json.dumps({"id": "new-1", "text": "New.", "published_at": "2026-01-01"}),
        encoding="utf-8",
    )

    collected = collect(ManualSource(tmp_path))

    filtered = collect(
        _with_window(ManualSource(tmp_path), since="2025-01-01")
    )

    assert len(collected) == 2
    assert [post.source_post_id for post in filtered] == ["new-1"]


def _with_window(source: ManualSource, **limits) -> ManualSource:
    source.discover = (  # type: ignore[method-assign]
        lambda **_: _filtered(source, **limits)
    )

    return source


def _filtered(source: ManualSource, **limits):
    try:
        for post in ManualSource.discover(source, **limits):
            yield post
    except CollectionStopped:
        return


def test_an_unparsable_date_is_kept(tmp_path):
    """
    A format change must not silently drop content, so a date that
    cannot be read is kept rather than filtered out.
    """

    directory = bundle(tmp_path, "odd-date")

    (directory / "capture.json").write_text(
        json.dumps({"id": "odd-1", "text": "Body", "published_at": "sometime"}),
        encoding="utf-8",
    )

    collected = collect(_with_window(ManualSource(tmp_path), since="2020-01-01"))

    assert len(collected) == 1


def test_a_missing_root_fails_clearly(tmp_path):
    with pytest.raises(CollectionStopped):
        list(ManualSource(tmp_path / "absent").discover())


def test_a_single_file_can_be_the_root(tmp_path):
    target = tmp_path / "one.txt"
    target.write_text("Single file.", encoding="utf-8")

    collected = collect(ManualSource(target))

    assert len(collected) == 1
    assert collected[0].text == "Single file."


# ---------------------------------------------------------------------
# The platform override
# ---------------------------------------------------------------------


def test_the_platform_can_be_overridden(tmp_path):
    source = ManualSource(tmp_path, platform="book")

    assert source.platform == "book"
    assert source.name == "manual"


def test_two_copies_of_one_capture_collapse_onto_one_post(tmp_path):
    """
    The property that makes a repeated drop safe. Two copies sit in
    directories with different names, so an identifier derived from the
    name would treat them as two posts.
    """

    for name in ("copy-one", "copy-two"):
        directory = bundle(tmp_path, name)
        (directory / "post.txt").write_text(
            "The same captured material, saved twice.", encoding="utf-8"
        )

    collected = collect(ManualSource(tmp_path))

    assert len(collected) == 2

    # Same identifier, because the identifier comes from the content.
    assert (
        collected[0].source_post_id == collected[1].source_post_id
    )


def test_different_material_does_not_collapse(tmp_path):
    for name in ("first", "second"):
        directory = bundle(tmp_path, name)
        (directory / "post.txt").write_text(
            f"Distinctly different material in {name}.", encoding="utf-8"
        )

    collected = collect(ManualSource(tmp_path))

    assert (
        collected[0].source_post_id != collected[1].source_post_id
    )


def test_an_identifier_is_readable(tmp_path):
    """
    A reader meets these identifiers in URLs and directory names, so
    they have to say something about the content.
    """

    directory = bundle(tmp_path, "opaque-name")
    (directory / "post.txt").write_text(
        "Star schema denormalisation tradeoffs", encoding="utf-8"
    )

    collected = collect(ManualSource(tmp_path))

    assert collected[0].source_post_id.startswith("star-schema")


def test_a_markdown_heading_becomes_the_identifier(tmp_path):
    directory = bundle(tmp_path, "heading")
    (directory / "content.md").write_text(
        "# Slowly Changing Dimensions\n\nType 2 keeps the history.",
        encoding="utf-8",
    )

    collected = collect(ManualSource(tmp_path))

    assert collected[0].source_post_id.startswith(
        "slowly-changing-dimensions"
    )
