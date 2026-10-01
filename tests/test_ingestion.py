"""
Tests for the ingestion layer.

The layer's job is to make adding a manually captured post safe and
painless, so these tests cover the three things that can actually go
wrong:

* a post that the pipeline cannot read, or that publishes nothing
* a capture that is imported twice, duplicating or destroying content
* media that reaches outside the post, or overwrites other content

The compatibility tests are just as important: the existing
``data/posts/*/post.json`` structure, the worker's loader and the
enrichment pipeline must all keep working untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ingestion import (
    IngestionError,
    InvalidPostError,
    MediaConflictError,
    PostExistsError,
    PostNotFoundError,
    UnsupportedMediaError,
    add_media,
    create_post,
    discover_posts,
    discover_posts,
    import_post,
    load_post,
    normalize_post_id,
    validate_post,
    validate_posts,
)
from src.ingestion.cli import main as cli_main
from src.ingestion.importer import (
    MediaImportResult,
    PostImportResult,
    safe_media_name,
)
from src.ingestion.post_document import (
    MEDIA_DIRECTORY_NAME,
    POST_FILE_NAME,
    PostDocument,
    media_type_for,
    resolve_media_path,
)


REPO_ROOT = Path(__file__).resolve().parents[1]

REAL_POSTS_ROOT = REPO_ROOT / "data" / "posts"

# One pixel, so image processing can still read it if a test needs it.
PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00"
    b"\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\n"
    b"IDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n\x2d\xb4\x00\x00"
    b"\x00\x00IEND\xaeB`\x82"
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def posts_root(tmp_path: Path, monkeypatch) -> Path:
    """
    A posts tree outside the checkout.

    The importer defaults to the repository's ``data/posts``, so the
    default is redirected as well: a test that forgets ``root=`` then
    fails on its own data instead of writing real posts into git.
    """

    root = tmp_path / "posts"
    root.mkdir()

    monkeypatch.setattr(
        "src.ingestion.importer.DEFAULT_POSTS_ROOT", root
    )
    monkeypatch.setattr(
        "src.ingestion.cli.DEFAULT_POSTS_ROOT", root
    )

    return root


def make_capture(
    tmp_path: Path,
    *,
    text: str = "Explain partition pruning.",
    media: dict[str, bytes] | None = None,
    post_json: dict | None = None,
    name: str = "capture",
) -> Path:
    """Build the directory a person has after a manual capture."""

    bundle = tmp_path / name
    bundle.mkdir()

    if text is not None:
        (bundle / "notes.md").write_text(text, encoding="utf-8")

    for filename, content in (media or {}).items():
        (bundle / filename).write_bytes(content)

    if post_json is not None:
        (bundle / POST_FILE_NAME).write_text(
            json.dumps(post_json, indent=2), encoding="utf-8"
        )

    return bundle


def read_document(directory: Path) -> dict:
    return json.loads(
        (directory / POST_FILE_NAME).read_text(encoding="utf-8")
    )


# ---------------------------------------------------------------------
# Post ids
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    ["sample_001", "2026-01-01-spark", "post.2", "a"],
)
def test_usable_post_ids_are_accepted(value):
    assert normalize_post_id(value) == value


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        "Sample_001",
        "../escape",
        "nested/post",
        "with space",
        "trailing/slash",
        ".hidden",
        "..",
        "post.json",
    ],
)
def test_unusable_post_ids_are_rejected(value):
    """
    A post id becomes a directory name, part of a job id and a URL
    segment, so anything unsafe is refused up front rather than
    discovered by a worker.
    """

    with pytest.raises(InvalidPostError):
        normalize_post_id(value)


def test_over_long_post_id_is_rejected():
    with pytest.raises(InvalidPostError):
        normalize_post_id("a" * 200)


# ---------------------------------------------------------------------
# Creating a post
# ---------------------------------------------------------------------


def test_new_post_uses_the_committed_document_structure(posts_root):
    create_post(
        "sample_100", text="Question about Spark.", root=posts_root
    )

    document = read_document(posts_root / "sample_100")

    # Exactly the keys the repository already ships, so the worker
    # needs no migration.
    assert set(document) == {
        "id",
        "source",
        "original_text",
        "media",
        "ai_analysis",
        "interview_questions",
        "classification",
    }

    assert document["id"] == "sample_100"
    assert document["original_text"] == "Question about Spark."
    assert document["media"] == []
    assert document["interview_questions"] == []
    assert document["ai_analysis"] == {
        "summary": None,
        "topics": [],
        "subtopics": [],
        "concepts": [],
        "image_descriptions": [],
    }
    assert document["classification"]["domain"] == "Data Engineering"


def test_new_post_records_provenance(posts_root):
    create_post(
        "sample_101",
        text="Question.",
        platform="manual",
        url="https://example.com/101",
        author="Interviewer",
        captured_at="2026-01-02T09:00:00+05:30",
        primary_topic="Databricks",
        secondary_topics=("Delta Lake",),
        interview_relevant=True,
        root=posts_root,
    )

    source = read_document(posts_root / "sample_101")["source"]

    assert source["platform"] == "manual"
    assert source["url"] == "https://example.com/101"
    assert source["author"] == "Interviewer"
    assert source["captured_at"] == "2026-01-02T09:00:00+05:30"

    classification = read_document(posts_root / "sample_101")[
        "classification"
    ]

    assert classification["primary_topic"] == "Databricks"
    assert classification["secondary_topics"] == ["Delta Lake"]
    assert classification["interview_relevant"] is True


def test_new_post_defaults_to_manual_provenance(posts_root):
    """
    A post added by hand must say so, rather than implying a named
    collection process.
    """

    create_post("sample_102", text="Question.", root=posts_root)

    assert read_document(posts_root / "sample_102")["source"][
        "platform"
    ] == "manual"


def test_new_post_is_immediately_valid(posts_root):
    create_post("sample_103", text="Question.", root=posts_root)

    report = validate_posts(root=posts_root)

    assert report.ok
    assert report.errors == ()


def test_new_post_is_immediately_loadable(posts_root):
    create_post(
        "sample_104", text="Question about joins.", root=posts_root
    )

    post = load_post(posts_root / "sample_104")

    assert post.id == "sample_104"
    assert post.original_text == "Question about joins."
    assert post.media == []


def test_creating_an_existing_post_is_refused(posts_root):
    create_post("sample_105", text="Question.", root=posts_root)

    with pytest.raises(PostExistsError) as error:
        create_post("sample_105", text="Different.", root=posts_root)

    assert "already exists" in str(error.value)

    # The original is untouched.
    assert read_document(posts_root / "sample_105")[
        "original_text"
    ] == "Question."


def test_overwrite_is_explicit(posts_root):
    create_post("sample_106", text="Question.", root=posts_root)

    create_post(
        "sample_106",
        text="Replacement.",
        overwrite=True,
        root=posts_root,
    )

    assert read_document(posts_root / "sample_106")[
        "original_text"
    ] == "Replacement."


def test_empty_post_is_refused_by_default(posts_root):
    with pytest.raises(InvalidPostError) as error:
        create_post("sample_107", root=posts_root)

    assert "--allow-empty" in str(error.value)

    assert not (posts_root / "sample_107").exists()


def test_empty_shell_can_be_created_explicitly(posts_root):
    create_post("sample_108", allow_empty=True, root=posts_root)

    report = validate_posts(root=posts_root)

    assert not report.ok
    assert "contributes nothing" in report.errors[0].message


# ---------------------------------------------------------------------
# Adding media
# ---------------------------------------------------------------------


def test_add_media_copies_and_declares(posts_root, tmp_path):
    create_post("sample_110", text="Question.", root=posts_root)

    shot = tmp_path / "shot.png"
    shot.write_bytes(PNG_BYTES)

    result = add_media("sample_110", [shot], root=posts_root)

    assert isinstance(result, MediaImportResult)
    assert result.added == ("shot.png",)

    media_directory = posts_root / "sample_110" / MEDIA_DIRECTORY_NAME

    assert (media_directory / "shot.png").read_bytes() == PNG_BYTES

    declared = read_document(posts_root / "sample_110")["media"]

    assert declared == [
        {"type": "image", "path": "media/shot.png"}
    ]


def test_declared_media_path_is_relative_to_the_post(
    posts_root, tmp_path
):
    """
    The repository has to stay portable, so a committed post may never
    record a path that depends on the runner's working directory.
    """

    create_post("sample_111", text="Question.", root=posts_root)

    shot = tmp_path / "shot.png"
    shot.write_bytes(PNG_BYTES)

    add_media("sample_111", [shot], root=posts_root)

    declared = read_document(posts_root / "sample_111")["media"][0]

    assert declared["path"] == "media/shot.png"
    assert not Path(declared["path"]).is_absolute()


def test_add_media_records_a_description(posts_root, tmp_path):
    create_post("sample_112", text="Question.", root=posts_root)

    shot = tmp_path / "shot.png"
    shot.write_bytes(PNG_BYTES)

    add_media(
        "sample_112",
        [shot],
        root=posts_root,
        description="The architecture diagram",
    )

    declared = read_document(posts_root / "sample_112")["media"][0]

    assert declared["description"] == "The architecture diagram"


def test_declared_media_reaches_the_loader(posts_root, tmp_path):
    create_post("sample_113", text="Question.", root=posts_root)

    shot = tmp_path / "shot.png"
    shot.write_bytes(PNG_BYTES)

    add_media(
        "sample_113", [shot], root=posts_root, description="A diagram"
    )

    post = load_post(posts_root / "sample_113")

    assert len(post.media) == 1
    assert post.media[0].type == "image"
    assert post.media[0].description == "A diagram"
    # Compared with forward slashes so the assertion is identical on
    # Windows and POSIX.
    assert post.media[0].path.replace("\\", "/").endswith(
        "media/shot.png"
    )


def test_add_media_is_idempotent(posts_root, tmp_path):
    create_post("sample_114", text="Question.", root=posts_root)

    shot = tmp_path / "shot.png"
    shot.write_bytes(PNG_BYTES)

    add_media("sample_114", [shot], root=posts_root)
    second = add_media("sample_114", [shot], root=posts_root)

    assert second.added == ()
    assert second.skipped == ("shot.png",)
    assert len(read_document(posts_root / "sample_114")["media"]) == 1


def test_conflicting_media_is_refused(posts_root, tmp_path):
    create_post("sample_115", text="Question.", root=posts_root)

    original = tmp_path / "shot.png"
    original.write_bytes(PNG_BYTES)

    add_media("sample_115", [original], root=posts_root)

    replacement = tmp_path / "other" / "shot.png"
    replacement.parent.mkdir()
    replacement.write_bytes(PNG_BYTES + b"changed")

    with pytest.raises(MediaConflictError) as error:
        add_media("sample_115", [replacement], root=posts_root)

    assert "--force" in str(error.value)

    # The committed bytes are unchanged.
    assert (
        posts_root
        / "sample_115"
        / MEDIA_DIRECTORY_NAME
        / "shot.png"
    ).read_bytes() == PNG_BYTES


def test_force_replaces_conflicting_media(posts_root, tmp_path):
    create_post("sample_116", text="Question.", root=posts_root)

    original = tmp_path / "shot.png"
    original.write_bytes(PNG_BYTES)

    add_media("sample_116", [original], root=posts_root)

    replacement = tmp_path / "reshoot" / "shot.png"
    replacement.parent.mkdir()
    replacement.write_bytes(PNG_BYTES + b"changed")

    result = add_media(
        "sample_116", [replacement], root=posts_root, force=True
    )

    assert result.added == ("shot.png",)
    assert len(read_document(posts_root / "sample_116")["media"]) == 1
    assert (
        posts_root
        / "sample_116"
        / MEDIA_DIRECTORY_NAME
        / "shot.png"
    ).read_bytes() == PNG_BYTES + b"changed"


def test_adding_media_from_a_directory(posts_root, tmp_path):
    create_post("sample_117", text="Question.", root=posts_root)

    bundle = tmp_path / "bundle"
    bundle.mkdir()

    (bundle / "a.png").write_bytes(PNG_BYTES)
    (bundle / "b.pdf").write_bytes(b"%PDF-1.4 fake")
    (bundle / ".DS_Store").write_bytes(b"junk")
    (bundle / "notes.md").write_text("ignored", encoding="utf-8")

    rendered = bundle / "a_pages"
    rendered.mkdir()
    (rendered / "page_001.png").write_bytes(PNG_BYTES)

    result = add_media("sample_117", [bundle], root=posts_root)

    # A directory is expanded like a capture bundle: only media files
    # are taken, hidden files and the notes are left behind, and a
    # subdirectory is enrichment output rather than a capture.
    assert result.added == ("a.png", "b.pdf")

    declared = read_document(posts_root / "sample_117")["media"]

    assert [entry["type"] for entry in declared] == ["image", "pdf"]


def test_an_explicitly_named_file_is_always_taken(posts_root, tmp_path):
    """
    A file named directly is an unambiguous instruction, even when a
    directory expansion would have skipped it.
    """

    create_post("sample_121", text="Question.", root=posts_root)

    notes = tmp_path / "notes.txt"
    notes.write_text("A transcript.", encoding="utf-8")

    result = add_media("sample_121", [notes], root=posts_root)

    assert result.added == ("notes.txt",)

    declared = read_document(posts_root / "sample_121")["media"]

    assert declared == [{"type": "other", "path": "media/notes.txt"}]


def test_awkward_media_names_are_made_portable(posts_root, tmp_path):
    create_post("sample_118", text="Question.", root=posts_root)

    shot = tmp_path / "Screenshot 2026-01-01 at 10.15.42.png"
    shot.write_bytes(PNG_BYTES)

    result = add_media("sample_118", [shot], root=posts_root)

    assert result.added == ("Screenshot_2026-01-01_at_10.15.42.png",)
    assert result.renamed == (
        (
            "Screenshot 2026-01-01 at 10.15.42.png",
            "Screenshot_2026-01-01_at_10.15.42.png",
        ),
    )


@pytest.mark.parametrize(
    "name",
    ["...", "///", "   "],
)
def test_unnameable_media_is_refused(name):
    with pytest.raises(UnsupportedMediaError):
        safe_media_name(name)


def test_add_media_requires_an_existing_post(posts_root, tmp_path):
    shot = tmp_path / "shot.png"
    shot.write_bytes(PNG_BYTES)

    with pytest.raises(PostNotFoundError) as error:
        add_media("sample_119", [shot], root=posts_root)

    assert "Create the post first" in str(error.value)


def test_add_media_requires_a_real_source(posts_root, tmp_path):
    create_post("sample_120", text="Question.", root=posts_root)

    with pytest.raises(IngestionError):
        add_media(
            "sample_120", [tmp_path / "absent.png"], root=posts_root
        )


def test_media_type_is_derived_from_the_extension():
    assert media_type_for("a.png") == "image"
    assert media_type_for("a.JPEG") == "image"
    assert media_type_for("a.pdf") == "pdf"
    assert media_type_for("a.txt") == "other"


# ---------------------------------------------------------------------
# Importing a capture bundle
# ---------------------------------------------------------------------


def test_import_creates_a_complete_post(posts_root, tmp_path):
    bundle = make_capture(
        tmp_path,
        text="Explain a broadcast join.",
        media={"shot.png": PNG_BYTES, "notes.pdf": b"%PDF-1.4 fake"},
    )

    result = import_post(
        "sample_130", bundle, root=posts_root, author="Interviewer"
    )

    assert isinstance(result, PostImportResult)
    assert result.created is True
    assert result.media_added == ("notes.pdf", "shot.png")

    directory = posts_root / "sample_130"

    document = read_document(directory)

    assert document["original_text"] == "Explain a broadcast join."
    assert document["source"]["author"] == "Interviewer"
    assert [entry["path"] for entry in document["media"]] == [
        "media/notes.pdf",
        "media/shot.png",
    ]

    assert (directory / MEDIA_DIRECTORY_NAME / "shot.png").is_file()
    assert (directory / MEDIA_DIRECTORY_NAME / "notes.pdf").is_file()

    # The bundle is left alone: an import copies, it never moves.
    assert (bundle / "shot.png").is_file()


def test_imported_post_is_valid_and_loadable(posts_root, tmp_path):
    bundle = make_capture(
        tmp_path, text="Question.", media={"shot.png": PNG_BYTES}
    )

    import_post("sample_131", bundle, root=posts_root)

    report = validate_posts(root=posts_root)

    assert report.ok, [str(issue) for issue in report.issues]

    post = load_post(posts_root / "sample_131")

    assert len(post.media) == 1
    assert post.media[0].type == "image"


def test_import_is_idempotent(posts_root, tmp_path):
    bundle = make_capture(
        tmp_path, text="Question.", media={"shot.png": PNG_BYTES}
    )

    first = import_post("sample_132", bundle, root=posts_root)
    second = import_post("sample_132", bundle, root=posts_root)

    assert first.created is True
    assert second.created is False
    assert second.media_added == ()
    assert second.media.skipped == ("shot.png",)

    assert len(read_document(posts_root / "sample_132")["media"]) == 1


def test_reimport_preserves_classification(posts_root, tmp_path):
    """
    A re-import must not drop what a human recorded earlier, which is
    the difference between a refresh and a silent reset.
    """

    bundle = make_capture(tmp_path, text="Question.")

    import_post(
        "sample_133",
        bundle,
        root=posts_root,
        primary_topic="Databricks",
        secondary_topics=("Delta Lake",),
        interview_relevant=True,
        author="Interviewer",
    )

    import_post("sample_133", bundle, root=posts_root)

    classification = read_document(posts_root / "sample_133")[
        "classification"
    ]

    assert classification["primary_topic"] == "Databricks"
    assert classification["secondary_topics"] == ["Delta Lake"]
    assert classification["interview_relevant"] is True

    assert read_document(posts_root / "sample_133")["source"][
        "author"
    ] == "Interviewer"


def test_reimport_preserves_enrichment(posts_root, tmp_path):
    """
    Enrichment output belongs to the pipeline. An import that erased it
    would silently throw away a worker's work.
    """

    bundle = make_capture(tmp_path, text="Question.")

    import_post("sample_134", bundle, root=posts_root)

    directory = posts_root / "sample_134"

    document = read_document(directory)
    document["ai_analysis"]["summary"] = "A summary from the worker."
    document["interview_questions"] = [
        {
            "question": "Explain partition pruning.",
            "type": "theory",
            "difficulty": "medium",
        }
    ]
    (directory / POST_FILE_NAME).write_text(
        json.dumps(document, indent=2), encoding="utf-8"
    )

    result = import_post("sample_134", bundle, root=posts_root)

    refreshed = read_document(directory)

    assert result.created is False
    assert refreshed["ai_analysis"]["summary"] == (
        "A summary from the worker."
    )
    assert len(refreshed["interview_questions"]) == 1
    assert any("enrichment" in note for note in result.notes)


def test_import_uses_a_bundled_post_json(posts_root, tmp_path):
    """
    The existing structure is honoured: a capture may ship a
    hand-written post.json, questions and all.
    """

    bundle = make_capture(
        tmp_path,
        text=None,
        post_json={
            "id": "sample_135",
            "source": {
                "platform": "manual",
                "url": "https://example.com/135",
                "captured_at": "2026-01-03T10:00:00+05:30",
                "author": "Interviewer",
            },
            "original_text": "A hand-written post.",
            "media": [],
            "interview_questions": [
                {
                    "question": "What is a broadcast join?",
                    "type": "theory",
                    "difficulty": "easy",
                }
            ],
        },
    )

    import_post("sample_135", bundle, root=posts_root)

    document = read_document(posts_root / "sample_135")

    assert document["original_text"] == "A hand-written post."
    assert document["source"]["url"] == "https://example.com/135"
    assert document["interview_questions"][0]["question"] == (
        "What is a broadcast join?"
    )

    assert validate_post("sample_135", root=posts_root) == []


def test_import_of_a_single_text_file(posts_root, tmp_path):
    notes = tmp_path / "notes.md"
    notes.write_text("A single captured note.", encoding="utf-8")

    import_post("sample_136", notes, root=posts_root)

    assert read_document(posts_root / "sample_136")[
        "original_text"
    ] == "A single captured note."


def test_explicit_text_wins_over_a_bundled_note(posts_root, tmp_path):
    bundle = make_capture(tmp_path, text="Bundled note.")

    import_post(
        "sample_137", bundle, root=posts_root, text="Corrected text."
    )

    assert read_document(posts_root / "sample_137")[
        "original_text"
    ] == "Corrected text."


def test_import_merges_secondary_topics(posts_root, tmp_path):
    bundle = make_capture(tmp_path, text="Question.")

    import_post(
        "sample_138", bundle, root=posts_root, secondary_topics=("A",)
    )
    import_post(
        "sample_138", bundle, root=posts_root, secondary_topics=("B",)
    )

    assert read_document(posts_root / "sample_138")["classification"][
        "secondary_topics"
    ] == ["A", "B"]


def test_import_refuses_a_bundle_that_does_not_exist(
    posts_root, tmp_path
):
    with pytest.raises(IngestionError):
        import_post("sample_139", tmp_path / "absent", root=posts_root)


def test_import_reports_a_mismatched_bundled_id(posts_root, tmp_path):
    bundle = make_capture(
        tmp_path,
        text=None,
        post_json={
            "id": "capture-name",
            "source": {"platform": "manual", "captured_at": None},
            "original_text": "Text.",
        },
    )

    result = import_post("sample_140", bundle, root=posts_root)

    assert any("capture-name" in note for note in result.notes)
    assert read_document(posts_root / "sample_140")["id"] == (
        "sample_140"
    )


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------


def test_validation_of_an_empty_tree_fails(posts_root):
    report = validate_posts(root=posts_root)

    assert not report.ok
    assert "no posts found" in report.errors[0].message


def test_validation_of_a_missing_tree_fails(tmp_path):
    with pytest.raises(IngestionError):
        validate_posts(root=tmp_path / "absent")


def test_a_directory_without_post_json_is_only_a_warning(posts_root):
    """
    A half-started post must not fail a pipeline run, but it must be
    visible.
    """

    (posts_root / "sample_150").mkdir()

    report = validate_posts(root=posts_root)

    assert report.ok
    assert len(report.warnings) == 1
    assert "not ingested" in report.warnings[0].message


def test_id_must_match_the_directory(posts_root):
    directory = posts_root / "sample_151"
    directory.mkdir()

    (directory / POST_FILE_NAME).write_text(
        json.dumps(
            {
                "id": "sample_other",
                "source": {
                    "platform": "manual",
                    "captured_at": "2026-01-01T00:00:00+00:00",
                },
                "original_text": "Text.",
            }
        ),
        encoding="utf-8",
    )

    issues = validate_post("sample_151", root=posts_root)

    assert any(issue.is_error for issue in issues)
    assert any("sample_other" in issue.message for issue in issues)


def test_missing_platform_is_an_error(posts_root):
    directory = posts_root / "sample_152"
    directory.mkdir()

    (directory / POST_FILE_NAME).write_text(
        json.dumps(
            {
                "id": "sample_152",
                "source": {"captured_at": "2026-01-01T00:00:00+00:00"},
                "original_text": "Text.",
            }
        ),
        encoding="utf-8",
    )

    issues = validate_post("sample_152", root=posts_root)

    assert any(
        "source.platform" in issue.message and issue.is_error
        for issue in issues
    )


def test_unparseable_capture_time_is_an_error(posts_root):
    directory = posts_root / "sample_153"
    directory.mkdir()

    (directory / POST_FILE_NAME).write_text(
        json.dumps(
            {
                "id": "sample_153",
                "source": {
                    "platform": "manual",
                    "captured_at": "last tuesday",
                },
                "original_text": "Text.",
            }
        ),
        encoding="utf-8",
    )

    issues = validate_post("sample_153", root=posts_root)

    assert any("ISO-8601" in issue.message for issue in issues)


def test_malformed_post_json_is_reported_clearly(posts_root):
    directory = posts_root / "sample_154"
    directory.mkdir()

    (directory / POST_FILE_NAME).write_text(
        "{not json", encoding="utf-8"
    )

    issues = validate_post("sample_154", root=posts_root)

    assert [issue.is_error for issue in issues] == [True]
    assert "not valid JSON" in issues[0].message


def test_post_json_without_an_id_is_reported(posts_root):
    directory = posts_root / "sample_155"
    directory.mkdir()

    (directory / POST_FILE_NAME).write_text(
        json.dumps({"original_text": "Text."}), encoding="utf-8"
    )

    issues = validate_post("sample_155", root=posts_root)

    assert any("'id'" in issue.message for issue in issues)


def test_declared_media_must_exist(posts_root):
    create_post("sample_156", text="Question.", root=posts_root)

    document = read_document(posts_root / "sample_156")
    document["media"] = [
        {"type": "image", "path": "media/absent.png"}
    ]
    (posts_root / "sample_156" / POST_FILE_NAME).write_text(
        json.dumps(document, indent=2), encoding="utf-8"
    )

    issues = validate_post("sample_156", root=posts_root)

    assert any("missing" in issue.message for issue in issues)


def test_media_path_may_not_escape_the_post(posts_root):
    """
    A post.json is a committed file that a hand editor controls, so it
    must never be able to point the loader at a file outside the post.
    """

    create_post("sample_157", text="Question.", root=posts_root)

    document = read_document(posts_root / "sample_157")
    document["media"] = [
        {"type": "image", "path": "../../etc/passwd"}
    ]
    (posts_root / "sample_157" / POST_FILE_NAME).write_text(
        json.dumps(document, indent=2), encoding="utf-8"
    )

    issues = validate_post("sample_157", root=posts_root)

    assert any(
        "stay inside" in issue.message and issue.is_error
        for issue in issues
    )

    with pytest.raises(InvalidPostError):
        resolve_media_path(
            posts_root / "sample_157", "../../etc/passwd"
        )


def test_media_type_must_match_the_extension(posts_root):
    create_post("sample_158", text="Question.", root=posts_root)

    media_directory = posts_root / "sample_158" / MEDIA_DIRECTORY_NAME
    media_directory.mkdir(parents=True)
    (media_directory / "shot.png").write_bytes(PNG_BYTES)

    document = read_document(posts_root / "sample_158")
    document["media"] = [{"type": "pdf", "path": "media/shot.png"}]
    (posts_root / "sample_158" / POST_FILE_NAME).write_text(
        json.dumps(document, indent=2), encoding="utf-8"
    )

    issues = validate_post("sample_158", root=posts_root)

    assert any("'pdf'" in issue.message for issue in issues)


def test_duplicate_media_declarations_are_an_error(posts_root):
    create_post("sample_159", text="Question.", root=posts_root)

    media_directory = posts_root / "sample_159" / MEDIA_DIRECTORY_NAME
    media_directory.mkdir(parents=True)
    (media_directory / "shot.png").write_bytes(PNG_BYTES)

    document = read_document(posts_root / "sample_159")
    document["media"] = [
        {"type": "image", "path": "media/shot.png"},
        {"type": "image", "path": "media/shot.png"},
    ]
    (posts_root / "sample_159" / POST_FILE_NAME).write_text(
        json.dumps(document, indent=2), encoding="utf-8"
    )

    issues = validate_post("sample_159", root=posts_root)

    assert any("declared twice" in issue.message for issue in issues)


def test_undeclared_media_is_only_a_warning(posts_root, tmp_path):
    """
    An undeclared file is still ingested, so this must never fail a
    run; it is a prompt to record a description.
    """

    create_post("sample_160", text="Question.", root=posts_root)

    media_directory = posts_root / "sample_160" / MEDIA_DIRECTORY_NAME
    media_directory.mkdir(parents=True)
    (media_directory / "shot.png").write_bytes(PNG_BYTES)

    report = validate_posts(root=posts_root)

    assert report.ok
    assert any(
        "undeclared" in issue.message for issue in report.warnings
    )


def test_unknown_top_level_key_is_a_warning(posts_root):
    create_post("sample_161", text="Question.", root=posts_root)

    document = read_document(posts_root / "sample_161")
    document["ai-analsis"] = {"summary": "typo"}
    (posts_root / "sample_161" / POST_FILE_NAME).write_text(
        json.dumps(document, indent=2), encoding="utf-8"
    )

    report = validate_posts(root=posts_root)

    assert report.ok
    assert any(
        "ai-analsis" in issue.message for issue in report.warnings
    )


def test_invalid_interview_question_is_an_error(posts_root):
    create_post("sample_162", text="Question.", root=posts_root)

    document = read_document(posts_root / "sample_162")
    document["interview_questions"] = [
        {"question": "Why?", "type": "nonsense", "difficulty": "easy"}
    ]
    (posts_root / "sample_162" / POST_FILE_NAME).write_text(
        json.dumps(document, indent=2), encoding="utf-8"
    )

    issues = validate_post("sample_162", root=posts_root)

    assert any(
        "interview_questions[0]" in issue.message for issue in issues
    )


def test_bad_classification_types_are_errors(posts_root):
    create_post("sample_163", text="Question.", root=posts_root)

    document = read_document(posts_root / "sample_163")
    document["classification"]["secondary_topics"] = "Delta Lake"
    document["classification"]["interview_relevant"] = "yes"
    (posts_root / "sample_163" / POST_FILE_NAME).write_text(
        json.dumps(document, indent=2), encoding="utf-8"
    )

    issues = validate_post("sample_163", root=posts_root)

    assert sum(1 for issue in issues if issue.is_error) == 2


def test_import_refuses_to_write_an_unusable_post(posts_root, tmp_path):
    """
    A rejected import must leave the posts tree exactly as it was,
    rather than half-ingesting a capture.
    """

    bundle = make_capture(
        tmp_path,
        text=None,
        post_json={
            "id": "sample_170",
            "source": {"platform": "", "captured_at": None},
            "original_text": "Text.",
            "interview_questions": [
                {"question": "Why?", "type": "nonsense"}
            ],
        },
    )

    with pytest.raises(InvalidPostError):
        import_post("sample_170", bundle, root=posts_root)

    assert not (posts_root / "sample_170" / POST_FILE_NAME).exists()


# ---------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------


def test_discovery_finds_every_post(posts_root):
    create_post("sample_180", text="One.", root=posts_root)
    create_post("sample_181", text="Two.", root=posts_root)

    found = {post.post_id for post in discover_posts(posts_root)}

    assert found == {"sample_180", "sample_181"}


def test_discovery_ignores_hidden_directories(posts_root):
    create_post("sample_182", text="One.", root=posts_root)

    (posts_root / ".staging").mkdir()
    (posts_root / "_wip").mkdir()

    found = [post.post_id for post in discover_posts(posts_root)]

    assert found == ["sample_182"]


def test_discovery_reports_a_directory_without_post_json(posts_root):
    (posts_root / "sample_183").mkdir()

    posts = discover_posts(posts_root)

    assert len(posts) == 1
    assert posts[0].has_post_file is False


def test_discovery_of_a_missing_root_is_empty(tmp_path):
    assert discover_posts(tmp_path / "absent") == []


def test_discovery_reports_media_counts(posts_root, tmp_path):
    create_post("sample_184", text="One.", root=posts_root)

    shot = tmp_path / "shot.png"
    shot.write_bytes(PNG_BYTES)

    add_media("sample_184", [shot], root=posts_root)

    summary = discover_posts(posts_root)[0]

    assert summary.declared_media == 1
    assert summary.media_count == 1
    assert summary.media_names == ("shot.png",)
    assert summary.text_length == len("One.")


# ---------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------


def test_cli_new_then_add_media_then_validate(posts_root, tmp_path):
    notes = tmp_path / "notes.md"
    notes.write_text("A captured question.", encoding="utf-8")

    shot = tmp_path / "shot.png"
    shot.write_bytes(PNG_BYTES)

    assert cli_main(
        [
            "--root",
            str(posts_root),
            "new",
            "sample_190",
            "--text-file",
            str(notes),
            "--primary-topic",
            "Databricks",
            "--interview-relevant",
        ]
    ) == 0

    assert cli_main(
        [
            "--root",
            str(posts_root),
            "add-media",
            "sample_190",
            str(shot),
            "--description",
            "A diagram",
        ]
    ) == 0

    assert cli_main(["--root", str(posts_root), "validate"]) == 0

    document = read_document(posts_root / "sample_190")

    assert document["original_text"] == "A captured question."
    assert document["media"][0]["description"] == "A diagram"
    assert document["classification"]["interview_relevant"] is True


def test_cli_import_round_trip(posts_root, tmp_path):
    bundle = make_capture(
        tmp_path, text="Question.", media={"shot.png": PNG_BYTES}
    )

    assert cli_main(
        [
            "--root",
            str(posts_root),
            "import",
            "sample_191",
            str(bundle),
            "--author",
            "Interviewer",
        ]
    ) == 0

    assert cli_main(["--root", str(posts_root), "list"]) == 0

    assert cli_main(["--root", str(posts_root), "validate"]) == 0


def test_cli_reports_a_failing_validate(posts_root):
    directory = posts_root / "sample_192"
    directory.mkdir()

    (directory / POST_FILE_NAME).write_text(
        json.dumps({"id": "sample_192"}), encoding="utf-8"
    )

    assert cli_main(["--root", str(posts_root), "validate"]) == 1


def test_cli_validate_one_post(posts_root):
    create_post("sample_193", text="Question.", root=posts_root)

    assert cli_main(
        ["--root", str(posts_root), "validate", "sample_193"]
    ) == 0


def test_cli_reports_an_ingestion_error(posts_root, capsys):
    assert cli_main(
        ["--root", str(posts_root), "new", "Not A Valid Id"]
    ) == 1

    assert "INGESTION_ERROR" in capsys.readouterr().err


def test_cli_without_a_command_prints_help(capsys):
    assert cli_main([]) == 0

    assert "usage:" in capsys.readouterr().out


# ---------------------------------------------------------------------
# Compatibility with what is already committed
# ---------------------------------------------------------------------


def test_committed_posts_are_all_valid():
    """
    The posts already in the repository must keep validating, or the
    pipeline would fail on a change that touched nothing.
    """

    report = validate_posts(root=REAL_POSTS_ROOT)

    assert report.ok, [str(issue) for issue in report.issues]

    # The samples are the committed fixtures. Posts collected from an
    # authorized source live alongside them and must validate too, so
    # the assertion is a superset check rather than an exact one.
    assert {"sample_001", "sample_002", "sample_003"} <= {
        post.post_id for post in report.posts
    }


def test_collected_posts_keep_their_provenance():
    """
    A post collected from a source must remain traceable to where it
    came from, and must not claim provenance it does not have.
    """

    identifiers = [
        summary.post_id
        for summary in discover_posts(REAL_POSTS_ROOT)
        if summary.post_id.startswith("urn-li-")
    ]

    if not identifiers:
        pytest.skip("no collected posts are present in this checkout")

    for identifier in identifiers:
        post = load_post(REAL_POSTS_ROOT / identifier)

        assert post.source.platform == "linkedin"
        assert post.source.captured_at
        assert post.source.url
        assert post.original_text.strip()

        # The identifier is the source permalink URN, slugged so it is
        # a safe directory name. That keeps it stable across runs and
        # traceable back to the original.
        assert post.id.startswith("urn-li-")
        assert "urn:li:" in post.source.url
        assert post.source.url.endswith("/")


def test_collected_posts_carry_no_credential_shaped_text():
    """
    Collected text comes from a page rendered by a browser that was
    signed in with local credentials. Nothing in a stored post may
    carry one of those values.
    """

    import os

    username = os.environ.get("LINKEDIN_USERNAME", "")
    password = os.environ.get("LINKEDIN_PASSWORD", "")

    if not username and not password:
        pytest.skip("no credentials are configured in this environment")

    blob = ""

    for path in REAL_POSTS_ROOT.rglob("*.json"):
        blob += path.read_text(encoding="utf-8")

    for secret in (username, password):
        if secret:
            assert secret not in blob


def test_a_published_date_reaches_the_document(tmp_path):
    """
    The date the source rendered is provenance the collector read, so it
    must survive into the stored post rather than being dropped between
    the source and the document.
    """

    from src.ingestion.sources.base import CollectedPost

    collected = CollectedPost(
        source_post_id="urn:li:activity:4242",
        text="A post about bronze medallion architecture.",
        url="https://www.linkedin.com/feed/update/urn:li:activity:4242/",
        published_at="Jan 15, 2025",
        author="someone",
    )

    document = collected.to_document(
        post_id="urn-li-activity-4242",
        platform="linkedin",
    )

    assert document.source["published_at"] == "Jan 15, 2025"


def test_a_missing_published_date_stays_missing(tmp_path):
    """
    Nothing is invented. A post whose source rendered no date has none,
    rather than a fabricated timestamp.
    """

    from src.ingestion.sources.base import CollectedPost

    document = CollectedPost(
        source_post_id="urn:li:activity:99",
        text="A post with no rendered date.",
    ).to_document(post_id="urn-li-activity-99", platform="linkedin")

    assert document.source["published_at"] is None


def test_the_published_date_survives_a_refresh(tmp_path):
    """
    A re-collection that yields no date must not erase the one already
    recorded, exactly as a missing author is preserved.
    """

    from src.ingestion.importer import create_post

    create_post(
        root=tmp_path,
        post_id="urn-li-activity-77",
        text="Original body.",
        platform="linkedin",
        published_at="Feb 2, 2024",
    )

    directory = tmp_path / "urn-li-activity-77"

    from src.ingestion.importer import import_post
    from src.ingestion.post_document import PostDocument

    existing = PostDocument.load_file(directory / "post.json")
    existing.merge_source(published_at=None)
    existing.save(directory)

    reloaded = PostDocument.load_file(directory / "post.json")

    assert reloaded.source["published_at"] == "Feb 2, 2024"


def test_a_relative_published_date_is_kept_verbatim():
    """
    "2 days ago" cannot be resolved into a date without the capture
    time, and guessing would put a wrong timestamp into the knowledge
    base. It is stored as rendered.
    """

    from src.ingestion.sources.base import CollectedPost

    document = CollectedPost(
        source_post_id="urn:li:activity:88",
        text="A post from this week.",
        published_at="2 days ago",
    ).to_document(post_id="urn-li-activity-88", platform="linkedin")

    assert document.source["published_at"] == "2 days ago"


def test_every_collected_post_validates_against_the_schema():
    """The canonical schema is what the rest of the pipeline reads."""

    report = validate_posts(root=REAL_POSTS_ROOT)

    collected = [
        issue
        for issue in report.issues
        if "urn-li-" in str(issue)
    ]

    assert collected == []


@pytest.mark.parametrize(
    "post_id", ["sample_001", "sample_002", "sample_003"]
)
def test_committed_posts_still_load(post_id):
    """
    The loader is what the enrichment pipeline depends on, so an
    existing post must load exactly as it did before the ingestion
    layer existed.
    """

    post = load_post(REAL_POSTS_ROOT / post_id)

    assert post.id == post_id
    assert post.original_text
    assert post.source.platform == "linkedin"
    assert post.classification.domain == "Data Engineering"


def test_committed_media_is_still_discovered():
    """
    sample_001 ships media that post.json does not declare. Discovery
    of undeclared files is the behaviour the worker already relied on.
    """

    post = load_post(REAL_POSTS_ROOT / "sample_001")

    names = {Path(item.path).name for item in post.media}

    assert "databricks_dashboard.png" in names
    assert "interview_notes.pdf" in names
    assert {item.type for item in post.media} == {"image", "pdf"}


def test_committed_media_survives_worker_processing(tmp_path):
    """
    The worker processes media before enriching. Declared and discovered
    media must both survive that step, so the ingestion layer cannot
    quietly drop an attachment.

    The post is copied first: processing writes rendered PDF pages next
    to the media, and a test must never write into the checkout.
    """

    import shutil

    from src.processing.media_processor import process_media

    directory = tmp_path / "sample_001"

    shutil.copytree(REAL_POSTS_ROOT / "sample_001", directory)

    post = load_post(directory)
    processed = process_media(post)

    assert len(processed.media) == 2

    by_type = {item.type: item for item in processed.media}

    assert by_type["image"].description
    assert by_type["pdf"].extracted_text is not None


def test_generated_media_directories_are_not_ingested():
    """
    The worker renders PDF pages into subdirectories of media/. Those
    are derived output and must not be treated as new media.
    """

    post = load_post(REAL_POSTS_ROOT / "sample_001")

    assert all(
        Path(item.path).parent.name == MEDIA_DIRECTORY_NAME
        for item in post.media
    )


def test_post_document_round_trips_a_hand_edited_file(tmp_path):
    """
    A human editing post.json by hand, and the layer re-reading it, must
    agree.
    """

    directory = tmp_path / "post"

    PostDocument.new(
        "sample_200", text="Text."
    ).save(directory)

    raw = read_document(directory)
    raw["interview_questions"] = [
        {
            "question": "Explain Z-Ordering.",
            "type": "scenario",
            "difficulty": "hard",
        }
    ]
    (directory / POST_FILE_NAME).write_text(
        json.dumps(raw, indent=2), encoding="utf-8"
    )

    document = PostDocument.load(directory)

    assert document.post_id == "sample_200"
    assert document.original_text == "Text."

    # A hand-written question list is content an import must keep, so
    # the document counts as already carrying more than a scaffold.
    assert document.has_enrichment() is True

    document.set_original_text("New text.")
    document.save(directory)

    assert load_post(directory).original_text == "New text."
    assert len(load_post(directory).interview_questions) == 1


def test_a_scaffold_is_not_reported_as_enriched(tmp_path):
    document = PostDocument.new("sample_202", text="Text.")

    assert document.has_enrichment() is False

    document.data["ai_analysis"]["summary"] = "A summary."

    assert document.has_enrichment() is True


def test_saving_a_post_leaves_no_temporary_file(tmp_path):
    directory = tmp_path / "post"

    PostDocument.new("sample_201", text="Text.").save(directory)

    assert sorted(
        path.name for path in directory.iterdir()
    ) == [POST_FILE_NAME]


def test_root_is_accepted_after_the_command(posts_root):
    """
    Both spellings are natural to type, and the pipeline uses the one
    that comes after the subcommand.
    """

    create_post("sample_210", text="Question.", root=posts_root)

    assert cli_main(["--root", str(posts_root), "list"]) == 0
    assert cli_main(["list", "--root", str(posts_root)]) == 0
    assert cli_main(["validate", "--root", str(posts_root)]) == 0
    assert cli_main(
        ["validate", "--root", str(posts_root), "sample_210"]
    ) == 0


def test_root_after_the_command_is_the_one_that_is_used(
    tmp_path, capsys
):
    """
    Without SUPPRESS defaults on the subparser, the second `--root`
    would be silently dropped and the wrong tree would be read.
    """

    first = tmp_path / "first"
    second = tmp_path / "second"

    first.mkdir()
    second.mkdir()

    create_post("sample_211", text="One.", root=first)

    assert cli_main(["list", "--root", str(second)]) == 0

    output = capsys.readouterr().out

    assert "No posts under" in output
    assert "sample_211" not in output


def test_new_post_reads_piped_text(posts_root, monkeypatch):
    """
    `cat notes.md | ... new <id>` is a natural way to add a capture, so
    piped text is used when no text was given.
    """

    import io

    monkeypatch.setattr(
        "src.ingestion.cli.sys.stdin",
        io.StringIO("A piped capture.\n"),
    )

    assert cli_main(
        ["--root", str(posts_root), "new", "sample_220"]
    ) == 0

    assert read_document(posts_root / "sample_220")[
        "original_text"
    ] == "A piped capture."


def test_new_post_does_not_read_an_interactive_terminal(
    posts_root, monkeypatch
):
    class Terminal:
        def isatty(self) -> bool:
            return True

        def read(self) -> str:
            raise AssertionError("a terminal must never be read")

    monkeypatch.setattr(
        "src.ingestion.cli.sys.stdin", Terminal()
    )

    # The empty capture is refused, and the refusal is reported
    # rather than raised, so nothing is written.
    assert cli_main(
        ["--root", str(posts_root), "new", "sample_221"]
    ) == 1

    assert not (posts_root / "sample_221").exists()


def test_an_unreadable_stream_is_treated_as_empty(
    posts_root, monkeypatch
):
    class Broken:
        def isatty(self) -> bool:
            return False

        def read(self) -> str:
            raise OSError("closed")

    monkeypatch.setattr(
        "src.ingestion.cli.sys.stdin", Broken()
    )

    assert cli_main(
        ["--root", str(posts_root), "new", "sample_222"]
    ) == 1

    assert not (posts_root / "sample_222").exists()
