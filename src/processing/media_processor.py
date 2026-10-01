"""
Process the media attached to a post.

One unreadable file must cost that file, not the post and not the run.
A truncated PDF, a screenshot that is not really an image and a file
that vanished between discovery and processing are all ordinary events
in a system that ingests whatever a user drops in, so each is recorded
and stepped over.

What is never done here is inventing. A file that could not be read
records why, and carries no extracted text, rather than carrying text
that was never in it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.models import KnowledgePost, MediaItem as MediaEntry
from src.processing.image_processor import inspect_image
from src.processing.pdf_processor import (
    extract_pdf_text,
    render_pdf_pages,
)


@dataclass
class MediaOutcome:
    """What happened to one media file."""

    path: str
    outcome: str
    detail: str = ""

    @property
    def failed(self) -> bool:
        return self.outcome in {"failed", "missing"}


@dataclass
class MediaReport:
    """Every file's outcome, so a failure is visible rather than silent."""

    outcomes: list[MediaOutcome] = field(default_factory=list)

    @property
    def failed(self) -> list[MediaOutcome]:
        return [item for item in self.outcomes if item.failed]

    @property
    def processed(self) -> list[MediaOutcome]:
        return [item for item in self.outcomes if not item.failed]

    def summary(self) -> str:
        return (
            f"{len(self.processed)} processed, "
            f"{len(self.failed)} failed"
        )


def process_media(
    post: KnowledgePost,
    render_pdfs: bool = True,
    *,
    report: MediaReport | None = None,
    on_progress=None,
) -> KnowledgePost:
    """
    Process all media attached to a KnowledgePost.

    Images get their technical metadata. PDFs get their selectable text
    extracted, and optionally have their pages rendered for later vision
    analysis.

    Returns the post, unchanged in shape. A file that fails is recorded
    on ``report`` and, when one is supplied, described in its own
    description so the failure travels with the post rather than being
    lost.
    """

    outcomes = report if report is not None else MediaReport()

    for media in list(post.media):
        outcome = _process_one(
            post, media, render_pdfs=render_pdfs, on_progress=on_progress
        )

        outcomes.outcomes.append(outcome)

    return post


def _process_one(
    post: KnowledgePost,
    media: MediaEntry,
    *,
    render_pdfs: bool,
    on_progress,
) -> MediaOutcome:
    """
    Process one media entry, converting any failure into an outcome.

    The post's directory is where a declared path is resolved from, so
    a relative path is portable rather than dependent on the working
    directory of whichever runner loaded the post.
    """

    media_path = _resolve(post, media)

    if media_path is None:
        detail = (
            f"{media.path} is not present in the post directory"
        )
        _note(media, detail)

        return MediaOutcome(path=media.path, outcome="missing", detail=detail)

    try:
        if media.type == "image":
            _process_image(media, media_path)

        elif media.type == "pdf":
            _process_pdf(
                media, media_path, render_pdfs=render_pdfs,
                on_progress=on_progress,
            )

        else:
            return MediaOutcome(
                path=media.path,
                outcome="skipped",
                detail=f"no processor for type {media.type!r}",
            )

    except Exception as exc:  # noqa: BLE001
        # A corrupt file is ordinary. Recording it keeps the post and
        # lets the rest of the run continue.
        detail = f"{type(exc).__name__}: {exc}"

        _note(media, detail)

        if on_progress:
            on_progress(f"Media failed {media.path}: {detail}")

        return MediaOutcome(
            path=media.path, outcome="failed", detail=detail
        )

    return MediaOutcome(
        path=media.path,
        outcome="processed",
        detail=media.description or "",
    )


def _resolve(post: KnowledgePost, media: MediaEntry) -> Path | None:
    """
    The file behind a declared media path.

    A declared path is relative to the post directory, which is what
    keeps the repository portable. An absolute path or one that escapes
    the directory is refused rather than followed.
    """

    declared = Path(media.path)

    if declared.is_absolute():
        return None

    base = Path(post.directory) if post.directory else Path(".")

    candidate = base / declared

    if candidate.is_file():
        return candidate

    # A post built in memory rather than loaded from disk has no
    # directory, so its path is already absolute or is relative to the
    # process. Falling back keeps the media stage usable in tests and in
    # a worker that builds a post from a bundle.
    fallback = declared

    return fallback if fallback.is_file() else None


def _note(media: MediaEntry, detail: str) -> None:
    """
    Record a failure on the media entry itself.

    Keeping the reason with the file means the wiki and any later run
    can show that the file was unreadable, instead of silently
    presenting a media entry with nothing behind it.
    """

    existing = (media.description or "").strip()

    note = f"Media could not be processed ({detail})."

    if note in existing:
        return

    media.description = f"{existing} {note}".strip()


def _process_image(media: MediaEntry, image_path: Path) -> None:
    """Collect image metadata."""

    metadata = inspect_image(image_path)

    media.description = (
        f"{metadata['format']} image, "
        f"{metadata['width']}x{metadata['height']} pixels"
    )


def _process_pdf(
    media: MediaEntry,
    pdf_path: Path,
    *,
    render_pdfs: bool,
    on_progress,
) -> None:
    """Extract PDF text and optionally render pages."""

    extracted_text = extract_pdf_text(pdf_path)

    media.extracted_text = extracted_text

    if not render_pdfs:
        return

    render_directory = pdf_path.parent / f"{pdf_path.stem}_pages"

    try:
        render_pdf_pages(pdf_path, render_directory)

    except Exception as exc:  # noqa: BLE001
        # Rendering is an aid to later vision analysis. Losing it does
        # not invalidate the text that was already extracted, so it is
        # reported and stepped over rather than failing the file.
        if on_progress:
            on_progress(
                f"Could not render pages for {media.path}: {exc}"
            )
