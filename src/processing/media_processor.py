from pathlib import Path

from src.models import KnowledgePost
from src.processing.image_processor import inspect_image
from src.processing.pdf_processor import (
    extract_pdf_text,
    render_pdf_pages,
)


def process_media(
    post: KnowledgePost,
    render_pdfs: bool = True,
) -> KnowledgePost:
    """
    Process all media attached to a KnowledgePost.

    Images:
        Collect technical metadata.

    PDFs:
        Extract selectable text.
        Optionally render pages for later AI vision analysis.
    """

    for media in post.media:

        media_path = Path(media.path)

        if media.type == "image":
            _process_image(media, media_path)

        elif media.type == "pdf":
            _process_pdf(
                media,
                media_path,
                render_pdfs=render_pdfs,
            )

    return post


def _process_image(media, image_path: Path) -> None:
    """Collect image metadata."""

    metadata = inspect_image(image_path)

    media.description = (
        f"{metadata['format']} image, "
        f"{metadata['width']}x{metadata['height']} pixels"
    )


def _process_pdf(
    media,
    pdf_path: Path,
    render_pdfs: bool,
) -> None:
    """Extract PDF text and optionally render pages."""

    extracted_text = extract_pdf_text(pdf_path)

    media.extracted_text = extracted_text

    if render_pdfs:
        render_directory = (
            pdf_path.parent
            / f"{pdf_path.stem}_pages"
        )

        render_pdf_pages(
            pdf_path,
            render_directory,
        )