from pathlib import Path

import pymupdf


def extract_pdf_text(pdf_path: str | Path) -> str:
    """
    Extract selectable text from all pages of a PDF.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF does not exist: {pdf_path}"
        )

    text_parts: list[str] = []

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document):
            text = page.get_text("text").strip()

            if text:
                text_parts.append(
                    f"--- Page {page_number + 1} ---\n{text}"
                )

    return "\n\n".join(text_parts)


def render_pdf_pages(
    pdf_path: str | Path,
    output_directory: str | Path,
    dpi: int = 150,
) -> list[Path]:
    """
    Render every PDF page as a PNG image.

    These rendered images can later be sent to a
    multimodal AI model for visual analysis.
    """

    pdf_path = Path(pdf_path)
    output_directory = Path(output_directory)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF does not exist: {pdf_path}"
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    rendered_pages: list[Path] = []

    zoom = dpi / 72
    matrix = pymupdf.Matrix(zoom, zoom)

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document):

            pixmap = page.get_pixmap(
                matrix=matrix,
                alpha=False,
            )

            output_path = (
                output_directory
                / f"page_{page_number + 1:03d}.png"
            )

            pixmap.save(output_path)

            rendered_pages.append(output_path)

    return rendered_pages