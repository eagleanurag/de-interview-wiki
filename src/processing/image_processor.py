from pathlib import Path

from PIL import Image


SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".gif",
    ".bmp",
}


def inspect_image(image_path: str | Path) -> dict:
    """
    Inspect an image and return normalized metadata.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image does not exist: {image_path}"
        )

    if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format: {image_path.suffix}"
        )

    with Image.open(image_path) as image:
        width, height = image.size
        image_format = image.format
        mode = image.mode

    file_size_bytes = image_path.stat().st_size

    return {
        "path": str(image_path),
        "filename": image_path.name,
        "format": image_format,
        "mode": mode,
        "width": width,
        "height": height,
        "file_size_bytes": file_size_bytes,
    }