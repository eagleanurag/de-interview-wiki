from src.ingestion.post_loader import load_post
from src.processing.media_processor import process_media


post = load_post("data/posts/sample_001")

process_media(post)

print(f"Media count: {len(post.media)}")

for media in post.media:
    print()
    print(f"TYPE: {media.type}")
    print(f"PATH: {media.path}")

    if media.description:
        print(f"DESCRIPTION: {media.description}")

    if media.extracted_text:
        print("EXTRACTED TEXT:")
        print(media.extracted_text)

print()
print("Unified media processing OK")