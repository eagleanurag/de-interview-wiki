from src.ai.enricher import AIEnricher
from src.ingestion.post_loader import load_post
from src.processing.media_processor import process_media


def main() -> None:
    post = load_post("data/posts/sample_001")

    process_media(post)

    enricher = AIEnricher()

    enriched_post = enricher.enrich(post)

    print("=== KNOWLEDGE POST ENRICHMENT ===")
    print(f"ID: {enriched_post.id}")
    print()
    print("Summary:")
    print(enriched_post.ai_analysis.summary)
    print()
    print("Topics:")
    print(enriched_post.ai_analysis.topics)
    print()
    print(
        f"Concepts: "
        f"{len(enriched_post.ai_analysis.concepts)}"
    )
    print(
        f"Questions: "
        f"{len(enriched_post.interview_questions)}"
    )

    assert enriched_post.ai_analysis.summary
    assert enriched_post.ai_analysis.topics
    assert enriched_post.ai_analysis.concepts
    assert enriched_post.interview_questions

    print()
    print("KnowledgePost enrichment OK")


if __name__ == "__main__":
    main()