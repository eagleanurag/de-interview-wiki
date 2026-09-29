from src.ai.opencode import OpenCodeClient
from src.ai.schemas import AIEnrichmentResponse


PROMPT = """
Analyze the following Data Engineering interview content.

Return ONLY JSON with these fields:

{
  "summary": "string",
  "topics": ["string"],
  "subtopics": ["string"],
  "concepts": [
    {
      "name": "string",
      "category": "string",
      "explanation": "string"
    }
  ],
  "interview_questions": [
    {
      "question": "string",
      "difficulty": "easy|medium|hard",
      "what_strong_answers_cover": ["string"]
    }
  ]
}

Generate useful interview-preparation knowledge from the content.

Content:

You have a large Delta Lake table in Databricks.
A query against the table is taking too long.

How would you investigate and optimize the workload?

Consider:
- partitioning
- data skipping
- Z-Ordering
- Spark execution plans
- file sizes
- query execution metrics
""".strip()


def main() -> None:
    client = OpenCodeClient()

    result = client.run(PROMPT)

    print("=== AI ENRICHMENT TEST ===")
    print(f"Session: {result.session_id}")
    print()

    print("Raw response:")
    print(result.text)
    print()

    enrichment = AIEnrichmentResponse.model_validate(
        result.data
    )

    print("=== VALIDATED RESPONSE ===")
    print(f"Summary: {enrichment.summary}")
    print(f"Topics: {len(enrichment.topics)}")
    print(f"Subtopics: {len(enrichment.subtopics)}")
    print(f"Concepts: {len(enrichment.concepts)}")
    print(
        f"Interview questions: "
        f"{len(enrichment.interview_questions)}"
    )

    assert enrichment.summary
    assert enrichment.topics
    assert enrichment.concepts
    assert enrichment.interview_questions

    print()
    print("AI enrichment validation OK")


if __name__ == "__main__":
    main()