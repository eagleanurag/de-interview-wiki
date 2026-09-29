from __future__ import annotations

from src.ai.opencode import OpenCodeClient
from src.ai.schemas import AIEnrichmentResponse
from src.models import InterviewQuestion, KnowledgePost


class AIEnricher:
    """Enrich a KnowledgePost using OpenCode and Space Bunny."""

    def __init__(
        self,
        client: OpenCodeClient | None = None,
    ) -> None:
        self.client = client or OpenCodeClient()

    def enrich(self, post: KnowledgePost) -> KnowledgePost:
        prompt = self._build_prompt(post)

        result = self.client.run(prompt)

        enrichment = AIEnrichmentResponse.model_validate(
            result.data
        )

        post.ai_analysis.summary = enrichment.summary
        post.ai_analysis.topics = enrichment.topics
        post.ai_analysis.subtopics = enrichment.subtopics
        post.ai_analysis.concepts = [
            concept.name
            for concept in enrichment.concepts
        ]

        post.interview_questions = [
            InterviewQuestion(
                question=question.question,
                type="scenario",
                difficulty=_normalize_difficulty(
                    question.difficulty
                ),
                answer="\n".join(
                    question.what_strong_answers_cover
                ),
            )
            for question in enrichment.interview_questions
        ]

        return post

    @staticmethod
    def _build_prompt(post: KnowledgePost) -> str:
        media_text_parts: list[str] = []

        for media in post.media:
            if media.extracted_text:
                media_text_parts.append(
                    media.extracted_text
                )

        media_text = "\n\n".join(media_text_parts)

        return f"""
Analyze this Data Engineering knowledge post.

Return ONLY JSON matching this structure:

{{
  "summary": "string",
  "topics": ["string"],
  "subtopics": ["string"],
  "concepts": [
    {{
      "name": "string",
      "category": "string",
      "explanation": "string"
    }}
  ],
  "interview_questions": [
    {{
      "question": "string",
      "difficulty": "easy|medium|hard",
      "what_strong_answers_cover": ["string"]
    }}
  ]
}}

Generate high-quality interview-preparation knowledge.

Original content:
{post.original_text}

Extracted media text:
{media_text}
""".strip()


def _normalize_difficulty(value: str) -> str:
    value = value.lower().strip()

    if value in {"easy", "medium", "hard"}:
        return value

    return "medium"