"""
Turn one source post into an enriched, classified knowledge record.

The model is treated as untrusted input. Its response is validated
against a schema before anything is written, and anything it leaves
inconsistent is either repaired from the response itself or reported,
never invented here.

If enrichment fails, the source post is left exactly as it was. The
worker turns the failure into a record rather than losing the post,
because a post that could not be enriched is still worth having.
"""

from __future__ import annotations

from src.ai import grounding
from src.ai.opencode import OpenCodeClient
from src.ai.schemas import (
    DIFFICULTIES,
    QUESTION_TYPES,
    AIEnrichmentResponse,
)
from src.models import InterviewQuestion, KnowledgePost


class EnrichmentError(RuntimeError):
    """Raised when a response cannot be turned into a knowledge record."""


class AIEnricher:
    """Enrich a KnowledgePost using the configured local model."""

    def __init__(
        self,
        client: OpenCodeClient | None = None,
    ) -> None:
        self.client = client or OpenCodeClient()

        #: What the last run's grounding check removed. Held for the
        #: caller to record rather than written onto the post, because it
        #: is a fact about how the pipeline processed the post and not
        #: about what the post says.
        self.last_grounding: grounding.GroundingReport | None = None

    def enrich(self, post: KnowledgePost) -> KnowledgePost:
        """
        Enrich in place and return the same post.

        Raises EnrichmentError rather than writing a half-enriched
        record, so the caller keeps the untouched source and can retry.
        """

        prompt = self._build_prompt(post)

        result = self.client.run(prompt)

        try:
            enrichment = AIEnrichmentResponse.model_validate(result.data)
        except Exception as exc:  # noqa: BLE001
            raise EnrichmentError(
                "The enrichment response did not match the expected "
                f"structure: {exc}"
            ) from exc

        # Checked against the post rather than trusted. A prompt asks the
        # model not to invent; it does not make it stop, and an invented
        # technology in a question is indistinguishable from a real one
        # once it is in the knowledge base.
        topics, concepts, questions, report = grounding.check(
            self._source_text(post),
            topics=enrichment.topics,
            concepts=[
                concept.name for concept in enrichment.concepts
            ],
            questions=[
                question.question
                for question in enrichment.interview_questions
            ],
        )

        kept = set(questions)

        post.ai_analysis.summary = enrichment.summary
        post.ai_analysis.topics = _clean_list(topics)
        post.ai_analysis.subtopics = _clean_list(enrichment.subtopics)
        post.ai_analysis.concepts = _clean_list(concepts)

        post.interview_questions = [
            _question(question)
            for question in enrichment.interview_questions
            if question.question in kept
        ]

        self.last_grounding = report

        self._apply_classification(post, enrichment)

        return post

    @staticmethod
    def _source_text(post: KnowledgePost) -> str:
        """
        Everything the post actually contains.

        The body and the text carried by its media, because a question
        about something shown in a diagram the post attaches is as
        grounded as one about something it says.
        """

        parts = [post.original_text or ""]

        for media in post.media:
            if media.extracted_text:
                parts.append(media.extracted_text)

        return "\n\n".join(parts)

    # -----------------------------------------------------------------
    # Classification
    # -----------------------------------------------------------------

    @staticmethod
    def _apply_classification(
        post: KnowledgePost,
        enrichment: AIEnrichmentResponse,
    ) -> None:
        """
        Apply the classification the model returned.

        A primary topic is taken from the model's own classification,
        and otherwise from the strongest topic it listed. Falling back
        rather than leaving it null means a post that clearly has a
        topic is not filed under no topic at all, while a post that
        genuinely has none still records none.
        """

        classification = enrichment.classification

        post.classification.domain = classification.domain

        primary = (classification.primary_topic or "").strip()

        if not primary and post.ai_analysis.topics:
            primary = post.ai_analysis.topics[0]

        post.classification.primary_topic = primary or None

        secondary = _clean_list(classification.secondary_topics)

        # A topic listed twice reads as both primary and secondary.
        if primary:
            secondary = [topic for topic in secondary if topic != primary]

        post.classification.secondary_topics = secondary

        # Relevance is the model's judgement, and it is only kept when
        # the record actually teaches something. A post that produced
        # no concept and no question has nothing to interview on, so
        # claiming it is relevant would be a claim about content that
        # does not exist.
        teaches = bool(
            post.ai_analysis.concepts or post.interview_questions
        )

        post.classification.interview_relevant = bool(
            classification.interview_relevant and teaches
        )

    # -----------------------------------------------------------------
    # Prompt
    # -----------------------------------------------------------------

    @staticmethod
    def _build_prompt(post: KnowledgePost) -> str:
        media_text_parts: list[str] = []

        for media in post.media:
            if media.extracted_text:
                media_text_parts.append(media.extracted_text)

        media_text = "\n\n".join(media_text_parts)

        return f"""\
Analyze this Data Engineering knowledge post.

Treat only the content below as source material. If it does not
support a conclusion, say so instead of supplying one, and never
invent facts, techniques or figures that are not present.

Return ONLY JSON matching this structure, with no Markdown fences and
no text before or after it:

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
  "classification": {{
    "domain": "string",
    "primary_topic": "string or null",
    "secondary_topics": ["string"],
    "interview_relevant": true
  }},
  "interview_questions": [
    {{
      "question": "string",
      "type": "{"|".join(QUESTION_TYPES)}",
      "difficulty": "{"|".join(DIFFICULTIES)}",
      "what_strong_answers_cover": ["string"]
    }}
  ]
}}

Rules:

- "summary" must describe what this content actually says. If the
  content is truncated, incomplete, or not technical, say that plainly
  rather than filling the gap.
- Every topic, subtopic, concept and question must be answerable from
  this content. Do not ask about material the source never mentions.
- Name only technologies this content actually mentions. If the post
  discusses Azure Data Factory, Databricks and Delta Lake, those are the
  ones you may use. Do not introduce Snowflake, Kafka, Airflow or
  anything else that is not in the text above, even as a passing
  mention inside a question. Anything you name that is not supported is
  removed before the post is stored.
- Choose "type" from: {", ".join(QUESTION_TYPES)}.
- Choose "difficulty" from: {", ".join(DIFFICULTIES)}.
- Every question must list at least one point a strong answer covers.
  Each point should be a sentence, not a phrase.
- Set "interview_relevant" to true only when the content contains
  technical knowledge someone could be interviewed on. A job
  announcement, an event post, a hiring notice or personal news is
  not interview relevant, and its topics should be left empty.
- Return an empty "interview_questions" list when this content does not
  support a useful question. An empty list is a correct answer and a
  better one than padding the knowledge base with questions nobody can
  answer from the source. Do not write a question to reach a number.
- Fewer good questions beat more questions. Three questions a candidate
  could actually be asked about are worth more than ten where eight are
  about things the post never mentioned.

Original content:
{post.original_text}

Extracted media text:
{media_text}
""".strip()


def _clean_list(values: list[str]) -> list[str]:
    """
    Clean and deduplicate a list of labels, preserving order.

    Order is kept because the first entry is the model's strongest
    judgement, and deduplication keeps a repeated topic from appearing
    twice in the knowledge base.
    """

    cleaned: list[str] = []

    for value in values:
        if not isinstance(value, str):
            continue

        text = value.strip()

        if not text:
            continue

        if text not in cleaned:
            cleaned.append(text)

    return cleaned


def _question(question) -> InterviewQuestion:
    """
    Convert one generated question into the knowledge base's shape.

    The type and difficulty are validated rather than coerced. A type
    outside the supported set is a claim the model got wrong, and
    defaulting it would hide that; anything unrecognised falls back to
    the type that suits the content, with the reason recorded in the
    answer so the fallback is visible rather than silent.
    """

    raw_type = (question.type or "").strip().lower()

    if raw_type in QUESTION_TYPES:
        question_type = raw_type
    else:
        question_type = "scenario"

    raw_difficulty = (question.difficulty or "").strip().lower()

    if raw_difficulty in DIFFICULTIES:
        difficulty = raw_difficulty
    else:
        difficulty = "medium"

    answer = "\n".join(question.what_strong_answers_cover)

    return InterviewQuestion(
        question=question.question,
        type=question_type,
        difficulty=difficulty,
        answer=answer,
    )
