from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class MediaItem(BaseModel):
    type: Literal["image", "pdf", "video", "other"]
    path: str
    description: str | None = None
    extracted_text: str | None = None


class SourceInfo(BaseModel):
    platform: str
    url: str | None = None
    captured_at: datetime
    author: str | None = None

    # When the source itself published the content, exactly as it
    # rendered it. A string rather than a datetime because a relative
    # form such as "2 days ago" is only resolvable against the capture
    # date, and a wrong timestamp would be worse than the original.
    published_at: str | None = None


class AIAnalysis(BaseModel):
    summary: str | None = None
    topics: list[str] = Field(default_factory=list)
    subtopics: list[str] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    image_descriptions: list[str] = Field(default_factory=list)


class InterviewQuestion(BaseModel):
    question: str
    type: Literal[
        "theory",
        "coding",
        "scenario",
        "architecture",
        "troubleshooting",
    ]
    difficulty: Literal["easy", "medium", "hard"]
    answer: str | None = None


class Classification(BaseModel):
    domain: str = "Data Engineering"
    primary_topic: str | None = None
    secondary_topics: list[str] = Field(default_factory=list)
    interview_relevant: bool = False


class KnowledgePost(BaseModel):
    id: str
    source: SourceInfo
    original_text: str = ""
    media: list[MediaItem] = Field(default_factory=list)

    ai_analysis: AIAnalysis = Field(default_factory=AIAnalysis)

    interview_questions: list[InterviewQuestion] = Field(
        default_factory=list
    )

    classification: Classification = Field(
        default_factory=Classification
    )