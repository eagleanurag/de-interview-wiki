"""
The Gemini package, as a source this project can read.

Five files produced by a machine reading pictures, imported as derived
evidence about media this project already holds. Not the posts, not a
human's transcription, and not a replacement for either.

Three things this package is careful about, each of which cost
something to establish.

**The package's own prose is boilerplate.** Every post in
``LINKEDIN_KNOWLEDGE_ARCHIVE_COMPLETE.md`` carries the identical two
sentences under "Visual Description" and "Source-Derived Explanation".
They describe the method, not any slide, so they are recognised and
dropped rather than stored. Keeping them would put a sentence about
policy into the knowledge base once per image.

**Its "answers" are the transcription repeated.** The interview question
bank's answer field is the slide's OCR echoed back. It is kept as
``source_excerpt`` and never promoted to an answer.

**Its "questions" are heuristics over slide text.** Reading the bank
directly finds a bare YouTube link presented as a question and the
symbol debris ``cn oh? Azure`` presented as another. Every entry is
therefore a candidate carrying one of four verdicts, and only ACCEPT
and REWRITE reach the knowledge base.

The modules, in dependency order:

* :mod:`models` -- what a record is, and how far it can be trusted.
* :mod:`package_reader` -- the five files, parsed.
* :mod:`ocr_quality` -- what is observably true of a transcription.
* :mod:`crosscheck` -- the package against the real archive.
* :mod:`taxonomy` -- technology claims and the topic index.
* :mod:`questions` -- the candidate bank and its quality gate.
* :mod:`importer` -- build, then write, skipping when nothing changed.
* :mod:`bridge` -- transcriptions in the shape the pipeline consumes.
* :mod:`safety` -- what may not leave this machine.
"""

from __future__ import annotations

from src.gemini.importer import (
    DEFAULT_OUTPUT,
    Gap,
    GapReason,
    ImportResult,
    archive_fingerprint,
    build,
    load_candidates,
    load_gaps,
    load_records,
    read_manifest,
    run,
)
from src.gemini.models import (
    CandidateVerdict,
    CrossCheckReport,
    GeminiCandidateQuestion,
    GeminiImageRecord,
    GeminiPackageInfo,
    GeminiPostGroup,
    GeminiPostKnowledge,
    GeminiTopicIndex,
    ImportManifest,
    MatchVerdict,
    OcrQuality,
    OcrStatus,
    SourceKind,
)
from src.gemini.questions import accepted as accepted_questions
from src.gemini.safety import PublicOutputError, assert_public


__all__ = [
    "DEFAULT_OUTPUT",
    "CandidateVerdict",
    "CrossCheckReport",
    "Gap",
    "GapReason",
    "GeminiCandidateQuestion",
    "GeminiImageRecord",
    "GeminiPackageInfo",
    "GeminiPostGroup",
    "GeminiPostKnowledge",
    "GeminiTopicIndex",
    "ImportManifest",
    "ImportResult",
    "MatchVerdict",
    "OcrQuality",
    "OcrStatus",
    "PublicOutputError",
    "SourceKind",
    "accepted_questions",
    "archive_fingerprint",
    "assert_public",
    "build",
    "load_candidates",
    "load_gaps",
    "load_records",
    "read_manifest",
    "run",
]