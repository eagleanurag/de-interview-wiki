"""
The package's 180 candidate questions, and which of them are worth keeping.

This module exists because the candidate bank cannot be taken at face
value. Three of its entries, read directly from the file, are::

    Question: Describe the Explain the TASK
    Question: Link 1: https://www_youtube.com/watch?v=lfN8RDA7kVA&t=1090s
    Question: cn oh? Azure

The first is a two-column diagram whose OCR interleaved the columns. The
second is a link. The third is symbol debris that happened to pick up
one real technology name. Publishing these would put noise in front of
someone looking for something to study, and the package's own framing
promises as much: it says its questions come from "question-like or
instruction-like text captured from the uploaded slides", which is a
description of a heuristic, not of a curated set.

So every entry is a candidate, and each gets one of four verdicts.

**ACCEPT** -- readable, genuinely a question, traceable to a specific
slide. Used as written.

**REWRITE** -- the wording is damaged but the repair is recoverable
*and checkable*. The only repairs offered are listed in
:data:`REPAIRS`, and each one has to be verified against the source
transcription before it is accepted. "Make it sound better" is not a
repair; dropping an OCR-merged clause whose remainder appears verbatim
in the source is.

**REJECT** -- not a question, or damaged past recovery. Kept in full in
the imported record with its reason, because a rejected entry that
vanishes cannot be argued with.

**NEEDS_REVIEW** -- plausibly useful, but the intent cannot be
reconstructed without a person. This is the honest destination for
most of the column-merge damage, and it is a destination rather than a
failure: the original text and the source excerpt are both preserved so
a human has what they need.

The "Answer / source-bounded response" field is never used as an
answer. In the file it is the slide transcription repeated verbatim, so
promoting it would put a copy of the OCR in an answer field and call it
knowledge. It is kept as ``source_excerpt`` and nothing else.
"""

from __future__ import annotations

import hashlib
import re
from typing import Callable

from src.gemini.models import (
    CandidateVerdict,
    GeminiCandidateQuestion,
    GeminiImageRecord,
    OcrQuality,
)
from src.gemini.ocr_quality import assess, as_search_text, sentence_share
from src.gemini.package_reader import QUESTION_MD
from src.redaction import redact_text


#: How the bank labels its own entries.
_ENTRY = re.compile(r"^#{2,3}\s+Question\s+(?P<index>\d+)\s*$", re.IGNORECASE)

_FIELD = re.compile(r"^\*\*(?P<label>[^*]+):\*\*\s*(?P<value>.*)$")

_SOURCE_POST = re.compile(r"(?P<group>POST-\d+)", re.IGNORECASE)
_SOURCE_SLIDES = re.compile(
    r"slides?\s+(?P<first>\d+)"
    r"(?:\s*(?:,|and|&)\s*(?P<rest>[\d,\sand&]*))?",
    re.IGNORECASE,
)

_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

#: Words that make a line an interview question or an instruction to
#: answer one. Deliberately a closed list rather than a pattern: an
#: open-ended pattern matches whatever noise happens to contain a
#: question mark, and the package's '?' is very often OCR debris rather
#: than a mark the writer chose.
OPENERS = frozenset(
    {
        "what", "why", "how", "when", "where", "which", "who", "whom",
        "whose",
        "describe", "explain", "list", "define", "compare", "contrast",
        "discuss", "outline", "differentiate", "distinguish", "identify",
        "write", "state", "illustrate", "walkthrough", "calculate",
        "compute", "solve", "design", "recommend", "elaborate", "expand",
        "summarize", "summarise", "tell", "show", "give", "provide",
    }
)

#: Prefixes that are labels rather than question text.
_LABEL_PREFIX = re.compile(
    r"^\s*(?:q(?:uestion)?\.?\s*#?\s*\d+\s*[:.)\-]\s*)", re.IGNORECASE
)

#: A run of symbols at the end of a line: OCR ruling lines, table edges,
#: bullet glyphs that survived as text.
_TRAILING_SYMBOLS = re.compile(r"[\s|_=\-~*·•.,:;!?/\\<>^#@%\[\]{}()\"'`]+$")

#: A URL together with any words that are only there to introduce it.
_LINK_PREFIX = re.compile(
    r"^\s*(?:link|reference|source|url|read more|see)\s*"
    r"\d*\s*[:.\-–]?\s*",
    re.IGNORECASE,
)

_WORD = re.compile(r"[^\W\d_]{2,}", re.UNICODE)

#: Below this, the text is debris. Set where the two observed cases sit:
#: a coherent transcription is far above it, and ``cn oh? Azure`` is
#: below it in content though not in letter share -- which is why
#: letter share alone is never the only test.
_MIN_USEFUL_WORDS = 3


def _normalise(text: str) -> str:
    """Lowercased, punctuation-collapsed, for containment tests only."""

    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()


def _words(text: str) -> list[str]:
    return _WORD.findall(text)


def openers_in(text: str) -> list[int]:
    """Where question wording begins, as word indices."""

    found: list[int] = []

    for index, word in enumerate(_words(text.lower())):
        if word in OPENERS:
            found.append(index)

    return found


def is_question_like(text: str) -> tuple[bool, str]:
    """
    Whether this reads as a question, and why.

    A recognised opener is the evidence. A question mark alone is not
    accepted as evidence, because in this package the mark frequently
    arrives attached to noise -- ``cn oh? Azure`` has one -- and treating
    it as a question would admit precisely the fragments the quality
    gate exists to exclude. A bare mark is allowed only alongside enough
    words to be a sentence.
    """

    body = text.strip()

    if not body:
        return False, "there is no text"

    found = openers_in(body)

    if found:
        word = _words(body.lower())[found[0]]

        return True, f"question wording begins with {word!r}"

    words = _words(body)

    if body.rstrip().endswith("?") and len(words) >= 4:
        return True, (
            f"ends with a question mark and carries {len(words)} words"
        )

    if body.rstrip().endswith("?"):
        return False, (
            "the question mark is attached to too few words to be a "
            "question; in this package that mark is usually OCR debris"
        )

    return False, "no interrogative or instructional wording anywhere in it"


def looks_like_link(text: str) -> bool:
    """
    Whether the text is *only* a link, with nothing else in it.

    The word counts as well as the URL, and that distinction matters.
    ``Link 1: https://...`` is a link and is rejected. But a slide line
    reading ``Explain the MERGE INTO statement in this clause.
    https://...`` is a question that happens to end in a link, and
    rejecting it would discard a usable question while
    :func:`_repair_trailing_link` -- which drops exactly that link, and
    verifies the remainder against the source -- could not run.

    So this asks whether anything is left once the link is removed, not
    whether a link is present. Checking for presence would make the
    trailing-link repair unreachable, since the repair is only ever
    offered for text that contains a link.
    """

    body = _LINK_PREFIX.sub("", text.strip())

    if not _URL.search(body):
        return False

    return len(_words(_URL.sub(" ", body))) < _MIN_USEFUL_WORDS


def _supported_by(candidate: str, source: str) -> bool:
    """
    Whether a proposed rewrite is already present in the source.

    The single rule that keeps rewriting honest. A rewrite must be a
    narrowing of what the slide actually said, not an improvement to it,
    and the only way to tell those apart without a model is to require
    the text to be there. Checking against the slide transcription
    rather than against the candidate's own text matters: the
    candidate is damaged, so agreement with itself proves nothing.
    """

    if not source:
        return False

    return _normalise(candidate) in _normalise(source)


# ---------------------------------------------------------------------
# Repairs
# ---------------------------------------------------------------------
#
# Each is conservative, each states what it removed, and each is used
# only if its output is verified against the source. They are tried in
# order and the first that verifies wins; a repair that does not verify
# is not attempted again in weakened form.


def _repair_trailing_link(text: str, source: str) -> tuple[str, str] | None:
    """Drop a trailing link, when the words before it stand alone."""

    match = _URL.search(text)

    if not match:
        return None

    kept = text[: match.start()].strip()

    if len(_words(kept)) < _MIN_USEFUL_WORDS:
        return None

    if not is_question_like(kept)[0]:
        return None

    if not _supported_by(kept, source):
        return None

    return kept, f"dropped the trailing link ({len(text) - len(kept)} characters)"


def _repair_label_prefix(text: str, source: str) -> tuple[str, str] | None:
    """Drop a leading ``Q3:``-style label."""

    kept = _LABEL_PREFIX.sub("", text).strip()

    if kept == text.strip():
        return None

    if len(_words(kept)) < _MIN_USEFUL_WORDS:
        return None

    if not _supported_by(kept, source):
        return None

    return kept, "dropped a leading question-number label"


def _repair_trailing_symbols(
    text: str, source: str
) -> tuple[str, str] | None:
    """
    Drop OCR ruling lines and bullet glyphs from the end.

    Requires at least two characters to have been removed, which is what
    separates debris from punctuation. Without that floor, trimming the
    single ``?`` off the end of an ordinary question counted as a
    "repair", every clean question came back REWRITE with a note about
    OCR ruling lines that were never there.
    """

    kept = _TRAILING_SYMBOLS.sub("", text).strip()

    if kept == text.strip():
        return None

    if len(text.strip()) - len(kept) < 2:
        return None

    if len(_words(kept)) < _MIN_USEFUL_WORDS:
        return None

    if not _supported_by(kept, source):
        return None

    return kept, "dropped trailing OCR ruling and bullet glyphs"


def _repair_merged_clause(text: str, source: str) -> tuple[str, str] | None:
    """
    Reduce a column merge to the one clause that survives in the source.

    ``Describe the Explain the TASK`` is two columns of a diagram run
    together: one column says *describe the situation*, the other says
    *explain the task*. Both openers are genuine, so neither can be
    dismissed as noise. What settles it is the source: only one of the
    two readings appears verbatim in the slide transcription.

    The rewrite is therefore the last clause, and it is accepted only if
    the source contains it. In the example the source does not -- the
    OCR interleaved the columns there too, so ``explain the task that
    needed to be done`` is nowhere in it -- and the honest result is
    NEEDS_REVIEW rather than a confident guess at what the speaker was
    going to say.

    Gated on the text containing no sentence punctuation at all. That is
    the observable signal that this is a column merge rather than an
    ordinary question: a real question is a sentence and carries a mark,
    while ``Describe the Explain the TASK`` is two diagram cells fused
    with nothing between them. Without the gate a well-formed question
    containing two interrogatives -- *"What is an INNER JOIN and when
    should you use one?"* -- would be eligible, and the only thing
    stopping it would be the source check happening to fail.
    """

    if sentence_share(text) > 0.0:
        return None

    found = openers_in(text)

    if len(found) < 2:
        return None

    words = _words(text)

    last = found[-1]

    kept_words = words[last:]

    if len(kept_words) < _MIN_USEFUL_WORDS:
        return None

    kept = " ".join(kept_words)

    if not is_question_like(kept)[0]:
        return None

    if not _supported_by(kept, source):
        return None

    return kept, (
        "OCR ran two diagram columns together; kept the final clause, "
        "which appears verbatim in the source transcription"
    )


REPAIRS: tuple[Callable[[str, str], "tuple[str, str] | None"], ...] = (
    _repair_trailing_link,
    _repair_label_prefix,
    _repair_merged_clause,
    _repair_trailing_symbols,
)


def _tidy(text: str) -> str:
    """
    Whitespace collapsed and the first letter capitalised. Nothing else.

    Deliberately minimal. Cleaning a question means changing its wording,
    and every change beyond collapsing runs of spaces is a change the
    source may not support.
    """

    body = as_search_text(text)

    if not body:
        return body

    return body[0].upper() + body[1:]


# ---------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------


#: The labels this parser understands, mapped to the short names the rest
#: of the module uses.
#:
#: The bank writes ``Answer / source-bounded response``, not ``Answer``.
#: Matching on the file's own spelling rather than normalising it meant
#: every lookup for the answer came back empty and every candidate
#: silently lost its source excerpt -- no error, just nothing there.
_FIELD_ALIASES = {
    "question": "question",
    "type": "type",
    "category": "type",
    "difficulty": "difficulty",
    "answer": "answer",
    "answer / source-bounded response": "answer",
    "response": "answer",
    "source": "source",
    "sources": "source",
}


def _canonical_label(label: str) -> str:
    """The short name for a field label as the bank spells it."""

    key = " ".join(str(label or "").split()).lower()

    return _FIELD_ALIASES.get(key, key)


class RawEntry:
    """One candidate, exactly as the bank wrote it."""

    __slots__ = ("index", "fields")

    def __init__(self, index: int) -> None:
        self.index = index
        self.fields: dict[str, list[str]] = {}

    def get(self, label: str) -> str:
        return "\n".join(
            self.fields.get(_canonical_label(label), [])
        ).strip()

    def set(self, label: str, value: str) -> None:
        self.fields.setdefault(_canonical_label(label), []).append(value)

    def labels(self) -> list[str]:
        """The field names present, canonicalised and sorted."""

        return sorted(self.fields)


def parse_bank(contents: dict[str, str]) -> list[RawEntry]:
    """
    The candidate list, unmodified.

    Field values are collected across lines rather than taken from one
    line, because an OCR block that broke across a line break is still
    one field, and truncating at the newline is how a two-line question
    turns into a fragment.
    """

    body = contents.get(QUESTION_MD, "")

    entries: list[RawEntry] = []

    current: RawEntry | None = None

    label: str | None = None

    for line in body.splitlines():
        stripped = line.strip()

        heading = _ENTRY.match(stripped)

        if heading:
            current = RawEntry(int(heading.group("index")))

            entries.append(current)

            label = None

            continue

        if current is None:
            continue

        match = _FIELD.match(stripped)

        if match:
            label = match.group("label").strip().lower()

            value = match.group("value").strip()

            if value:
                current.set(label, value)

            continue

        if label and stripped:
            current.set(label, stripped)

        elif label and not stripped:
            # A blank line does **not** end the field.

            # The bank's answer field is a whole slide transcription and
            # those contain blank lines; resetting here truncated every
            # answer at its first paragraph, silently, and the tail was
            # simply gone. The field ends at the next ``**Label:**``, at
            # the ``---`` between entries, or at the next heading, and
            # ``RawEntry.get`` strips the trailing blanks.

            current.set(label, "")

        elif stripped.startswith("---"):
            label = None

    return entries


def _declared_slides(text: str) -> list[int]:
    """The slide numbers a ``Source:`` line names, in order."""

    match = _SOURCE_SLIDES.search(text)

    if not match:
        return []

    numbers = [int(match.group("first"))]

    rest = match.group("rest") or ""

    numbers.extend(
        int(part) for part in re.findall(r"\d+", rest)
    )

    return numbers


def _resolve(
    entry: RawEntry,
    records: list[GeminiImageRecord],
) -> tuple[str, int | None, str, bool, str]:
    """
    Attach an entry to a specific image.

    The bank's source line says ``POST-002, slide 0``. The number is
    ambiguous by one: the knowledge archive prints ``## Slide 1`` for a
    file named ``slide_0``, so the package counts from zero in one place
    and from one in another. Both readings are tried against the
    inventory and the one that names a real file wins, and which reading
    won is recorded rather than assumed.

    Returns the filename, the slide number, the activity, whether a file
    was actually found, and the convention used.
    """

    source = entry.get("source")

    post_match = _SOURCE_POST.search(source)

    group = post_match.group("group").upper() if post_match else ""

    declared = _declared_slides(source)

    if not declared:
        return "", None, "", False, "unresolved"

    candidates = [
        record
        for record in records
        if record.post_id.upper() == group
    ]

    if not candidates:
        return "", None, "", False, "unresolved"

    by_number = {record.slide_number: record for record in candidates}

    for number in declared:
        if number in by_number:
            record = by_number[number]

            return (
                record.filename,
                record.slide_number,
                record.activity_id,
                True,
                "zero_based",
            )

    # The package printed a one-based heading while the inventory counts
    # from zero.
    for number in declared:
        if number - 1 in by_number:
            record = by_number[number - 1]

            return (
                record.filename,
                record.slide_number,
                record.activity_id,
                True,
                "one_based",
            )

    return "", None, "", False, "unresolved"


def _fingerprint(entry: RawEntry) -> str:
    digest = hashlib.sha256()

    for value in (
        str(entry.index),
        entry.get("question"),
        entry.get("source"),
    ):
        digest.update(value.encode("utf-8", "replace"))

        digest.update(b"\x1f")

    return digest.hexdigest()


def suspicious_merge(text: str) -> bool:
    """
    Whether the text looks like two diagram columns run together.

    Two genuine question openings with no sentence punctuation between
    them is the observable shape of a column merge: ``Describe the
    Explain the TASK`` is not a question anyone wrote, it is one cell of
    a diagram fused to another.

    Both conditions are needed. Two openers in a sentence -- *"What is
    an INNER JOIN and when should you use one?"* -- is ordinary English,
    and the punctuation is what tells the two apart. Neither condition
    alone is evidence, and the letter-share measure that
    :func:`assess` reports calls the merged text perfectly readable,
    because it *is* mostly letters. That is why this check exists
    separately rather than as another quality threshold.

    Used to withhold acceptance. A merged fragment must either narrow to
    something the source supports or wait for a person; publishing it as
    an interview question is the specific failure this gate prevents.
    """

    return len(openers_in(text)) >= 2 and sentence_share(text) == 0.0


def judge(
    entry: RawEntry,
    source: str,
) -> tuple[CandidateVerdict, str, str | None, str]:
    """
    Decide one candidate. Returns verdict, reason, question, rewrite note.

    The ladder runs from the cheapest disqualifications to the most
    expensive, so that a bare link is never subjected to the repair
    machinery, and a question that is already clean is not rewritten for
    the sake of it.
    """

    text = entry.get("question")

    if not text.strip():
        return (
            CandidateVerdict.REJECT,
            "the entry has no question text",
            None,
            "",
        )

    if looks_like_link(text):
        return (
            CandidateVerdict.REJECT,
            "the entry is a link, and a link is not an interview question",
            None,
            "",
        )

    words = _words(text)

    if len(words) < _MIN_USEFUL_WORDS:
        return (
            CandidateVerdict.REJECT,
            f"only {len(words)} word(s); too little to be a question",
            None,
            "",
        )

    question_like, why = is_question_like(text)

    if not question_like:
        return (
            CandidateVerdict.REJECT,
            f"not question-like: {why}",
            None,
            "",
        )

    quality, evidence = assess(text)

    if not source:
        return (
            CandidateVerdict.NEEDS_REVIEW,
            (
                "question-like, but its slide did not resolve to an "
                f"image, so the wording cannot be checked against "
                f"anything ({why}; OCR {evidence})"
            ),
            None,
            "",
        )

    if quality is OcrQuality.GARBLED:
        return (
            CandidateVerdict.NEEDS_REVIEW,
            (
                f"question-like, but the transcription is debris and the "
                f"intended wording cannot be reconstructed from it "
                f"({evidence})"
            ),
            None,
            "",
        )

    for repair in REPAIRS:
        outcome = repair(text, source)

        if outcome is None:
            continue

        cleaned, note = outcome

        return (
            CandidateVerdict.REWRITE,
            f"repaired from a damaged transcription ({note})",
            _tidy(cleaned),
            note,
        )

    if suspicious_merge(text):
        # Either a repair above narrowed it, or none could. What is left
        # is a fused fragment that happens to be mostly letters, which
        # the quality measure alone would have called readable.
        return (
            CandidateVerdict.NEEDS_REVIEW,
            (
                "two question openings with no punctuation between them, "
                "which is the shape OCR leaves when it runs two diagram "
                "columns together; no narrowing of it appears in the "
                "source transcription, so reconstructing it would be a "
                "guess"
            ),
            None,
            "",
        )

    if quality is not OcrQuality.READABLE:
        return (
            CandidateVerdict.NEEDS_REVIEW,
            (
                "question-like, but the transcription is too fragmentary "
                f"to rewrite safely without guessing ({evidence})"
            ),
            None,
            "",
        )

    return (
        CandidateVerdict.ACCEPT,
        f"readable and question-like ({why}; OCR {evidence})",
        _tidy(text),
        "",
    )


def build_candidates(
    contents: dict[str, str],
    records: list[GeminiImageRecord],
) -> list[GeminiCandidateQuestion]:
    """
    Every candidate, with its verdict. Nothing is dropped.

    A rejected entry is kept complete -- original text, source excerpt,
    group, slide, reason -- because a rejected entry that disappears
    cannot be argued with, and the counts are how anyone finds out
    whether the gate is too strict or too lenient.

    Only the slide's own transcription, looked up by filename, is used to
    *verify* a repair. The bank's answer field is not, and using it would
    make the check circular: the candidate was extracted from that same
    text, so any narrowing of it trivially "appears in the source" and
    the verification would approve anything. The answer is still kept as
    ``source_excerpt`` for a human to read.
    """

    by_filename = {record.filename: record for record in records}

    candidates: list[GeminiCandidateQuestion] = []

    for entry in parse_bank(contents):
        filename, slide, activity, resolved, convention = _resolve(
            entry, records
        )

        record = by_filename.get(filename) if resolved else None

        source = (
            record.readable_text if record is not None else ""
        )

        verdict, reason, question, note = judge(entry, source)

        post_match = _SOURCE_POST.search(entry.get("source"))

        candidates.append(
            GeminiCandidateQuestion(
                index=entry.index,
                candidate_text=entry.get("question"),
                declared_type=entry.get("type"),
                declared_difficulty=entry.get("difficulty"),
                source_excerpt=redact_text(entry.get("answer")),
                group_id=(
                    post_match.group("group").upper() if post_match else ""
                ),
                activity_id=activity,
                slide_number=slide,
                slide_convention=convention,
                filename=filename,
                source_resolved=resolved,
                verdict=verdict,
                verdict_reason=reason,
                question=question,
                rewrite_note=note,
                fingerprint=_fingerprint(entry),
            )
        )

    return candidates


def accepted(candidates: list[GeminiCandidateQuestion]) -> list[GeminiCandidateQuestion]:
    """Only the questions that will actually be used."""

    return [
        candidate
        for candidate in candidates
        if candidate.verdict
        in {CandidateVerdict.ACCEPT, CandidateVerdict.REWRITE}
        and candidate.question
    ]


__all__ = [
    "OPENERS",
    "REPAIRS",
    "RawEntry",
    "accepted",
    "build_candidates",
    "is_question_like",
    "judge",
    "looks_like_link",
    "openers_in",
    "parse_bank",
    "suspicious_merge",
]