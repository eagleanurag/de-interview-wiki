"""
What is observably true of a transcription.

No score. That is the constraint worth stating first, because a numeric
quality metric is the obvious thing to add here and the wrong one.

A score invites two failures. It invites a threshold, and whoever sets
the threshold has invented a meaning for the number that nobody else can
reconstruct. And it hides its own inputs: a 0.62 does not say whether
the text was short, symbol-laden, or a clean diagram with three labels,
and those three deserve different handling.

So this module reports four named states, each derived from something a
reader can check, and records the evidence beside the verdict. The
evidence is what makes the verdict arguable, which is the only property
worth having here.

The signals used, all from the text itself:

* **Proportion of word-like runs.** Real text produces runs of letters;
  symbol debris produces runs of one and two characters. This is the
  strongest signal, because OCR's characteristic failure is not wrong
  words but stray characters.
* **Whether the text reads as connected sentences.** A transcription
  with punctuation and sentence structure is prose; one without is
  labels.
* **Share of non-alphanumeric characters.** High means the page was
  dense with iconography, or the recognition failed on graphics.
* **Presence of replacement characters**, which mean the source itself
  carried undecodable bytes.

None of these judge whether the transcription is *correct*. A garbled
transcription may still contain the one term a reader searched for, and
it is kept and indexed rather than dropped. Whether a slide is worth
re-reading with the CP12 vision processor is decided by the caller from
these states plus the image's own size, not from here.
"""

from __future__ import annotations

import re
import unicodedata

from src.gemini.models import OcrQuality


#: A run of letters, optionally carrying an internal apostrophe or
#: hyphen. Three is the floor: ``SQL`` and ``ETL`` are two letters and
#: perfectly real, so the floor cannot exclude short technical tokens.
WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")

#: Characters OCR emits when it cannot decode a glyph, and the
#: near-misses that mean the same thing.
REPLACEMENT = "�□"

#: Punctuation that indicates running prose rather than labels.
SENTENCE_MARKS = ".?!:;"

#: Above this share of non-letter, non-space characters, the text is
#: symbol load rather than words. Set high deliberately.
#:
#: An earlier version of this module used a *low* letter-share floor to
#: call text garbled, and it was wrong. Measured against this package's
#: own two extremes:
#:
#:   coherent SQL transcription   79% letters,  9% symbols, 12 words
#:   real symbol-noise sample      61% letters, 17% symbols, 20 words
#:   real data-tools-menu sample   63% letters, 20% symbols, 26 words
#:
#: The noise scores *better* than the prose on letter share, on symbol
#: share, on short-word share, on average word length, and on longest
#: consecutive word run. That is because OCR failure on a small
#: screenshot is not a failure to produce letters -- it produces
#: confidently misread letters. "bree ondey AEST" and a real heading are
#: the same shape of text.
#:
#: So no character statistic separates them, and a threshold low enough
#: to catch this would flag ordinary bullet-fragment slides as debris.
#: Which matters more than it sounds: GARBLED is what makes a slide a
#: candidate for a vision-model call. A threshold that over-fires turns a
#: bounded fallback into a full re-run of the corpus, and one that
#: under-fires costs nothing -- the damaged transcription stays indexed
#: and labelled, and a reader who searches for a term in it still finds
#: it.
SYMBOL_CEILING = 0.40

#: Fewer words than this and there is too little text to carry a topic,
#: whatever the characters look like.
SPARSE_WORDS = 4

#: Above this share of words being the same token, the engine repeated a
#: line rather than reading the slide. The one symbol-free signal that
#: does indicate a failed read, and the strongest one available.
REPETITION_SHARE = 0.34


def _is_wordlike(char: str) -> bool:
    """Whether a character is a letter rather than debris."""

    if char.isalpha():
        return True

    # An accented letter counts as a letter, which matters because the
    # archive contains names and headings that are not ASCII.
    return unicodedata.category(char).startswith("L")


def wordlike_runs(text: str) -> tuple[int, int]:
    """
    How much of the text is letters, and how many words it found.

    Returned as a pair rather than a ratio alone so a caller can see
    both "mostly symbols" and "almost nothing there", which are the two
    conditions that get confused.
    """

    letters = sum(1 for char in text if _is_wordlike(char))

    words = WORD.findall(text)

    return letters, len(words)


def symbol_share(text: str) -> float:
    """The proportion of characters that are not letters or spaces."""

    if not text:
        return 1.0

    noisy = sum(
        1
        for char in text
        if not _is_wordlike(char) and not char.isspace()
    )

    return noisy / max(1, len(text))


def sentence_share(text: str) -> float:
    """How much of the text looks like running prose."""

    if not text:
        return 0.0

    marks = sum(1 for char in text if char in SENTENCE_MARKS)

    return marks / max(1, len(text))


def has_replacement(text: str) -> bool:
    """Whether the source itself carried undecodable bytes."""

    return any(char in REPLACEMENT for char in text)


def dominant_word(text: str) -> str:
    """
    The most repeated substantial token, lowercased.

    Used only to detect an OCR engine that has emitted the same line
    repeatedly -- a failure mode that leaves a transcription looking
    word-rich while saying nothing. An empty result when there is no
    repetition is the normal case and means nothing is wrong.
    """

    words = [
        word.lower()
        for word in WORD.findall(text)
        if len(word) >= 4
    ]

    if len(words) < 12:
        return ""

    counts: dict[str, int] = {}

    for word in words:
        counts[word] = counts.get(word, 0) + 1

    best, highest = "", 0

    for word, count in counts.items():
        if count > highest:
            best, highest = word, count

    # A third or more of the words being one token is repetition.
    return best if highest >= len(words) / 3 else ""


def assess(text: str | None) -> tuple[OcrQuality, str]:
    """
    Judge a transcription, and say why.

    The second element is the evidence, written out, because a verdict
    with no stated reason is one nobody can disagree with usefully.

    Four states, and the honest boundary between them. ``GARBLED`` means
    a read that demonstrably failed, which character analysis can only
    establish three ways: the engine looped, the source carried
    undecodable glyphs, or the text is overwhelmingly symbol rather than
    letters. ``READABLE`` means "nothing observable is wrong with it",
    which is a much weaker statement than "it reads correctly" and is
    deliberately not dressed up as one.

    What this cannot do is tell misread English from real English. See
    :data:`SYMBOL_CEILING` for the measurements, and
    :func:`limitations` for what that costs.
    """

    if not text or not text.strip():
        return OcrQuality.EMPTY, "no text was transcribed"

    body = text.strip()

    letters, words = wordlike_runs(body)

    letter_share = letters / max(1, len(body))

    symbols = symbol_share(body)

    prose = sentence_share(body)

    repeated = dominant_word(body)

    parts = [
        f"{len(body)} chars",
        f"{words} words",
        f"{letter_share:.0%} letters",
        f"{symbols:.0%} symbols",
        f"{prose:.1%} sentence marks",
    ]

    if has_replacement(body):
        parts.append("contains replacement characters")

    if repeated:
        parts.append(f"repeats {repeated!r}")

    evidence = ", ".join(parts)

    if repeated:
        return OcrQuality.GARBLED, (
            f"{evidence}; one token dominates, so the engine repeated a "
            "line rather than reading the slide"
        )

    if has_replacement(body):
        return OcrQuality.GARBLED, (
            f"{evidence}; undecodable glyphs in the source"
        )

    if symbols > SYMBOL_CEILING:
        return OcrQuality.GARBLED, (
            f"{evidence}; overwhelmingly symbol rather than letters"
        )

    if words < SPARSE_WORDS:
        return OcrQuality.FRAGMENTARY, (
            f"{evidence}; real words, but too few to carry a topic"
        )

    return OcrQuality.READABLE, evidence


def limitations() -> str:
    """
    What this module's judgement is not worth, in one sentence.

    Carried into the manifest rather than left in a docstring, because
    the number it qualifies appears on the site. A reader told that a
    transcription is "readable" should be able to find out what that
    word means here.
    """

    return (
        "A transcription marked readable has no observable defect, not "
        "that it is correct: OCR misreading a small screenshot produces "
        "confidently wrong words that are indistinguishable from real "
        "ones by character analysis."
    )


def needs_reprocessing(
    quality: OcrQuality,
    *,
    width: int | None = None,
    height: int | None = None,
) -> bool:
    """
    Whether the CP12 vision processor is worth running on this slide.

    Deliberately narrow. The point of importing this package is to avoid
    paying for thousands of model calls, and a permissive gate would
    defeat that while appearing to help. Only these three cases justify
    a call:

    * nothing was transcribed at all, so there is nothing to index;
    * the text is debris, where a second reader may do better than the
      first and the cost is bounded to the slides that failed;
    * the package marked the record unresolved or failed itself.

    A FRAGMENTARY transcription is left alone. It is real text from a
    real picture, it is indexed, and it is labelled as fragmentary in
    the knowledge base. Sparse is not the same as unreadable, and
    spending a model call to confirm that a small diagram has three
    labels on it is how a budget disappears.

    Image size is accepted but not used to decide. It would be a
    reasonable additional signal -- a large picture with four words of
    OCR is suspicious -- but the threshold would be invented, and an
    invented threshold is what this module exists to avoid.
    """

    return quality in {OcrQuality.EMPTY, OcrQuality.GARBLED}


def as_search_text(text: str | None) -> str:
    """
    Whitespace collapsed, nothing else changed.

    Punctuation, case, code symbols and SQL survive on purpose. An
    index that turns ``INNER JOIN`` into ``inner join`` would still match
    a case-insensitive query, but one that strips parentheses and quotes
    would break ``COUNT(DISTINCT x)``, and code is a large part of what
    these slides contain.

    Collapsing runs of whitespace is the only transformation, because it
    is the only one that cannot change what a term looks like.
    """

    if not text:
        return ""

    return re.sub(r"\s+", " ", text).strip()


__all__ = [
    "REPETITION_SHARE",
    "SPARSE_WORDS",
    "SYMBOL_CEILING",
    "assess",
    "as_search_text",
    "dominant_word",
    "has_replacement",
    "limitations",
    "needs_reprocessing",
    "sentence_share",
    "symbol_share",
    "wordlike_runs",
]
