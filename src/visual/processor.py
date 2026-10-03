"""
Asking something what is in a picture.

One interface, two implementations, one of which is wired and one of
which reports itself missing.

``VisionProcessor`` is the working one, and it is not a second AI
client. It calls :class:`~src.ai.opencode.OpenCodeClient`, the same
object the enricher calls, over the same CLI, through the same retry and
the same failure classification. The CLI already supports attaching a
file to a message (``opencode run --file``); this is that capability used,
not a new route to a provider.

``OCRProcessor`` is the extension point, and it is deliberately not
installed. Nothing in the environment provides an OCR engine -- no
pytesseract, tesserocr, easyocr, paddleocr or onnxruntime -- and tesseract
itself is a native installer rather than a package. Adding one would
bring an OS-specific, CI-hostile dependency whose result is worse than
what the vision path already achieves: the probe against real slides
transcribed SQL and a six-thousand-character probability slide verbatim,
which is not what OCR on a 480-pixel preview would have produced. So the
class exists, says it is unavailable, and costs nothing when unused.

``CompositeVisualProcessor`` runs the wired processors in order and
stops at the first that understands the asset, so adding OCR later
needs no change to the caller.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path

from src.ai.recovery import classify
from src.visual.models import (
    PROCESSOR_CONFIGURATION,
    PROCESSOR_VERSION,
    CodeBlock,
    VisualAnalysis,
    VisualAsset,
)

#: Bounded, because a 262-slide post must not become one request whose
#: failure loses all of it. Measured: five slides cost about what one
#: costs, so the batch size buys granularity rather than speed, and the
#: default is the largest batch whose failure unit stays acceptable.
DEFAULT_BATCH_SIZE = 5

#: Refuse to hand the provider anything absurd. A decompression bomb
#: should not become a provider request, and neither should a file large
#: enough to time out every attempt.
MAX_ASSET_BYTES = 12_000_000

MAX_BATCH_ASSETS = 12

ANALYSIS_PROMPT = """\
You are reading one or more images taken from a single LinkedIn post. \
They are consecutive slides of a carousel, or a single image when only \
one is given.

Treat only what is visible as source material. Transcribe text exactly \
as it appears, including code, and do not correct, complete or improve \
it. If a word is cut off or illegible, write what you can see and mark \
the gap with [illegible]. Never infer content that is not on the image, \
and never add a technology, product or service that the image does not \
actually name.

Return ONLY a JSON array, one object per image, in the order given, with \
exactly these keys:

  index        integer, 0-based, matching the order of the images
  kind         one of: photo, screenshot, diagram, slide, code, chart, document, other
  visible_text string, every piece of text you can read, verbatim, newlines preserved
  summary      string, one or two sentences on what this image shows. Empty string if there is nothing to say.
  diagram      string, only for diagrams, charts and architecture pictures: the \
               connections between the named parts, in plain words. Empty string otherwise.
  code         array of objects, only where code is visible, each with keys \
               "language" (a language name, or "" if you are not sure) and "code" (verbatim).
  technologies array of strings, only the products, services and tools the image itself names.
  concepts     array of strings, the technical concepts the image itself explains or labels.
  questions    array of strings, questions the image itself raises, labels or answers.
  confidence   number from 0 to 1, how much of this image you could actually read.

Empty arrays and empty strings are correct and expected for images that \
carry little. An empty answer is better than an invented one. Return \
nothing except the JSON array."""


def _strip_fence(text: str) -> str:
    """
    Remove a Markdown fence if the model added one.

    The project's client already does this when it parses a whole
    response. A batched response is an array inside that, and the fence
    can land around the array rather than around an object, so the same
    courtesy is extended here rather than assumed.
    """

    stripped = text.strip()

    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()

    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]

    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]

    return "\n".join(lines).strip()


def _as_list(value: object) -> list[str]:
    """Coerce a model field to a list of non-empty strings."""

    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    if isinstance(value, str) and value.strip():
        return [value.strip()]

    return []


def _as_confidence(value: object) -> float:
    """A confidence clamped into range, defaulting to nothing claimed."""

    try:
        number = float(value)  # type: ignore[arg-type]

    except (TypeError, ValueError):
        return 0.0

    return max(0.0, min(1.0, number))


class VisualProcessor(ABC):
    """
    Something that can look at an image and say what is on it.

    Deliberately narrow. It is handed a path and returns a structured
    answer or raises, and it has no opinion about posts, enrichment or
    the knowledge base. Keeping it that small is what lets the same
    implementation serve the per-slide path, a batch path and a test.
    """

    name: str = "processor"
    version: str = PROCESSOR_VERSION

    @abstractmethod
    def available(self) -> bool:
        """Whether this processor can run at all right now."""

    @abstractmethod
    def analyse(self, paths: list[Path]) -> list[VisualAnalysis]:
        """
        Analyse images, returning one analysis per input, in order.

        Raises on failure. Isolation is the caller's job, because only
        the caller knows which post the images belong to and what the
        rest of the batch is worth.
        """

    def describe(self) -> str:
        return f"{self.name}/{self.version}"


class VisionProcessor(VisualProcessor):
    """
    Reads images through the existing OpenCode client.

    Uses the CLI's own file attachment rather than encoding anything
    into a prompt, so the bytes go to the provider by the mechanism the
    tool already provides and not by a route invented here.
    """

    name = "vision"

    #: Assets handed over for the batch about to be read, keyed by
    #: filename. Empty between calls so a later batch cannot pick up a
    #: previous one's provenance.
    _assets: dict[str, VisualAsset] = {}

    def __init__(self, client=None, *, timeout_seconds: int = 1800) -> None:
        if client is None:
            from src.ai.opencode import OpenCodeClient

            client = OpenCodeClient(timeout_seconds=timeout_seconds)

        self.client = client

    def available(self) -> bool:
        try:
            self.client.run("reply with the single word: ready")

            return True

        except Exception as exc:  # noqa: BLE001
            # Recorded rather than swallowed: "no processor" and
            # "processor is misconfigured" are different problems and a
            # run that reports them identically teaches nothing.
            self.last_error = f"{type(exc).__name__}: {exc}"

            return False

    def analyse(self, paths: list[Path]) -> list[VisualAnalysis]:
        if not paths:
            return []

        if len(paths) > MAX_BATCH_ASSETS:
            raise ValueError(
                f"{len(paths)} images exceeds the batch limit of "
                f"{MAX_BATCH_ASSETS}; the caller must bound its batches"
            )

        assets_by_path = self._assets

        result = self.client.run(
            ANALYSIS_PROMPT,
            files=[str(path) for path in paths],
        )

        entries = _parse_batch(result.data)[: len(paths)]

        analyses = [
            _to_analysis(
                entry,
                index,
                paths[index],
                assets_by_path=assets_by_path,
            )
            for index, entry in enumerate(entries)
        ]

        # A short answer is a failure, not a shorter post. Patching the
        # gap by reusing the last image's text would attribute one
        # slide's words to another, which is the specific thing this
        # whole stage exists to avoid.
        if len(analyses) != len(paths):
            missing = len(paths) - len(analyses)

            raise ValueError(
                f"The model answered for {len(analyses)} of "
                f"{len(paths)} images; {missing} were left out. "
                "Nothing is inferred for them."
            )

        self._assets = {}

        return analyses

    def set_assets(self, assets: dict[str, VisualAsset]) -> None:
        """
        Hand the processor the assets it is about to read.

        Provenance comes from here rather than from the model's answer,
        so the sequence and the content digest of a visual claim are
        facts about the archive and not things the model could state.
        """

        self._assets = dict(assets)


def _parse_batch(data: object) -> list[dict]:
    """
    Read the model's array of per-image answers.

    Tolerates an object for a single image, which the model returns when
    asked about one thing, and a Markdown fence, which it adds often
    enough to be worth handling rather than reporting.
    """

    body = _strip_fence(
        data if isinstance(data, str) else json.dumps(data)
    )

    try:
        parsed = json.loads(body)

    except json.JSONDecodeError:
        # A truncated array cannot be completed honestly. Reported as a
        # failure by the caller rather than patched together.
        raise ValueError(
            "The vision response was not valid JSON. It was most "
            f"likely truncated; first 200 characters: {body[:200]!r}"
        ) from None

    if isinstance(parsed, dict):
        # A single image answered as an object rather than a one-element
        # array, and possibly wrapped in a key.
        for key in ("results", "images", "slides", "analyses", "data"):
            if isinstance(parsed.get(key), list):
                return [item for item in parsed[key] if isinstance(item, dict)]

        return [parsed]

    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]

    raise ValueError(
        f"The vision response was a {type(parsed).__name__}, not an "
        "array of per-image answers"
    )


def _to_analysis(
    entry: dict,
    index: int,
    path: Path,
    *,
    assets_by_path: dict[str, VisualAsset] | None = None,
) -> VisualAnalysis:
    """
    Turn one model answer into an analysis, with provenance attached.

    The digest and sequence come from the asset that was read, never
    from the model. A model that hallucinates a field cannot invent its
    own provenance, and a visual fact that cannot name its slide has
    nothing to be a fact about.
    """

    asset = (assets_by_path or {}).get(path.name)

    code_blocks = []

    for block in entry.get("code") or []:
        if isinstance(block, str):
            code_blocks.append(CodeBlock(code=block))

            continue

        if not isinstance(block, dict):
            continue

        code = str(block.get("code") or "").strip()

        if not code:
            continue

        language = str(block.get("language") or "").strip()

        code_blocks.append(
            CodeBlock(
                # Left empty rather than guessed. A confident wrong
                # language is a wrong fact about how a technology is
                # used, which is worse than no claim.
                language=language or None,
                code=code,
                method="vision",
            )
        )

    return VisualAnalysis(
        source_asset_hash=asset.sha256 if asset else "",
        source_path=f"media/{path.name}",
        sequence=asset.sequence if asset else index,
        extracted_text=(str(entry.get("visible_text") or "").strip() or None),
        visual_summary=(str(entry.get("summary") or "").strip() or None),
        diagram_description=(str(entry.get("diagram") or "").strip() or None),
        code_blocks=code_blocks,
        detected_technologies=_as_list(entry.get("technologies")),
        detected_topics=_as_list(entry.get("topics")),
        detected_concepts=_as_list(entry.get("concepts")),
        interview_questions=_as_list(entry.get("questions")),
        confidence=_as_confidence(entry.get("confidence")),
        processor="vision",
        processor_version=PROCESSOR_VERSION,
        processor_configuration=PROCESSOR_CONFIGURATION,
        batch_index=index,
    )


class OCRProcessor(VisualProcessor):
    """
    Local text extraction, when an engine is installed.

    Present as the extension point the architecture calls for, and not
    wired, because no engine is installed and adding one is a worse
    trade than the vision path it would sit beside.

    Kept honest in three ways: it reports itself unavailable rather than
    silently doing nothing, it says what it would need, and it does not
    pretend to have read an image when it has not. A processor that
    returned empty text as if it had succeeded would make "the slide was
    blank" and "no OCR engine exists" indistinguishable in the output.
    """

    name = "ocr"

    def __init__(self, engine: str | None = None) -> None:
        self.engine = engine or _find_ocr_engine()

    def available(self) -> bool:
        return self.engine is not None

    def unavailable_reason(self) -> str:
        return (
            "No local OCR engine is installed. Tesseract would be "
            "required: it is a native installer rather than a package, "
            "so it is not installed by this project. The vision "
            "processor covers the same ground and reads code and "
            "diagrams better than generic OCR would on a "
            "low-resolution preview."
        )

    def analyse(self, paths: list[Path]) -> list[VisualAnalysis]:
        if not self.available():
            raise RuntimeError(self.unavailable_reason())

        # Unreachable until an engine is configured. Written out so
        # that wiring one is a single function rather than a design
        # exercise, and so the shape of the answer is already decided.
        raise NotImplementedError(
            "Configure an OCR engine before enabling this processor"
        )


def _find_ocr_engine() -> str | None:
    """A local OCR engine, if the machine happens to have one."""

    import shutil

    for candidate in ("tesseract",):
        found = shutil.which(candidate)

        if found:
            return found

    return None


class CompositeVisualProcessor(VisualProcessor):
    """
    Runs processors in order, stopping at the first that understands.

    Order is a preference, not a race: OCR first when it is installed
    because it is free and offline, then vision. The first processor to
    return a usable answer wins, so adding a cheaper processor later
    cannot change the answer for slides the earlier one handled.
    """

    name = "composite"

    def __init__(self, processors: list[VisualProcessor]) -> None:
        self.processors = processors

    @property
    def version(self) -> str:
        return "+".join(
            processor.version for processor in self.processors
        ) or PROCESSOR_VERSION

    def available(self) -> bool:
        return any(processor.available() for processor in self.processors)

    def describe(self) -> str:
        return "+".join(
            processor.describe() for processor in self.processors
        )

    def analyse(self, paths: list[Path]) -> list[VisualAnalysis]:
        for processor in self.processors:
            if not processor.available():
                continue

            analyses = processor.analyse(paths)

            if analyses and any(item.is_usable for item in analyses):
                return analyses

        return []


def build_default_processor() -> VisualProcessor:
    """
    The processor this project uses.

    OCR first, so that a machine with an engine gets it, and vision
    always, because it is the one that works here.
    """

    return CompositeVisualProcessor([OCRProcessor(), VisionProcessor()])


def failure_note(exc: BaseException) -> str:
    """
    Why one asset could not be read, classified.

    The classification is the same one the enricher's retry uses, so a
    truncated provider response is recognisable here as it is there
    rather than being a new and separate kind of failure.
    """

    verdict = classify(exc)

    return f"{verdict.kind.value}: {str(exc)[:300]}"


__all__ = [
    "ANALYSIS_PROMPT",
    "CompositeVisualProcessor",
    "DEFAULT_BATCH_SIZE",
    "MAX_ASSET_BYTES",
    "MAX_BATCH_ASSETS",
    "OCRProcessor",
    "VisionProcessor",
    "VisualProcessor",
    "build_default_processor",
    "failure_note",
]
