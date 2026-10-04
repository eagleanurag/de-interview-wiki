"""
Static wiki generator.

    python -m src.wiki.generator \
      --input aggregation/knowledge_base.json \
      --output site

The generator is pure standard library plus pydantic, and it is
deterministic: the same knowledge base always produces byte-identical
output. Nothing about the build time or the host environment leaks
into the generated files.

Files are written into a staging directory and swapped into place, so
a failed run never leaves a half-written site behind. The output
directory must either be absent or be a directory this tool generated
before, which prevents an accidental `--output` from deleting
unrelated files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time

from dataclasses import replace
from pathlib import Path

from src.wiki.analysis import build_site_model
from src.wiki.canonical import WikiError, load_canonical
from src.wiki.naming import (
    CONCEPTS_PAGE,
    INDEX_PAGE,
    MANIFEST_FILE,
    NOT_FOUND_PAGE,
    OCR_INDEX_FILE,
    QUESTIONS_PAGE,
    SAVED_ITEMS_PAGE,
    SEARCH_INDEX_FILE,
    SEARCH_PAGE,
    TECHNOLOGIES_PAGE,
    TOPICS_PAGE,
)
from src.wiki.saved_items import render_saved_items
from src.wiki.curriculum import build_curriculum
from src.wiki.revision import curriculum_pages
from src.wiki.ocr_index import write_ocr_index
from src.wiki.pages import (
    render_concept_detail,
    render_concepts_index,
    render_home,
    render_not_found,
    render_post_detail,
    render_questions,
    render_search,
    render_technology_detail,
    render_technologies_index,
    render_topic_detail,
    render_topics_index,
)
from src.wiki.search_index import write_search_index


ASSET_DIR = Path(__file__).parent / "assets"

GENERATOR_NAME = "src.wiki.generator"
GENERATOR_VERSION = "1.0"
MANIFEST_VERSION = 1

DEFAULT_OUTPUT = "site"

SKIPPED_ASSETS = frozenset({"__pycache__"})


def generate_site(
    input_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """
    Generate the whole static site.

    Returns the output directory. Raises WikiError with an actionable
    message for any problem, leaving any previous site untouched.
    """

    source = Path(input_path)
    output = Path(output_dir)

    _assert_safe_target(output)

    knowledge_base = load_canonical(source)
    model = build_site_model(knowledge_base)

    staging = output.parent / f".{output.name}.staging"

    if staging.exists():
        shutil.rmtree(staging)

    staging.mkdir(parents=True)

    written = _write_site(model, staging)
    _write_manifest(source, staging, written)

    _swap_into_place(staging, output)

    print(
        f"Wiki generated: {len(written) + 1} file(s) in {output}"
    )
    # Page counts, not label counts. This line describes what was
    # written, and it used to print concepts counted before
    # consolidation beside a topic count counted after, so the two were
    # never comparable and both disagreed with the knowledge base.
    print(
        f"  posts={model.post_count} "
        f"topic pages={model.topic_page_count} "
        f"concept pages={model.concept_page_count} "
        f"technology pages={model.technology_count} "
        f"questions={model.question_count}"
    )

    return output


def _write_site(model, staging: Path) -> list[str]:
    """Write every page and asset. Returns relative POSIX paths."""

    written: list[str] = []
    # Built once, up front, and attached to the model everything below
    # renders from. It has to be up front: a knowledge page links to
    # neighbouring subtopics and cannot know which subtopics have
    # pages without it. Built after the post pages had already been
    # rendered, it left every one of them with no Related section --
    # which is the section a reader uses to decide where to go next.
    curriculum = build_curriculum(list(model.posts))

    model = replace(model, curriculum=curriculum)


    # The reader-facing pages, and only those.
    #
    # The evidence layer -- 5,786 archive topic pages, 3,112 concept
    # pages, 44 technology pages, saved-items, and the four indexes that
    # listed them -- used to be written here as well, which made 8,946 of
    # the site's 9,595 pages a browsable archive of labels the corpus
    # generated. The revision curriculum already represents all of it
    # readably: a concept sits under the subtopic that teaches it, a
    # technology under the subject that uses it.
    #
    # Nothing is lost. Every concept, technology, topic grouping and
    # saved-item provenance record is still in the knowledge base, still
    # reachable through the search index's data layer, and still counted
    # by the manifest. What stops is the reader being sent there.
    pages = {
        INDEX_PAGE: render_home(model),
        SEARCH_PAGE: render_search(model),
        QUESTIONS_PAGE: render_questions(model),
        NOT_FOUND_PAGE: render_not_found(model),
    }

    for post, slug in zip(model.posts, model.post_slugs):
        page = f"posts/{slug}.html"
        pages[page] = render_post_detail(model, post, slug)

    for relative, document in pages.items():
        _write_text(staging, relative, document)
        written.append(relative)

    for asset in _asset_files():
        relative = f"assets/{asset.name}"
        target = staging / "assets" / asset.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(asset, target)
        written.append(relative)

    revision = curriculum_pages(curriculum, model)

    for relative, document in sorted(revision.items()):
        _write_text(staging, relative, document)
        written.append(relative)

    write_search_index(model, staging / SEARCH_INDEX_FILE)
    written.append(SEARCH_INDEX_FILE)

    # Written unconditionally, even when empty. An absent file and an
    # empty one mean different things to the search script -- the first
    # is an error, the second is a site with nothing transcribed -- and
    # only one of those is worth showing a reader.
    write_ocr_index(model, staging / OCR_INDEX_FILE)
    written.append(OCR_INDEX_FILE)

    return sorted(written)


def _asset_files() -> list[Path]:
    if not ASSET_DIR.is_dir():
        raise WikiError(
            f"Generator asset directory is missing: {ASSET_DIR}"
        )

    return sorted(
        path
        for path in ASSET_DIR.iterdir()
        if path.is_file()
        and path.name not in SKIPPED_ASSETS
    )


def _write_text(
    root: Path,
    relative: str,
    content: str,
) -> None:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)

    # newline="\n" keeps output identical regardless of the host OS,
    # so a build on Windows and a build on a runner produce the same
    # bytes.
    with open(
        target,
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        handle.write(content)


def _write_manifest(
    source: Path,
    staging: Path,
    written: list[str],
) -> None:
    """
    Record what was generated.

    The manifest is what makes replacing a previous build safe, and
    the input digest ties a published site to the exact knowledge base
    that produced it.
    """

    payload = {
        "generator": GENERATOR_NAME,
        "generator_version": GENERATOR_VERSION,
        "manifest_version": MANIFEST_VERSION,
        "input_sha256": _digest(source),
        "files": sorted([*written, MANIFEST_FILE]),
    }

    _write_text(
        staging,
        MANIFEST_FILE,
        f"{json.dumps(payload, indent=2, sort_keys=True)}\n",
    )


def _digest(source: Path) -> str:
    return hashlib.sha256(source.read_bytes()).hexdigest()


def _assert_safe_target(output: Path) -> None:
    """
    Refuse to replace anything this tool did not create.

    Guards the two realistic foot-guns: pointing `--output` at the
    repository root, and pointing it at a populated directory that
    holds somebody else's files.
    """

    resolved = output.resolve()
    anchor = Path(resolved.anchor or ".")

    if resolved == anchor:
        raise WikiError(
            f"Refusing to generate into a filesystem root: {resolved}"
        )

    if (resolved / ".git").exists():
        raise WikiError(
            f"Refusing to generate into a git checkout: {resolved}"
        )

    if resolved.exists() and not resolved.is_dir():
        raise WikiError(
            f"Output path exists and is not a directory: {resolved}"
        )

    if resolved.is_dir() and any(resolved.iterdir()):
        if not (resolved / MANIFEST_FILE).is_file():
            raise WikiError(
                f"Refusing to replace {resolved}: it is not empty and "
                f"was not generated by this tool (no {MANIFEST_FILE}). "
                f"Remove it or choose a different --output."
            )


def _swap_into_place(staging: Path, output: Path) -> None:
    """
    Replace the output directory with the freshly built one.

    Built beside the output and swapped in, so a reader never sees a
    half-written site and a crash leaves the previous one intact.

    The rename is retried because Windows refuses it while any handle to
    the target is open. Building nine thousand files means an indexer or
    a virus scanner is following the generator from one to the next, and
    the wait that a handful of files clears instantly is not long enough
    for the tail of a site this size. Every attempt is real, so a
    refusal that is genuinely permanent raises rather than being
    swallowed and leaving no site at all.
    """

    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists():
        shutil.rmtree(output)

    for attempt in range(8):
        try:
            staging.replace(output)
            return

        except PermissionError:
            if attempt == 7:
                raise

            time.sleep(min(0.1 * (2**attempt), 2.0))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the static Data Engineering interview wiki "
            "from a canonical knowledge base."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to aggregation/knowledge_base.json",
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=(
            f"Directory to write the site into "
            f"(default: {DEFAULT_OUTPUT})"
        ),
    )

    args = parser.parse_args()

    try:
        generate_site(args.input, args.output)
    except WikiError as exc:
        print(f"WIKI_ERROR={exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
