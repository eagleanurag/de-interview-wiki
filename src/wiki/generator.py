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
from pathlib import Path

from src.wiki.analysis import build_site_model
from src.wiki.canonical import WikiError, load_canonical
from src.wiki.naming import (
    CONCEPTS_PAGE,
    INDEX_PAGE,
    MANIFEST_FILE,
    NOT_FOUND_PAGE,
    QUESTIONS_PAGE,
    SAVED_ITEMS_PAGE,
    SEARCH_INDEX_FILE,
    SEARCH_PAGE,
    TECHNOLOGIES_PAGE,
    TOPICS_PAGE,
)
from src.wiki.saved_items import render_saved_items
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
    print(
        f"  posts={model.post_count} "
        f"topics={model.topic_count} "
        f"concepts={model.concept_count} "
        f"questions={model.question_count}"
    )

    return output


def _write_site(model, staging: Path) -> list[str]:
    """Write every page and asset. Returns relative POSIX paths."""

    written: list[str] = []

    pages = {
        INDEX_PAGE: render_home(model),
        SEARCH_PAGE: render_search(model),
        TOPICS_PAGE: render_topics_index(model),
        CONCEPTS_PAGE: render_concepts_index(model),
        TECHNOLOGIES_PAGE: render_technologies_index(model),
        QUESTIONS_PAGE: render_questions(model),
        SAVED_ITEMS_PAGE: render_saved_items(model),
        NOT_FOUND_PAGE: render_not_found(model),
    }

    for post, slug in zip(model.posts, model.post_slugs):
        page = f"posts/{slug}.html"
        pages[page] = render_post_detail(model, post, slug)

    for topic in model.topics:
        pages[topic.page] = render_topic_detail(model, topic)

    for concept in model.concept_entries:
        pages[concept.page] = render_concept_detail(model, concept)

    for technology in model.technology_entries:
        pages[technology.page] = render_technology_detail(
            model, technology
        )

    for relative, document in pages.items():
        _write_text(staging, relative, document)
        written.append(relative)

    for asset in _asset_files():
        relative = f"assets/{asset.name}"
        target = staging / "assets" / asset.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(asset, target)
        written.append(relative)

    write_search_index(model, staging / SEARCH_INDEX_FILE)
    written.append(SEARCH_INDEX_FILE)

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
    """Replace the output directory with the freshly built one."""

    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists():
        shutil.rmtree(output)

    staging.replace(output)


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
