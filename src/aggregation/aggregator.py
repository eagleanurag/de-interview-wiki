from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from src.aggregation.consolidation import consolidate, verify
from src.models import KnowledgePost


WORKER_RESULT_GLOB = "cloud_worker_*.json"

#: What makes a payload a post rather than something else. These are
#: the fields that carry identity and provenance.
#:
#: Enrichment fields are deliberately not required. A post that has not
#: been enriched is still real content with real provenance, and
#: requiring its analysis to aggregate it would drop the source
#: entirely. It is aggregated, labelled "unenriched", and carried
#: through to the site so a reader can still reach it.
REQUIRED_POST_FIELDS = frozenset(
    {
        "id",
        "source",
        "original_text",
    }
)

#: Retained under the old name for callers that referenced it.
REQUIRED_ENRICHED_FIELDS = REQUIRED_POST_FIELDS


class AggregationError(RuntimeError):
    """Raised when worker results cannot be safely aggregated."""


def _is_job_manifest(raw: dict) -> bool:
    """
    Identify a worker job manifest.

    A job manifest carries a job_id and never carries a post id.
    A KnowledgePost always carries an id and never carries a job_id.
    Checking the two shapes directly is more reliable than relying on
    which optional fields a manifest happens to contain.
    """

    return "job_id" in raw and "id" not in raw


def _load_json_object(result_file: Path) -> object:
    """Read and decode one worker file."""

    try:
        return json.loads(
            result_file.read_text(encoding="utf-8-sig")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise AggregationError(
            f"Could not read result file {result_file}: {exc}"
        ) from exc


def _describe(path: Path, root: Path) -> str:
    """Render a stable, root-relative label for reporting."""

    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def aggregate_results(
    input_directory: str | Path,
    output_path: str | Path,
    expected_post_count: int | None = None,
) -> Path:
    """
    Aggregate worker result JSON files into one canonical knowledge base.

    Job manifests uploaded alongside worker results are skipped because
    they are not KnowledgePost objects.

    Duplicate post IDs within the same aggregation input are treated as
    an error so that one post is never silently overwritten.

    `expected_post_count`, when provided, must match the number of
    posts actually aggregated. This turns a silently missing worker
    result into a failed run.
    """

    input_dir = Path(input_directory)
    output_file = Path(output_path)

    if not input_dir.exists():
        raise AggregationError(
            f"Input directory does not exist: {input_dir}"
        )

    result_files = sorted(input_dir.rglob(WORKER_RESULT_GLOB))

    if not result_files:
        raise AggregationError(
            f"No worker JSON files matching "
            f"{WORKER_RESULT_GLOB} found under: {input_dir}"
        )

    posts: list[KnowledgePost] = []
    seen_ids: set[str] = set()
    skipped_files: list[str] = []

    for result_file in result_files:
        raw = _load_json_object(result_file)

        if not isinstance(raw, dict):
            skipped_files.append(
                f"{_describe(result_file, input_dir)}: "
                f"not a JSON object"
            )
            continue

        if _is_job_manifest(raw):
            skipped_files.append(
                f"{_describe(result_file, input_dir)}: "
                f"worker job manifest"
            )
            continue

        if not REQUIRED_POST_FIELDS.issubset(raw):
            skipped_files.append(
                f"{_describe(result_file, input_dir)}: "
                f"missing required KnowledgePost fields"
            )
            continue

        try:
            post = KnowledgePost.model_validate(raw)
        except ValidationError as exc:
            raise AggregationError(
                f"Invalid KnowledgePost in {result_file}:\n{exc}"
            ) from exc

        if post.id in seen_ids:
            raise AggregationError(
                f"Duplicate post ID detected during aggregation: "
                f"{post.id}"
            )

        seen_ids.add(post.id)
        posts.append(post)

    if expected_post_count is not None:
        if len(posts) != expected_post_count:
            raise AggregationError(
                f"Expected {expected_post_count} aggregated post(s) "
                f"but collected {len(posts)} from "
                f"{len(result_files)} worker file(s) in {input_dir}. "
                f"A worker result is missing or was skipped."
            )

    posts.sort(key=lambda post: post.id)

    # Technologies read out of an attached image, when a package has been
    # imported. Absent an import this is empty and the consolidation is
    # exactly what it was before, which is the point: the imported layer
    # is optional and adds nothing on its own.
    from src.gemini.derived import derived_claims_for_posts

    derived = derived_claims_for_posts(posts)

    # Fold the claims onto the posts before consolidating.
    #
    # There were two channels here and only one of them reached the site.
    # The claims arrive as a mapping keyed by post; the site builder
    # reads ``ai_analysis.derived_technologies`` on the post itself. A
    # technology present in the knowledge base but absent from its own
    # site -- Azure Data Factory, in the first run of this -- is exactly
    # what that split produces. Writing the names onto the post first
    # makes the post the single channel both sides read, which is the
    # only arrangement in which "the knowledge base and the site cannot
    # disagree" is true rather than merely intended.
    by_id = {post.id: post for post in posts}

    for post_id, claims in derived.items():
        target = by_id.get(post_id)

        if target is None:
            continue

        existing = target.ai_analysis.derived_technologies

        for claim in claims:
            if claim.name not in existing:
                existing.append(claim.name)

        existing.sort()

    # Consolidation turns the post list into navigable knowledge areas.
    # It runs before the payload is written so a reference to a post
    # that was never aggregated fails the run rather than producing a
    # knowledge base that points at nothing.
    index = consolidate(posts, derived_technologies=derived)

    verify(index, posts)

    payload = {
        "schema_version": 2,
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "stats": {
            "result_files_found": len(result_files),
            "posts_aggregated": len(posts),
            "files_skipped": len(skipped_files),
            "topics_consolidated": len(index.topics),
            "subtopics_consolidated": len(index.subtopics),
            "concepts_consolidated": len(index.concepts),
            "technologies_consolidated": len(index.technologies),
            "questions_consolidated": len(index.questions),
            "posts_interview_relevant": sum(
                1
                for post in posts
                if post.classification.interview_relevant
            ),
            # How many posts actually carry an analysis, as opposed to
            # being present but not yet enriched. Reported because a
            # knowledge base with empty topics is otherwise
            # indistinguishable from a knowledge base with no content.
            "posts_enriched": sum(
                1
                for post in posts
                if (post.ai_analysis.summary or "").strip()
            ),
        },
        "skipped_files": skipped_files,
        "posts": [
            post.model_dump(mode="json") for post in posts
        ],
        "knowledge": index.as_dict(),
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)

    temporary_file = output_file.with_suffix(
        output_file.suffix + ".tmp"
    )

    temporary_file.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    temporary_file.replace(output_file)

    print(
        f"Aggregated {len(posts)} post(s) "
        f"from {len(result_files)} worker file(s)."
    )

    print(
        f"Consolidated {len(index.topics)} topic(s), "
        f"{len(index.concepts)} concept(s), "
        f"{len(index.technologies)} technology(ies), "
        f"{len(index.questions)} question(s)."
    )

    if skipped_files:
        print(
            f"Skipped {len(skipped_files)} non-result file(s)."
        )

        for skipped in skipped_files:
            print(f"  - {skipped}")

    print(f"Knowledge base written: {output_file}")

    return output_file


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate cloud worker results."
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing downloaded worker artifacts.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path of the canonical knowledge base JSON.",
    )

    parser.add_argument(
        "--expected-post-count",
        type=int,
        default=None,
        help=(
            "Fail unless exactly this many posts are aggregated. "
            "Use the number of discovered posts to detect a missing "
            "worker result."
        ),
    )

    args = parser.parse_args()

    try:
        aggregate_results(
            args.input_dir,
            args.output,
            expected_post_count=args.expected_post_count,
        )
    except AggregationError as exc:
        print(f"AGGREGATION_ERROR={exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
