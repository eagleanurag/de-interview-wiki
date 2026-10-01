"""
The manual source.

Content the user has already captured: text they pasted, a file they
dropped in, or a bundle directory. This is the default source because
it requires no credentials and no network, and it is what makes the
rest of the pipeline testable without any live source.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

from src.ingestion.errors import InvalidPostError
from src.ingestion.sources.base import (
    CollectedPost,
    CollectionStopped,
    CollectionState,
    StopReason,
    Source,
)


class ManualSource(Source):
    """
    Reads posts from a directory of capture bundles.

    A bundle is either a directory containing ``capture.json`` (or a
    ``post.json``) or a single ``.txt``/``.md`` file. Directories are
    walked in sorted order so a run is reproducible.
    """

    name = "manual"
    platform = "manual"

    def __init__(
        self,
        bundle_root: str | Path,
        *,
        platform: str | None = None,
    ) -> None:
        self.bundle_root = Path(bundle_root)
        self._platform = platform

    def discover(self, **limits: object):
        """
        Yield one collected post per bundle, in sorted order.

        Respects ``max_posts`` and ``since``/``until`` the same way the
        browser source does, so both sources behave identically for the
        collector.
        """

        max_posts = limits.get("max_posts")
        since = limits.get("since")
        until = limits.get("until")

        if not self.bundle_root.exists():
            raise CollectionStopped(
                StopReason.FAILED,
                f"Manual source root does not exist: "
                f"{self.bundle_root}",
            )

        produced = 0

        for bundle in self._bundles():
            if isinstance(max_posts, int) and produced >= max_posts:
                raise CollectionStopped(
                    StopReason.MAX_POSTS,
                    f"Reached the configured limit of {max_posts}.",
                )

            collected = self._read_bundle(bundle)

            if collected is None:
                continue

            if not self._within_window(
                collected.published_at, since=since, until=until
            ):
                continue

            collected.extra["bundle"] = str(bundle)

            yield collected

            produced += 1

        raise CollectionStopped(StopReason.EXHAUSTED)

    def _bundles(self) -> list[Path]:
        if self.bundle_root.is_file():
            return [self.bundle_root]

        bundles: list[Path] = []

        for path in sorted(self.bundle_root.rglob("*")):
            if not path.is_file():
                continue

            if path.name in {"capture.json", "post.json"}:
                bundles.append(path.parent)
                continue

            if path.suffix.lower() in {".txt", ".md", ".jsonl"}:
                bundles.append(path)

        seen: set[Path] = set()
        unique: list[Path] = []

        for bundle in bundles:
            resolved = bundle.resolve()

            if resolved in seen:
                continue

            seen.add(resolved)
            unique.append(bundle)

        return unique

    def _read_bundle(self, bundle: Path) -> CollectedPost | None:
        if bundle.is_file() and bundle.suffix.lower() != ".jsonl":
            return self._read_text_bundle(bundle)

        if bundle.suffix.lower() == ".jsonl":
            return self._read_jsonl_bundle(bundle)

        capture = bundle / "capture.json"

        if not capture.is_file():
            capture = bundle / "post.json"

        if not capture.is_file():
            return None

        payload = self._read_json(capture)

        return self._from_payload(payload, bundle)

    def _read_text_bundle(self, bundle: Path) -> CollectedPost:
        text = bundle.read_text(
            encoding="utf-8", errors="replace"
        ).strip()

        if not text:
            return CollectedPost(
                source_post_id=bundle.stem,
                text="",
                extra={"empty": True},
            )

        return CollectedPost(
            source_post_id=bundle.stem,
            text=text,
            extra={"empty": True},
        )

    def _read_jsonl_bundle(self, bundle: Path) -> CollectedPost:
        """
        Read a JSON Lines export.

        Each line is one post, which is the shape an official data
        export usually takes.
        """

        collected: list[CollectedPost] = []

        for number, line in enumerate(
            bundle.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines(),
            start=1,
        ):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise InvalidPostError(
                    f"{bundle.name} line {number} is not valid JSON: "
                    f"{exc}"
                ) from exc

            if not isinstance(payload, dict):
                continue

            entry = self._from_payload(payload, bundle)

            if entry is not None:
                collected.append(entry)

        text_lines = [
            f"## {entry.source_post_id}\n\n{entry.text}"
            for entry in collected
            if entry.text
        ]

        return CollectedPost(
            source_post_id=bundle.stem,
            text="\n\n".join(text_lines),
            extra={"entries": len(collected)},
        )

    def _from_payload(
        self,
        payload: dict,
        bundle: Path,
    ) -> CollectedPost | None:
        text = (
            payload.get("text")
            or payload.get("original_text")
            or payload.get("content")
            or ""
        )

        if not isinstance(text, str):
            text = ""

        identifier = (
            payload.get("source_post_id")
            or payload.get("id")
            or payload.get("urn")
            or bundle.stem
        )

        media = self._media_paths(payload, bundle)

        return CollectedPost(
            source_post_id=str(identifier),
            text=text.strip(),
            url=_optional_str(payload.get("url")),
            published_at=_optional_str(
                payload.get("published_at")
                or payload.get("publishedAt")
            ),
            author=_optional_str(
                payload.get("author") or payload.get("authorName")
            ),
            media=media,
            extra={
                key: value
                for key, value in payload.items()
                if key
                not in {
                    "text",
                    "original_text",
                    "content",
                    "url",
                    "published_at",
                    "publishedAt",
                    "author",
                    "authorName",
                    "media",
                    "images",
                }
            },
        )

    def _media_paths(
        self,
        payload: dict,
        bundle: Path,
    ) -> list[Path]:
        raw = payload.get("media") or payload.get("images") or []

        if not isinstance(raw, list):
            return []

        resolved: list[Path] = []

        for item in raw:
            if not isinstance(item, str) or not item.strip():
                continue

            candidate = Path(item.strip())

            if not candidate.is_absolute():
                candidate = bundle / candidate

            if candidate.is_file():
                resolved.append(candidate)

        return resolved

    def _read_json(self, path: Path) -> dict:
        try:
            payload = json.loads(
                path.read_text(
                    encoding="utf-8", errors="replace"
                )
            )
        except json.JSONDecodeError as exc:
            raise InvalidPostError(
                f"{path} is not valid JSON: {exc}"
            ) from exc

        if not isinstance(payload, dict):
            raise InvalidPostError(
                f"{path} must contain a JSON object"
            )

        return payload

    @property
    def platform(self) -> str:  # type: ignore[override]
        return self._platform or self.name

    @staticmethod
    def _within_window(
        published_at: str | None,
        *,
        since: object,
        until: object,
    ) -> bool:
        if not published_at or (since is None and until is None):
            return True

        moment = _parse_moment(published_at)

        if moment is None:
            # An unparsable timestamp is kept rather than dropped, so
            # a format change never silently loses content.
            return True

        lower = _parse_moment(str(since)) if since else None
        upper = _parse_moment(str(until)) if until else None

        if lower and moment < lower:
            return False

        if upper and moment > upper:
            return False

        return True


def _optional_str(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()

    return None


def _parse_moment(value: str) -> datetime | None:
    """
    Parse a timestamp to a naive UTC value.

    Normalized to naive UTC because sources are inconsistent about
    offsets, and comparing an offset-aware value with a naive one
    raises. Naive UTC keeps every comparison well defined.
    """

    text = value.strip()

    if not text:
        return None

    candidate = text.replace("Z", "+00:00")

    for attempt in (candidate, text[:10]):
        try:
            parsed = datetime.fromisoformat(attempt)
        except ValueError:
            continue

        if parsed.tzinfo is None:
            return parsed

        return parsed.astimezone(timezone.utc).replace(tzinfo=None)

    return None


def parse_day(value: str) -> date:
    """Parse a ``--since``/``--until`` day boundary."""

    parsed = _parse_moment(value)

    if parsed is None:
        raise ValueError(
            f"Not a valid date: {value!r}. Use YYYY-MM-DD."
        )

    return parsed.date()


def merge_states(
    previous: CollectionState | None,
    current: CollectionState,
) -> CollectionState:
    """Add a resumed run's counts to the recorded totals."""

    if previous is None:
        return current

    return CollectionState(
        discovered=previous.discovered + current.discovered,
        persisted=previous.persisted + current.persisted,
        duplicates=previous.duplicates + current.duplicates,
        failed=previous.failed + current.failed,
        last_post_id=current.last_post_id
        or previous.last_post_id,
        last_url=current.last_url or previous.last_url,
        stopped_because=current.stopped_because,
        started_at=previous.started_at or current.started_at,
        updated_at=current.updated_at,
    )