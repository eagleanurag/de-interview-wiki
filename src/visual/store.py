"""
Per-asset storage, so a long run can be interrupted.

A post with 262 slides must not be one indivisible unit of work. If it
fails at slide 200, the other 199 are still worth keeping, and the next
run should not pay for them again.

So the unit of storage is the asset, keyed by what it contains and what
was done to it: the SHA-256 of the bytes, the processor version, and the
processor configuration. A stored analysis is reused when all three
match, and reprocessed when any of them does not. That is the same rule
the enricher's fingerprints already use, applied to a finer grain, and
for the same reason: reuse must be decidable without asking a model.

Stored under ``build/visual/``, beside the other pipeline output and
git-ignored, because it is a cache of a run and not a source of truth.
The truth is the archive; this is what was learned from it.

One file per asset. Small, and it means a corrupt write can cost one
slide rather than a post, because there is nothing to lose but the one
file.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from src.visual.models import VisualAnalysis

#: Two characters of the digest, so the tree is browsable and no
#: directory holds more than a few hundred entries.
SHARD = 2

#: One lock per destination path, created on demand.
#:
#: Keyed by the file being replaced rather than global, so posts reading
#: different pictures never wait on each other. The lock table itself is
#: guarded, and entries are never removed: the number of distinct
#: analyses in a run is bounded by the number of images, which is a few
#: thousand small objects.
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    with _LOCKS_GUARD:
        lock = _LOCKS.get(str(path))

        if lock is None:
            lock = threading.Lock()
            _LOCKS[str(path)] = lock

        return lock


def _replace(source: str, target: Path, attempts: int = 8) -> None:
    """
    Move a temporary file into place, retrying a refusal.

    Windows refuses ``os.replace`` while anything holds the destination
    open. An indexer or virus scanner following a few thousand freshly
    written files does exactly that, at least once, and the project's
    knowledge-base write already handles it the same way for the same
    reason.

    Bounded, and with a growing wait: a file held for a quarter of a
    second is a different thing from one held forever, and giving up
    eventually is what keeps a stuck file from stalling a run.
    """

    for attempt in range(attempts):
        try:
            os.replace(source, target)

            return

        except PermissionError:
            if attempt == attempts - 1:
                raise

            time.sleep(min(0.05 * (2**attempt), 2.0))

        except OSError:
            if attempt == attempts - 1:
                raise

            time.sleep(min(0.05 * (2**attempt), 2.0))


class VisualStore:
    """
    Reads and writes one file per analysed asset.

    Deliberately not a database. The whole cache is a few thousand small
    documents, it is meant to be inspectable with a text editor when
    something looks wrong, and it has to be deletable without leaving the
    repository inconsistent -- which is the situation this project is in
    every time a processor is upgraded.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path_for(self, asset_hash: str, version: str, configuration: str) -> Path:
        """Where one asset's analysis lives."""

        if not asset_hash:
            # Nothing readable, so nothing to key on. The caller records
            # a failure rather than reaching here.
            raise ValueError("an analysis needs a non-empty asset digest")

        shard = asset_hash[:SHARD]

        return (
            self.root
            / version
            / configuration.replace("+", "_").replace("/", "_")
            / shard
            / f"{asset_hash}.json"
        )

    def load(
        self,
        asset_hash: str,
        version: str,
        configuration: str,
    ) -> VisualAnalysis | None:
        """
        A stored analysis, if one is current.

        Returns None for anything unreadable rather than raising. A
        cache that cannot be read is a cache that will be rewritten, and
        refusing to start because a file is malformed would turn a
        nuisance into a blocker.
        """

        try:
            path = self.path_for(asset_hash, version, configuration)

        except ValueError:
            return None

        if not path.is_file():
            return None

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))

        except (OSError, json.JSONDecodeError):
            return None

        if not isinstance(payload, dict):
            return None

        try:
            analysis = VisualAnalysis.model_validate(payload)

        except ValidationError:
            return None

        # A stored analysis that names a different asset is not an
        # analysis of this one, whatever its filename says.
        if analysis.source_asset_hash != asset_hash:
            return None

        return analysis

    def save(
        self,
        analysis: VisualAnalysis,
        version: str,
        configuration: str,
    ) -> Path:
        """
        Write one analysis, atomically.

        Atomic because a half-written JSON file is not an analysis, it is
        a file that parses as something else on the next run. Written to
        a temporary file in the same directory and moved into place.

        Locked, because writing is genuinely shared. Four posts showing
        the same three slides is ordinary in an archive with duplicates,
        so all four analyse the same digests on a cold cache and all four
        replace onto the same paths. On Windows that race fails with
        ``PermissionError``, which is how this was found: two posts
        succeeded and two were lost to a filesystem error that had
        nothing to do with either of them.

        Locked per digest rather than globally, so two posts reading
        different pictures still write in parallel.
        """

        path = self.path_for(
            analysis.source_asset_hash, version, configuration
        )

        with _lock_for(path):
            path.parent.mkdir(parents=True, exist_ok=True)

            handle = tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f"{path.name}.",
                suffix=".tmp",
                delete=False,
            )

            try:
                with handle:
                    json.dump(
                        analysis.model_dump(mode="json"),
                        handle,
                        indent=2,
                        ensure_ascii=False,
                    )

                _replace(handle.name, path)

            except BaseException:
                try:
                    os.unlink(handle.name)

                except OSError:
                    pass

                raise

        return path

    def count(self, version: str, configuration: str) -> int:
        """How many analyses are stored for this processor."""

        directory = (
            self.root
            / version
            / configuration.replace("+", "_").replace("/", "_")
        )

        if not directory.is_dir():
            return 0

        return sum(1 for _ in directory.rglob("*.json"))

    def prune(self, version: str, configuration: str) -> int:
        """
        Remove stored analyses for a superseded processor version.

        Offered rather than automatic. Deleting a cache on upgrade is
        safe -- it can be rebuilt from the archive -- but "safe to
        rebuild" and "should be rebuilt now" are different decisions,
        and a run that silently threw away a cache would make its own
        cost unpredictable.
        """

        removed = 0

        for path in self.root.glob("*/*/**/*.json"):
            relative = path.relative_to(self.root)

            if relative.parts[0] == version:
                continue

            try:
                path.unlink()
                removed += 1

            except OSError:
                continue

        return removed

    def summary(self, version: str, configuration: str) -> dict[str, int]:
        """Counts by processor, for a report or a plan."""

        summary: dict[str, int] = {}

        if not self.root.is_dir():
            return summary

        for directory in sorted(self.root.iterdir()):
            if not directory.is_dir():
                continue

            for configuration_dir in sorted(directory.iterdir()):
                if not configuration_dir.is_dir():
                    continue

                count = sum(1 for _ in configuration_dir.rglob("*.json"))

                if count:
                    summary[f"{directory.name}/{configuration_dir.name}"] = count

        return summary


def store_root() -> Path:
    """Where the cache lives. Under build/, git-ignored, disposable."""

    return Path("build") / "visual"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = ["VisualStore", "store_root"]
