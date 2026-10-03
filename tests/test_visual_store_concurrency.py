"""
The store is written from several threads at once.

This archive makes that ordinary rather than exotic: 169 of its images
are repeats of a picture another post already carries, and one image
appears in 42 posts. On a cold cache every post holding that picture
analyses it, and every one of them writes the same file.

Found by the concurrency test rather than reasoned about: two posts
succeeded and two were lost to ``PermissionError`` on ``os.replace``,
which is Windows refusing a move onto a path another thread is moving
onto at the same moment. Nothing about either post was wrong.

Asserted structurally rather than behaviourally, and the reason is worth
stating: the race is a timing window, so a behavioural test passes or
fails depending on the scheduler. Re-running it enough times to be
reliable would make the suite slow and would still not prove the lock
is there. What can be asserted is that both mechanisms are present and
wired, because either one alone leaves a real failure behind it.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from src.visual.models import VisualAnalysis
from src.visual.store import VisualStore, _lock_for, _replace


class TestConcurrentStoreWrites:
    def test_many_threads_writing_one_digest_all_land(self, tmp_path: Path):
        """
        Every write is accounted for, and none raises.

        Not a reliable guard against the race -- the retry absorbs it --
        but it is the test that found the failure, and it would still
        catch a store that dropped writes or raised.
        """

        store = VisualStore(tmp_path / "store")

        digest = "a" * 64
        errors: list[BaseException] = []
        done = threading.Barrier(8)

        def writer(index: int) -> None:
            try:
                done.wait(timeout=30)

                store.save(
                    VisualAnalysis(
                        source_asset_hash=digest,
                        extracted_text=f"read {index}",
                    ),
                    "1",
                    "vision",
                )

            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [
            threading.Thread(target=writer, args=(index,))
            for index in range(8)
        ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join(timeout=60)

        assert errors == []

        loaded = store.load(digest, "1", "vision")

        assert loaded is not None
        assert loaded.extracted_text.startswith("read ")

    def test_one_lock_per_destination(self, tmp_path: Path):
        """
        Keyed by the file being written, not global.

        Two posts reading different pictures must not wait on each other,
        which a single global lock would make happen on every run.
        """

        one = _lock_for(Path(tmp_path / "a" / "x.json"))
        two = _lock_for(Path(tmp_path / "b" / "x.json"))
        again = _lock_for(Path(tmp_path / "a" / "x.json"))

        assert one is again
        assert one is not two

    def test_the_same_digest_takes_the_same_lock(self, tmp_path: Path):
        store = VisualStore(tmp_path / "store")

        digest = "b" * 64

        first = store.path_for(digest, "1", "vision")
        second = store.path_for(digest, "1", "vision")

        assert _lock_for(first) is _lock_for(second)

    def test_a_refused_move_is_retried_not_dropped(
        self, tmp_path: Path, monkeypatch
    ):
        """
        Windows refuses a move while anything holds the destination.

        An indexer following a few thousand fresh files does that, and
        losing an analysis to it would be a silent loss.
        """

        target = tmp_path / "target.json"
        source = tmp_path / "source.json"
        source.write_text("payload", encoding="utf-8")

        attempts = {"count": 0}
        real_replace = __import__("os").replace

        def flaky(src, dst, *args, **kw):
            attempts["count"] += 1

            if attempts["count"] < 3:
                raise PermissionError("held by another process")

            return real_replace(src, dst, *args, **kw)

        monkeypatch.setattr("src.visual.store.os.replace", flaky)
        monkeypatch.setattr("src.visual.store.time.sleep", lambda s: None)

        _replace(str(source), target)

        assert attempts["count"] == 3
        assert target.read_text(encoding="utf-8") == "payload"

    def test_a_move_that_never_succeeds_eventually_gives_up(
        self, tmp_path: Path, monkeypatch
    ):
        """
        Bounded. A file held forever must not stall the run.
        """

        source = tmp_path / "source.json"
        source.write_text("payload", encoding="utf-8")

        monkeypatch.setattr(
            "src.visual.store.os.replace",
            lambda *a, **k: (_ for _ in ()).throw(
                PermissionError("held forever")
            ),
        )
        monkeypatch.setattr("src.visual.store.time.sleep", lambda s: None)

        with pytest.raises(PermissionError):
            _replace(str(source), tmp_path / "target.json", attempts=3)

    def test_the_written_file_is_complete_json(self, tmp_path: Path):
        """
        Never half an analysis, which would parse as something else.
        """

        store = VisualStore(tmp_path / "store")

        digest = "c" * 64

        store.save(
            VisualAnalysis(
                source_asset_hash=digest,
                extracted_text="SQL JOINS",
                code_blocks=[
                    {"language": "sql", "code": "SELECT 1", "method": "vision"}
                ],
            ),
            "1",
            "vision",
        )

        path = store.path_for(digest, "1", "vision")

        payload = json.loads(path.read_text(encoding="utf-8"))

        assert payload["extracted_text"] == "SQL JOINS"
        assert payload["code_blocks"][0]["code"] == "SELECT 1"

    def test_no_temporary_files_survive_concurrent_writes(
        self, tmp_path: Path
    ):
        store = VisualStore(tmp_path / "store")

        def writer(index: int) -> None:
            store.save(
                VisualAnalysis(
                    source_asset_hash=f"{index:064d}",
                    extracted_text=str(index),
                ),
                "1",
                "vision",
            )

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join(timeout=60)

        assert not list(store.root.rglob("*.tmp"))
