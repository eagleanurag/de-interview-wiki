"""
Build the synthetic fixture set.

Run to regenerate the files under ``tests/fixtures/``:

    python -m tests.build_fixtures

Everything here is fabricated. No fixture contains a real credential, a
real person, or content copied from any source. The PDFs and PNGs are
constructed byte by byte so the media pipeline is exercised on true
files rather than on stubs that merely have the right extension.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path


ROOT = Path(__file__).parent / "fixtures"


def png(width: int = 24, height: int = 16, colour=(30, 90, 200)) -> bytes:
    """A real PNG, written with zlib rather than copied."""

    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload

        return (
            len(payload).to_bytes(4, "big")
            + body
            + zlib.crc32(body).to_bytes(4, "big")
        )

    header = (
        width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + bytes([8, 2, 0, 0, 0])
    )

    row = bytes([0]) + bytes(colour) * width
    raw = row * height

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def pdf(lines: list[str], *, broken: bool = False) -> bytes:
    """
    A real single-page PDF holding the given text.

    ``broken`` emits a file with a valid header and truncated structure,
    which is what a failed download actually looks like, so the failure
    path is exercised on a real file rather than an empty one.
    """

    content = b"BT /F1 12 Tf 72 720 Td 14 TL\n"

    for index, line in enumerate(lines):
        escaped = (
            line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        )
        prefix = b"" if index == 0 else b"T* "
        content += prefix + f"({escaped}) Tj\n".encode()

    content += b"ET"

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n"
        + content
        + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = []

    for number, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + obj + b"\nendobj\n"

    xref_at = len(out)

    out += b"xref\n0 " + str(len(objects) + 1).encode() + b"\n"
    out += b"0000000000 65535 f \n"

    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()

    out += (
        b"trailer\n<< /Size "
        + str(len(objects) + 1).encode()
        + b" /Root 1 0 R >>\nstartxref\n"
        + str(xref_at).encode()
        + b"\n%%EOF\n"
    )

    if broken:
        # A header and a truncated body: what a half-written download
        # leaves behind.
        return bytes(out[: len(out) // 3])

    return bytes(out)


def write(relative: str, content: bytes | str) -> Path:
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(content, str):
        path.write_text(content, encoding="utf-8")
    else:
        path.write_bytes(content)

    return path


def build() -> list[str]:
    written: list[str] = []

    # 1. A plain text interview post.
    write(
        "incoming/plain-text-interview-post/post.txt",
        "What is a window function, and when would you reach for one "
        "instead of a GROUP BY?\n\n"
        "A window function computes across a set of related rows "
        "without collapsing them, so the original rows stay visible "
        "alongside the aggregate. That matters when each row needs a "
        "comparison against its own partition: a running total, a "
        "previous-value delta, or a rank within a group.\n",
    )
    written.append("plain-text-interview-post")

    # 2. A technical post in Markdown.
    write(
        "incoming/technical-markdown-post/content.md",
        "# Delta Lake compaction\n\n"
        "Compaction rewrites small files into fewer, larger files.\n\n"
        "## Why it matters\n\n"
        "Small files defeat parallelism: every task pays planning "
        "overhead before it reads anything, so throughput falls even "
        "when the data volume is unchanged.\n\n"
        "## When to compact\n\n"
        "- Many small files accumulate after a streaming write.\n"
        "- A query is scanning file counts rather than bytes.\n\n"
        "## Trade-off\n\n"
        "Compaction rewrites data, so it costs compute and I/O. Doing "
        "it continuously can cost more than the reads it saves.\n",
    )
    written.append("technical-markdown-post")

    # 3. A screenshot question with an image.
    write(
        "incoming/screenshot-question/screenshot.png",
        png(48, 30, (200, 60, 60)),
    )
    write(
        "incoming/screenshot-question/notes.txt",
        "Screenshot of an interview screen. The question text has to be "
        "read off the image, because the capture is a picture.\n",
    )
    written.append("screenshot-question")

    # 4. PDF interview notes.
    write(
        "incoming/pdf-interview-notes/notes.pdf",
        pdf(
            [
                "Slowly changing dimension type 2",
                "",
                "SCD2 keeps the history: a row is closed with an end",
                "date when its tracked attributes change, and a new row",
                "is inserted with the new values.",
                "",
                "A row is never updated in place, so a point-in-time",
                "query reconstructs any past state.",
            ]
        ),
    )
    written.append("pdf-interview-notes")

    # 5. An exact duplicate of another bundle.
    write(
        "incoming/duplicate-of-plain-text/post.txt",
        "What is a window function, and when would you reach for one "
        "instead of a GROUP BY?\n\n"
        "A window function computes across a set of related rows "
        "without collapsing them, so the original rows stay visible "
        "alongside the aggregate. That matters when each row needs a "
        "comparison against its own partition: a running total, a "
        "previous-value delta, or a rank within a group.\n",
    )
    written.append("duplicate-of-plain-text")

    # 6. A near duplicate: same topic, different wording.
    write(
        "incoming/near-duplicate/content.md",
        "# Window functions versus GROUP BY\n\n"
        "A window function aggregates across a partition but keeps the "
        "rows, where GROUP BY collapses them into one row per group. "
        "Use a window when every row still needs to be visible next to "
        "its own aggregate.\n",
    )
    written.append("near-duplicate")

    # 7. Irrelevant content.
    write(
        "incoming/irrelevant-social/post.txt",
        "Thrilled to share that I have started a new journey. Grateful "
        "to everyone who helped along the way. See you at the next "
        "meetup!\n",
    )
    written.append("irrelevant-social")

    # 8. Malformed media.
    write("incoming/malformed-media/broken.pdf", pdf(["x"], broken=True))
    write(
        "incoming/malformed-media/notes.txt",
        "This bundle carries a document that failed to download, so the "
        "extraction stage has to survive it.\n",
    )
    written.append("malformed-media")

    # 9. A structured capture with declared media.
    write(
        "incoming/structured-capture/capture.json",
        json.dumps(
            {
                "id": "capture-spark-skew",
                "text": (
                    "Data skew means one task in a stage does far more "
                    "work than the others, so the stage waits for it. "
                    "Salting a hot key, or splitting it across several "
                    "keys, spreads the work."
                ),
                "url": "https://example.invalid/capture-spark-skew",
                "author": "fixture-author",
                "published_at": "2026-02-02T12:00:00+00:00",
                "media": ["diagram.png"],
            },
            indent=2,
        ),
    )
    write(
        "incoming/structured-capture/diagram.png",
        png(60, 40, (40, 160, 90)),
    )
    written.append("structured-capture")

    # 10. A JSONL export, one post per line.
    write(
        "incoming/jsonl-export/export.jsonl",
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "export-1",
                        "text": "Z-Ordering co-locates related rows so "
                        "a predicate can skip whole files.",
                    }
                ),
                json.dumps(
                    {
                        "id": "export-2",
                        "text": "Partitioning is about pruning; "
                        "clustering is about ordering. They answer "
                        "different questions.",
                    }
                ),
            ]
        ),
    )
    written.append("jsonl-export")

    written.extend(build_saved_items())

    return written


# ---------------------------------------------------------------------
# Saved Items
# ---------------------------------------------------------------------

#: The cases the saved-items workflow has to handle. Each is a thing a
#: person actually saved, not a shape a test invented.
SAVED_ITEMS = {
    "delta-lake": "https://www.linkedin.com/posts/alice_delta-lake-101",
    "sql-window": "https://www.linkedin.com/posts/bob_sql-window-202",
    "pyspark-joins": "https://www.linkedin.com/pulse/carol_pyspark-303",
    "databricks": "https://www.linkedin.com/posts/dan_databricks-404",
    "data-factory": "https://www.linkedin.com/posts/erin_data-factory-505",
    "architecture": "https://www.linkedin.com/posts/frank_architecture-606",
    "hiring": "https://www.linkedin.com/posts/grace_hiring-707",
    "image-only": "https://www.linkedin.com/posts/heidi_diagram-808",
    "document": "https://www.linkedin.com/posts/ivan_backpressure-909",
    "metadata-only": (
        "https://www.linkedin.com/posts/judy_never-captured-1010"
    ),
    "corrupt": "https://www.linkedin.com/posts/ken_corrupt-1110",
    # The same link as delta-lake, written the way a second export does.
    # A row and nothing else: a repeat in a list is a repeat, and giving
    # it a folder would pretend it were a different post.
    "duplicate": "https://www.linkedin.com/posts/alice_delta-lake-101",
}

UNCLAIMED_URL = "https://www.linkedin.com/posts/kim_unclaimed-1111"


def saved_item_ids() -> dict[str, str]:
    """
    The source id of each fixture item.

    Derived rather than hardcoded, so a change to URL normalization
    cannot leave the fixture folders named after ids nothing claims.
    """
    from src.ingestion.saved_items.urls import normalize_linkedin_url

    return {
        name: normalize_linkedin_url(url).source_id
        for name, url in SAVED_ITEMS.items()
    }


def build_saved_items() -> list[str]:
    """A saved-items drop zone covering every case the phase names."""
    written: list[str] = []

    base = "saved-items"

    ids = saved_item_ids()

    def folder(name: str) -> str:
        return f"{base}/captures/{ids[name].replace(':', '-')}"

    rows = ["URL,Saved Date,Title,Author,Notes"]

    for number, (name, url) in enumerate(SAVED_ITEMS.items()):
        rows.append(
            f"{url},2026-01-{number + 1:02d},"
            f"{name.replace('-', ' ').title()},Fixture Author,note {name}"
        )

    write(f"{base}/manifest.csv", "\n".join(rows) + "\n")

    # 1. A technical Data Engineering post, as Markdown.
    write(
        f"{folder('delta-lake')}/content.md",
        "# Delta Lake gives a data lake ACID guarantees\n\n"
        "Delta Lake stores a transaction log beside the data, so a "
        "commit is atomic and a reader never sees a half-written "
        "table. Schema evolution lets you add, rename or drop a column "
        "without rewriting the files that are already there. Time "
        "travel queries an earlier version of the table by version "
        "number, which is how you compare a result against what the "
        "table looked like when the job ran.\n\n"
        "On Databricks these tables sit on Apache Spark, and the file "
        "layout is what makes predicate pushdown and file skipping "
        "possible at all.\n",
    )

    # 2. A SQL post, as a saved page that claims itself by URL.
    write(
        f"{folder('sql-window')}/capture.json",
        json.dumps(
            {
                "url": SAVED_ITEMS["sql-window"],
                "title": "Window functions beat GROUP BY",
                "notes": "worth re-reading before the interview",
            },
            indent=2,
        ),
    )
    write(
        f"{folder('sql-window')}/page.html",
        "<!doctype html><html><head>"
        '<meta property="og:title" content="Window functions beat '
        'GROUP BY">'
        '<meta property="og:description" content="A window function '
        "computes across a set of related rows without collapsing them, "
        "so each row keeps a comparison against its own partition. That "
        "is what a running total, a period-over-period delta and a rank "
        'within a group all need.">'
        '<meta name="author" content="Bob Fixture">'
        '<meta property="article:published_time" '
        'content="2026-01-02T09:00:00Z">'
        "</head><body><nav>Home Jobs Sign in</nav></body></html>",
    )

    # 3. A PySpark post, as plain text.
    write(
        f"{folder('pyspark-joins')}/content.txt",
        "A BroadcastHashJoin in PySpark avoids a shuffle when one side "
        "of the join is small enough to sit in memory on every "
        "executor. Spark picks it automatically, but it is worth "
        "checking the plan, because a table that was small yesterday "
        "can stop being small after a backfill.\n\n"
        "A SortMergeJoin is the other side of the trade: it streams, "
        "so it handles either side being large, at the cost of sorting "
        "both. A skewed key defeats it either way, and salting the hot "
        "key is the usual repair.\n",
    )

    # 4. A Databricks post, as Markdown with a diagram beside it.
    write(
        f"{folder('databricks')}/content.md",
        "# What Unity Catalog actually changes\n\n"
        "Unity Catalog is a catalogue over the tables in a workspace: "
        "it records which principal may read which table, and it keeps "
        "lineage from a dashboard back to the columns it reads. On "
        "Databricks that replaces per-table grants scattered across "
        "notebooks, and it is what makes a table safe to share outside "
        "the team that wrote it.\n",
    )
    write(
        f"{folder('databricks')}/lineage.png", png(48, 32, (120, 60, 200))
    )

    # 5. An Azure Data Factory post, claimed by a capture.json URL.
    write(
        f"{folder('data-factory')}/capture.json",
        json.dumps({"url": SAVED_ITEMS["data-factory"]}),
    )
    write(
        f"{folder('data-factory')}/page.html",
        "<!doctype html><html><head>"
        '<meta property="og:description" content="An Azure Data '
        "Factory pipeline copies rows from an on-premises SQL Server "
        "into a Delta Lake table through a self-hosted integration "
        'runtime, and retries a failed slice on its own.">'
        "</head><body></body></html>",
    )

    # 6. An architecture scenario, as Markdown.
    write(
        f"{folder('architecture')}/content.md",
        "# Ingesting a change stream without double counting\n\n"
        "The scenario: a Kafka topic carries change data capture events "
        "from a transactional database, and they have to land in a "
        "Delta Lake table exactly once. Writing to Delta Lake gives "
        "atomic commits, so a replayed batch converges on the same "
        "table rather than duplicating rows, which is the property that "
        "makes at-least-once delivery safe here.\n\n"
        "The part that goes wrong in practice is offset management. A "
        "consumer that records its offset after writing, rather than in "
        "the same transaction as the write, re-reads a batch on a crash "
        "and the idempotent table absorbs it. One that records before "
        "writing loses it instead, and nothing notices until someone "
        "queries for a range that should have data.\n",
    )

    # 7. A non-technical post. Real, saved, and not interview material.
    write(
        f"{folder('hiring')}/content.md",
        "# We are hiring two data engineers\n\n"
        "Our team is growing and we are looking for two data engineers "
        "to join a small platform group. There is a competitive salary, "
        "hybrid working and a learning budget. If that sounds like you, "
        "send a note and we will arrange a call.\n",
    )

    # 8. An image-only capture, which is most of what people save.
    write(
        f"{folder('image-only')}/screenshot.png",
        png(80, 60, (20, 150, 120)),
    )

    # 9. A document capture, with text the pipeline can read.
    write(
        f"{folder('document')}/document.pdf",
        pdf(
            [
                "Consumer lag is how far a Kafka consumer group is",
                "behind its committed offsets. It grows when processing",
                "takes longer than arrivals and shrinks when the group",
                "catches up.",
                "Backpressure shows up as lag, but the cause is usually",
                "downstream: a slow sink, a rebalance storm, or a",
                "partition that is far hotter than its neighbours.",
            ]
        ),
    )

    # 10. A corrupt capture, which must be reported and not imported.
    write(
        f"{folder('corrupt')}/document.pdf",
        pdf(
            ["A deliberately broken file for the failure path."],
            broken=True,
        ),
    )
    write(
        f"{folder('corrupt')}/notes.md",
        "The notes survived even though the document did not.\n",
    )

    # 11. A capture nothing claims, which does name its post.
    write(
        f"{base}/captures/unclaimed-capture/capture.json",
        json.dumps({"url": UNCLAIMED_URL}, indent=2),
    )
    write(
        f"{base}/captures/unclaimed-capture/content.md",
        "# A capture for a post that was never in the list\n\n"
        "The user captured this before the export was refreshed.\n",
    )

    # 12. A capture nothing claims and that says nothing about its post.
    write(
        f"{base}/captures/no-claim/notes.md",
        "Some notes with no way to tell which saved post they are.\n",
    )

    for name in SAVED_ITEMS:
        written.append(f"{base}/captures/{name}")

    written.append(f"{base}/manifest.csv")
    written.append(f"{base}/captures/unclaimed-capture")
    written.append(f"{base}/captures/no-claim")

    return written


def main() -> int:
    written = build()

    print(f"fixtures written under {ROOT}")

    for name in written:
        print(f"  {name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
