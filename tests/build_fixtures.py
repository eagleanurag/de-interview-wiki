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

    return written


def main() -> int:
    written = build()

    print(f"fixtures written under {ROOT}")

    for name in written:
        print(f"  {name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
