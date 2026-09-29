from pathlib import Path

from src.workers.worker import process_enrichment_job


def main() -> None:
    output_path = Path(
        "data/results/sample_001_enriched.json"
    )

    result = process_enrichment_job(
        post_directory="data/posts/sample_001",
        output_path=output_path,
    )

    print()
    print("=== WORKER TEST ===")
    print(f"Result: {result}")
    print(f"Exists: {result.exists()}")

    assert result.exists()
    assert result.stat().st_size > 0

    print()
    print("Worker job OK")


if __name__ == "__main__":
    main()