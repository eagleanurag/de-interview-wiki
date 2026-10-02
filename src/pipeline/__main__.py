"""
``python -m src.pipeline`` entry point.

Kept separate from the package body so that importing the pipeline --
which tests and other modules do -- never parses arguments or runs a
stage.
"""

from __future__ import annotations

from src.pipeline.cli import main


if __name__ == "__main__":
    raise SystemExit(main())