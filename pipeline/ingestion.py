"""Log ingestion helpers."""

from __future__ import annotations

from pathlib import Path
from typing import List


def read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return handle.readlines()
