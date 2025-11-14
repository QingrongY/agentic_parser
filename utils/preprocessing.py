"""Preprocessing utilities for ingestion phase."""

from __future__ import annotations

import re
from typing import Iterable, List

from utils.types import ProcessedLogLine


_WHITESPACE = re.compile(r"\s+")


def normalize(line: str) -> ProcessedLogLine:
    raw = line.rstrip("\r\n")
    transformed = _WHITESPACE.sub(" ", raw).strip()
    return ProcessedLogLine(raw=raw, transformed=transformed)


def normalize_many(lines: Iterable[str]) -> List[ProcessedLogLine]:
    return [normalize(line) for line in lines]
