"""Matching utilities reused in multiple phases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from knowledge.template_store import TemplateLibrary
from utils.types import ProcessedLogLine


@dataclass
class MatchResult:
    line_number: int
    template_id: Optional[str]
    variables: Dict[str, str]
    raw: str


def match_all(library: TemplateLibrary, lines: Iterable[ProcessedLogLine]) -> List[MatchResult]:
    results: List[MatchResult] = []
    for idx, line in enumerate(lines, start=1):
        match = library.match(line)
        if match:
            record, variables = match
            results.append(MatchResult(idx, record.template_id, variables, line.raw))
        else:
            results.append(MatchResult(idx, None, {}, line.raw))
    return results
