"""Centralized metrics storage for the new pipeline."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class PipelineCounters:
    processed_lines: int = 0
    matched_lines: int = 0
    learned_templates: int = 0
    escalations: int = 0

    def asdict(self) -> Dict[str, int]:
        return asdict(self)


class MetricsStore:
    def __init__(self) -> None:
        self.pipeline = PipelineCounters()
        self.token_usage: Dict[str, int] = {}

    def increment(self, **values: int) -> None:
        for key, delta in values.items():
            if hasattr(self.pipeline, key):
                setattr(self.pipeline, key, getattr(self.pipeline, key) + delta)

    def add_tokens(self, provider: str, tokens: int) -> None:
        self.token_usage[provider] = self.token_usage.get(provider, 0) + tokens

    def snapshot(self) -> Dict:
        return {
            "pipeline": self.pipeline.asdict(),
            "tokens": dict(self.token_usage),
        }
