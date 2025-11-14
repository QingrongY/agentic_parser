"""Shared dataclasses used across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessedLogLine:
    raw: str
    transformed: str
