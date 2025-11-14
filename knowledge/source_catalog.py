"""Catalog storing routing decisions and metadata."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class SourceDescriptor:
    source_id: str
    device_type: str
    vendor: str
    metadata: Dict[str, str]


class SourceCatalog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: Dict[str, SourceDescriptor] = {}
        if path.exists():
            self._load()

    def _load(self) -> None:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        for source_id, data in payload.items():
            self.entries[source_id] = SourceDescriptor(
                source_id=source_id,
                device_type=data.get("device_type", "unknown"),
                vendor=data.get("vendor", "unknown"),
                metadata=data.get("metadata", {}),
            )

    def register(self, descriptor: SourceDescriptor) -> None:
        self.entries[descriptor.source_id] = descriptor

    def save(self) -> None:
        snapshot = {source_id: asdict(desc) for source_id, desc in self.entries.items()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
