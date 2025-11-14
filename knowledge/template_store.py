"""Template storage for the new agentic parser."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from agentic_parser.utils.types import ProcessedLogLine


@dataclass
class TemplateRecord:
    template_id: str
    source_id: str
    regex: str
    notes: str = ""
    is_active: bool = True

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "TemplateRecord":
        return cls(
            template_id=data["template_id"],
            source_id=data["source_id"],
            regex=data["regex"],
            notes=data.get("notes", ""),
            is_active=data.get("is_active", True),
        )


class TemplateLibrary:
    def __init__(self, source_id: str, path: Path) -> None:
        self.source_id = source_id
        self.path = path
        self.templates: Dict[str, TemplateRecord] = {}
        self._compiled: Dict[str, re.Pattern] = {}
        self._sequence = 1
        if path.exists():
            self._load()

    def _load(self) -> None:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        for data in payload.get("templates", []):
            record = TemplateRecord.from_dict(data)
            self.templates[record.template_id] = record
            self._compiled[record.template_id] = re.compile(record.regex)
            try:
                suffix = int(record.template_id.rsplit("-", 1)[-1])
            except ValueError:
                continue
            self._sequence = max(self._sequence, suffix + 1)

    def save(self) -> None:
        snapshot = {
            "source_id": self.source_id,
            "templates": [record.to_dict() for record in self.templates.values()],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")

    def allocate_id(self) -> str:
        template_id = f"{self.source_id}-{self._sequence:04d}"
        self._sequence += 1
        return template_id

    def add(self, record: TemplateRecord) -> TemplateRecord:
        if not record.template_id:
            record.template_id = self.allocate_id()
        self.templates[record.template_id] = record
        self._compiled[record.template_id] = re.compile(record.regex)
        return record

    def deactivate(self, template_id: str) -> None:
        if template_id in self.templates:
            self.templates[template_id].is_active = False

    def match(self, processed: ProcessedLogLine) -> Optional[Tuple[TemplateRecord, Dict[str, str]]]:
        for template_id, record in self.templates.items():
            if not record.is_active:
                continue
            pattern = self._compiled.get(template_id)
            if not pattern:
                pattern = re.compile(record.regex)
                self._compiled[template_id] = pattern
            match = pattern.fullmatch(processed.transformed)
            if match:
                return record, match.groupdict()
        return None


class TemplateStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._cache: Dict[str, TemplateLibrary] = {}

    def library(self, source_id: str) -> TemplateLibrary:
        if source_id not in self._cache:
            path = self.root / f"{source_id}.json"
            self._cache[source_id] = TemplateLibrary(source_id, path)
        return self._cache[source_id]

    def save_all(self) -> None:
        for library in self._cache.values():
            library.save()
