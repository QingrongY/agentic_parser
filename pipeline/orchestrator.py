"""Task orchestrator implementing the three-phase flow."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List

from agentic_parser.agents.base_agent import BaseAgent
from agentic_parser.agents.error_agent import ErrorAgent
from agentic_parser.agents.router_agent import RouterAgent
from agentic_parser.agents.template_agent import TemplateAgent
from agentic_parser.agents.validation_agent import ValidationAgent
from agentic_parser.agents.repair_agent import RepairAgent
from agentic_parser.agents.update_agent import UpdateAgent
from agentic_parser.interface.interaction_service import InteractionService
from agentic_parser.knowledge.metrics import MetricsStore
from agentic_parser.knowledge.source_catalog import SourceCatalog, SourceDescriptor
from agentic_parser.knowledge.template_store import TemplateStore
from agentic_parser.llm.client import LLMClient
from agentic_parser.pipeline.ingestion import read_lines
from agentic_parser.pipeline.learning import LearningEngine
from agentic_parser.pipeline.matching import match_all
from agentic_parser.utils.preprocessing import normalize_many
from agentic_parser.utils.types import ProcessedLogLine


@dataclass
class ParseArtifacts:
    structured_output: Path
    templates_snapshot: Path
    metrics_path: Path


@dataclass
class ParseReport:
    routing: SourceDescriptor
    processed_lines: int
    matched_lines: int
    learned_templates: int
    artifacts: ParseArtifacts


class LogParsingOrchestrator:
    def __init__(
        self,
        llm_client: LLMClient,
        *,
        state_dir: Path,
        interaction_service: InteractionService | None = None,
    ) -> None:
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir = self.state_dir / "outputs"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        self.catalog = SourceCatalog(self.state_dir / "source_catalog.json")
        self.template_store = TemplateStore(self.state_dir / "template_libraries")
        self.metrics = MetricsStore()
        self.interaction_service = interaction_service or InteractionService()

        self.router = RouterAgent(llm_client)
        error_agent = ErrorAgent(llm_client)
        BaseAgent.register_error_agent(error_agent)
        template_agent = TemplateAgent(llm_client)
        validation_agent = ValidationAgent(llm_client)
        repair_agent = RepairAgent(template_agent)
        update_agent = UpdateAgent(llm_client, self.interaction_service)
        self.learning_engine = LearningEngine(
            template_agent=template_agent,
            validation_agent=validation_agent,
            repair_agent=repair_agent,
            update_agent=update_agent,
            metrics=self.metrics,
        )

    # ------------------------------------------------------------------
    def process(self, log_path: Path) -> ParseReport:
        lines = read_lines(log_path)
        processed = normalize_many(lines)
        samples = [line.raw for line in processed[:12] if line.raw]
        decision = self.router.identify(samples)
        descriptor = SourceDescriptor(
            source_id=decision.source_id,
            device_type=decision.device_type,
            vendor=decision.vendor,
            metadata={"reasoning": decision.reasoning},
        )
        self.catalog.register(descriptor)

        library = self.template_store.library(descriptor.source_id)
        example_cache: Dict[str, ProcessedLogLine] = {}
        context = f"device={descriptor.device_type}, vendor={descriptor.vendor}"

        # Phase 1 streaming match & learning
        for idx, line in enumerate(processed, start=1):
            self.metrics.increment(processed_lines=1)
            match = library.match(line)
            if match:
                record, _ = match
                example_cache.setdefault(record.template_id, line)
                self.metrics.increment(matched_lines=1)
                continue
            outcome = self.learning_engine.process_line(
                line,
                context=context,
                library=library,
                examples=example_cache,
            )
            if outcome.template_id:
                example_cache.setdefault(outcome.template_id, line)

        # Phase 2 final pass
        final_matches = match_all(library, processed)
        artifacts = self._write_outputs(log_path, final_matches, library)

        self.template_store.save_all()
        self.catalog.save()

        return ParseReport(
            routing=descriptor,
            processed_lines=len(processed),
            matched_lines=self.metrics.pipeline.matched_lines,
            learned_templates=self.metrics.pipeline.learned_templates,
            artifacts=artifacts,
        )

    def _write_outputs(self, log_path: Path, matches, library) -> ParseArtifacts:
        base = log_path.stem
        structured = self.outputs_dir / f"{base}.parsed.tsv"
        snapshot = self.outputs_dir / f"{base}.templates.json"
        metrics_path = self.outputs_dir / f"{base}.metrics.json"

        with structured.open("w", encoding="utf-8") as handle:
            handle.write("line\ttemplate_id\traw\n")
            for result in matches:
                template_id = result.template_id or ""
                safe_raw = result.raw.replace("\t", " ").replace("\n", " ")
                handle.write(f"{result.line_number}\t{template_id}\t{safe_raw}\n")

        snapshot.write_text(
            json.dumps({
                "source_id": library.source_id,
                "templates": [record.to_dict() for record in library.templates.values()],
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        metrics_path.write_text(json.dumps(self.metrics.snapshot(), indent=2), encoding="utf-8")

        return ParseArtifacts(
            structured_output=structured,
            templates_snapshot=snapshot,
            metrics_path=metrics_path,
        )
