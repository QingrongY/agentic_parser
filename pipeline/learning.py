"""Learning engine that chains dedicated agents."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List

from agents.base_agent import AgentError
from agents.template_agent import TemplateAgent
from agents.validation_agent import ValidationAgent
from agents.repair_agent import RepairAgent
from agents.update_agent import UpdateAgent
from knowledge.metrics import MetricsStore
from knowledge.template_store import TemplateLibrary, TemplateRecord
from utils.types import ProcessedLogLine


@dataclass
class LearningOutcome:
    status: str
    template_id: str | None
    detail: str


class LearningEngine:
    def __init__(
        self,
        template_agent: TemplateAgent,
        validation_agent: ValidationAgent,
        repair_agent: RepairAgent,
        update_agent: UpdateAgent,
        metrics: MetricsStore,
    ) -> None:
        self.template_agent = template_agent
        self.validation_agent = validation_agent
        self.repair_agent = repair_agent
        self.update_agent = update_agent
        self.metrics = metrics

    def process_line(
        self,
        line: ProcessedLogLine,
        *,
        context: str,
        library: TemplateLibrary,
        examples: Dict[str, ProcessedLogLine],
    ) -> LearningOutcome:
        line_num = self.metrics.pipeline.processed_lines
        self.template_agent.set_line_context(line_num)
        self.validation_agent.set_line_context(line_num)
        self.repair_agent.set_line_context(line_num)
        self.update_agent.set_line_context(line_num)

        proposal = self.template_agent.derive([line], context=context)
        report = self.validation_agent.review(proposal.regex, line, context)
        if not report.approved:
            try:
                refined_regex = self.repair_agent.refine(
                    regex=proposal.regex,
                    issues=report.issues,
                    suggestions=report.suggestions,
                    sample=line,
                    context=context,
                )
            except AgentError as exc:
                self.metrics.increment(escalations=1)
                return LearningOutcome(status="repair_failed", template_id=None, detail=str(exc))
            report = self.validation_agent.review(refined_regex, line, context)
            regex = refined_regex
        else:
            regex = proposal.regex

        if not report.approved:
            self.metrics.increment(escalations=1)
            return LearningOutcome(status="rejected", template_id=None, detail="; ".join(report.issues))

        candidate = TemplateRecord(
            template_id="",
            source_id=library.source_id,
            regex=regex,
            notes=proposal.reasoning,
        )
        conflicts = self._detect_conflicts(regex, library, examples)
        if not conflicts:
            stored = library.add(candidate)
            examples[stored.template_id] = line
            self.metrics.increment(learned_templates=1)
            return LearningOutcome(status="stored", template_id=stored.template_id, detail="stored without conflict")

        try:
            plan = self.update_agent.resolve_conflict(
                candidate=candidate,
                sample=line,
                context=context,
                conflicts=conflicts,
            )
        except AgentError as exc:
            self._escalate(candidate, line, {"error": str(exc)}, reason="conflict resolution error")
            return LearningOutcome(status="escalated", template_id=None, detail=str(exc))
        return self._apply_conflict_plan(
            plan=plan,
            candidate=candidate,
            line=line,
            context=context,
            library=library,
            examples=examples,
        )

    def _detect_conflicts(
        self,
        regex: str,
        library: TemplateLibrary,
        examples: Dict[str, ProcessedLogLine],
    ) -> List[tuple[TemplateRecord, str]]:
        try:
            pattern = re.compile(regex)
        except re.error:
            return []

        conflicts: List[tuple[TemplateRecord, str]] = []
        for template_id, record in library.templates.items():
            if not record.is_active:
                continue
            sample = examples.get(template_id)
            if not sample:
                continue
            if pattern.fullmatch(sample.transformed):
                conflicts.append((record, sample.transformed))
        return conflicts

    def _apply_conflict_plan(
        self,
        *,
        plan: dict,
        candidate: TemplateRecord,
        line: ProcessedLogLine,
        context: str,
        library: TemplateLibrary,
        examples: Dict[str, ProcessedLogLine],
    ) -> LearningOutcome:
        decision = plan.get("decision")
        new_regex = plan.get("new_regex")
        reasoning = plan.get("reasoning", "")
        if not isinstance(new_regex, str) or not new_regex.strip():
            self._escalate(candidate, line, plan, reason="missing new_regex in conflict plan")
            return LearningOutcome(status="escalated", template_id=None, detail="conflict plan missing regex")
        candidate.regex = new_regex.strip()

        validation = self.validation_agent.review(candidate.regex, line, context)
        if not validation.approved:
            self._escalate(candidate, line, plan, reason="refined regex rejected by validator")
            return LearningOutcome(status="escalated", template_id=None, detail="validator rejected refined regex")

        if decision == "replace_conflicting":
            replaced_ids = plan.get("replaced_ids") or []
            if not isinstance(replaced_ids, list):
                replaced_ids = [str(replaced_ids)]
            for template_id in replaced_ids:
                library.deactivate(template_id)
                examples.pop(template_id, None)
            stored = library.add(candidate)
            examples[stored.template_id] = line
            self.metrics.increment(learned_templates=1)
            return LearningOutcome(status="replaced", template_id=stored.template_id, detail=reasoning)

        if decision == "refine_candidate":
            stored = library.add(candidate)
            examples[stored.template_id] = line
            self.metrics.increment(learned_templates=1)
            return LearningOutcome(status="refined", template_id=stored.template_id, detail=reasoning)

        self._escalate(candidate, line, plan, reason="unsupported decision")
        return LearningOutcome(status="escalated", template_id=None, detail="unsupported conflict decision")

    def _escalate(
        self,
        candidate: TemplateRecord,
        line: ProcessedLogLine,
        plan: dict,
        reason: str,
    ) -> None:
        self.metrics.increment(escalations=1)
        self.update_agent.interaction_service.enqueue(
            task_type="template_conflict",
            description=reason,
            payload={
                "candidate": candidate.to_dict(),
                "plan": plan,
                "sample": line.transformed,
            },
        )
