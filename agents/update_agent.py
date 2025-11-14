"""Conflict resolution agent mirroring the original prompt design."""

from __future__ import annotations

from agentic_parser.agents.base_agent import BaseAgent, AgentError
from agentic_parser.interface.interaction_service import InteractionService
from agentic_parser.llm.client import Message
from agentic_parser.knowledge.template_store import TemplateRecord
from agentic_parser.utils.types import ProcessedLogLine


class UpdateAgent(BaseAgent):
    SYSTEM_PROMPT = (
        BaseAgent.COMMON_KNOWLEDGE
        + "\n\nYour role: resolve conflicts between regex templates while preserving STRUCTURE as literal text and keeping "
        "BUSINESS DATA variables only where appropriate. Respond with JSON only."
    )

    def __init__(self, client, interaction_service: InteractionService) -> None:
        super().__init__(client)
        self.interaction_service = interaction_service

    def resolve_conflict(
        self,
        *,
        candidate: TemplateRecord,
        sample: ProcessedLogLine,
        context: str,
        conflicts: list[tuple[TemplateRecord, str]],
    ) -> dict:
        prompt = self._build_conflict_prompt(candidate=candidate, sample=sample, conflicts=conflicts, context=context)
        messages = [
            Message(role="system", content=self.SYSTEM_PROMPT),
            Message(role="user", content=prompt),
        ]
        try:
            data, _, _ = self._call_json(messages, error_description="Conflict resolver must return JSON")
        except AgentError:
            self.interaction_service.enqueue(
                task_type="template_conflict",
                description="Conflict resolution failed (invalid JSON)",
                payload={
                    "candidate": candidate.to_dict(),
                    "sample": sample.transformed,
                    "conflicts": [rec.to_dict() for rec, _ in conflicts],
                },
            )
            raise
        self._log(f"decision={data.get('decision')} reasoning={data.get('reasoning', '')}")
        return data

    def _build_conflict_prompt(
        self,
        *,
        candidate: TemplateRecord,
        sample: ProcessedLogLine,
        conflicts: list[tuple[TemplateRecord, str]],
        context: str,
    ) -> str:
        conflicts_desc = "\n".join(
            (
                f"- template_id: {rec.template_id}\n"
                f"  regex: {rec.regex}\n"
                f"  example: {example}"
            )
            for rec, example in conflicts
        )
        return (
            f"CONFLICT DETECTED:\n"
            f"{context}\n"
            f"Your regex pattern matches not only your example but also examples from existing templates.\n"
            f"This creates ambiguity: the same log line could match multiple templates.\n\n"
            f"Your template:\n"
            f"  regex: {candidate.regex}\n"
            f"  example: {sample.transformed}\n"
            f"  your reasoning: {candidate.notes}\n\n"
            f"Conflicting templates (your regex also matches their examples):\n"
            f"{conflicts_desc}\n\n"
            "Analyze the conflict and choose one decision:\n\n"
            "1. replace_conflicting:\n"
            "   Use when the candidate template correctly identifies business variables that conflicting templates incorrectly hardcoded.\n"
            "   Result: Delete/deactivate all conflicting templates, use the candidate template instead.\n"
            "   WARNING: Only use this when the captured position is truly a BUSINESS VARIABLE.\n"
            "   Return JSON:\n"
            "{\n"
            '  "reasoning": "explanation of why this decision is correct",\n'
            '  "decision": "replace_conflicting",\n'
            '  "new_regex": "the regex to use (candidate as-is)",\n'
            '  "replaced_ids": ["template_id1", "template_id2"]\n'
            "}\n\n"
            "2. refine_candidate:\n"
            "   Use when the candidate template is overly generalized, capturing structural constants as variables.\n"
            "   Result: Adjust the candidate regex to be more specific so it doesn't conflict.\n"
            "   Return JSON:\n"
            "{\n"
            '  "reasoning": "explanation of why this decision is correct",\n'
            '  "decision": "refine_candidate",\n'
            '  "new_regex": "the new more specific regex to use"\n'
            "}\n"
            "Respond with JSON only."
        )
