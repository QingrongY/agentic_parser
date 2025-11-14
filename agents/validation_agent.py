"""Validation agent ensures templates meet structural requirements."""

from __future__ import annotations

from dataclasses import dataclass
import re

from typing import Iterable, List

from agents.base_agent import BaseAgent, AgentError
from llm.client import Message
from utils.types import ProcessedLogLine


@dataclass
class ValidationReport:
    approved: bool
    issues: list[str]
    suggestions: list[str]
    reasoning: str


class ValidationAgent(BaseAgent):
    SYSTEM_PROMPT = (
        BaseAgent.COMMON_KNOWLEDGE
        + "\n\nYour role: review regex templates to ensure STRUCTURE stays literal and only BUSINESS DATA is captured. "
        "When rejecting, provide concrete fix suggestions that can be applied directly. "
        'Respond with JSON {"approved": bool, "reasoning": str, "issues": [], "suggestions": []}.'
    )

    def __init__(self, client) -> None:
        super().__init__(client)
        self._directives: List[str] = []

    def set_directives(self, directives: Iterable[str]) -> None:
        self._directives = [item.strip() for item in directives if item and item.strip()]

    def add_directive(self, directive: str) -> None:
        cleaned = directive.strip()
        if cleaned:
            self._directives.append(cleaned)

    def review(self, regex: str, sample: ProcessedLogLine, context: str) -> ValidationReport:
        prompt = self._build_prompt(regex=regex, sample=sample, context=context)
        messages = [
            Message(role="system", content=self.SYSTEM_PROMPT),
            Message(role="user", content=prompt),
        ]
        data, _, _ = self._call_json(messages, error_description="Validator must return JSON verdict")
        approved = bool(data.get("approved"))
        issues = data.get("issues") or []
        if not isinstance(issues, list):
            issues = [str(issues)]
        suggestions = data.get("suggestions") or []
        if not isinstance(suggestions, list):
            suggestions = [str(suggestions)]
        reasoning = data.get("reasoning", "")
        report = ValidationReport(approved=approved, issues=issues, suggestions=suggestions, reasoning=reasoning)
        return report

    def _build_prompt(self, regex: str, sample: ProcessedLogLine, context: str) -> str:
        captures_desc = self._describe_captures(regex, sample)
        directives = ""
        if self._directives:
            directives_list = "\n".join(f"  - {item}" for item in self._directives)
            directives = f"\nAdditional instructions:\n{directives_list}\n"
        return (
            "TEMPLATE REVIEW REQUEST\n\n"
            f"Context:\n"
            f"  {context}\n\n"
            f"Template to review:\n"
            f"  regex: {regex}\n\n"
            f"Example log line:\n"
            f"  raw: {sample.raw}\n"
            f"  transformed: {sample.transformed}\n\n"
            f"Captured variables:\n"
            f"{captures_desc}\n\n"
            f"{directives}"
            "Task: Review this template and determine if it correctly distinguishes "
            "business variables from structural constants. Return JSON only."
        )

    def _describe_captures(self, regex: str, sample: ProcessedLogLine) -> str:
        try:
            pattern = re.compile(regex)
            match = pattern.fullmatch(sample.transformed)
        except re.error:
            return "  (regex failed to compile)"
        if not match:
            return "  (regex does not match sample)"
        captures = match.groupdict()
        if not captures:
            return "  (no named groups)"
        return "\n".join(f"  - {name}: '{value}'" for name, value in captures.items())
