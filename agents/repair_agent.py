"""Repair agent that continues the template agent conversation for refinements."""

from __future__ import annotations

from agents.base_agent import AgentError, BaseAgent
from agents.template_agent import TemplateAgent
from utils.types import ProcessedLogLine


class RepairAgent(BaseAgent):
    def __init__(self, template_agent: TemplateAgent) -> None:
        super().__init__(template_agent._client)
        self.template_agent = template_agent

    def refine(
        self,
        regex: str,
        issues: list[str],
        suggestions: list[str],
        sample: ProcessedLogLine,
        context: str,
    ) -> str:
        prompt = self._build_feedback_prompt(
            regex=regex,
            issues=issues,
            suggestions=suggestions,
            sample=sample,
            context=context,
        )
        try:
            data = self.template_agent.follow_up(prompt)
        except AgentError as exc:
            return self.retry_from_error(
                error_message=str(exc),
                sample=sample,
                context=context,
            )
        return self._extract_regex(data)

    def retry_from_error(
        self,
        *,
        error_message: str,
        sample: ProcessedLogLine,
        context: str,
    ) -> str:
        last_reply = self.template_agent.last_response or "[no previous response]"
        prompt = (
            "ERROR CORRECTION REQUEST\n"
            f"Context: {context}\n"
            f"Sample: {sample.transformed}\n\n"
            "Previous LLM response:\n"
            f"{last_reply}\n\n"
            "Processing error encountered:\n"
            f"{error_message}\n\n"
            "Analyze the issue and regenerate a correct JSON response that follows the original instructions."
        )
        data = self.template_agent.follow_up(prompt)
        return self._extract_regex(data)

    def _extract_regex(self, payload: dict) -> str:
        new_regex = payload.get("regex")
        if not isinstance(new_regex, str) or not new_regex.strip():
            raise AgentError("repair agent missing regex after follow-up")
        return new_regex.strip()

    def _build_feedback_prompt(
        self,
        regex: str,
        issues: list[str],
        suggestions: list[str],
        sample: ProcessedLogLine,
        context: str,
    ) -> str:
        issues_block = "\n".join(f"- {issue}" for issue in issues) or "- unspecified validator issue"
        suggestions_block = "\n".join(f"- {item}" for item in suggestions) or "- ensure structure stays literal and variables remain (?P<name>.*)"
        instructions = (
            "VALIDATION FEEDBACK\n"
            f"Context: {context}\n"
            f"Sample: {sample.transformed}\n\n"
            "Issues reported:\n"
            f"{issues_block}\n\n"
            "Suggested fixes:\n"
            f"{suggestions_block}\n\n"
            "Regenerate a JSON response with an improved regex that satisfies the feedback."
        )
        return instructions
