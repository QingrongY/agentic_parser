"""Agent responsible for deriving parsing templates via LLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from agentic_parser.agents.base_agent import BaseAgent, AgentError
from agentic_parser.llm.client import LLMClient, Message
from agentic_parser.utils.types import ProcessedLogLine


@dataclass
class TemplateProposal:
    regex: str
    reasoning: str


class TemplateAgent(BaseAgent):
    SYSTEM_PROMPT = (
        BaseAgent.COMMON_KNOWLEDGE
        + "\n\nYour role: derive a regex template that captures every BUSINESS DATA value with named groups "
        "using (?P<name>.*) and keeps all STRUCTURE literal. The regex must match the entire log line. "
        "If JSON payloads appear, capture the whole object as one variable without inspecting its fields.\n"
        'Respond with JSON {"reasoning": str, "regex": str}.'
    )

    def __init__(self, client: LLMClient) -> None:
        super().__init__(client)
        self.conversation_history: List[Message] = []
        self.last_response: str = ""

    def derive(self, samples: Iterable[ProcessedLogLine], *, context: str) -> TemplateProposal:
        sample_list = [sample for sample in samples if sample.transformed]
        if not sample_list:
            raise AgentError("no content to derive template from")
        primary = sample_list[0]
        user_sections: List[str] = [
            f"Context: {context}",
            "Generate a regex template for this log line:",
            f"Log line: {primary.transformed}",
        ]
        conversation = [
            Message(role="system", content=self.SYSTEM_PROMPT),
            Message(
                role="user",
                content="\n\n".join(
                    user_sections
                    + [
                        "Remember: Distinguish between business data variables and structural elements. "
                        "Return ONLY the JSON object."
                    ]
                ),
            ),
        ]
        data, response, history = self._call_json(
            conversation,
            error_description="Template derivation must return JSON",
        )
        self.last_response = response
        self.conversation_history = history
        regex = data.get("regex")
        if not isinstance(regex, str) or not regex.strip():
            raise AgentError("template agent missing regex")
        return TemplateProposal(regex=regex.strip(), reasoning=data.get("reasoning", ""))

    def follow_up(self, user_prompt: str) -> dict:
        if not self.conversation_history:
            raise AgentError("no existing conversation to extend")
        conversation = list(self.conversation_history) + [Message(role="user", content=user_prompt)]
        data, response, history = self._call_json(
            conversation,
            error_description="Template follow-up must return JSON",
        )
        self.last_response = response
        self.conversation_history = history
        return data
