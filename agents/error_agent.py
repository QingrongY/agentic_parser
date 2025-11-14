"""Generic error-handling agent that produces correction prompts."""

from __future__ import annotations

from typing import List

from agents.base_agent import BaseAgent
from llm.client import LLMClient, Message


class ErrorAgent(BaseAgent):
    SYSTEM_PROMPT = "You analyze prior prompts and assistant responses to fix errors. Respond with a corrected assistant reply."

    def __init__(self, client: LLMClient) -> None:
        super().__init__(client)

    def repair(
        self,
        *,
        messages: List[Message],
        last_response: str,
        error_description: str,
    ) -> str:
        original = "\n".join(f"{msg.role.upper()}: {msg.content}" for msg in messages)
        user_prompt = (
            "ORIGINAL CONVERSATION:\n"
            f"{original}\n\n"
            "PREVIOUS ASSISTANT RESPONSE:\n"
            f"{last_response}\n\n"
            "ERROR:\n"
            f"{error_description}\n\n"
            "Regenerate a corrected response that satisfies the original instructions and output JSON only if requested."
        )
        response = self._request([
            Message(role="system", content=self.SYSTEM_PROMPT),
            Message(role="user", content=user_prompt),
        ])
        return response
