"""Common utilities shared by agent classes."""

from __future__ import annotations

import json
import re
from typing import Iterable, List, Optional

from llm.client import LLMClient, Message


class AgentError(RuntimeError):
    """Raised when an LLM call returns unusable content."""


class BaseAgent:
    COMMON_KNOWLEDGE = (
        "Shared concepts:\n"
        "  • BUSINESS DATA (variables) are instance-specific values such as timestamps, user/device identifiers, "
        "IP/MAC addresses, counters, metrics, and JSON payloads. They come from unbounded domains and replacing "
        "them does not change the semantic meaning of the event. It should not include multiple words.\n"
        "  • STRUCTURE (constants) are system-defined tokens such as event skeletons, log levels, module names, "
        "protocol keywords, message sentences, and syntactic separators (colons, brackets, pipes). They draw from finite sets and "
        "altering them would change what the log entry represents.\n"
        "Always preserve STRUCTURE as literal text and capture only BUSINESS DATA."
    )

    _error_agent = None  # class-level reference shared across agents

    def __init__(self, client: LLMClient) -> None:
        self._client = client
        self._agent_name = self.__class__.__name__
        self._line_context: str = ""
        self.last_messages: List[Message] = []

    def _request(self, messages: Iterable[Message]) -> str:
        convo = list(messages)
        self.last_messages = convo
        response = self._client.chat(convo)
        if not isinstance(response, str) or not response.strip():
            raise AgentError("empty response from LLM")
        cleaned = response.strip()
        single_line = " ".join(cleaned.split())
        self._log(single_line)
        return cleaned

    @staticmethod
    def _extract_json(payload: str) -> Optional[dict]:
        text = payload.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text)
            text = re.sub(r"```$", "", text).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    return None
        return None

    def set_line_context(self, line_number: int) -> None:
        self._line_context = f"[L{line_number}]"

    def _log(self, message: str) -> None:
        line_tag = self._line_context or "[L-]"
        prefix = f"{line_tag}[AGENT][{self._agent_name}]"
        print(f"{prefix} {message}")

    @classmethod
    def register_error_agent(cls, agent) -> None:
        cls._error_agent = agent

    def _call_json(
        self,
        messages: List[Message],
        *,
        error_description: str,
    ) -> tuple[dict, str, List[Message]]:
        history = list(messages)
        response = self._request(history)
        data = self._extract_json(response)
        if data:
            history.append(Message(role="assistant", content=response))
            return data, response, history

        repaired_response = self._attempt_repair(
            messages=history,
            last_response=response,
            error_description=error_description,
        )
        if not repaired_response:
            raise AgentError("LLM failed to provide valid JSON")
        retry_data = self._extract_json(repaired_response)
        if not retry_data:
            raise AgentError("LLM failed to provide valid JSON after retry")
        history.append(Message(role="assistant", content=repaired_response))
        return retry_data, repaired_response, history

    def _attempt_repair(
        self,
        *,
        messages: List[Message],
        last_response: str,
        error_description: str,
    ) -> Optional[str]:
        if not BaseAgent._error_agent:
            return None
        return BaseAgent._error_agent.repair(
            messages=messages,
            last_response=last_response,
            error_description=error_description,
        )
