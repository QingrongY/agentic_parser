"""Router agent that classifies log sources via LLM instructions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from agentic_parser.agents.base_agent import BaseAgent, AgentError
from agentic_parser.llm.client import Message


@dataclass(frozen=True)
class RoutingDecision:
    device_type: str
    vendor: str
    reasoning: str

    @property
    def source_id(self) -> str:
        device = self.device_type or "unknown"
        vendor = self.vendor or "unknown"
        return f"{device}_{vendor}".replace(" ", "_")


class RouterAgent(BaseAgent):
    """Uses an LLM to categorize the origin of a log sample batch."""

    SYSTEM_PROMPT = "You are a precise classifier. Respond with JSON only."

    def identify(self, samples: Iterable[str]) -> RoutingDecision:
        sample_list = [line.strip() for line in samples if line.strip()]
        if not sample_list:
            return RoutingDecision(device_type="unknown", vendor="unknown", reasoning="no samples provided")
        allowed_types = (
            "wifi_router, wifi_network, firewall, switch, application, "
            "mobile_device, server, storage, security, unknown"
        )
        user_prompt = (
            "You are a log routing specialist. "
            "Analyze the following log samples and classify the OVERALL log source. "
            "All samples are from the SAME device/system. "
            "Respond with a SINGLE JSON object (not an array):\n"
            '{"device_type": "<category>", "vendor": "<vendor>", "reasoning": "<brief reasoning>"}\n\n'
            f"Allowed device_type values: {allowed_types}.\n"
            "Vendor examples: aruba, ubiquiti, cisco, meraki, palo_alto, apple, android, generic, unknown.\n\n"
            "Log samples:\n" + "\n".join(sample_list[:12])
        )
        messages = [
            Message(role="system", content=self.SYSTEM_PROMPT),
            Message(role="user", content=user_prompt),
        ]
        data, _, _ = self._call_json(messages, error_description="Router must return JSON payload")
        decision = RoutingDecision(
            device_type=(data.get("device_type") or "unknown").strip(),
            vendor=(data.get("vendor") or "unknown").strip(),
            reasoning=data.get("reasoning", ""),
        )
        return decision
