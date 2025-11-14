"""Single AIML-backed LLM client used by all agents."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Protocol

import requests


@dataclass(frozen=True)
class Message:
    role: str
    content: str


class LLMClient(Protocol):
    """Minimal protocol for chat completions."""

    def chat(self, messages: Iterable[Message]) -> str:  # pragma: no cover - type contract only
        ...


class AIMLLLMClient(LLMClient):
    """Direct HTTP client for AIML's chat completion API."""

    def __init__(
        self,
        *,
        config_path: Path,
        model: str,
        base_url: str = "https://api.aimlapi.com/v1",
        timeout: int = 60,
    ) -> None:
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        api_key = config.get("AIML_API_KEY")
        if not api_key:
            raise ValueError("AIML_API_KEY missing from config file")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat(self, messages: Iterable[Message]) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "temperature": 0.0,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"AIML API response missing content: {data}") from exc
        if not isinstance(content, str):
            raise RuntimeError(f"AIML API content is not string: {content!r}")
        return content.strip()
