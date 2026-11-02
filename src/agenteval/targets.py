"""Targets: the agent under test.

A target receives the conversation so far and returns the agent's next reply.

- :class:`HttpTarget` — your real agent behind an HTTP endpoint
- :class:`SimulatedTarget` — no endpoint yet; the simulator model plays your agent from its system prompt
"""

from __future__ import annotations

from typing import Any, Protocol

import httpx

from .llm import LLM
from .models import AgentConfig, Turn


class Target(Protocol):
    def respond(self, transcript: list[Turn]) -> str: ...

    def describe(self) -> str: ...


def get_path(data: Any, path: str) -> Any:
    """Read a dotted path such as ``reply`` or ``choices.0.message.content`` from parsed JSON."""
    cur = data
    for part in path.split("."):
        if isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return None
        elif isinstance(cur, dict):
            if part not in cur:
                return None
            cur = cur[part]
        else:
            return None
    return cur


class HttpTarget:
    """POSTs ``{"messages": [{"role", "content"}, ...], **extra_body}`` and reads ``reply_field`` from the JSON."""

    def __init__(
        self,
        url: str,
        reply_field: str = "reply",
        headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ):
        if not url:
            raise ValueError("HTTP target needs a url")
        self.url = url
        self.reply_field = reply_field or "reply"
        self.headers = headers or {}
        self.extra_body = extra_body or {}
        self.timeout = timeout
        self._transport = transport

    def describe(self) -> str:
        return f"http {self.url}"

    def respond(self, transcript: list[Turn]) -> str:
        payload = {
            **self.extra_body,
            "messages": [{"role": t.role, "content": t.content} for t in transcript],
        }
        try:
            with httpx.Client(timeout=self.timeout, transport=self._transport) as http:
                r = http.post(self.url, json=payload, headers=self.headers)
                r.raise_for_status()
                value = get_path(r.json(), self.reply_field)
        except Exception as e:  # the judge should see agent errors as part of the conversation
            return f"(AGENT ERROR: {e})"
        if value is None or value == "":
            return f"(AGENT ERROR: no '{self.reply_field}' field in the response)"
        return str(value)


class SimulatedTarget:
    """Demo mode: the simulator model plays your agent using the given system prompt."""

    def __init__(self, llm: LLM, system_prompt: str):
        self.llm = llm
        self.system_prompt = system_prompt or "You are a helpful customer service agent."

    def describe(self) -> str:
        return "simulated (system prompt)"

    def respond(self, transcript: list[Turn]) -> str:
        messages = [{"role": t.role, "content": t.content} for t in transcript]
        return self.llm.complete("simulator", self.system_prompt, messages, max_tokens=500)


def target_from_agent_config(agent: AgentConfig, llm: LLM) -> Target:
    """Build a target from the web UI's AgentConfig."""
    if agent.mode == "endpoint":
        return HttpTarget(url=agent.endpoint_url or "", reply_field=agent.reply_field)
    return SimulatedTarget(llm, agent.system_prompt or "")
