"""Anthropic (Claude) provider."""

from __future__ import annotations

from .base import LLMError, Role

DEFAULT_MODELS: dict[str, str] = {
    "generator": "claude-haiku-4-5-20251001",
    "simulator": "claude-haiku-4-5-20251001",
    "judge": "claude-sonnet-4-6",
}


class AnthropicLLM:
    provider = "anthropic"

    def __init__(self, api_key: str, models: dict[str, str] | None = None, max_retries: int = 3):
        if not api_key or not api_key.strip():
            raise LLMError("No Anthropic API key. Set ANTHROPIC_API_KEY or pass one in the config.")
        from anthropic import Anthropic  # imported lazily so other providers don't need it

        self._client = Anthropic(api_key=api_key.strip(), max_retries=max_retries)
        self.models = {**DEFAULT_MODELS, **(models or {})}

    def complete(
        self,
        role: Role,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1000,
    ) -> str:
        import anthropic

        try:
            resp = self._client.messages.create(
                model=self.models[role],
                max_tokens=max_tokens,
                system=system,
                messages=messages,
            )
        except anthropic.AuthenticationError as e:
            raise LLMError("Anthropic rejected the API key (401). Check that it is correct and active.") from e
        except anthropic.NotFoundError as e:
            raise LLMError(f"Model '{self.models[role]}' was not found for role '{role}'.") from e
        return "".join(b.text for b in resp.content if b.type == "text").strip()
