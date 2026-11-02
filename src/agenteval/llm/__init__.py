"""LLM providers. Add a provider by implementing :class:`LLM` and registering it in :func:`create_llm`."""

from __future__ import annotations

from .base import LLM, ROLES, LLMError, Role

PROVIDERS = ("anthropic",)


def create_llm(provider: str, api_key: str, models: dict[str, str] | None = None) -> LLM:
    provider = (provider or "anthropic").lower()
    if provider == "anthropic":
        from .anthropic import AnthropicLLM

        return AnthropicLLM(api_key=api_key, models=models)
    raise LLMError(f"Unknown provider '{provider}'. Available: {', '.join(PROVIDERS)}")


def default_api_key_env(provider: str) -> str:
    return {"anthropic": "ANTHROPIC_API_KEY"}.get(provider.lower(), f"{provider.upper()}_API_KEY")


__all__ = ["LLM", "LLMError", "ROLES", "Role", "PROVIDERS", "create_llm", "default_api_key_env"]
