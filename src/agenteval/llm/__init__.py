"""LLM providers. Add a provider by implementing :class:`LLM` and registering it in :func:`create_llm`."""

from __future__ import annotations

from .base import LLM, ROLES, LLMError, Role

PROVIDERS = ("anthropic", "gemini")


def create_llm(
    provider: str,
    api_key: str | None,
    models: dict[str, str] | None = None,
    **options,
) -> LLM:
    """Create a provider. ``options`` are provider-specific (Gemini: vertexai, project, location, thinking)."""
    provider = (provider or "anthropic").lower()
    if provider == "anthropic":
        from .anthropic import AnthropicLLM

        return AnthropicLLM(api_key=api_key or "", models=models)
    if provider in ("gemini", "google"):
        from .gemini import GeminiLLM

        return GeminiLLM(api_key=api_key, models=models, **options)
    raise LLMError(f"Unknown provider '{provider}'. Available: {', '.join(PROVIDERS)}")


def default_api_key_env(provider: str) -> str:
    return {"anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY", "google": "GEMINI_API_KEY"}.get(
        provider.lower(), f"{provider.upper()}_API_KEY"
    )


class RoutedLLM:
    """Sends each role to its own provider — e.g. Gemini plays the attacker, Claude judges.

    Using a different model family as judge avoids a model grading its own kind of output.
    """

    def __init__(self, routes: dict[str, LLM]):
        missing = set(ROLES) - set(routes)
        if missing:
            raise ValueError(f"RoutedLLM is missing role(s): {sorted(missing)}")
        self.routes = routes
        providers = []
        for r in ROLES:
            if routes[r].provider not in providers:
                providers.append(routes[r].provider)
        self.provider = " + ".join(providers)
        self.models = {r: f"{routes[r].provider}:{routes[r].models[r]}" if len(providers) > 1 else routes[r].models[r]
                       for r in ROLES}

    def complete(self, role: Role, system: str, messages: list[dict[str, str]], max_tokens: int = 1000) -> str:
        return self.routes[role].complete(role, system, messages, max_tokens)


__all__ = ["LLM", "LLMError", "ROLES", "Role", "PROVIDERS", "RoutedLLM", "create_llm", "default_api_key_env"]
