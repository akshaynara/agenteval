"""Google Gemini provider — Gemini API (API key) or Vertex AI (Application Default Credentials).

Install with:  pip install "agenteval[google]"
"""

from __future__ import annotations

import httpx

from .base import LLMError, Role

DEFAULT_MODELS: dict[str, str] = {
    "generator": "gemini-3.8-flash",
    "simulator": "gemini-3.8-flash",
    "judge": "gemini-3.8-flash",
}

# Gemini 3 models think before answering. Keep that light for high-volume roles.
DEFAULT_THINKING: dict[str, str] = {"generator": "low", "simulator": "low", "judge": "medium"}

# Thinking tokens count toward max_output_tokens, so leave room for them on top of the answer budget.
THINKING_HEADROOM = 8192


def _import_genai():
    try:
        from google import genai
        from google.genai import errors, types
    except ImportError as e:
        raise LLMError('The Gemini provider needs the Google SDK: pip install "agenteval[google]"') from e
    return genai, types, errors


class GeminiLLM:
    provider = "gemini"

    def __init__(
        self,
        api_key: str | None = None,
        models: dict[str, str] | None = None,
        vertexai: bool = False,
        project: str | None = None,
        location: str | None = None,
        thinking: dict[str, str] | None = None,
    ):
        genai, types, _ = _import_genai()
        if not vertexai and not (api_key and api_key.strip()):
            raise LLMError("No Gemini API key. Set GEMINI_API_KEY, or use Vertex AI (provider.vertexai: true).")
        # extra retries: free-tier rate limits are tight
        http_options = types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=6))
        if vertexai:
            self._client = genai.Client(vertexai=True, project=project, location=location or "global",
                                        http_options=http_options)
            self.provider = "gemini (vertex ai)"
        else:
            self._client = genai.Client(api_key=api_key.strip(), http_options=http_options)
        self._types = types
        self.models = {**DEFAULT_MODELS, **(models or {})}
        self.thinking = {**DEFAULT_THINKING, **(thinking or {})}

    def complete(
        self,
        role: Role,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1000,
    ) -> str:
        types = self._types
        _, _, errors = _import_genai()
        contents = [
            types.Content(role="model" if m["role"] == "assistant" else "user", parts=[types.Part(text=m["content"])])
            for m in messages
        ]
        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens + THINKING_HEADROOM,
        )
        level = self.thinking.get(role)
        # thinking_level is a Gemini 3+ setting; older models use thinking budgets instead.
        if level and level != "default" and not self.models[role].startswith(("gemini-1", "gemini-2")):
            config.thinking_config = types.ThinkingConfig(thinking_level=level.upper())
        try:
            resp = self._client.models.generate_content(model=self.models[role], contents=contents, config=config)
        except errors.ClientError as e:
            code = getattr(e, "code", None)
            if code in (401, 403):
                raise LLMError(f"Gemini rejected the credentials ({code}). Check your API key or Vertex AI access.") from e
            if code == 429:
                raise LLMError(
                    "Gemini rate limit reached, even after retrying. On the free tier, run fewer personas/probes "
                    "or lower concurrency."
                ) from e
            if code == 404:
                raise LLMError(f"Gemini model '{self.models[role]}' was not found for role '{role}'.") from e
            if code == 400 and "thinking" in str(e).lower():
                raise LLMError(
                    f"Model '{self.models[role]}' rejected thinking level '{level}'. "
                    f"Set provider.thinking.{role} to a supported level (or 'default')."
                ) from e
            raise
        except errors.ServerError as e:
            raise LLMError(f"Gemini had a server error after retries ({getattr(e, 'code', '?')}). Try again shortly.") from e
        except httpx.TransportError as e:
            raise LLMError(f"Couldn't reach the Gemini API: {e}") from e
        return (resp.text or "").strip()
