"""Provider-neutral interface for the LLMs AgentEval drives.

AgentEval uses a model in three roles:

- ``generator`` — writes personas and safety probes (high volume, cheap model)
- ``simulator`` — plays the customer / attacker, and the agent in simulated mode
- ``judge``     — decides whether the agent failed (quality matters most here)

Every provider implements :class:`LLM`, so the engine never imports a vendor SDK.
"""

from __future__ import annotations

from typing import Literal, Protocol

Role = Literal["generator", "simulator", "judge"]
ROLES: tuple[Role, ...] = ("generator", "simulator", "judge")


class LLMError(RuntimeError):
    """Raised when a provider call fails in a way the user should see."""


class LLM(Protocol):
    provider: str
    models: dict[str, str]

    def complete(
        self,
        role: Role,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1000,
    ) -> str:
        """Return the model's text reply. ``messages`` alternate user/assistant, starting with user."""
        ...
