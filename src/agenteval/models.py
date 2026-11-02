"""Data models shared by the engine, the CLI, and the web API.

The request/response models used by the web UI (`TestConfig`, `RAITestConfig`,
`TestReport`, `RAITestReport`, ...) keep the exact field names of v2 so the
deployed frontend keeps working.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Severity = Literal["critical", "high", "medium", "low", "pass"]
RiskLevel = Literal["baseline", "edge_case", "high_risk", "critical"]
RAICategory = Literal[
    "jailbreak",
    "harmful_content",
    "bias",
    "pii_leakage",
    "scope_violation",
    "prompt_injection",
    "manipulation_compliance",
]

RAI_CATEGORIES: list[str] = [
    "jailbreak",
    "harmful_content",
    "bias",
    "pii_leakage",
    "scope_violation",
    "prompt_injection",
    "manipulation_compliance",
]

# Lower number = worse. Used for sorting findings and for --fail-on thresholds.
SEVERITY_RANK: dict[str, int] = {"critical": 0, "high": 1, "medium": 2, "low": 3, "pass": 4}


# ─── Web API request models (v2 contract) ──────────────────────────────────


class AgentConfig(BaseModel):
    """How to reach the agent under test (web UI shape)."""

    mode: Literal["endpoint", "prompt"] = "endpoint"
    endpoint_url: str | None = None
    reply_field: str = "reply"
    system_prompt: str | None = None


class TestConfig(BaseModel):
    """A full operational stress-test request."""

    __test__ = False  # not a pytest test class

    api_key: str = Field(..., description="User's API key for the chosen provider (BYOK)")
    provider: Literal["anthropic", "gemini"] = "anthropic"
    domain: str = Field(..., description="e.g. 'Banking — Fraud & Card Services'")
    agent: AgentConfig
    num_personas: int = Field(10, ge=1, le=50)
    turns_per_conversation: int = Field(4, ge=2, le=8)


class RAITestConfig(BaseModel):
    """A full RAI safety red-team test request."""

    api_key: str = Field(..., description="User's API key for the chosen provider (BYOK)")
    provider: Literal["anthropic", "gemini"] = "anthropic"
    domain: str = Field(..., description="e.g. 'Banking — Fraud & Card Services'")
    agent: AgentConfig
    num_probes: int = Field(10, ge=1, le=50)
    turns_per_conversation: int = Field(4, ge=2, le=8)
    categories: list[RAICategory] = Field(default_factory=lambda: list(RAI_CATEGORIES))


# ─── Test cases ────────────────────────────────────────────────────────────


class Persona(BaseModel):
    id: str
    name: str
    description: str
    intent: str
    communication_style: str
    risk_level: RiskLevel
    opening_message: str
    success_criteria: str


class RAIProbe(BaseModel):
    """A single RAI safety probe — adversarial attack targeting a specific safety category."""

    id: str
    name: str
    category: RAICategory
    description: str
    attack_strategy: str
    opening_message: str
    success_criteria: str


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    # What the platform reported about this agent turn (matched intent, page, tool calls, ...).
    # Shown to the judge, never to the simulated customer.
    meta: dict[str, Any] | None = None


# ─── Findings & reports ────────────────────────────────────────────────────


class Finding(BaseModel):
    persona_id: str
    persona_name: str
    failed: bool
    failure_type: str | None = None
    severity: Severity | None = None
    title: str
    explanation: str
    was_unexpected: bool = False


class RAIFinding(BaseModel):
    """Finding from a RAI safety probe."""

    probe_id: str
    probe_name: str
    category: str
    failed: bool
    severity: Severity
    title: str
    explanation: str
    attack_succeeded: bool


class ConversationResult(BaseModel):
    persona: Persona
    transcript: list[Turn]
    finding: Finding


class RAIConversationResult(BaseModel):
    probe: RAIProbe
    transcript: list[Turn]
    finding: RAIFinding


class TestReport(BaseModel):
    """Wind Tunnel (operational) results."""

    __test__ = False

    domain: str
    total_personas: int
    total_failures: int
    unexpected_failures: int
    passed: int
    findings: list[Finding]
    conversations: list[ConversationResult]


class RAITestReport(BaseModel):
    """Red-team (safety) results."""

    domain: str
    total_probes: int
    total_failures: int
    passed: int
    categories_tested: list[str]
    findings: list[RAIFinding]
    conversations: list[RAIConversationResult]


class RunSummary(BaseModel):
    total_cases: int = 0
    total_failures: int = 0
    passed: int = 0
    unexpected_failures: int = 0
    severity_counts: dict[str, int] = Field(default_factory=dict)


class GateResult(BaseModel):
    """Whether this run should fail a CI pipeline."""

    fail_on: str | None = None
    passed: bool = True
    reason: str = ""


class RunReport(BaseModel):
    """Everything one `agenteval run` produced. Saved as JSON; reloadable for regression later."""

    schema_version: int = 1
    agenteval_version: str
    name: str
    domain: str
    target: str
    provider: str
    models: dict[str, str]
    started_at: str
    finished_at: str
    duration_seconds: float
    wind_tunnel: TestReport | None = None
    red_team: RAITestReport | None = None
    summary: RunSummary
    gate: GateResult = Field(default_factory=GateResult)
