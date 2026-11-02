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


# ─── Accent & language variation (V3) ──────────────────────────────────────


class LanguageTestConfig(BaseModel):
    """A full accent & language variation test request (web UI)."""

    api_key: str = Field(..., description="User's API key for the chosen provider (BYOK)")
    provider: Literal["anthropic", "gemini"] = "anthropic"
    domain: str
    agent: AgentConfig
    num_scenarios: int = Field(3, ge=1, le=10)
    turns_per_conversation: int = Field(2, ge=1, le=6)
    accents: list[str] = Field(default_factory=lambda: ["indian_english", "british_english", "american_english"])
    code_switch: list[str] = Field(default_factory=lambda: ["hinglish"])
    asr: list[str] = Field(default_factory=lambda: ["indian_english"])
    asr_error_rate: float = Field(0.3, ge=0.0, le=1.0)


class LanguageVariant(BaseModel):
    id: str
    kind: Literal["control", "accent", "code_switch", "asr"]
    profile: str  # e.g. indian_english, hinglish; "plain" for the control
    label: str
    opening_message: str


class LanguageScenario(BaseModel):
    id: str
    name: str
    intent: str
    canonical_message: str
    success_criteria: str
    variants: list[LanguageVariant]


class VariantResult(BaseModel):
    variant: LanguageVariant
    transcript: list[Turn]
    handled_correctly: bool
    issue: str = ""


class LanguageFinding(BaseModel):
    scenario_id: str
    scenario_name: str
    failed: bool
    severity: Severity
    title: str
    explanation: str
    consistency_score: int  # 0-100: how equally every variant was served
    bias_detected: bool = False
    failing_variants: list[str] = Field(default_factory=list)  # labels
    control_failed: bool = False  # the plain-English control failed too: not a language issue


class LanguageScenarioResult(BaseModel):
    scenario: LanguageScenario
    results: list[VariantResult]
    finding: LanguageFinding


class LanguageTestReport(BaseModel):
    domain: str
    total_scenarios: int
    total_variants: int
    total_failures: int
    passed: int
    profiles_tested: list[str]
    average_consistency: float
    findings: list[LanguageFinding]
    scenarios: list[LanguageScenarioResult]


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


class CaseComparison(BaseModel):
    suite: str
    case_id: str
    case: str
    status: Literal["fixed", "regressed", "still_failing", "still_passing", "new", "removed"]
    before: str  # severity, or "pass"
    after: str
    title: str = ""


class Comparison(BaseModel):
    """How a run compares with an earlier one on the same test cases."""

    baseline_started_at: str
    baseline_name: str
    fixed: int = 0
    regressed: int = 0
    still_failing: int = 0
    still_passing: int = 0
    cases: list[CaseComparison] = Field(default_factory=list)


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
    language: LanguageTestReport | None = None
    summary: RunSummary
    comparison: Comparison | None = None
    gate: GateResult = Field(default_factory=GateResult)
