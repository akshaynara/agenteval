"""Data models for the CX Wind Tunnel API."""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class AgentConfig(BaseModel):
    """How to reach the agent under test."""
    mode: Literal["endpoint", "prompt"] = "endpoint"
    endpoint_url: Optional[str] = None
    reply_field: str = "reply"
    system_prompt: Optional[str] = None


class TestConfig(BaseModel):
    """A full operational stress-test request."""
    api_key: str = Field(..., description="User's Anthropic API key (BYOK)")
    domain: str = Field(..., description="e.g. 'Banking — Fraud & Card Services'")
    agent: AgentConfig
    num_personas: int = Field(10, ge=1, le=50)
    turns_per_conversation: int = Field(4, ge=2, le=8)


class RAITestConfig(BaseModel):
    """A full RAI safety red-team test request."""
    api_key: str = Field(..., description="User's Anthropic API key (BYOK)")
    domain: str = Field(..., description="e.g. 'Banking — Fraud & Card Services'")
    agent: AgentConfig
    num_probes: int = Field(10, ge=1, le=50)
    turns_per_conversation: int = Field(4, ge=2, le=8)
    # Which RAI categories to test — defaults to all
    categories: List[Literal[
        "jailbreak",
        "harmful_content",
        "bias",
        "pii_leakage",
        "scope_violation",
        "prompt_injection",
        "manipulation_compliance"
    ]] = Field(default=[
        "jailbreak", "harmful_content", "bias",
        "pii_leakage", "scope_violation",
        "prompt_injection", "manipulation_compliance"
    ])


class Persona(BaseModel):
    id: str
    name: str
    description: str
    intent: str
    communication_style: str
    risk_level: Literal["baseline", "edge_case", "high_risk", "critical"]
    opening_message: str
    success_criteria: str


class RAIProbe(BaseModel):
    """A single RAI safety probe — adversarial attack targeting a specific safety category."""
    id: str
    name: str
    category: Literal[
        "jailbreak", "harmful_content", "bias",
        "pii_leakage", "scope_violation",
        "prompt_injection", "manipulation_compliance"
    ]
    description: str
    attack_strategy: str   # how this probe tries to break the agent
    opening_message: str
    success_criteria: str  # what agent MUST do to pass


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class Finding(BaseModel):
    persona_id: str
    persona_name: str
    failed: bool
    failure_type: Optional[str] = None
    severity: Optional[Literal["critical", "high", "medium", "low", "pass"]] = None
    title: str
    explanation: str
    was_unexpected: bool = False


class RAIFinding(BaseModel):
    """Finding from a RAI safety probe."""
    probe_id: str
    probe_name: str
    category: str
    failed: bool          # True = safety failure (agent was broken)
    severity: Literal["critical", "high", "medium", "low", "pass"]
    title: str
    explanation: str
    attack_succeeded: bool  # Did the adversarial attack work?


class ConversationResult(BaseModel):
    persona: Persona
    transcript: List[Turn]
    finding: Finding


class RAIConversationResult(BaseModel):
    probe: RAIProbe
    transcript: List[Turn]
    finding: RAIFinding


class TestReport(BaseModel):
    domain: str
    total_personas: int
    total_failures: int
    unexpected_failures: int
    passed: int
    findings: List[Finding]
    conversations: List[ConversationResult]


class RAITestReport(BaseModel):
    domain: str
    total_probes: int
    total_failures: int
    passed: int
    categories_tested: List[str]
    findings: List[RAIFinding]
    conversations: List[RAIConversationResult]
