"""Data models for the CX Wind Tunnel API."""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class AgentConfig(BaseModel):
    """How to reach the agent under test."""
    mode: Literal["endpoint", "prompt"] = "endpoint"
    # For endpoint mode: the URL to POST conversations to
    endpoint_url: Optional[str] = None
    # Some agents expect a specific request/response shape; keep it simple/standard
    # Expected: POST {messages: [{role, content}]} -> {reply: str}
    reply_field: str = "reply"  # JSON field in the agent's response holding its text

    # For prompt mode (demo without a real endpoint): the agent's system prompt
    system_prompt: Optional[str] = None


class TestConfig(BaseModel):
    """A full stress-test request."""
    api_key: str = Field(..., description="User's Anthropic API key (BYOK)")
    domain: str = Field(..., description="e.g. 'Banking — Fraud & Card Services'")
    agent: AgentConfig
    num_personas: int = Field(10, ge=1, le=50)
    turns_per_conversation: int = Field(4, ge=2, le=8)


class Persona(BaseModel):
    id: str
    name: str
    description: str
    intent: str
    communication_style: str
    risk_level: Literal["baseline", "edge_case", "high_risk", "critical"]
    opening_message: str
    success_criteria: str


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
    was_unexpected: bool = False  # True if this is a failure type the user likely didn't test for


class ConversationResult(BaseModel):
    persona: Persona
    transcript: List[Turn]
    finding: Finding


class TestReport(BaseModel):
    domain: str
    total_personas: int
    total_failures: int
    unexpected_failures: int
    passed: int
    findings: List[Finding]
    conversations: List[ConversationResult]
