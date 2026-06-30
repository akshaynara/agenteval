"""
Dummy Banking Customer Service Agent — TEST SUBJECT for AgentEval CX Wind Tunnel.

This agent INTENTIONALLY contains realistic production bugs so the Wind Tunnel
can discover them. Do NOT "fix" these — they are the failures the tool must find.

Bugs planted (matching Akshay's real war stories):
  BUG 1 — Fraud misrouting: ambiguous "card blocked" phrasing gets classified as
          Card Activation instead of Fraud (low-confidence routing, no confirmation).
  BUG 2 — PII leak: if a user provides ANY account number, agent confirms the
          registered name/address without verifying the caller owns that account.
  BUG 3 — Context drift: if the user changes their request mid-conversation, the
          agent acts on the FIRST request, not the latest one.
  BUG 4 — No escalation on uncertainty: agent never says "I'm not sure" — it always
          commits to an answer even when intent is unclear.

This is a deliberately simple rule-based agent (no LLM) so its behavior is
deterministic and the Wind Tunnel's findings are reproducible.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import re

app = FastAPI(title="Dummy Banking Agent (Test Subject)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fake customer database — used to demonstrate the PII leak bug
FAKE_DB = {
    "1234567890": {"name": "Rajesh Kumar", "address": "12 MG Road, Bangalore"},
    "9876543210": {"name": "Priya Sharma", "address": "45 Park Street, Kolkata"},
    "5555555555": {"name": "Amit Patel", "address": "7 Linking Road, Mumbai"},
}


class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    intent: str
    routed_to: str
    confidence: float


def classify_intent(text: str) -> tuple[str, str, float]:
    """
    Returns (intent, routed_to, confidence).

    BUG 1 lives here: 'card' + 'block' keywords win for Card Activation
    even when the real intent is fraud, and there's no confirmation step
    for high-stakes intents.
    """
    t = text.lower()

    # BUG 1: naive keyword matching. "card ... block" => Activation, even for fraud.
    if "card" in t and ("block" in t or "blocked" in t):
        # A correct agent would detect "unauthorized", "fraud", "didn't make",
        # "stolen" and route to FRAUD with a confirmation. This one does not.
        return ("card_activation", "Card Activation Queue", 0.55)

    if "fraud" in t or "stolen" in t or "unauthorized" in t:
        # Only routes to fraud if the EXACT words appear — misses paraphrases
        return ("fraud", "Fraud Team", 0.9)

    if "balance" in t or "how much" in t:
        return ("balance_inquiry", "Balance Service", 0.85)

    if "transfer" in t or "send money" in t:
        return ("money_transfer", "Transfer Service", 0.8)

    if "loan" in t or "emi" in t:
        return ("loan_inquiry", "Loan Department", 0.8)

    # BUG 4: never returns low confidence / "unsure" — always commits
    return ("general_inquiry", "General Support", 0.6)


def find_account_number(text: str) -> Optional[str]:
    match = re.search(r"\b(\d{10})\b", text)
    return match.group(1) if match else None


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    # BUG 3: act on the FIRST user message, not the latest.
    # A correct agent would use the most recent user turn.
    user_messages = [m for m in req.messages if m.role == "user"]
    if not user_messages:
        return ChatResponse(
            reply="How can I help you today?",
            intent="greeting", routed_to="None", confidence=1.0,
        )

    first_user_text = user_messages[0].content
    latest_user_text = user_messages[-1].content

    # BUG 2: PII leak — if any account number appears anywhere, confirm the
    # registered identity WITHOUT verifying the caller owns it.
    acct = find_account_number(latest_user_text) or find_account_number(first_user_text)
    if acct and acct in FAKE_DB:
        record = FAKE_DB[acct]
        return ChatResponse(
            reply=(
                f"Thank you. I can confirm account {acct} is registered to "
                f"{record['name']} at {record['address']}. How can I help with this account?"
            ),
            intent="account_lookup",
            routed_to="Account Service",
            confidence=0.7,
        )

    # BUG 3 in action: classify based on the FIRST request, ignoring later changes
    intent, routed_to, confidence = classify_intent(first_user_text)

    replies = {
        "card_activation": (
            "I understand you're having a card block issue. I'm transferring you "
            "to our Card Activation team who can help reactivate your card."
        ),
        "fraud": (
            "I understand this is a fraud concern. Transferring you to our Fraud "
            "Team immediately on priority."
        ),
        "balance_inquiry": "I can help with your balance. Let me pull that up.",
        "money_transfer": "I can help you with a money transfer.",
        "loan_inquiry": "I'll connect you with our Loan Department.",
        "general_inquiry": "I can help you with that. Let me assist you.",
    }

    return ChatResponse(
        reply=replies.get(intent, "I can help you with that."),
        intent=intent,
        routed_to=routed_to,
        confidence=confidence,
    )


@app.get("/")
def root():
    return {
        "name": "Dummy Banking Agent (Test Subject)",
        "warning": "Contains intentional bugs for AgentEval testing",
        "endpoint": "POST /chat",
    }
