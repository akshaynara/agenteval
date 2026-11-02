"""
Demo banking agent — a TEST SUBJECT for AgentEval, with bugs planted on purpose.

Run it with:  agenteval demo-agent      (serves POST http://localhost:9000/chat)

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

`agenteval demo-agent --fixed` (or AGENTEVAL_DEMO_FIXED=1) serves the same agent with
all four bugs fixed — run it after a baseline to demo regression mode on stage.
"""

import os
import re

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
    messages: list[Message]
    session_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    intent: str
    routed_to: str
    confidence: float


FRAUD_SIGNALS = ("fraud", "stolen", "unauthori", "didn't make", "did not make", "not me", "wasn't me",
                 "don't recognize", "don't recognise", "someone used", "lost my card", "hacked", "scam")


def fixed_mode() -> bool:
    return os.environ.get("AGENTEVAL_DEMO_FIXED", "").lower() in ("1", "true", "yes")


def classify_intent_fixed(text: str) -> tuple[str, str, float]:
    """The fixed classifier: fraud signals win, and unclear requests escalate instead of guessing."""
    t = text.lower()
    if any(s in t for s in FRAUD_SIGNALS):
        return ("fraud", "Fraud Team", 0.92)
    if "card" in t and ("block" in t or "blocked" in t):
        return ("card_blocked_unclear", "Confirm with customer", 0.5)
    if "balance" in t or "how much" in t:
        return ("balance_inquiry", "Balance Service", 0.85)
    if "transfer" in t or "send money" in t:
        return ("money_transfer", "Transfer Service", 0.8)
    if "loan" in t or "emi" in t:
        return ("loan_inquiry", "Loan Department", 0.8)
    return ("unclear", "Human agent", 0.3)


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


def find_account_number(text: str) -> str | None:
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
    if fixed_mode():
        return _chat_fixed(user_messages[-1].content)
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


def _chat_fixed(latest: str) -> ChatResponse:
    """All four bugs fixed: latest message wins, fraud first, no PII before verification, escalate when unsure."""
    if find_account_number(latest):
        return ChatResponse(
            reply=("For your security I can't confirm any details on an account over chat. "
                   "I'll send a one-time code to the phone number registered on the account — can you read it back to me?"),
            intent="account_lookup", routed_to="Identity Verification", confidence=0.9)
    intent, routed_to, confidence = classify_intent_fixed(latest)
    replies = {
        "fraud": ("I'm sorry — that sounds like fraud. I've frozen your card right now so no new charges can go "
                  "through, and I'm connecting you to our Fraud Team on priority."),
        "card_blocked_unclear": ("I can help with a blocked card. To route you correctly: did you or someone you "
                                 "know make the recent transactions, or do you not recognise them?"),
        "balance_inquiry": "I can help with your balance once I've verified your identity. I'll send you a one-time code.",
        "money_transfer": "I can help you with a transfer. Who would you like to send money to, and how much?",
        "loan_inquiry": "I'll connect you with our Loan Department.",
        "unclear": ("I want to make sure I get this right, so I'm connecting you with a colleague who can help. "
                    "Could you tell me a bit more about what you need while I do that?"),
    }
    return ChatResponse(reply=replies[intent], intent=intent, routed_to=routed_to, confidence=confidence)
