"""Connector tests. Every platform is mocked at the HTTP layer, so these check the exact wire format."""

import json

import httpx
import pytest

from agenteval import engine
from agenteval.models import Turn
from agenteval.targets import (
    CXAgentStudioTarget,
    DialogflowCXTarget,
    HttpTarget,
    OpenAITarget,
    TargetError,
    agent_turn,
    check_reachable,
)


class Recorder:
    """An httpx transport that records requests and answers with a scripted function."""

    def __init__(self, respond):
        self.requests: list[httpx.Request] = []
        self._respond = respond

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._respond(request)

    @property
    def transport(self):
        return httpx.MockTransport(self)

    def body(self, i=-1):
        return json.loads(self.requests[i].content)


def convo(*texts):
    roles = ["user", "assistant"]
    return [Turn(role=roles[i % 2], content=t) for i, t in enumerate(texts)]


# ─── HTTP ──────────────────────────────────────────────────────────────────


def test_http_session_field_is_stable_within_a_conversation():
    rec = Recorder(lambda r: httpx.Response(200, json={"reply": "ok"}))
    t = HttpTarget("http://agent/chat", session_field="session_id", extra_body={"channel": "web"},
                   transport=rec.transport)
    s1, s2 = t.open_session(), t.open_session()
    s1.respond(convo("hi"))
    s1.respond(convo("hi", "ok", "again"))
    s2.respond(convo("hi"))
    ids = [rec.body(i)["session_id"] for i in range(3)]
    assert ids[0] == ids[1] != ids[2]
    assert rec.body(0)["channel"] == "web" and rec.body(1)["messages"][-1]["content"] == "again"


# ─── OpenAI-compatible ─────────────────────────────────────────────────────


def test_openai_target_wire_format():
    rec = Recorder(lambda r: httpx.Response(200, json={"choices": [{"message": {"content": " Hello! "}}]}))
    t = OpenAITarget("https://llm.example.com/v1/", "my-model", api_key="k123", system_prompt="Be a bank.",
                     extra_body={"temperature": 0}, transport=rec.transport)
    reply = t.open_session().respond(convo("hi"))
    assert reply.text == "Hello!"
    req = rec.requests[0]
    assert str(req.url) == "https://llm.example.com/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer k123"
    body = rec.body()
    assert body["model"] == "my-model" and body["temperature"] == 0
    assert body["messages"][0] == {"role": "system", "content": "Be a bank."}
    assert body["messages"][1] == {"role": "user", "content": "hi"}


def test_openai_auth_failure_stops_the_run():
    rec = Recorder(lambda r: httpx.Response(401, json={"error": "bad key"}))
    t = OpenAITarget("https://llm.example.com/v1", "m", transport=rec.transport)
    with pytest.raises(TargetError, match="401"):
        t.respond(convo("hi"))


# ─── Dialogflow CX ─────────────────────────────────────────────────────────

AGENT = "projects/acme/locations/asia-south1/agents/abc-123"


def dfcx_reply(texts, page="Card Activation", intent="card.blocked", end=False):
    return {
        "queryResult": {
            "responseMessages": [{"text": {"text": texts}}] + ([{"endInteraction": {}}] if end else []),
            "match": {"intent": {"displayName": intent}, "matchType": "INTENT", "confidence": 0.62},
            "currentPage": {"displayName": page, "name": f"{AGENT}/flows/x/pages/{page}"},
        }
    }


def test_dialogflow_cx_wire_format_and_routing_meta():
    rec = Recorder(lambda r: httpx.Response(200, json=dfcx_reply(["Sure.", "Transferring you to Card Activation."])))
    t = DialogflowCXTarget(AGENT, language_code="en-IN", access_token="tok", quota_project="billing-proj",
                           transport=rec.transport)
    session = t.open_session()
    turn, ended = agent_turn(session, convo("my card is blocked", "Sure.", "someone used it, not me"))
    req = rec.requests[0]
    assert req.url.host == "asia-south1-dialogflow.googleapis.com"
    assert req.url.path.startswith(f"/v3/{AGENT}/sessions/agenteval-") and req.url.path.endswith(":detectIntent")
    assert req.headers["authorization"] == "Bearer tok"
    assert req.headers["x-goog-user-project"] == "billing-proj"
    # only the latest customer message is sent; the platform keeps the session state
    assert rec.body() == {"queryInput": {"text": {"text": "someone used it, not me"}, "languageCode": "en-IN"}}
    assert turn.content == "Sure.\nTransferring you to Card Activation."
    assert turn.meta == {"intent": "card.blocked", "match_type": "INTENT", "confidence": 0.62, "page": "Card Activation"}
    assert not ended


def test_dialogflow_cx_environment_global_host_and_end_session():
    rec = Recorder(lambda r: httpx.Response(200, json=dfcx_reply(["Bye"], page="End Session", end=True)))
    agent = "projects/acme/locations/global/agents/abc"
    t = DialogflowCXTarget(agent, environment="prod", access_token="tok", transport=rec.transport)
    turn, ended = agent_turn(t.open_session(), convo("bye"))
    assert rec.requests[0].url.host == "dialogflow.googleapis.com"
    assert f"/v3/{agent}/environments/prod/sessions/" in rec.requests[0].url.path
    assert ended


def test_dialogflow_cx_permission_error_is_fatal():
    rec = Recorder(lambda r: httpx.Response(403, json={"error": {"message": "Permission denied"}}))
    t = DialogflowCXTarget(AGENT, access_token="tok", transport=rec.transport)
    with pytest.raises(TargetError, match="403"):
        t.open_session().respond(convo("hi"))


def test_dialogflow_cx_rejects_bad_resource_name():
    with pytest.raises(ValueError, match="projects/P/locations/L/agents"):
        DialogflowCXTarget("abc-123")


# ─── CX Agent Studio ───────────────────────────────────────────────────────

APP = "projects/acme/locations/us/apps/medicare-ivr"


def test_cx_agent_studio_wire_format():
    outputs = {"outputs": [
        {"toolCalls": {"toolCalls": [{"displayName": "lookup_member", "args": {}}]}},
        {"text": "I found your plan.", "turnIndex": 1},
        {"text": "Anything else?", "turnIndex": 1, "turnCompleted": True},
    ]}
    rec = Recorder(lambda r: httpx.Response(200, json=outputs))
    t = CXAgentStudioTarget(APP, deployment="dep-1", entry_agent="root", use_tool_fakes=True, access_token="tok",
                            transport=rec.transport)
    session = t.open_session()
    turn, ended = agent_turn(session, convo("what's my plan?"))
    agent_turn(session, convo("what's my plan?", "I found your plan.", "and my deductible?"))

    req = rec.requests[0]
    assert req.url.host == "ces.googleapis.com"
    assert req.url.path.startswith(f"/v1/{APP}/sessions/agenteval-") and req.url.path.endswith(":runSession")
    body = rec.body(0)
    session_name = body["config"]["session"]
    assert session_name.split("/sessions/")[1].startswith("agenteval-")
    assert body["config"]["deployment"] == f"{APP}/deployments/dep-1"
    assert body["config"]["entryAgent"] == f"{APP}/agents/root"
    assert body["config"]["useToolFakes"] is True
    assert body["inputs"] == [{"text": "what's my plan?"}]
    assert rec.body(1)["config"]["session"] == session_name
    assert rec.body(1)["inputs"] == [{"text": "and my deductible?"}]
    assert turn.content == "I found your plan.\nAnything else?"
    assert turn.meta == {"tool_calls": ["lookup_member"]}
    assert not ended


def test_cx_agent_studio_session_end():
    rec = Recorder(lambda r: httpx.Response(200, json={"outputs": [{"text": "Goodbye."}, {"endSession": {}}]}))
    t = CXAgentStudioTarget(APP, access_token="tok", transport=rec.transport)
    turn, ended = agent_turn(t.open_session(), convo("bye"))
    assert ended and turn.meta == {"ended_session": True}
    assert "deployment" not in rec.body()["config"] and "useToolFakes" not in rec.body()["config"]


def test_session_ids_match_ces_rules():
    import re

    t = CXAgentStudioTarget(APP, access_token="tok")
    sid = t.open_session().session.split("/sessions/")[1]
    assert re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9-_]{4,62}", sid)


# ─── Engine integration ────────────────────────────────────────────────────


def test_conversation_stops_when_agent_ends_session_and_judge_sees_routing(fake_llm):
    rec = Recorder(lambda r: httpx.Response(200, json=dfcx_reply(["Goodbye."], page="End Session", end=True)))
    target = DialogflowCXTarget(AGENT, access_token="tok", transport=rec.transport)
    seen = []
    original = fake_llm.complete

    def spy(role, system, messages, max_tokens=1000):
        if role == "judge":
            seen.append((system, messages[-1]["content"]))
        return original(role, system, messages, max_tokens)

    fake_llm.complete = spy
    report = engine.run_wind_tunnel(fake_llm, target, "Banking", num_personas=1, turns=4)
    assert [t.role for t in report.conversations[0].transcript] == ["user", "assistant"]
    system, judged = seen[0]
    assert "[platform: intent=card.blocked" in judged and "page=End Session" in judged
    assert "[platform: ...]" in system


def test_check_reachable_fails_fast():
    t = HttpTarget("http://down/chat", transport=httpx.MockTransport(lambda r: httpx.Response(502)))
    with pytest.raises(TargetError, match="didn't answer"):
        check_reachable(t)


def test_engine_checks_target_before_generating(fake_llm):
    t = HttpTarget("http://down/chat", transport=httpx.MockTransport(lambda r: httpx.Response(502)))
    with pytest.raises(TargetError):
        engine.run_wind_tunnel(fake_llm, t, "Banking", num_personas=3)
    assert fake_llm.calls["generator"] == 0  # no tokens spent
