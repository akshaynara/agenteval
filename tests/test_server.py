import json

from fastapi.testclient import TestClient

from agenteval.llm import LLMError
from agenteval.server import create_app

from .conftest import FakeLLM


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        name, data = "message", ""
        for line in block.split("\n"):
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data += line[5:].strip()
        events.append((name, json.loads(data)))
    return events


def client() -> TestClient:
    return TestClient(create_app(llm_factory=lambda key, provider: FakeLLM()))


AGENT = {"mode": "prompt", "system_prompt": "You are a bank agent."}


def test_run_test_keeps_v2_event_contract():
    r = client().post("/run-test", json={"api_key": "k", "domain": "Banking", "agent": AGENT,
                                         "num_personas": 3, "turns_per_conversation": 2})
    assert r.status_code == 200
    events = parse_sse(r.text)
    names = [n for n, _ in events]
    assert names.count("conversation") == 3
    assert "personas" in names and "report" in names
    report = dict(events)["report"]["report"]
    assert {"total_failures", "unexpected_failures", "passed", "total_personas", "findings"} <= report.keys()
    assert events[-1] == ("status", {"message": "Test complete.", "done": True})
    progress = [d for n, d in events if n == "status" and "current" in d]
    assert {d["total"] for d in progress} == {3}


def test_run_rai_test_keeps_v2_event_contract():
    r = client().post("/run-rai-test", json={"api_key": "k", "domain": "Banking", "agent": AGENT,
                                             "num_probes": 2, "categories": ["jailbreak", "bias"]})
    names = [n for n, _ in parse_sse(r.text)]
    assert names.count("rai_conversation") == 2
    assert "probes" in names and "rai_report" in names


def test_provider_errors_reach_the_ui():
    def bad_key(_key, _provider):
        raise LLMError("Anthropic rejected the API key (401).")

    c = TestClient(create_app(llm_factory=bad_key))
    r = c.post("/run-test", json={"api_key": "bad", "domain": "d", "agent": AGENT})
    events = parse_sse(r.text)
    assert events == [("error", {"message": "Anthropic rejected the API key (401)."})]


def test_serves_ui_and_health():
    c = client()
    assert c.get("/health").json()["status"] == "ready"
    page = c.get("/")
    assert page.status_code == 200 and "<html" in page.text.lower()
    assert c.get("/app.js").status_code == 200


def test_validation_matches_v2_limits():
    r = client().post("/run-test", json={"api_key": "k", "domain": "d", "agent": AGENT, "num_personas": 99})
    assert r.status_code == 422
