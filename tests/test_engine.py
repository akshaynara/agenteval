import pytest

from agenteval import engine
from agenteval._parsing import extract_json
from agenteval.targets import HttpTarget, SimulatedTarget, get_path

from .conftest import demo_agent_transport


def test_extract_json_tolerates_fences_and_prose():
    assert extract_json('Here:\n```json\n[{"a": 1}]\n```\nDone', list) == [{"a": 1}]
    assert extract_json('verdict: {"failed": true} ok', dict) == {"failed": True}
    with pytest.raises(ValueError):
        extract_json("no json here", list)


def test_get_path_reads_dotted_paths():
    data = {"choices": [{"message": {"content": "hi"}}], "reply": "x"}
    assert get_path(data, "reply") == "x"
    assert get_path(data, "choices.0.message.content") == "hi"
    assert get_path(data, "choices.5.message") is None
    assert get_path(data, "missing") is None


def test_wind_tunnel_finds_demo_agent_pii_leak(fake_llm):
    target = HttpTarget("http://demo/chat", transport=demo_agent_transport())
    events = []
    report = engine.run_wind_tunnel(fake_llm, target, "Banking", num_personas=3, turns=3,
                                    on_event=lambda n, d: events.append(n))

    assert report.total_personas == 3
    assert report.total_failures == 1
    leak = next(f for f in report.findings if f.failed)
    assert leak.persona_name == "The Account Number Prober"
    assert leak.severity == "critical" and leak.was_unexpected
    # risk level normalized from "EDGE_CASE"
    assert report.conversations[2].persona.risk_level == "edge_case"
    assert events[0] == "status" and "personas" in events and events[-1] == "report"
    assert events.count("conversation") == 3
    # the persona said "(END)", so the conversation stops after the agent's first reply
    assert [t.role for t in leak_transcript(report)] == ["user", "assistant"]


def leak_transcript(report):
    return next(c.transcript for c in report.conversations if c.finding.failed)


def test_red_team_normalizes_categories(fake_llm):
    target = HttpTarget("http://demo/chat", transport=demo_agent_transport())
    report = engine.run_red_team(fake_llm, target, "Banking", ["jailbreak"], num_probes=2, turns=2)
    assert [c.probe.category for c in report.conversations] == ["jailbreak", "scope_violation"]
    assert report.total_failures == 1 and report.findings[0].attack_succeeded


def test_concurrency_preserves_case_order(fake_llm):
    target = SimulatedTarget(fake_llm, "You are a bank agent.")
    seq = engine.run_wind_tunnel(fake_llm, target, "Banking", 3, 2, concurrency=1)
    par = engine.run_wind_tunnel(fake_llm, target, "Banking", 3, 2, concurrency=3)
    assert [c.persona.id for c in par.conversations] == [c.persona.id for c in seq.conversations] == ["p01", "p02", "p03"]


def test_http_target_reports_agent_errors_in_transcript():
    import httpx

    target = HttpTarget("http://x/chat", transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    assert target.respond([]).startswith("(AGENT ERROR")
    target = HttpTarget("http://x/chat", reply_field="text",
                        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"reply": "hi"})))
    assert "no 'text' field" in target.respond([])


@pytest.mark.parametrize(
    "fail_on, passed",
    [("critical", False), ("high", False), ("none", True), (None, True)],
)
def test_gate(fake_llm, fail_on, passed):
    target = HttpTarget("http://demo/chat", transport=demo_agent_transport())
    report = engine.run(llm=fake_llm, target=target, name="t", domain="Banking",
                        wind_tunnel_opts={"personas": 3, "turns": 2}, red_team_opts=None, fail_on=fail_on)
    assert report.gate.passed is passed
    assert report.summary.severity_counts["critical"] == 1
    assert report.summary.total_cases == 3


def test_gate_rejects_bad_threshold(fake_llm):
    target = SimulatedTarget(fake_llm, "x")
    with pytest.raises(ValueError):
        engine.run(llm=fake_llm, target=target, name="t", domain="d", wind_tunnel_opts={"personas": 1},
                   red_team_opts=None, fail_on="pass")
