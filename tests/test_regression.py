import json
import re

from typer.testing import CliRunner

from agenteval import cli, engine, factory
from agenteval.html_report import render_html
from agenteval.regression import compare
from agenteval.report import load_json, save_json
from agenteval.targets import HttpTarget, SimulatedTarget

from .conftest import FakeLLM, demo_agent_transport


def demo_run(llm, **kw):
    target = HttpTarget("http://demo/chat", transport=demo_agent_transport())
    return engine.run(llm=llm, target=target, name="demo", domain="Banking",
                      wind_tunnel_opts={"personas": 3, "turns": 2}, red_team_opts={"probes": 2, "turns": 2},
                      fail_on="high", **kw)


class FixedAgentLLM(FakeLLM):
    """The judge now says the PII leak is fixed, but a different case breaks."""

    def complete(self, role, system, messages, max_tokens=1000):
        if role == "judge" and "EQUALLY" not in system:
            text = messages[-1]["content"]
            broke = "Calm Balance Checker" in text
            return json.dumps({"failed": broke, "attack_succeeded": False, "failure_type": "wrong_routing" if broke else None,
                               "severity": "high" if broke else "pass", "title": "Balance request misrouted" if broke else "OK",
                               "explanation": "x", "was_unexpected": False})
        return super().complete(role, system, messages, max_tokens)


def test_baseline_replays_the_same_cases_without_generating(fake_llm):
    before = demo_run(fake_llm)
    llm2 = FixedAgentLLM()
    after = demo_run(llm2, baseline=before)
    assert llm2.calls["generator"] == 0  # nothing regenerated
    assert [c.persona for c in after.wind_tunnel.conversations] == [c.persona for c in before.wind_tunnel.conversations]
    assert [c.probe for c in after.red_team.conversations] == [c.probe for c in before.red_team.conversations]

    c = after.comparison
    statuses = {(x.suite, x.case): x.status for x in c.cases}
    assert statuses[("wind tunnel", "The Account Number Prober")] == "fixed"
    assert statuses[("red team", "The Auditor Override")] == "fixed"
    assert statuses[("wind tunnel", "The Calm Balance Checker")] == "regressed"
    assert (c.fixed, c.regressed) == (2, 1)
    assert c.cases[0].status == "regressed"  # worst news first
    assert not after.gate.passed and "regressed since the baseline" in after.gate.reason


def test_regression_gate_can_be_turned_off(fake_llm):
    before = demo_run(fake_llm)
    after = demo_run(FixedAgentLLM(), baseline=before, fail_on_regression=False)
    assert "regressed" not in after.gate.reason  # falls back to the severity gate
    assert not after.gate.passed  # the new high finding still trips fail_on: high


def test_compare_handles_new_and_removed_cases(fake_llm):
    a = demo_run(fake_llm)
    b = a.model_copy(deep=True)
    b.red_team = None
    cmp = compare(a, b)
    assert {x.status for x in cmp.cases if x.suite == "red team"} == {"removed"}


def test_language_scenarios_replay(fake_llm):
    target = SimulatedTarget(fake_llm, "You are a bank agent.")
    kw = dict(llm=fake_llm, target=target, name="t", domain="Banking", wind_tunnel_opts=None, red_team_opts=None,
              language_opts={"scenarios": 2, "accents": ["british_english"], "code_switch": ["hinglish"]})
    before = engine.run(**kw)
    gen_calls = fake_llm.calls["generator"]
    after = engine.run(**kw, baseline=before)
    assert fake_llm.calls["generator"] == gen_calls
    assert [s.scenario for s in after.language.scenarios] == [s.scenario for s in before.language.scenarios]
    assert after.comparison.still_failing == 2


def test_html_report_is_self_contained_and_escaped(fake_llm, tmp_path):
    report = demo_run(fake_llm)
    report.name = "<script>alert(1)</script>"
    html = render_html(report)
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;" in html
    assert "Confirmed a stranger&#x27;s identity" in html
    assert "Gate failed" in html
    assert not re.search(r"<(link|img)[^>]+(src|href)=['\"]https?://", html)  # no external assets
    raw = re.search(r"<script type='application/json' id='agenteval-report'>(.*?)</script>", html, re.S).group(1)
    assert json.loads(raw.replace("<\\/", "</"))["name"] == "<script>alert(1)</script>"


def test_html_report_with_comparison_and_language(fake_llm):
    before = demo_run(fake_llm)
    after = demo_run(FixedAgentLLM(), baseline=before)
    html = render_html(after)
    assert "Compared with the baseline" in html and "regressed" in html
    target = SimulatedTarget(fake_llm, "x")
    lang = engine.run(llm=fake_llm, target=target, name="t", domain="d", wind_tunnel_opts=None, red_team_opts=None,
                      language_opts={"scenarios": 1, "code_switch": ["hinglish"]})
    html = render_html(lang)
    assert "language consistency" in html and "Hinglish" in html


# ─── CLI ───────────────────────────────────────────────────────────────────

CONFIG = """
name: Regression test
domain: Banking
target: {type: simulated, system_prompt: You are a bank agent.}
suites:
  wind_tunnel: {personas: 3, turns: 2}
  red_team: {enabled: false}
  language: {scenarios: 1, accents: [british_english], code_switch: [hinglish], asr: []}
run: {fail_on: none, output: run1.json}
"""


def test_cli_regression_flow(tmp_path, monkeypatch):
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.setattr(factory, "create_llm", lambda *a, **k: FakeLLM())
    (tmp_path / "agenteval.yaml").write_text(CONFIG)

    r = runner.invoke(cli.app, ["run", "--suite", "language", "--html", "run1.html"])
    assert r.exit_code == 0, r.output
    first = load_json(tmp_path / "run1.json")
    assert first.language and first.wind_tunnel is None
    assert (tmp_path / "run1.html").read_text().startswith("<!doctype html>")

    r = runner.invoke(cli.app, ["run", "--baseline", "run1.json", "-o", "run2.json"])
    assert r.exit_code == 0, r.output
    assert "vs baseline" in r.output
    second = load_json(tmp_path / "run2.json")
    assert second.wind_tunnel is None  # only replays the suites the baseline ran
    assert second.comparison.still_failing == 1

    r = runner.invoke(cli.app, ["compare", "run1.json", "run2.json", "--html", "diff.html"])
    assert r.exit_code == 0 and (tmp_path / "diff.html").exists()

    r = runner.invoke(cli.app, ["show", "run2.json", "--html", "again.html"])
    assert r.exit_code == 0 and (tmp_path / "again.html").exists()


def test_cli_compare_exits_1_on_regression(tmp_path, fake_llm):
    before = demo_run(fake_llm)
    after = demo_run(FixedAgentLLM(), baseline=before)
    save_json(before, tmp_path / "a.json")
    save_json(after, tmp_path / "b.json")
    r = CliRunner().invoke(cli.app, ["compare", str(tmp_path / "a.json"), str(tmp_path / "b.json")])
    assert r.exit_code == cli.EXIT_GATE_FAILED
    assert "1 regressed" in r.output


def test_fixed_demo_agent_turns_failures_into_fixes(fake_llm, monkeypatch):
    before = demo_run(fake_llm)
    assert before.summary.severity_counts["critical"] == 2  # PII leaks
    monkeypatch.setenv("AGENTEVAL_DEMO_FIXED", "1")
    after = demo_run(FakeLLM(), baseline=before)
    assert after.comparison.fixed == 2 and after.comparison.regressed == 0
    assert after.gate.passed
    reply = after.wind_tunnel.conversations[1].transcript[1].content
    assert "can't confirm" in reply and "registered to" not in reply
