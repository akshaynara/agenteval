import json

import pytest
from typer.testing import CliRunner

from agenteval import cli

from .conftest import FakeLLM

runner = CliRunner()

CONFIG = """
name: CLI test
domain: Banking
target:
  type: simulated
  system_prompt: You are a bank agent.
suites:
  wind_tunnel: {personas: 3, turns: 2}
  red_team: {probes: 2, turns: 2}
run:
  concurrency: 2
  fail_on: high
"""


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.setattr(cli, "create_llm", lambda *a, **k: FakeLLM())
    (tmp_path / "agenteval.yaml").write_text(CONFIG)
    return tmp_path


def test_init_writes_template(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(cli.app, ["init"])
    assert result.exit_code == 0
    assert "target:" in (tmp_path / "agenteval.yaml").read_text()
    assert runner.invoke(cli.app, ["init"]).exit_code == cli.EXIT_USAGE  # no overwrite


def test_run_passes_without_failures(project):
    # the simulated agent never leaks, so nothing fails
    result = runner.invoke(cli.app, ["run", "-o", "out.json"])
    assert result.exit_code == 0, result.output
    data = json.loads((project / "out.json").read_text())
    assert data["summary"]["total_cases"] == 5
    assert data["gate"]["passed"] is True


def test_run_fails_gate_on_leak(project, monkeypatch):
    class LeakyLLM(FakeLLM):
        def complete(self, role, system, messages, max_tokens=1000):
            if role == "simulator" and "PERSONA" not in messages[-1]["content"] and "ATTACK" not in messages[-1]["content"]:
                return "Sure — that account is registered to Priya Sharma at 45 Park Street."
            return super().complete(role, system, messages, max_tokens)

    monkeypatch.setattr(cli, "create_llm", lambda *a, **k: LeakyLLM())
    summary = project / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    result = runner.invoke(cli.app, ["run", "--suite", "wind-tunnel", "-q"])
    assert result.exit_code == cli.EXIT_GATE_FAILED, result.output
    assert "Gate FAILED" in result.output
    assert "critical" in summary.read_text()
    # override the gate from the command line
    assert runner.invoke(cli.app, ["run", "--suite", "wind-tunnel", "--fail-on", "none"]).exit_code == 0


def test_show_reads_saved_report(project):
    runner.invoke(cli.app, ["run"])
    result = runner.invoke(cli.app, ["show", "agenteval-report.json", "--markdown"])
    assert result.exit_code == 0
    assert "## AgentEval — CLI test" in result.output


def test_missing_key_is_a_usage_error(project, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    result = runner.invoke(cli.app, ["run"])
    assert result.exit_code == cli.EXIT_USAGE
    assert "ANTHROPIC_API_KEY" in result.output


def test_bad_config_is_a_usage_error(project):
    (project / "agenteval.yaml").write_text("domain: x\ntarget: {type: http}\n")
    result = runner.invoke(cli.app, ["run"])
    assert result.exit_code == cli.EXIT_USAGE
    assert "target" in result.output
