import pytest

from agenteval.config import ConfigError, parse_config


def base(**over):
    cfg = {"domain": "Banking", "target": {"type": "http", "url": "http://localhost:9000/chat"}}
    cfg.update(over)
    return cfg


def test_defaults():
    cfg = parse_config(base())
    assert cfg.suites.wind_tunnel.enabled and cfg.suites.red_team.enabled
    assert len(cfg.suites.red_team.categories) == 7
    assert cfg.run.fail_on == "high"
    assert cfg.provider.name == "anthropic"


def test_env_expansion(monkeypatch):
    monkeypatch.setenv("AGENT_TOKEN", "s3cret")
    cfg = parse_config(base(target={"url": "${AGENT_URL:-http://fallback/chat}",
                                    "headers": {"Authorization": "Bearer ${AGENT_TOKEN}"}}))
    assert cfg.target.url == "http://fallback/chat"
    assert cfg.target.headers["Authorization"] == "Bearer s3cret"


def test_missing_env_is_reported(monkeypatch):
    monkeypatch.delenv("NOPE", raising=False)
    with pytest.raises(ConfigError, match="NOPE"):
        parse_config(base(target={"url": "${NOPE}"}))


@pytest.mark.parametrize(
    "data, msg",
    [
        ({"target": {"type": "http"}}, "target"),
        ({"domain": "x", "target": {"type": "simulated"}}, "system_prompt"),
        ({**base(), "suites": {"red_team": {"categories": ["telepathy"]}}}, "telepathy"),
        ({**base(), "run": {"fail_on": "pass"}}, "fail_on"),
        ({**base(), "provider": {"models": {"writer": "x"}}}, "writer"),
    ],
)
def test_validation_errors(data, msg):
    with pytest.raises(ConfigError, match=msg):
        parse_config(data)


def test_system_prompt_file_is_relative_to_config(tmp_path):
    (tmp_path / "prompt.md").write_text("You are a bank agent.")
    cfg = parse_config({"domain": "d", "target": {"type": "simulated", "system_prompt_file": "prompt.md"}},
                       base_dir=tmp_path)
    assert cfg.system_prompt() == "You are a bank agent."


def test_example_configs_are_valid():
    from pathlib import Path

    from agenteval.config import load_config

    root = Path(__file__).resolve().parents[1]
    configs = sorted((root / "examples").glob("*/agenteval.yaml"))
    assert configs
    for path in configs:
        cfg = load_config(path)
        if cfg.target.type == "simulated":
            assert cfg.system_prompt().strip()


def test_init_template_is_valid():
    from importlib import resources

    import yaml

    data = yaml.safe_load(resources.files("agenteval").joinpath("templates/agenteval.yaml").read_text())
    assert parse_config(data).target.url == "http://localhost:9000/chat"
