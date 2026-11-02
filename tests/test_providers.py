"""Provider + factory tests. The Gemini SDK client is replaced with a recorder; no network."""

import pytest

from agenteval import factory
from agenteval.config import ConfigError, parse_config
from agenteval.llm import LLMError, RoutedLLM
from agenteval.targets import CXAgentStudioTarget, DialogflowCXTarget, OpenAITarget

from .conftest import FakeLLM


class FakeGenAIClient:
    last = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        FakeGenAIClient.last = self
        outer = self

        class Models:
            def generate_content(self, model, contents, config):
                outer.calls.append({"model": model, "contents": contents, "config": config})

                class R:
                    text = " judged "

                return R()

        self.models = Models()


@pytest.fixture
def fake_genai(monkeypatch):
    from google import genai

    monkeypatch.setattr(genai, "Client", FakeGenAIClient)
    return FakeGenAIClient


def test_gemini_maps_roles_and_thinking(fake_genai):
    from agenteval.llm.gemini import GeminiLLM

    llm = GeminiLLM(api_key="AIza-test", models={"judge": "gemini-3.1-pro-preview"})
    out = llm.complete("judge", "You judge.", [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"},
                                               {"role": "user", "content": "verdict?"}], max_tokens=800)
    assert out == "judged"
    call = fake_genai.last.calls[0]
    assert call["model"] == "gemini-3.1-pro-preview"
    assert [c.role for c in call["contents"]] == ["user", "model", "user"]
    cfg = call["config"]
    assert cfg.system_instruction == "You judge."
    assert cfg.thinking_config.thinking_level.value == "MEDIUM"
    assert cfg.max_output_tokens > 800  # leaves room for thinking tokens
    assert fake_genai.last.kwargs["api_key"] == "AIza-test"


def test_gemini_skips_thinking_level_for_older_models(fake_genai):
    from agenteval.llm.gemini import GeminiLLM

    llm = GeminiLLM(api_key="k", models={"simulator": "gemini-2.5-flash"})
    llm.complete("simulator", "s", [{"role": "user", "content": "hi"}])
    assert fake_genai.last.calls[0]["config"].thinking_config is None


def test_gemini_vertex_needs_no_key(fake_genai):
    from agenteval.llm.gemini import GeminiLLM

    llm = GeminiLLM(vertexai=True, project="acme", location="us-central1")
    assert fake_genai.last.kwargs["vertexai"] is True and fake_genai.last.kwargs["project"] == "acme"
    assert "vertex" in llm.provider
    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        GeminiLLM(api_key="")


def test_gemini_auth_error_is_readable(fake_genai, monkeypatch):
    from google.genai import errors

    from agenteval.llm.gemini import GeminiLLM

    llm = GeminiLLM(api_key="bad")

    def boom(**kwargs):
        raise errors.ClientError(403, {"error": {"message": "API key not valid"}})

    monkeypatch.setattr(fake_genai.last.models, "generate_content", boom)
    with pytest.raises(LLMError, match="Gemini rejected the credentials"):
        llm.complete("judge", "s", [{"role": "user", "content": "x"}])


# ─── factory ───────────────────────────────────────────────────────────────


def cfg(**over):
    data = {"domain": "Banking", "target": {"type": "http", "url": "http://x/chat"}}
    data.update(over)
    return parse_config(data)


def test_build_llm_reads_the_right_key(fake_genai):
    llm = factory.build_llm(cfg(provider={"name": "gemini"}).provider, env={"GOOGLE_API_KEY": "AIza-g"})
    assert llm.provider == "gemini" and fake_genai.last.kwargs["api_key"] == "AIza-g"
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        factory.build_llm(cfg(provider={"name": "gemini"}).provider, env={})
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        factory.build_llm(cfg().provider, env={})


def test_cross_provider_judge(monkeypatch):
    made = []

    def fake_create(name, key, models=None, **opts):
        llm = FakeLLM()
        llm.provider = name
        llm.models = {"generator": f"{name}-gen", "simulator": f"{name}-sim", "judge": (models or {}).get("judge", f"{name}-judge")}
        made.append((name, key))
        return llm

    monkeypatch.setattr(factory, "create_llm", fake_create)
    spec = cfg(provider={"name": "gemini", "judge": {"name": "anthropic", "model": "claude-sonnet-4-6"}}).provider
    llm = factory.build_llm(spec, env={"GEMINI_API_KEY": "g", "ANTHROPIC_API_KEY": "a"})
    assert isinstance(llm, RoutedLLM)
    assert made == [("gemini", "g"), ("anthropic", "a")]
    assert llm.provider == "gemini + anthropic"
    assert llm.models["judge"] == "anthropic:claude-sonnet-4-6" and llm.models["simulator"] == "gemini:gemini-sim"
    assert llm.routes["judge"].provider == "anthropic"


@pytest.mark.parametrize(
    "target, cls",
    [
        ({"type": "openai", "base_url": "http://localhost:11434/v1", "model": "llama3"}, OpenAITarget),
        ({"type": "dialogflow_cx", "agent": "projects/p/locations/global/agents/a"}, DialogflowCXTarget),
        ({"type": "cx_agent_studio", "app": "projects/p/locations/us/apps/a", "use_tool_fakes": True}, CXAgentStudioTarget),
    ],
)
def test_build_each_target_type(target, cls):
    c = cfg(target=target)
    t = factory.build_target(c, FakeLLM(), env={})
    assert isinstance(t, cls)


@pytest.mark.parametrize(
    "target, msg",
    [
        ({"type": "openai", "base_url": "http://x/v1"}, "target.model"),
        ({"type": "dialogflow_cx"}, "target.agent"),
        ({"type": "dialogflow_cx", "agent": "my-agent"}, "projects/P/locations/L/agents"),
        ({"type": "cx_agent_studio", "app": "projects/p/apps"}, "projects/P/locations/L/apps"),
        ({"type": "carrier_pigeon"}, "type"),
    ],
)
def test_target_validation(target, msg):
    with pytest.raises(ConfigError, match=msg):
        cfg(target=target)


def test_openai_target_key_env(monkeypatch):
    c = cfg(target={"type": "openai", "base_url": "https://api.openai.com/v1", "model": "gpt", "api_key_env": "OPENAI_API_KEY"})
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        factory.build_target(c, FakeLLM(), env={})
    t = factory.build_target(c, FakeLLM(), env={"OPENAI_API_KEY": "sk-x"})
    assert t.headers["Authorization"] == "Bearer sk-x"


def test_gemini_network_errors_are_readable(fake_genai, monkeypatch):
    import httpx

    from agenteval.llm.gemini import GeminiLLM

    llm = GeminiLLM(api_key="k")

    def down(**kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(fake_genai.last.models, "generate_content", down)
    with pytest.raises(LLMError, match="Couldn't reach the Gemini API"):
        llm.complete("judge", "s", [{"role": "user", "content": "x"}])
