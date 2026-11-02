"""Build providers and targets from a :class:`~agenteval.config.RunConfig`."""

from __future__ import annotations

import os
from collections.abc import Mapping

from .config import ConfigError, JudgeProviderSpec, ProviderSpec, RunConfig
from .llm import LLM, RoutedLLM, create_llm, default_api_key_env
from .targets import (
    CXAgentStudioTarget,
    DialogflowCXTarget,
    HttpTarget,
    OpenAITarget,
    SimulatedTarget,
    Target,
)


def _api_key(name: str, key_env: str | None, vertexai: bool, env: Mapping[str, str]) -> str | None:
    if vertexai:
        return None
    var = key_env or default_api_key_env(name)
    key = env.get(var, "")
    if not key and name == "gemini" and not key_env:
        key = env.get("GOOGLE_API_KEY", "")  # the Google SDK's other conventional name
    if not key:
        hint = " (or set provider.vertexai: true to use Vertex AI)" if name == "gemini" else ""
        raise ConfigError(f"Set {var} to your {name} API key{hint}. It's only used for this run.")
    return key


def _gemini_options(spec: ProviderSpec | JudgeProviderSpec) -> dict:
    if spec.name != "gemini":
        return {}
    opts = {"vertexai": spec.vertexai, "project": spec.project, "location": spec.location}
    if isinstance(spec, ProviderSpec) and spec.thinking:
        opts["thinking"] = spec.thinking
    return opts


def build_llm(spec: ProviderSpec, env: Mapping[str, str] | None = None) -> LLM:
    env = os.environ if env is None else env
    main = create_llm(spec.name, _api_key(spec.name, spec.api_key_env, spec.vertexai, env), spec.models,
                      **_gemini_options(spec))
    if not spec.judge:
        return main
    j = spec.judge
    if j.name not in ("anthropic", "gemini"):
        raise ConfigError(f"provider.judge.name must be anthropic or gemini, got '{j.name}'")
    judge = create_llm(j.name, _api_key(j.name, j.api_key_env, j.vertexai, env),
                       {"judge": j.model} if j.model else None, **_gemini_options(j))
    return RoutedLLM({"generator": main, "simulator": main, "judge": judge})


def build_target(cfg: RunConfig, llm: LLM, env: Mapping[str, str] | None = None) -> Target:
    env = os.environ if env is None else env
    t = cfg.target
    if t.type == "http":
        return HttpTarget(url=t.url or "", reply_field=t.reply_field, headers=t.headers, extra_body=t.body,
                          timeout=t.timeout or 30.0, session_field=t.session_field)
    if t.type == "simulated":
        return SimulatedTarget(llm, cfg.system_prompt())
    if t.type == "openai":
        api_key = env.get(t.api_key_env, "") if t.api_key_env else None
        if t.api_key_env and not api_key:
            raise ConfigError(f"Set {t.api_key_env} (target.api_key_env) for your agent's endpoint.")
        prompt = cfg.system_prompt() if (t.system_prompt or t.system_prompt_file) else None
        return OpenAITarget(base_url=t.base_url or "", model=t.model or "", api_key=api_key, system_prompt=prompt,
                            headers=t.headers, extra_body=t.body, timeout=t.timeout or 60.0)
    if t.type == "dialogflow_cx":
        return DialogflowCXTarget(agent=t.agent or "", environment=t.environment, language_code=t.language_code,
                                  access_token=t.access_token, quota_project=t.quota_project, endpoint=t.endpoint,
                                  timeout=t.timeout or 30.0)
    if t.type == "cx_agent_studio":
        return CXAgentStudioTarget(app=t.app or "", deployment=t.deployment, entry_agent=t.entry_agent,
                                   use_tool_fakes=t.use_tool_fakes, access_token=t.access_token,
                                   quota_project=t.quota_project, endpoint=t.endpoint or "ces.googleapis.com",
                                   api_version=t.api_version, timeout=t.timeout or 60.0)
    raise ConfigError(f"Unknown target type '{t.type}'")
