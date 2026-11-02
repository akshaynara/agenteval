"""YAML configuration for `agenteval run`.

Strings may reference environment variables as ``${NAME}`` or ``${NAME:-default}``,
so secrets (agent tokens, API keys) stay out of the file.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from .models import RAI_CATEGORIES, SEVERITY_RANK

DEFAULT_CONFIG_NAMES = ("agenteval.yaml", "agenteval.yml")
_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ConfigError(ValueError):
    pass


class TargetSpec(BaseModel):
    type: Literal["http", "simulated"] = "http"
    # http
    url: str | None = None
    reply_field: str = "reply"
    headers: dict[str, str] = Field(default_factory=dict)
    body: dict[str, Any] = Field(default_factory=dict, description="Extra JSON fields sent with every request")
    timeout: float = Field(30.0, gt=0)
    # simulated
    system_prompt: str | None = None
    system_prompt_file: str | None = None

    @model_validator(mode="after")
    def _check(self) -> TargetSpec:
        if self.type == "http" and not self.url:
            raise ValueError("target.url is required when target.type is 'http'")
        if self.type == "simulated" and not (self.system_prompt or self.system_prompt_file):
            raise ValueError("target.system_prompt (or system_prompt_file) is required when target.type is 'simulated'")
        return self


class ProviderSpec(BaseModel):
    name: str = "anthropic"
    api_key_env: str | None = None
    models: dict[str, str] = Field(default_factory=dict)

    @field_validator("models")
    @classmethod
    def _roles(cls, v: dict[str, str]) -> dict[str, str]:
        bad = set(v) - {"generator", "simulator", "judge"}
        if bad:
            raise ValueError(f"unknown model role(s) {sorted(bad)}; use generator, simulator, judge")
        return v


class WindTunnelSpec(BaseModel):
    enabled: bool = True
    personas: int = Field(10, ge=1, le=200)
    turns: int = Field(4, ge=1, le=12)


class RedTeamSpec(BaseModel):
    enabled: bool = True
    probes: int = Field(7, ge=1, le=200)
    turns: int = Field(4, ge=1, le=12)
    categories: list[str] = Field(default_factory=lambda: list(RAI_CATEGORIES))

    @field_validator("categories")
    @classmethod
    def _cats(cls, v: list[str]) -> list[str]:
        bad = [c for c in v if c not in RAI_CATEGORIES]
        if bad:
            raise ValueError(f"unknown categories {bad}; choose from {RAI_CATEGORIES}")
        return v or list(RAI_CATEGORIES)


class SuitesSpec(BaseModel):
    wind_tunnel: WindTunnelSpec = Field(default_factory=WindTunnelSpec)
    red_team: RedTeamSpec = Field(default_factory=RedTeamSpec)


class RunSpec(BaseModel):
    concurrency: int = Field(4, ge=1, le=32)
    fail_on: str = "high"
    output: str | None = "agenteval-report.json"

    @field_validator("fail_on")
    @classmethod
    def _fail_on(cls, v: str) -> str:
        v = (v or "none").lower()
        if v != "none" and (v not in SEVERITY_RANK or v == "pass"):
            raise ValueError("fail_on must be one of critical, high, medium, low, none")
        return v


class RunConfig(BaseModel):
    name: str = "AgentEval run"
    domain: str
    target: TargetSpec
    provider: ProviderSpec = Field(default_factory=ProviderSpec)
    suites: SuitesSpec = Field(default_factory=SuitesSpec)
    run: RunSpec = Field(default_factory=RunSpec)

    # Resolved at load time; not part of the YAML.
    base_dir: Path = Field(default=Path("."), exclude=True)

    def system_prompt(self) -> str:
        if self.target.system_prompt:
            return self.target.system_prompt
        path = Path(self.target.system_prompt_file or "")
        if not path.is_absolute():
            path = self.base_dir / path
        try:
            return path.read_text(encoding="utf-8")
        except OSError as e:
            raise ConfigError(f"Could not read target.system_prompt_file '{path}': {e}") from e


def expand_env(value: Any, missing: set[str]) -> Any:
    if isinstance(value, str):

        def sub(m: re.Match[str]) -> str:
            name, default = m.group(1), m.group(2)
            if name in os.environ:
                return os.environ[name]
            if default is not None:
                return default
            missing.add(name)
            return ""

        return _ENV_RE.sub(sub, value)
    if isinstance(value, dict):
        return {k: expand_env(v, missing) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v, missing) for v in value]
    return value


def find_config(path: str | os.PathLike[str] | None) -> Path:
    if path:
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"Config file not found: {p}")
        return p
    for name in DEFAULT_CONFIG_NAMES:
        if Path(name).exists():
            return Path(name)
    raise ConfigError("No agenteval.yaml in this folder. Create one with `agenteval init`, or pass a path.")


def parse_config(data: dict[str, Any], base_dir: Path = Path(".")) -> RunConfig:
    if not isinstance(data, dict):
        raise ConfigError("The config file must be a YAML mapping (key: value pairs).")
    missing: set[str] = set()
    data = expand_env(data, missing)
    if missing:
        raise ConfigError("Missing environment variable(s) used in the config: " + ", ".join(sorted(missing)))
    try:
        cfg = RunConfig.model_validate(data)
    except ValidationError as e:
        lines = []
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"]) or "(root)"
            lines.append(f"  {loc}: {err['msg']}")
        raise ConfigError("Invalid config:\n" + "\n".join(lines)) from None
    cfg.base_dir = base_dir
    return cfg


def load_config(path: str | os.PathLike[str] | None = None) -> RunConfig:
    p = find_config(path)
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"{p} is not valid YAML: {e}") from e
    return parse_config(data, base_dir=p.parent.resolve())
