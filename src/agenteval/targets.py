"""Targets: the agent under test.

Each test conversation opens its own session on the target, so stateful platforms
(Dialogflow CX, CX Agent Studio, agents keyed by session id) see one clean session
per synthetic customer.

- :class:`HttpTarget`          your agent behind any JSON-over-HTTP endpoint
- :class:`OpenAITarget`        any OpenAI-compatible chat completions endpoint
- :class:`DialogflowCXTarget`  a Dialogflow CX agent (detectIntent)
- :class:`CXAgentStudioTarget` a CX Agent Studio app (runSession)
- :class:`SimulatedTarget`     no endpoint yet: the simulator model plays your agent from its system prompt
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from .llm import LLM
from .models import AgentConfig, Turn


class TargetError(RuntimeError):
    """The agent can't be reached or rejected our credentials — stop the run instead of judging errors."""


@dataclass
class AgentReply:
    text: str
    meta: dict[str, Any] | None = None  # what the platform reported: intent, page, tool calls...
    end: bool = False  # the agent ended the session


class Session(Protocol):
    def respond(self, transcript: list[Turn]) -> AgentReply: ...


class Target(Protocol):
    def describe(self) -> str: ...

    def open_session(self) -> Session: ...


def get_path(data: Any, path: str) -> Any:
    """Read a dotted path such as ``reply`` or ``choices.0.message.content`` from parsed JSON."""
    cur = data
    for part in path.split("."):
        if isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return None
        elif isinstance(cur, dict):
            if part not in cur:
                return None
            cur = cur[part]
        else:
            return None
    return cur


def _messages(transcript: list[Turn]) -> list[dict[str, str]]:
    return [{"role": t.role, "content": t.content} for t in transcript]


def _latest_user_text(transcript: list[Turn]) -> str:
    for t in reversed(transcript):
        if t.role == "user":
            return t.content
    return ""


def _agent_error(e: Exception) -> AgentReply:
    return AgentReply(text=f"(AGENT ERROR: {e})", meta={"error": True})


# ─── Generic HTTP ──────────────────────────────────────────────────────────


class HttpTarget:
    """POSTs ``{"messages": [...], **body}`` and reads ``reply_field`` from the JSON response.

    Set ``session_field`` (e.g. ``"session_id"``) to also send a fresh id per test conversation.
    """

    def __init__(
        self,
        url: str,
        reply_field: str = "reply",
        headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout: float = 30.0,
        session_field: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        if not url:
            raise ValueError("HTTP target needs a url")
        self.url = url
        self.reply_field = reply_field or "reply"
        self.headers = headers or {}
        self.extra_body = extra_body or {}
        self.timeout = timeout
        self.session_field = session_field
        self._transport = transport

    def describe(self) -> str:
        return f"http {self.url}"

    def open_session(self) -> _HttpSession:
        return _HttpSession(self, f"agenteval-{uuid.uuid4().hex[:16]}" if self.session_field else None)

    def respond(self, transcript: list[Turn]) -> str:
        """One stateless call (kept for convenience and backwards compatibility)."""
        return self.open_session().respond(transcript).text


class _HttpSession:
    def __init__(self, target: HttpTarget, session_id: str | None):
        self.t = target
        self.session_id = session_id

    def respond(self, transcript: list[Turn]) -> AgentReply:
        t = self.t
        payload: dict[str, Any] = {**t.extra_body, "messages": _messages(transcript)}
        if t.session_field and self.session_id:
            payload[t.session_field] = self.session_id
        try:
            with httpx.Client(timeout=t.timeout, transport=t._transport) as http:
                r = http.post(t.url, json=payload, headers=t.headers)
                r.raise_for_status()
                value = get_path(r.json(), t.reply_field)
        except Exception as e:  # the judge sees agent errors as part of the conversation
            return _agent_error(e)
        if value is None or value == "":
            return AgentReply(f"(AGENT ERROR: no '{t.reply_field}' field in the response)", meta={"error": True})
        return AgentReply(str(value))


# ─── OpenAI-compatible ─────────────────────────────────────────────────────


class OpenAITarget:
    """Any OpenAI-compatible ``/chat/completions`` endpoint: OpenAI, Azure OpenAI, vLLM, Ollama, LiteLLM, ..."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        system_prompt: str | None = None,
        headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ):
        if not base_url or not model:
            raise ValueError("OpenAI-compatible target needs base_url and model")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.system_prompt = system_prompt
        self.headers = dict(headers or {})
        if api_key:
            self.headers.setdefault("Authorization", f"Bearer {api_key}")
        self.extra_body = extra_body or {}
        self.timeout = timeout
        self._transport = transport

    def describe(self) -> str:
        return f"openai-compatible {self.model} @ {self.url}"

    def open_session(self) -> OpenAITarget:
        return self  # stateless: the full history is sent every turn

    def respond(self, transcript: list[Turn]) -> AgentReply:
        messages = _messages(transcript)
        if self.system_prompt:
            messages = [{"role": "system", "content": self.system_prompt}, *messages]
        payload = {**self.extra_body, "model": self.model, "messages": messages}
        try:
            with httpx.Client(timeout=self.timeout, transport=self._transport) as http:
                r = http.post(self.url, json=payload, headers=self.headers)
                if r.status_code in (401, 403, 404):
                    raise TargetError(f"{self.url} returned {r.status_code}: {r.text[:200]}")
                r.raise_for_status()
                text = get_path(r.json(), "choices.0.message.content")
        except TargetError:
            raise
        except Exception as e:
            return _agent_error(e)
        return AgentReply(str(text or "").strip() or "(AGENT ERROR: empty completion)")


# ─── Google Cloud (Dialogflow CX, CX Agent Studio) ─────────────────────────


class _GoogleToken:
    """A bearer token: a fixed one you pass in, or Application Default Credentials refreshed as needed."""

    SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

    def __init__(self, access_token: str | None = None, credentials: Any = None):
        self._fixed = access_token
        self._creds = credentials
        self._lock = threading.Lock()

    def header(self) -> dict[str, str]:
        if self._fixed:
            return {"Authorization": f"Bearer {self._fixed}"}
        with self._lock:
            if self._creds is None:
                try:
                    import google.auth
                    import google.auth.transport.requests  # noqa: F401
                except ImportError as e:
                    raise TargetError(
                        'Google targets need google-auth: pip install "agenteval[google]" '
                        "(or set target.access_token, e.g. from `gcloud auth print-access-token`)."
                    ) from e
                try:
                    self._creds, _ = google.auth.default(scopes=self.SCOPES)
                except Exception as e:
                    raise TargetError(
                        "No Google credentials found. Run `gcloud auth application-default login`, "
                        "set GOOGLE_APPLICATION_CREDENTIALS, or set target.access_token."
                    ) from e
            if not self._creds.valid:
                import google.auth.transport.requests

                self._creds.refresh(google.auth.transport.requests.Request())
            return {"Authorization": f"Bearer {self._creds.token}"}


def _google_post(url: str, body: dict, token: _GoogleToken, timeout: float, transport, quota_project: str | None):
    headers = token.header()
    if quota_project:
        headers["x-goog-user-project"] = quota_project
    with httpx.Client(timeout=timeout, transport=transport) as http:
        r = http.post(url, json=body, headers=headers)
    if r.status_code in (401, 403, 404):
        raise TargetError(f"Google API returned {r.status_code} for {url}: {r.text[:300]}")
    r.raise_for_status()
    return r.json()


def _location_of(resource: str) -> str:
    parts = resource.split("/")
    return parts[parts.index("locations") + 1] if "locations" in parts else "global"


class DialogflowCXTarget:
    """A Dialogflow CX agent. Sends each customer message to ``detectIntent`` in a fresh session.

    ``agent`` is the full resource name: ``projects/P/locations/L/agents/A``.
    The judge also sees which intent matched and which page the agent landed on.
    """

    def __init__(
        self,
        agent: str,
        environment: str | None = None,
        language_code: str = "en",
        access_token: str | None = None,
        quota_project: str | None = None,
        endpoint: str | None = None,
        timeout: float = 30.0,
        credentials: Any = None,
        transport: httpx.BaseTransport | None = None,
    ):
        agent = agent.strip().strip("/")
        if not agent.startswith("projects/") or "/agents/" not in agent:
            raise ValueError("target.agent must look like projects/P/locations/L/agents/AGENT_ID")
        self.agent = agent
        self.environment = environment
        self.language_code = language_code
        loc = _location_of(agent)
        host = endpoint or ("dialogflow.googleapis.com" if loc == "global" else f"{loc}-dialogflow.googleapis.com")
        self.base = f"https://{host}/v3"
        self.timeout = timeout
        self.quota_project = quota_project
        self._token = _GoogleToken(access_token, credentials)
        self._transport = transport

    def describe(self) -> str:
        return f"dialogflow-cx {self.agent.split('/')[-1]}" + (f" ({self.environment})" if self.environment else "")

    def open_session(self) -> _DialogflowSession:
        prefix = self.agent
        if self.environment:
            env = self.environment if "/" in self.environment else f"{self.agent}/environments/{self.environment}"
            prefix = env
        return _DialogflowSession(self, f"{prefix}/sessions/agenteval-{uuid.uuid4().hex[:20]}")


class _DialogflowSession:
    def __init__(self, target: DialogflowCXTarget, session: str):
        self.t = target
        self.session = session

    def respond(self, transcript: list[Turn]) -> AgentReply:
        t = self.t
        body = {
            "queryInput": {"text": {"text": _latest_user_text(transcript)}, "languageCode": t.language_code}
        }
        try:
            data = _google_post(f"{t.base}/{self.session}:detectIntent", body, t._token, t.timeout, t._transport,
                                t.quota_project)
        except TargetError:
            raise
        except Exception as e:
            return _agent_error(e)
        qr = data.get("queryResult", {})
        texts: list[str] = []
        end = False
        for m in qr.get("responseMessages", []):
            texts += [s for s in (m.get("text", {}) or {}).get("text", []) if s]
            if "endInteraction" in m:
                end = True
        page = qr.get("currentPage", {}) or {}
        if page.get("name", "").endswith("END_SESSION") or page.get("displayName") == "End Session":
            end = True
        match = qr.get("match", {}) or {}
        meta = {
            k: v
            for k, v in {
                "intent": (match.get("intent") or {}).get("displayName") or (qr.get("intent") or {}).get("displayName"),
                "match_type": match.get("matchType"),
                "confidence": match.get("confidence"),
                "page": page.get("displayName"),
            }.items()
            if v not in (None, "")
        }
        return AgentReply(text="\n".join(texts) or "(no text response)", meta=meta or None, end=end)


class CXAgentStudioTarget:
    """A CX Agent Studio app. Sends each customer message to ``runSession`` in a fresh session.

    ``app`` is the full resource name: ``projects/P/locations/L/apps/APP_ID``.
    Set ``use_tool_fakes`` to red-team the agent without its tools touching real backends.
    """

    def __init__(
        self,
        app: str,
        deployment: str | None = None,
        entry_agent: str | None = None,
        use_tool_fakes: bool = False,
        access_token: str | None = None,
        quota_project: str | None = None,
        endpoint: str = "ces.googleapis.com",
        api_version: str = "v1",
        timeout: float = 60.0,
        credentials: Any = None,
        transport: httpx.BaseTransport | None = None,
    ):
        app = app.strip().strip("/")
        if not app.startswith("projects/") or "/apps/" not in app:
            raise ValueError("target.app must look like projects/P/locations/L/apps/APP_ID")
        self.app = app
        self.deployment = deployment if not deployment or "/" in deployment else f"{app}/deployments/{deployment}"
        self.entry_agent = entry_agent if not entry_agent or "/" in entry_agent else f"{app}/agents/{entry_agent}"
        self.use_tool_fakes = use_tool_fakes
        self.base = f"https://{endpoint}/{api_version}"
        self.timeout = timeout
        self.quota_project = quota_project
        self._token = _GoogleToken(access_token, credentials)
        self._transport = transport

    def describe(self) -> str:
        return f"cx-agent-studio {self.app.split('/')[-1]}" + (" (tool fakes)" if self.use_tool_fakes else "")

    def open_session(self) -> _CESSession:
        # session ids must match [a-zA-Z0-9][a-zA-Z0-9-_]{4,62}
        return _CESSession(self, f"{self.app}/sessions/agenteval-{uuid.uuid4().hex[:24]}")


class _CESSession:
    def __init__(self, target: CXAgentStudioTarget, session: str):
        self.t = target
        self.session = session

    def respond(self, transcript: list[Turn]) -> AgentReply:
        t = self.t
        config: dict[str, Any] = {"session": self.session}
        if t.deployment:
            config["deployment"] = t.deployment
        if t.entry_agent:
            config["entryAgent"] = t.entry_agent
        if t.use_tool_fakes:
            config["useToolFakes"] = True
        body = {"config": config, "inputs": [{"text": _latest_user_text(transcript)}]}
        try:
            data = _google_post(f"{t.base}/{self.session}:runSession", body, t._token, t.timeout, t._transport,
                                t.quota_project)
        except TargetError:
            raise
        except Exception as e:
            return _agent_error(e)
        texts: list[str] = []
        tools: list[str] = []
        end = False
        for out in data.get("outputs", []):
            if out.get("text"):
                texts.append(out["text"])
            if "endSession" in out:
                end = True
            for call in (out.get("toolCalls") or {}).get("toolCalls", []):
                name = call.get("displayName") or call.get("tool") or call.get("name")
                if name:
                    tools.append(str(name).split("/")[-1])
        meta = {"tool_calls": tools} if tools else None
        if end:
            meta = {**(meta or {}), "ended_session": True}
        return AgentReply(text="\n".join(texts) or "(no text response)", meta=meta, end=end)


# ─── Simulated ─────────────────────────────────────────────────────────────


@dataclass
class SimulatedTarget:
    """Demo mode: the simulator model plays your agent using the given system prompt."""

    llm: LLM
    system_prompt: str = field(default="You are a helpful customer service agent.")

    def __post_init__(self):
        self.system_prompt = self.system_prompt or "You are a helpful customer service agent."

    def describe(self) -> str:
        return "simulated (system prompt)"

    def open_session(self) -> SimulatedTarget:
        return self

    def respond(self, transcript: list[Turn]) -> AgentReply:
        return AgentReply(self.llm.complete("simulator", self.system_prompt, _messages(transcript), max_tokens=500))


# ─── Helpers ───────────────────────────────────────────────────────────────


def agent_turn(session: Session, transcript: list[Turn]) -> tuple[Turn, bool]:
    """Ask the agent for its next reply. Returns the turn and whether the agent ended the session."""
    reply = session.respond(transcript)
    if isinstance(reply, str):  # tolerate simple custom targets that return plain text
        reply = AgentReply(reply)
    return Turn(role="assistant", content=reply.text, meta=reply.meta), reply.end


def check_reachable(target: Target) -> None:
    """Send one greeting before spending tokens on a run; raise TargetError if the agent can't be reached."""
    if isinstance(target, SimulatedTarget):
        return
    turn, _ = agent_turn(target.open_session(), [Turn(role="user", content="Hello")])
    if turn.content.startswith("(AGENT ERROR"):
        raise TargetError(f"Your agent at {target.describe()} didn't answer a test message: {turn.content}")


def target_from_agent_config(agent: AgentConfig, llm: LLM) -> Target:
    """Build a target from the web UI's AgentConfig."""
    if agent.mode == "endpoint":
        return HttpTarget(url=agent.endpoint_url or "", reply_field=agent.reply_field)
    return SimulatedTarget(llm, agent.system_prompt or "")
