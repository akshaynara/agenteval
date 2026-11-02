"""Web API + UI. Same engine as the CLI, streamed to the browser as server-sent events.

Endpoints (unchanged from v2, so the hosted frontend keeps working):
  GET  /health         health check
  POST /run-test       Wind Tunnel (operational) test, SSE
  POST /run-rai-test   Red-team (safety) test, SSE
  POST /run-language-test  Accent & language variation test, SSE
  GET  /language-profiles  Built-in accent / code-switch / ASR profiles
  GET  /               the web UI, when the frontend files are available

Bring Your Own Key: the user's API key is passed per request and never stored.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import traceback
from collections.abc import Callable, Iterator
from importlib import resources
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from . import __version__, engine
from .llm import LLM, LLMError, create_llm
from .models import LanguageTestConfig, RAITestConfig, TestConfig
from .targets import TargetError, target_from_agent_config

WEB_CONCURRENCY = int(os.environ.get("AGENTEVAL_WEB_CONCURRENCY", "3"))
# Free Gemini keys have low per-minute limits; fewer parallel conversations avoids hitting them.
GEMINI_WEB_CONCURRENCY = int(os.environ.get("AGENTEVAL_GEMINI_WEB_CONCURRENCY", "2"))


def _concurrency(provider: str) -> int:
    return GEMINI_WEB_CONCURRENCY if provider == "gemini" else WEB_CONCURRENCY
_ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("AGENTEVAL_CORS_ORIGINS", "*").split(",") if o.strip()]


def _default_llm_factory(api_key: str, provider: str = "anthropic") -> LLM:
    return create_llm(provider, api_key)


class _Cancelled(Exception):
    pass


def _find_web_dir() -> Path | None:
    """The UI ships inside the wheel as agenteval/web; in a source checkout it's ../../frontend."""
    candidates = [
        Path(str(resources.files("agenteval").joinpath("web"))),
        Path(__file__).resolve().parents[2] / "frontend",
    ]
    for c in candidates:
        if (c / "index.html").is_file():
            return c
    return None


def create_app(llm_factory: Callable[[str, str], LLM] = _default_llm_factory) -> FastAPI:
    app = FastAPI(title="AgentEval", version=__version__)
    app.add_middleware(CORSMiddleware, allow_origins=_ALLOWED_ORIGINS, allow_methods=["*"], allow_headers=["*"])
    app.state.llm_factory = llm_factory

    def sse(event: str, data: dict[str, Any]) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    def stream(job: Callable[[engine.EventHandler], None], done_message: str) -> StreamingResponse:
        events: queue.Queue[tuple[str, dict[str, Any]] | None] = queue.Queue()
        cancelled = threading.Event()

        def on_event(name: str, data: dict[str, Any]) -> None:
            if cancelled.is_set():
                raise _Cancelled()
            events.put((name, data))

        def worker() -> None:
            try:
                job(on_event)
                events.put(("status", {"message": done_message, "done": True}))
            except _Cancelled:
                pass
            except (LLMError, TargetError) as e:
                events.put(("error", {"message": str(e)}))
            except Exception as e:  # surface everything else to the UI
                events.put(("error", {"message": str(e), "detail": traceback.format_exc()[-500:]}))
            finally:
                events.put(None)

        def gen() -> Iterator[str]:
            threading.Thread(target=worker, daemon=True).start()
            try:
                while True:
                    item = events.get()
                    if item is None:
                        return
                    yield sse(*item)
            finally:
                cancelled.set()  # client went away: stop spending the user's tokens

        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"name": "AgentEval", "version": __version__, "status": "ready"}

    @app.post("/run-test")
    def run_test(config: TestConfig) -> StreamingResponse:
        def job(on_event: engine.EventHandler) -> None:
            llm = app.state.llm_factory(config.api_key, config.provider)
            target = target_from_agent_config(config.agent, llm)
            engine.run_wind_tunnel(llm, target, config.domain, config.num_personas, config.turns_per_conversation,
                                   concurrency=_concurrency(config.provider), on_event=on_event)

        return stream(job, "Test complete.")

    @app.post("/run-rai-test")
    def run_rai_test(config: RAITestConfig) -> StreamingResponse:
        def job(on_event: engine.EventHandler) -> None:
            llm = app.state.llm_factory(config.api_key, config.provider)
            target = target_from_agent_config(config.agent, llm)
            engine.run_red_team(llm, target, config.domain, list(config.categories), config.num_probes,
                                config.turns_per_conversation, concurrency=_concurrency(config.provider), on_event=on_event)

        return stream(job, "RAI test complete.")

    @app.post("/run-language-test")
    def run_language_test(config: LanguageTestConfig) -> StreamingResponse:
        def job(on_event: engine.EventHandler) -> None:
            llm = app.state.llm_factory(config.api_key, config.provider)
            target = target_from_agent_config(config.agent, llm)
            engine.run_language(llm, target, config.domain, config.num_scenarios, config.turns_per_conversation,
                                accents=config.accents, code_switch=config.code_switch, asr=config.asr,
                                asr_error_rate=config.asr_error_rate, concurrency=_concurrency(config.provider),
                                on_event=on_event)

        return stream(job, "Language test complete.")

    @app.get("/language-profiles")
    def language_profiles() -> dict[str, Any]:
        from .suites.language import ACCENT_PROFILES, ASR_CONFUSIONS, CODE_SWITCH_PROFILES

        return {
            "accents": {k: v["label"] for k, v in ACCENT_PROFILES.items()},
            "code_switch": {k: v["label"] for k, v in CODE_SWITCH_PROFILES.items()},
            "asr": sorted(ASR_CONFUSIONS),
        }

    web = _find_web_dir()
    if web:
        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(web / "index.html")

        @app.get("/app.js", include_in_schema=False)
        def app_js() -> FileResponse:
            return FileResponse(web / "app.js", media_type="application/javascript")
    else:

        @app.get("/", include_in_schema=False)
        def root() -> JSONResponse:
            return JSONResponse(health())

    return app


app = create_app()
