"""
AgentEval CX Wind Tunnel — main FastAPI backend.

Endpoints:
  GET  /                 health check
  POST /run-test         V1 operational stress test (SSE)
  POST /run-rai-test     V2 RAI safety red-team test (SSE)

Bring Your Own Key: the user's Anthropic API key is passed per-request and never stored.
"""

import json
import traceback
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from anthropic import Anthropic

from .models import TestConfig, RAITestConfig, ConversationResult, RAIConversationResult, TestReport, RAITestReport
from .persona_generator import generate_personas
from .conversation_runner import run_conversation
from .judge import judge_conversation
from .rai_probe_generator import generate_rai_probes
from .rai_conversation_runner import run_rai_conversation
from .rai_judge import judge_rai_conversation

app = FastAPI(title="AgentEval — CX Wind Tunnel v2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/")
def root():
    return {"name": "AgentEval CX Wind Tunnel", "version": "2.0", "status": "ready"}


# ─── V1: Operational Stress Test ───────────────────────────────────────────

@app.post("/run-test")
def run_test(config: TestConfig):
    """V1 — Operational stress test. Streams SSE."""

    def event_stream():
        try:
            client = Anthropic(api_key=config.api_key.strip())
            yield _sse("status", {"message": f"Generating {config.num_personas} synthetic customers for '{config.domain}'..."})

            personas = generate_personas(client, config.domain, config.num_personas)
            yield _sse("personas", {"personas": [p.model_dump() for p in personas]})
            yield _sse("status", {"message": f"{len(personas)} personas ready. Running conversations..."})

            conversations: list[ConversationResult] = []
            total_failures = 0
            unexpected_failures = 0
            passed = 0

            for idx, persona in enumerate(personas):
                yield _sse("status", {
                    "message": f"[{idx+1}/{len(personas)}] Testing: {persona.name}",
                    "current": idx + 1,
                    "total": len(personas),
                })
                transcript = run_conversation(client, persona, config.agent, config.turns_per_conversation)
                finding = judge_conversation(client, persona, transcript)

                if finding.failed:
                    total_failures += 1
                    if finding.was_unexpected:
                        unexpected_failures += 1
                else:
                    passed += 1

                result = ConversationResult(persona=persona, transcript=transcript, finding=finding)
                conversations.append(result)
                yield _sse("conversation", {"result": result.model_dump()})

            report = TestReport(
                domain=config.domain,
                total_personas=len(personas),
                total_failures=total_failures,
                unexpected_failures=unexpected_failures,
                passed=passed,
                findings=[c.finding for c in conversations],
                conversations=conversations,
            )
            yield _sse("report", {"report": report.model_dump()})
            yield _sse("status", {"message": "Test complete.", "done": True})

        except Exception as e:
            yield _sse("error", {"message": str(e), "detail": traceback.format_exc()[-500:]})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ─── V2: RAI Safety Red-Team Test ──────────────────────────────────────────

@app.post("/run-rai-test")
def run_rai_test(config: RAITestConfig):
    """V2 — RAI Safety Red-Team test. Streams SSE."""

    def event_stream():
        try:
            client = Anthropic(api_key=config.api_key.strip())
            cats = ", ".join(config.categories)
            yield _sse("status", {"message": f"Generating {config.num_probes} safety probes for: {cats}..."})

            probes = generate_rai_probes(client, config.domain, config.categories, config.num_probes)
            yield _sse("probes", {"probes": [p.model_dump() for p in probes]})
            yield _sse("status", {"message": f"{len(probes)} probes ready. Launching attacks against your agent..."})

            conversations: list[RAIConversationResult] = []
            total_failures = 0
            passed = 0

            for idx, probe in enumerate(probes):
                yield _sse("status", {
                    "message": f"[{idx+1}/{len(probes)}] [{probe.category.upper()}] {probe.name}",
                    "current": idx + 1,
                    "total": len(probes),
                })
                transcript = run_rai_conversation(client, probe, config.agent, config.turns_per_conversation)
                finding = judge_rai_conversation(client, probe, transcript)

                if finding.failed:
                    total_failures += 1
                else:
                    passed += 1

                result = RAIConversationResult(probe=probe, transcript=transcript, finding=finding)
                conversations.append(result)
                yield _sse("rai_conversation", {"result": result.model_dump()})

            report = RAITestReport(
                domain=config.domain,
                total_probes=len(probes),
                total_failures=total_failures,
                passed=passed,
                categories_tested=config.categories,
                findings=[c.finding for c in conversations],
                conversations=conversations,
            )
            yield _sse("rai_report", {"report": report.model_dump()})
            yield _sse("status", {"message": "RAI test complete.", "done": True})

        except Exception as e:
            yield _sse("error", {"message": str(e), "detail": traceback.format_exc()[-500:]})

    return StreamingResponse(event_stream(), media_type="text/event-stream")

