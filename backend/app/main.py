"""
AgentEval CX Wind Tunnel — main FastAPI backend.

Endpoints:
  GET  /                 health check
  POST /run-test         run a full stress test, stream progress (SSE)

Bring Your Own Key: the user's Anthropic API key is passed per-request and never
stored. It's used only for that test run.
"""

import json
import traceback
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from anthropic import Anthropic

from .models import TestConfig, ConversationResult, TestReport
from .persona_generator import generate_personas
from .conversation_runner import run_conversation
from .judge import judge_conversation

app = FastAPI(title="AgentEval — CX Wind Tunnel")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your GitBook/site domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)


def _sse(event: str, data: dict) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/")
def root():
    return {"name": "AgentEval CX Wind Tunnel", "status": "ready"}


@app.post("/run-test")
def run_test(config: TestConfig):
    """
    Streams the full stress test as Server-Sent Events so the frontend can show
    live progress. Events:
      status      — human-readable progress messages
      personas    — the generated personas
      conversation— one completed conversation + finding
      report      — final aggregated report
      error       — something went wrong
    """

    def event_stream():
        try:
            client = Anthropic(api_key=config.api_key)

            yield _sse("status", {"message": f"Generating {config.num_personas} synthetic customers for '{config.domain}'..."})

            personas = generate_personas(client, config.domain, config.num_personas)
            yield _sse("personas", {"personas": [p.model_dump() for p in personas]})
            yield _sse("status", {"message": f"{len(personas)} personas ready. Running conversations against your agent..."})

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

                transcript = run_conversation(
                    client, persona, config.agent, config.turns_per_conversation
                )
                finding = judge_conversation(client, persona, transcript)

                if finding.failed:
                    total_failures += 1
                    if finding.was_unexpected:
                        unexpected_failures += 1
                else:
                    passed += 1

                result = ConversationResult(
                    persona=persona, transcript=transcript, finding=finding
                )
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
            yield _sse("error", {
                "message": str(e),
                "detail": traceback.format_exc()[-500:],
            })

    return StreamingResponse(event_stream(), media_type="text/event-stream")
