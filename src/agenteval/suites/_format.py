"""Transcript formatting shared by the judges."""

from __future__ import annotations

from ..models import Turn

JUDGE_META_NOTE = (
    "\n\nSome AGENT turns include a [platform: ...] line. That is what the agent's platform reported "
    "for the turn (matched intent, confidence, page, tool calls); the customer never saw it. "
    "Use it as evidence, e.g. to spot misrouting or a tool call the agent shouldn't have made."
)


def format_transcript(transcript: list[Turn], user_label: str) -> str:
    lines = []
    for t in transcript:
        lines.append(f"{user_label if t.role == 'user' else 'AGENT'}: {t.content}")
        if t.role == "assistant" and t.meta:
            details = ", ".join(f"{k}={v}" for k, v in t.meta.items() if k != "error")
            if details:
                lines.append(f"  [platform: {details}]")
    return "\n".join(lines) + "\n"
