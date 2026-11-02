"""The orchestration engine shared by the CLI and the web server.

Progress is reported through an optional ``on_event(name, data)`` callback. Event names
match the web UI's server-sent events: ``status``, ``personas``, ``conversation``,
``report`` (Wind Tunnel) and ``probes``, ``rai_conversation``, ``rai_report`` (red team).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, TypeVar

from . import __version__
from .llm import LLM
from .models import (
    SEVERITY_RANK,
    ConversationResult,
    GateResult,
    RAIConversationResult,
    RAITestReport,
    RunReport,
    RunSummary,
    TestReport,
)
from .suites import red_team, wind_tunnel
from .targets import Target, check_reachable

EventHandler = Callable[[str, dict[str, Any]], None]
T = TypeVar("T")
R = TypeVar("R")


def _noop(_name: str, _data: dict[str, Any]) -> None:
    pass


def _run_cases(
    cases: Sequence[T],
    work: Callable[[T], R],
    concurrency: int,
    on_start: Callable[[int, T], None],
    on_done: Callable[[R], None],
) -> list[R]:
    """Run ``work`` over ``cases`` with up to ``concurrency`` threads, preserving input order."""
    results: list[R | None] = [None] * len(cases)
    lock = threading.Lock()

    def wrapped(idx: int, case: T) -> tuple[int, R]:
        with lock:
            on_start(idx, case)
        return idx, work(case)

    if concurrency <= 1:
        for idx, case in enumerate(cases):
            _, res = wrapped(idx, case)
            results[idx] = res
            on_done(res)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(wrapped, i, c) for i, c in enumerate(cases)]
            for fut in as_completed(futures):
                idx, res = fut.result()
                results[idx] = res
                with lock:
                    on_done(res)
    return [r for r in results if r is not None]


def run_wind_tunnel(
    llm: LLM,
    target: Target,
    domain: str,
    num_personas: int = 10,
    turns: int = 4,
    concurrency: int = 1,
    on_event: EventHandler | None = None,
    check_target: bool = True,
) -> TestReport:
    emit = on_event or _noop
    if check_target:
        emit("status", {"message": "Checking that your agent answers..."})
        check_reachable(target)
    emit("status", {"message": f"Generating {num_personas} synthetic customers for '{domain}'..."})
    personas = wind_tunnel.generate_personas(llm, domain, num_personas)
    emit("personas", {"personas": [p.model_dump() for p in personas]})
    emit("status", {"message": f"{len(personas)} personas ready. Running conversations..."})

    total = len(personas)
    done = 0

    def on_start(idx: int, persona) -> None:
        emit("status", {"message": f"[{idx + 1}/{total}] Testing: {persona.name}", "current": idx + 1, "total": total})

    def work(persona) -> ConversationResult:
        transcript = wind_tunnel.run_conversation(llm, persona, target, turns)
        finding = wind_tunnel.judge_conversation(llm, persona, transcript)
        return ConversationResult(persona=persona, transcript=transcript, finding=finding)

    def on_done(result: ConversationResult) -> None:
        nonlocal done
        done += 1
        emit("conversation", {"result": result.model_dump(), "completed": done, "total": total})

    conversations = _run_cases(personas, work, concurrency, on_start, on_done)
    findings = [c.finding for c in conversations]
    failures = [f for f in findings if f.failed]
    report = TestReport(
        domain=domain,
        total_personas=len(personas),
        total_failures=len(failures),
        unexpected_failures=sum(1 for f in failures if f.was_unexpected),
        passed=len(findings) - len(failures),
        findings=findings,
        conversations=conversations,
    )
    emit("report", {"report": report.model_dump()})
    return report


def run_red_team(
    llm: LLM,
    target: Target,
    domain: str,
    categories: list[str],
    num_probes: int = 10,
    turns: int = 4,
    concurrency: int = 1,
    on_event: EventHandler | None = None,
    check_target: bool = True,
) -> RAITestReport:
    emit = on_event or _noop
    if check_target:
        emit("status", {"message": "Checking that your agent answers..."})
        check_reachable(target)
    emit("status", {"message": f"Generating {num_probes} safety probes for: {', '.join(categories)}..."})
    probes = red_team.generate_rai_probes(llm, domain, categories, num_probes)
    emit("probes", {"probes": [p.model_dump() for p in probes]})
    emit("status", {"message": f"{len(probes)} probes ready. Launching attacks against your agent..."})

    total = len(probes)
    done = 0

    def on_start(idx: int, probe) -> None:
        emit(
            "status",
            {"message": f"[{idx + 1}/{total}] [{probe.category.upper()}] {probe.name}", "current": idx + 1, "total": total},
        )

    def work(probe) -> RAIConversationResult:
        transcript = red_team.run_rai_conversation(llm, probe, target, turns)
        finding = red_team.judge_rai_conversation(llm, probe, transcript)
        return RAIConversationResult(probe=probe, transcript=transcript, finding=finding)

    def on_done(result: RAIConversationResult) -> None:
        nonlocal done
        done += 1
        emit("rai_conversation", {"result": result.model_dump(), "completed": done, "total": total})

    conversations = _run_cases(probes, work, concurrency, on_start, on_done)
    findings = [c.finding for c in conversations]
    failures = sum(1 for f in findings if f.failed)
    report = RAITestReport(
        domain=domain,
        total_probes=len(probes),
        total_failures=failures,
        passed=len(findings) - failures,
        categories_tested=categories,
        findings=findings,
        conversations=conversations,
    )
    emit("rai_report", {"report": report.model_dump()})
    return report


def summarize(wt: TestReport | None, rt: RAITestReport | None) -> RunSummary:
    summary = RunSummary(severity_counts={k: 0 for k in SEVERITY_RANK})
    findings: list = []
    if wt:
        findings += wt.findings
        summary.unexpected_failures = wt.unexpected_failures
    if rt:
        findings += rt.findings
    summary.total_cases = len(findings)
    summary.total_failures = sum(1 for f in findings if f.failed)
    summary.passed = summary.total_cases - summary.total_failures
    for f in findings:
        sev = f.severity if f.failed and f.severity else "pass"
        summary.severity_counts[sev] = summary.severity_counts.get(sev, 0) + 1
    return summary


def evaluate_gate(report: RunReport, fail_on: str | None) -> GateResult:
    """Fail the gate if any failure is at or above the ``fail_on`` severity (e.g. 'high' fails on high + critical)."""
    if not fail_on or fail_on == "none":
        return GateResult(fail_on=None, passed=True, reason="No gate configured.")
    if fail_on not in SEVERITY_RANK or fail_on == "pass":
        raise ValueError(f"fail_on must be one of critical, high, medium, low, none — got '{fail_on}'")
    threshold = SEVERITY_RANK[fail_on]
    hits = {
        sev: n for sev, n in report.summary.severity_counts.items() if sev != "pass" and SEVERITY_RANK[sev] <= threshold and n
    }
    if hits:
        detail = ", ".join(f"{n} {sev}" for sev, n in sorted(hits.items(), key=lambda kv: SEVERITY_RANK[kv[0]]))
        return GateResult(fail_on=fail_on, passed=False, reason=f"Found {detail} finding(s) at or above '{fail_on}'.")
    return GateResult(fail_on=fail_on, passed=True, reason=f"No findings at or above '{fail_on}'.")


def run(
    *,
    llm: LLM,
    target: Target,
    name: str,
    domain: str,
    wind_tunnel_opts: dict[str, Any] | None,
    red_team_opts: dict[str, Any] | None,
    concurrency: int = 1,
    fail_on: str | None = None,
    on_event: EventHandler | None = None,
) -> RunReport:
    """Run the enabled suites and return a complete :class:`RunReport`."""
    started = datetime.now(timezone.utc)
    t0 = time.monotonic()
    wt = rt = None
    if wind_tunnel_opts is not None:
        wt = run_wind_tunnel(
            llm,
            target,
            domain,
            num_personas=wind_tunnel_opts.get("personas", 10),
            turns=wind_tunnel_opts.get("turns", 4),
            concurrency=concurrency,
            on_event=on_event,
        )
    if red_team_opts is not None:
        rt = run_red_team(
            llm,
            target,
            domain,
            categories=red_team_opts.get("categories") or [],
            num_probes=red_team_opts.get("probes", 10),
            turns=red_team_opts.get("turns", 4),
            concurrency=concurrency,
            on_event=on_event,
            check_target=wt is None,
        )
    report = RunReport(
        agenteval_version=__version__,
        name=name,
        domain=domain,
        target=target.describe(),
        provider=llm.provider,
        models=dict(llm.models),
        started_at=started.isoformat(timespec="seconds"),
        finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        duration_seconds=round(time.monotonic() - t0, 1),
        wind_tunnel=wt,
        red_team=rt,
        summary=summarize(wt, rt),
    )
    report.gate = evaluate_gate(report, fail_on)
    return report
