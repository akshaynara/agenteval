"""Rendering and persistence for run reports: terminal, Markdown (CI summaries), JSON. HTML lives in html_report."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .models import SEVERITY_RANK, RunReport, Turn

SEVERITY_STYLE = {"critical": "bold white on red", "high": "bold red", "medium": "yellow", "low": "cyan", "pass": "green"}


@dataclass
class Row:
    suite: str
    case_id: str
    case: str
    severity: str
    kind: str
    title: str
    explanation: str
    unexpected: bool
    transcripts: list[tuple[str, list[Turn]]]  # (label, transcript); language findings have one per failing variant


def failure_rows(report: RunReport) -> list[Row]:
    rows: list[Row] = []
    if report.wind_tunnel:
        for c in report.wind_tunnel.conversations:
            f = c.finding
            if f.failed:
                rows.append(Row("wind tunnel", f.persona_id, f.persona_name, f.severity or "medium",
                                f.failure_type or "-", f.title, f.explanation, f.was_unexpected,
                                [("", c.transcript)]))
    if report.red_team:
        for c in report.red_team.conversations:
            f = c.finding
            if f.failed:
                rows.append(Row("red team", f.probe_id, f.probe_name, f.severity, f.category,
                                f.title, f.explanation, False, [("", c.transcript)]))
    if report.language:
        for sc in report.language.scenarios:
            f = sc.finding
            if f.failed:
                kind = ", ".join(f.failing_variants) or ("bias" if f.bias_detected else "inconsistent")
                shown = [r for r in sc.results if r.variant.kind == "control" or not r.handled_correctly]
                rows.append(Row("language", f.scenario_id, f.scenario_name, f.severity, kind,
                                f.title, f"{f.explanation} (consistency {f.consistency_score}/100)", f.bias_detected,
                                [(r.variant.label, r.transcript) for r in shown]))
    rows.sort(key=lambda r: (SEVERITY_RANK.get(r.severity, 9), r.suite, r.case_id))
    return rows


def save_json(report: RunReport, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return p


def load_json(path: str | Path) -> RunReport:
    return RunReport.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def print_report(report: RunReport, console: Console | None = None, transcripts: bool = False) -> None:
    console = console or Console()
    s = report.summary
    head = Text()
    head.append(f"{report.name}\n", style="bold")
    head.append(f"{report.domain}  ·  target: {report.target}  ·  {report.provider}  ·  {report.duration_seconds:.0f}s")
    console.print(Panel(head, title="AgentEval report", border_style="magenta"))

    stats = Table.grid(padding=(0, 3))
    stats.add_row(
        Text(f"{s.total_failures} failed", style="bold red" if s.total_failures else "bold green"),
        Text(f"{s.passed} passed", style="green"),
        Text(f"{s.total_cases} cases"),
        Text(f"{s.unexpected_failures} you likely never tested for", style="bold magenta") if report.wind_tunnel else Text(""),
    )
    console.print(stats)
    sev = "  ".join(
        f"[{SEVERITY_STYLE[k]}]{k}[/] {s.severity_counts.get(k, 0)}" for k in ("critical", "high", "medium", "low")
    )
    console.print(f"Failures by severity:  {sev}")
    if report.language:
        console.print(f"Language: average consistency [bold]{report.language.average_consistency:.0f}/100[/] across "
                      f"{report.language.total_variants} variants ({', '.join(report.language.profiles_tested)})")
    console.print()
    if report.comparison:
        print_comparison(report, console)

    rows = failure_rows(report)
    if rows:
        t = Table(title="Failures, worst first", show_lines=False, expand=True)
        t.add_column("Severity", no_wrap=True)
        t.add_column("Suite", no_wrap=True)
        t.add_column("Case")
        t.add_column("Type")
        t.add_column("What happened", ratio=3)
        for r in rows:
            flag = " [magenta](bias)[/]" if r.suite == "language" else " [magenta](unexpected)[/]"
            title = r.title + (flag if r.unexpected else "")
            t.add_row(Text(r.severity, style=SEVERITY_STYLE.get(r.severity, "")), r.suite, r.case, r.kind,
                      f"[bold]{title}[/]\n[dim]{r.explanation}[/]")
        console.print(t)
        if transcripts:
            for r in rows:
                console.rule(f"[{SEVERITY_STYLE.get(r.severity, '')}]{r.severity}[/] {r.case} — {r.title}")
                for label, transcript in r.transcripts:
                    if label:
                        console.print(f"[bold]{label}[/]")
                    for turn in transcript:
                        who = "[cyan]customer[/]" if turn.role == "user" else "[yellow]agent[/]   "
                        console.print(f"{who}  {turn.content}")
                        if turn.meta and turn.role == "assistant":
                            console.print(f"[dim]          platform: {turn.meta}[/]")
                    console.print()
    else:
        console.print("[green]No failures found.[/]")

    g = report.gate
    if g.fail_on or not g.passed:
        style = "bold green" if g.passed else "bold red"
        console.print(f"\n[{style}]Gate {'PASSED' if g.passed else 'FAILED'}[/] (fail_on: {g.fail_on}) — {g.reason}")


def to_markdown(report: RunReport) -> str:
    """A compact Markdown summary, used for GitHub Actions job summaries and PR comments."""
    s = report.summary
    gate = report.gate
    status = "✅ passed" if gate.passed else "❌ failed"
    lines = [
        f"## AgentEval — {report.name}",
        "",
        f"**{report.domain}** · target `{report.target}` · {report.duration_seconds:.0f}s",
        "",
        "| Failed | Passed | Cases | Critical | High | Medium | Low |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {s.total_failures} | {s.passed} | {s.total_cases} | {s.severity_counts.get('critical', 0)} | "
        f"{s.severity_counts.get('high', 0)} | {s.severity_counts.get('medium', 0)} | {s.severity_counts.get('low', 0)} |",
        "",
    ]
    if report.language:
        lines += [f"Language: average consistency **{report.language.average_consistency:.0f}/100** across "
                  f"{report.language.total_variants} variants", ""]
    if report.comparison:
        c = report.comparison
        lines += [f"**vs baseline** ({c.baseline_started_at}): ✅ {c.fixed} fixed · ❌ {c.regressed} regressed · "
                  f"{c.still_failing} still failing · {c.still_passing} still passing", ""]
        changed = [x for x in c.cases if x.status in ("fixed", "regressed")]
        if changed:
            lines += ["| Change | Suite | Case | Before → After |", "|---|---|---|---|"]
            for x in changed:
                lines.append(f"| {'✅ fixed' if x.status == 'fixed' else '❌ regressed'} | {x.suite} | "
                             f"{x.case.replace('|', '/')} | {x.before} → {x.after} |")
            lines.append("")
    if gate.fail_on or not gate.passed:
        lines += [f"**Gate {status}** (fail_on: `{gate.fail_on or 'none'}`) — {gate.reason}", ""]
    rows = failure_rows(report)
    if rows:
        lines += ["| Severity | Suite | Case | Type | Finding |", "|---|---|---|---|---|"]
        for r in rows:
            flag = " _(bias)_" if r.suite == "language" else " _(unexpected)_"
            finding = f"**{r.title}**{flag if r.unexpected else ''}<br>{r.explanation}".replace("|", "\\|")
            lines.append(f"| {r.severity} | {r.suite} | {r.case.replace('|', '/')} | {r.kind} | {finding} |")
    return "\n".join(lines) + "\n"


STATUS_STYLE = {"fixed": "bold green", "regressed": "bold red", "still_failing": "yellow", "still_passing": "dim",
                "new": "cyan", "removed": "dim"}


def print_comparison(report: RunReport, console: Console) -> None:
    c = report.comparison
    if not c:
        return
    console.print(Panel(
        f"[bold green]{c.fixed} fixed[/]   [bold red]{c.regressed} regressed[/]   "
        f"[yellow]{c.still_failing} still failing[/]   [dim]{c.still_passing} still passing[/]",
        title=f"vs baseline · {c.baseline_started_at}", border_style="cyan"))
    changed = [x for x in c.cases if x.status not in ("still_passing",)]
    if changed:
        t = Table(expand=True)
        for col in ("Change", "Suite", "Case", "Before", "After", "Finding"):
            t.add_column(col, ratio=3 if col == "Finding" else None)
        for x in changed:
            t.add_row(Text(x.status.replace("_", " "), style=STATUS_STYLE[x.status]), x.suite, x.case,
                      Text(x.before, style=SEVERITY_STYLE.get(x.before, "")),
                      Text(x.after, style=SEVERITY_STYLE.get(x.after, "")), x.title)
        console.print(t)
    console.print()
