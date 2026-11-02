"""A single self-contained HTML file for a run: share it, attach it to a PR, or open it on stage."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

from .models import RunReport, Turn
from .report import failure_rows

SEV_ORDER = ("critical", "high", "medium", "low")

CSS = """
:root{--bg:#f7f7f9;--card:#fff;--text:#16161d;--muted:#5d5d6b;--line:#e4e4ea;--soft:#f0f0f4;
--crit:#b42318;--crit-bg:#fde8e7;--high:#b54708;--high-bg:#fef0e1;--med:#8a6d00;--med-bg:#fdf6d8;
--low:#3b5bdb;--low-bg:#e8edfd;--pass:#1a7f4b;--pass-bg:#e3f5ea;--accent:#7c3aed;
--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;--sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
@media (prefers-color-scheme:dark){:root{--bg:#0e0e13;--card:#17171f;--text:#ececf1;--muted:#a0a0b0;--line:#2a2a36;
--soft:#1f1f29;--crit:#ff8a80;--crit-bg:#3d1714;--high:#ffb366;--high-bg:#3a250f;--med:#f2d36b;--med-bg:#332b0d;
--low:#9db2ff;--low-bg:#1b2242;--pass:#6ee7a8;--pass-bg:#11301f;--accent:#b794f6}}
a{color:var(--accent)}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 var(--sans)}
main{max-width:1040px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:18px;margin:32px 0 12px}
.sub{color:var(--muted);font-size:14px}.brand{font:600 12px var(--mono);letter-spacing:.08em;color:var(--accent);margin-bottom:10px}
.gate{display:inline-block;margin-top:14px;padding:6px 12px;border-radius:8px;font-weight:600;font-size:14px}
.gate.ok{background:var(--pass-bg);color:var(--pass)}.gate.bad{background:var(--crit-bg);color:var(--crit)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:22px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.tile b{display:block;font:700 28px var(--mono)}.tile span{color:var(--muted);font-size:13px}
.bar{display:flex;height:10px;border-radius:6px;overflow:hidden;background:var(--soft);margin-top:14px}
.bar i{display:block}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:13px;color:var(--muted);margin-top:8px}
.sev{font:600 11px var(--mono);text-transform:uppercase;letter-spacing:.04em;padding:3px 8px;border-radius:6px;white-space:nowrap}
.sev.critical{background:var(--crit-bg);color:var(--crit)}.sev.high{background:var(--high-bg);color:var(--high)}
.sev.medium{background:var(--med-bg);color:var(--med)}.sev.low{background:var(--low-bg);color:var(--low)}
.sev.pass{background:var(--pass-bg);color:var(--pass)}
.filters{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px}
.filters button{font:500 13px var(--sans);padding:6px 12px;border-radius:999px;border:1px solid var(--line);
background:var(--card);color:var(--text);cursor:pointer}.filters button.on{border-color:var(--accent);color:var(--accent)}
details.f{background:var(--card);border:1px solid var(--line);border-radius:12px;margin-bottom:10px}
details.f>summary{list-style:none;cursor:pointer;padding:14px 16px;display:flex;gap:12px;align-items:flex-start}
details.f>summary::-webkit-details-marker{display:none}
.f .what{flex:1;min-width:0}.f .title{font-weight:600}.f .meta{color:var(--muted);font-size:13px;margin-top:2px}
.f .why{margin-top:6px;font-size:14px}.tag{font:500 11px var(--mono);color:var(--accent);margin-left:6px}
.body{padding:0 16px 16px;border-top:1px solid var(--line)}
.variant{font-weight:600;margin:14px 0 6px;font-size:14px}
.turn{display:grid;grid-template-columns:86px 1fr;gap:10px;padding:6px 0;font-size:14px}
.turn .who{font:600 11px var(--mono);text-transform:uppercase;color:var(--muted);padding-top:3px}
.turn.agent .who{color:var(--accent)}.turn .txt{white-space:pre-wrap;word-break:break-word}
.platform{grid-column:2;font:12px var(--mono);color:var(--muted);background:var(--soft);padding:4px 8px;border-radius:6px}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden;font-size:14px}
th,td{text-align:left;padding:9px 12px;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--muted);font-weight:500;font-size:13px}
tr:last-child td{border-bottom:none}.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
code{word-break:break-all}.fixed{color:var(--pass);font-weight:600}.regressed{color:var(--crit);font-weight:600}
.passed-list{columns:2 280px;font-size:14px;color:var(--muted)}.passed-list div{break-inside:avoid;padding:2px 0}
footer{margin-top:40px;color:var(--muted);font-size:13px}
@media print{.filters{display:none}details.f{break-inside:avoid}details.f .body{display:block}}
@media (max-width:600px){.turn{grid-template-columns:1fr}.platform{grid-column:1}}
"""

JS = """
const btns=document.querySelectorAll('.filters button');
btns.forEach(b=>b.addEventListener('click',()=>{btns.forEach(x=>x.classList.remove('on'));b.classList.add('on');
const f=b.dataset.f;document.querySelectorAll('details.f').forEach(d=>{
d.style.display=(f==='all'||d.dataset.suite===f||d.dataset.sev===f)?'':'none';});}));
window.addEventListener('beforeprint',()=>document.querySelectorAll('details.f').forEach(d=>d.open=true));
"""


def _e(s: object) -> str:
    return escape(str(s), quote=True)


def _transcript(turns: list[Turn]) -> str:
    out = []
    for t in turns:
        cls, who = ("user", "Customer") if t.role == "user" else ("agent", "Agent")
        out.append(f'<div class="turn {cls}"><div class="who">{who}</div><div class="txt">{_e(t.content)}</div>')
        if t.meta and t.role == "assistant":
            meta = ", ".join(f"{k}={v}" for k, v in t.meta.items() if k != "error")
            if meta:
                out.append(f'<div class="platform">platform: {_e(meta)}</div>')
        out.append("</div>")
    return "".join(out)


def _passed_cases(report: RunReport) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if report.wind_tunnel:
        out += [("wind tunnel", f.persona_name) for f in report.wind_tunnel.findings if not f.failed]
    if report.red_team:
        out += [("red team", f.probe_name) for f in report.red_team.findings if not f.failed]
    if report.language:
        out += [("language", f.scenario_name) for f in report.language.findings if not f.failed]
    return out


def render_html(report: RunReport) -> str:
    s = report.summary
    rows = failure_rows(report)
    gate = report.gate
    total_fail = max(1, sum(s.severity_counts.get(k, 0) for k in SEV_ORDER))
    colors = {"critical": "var(--crit)", "high": "var(--high)", "medium": "var(--med)", "low": "var(--low)"}

    parts: list[str] = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>AgentEval report · {_e(report.name)}</title><style>{CSS}</style></head><body><main>",
        "<div class='brand'>AGENTEVAL REPORT</div>",
        f"<h1>{_e(report.name)}</h1>",
        f"<div class='sub'>{_e(report.domain)} · target <code>{_e(report.target)}</code> · {_e(report.provider)} · "
        f"{_e(report.started_at.replace('T', ' ')[:16])} UTC · {report.duration_seconds:.0f}s</div>",
    ]
    if gate.fail_on or not gate.passed:
        parts.append(f"<div class='gate {'ok' if gate.passed else 'bad'}'>Gate {'passed' if gate.passed else 'failed'}"
                     f" · {_e(gate.reason)}</div>")

    tiles = [
        (s.total_failures, "failures"),
        (s.passed, "passed"),
        (s.total_cases, "test cases"),
    ]
    if report.wind_tunnel:
        tiles.append((s.unexpected_failures, "failures you likely never tested for"))
    if report.language:
        tiles.append((f"{report.language.average_consistency:.0f}", "language consistency / 100"))
    parts.append("<div class='tiles'>" + "".join(f"<div class='tile'><b>{_e(v)}</b><span>{_e(k)}</span></div>"
                                                  for v, k in tiles) + "</div>")
    if s.total_failures:
        parts.append("<div class='bar'>" + "".join(
            f"<i style='width:{100 * s.severity_counts.get(k, 0) / total_fail:.1f}%;background:{colors[k]}'></i>"
            for k in SEV_ORDER) + "</div>")
        parts.append("<div class='legend'>" + "".join(
            f"<span><span class='sev {k}'>{k}</span> {s.severity_counts.get(k, 0)}</span>" for k in SEV_ORDER) + "</div>")

    if report.comparison:
        c = report.comparison
        parts.append(f"<h2>Compared with the baseline ({_e(c.baseline_started_at.replace('T', ' ')[:16])} UTC)</h2>")
        parts.append(f"<div class='tiles'><div class='tile'><b class='fixed'>{c.fixed}</b><span>fixed</span></div>"
                     f"<div class='tile'><b class='regressed'>{c.regressed}</b><span>regressed</span></div>"
                     f"<div class='tile'><b>{c.still_failing}</b><span>still failing</span></div>"
                     f"<div class='tile'><b>{c.still_passing}</b><span>still passing</span></div></div>")
        changed = [x for x in c.cases if x.status != "still_passing"]
        if changed:
            parts.append("<div class='scroll'><table style='margin-top:12px'><tr><th>Change</th><th>Suite</th><th>Case</th>"
                         "<th>Before</th><th>After</th><th>Finding</th></tr>")
            for x in changed:
                parts.append(
                    f"<tr><td class='{_e(x.status)}'>{_e(x.status.replace('_', ' '))}</td><td>{_e(x.suite)}</td>"
                    f"<td>{_e(x.case)}</td><td><span class='sev {_e(x.before)}'>{_e(x.before)}</span></td>"
                    f"<td><span class='sev {_e(x.after)}'>{_e(x.after)}</span></td><td>{_e(x.title)}</td></tr>")
            parts.append("</table></div>")

    parts.append("<h2>Failures, worst first</h2>")
    if rows:
        suites = sorted({r.suite for r in rows})
        sevs = [k for k in SEV_ORDER if any(r.severity == k for r in rows)]
        parts.append("<div class='filters'><button class='on' data-f='all'>All</button>"
                     + "".join(f"<button data-f='{_e(x)}'>{_e(x)}</button>" for x in suites + sevs) + "</div>")
        for r in rows:
            tag = ("bias" if r.suite == "language" else "you likely never tested for") if r.unexpected else ""
            body = "".join(
                (f"<div class='variant'>{_e(label)}</div>" if label else "") + _transcript(tr)
                for label, tr in r.transcripts
            )
            parts.append(
                f"<details class='f' data-suite='{_e(r.suite)}' data-sev='{_e(r.severity)}'><summary>"
                f"<span class='sev {_e(r.severity)}'>{_e(r.severity)}</span><div class='what'>"
                f"<div class='title'>{_e(r.title)}{f'<span class=tag>{_e(tag)}</span>' if tag else ''}</div>"
                f"<div class='meta'>{_e(r.suite)} · {_e(r.case)} · {_e(r.kind)}</div>"
                f"<div class='why'>{_e(r.explanation)}</div></div></summary><div class='body'>{body}</div></details>")
    else:
        parts.append("<p>No failures found.</p>")

    passed = _passed_cases(report)
    if passed:
        parts.append(f"<h2>Passed ({len(passed)})</h2><div class='passed-list'>"
                     + "".join(f"<div>✓ {_e(name)} <span class='sub'>· {_e(suite)}</span></div>" for suite, name in passed)
                     + "</div>")

    models = ", ".join(f"{k}: {v}" for k, v in report.models.items())
    parts.append(f"<footer>Generated by AgentEval {_e(report.agenteval_version)} · models — {_e(models)} · "
                 f"<a href='https://github.com/akshaynara/agenteval'>github.com/akshaynara/agenteval</a></footer>")
    # Embed the raw report so the HTML file alone is enough to reload or diff later.
    raw = json.dumps(report.model_dump(mode="json")).replace("</", "<\\/")
    parts.append(f"<script type='application/json' id='agenteval-report'>{raw}</script>")
    parts.append(f"<script>{JS}</script></main></body></html>")
    return "".join(parts)


def save_html(report: RunReport, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_html(report), encoding="utf-8")
    return p
