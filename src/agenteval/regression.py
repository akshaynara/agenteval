"""Regression mode: replay a saved run's test cases and see what got fixed and what broke.

    agenteval run --baseline before.json     # same personas, probes and scenarios, compared with before.json
    agenteval compare before.json after.json

Opening messages are replayed exactly; the simulated customer's follow-ups are generated
fresh, so a case can occasionally flip on model variance. Re-run a flipped case before
treating it as a real change.
"""

from __future__ import annotations

from .models import SEVERITY_RANK, CaseComparison, Comparison, RunReport


def _outcomes(report: RunReport) -> dict[tuple[str, str], tuple[str, str, str]]:
    """(suite, case_id) -> (case name, severity-or-pass, title)."""
    out: dict[tuple[str, str], tuple[str, str, str]] = {}
    if report.wind_tunnel:
        for f in report.wind_tunnel.findings:
            out[("wind tunnel", f.persona_id)] = (f.persona_name, (f.severity or "medium") if f.failed else "pass", f.title)
    if report.red_team:
        for f in report.red_team.findings:
            out[("red team", f.probe_id)] = (f.probe_name, f.severity if f.failed else "pass", f.title)
    if report.language:
        for f in report.language.findings:
            out[("language", f.scenario_id)] = (f.scenario_name, f.severity if f.failed else "pass", f.title)
    return out


def compare(baseline: RunReport, current: RunReport) -> Comparison:
    before = _outcomes(baseline)
    after = _outcomes(current)
    cmp = Comparison(baseline_started_at=baseline.started_at, baseline_name=baseline.name)
    for key in [*before, *[k for k in after if k not in before]]:
        suite, case_id = key
        b = before.get(key)
        a = after.get(key)
        if b and a:
            failed_before, failed_after = b[1] != "pass", a[1] != "pass"
            status = {
                (True, False): "fixed",
                (False, True): "regressed",
                (True, True): "still_failing",
                (False, False): "still_passing",
            }[(failed_before, failed_after)]
            case = CaseComparison(suite=suite, case_id=case_id, case=a[0], status=status, before=b[1], after=a[1],
                                  title=a[2] if failed_after else b[2])
        elif a:
            case = CaseComparison(suite=suite, case_id=case_id, case=a[0], status="new", before="-", after=a[1],
                                  title=a[2])
        else:
            assert b is not None
            case = CaseComparison(suite=suite, case_id=case_id, case=b[0], status="removed", before=b[1], after="-",
                                  title=b[2])
        cmp.cases.append(case)
    for c in cmp.cases:
        if c.status in ("fixed", "regressed", "still_failing", "still_passing"):
            setattr(cmp, c.status, getattr(cmp, c.status) + 1)
    order = {"regressed": 0, "still_failing": 1, "fixed": 2, "new": 3, "removed": 4, "still_passing": 5}
    cmp.cases.sort(key=lambda c: (order[c.status], SEVERITY_RANK.get(c.after, 9), c.suite, c.case_id))
    return cmp
