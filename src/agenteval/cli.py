"""Command-line interface: `agenteval init | run | show | serve | demo-agent`."""

from __future__ import annotations

import os
from enum import Enum
from importlib import resources
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from . import __version__, engine
from .config import ConfigError, RunSpec, load_config
from .factory import build_llm, build_target
from .llm import LLMError
from .report import load_json, print_report, save_json, to_markdown
from .targets import TargetError

EXIT_OK, EXIT_GATE_FAILED, EXIT_USAGE, EXIT_ERROR = 0, 1, 2, 3

app = typer.Typer(
    name="agenteval",
    help="Red-team and stress-test conversational AI agents with synthetic customers and safety probes.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()
err = Console(stderr=True)


class Suite(str, Enum):
    all = "all"
    wind_tunnel = "wind-tunnel"
    red_team = "red-team"


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"agenteval {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: bool | None = typer.Option(None, "--version", callback=_version_callback, is_eager=True,
                                           help="Show the version and exit."),
) -> None:
    pass


@app.command()
def init(
    path: Path = typer.Argument(Path("agenteval.yaml"), help="Where to write the config."),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite an existing file."),
) -> None:
    """Create a starter [bold]agenteval.yaml[/] in this folder."""
    if path.exists() and not force:
        err.print(f"[red]{path} already exists.[/] Use --force to overwrite.")
        raise typer.Exit(EXIT_USAGE)
    template = resources.files("agenteval").joinpath("templates/agenteval.yaml").read_text(encoding="utf-8")
    path.write_text(template, encoding="utf-8")
    console.print(f"[green]Created {path}[/]\n")
    console.print("Next steps:")
    console.print("  1. export ANTHROPIC_API_KEY=sk-ant-...")
    console.print("  2. agenteval demo-agent          [dim]# in another terminal: a buggy banking agent to test[/]")
    console.print("  3. agenteval run")


class _ProgressPrinter:
    """Turns engine events into a progress bar plus one line per finished conversation."""

    def __init__(self, progress: Progress, quiet: bool):
        self.progress = progress
        self.quiet = quiet
        self.task = None

    def __call__(self, name: str, data: dict[str, Any]) -> None:
        if name == "status" and "current" not in data and not data.get("done"):
            self.progress.console.print(f"[dim]{data['message']}[/]")
        elif name in ("personas", "probes"):
            items = data[name]
            label = "Wind tunnel" if name == "personas" else "Red team"
            if self.task is not None:
                self.progress.remove_task(self.task)
            self.task = self.progress.add_task(label, total=len(items))
        elif name in ("conversation", "rai_conversation"):
            if self.task is not None:
                self.progress.advance(self.task)
            if self.quiet:
                return
            res = data["result"]
            case = res.get("persona") or res.get("probe")
            f = res["finding"]
            if f["failed"]:
                sev = f.get("severity") or "medium"
                self.progress.console.print(f"  [red]✗[/] [bold]{case['name']}[/] — [red]{sev}[/] {f['title']}")
            else:
                self.progress.console.print(f"  [green]✓[/] {case['name']}")


@app.command()
def run(
    config: Path | None = typer.Argument(None, help="Config file (default: ./agenteval.yaml)."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Write the JSON report here."),
    fail_on: str | None = typer.Option(None, "--fail-on",
                                          help="Exit 1 if any finding is at/above: critical|high|medium|low|none."),
    suite: Suite = typer.Option(Suite.all, "--suite", "-s", help="Which suite(s) to run."),
    personas: int | None = typer.Option(None, "--personas", help="Override the number of personas."),
    probes: int | None = typer.Option(None, "--probes", help="Override the number of safety probes."),
    concurrency: int | None = typer.Option(None, "--concurrency", "-c", help="Conversations in parallel."),
    transcripts: bool = typer.Option(False, "--transcripts", "-t", help="Print the transcript of every failure."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Only show the final report."),
    markdown: Path | None = typer.Option(None, "--markdown", help="Also write a Markdown summary here."),
) -> None:
    """Run the test suites against your agent."""
    try:
        cfg = load_config(config)
        if fail_on is not None:
            cfg.run.fail_on = RunSpec(fail_on=fail_on).fail_on
        if concurrency is not None:
            cfg.run.concurrency = max(1, concurrency)
        if personas is not None:
            cfg.suites.wind_tunnel.personas = personas
        if probes is not None:
            cfg.suites.red_team.probes = probes
        llm = build_llm(cfg.provider)
        target = build_target(cfg, llm)
    except (ConfigError, LLMError, ValueError) as e:
        err.print(f"[red]Error:[/] {e}")
        raise typer.Exit(EXIT_USAGE) from None

    wt = cfg.suites.wind_tunnel
    rt = cfg.suites.red_team
    wt_opts = {"personas": wt.personas, "turns": wt.turns} if wt.enabled and suite in (Suite.all, Suite.wind_tunnel) else None
    rt_opts = (
        {"probes": rt.probes, "turns": rt.turns, "categories": rt.categories}
        if rt.enabled and suite in (Suite.all, Suite.red_team)
        else None
    )
    if wt_opts is None and rt_opts is None:
        err.print("[red]Error:[/] no suites enabled. Check `suites:` in your config or the --suite option.")
        raise typer.Exit(EXIT_USAGE)

    console.print(f"[bold magenta]AgentEval[/] {__version__} · {cfg.name} · target: {target.describe()}")
    progress = Progress(TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
                        console=console, transient=True)
    try:
        with progress:
            report = engine.run(
                llm=llm,
                target=target,
                name=cfg.name,
                domain=cfg.domain,
                wind_tunnel_opts=wt_opts,
                red_team_opts=rt_opts,
                concurrency=cfg.run.concurrency,
                fail_on=cfg.run.fail_on,
                on_event=_ProgressPrinter(progress, quiet),
            )
    except TargetError as e:
        err.print(f"[red]Can't reach your agent:[/] {e}")
        raise typer.Exit(EXIT_ERROR) from None
    except (LLMError, ValueError) as e:
        err.print(f"[red]Run failed:[/] {e}")
        raise typer.Exit(EXIT_ERROR) from None
    except KeyboardInterrupt:
        err.print("[yellow]Interrupted.[/]")
        raise typer.Exit(130) from None

    console.print()
    print_report(report, console, transcripts=transcripts)

    out = output or (Path(cfg.run.output) if cfg.run.output else None)
    if out:
        save_json(report, out)
        console.print(f"\n[dim]Report saved to {out} — view it again with `agenteval show {out}`[/]")
    if markdown:
        markdown.write_text(to_markdown(report), encoding="utf-8")
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a", encoding="utf-8") as fh:
            fh.write(to_markdown(report))

    raise typer.Exit(EXIT_OK if report.gate.passed else EXIT_GATE_FAILED)


@app.command()
def show(
    report_path: Path = typer.Argument(Path("agenteval-report.json"), help="A saved JSON report."),
    transcripts: bool = typer.Option(False, "--transcripts", "-t", help="Print the transcript of every failure."),
    markdown: bool = typer.Option(False, "--markdown", help="Print as Markdown instead."),
) -> None:
    """Show a saved report."""
    try:
        report = load_json(report_path)
    except (OSError, ValueError) as e:
        err.print(f"[red]Could not read {report_path}:[/] {e}")
        raise typer.Exit(EXIT_USAGE) from None
    if markdown:
        print(to_markdown(report))
    else:
        print_report(report, console, transcripts=transcripts)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Interface to bind."),
    port: int = typer.Option(8000, help="Port."),
) -> None:
    """Start the web UI and API (the same engine, in your browser)."""
    import uvicorn

    console.print(f"[bold magenta]AgentEval[/] web UI on http://{host}:{port}")
    uvicorn.run("agenteval.server:app", host=host, port=port, log_level="info")


@app.command("demo-agent")
def demo_agent(
    host: str = typer.Option("127.0.0.1", help="Interface to bind."),
    port: int = typer.Option(9000, help="Port."),
) -> None:
    """Run a deliberately buggy banking agent to test against (POST /chat)."""
    import uvicorn

    console.print(f"Demo banking agent on http://{host}:{port}/chat  [dim](4 bugs planted on purpose)[/]")
    uvicorn.run("agenteval.demo.banking_agent:app", host=host, port=port, log_level="warning")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
