# Contributing to AgentEval

Thanks for helping. Issues and pull requests are welcome, especially new connectors, providers, accent and language profiles, and failure stories from real agents.

## Set up

```bash
git clone https://github.com/akshaynara/agenteval && cd agenteval
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                 # fully offline: a scripted fake LLM drives the whole pipeline
ruff check src tests
```

No API key is needed for the tests. To try a real run: `agenteval demo-agent &` then `agenteval run examples/banking-demo/agenteval.yaml`.

## Where things live

| You want to add… | Touch |
|---|---|
| A **connector** for an agent platform | `targets.py` (a class with `describe()` and `open_session()`; the session's `respond(transcript)` returns an `AgentReply`), `config.py` (`TargetSpec`), `factory.py` (`build_target`) |
| A **model provider** | `llm/` (a class with `provider`, `models` and `complete(role, system, messages, max_tokens)`), registered in `llm/__init__.py:create_llm` |
| An **accent or language profile** | `ACCENT_PROFILES` / `CODE_SWITCH_PROFILES` / `ASR_CONFUSIONS` in `suites/language.py` — a description, a few real examples, and for ASR, real-word confusions an ASR layer actually produces |
| A **test suite** | `suites/`, then wire it into `engine.run`, the CLI, the report renderers and the server |

## Guidelines

- **Test at the wire.** Connector tests mock HTTP and assert the exact URL, headers and body (see `tests/test_targets.py`), so they catch API drift.
- **Keep prompts honest.** Judges must explain findings by citing the transcript. Don't add instructions that bias toward failing or passing.
- **Never store secrets.** Keys come from the environment or the request and are never written to reports.
- Keep the web API's existing request and event shapes stable; the hosted UI depends on them.
