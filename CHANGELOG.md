# Changelog

## 1.0.0 — 2026-11-03

First release as an installable package (`pip install "agenteval-cx[google]"`; the command is `agenteval`).

### Suites
- **Wind Tunnel** — adversarial synthetic customers across four risk tiers; flags failures you likely never wrote a test for.
- **Red Team** — seven safety categories: jailbreak, harmful content, bias, PII leakage, scope violation, prompt injection, manipulation compliance.
- **Accent & Language** — every customer need in plain English, accents, code-switching (Hinglish, Spanglish, Tanglish, Taglish) and seeded speech-to-text noise, with a consistency score and a bias flag.

### Providers
- Anthropic and Gemini (Gemini API, free tier included, or Vertex AI).
- Cross-model judging: one model family attacks, another grades.

### Connectors
- Any JSON-over-HTTP agent (with optional per-conversation session ids), any OpenAI-compatible endpoint, Google CX Agent Studio (with tool fakes), Dialogflow CX, or a system prompt alone.
- The judge sees platform metadata — matched intent and confidence, page, tool calls — that the simulated customer never sees.

### Workflow
- CLI: `init`, `run`, `show`, `compare`, `serve`, `demo-agent`.
- YAML config with `${ENV}` secrets; CI gate on severity (`fail_on`) and on regressions.
- Regression mode: `--baseline` replays a saved run's exact test cases and reports fixed / regressed / still failing.
- Reports: terminal, JSON, Markdown (GitHub job summaries) and a self-contained HTML report.
- GitHub Action: `uses: akshaynara/agenteval@v1`.
- Web UI with all three suites; Docker image.

## 0.x — 2026

V1 (CX Wind Tunnel) and V2 (RAI red-teaming) as a hosted web app at try-agenteval.netlify.app, from the EXL "Ideas to Impact" concept.
