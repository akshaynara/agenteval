# AgentEval — 5-minute stage demo

**Story:** *Your agent passed QA. Here's what happens when real customers show up — and how to catch it before they do.*

Everything runs from one terminal and one browser tab. Each live step has a pre-recorded fallback, so bad venue Wi-Fi can't sink the talk.

---

## The day before

```bash
pip install "agenteval-cx[google]"
export ANTHROPIC_API_KEY=sk-ant-...        # or GEMINI_API_KEY with `provider: {name: gemini}`
mkdir agenteval-demo && cd agenteval-demo
agenteval init
```

Edit `agenteval.yaml` for speed on stage:

```yaml
suites:
  wind_tunnel: { personas: 4, turns: 3 }
  red_team:    { probes: 3, turns: 2, categories: [pii_leakage, jailbreak, manipulation_compliance] }
  language:    { enabled: false, scenarios: 2, accents: [indian_english], code_switch: [hinglish], asr: [indian_english] }
run:
  concurrency: 6
  fail_on: high
```

**Record your fallbacks.** These runs are also your rehearsal.

```bash
agenteval demo-agent &                                    # the buggy agent
agenteval run -o before.json --html before.html
agenteval run --suite language -o language.json --html language.html
kill %1 && agenteval demo-agent --fixed &                 # the fixed agent
agenteval run --baseline before.json -o after.json --html after.html
kill %1
```

Open the three HTML files once to check them. They're self-contained, so they work with no internet.

## Ten minutes before

- Terminal font at 20pt or more, dark theme, window wide enough for the report table (about 120 columns).
- `agenteval demo-agent &` running in the demo folder.
- Browser tabs: `before.html`, `language.html`, `after.html`, and the GitHub repo with a QR code to it on your last slide.
- Hosted web UI instead? Open https://agenteval-backend.onrender.com/health first, since the free tier takes about a minute to wake up.

---

## Run of show

| Time | Say | Do |
|---|---|---|
| **0:00** | "Every agent I've shipped passed QA. Then production happened. One banking agent told a customer how to space out large transfers to avoid scrutiny. No test case asked that question, because nobody thought to." | Slide: that finding. |
| **0:30** | "AgentEval writes the customers you didn't think of, runs real conversations with your agent, and has a judge tell you where it broke. This is a deliberately buggy banking agent." | `agenteval run -o live.json` |
| **0:45** | While it streams: "Each line is a whole conversation. A panicked fraud victim. Someone probing for another customer's address. A 'bank auditor' who wants the rules bent." | Point at ✗ lines as they appear. |
| **2:00** | "Two critical findings. This one confirmed a stranger's name and address because someone typed an account number. And the gate failed, so in CI this pull request can't merge." | Scroll the report; point at **Gate FAILED**. |
| **2:30** | "Now the part most teams never test: does it serve every caller equally? Same need in plain English, Indian English, Hinglish, and through a voice channel's speech-to-text." | Switch to `language.html`. |
| **2:45** | "Plain English worked. Hinglish didn't. And on voice, *'I did **not** make the fifteen dollar payment'* arrived as *'I did make the fifty dollar payment'*. Your plain-English tests would pass every one of these." | Open a finding and show the variant transcripts. Quote the real examples from *your* `language.html`. |
| **3:30** | "So we fix the agent and replay the exact same customers." | `kill %1; agenteval demo-agent --fixed &` then `agenteval run --baseline live.json` |
| **4:00** | "Fixed, fixed, and nothing regressed. That's the loop: find it, fix it, prove it stayed fixed." | Point at the **vs baseline** panel. |
| **4:20** | "It runs in CI on every pull request, works with Claude or Gemini, and connects straight to CX Agent Studio, Dialogflow CX, or any HTTP or OpenAI-compatible agent." | Slide: the 8-line GitHub Action. |
| **4:45** | "Open source and free. `pip install agenteval-cx`. Red-team your agent before your customers do." | QR code to the repo. |

### If something goes wrong

- **Slow API or bad Wi-Fi:** say "here's one I ran this morning" and open `before.html` / `after.html`. The story is identical.
- **The run takes too long:** press `Ctrl+C` and show `agenteval show before.json`.
- **Model variance:** a case can flip between runs. If a "fixed" case still fails, say so honestly. That's why the tool shows transcripts, not just scores.

---

## Cheat sheet

```bash
agenteval demo-agent [--fixed]                 # the test subject
agenteval run [-o out.json] [--html out.html]  # all enabled suites
agenteval run --suite language                 # just accent & language
agenteval run --baseline before.json           # replay + compare
agenteval show before.json [--transcripts]     # re-print a saved run
agenteval compare before.json after.json       # fixed / regressed
agenteval serve                                # the web UI on :8000
```
