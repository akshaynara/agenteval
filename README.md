# AgentEval — CX Wind Tunnel

The synthetic customer simulation engine for testing Agentic AI before production.

Drop in your agent → AgentEval generates dozens of adversarial synthetic customers → runs them against your agent → surfaces the failures you never thought to test for.

Built by Akshay Nara. Companion to the [AgentEval Handbook](https://agenteval.gitbook.io/agenteval-docs/).

---

## What's in this project

```
agenteval/
├── backend/              FastAPI backend (the engine)
│   ├── app/
│   │   ├── main.py               API + SSE streaming orchestration
│   │   ├── models.py             Pydantic data models
│   │   ├── persona_generator.py  Claude generates adversarial personas
│   │   ├── conversation_runner.py Personas talk to your agent
│   │   └── judge.py              Claude judges each conversation for failures
│   └── requirements.txt
├── frontend/             Single-page UI (no build step)
│   ├── index.html
│   └── app.js
├── dummy_agent/          A test-subject agent with intentional bugs
│   └── agent.py
└── README.md
```

---

## How it works

1. **You provide your agent** — either a live endpoint (`POST /chat`) or just its system prompt (demo mode, Claude simulates it).
2. **Claude generates personas** — adversarial synthetic customers across 4 risk tiers (baseline → edge_case → high_risk → critical), tailored to your domain.
3. **Each persona talks to your agent** — real multi-turn conversations.
4. **Claude judges each conversation** — did the agent fail? Which failure type? Was it a failure you'd likely never have tested for?
5. **You get a report** — failures ranked by severity, with the headline metric: *how many failures you didn't know about.*

### Bring Your Own Key (BYOK)
The user enters their own Anthropic API key in the UI. It's used only for that test run and never stored. This keeps your hosting cost at zero — every user pays for their own API usage.

A full test run (8–50 personas) costs roughly **$1–2** in Anthropic API usage. The engine uses Haiku for persona generation + conversations (cheap, high volume) and Sonnet for judging (quality where it matters).

---

## Running locally

### 1. Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Backend is now at `http://localhost:8000`.

### 2. Frontend

The frontend is a static file. Just open it, but it needs to reach the backend.

Edit `frontend/app.js` line 4 if your backend isn't on localhost:8000:
```js
const BACKEND_URL = "http://localhost:8000";
```

Then serve the frontend (any static server works):
```bash
cd frontend
python3 -m http.server 3000
```

Open `http://localhost:3000`.

### 3. (Optional) Run the dummy test agent

To test the Wind Tunnel against a real endpoint with known bugs:

```bash
cd dummy_agent
pip install fastapi uvicorn
uvicorn agent:app --reload --port 9000
```

Then in the UI, use **Live Endpoint** mode with URL `http://localhost:9000/chat`.

The dummy agent has 4 intentional bugs (fraud misrouting, PII leak, context drift, no escalation) — the Wind Tunnel should find them.

---

## Deploying for real (so others can use it)

### Backend → Render (free tier)

1. Push this repo to GitHub.
2. On [render.com](https://render.com): New → Web Service → connect your repo.
3. Settings:
   - Root directory: `backend`
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Deploy. You get a URL like `https://agenteval-api.onrender.com`.

### Frontend → Netlify / Vercel / GitHub Pages

1. Update `frontend/app.js` → set `BACKEND_URL` to your Render URL.
2. Deploy the `frontend/` folder as a static site.

### Lock down CORS

In `backend/app/main.py`, change `allow_origins=["*"]` to your frontend's domain once deployed.

---

## Linking from the Handbook

Once deployed, add a "Try the Tool" button in your GitBook handbook pointing to the frontend URL. The handbook teaches the concepts; the tool lets people act on them.

---

## Roadmap

- [x] CX Wind Tunnel v1 — persona generation, conversation running, failure judging
- [ ] Persona library — save & reuse persona sets
- [ ] Regression mode — re-run the same personas after an agent change
- [ ] Export reports (PDF/JSON)
- [ ] CI integration — run the Wind Tunnel in your pipeline
