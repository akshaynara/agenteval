// AgentEval CX Wind Tunnel — frontend logic

// CHANGE THIS to your deployed backend URL (e.g. https://agenteval-api.onrender.com)
const BACKEND_URL = "http://localhost:8000";

let currentMode = "endpoint";

function setMode(mode) {
  currentMode = mode;
  document.getElementById("modeEndpoint").classList.toggle("active", mode === "endpoint");
  document.getElementById("modePrompt").classList.toggle("active", mode === "prompt");
  document.getElementById("endpointFields").style.display = mode === "endpoint" ? "block" : "none";
  document.getElementById("promptFields").style.display = mode === "prompt" ? "block" : "none";
}

// Slider value displays
document.getElementById("numPersonas").addEventListener("input", (e) => {
  document.getElementById("numPersonasVal").textContent = e.target.value;
});
document.getElementById("turns").addEventListener("input", (e) => {
  document.getElementById("turnsVal").textContent = e.target.value;
});

function severityRank(sev) {
  return { critical: 0, high: 1, medium: 2, low: 3, pass: 4 }[sev] ?? 5;
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

async function runTest() {
  const apiKey = document.getElementById("apiKey").value.trim();
  const domain = document.getElementById("domain").value.trim();
  const numPersonas = parseInt(document.getElementById("numPersonas").value);
  const turns = parseInt(document.getElementById("turns").value);

  // Validation
  if (!apiKey) { showError("Please enter your Anthropic API key."); return; }
  if (!domain) { showError("Please enter a test domain."); return; }

  const agent = { mode: currentMode, reply_field: "reply" };
  if (currentMode === "endpoint") {
    agent.endpoint_url = document.getElementById("endpointUrl").value.trim();
    agent.reply_field = document.getElementById("replyField").value.trim() || "reply";
    if (!agent.endpoint_url) { showError("Please enter your agent endpoint URL, or switch to Prompt mode."); return; }
  } else {
    agent.system_prompt = document.getElementById("systemPrompt").value.trim();
    if (!agent.system_prompt) { showError("Please paste your agent's system prompt, or switch to Endpoint mode."); return; }
  }

  // Reset UI
  hideError();
  document.getElementById("report").style.display = "none";
  document.getElementById("findingsList").innerHTML = "";
  document.getElementById("progressArea").style.display = "block";
  document.getElementById("runBtn").disabled = true;
  setStatus("Connecting...");
  setProgress(0);

  const payload = {
    api_key: apiKey, domain, agent,
    num_personas: numPersonas, turns_per_conversation: turns,
  };

  try {
    const resp = await fetch(`${BACKEND_URL}/run-test`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!resp.ok) {
      throw new Error(`Server returned ${resp.status}. Is the backend running?`);
    }

    // Parse Server-Sent Events stream
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    const findings = [];
    let report = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const parts = buffer.split("\n\n");
      buffer = parts.pop(); // keep incomplete chunk

      for (const part of parts) {
        const lines = part.split("\n");
        let event = "message", data = "";
        for (const line of lines) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          if (line.startsWith("data:")) data += line.slice(5).trim();
        }
        if (!data) continue;
        const parsed = JSON.parse(data);
        handleEvent(event, parsed, findings);
        if (event === "report") report = parsed.report;
      }
    }

    if (report) renderReport(report);
  } catch (err) {
    showError(err.message || "Something went wrong.");
  } finally {
    document.getElementById("runBtn").disabled = false;
    document.querySelector(".spinner")?.style.setProperty("display", "none");
  }
}

function handleEvent(event, data, findings) {
  if (event === "status") {
    setStatus(data.message);
    if (data.current && data.total) {
      setProgress((data.current / data.total) * 100);
    }
    if (data.done) {
      document.querySelector(".spinner").style.display = "none";
      setStatus("Test complete. ✓");
      setProgress(100);
    }
  } else if (event === "personas") {
    setStatus(`${data.personas.length} synthetic customers generated. Running conversations...`);
  } else if (event === "conversation") {
    findings.push(data.result);
    appendFinding(data.result);
  } else if (event === "error") {
    showError(data.message + (data.detail ? "\n\n" + data.detail : ""));
  }
}

function appendFinding(result) {
  // Show report container as soon as first finding arrives
  document.getElementById("report").style.display = "block";
  const f = result.finding;
  const card = document.createElement("div");
  card.className = "finding-card";
  card.onclick = () => {
    const t = card.querySelector(".transcript");
    t.style.display = t.style.display === "block" ? "none" : "block";
  };

  const transcriptHtml = result.transcript.map(turn => `
    <div class="turn turn-${turn.role === "user" ? "user" : "agent"}">
      <div class="turn-role">${turn.role === "user" ? "CUSTOMER" : "AGENT"}</div>
      <div class="turn-content">${escapeHtml(turn.content)}</div>
    </div>`).join("");

  card.innerHTML = `
    <div class="finding-head">
      <span class="finding-persona">${escapeHtml(f.persona_name)}</span>
      <span class="finding-title">${escapeHtml(f.title)}</span>
      ${f.was_unexpected ? '<span class="unexpected-tag">UNEXPECTED</span>' : ''}
      <span class="sev-badge sev-${f.severity}">${f.severity.toUpperCase()}</span>
    </div>
    <div class="finding-explain">${escapeHtml(f.explanation)}</div>
    <div class="transcript">${transcriptHtml}</div>
  `;
  document.getElementById("findingsList").appendChild(card);
}

function renderReport(report) {
  document.getElementById("report").style.display = "block";
  document.getElementById("statFailures").textContent = report.total_failures;
  document.getElementById("statUnexpected").textContent = report.unexpected_failures;
  document.getElementById("statPassed").textContent = report.passed;
  document.getElementById("statTotal").textContent = report.total_personas;

  // Re-sort findings by severity
  const list = document.getElementById("findingsList");
  const cards = Array.from(list.children);
  cards.sort((a, b) => {
    const sa = a.querySelector(".sev-badge").className.split("sev-")[1];
    const sb = b.querySelector(".sev-badge").className.split("sev-")[1];
    return severityRank(sa) - severityRank(sb);
  });
  cards.forEach(c => list.appendChild(c));

  const ki = document.getElementById("keyInsight");
  if (report.unexpected_failures > 0) {
    ki.innerHTML = `
      <div class="ki-title">THE CORE VALUE</div>
      <p><strong>${report.unexpected_failures} of these ${report.total_failures} failures were ones you likely never wrote a test for.</strong> The Wind Tunnel found them by simulating customers you didn't think to try — before they reached production.</p>`;
    ki.style.display = "block";
  } else if (report.total_failures === 0) {
    ki.innerHTML = `
      <div class="ki-title">CLEAN RUN</div>
      <p>Your agent passed all ${report.total_personas} synthetic customers. Try increasing persona count or adding higher-risk domains to push it harder.</p>`;
    ki.style.display = "block";
  } else {
    ki.style.display = "none";
  }
}

function setStatus(msg) { document.getElementById("statusText").textContent = msg; }
function setProgress(pct) { document.getElementById("progressFill").style.width = pct + "%"; }
function showError(msg) {
  const box = document.getElementById("errorBox");
  box.textContent = msg;
  box.style.display = "block";
}
function hideError() { document.getElementById("errorBox").style.display = "none"; }
