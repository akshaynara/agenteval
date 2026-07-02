// AgentEval CX Wind Tunnel v2 — frontend logic

const BACKEND_URL = "https://agenteval-backend.onrender.com";

// switchTab is defined inline in index.html (before this script loads)

// ─── V1: OPERATIONAL MODE ─────────────────────────────────────────────────
let currentMode = "endpoint";

function setMode(mode) {
  currentMode = mode;
  document.getElementById("modeEndpoint").classList.toggle("active", mode === "endpoint");
  document.getElementById("modePrompt").classList.toggle("active", mode === "prompt");
  document.getElementById("endpointFields").style.display = mode === "endpoint" ? "block" : "none";
  document.getElementById("promptFields").style.display = mode === "prompt" ? "block" : "none";
}

document.getElementById("numPersonas").addEventListener("input", e => {
  document.getElementById("numPersonasVal").textContent = e.target.value;
});
document.getElementById("turns").addEventListener("input", e => {
  document.getElementById("turnsVal").textContent = e.target.value;
});

function escapeHtml(s) {
  const d = document.createElement("div"); d.textContent = s || ""; return d.innerHTML;
}
function severityRank(sev) {
  return { critical: 0, high: 1, medium: 2, low: 3, pass: 4 }[sev] ?? 5;
}

async function runTest() {
  const apiKey = document.getElementById("apiKey").value.trim().replace(/\s+/g, "");
  const domain = document.getElementById("domain").value.trim();
  const numPersonas = parseInt(document.getElementById("numPersonas").value);
  const turns = parseInt(document.getElementById("turns").value);

  if (!apiKey) { v1ShowError("Please enter your Anthropic API key."); return; }
  if (!domain) { v1ShowError("Please enter a test domain."); return; }

  const agent = { mode: currentMode, reply_field: "reply" };
  if (currentMode === "endpoint") {
    agent.endpoint_url = document.getElementById("endpointUrl").value.trim();
    agent.reply_field = document.getElementById("replyField").value.trim() || "reply";
    if (!agent.endpoint_url) { v1ShowError("Please enter your agent endpoint URL, or switch to Prompt mode."); return; }
  } else {
    agent.system_prompt = document.getElementById("systemPrompt").value.trim();
    if (!agent.system_prompt) { v1ShowError("Please paste your agent's system prompt."); return; }
  }

  v1HideError();
  document.getElementById("v1Report").style.display = "none";
  document.getElementById("v1FindingsList").innerHTML = "";
  document.getElementById("v1Progress").style.display = "block";
  document.getElementById("runBtn").disabled = true;
  v1SetStatus("Connecting...");
  v1SetProgress(0);

  const payload = { api_key: apiKey, domain, agent, num_personas: numPersonas, turns_per_conversation: turns };

  try {
    const resp = await fetch(`${BACKEND_URL}/run-test`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    if (!resp.ok) throw new Error(`Server returned ${resp.status}. Is the backend running?`);

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
      buffer = parts.pop();
      for (const part of parts) {
        const lines = part.split("\n");
        let event = "message", data = "";
        for (const line of lines) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          if (line.startsWith("data:")) data += line.slice(5).trim();
        }
        if (!data) continue;
        const parsed = JSON.parse(data);
        if (event === "status") {
          v1SetStatus(parsed.message);
          if (parsed.current && parsed.total) v1SetProgress((parsed.current / parsed.total) * 100);
          if (parsed.done) { document.getElementById("v1Spinner").style.display = "none"; v1SetProgress(100); }
        } else if (event === "conversation") {
          findings.push(parsed.result);
          v1AppendFinding(parsed.result);
          document.getElementById("v1Report").style.display = "block";
        } else if (event === "report") {
          report = parsed.report;
        } else if (event === "error") {
          v1ShowError(parsed.message);
        }
      }
    }
    if (report) v1RenderReport(report);
  } catch (err) {
    v1ShowError(err.message || "Something went wrong.");
  } finally {
    document.getElementById("runBtn").disabled = false;
  }
}

function v1AppendFinding(result) {
  const f = result.finding;
  const card = document.createElement("div");
  card.className = "finding-card";
  card.onclick = () => { const t = card.querySelector(".transcript"); t.style.display = t.style.display === "block" ? "none" : "block"; };
  const transcriptHtml = result.transcript.map(t => `
    <div class="turn turn-${t.role === "user" ? "user" : "agent"}">
      <div class="turn-role">${t.role === "user" ? "CUSTOMER" : "AGENT"}</div>
      <div class="turn-content">${escapeHtml(t.content)}</div>
    </div>`).join("");
  card.innerHTML = `
    <div class="finding-head">
      <span class="finding-persona">${escapeHtml(f.persona_name)}</span>
      <span class="finding-title">${escapeHtml(f.title)}</span>
      ${f.was_unexpected ? '<span class="unexpected-tag">UNEXPECTED</span>' : ''}
      <span class="sev-badge sev-${f.severity}">${(f.severity || "").toUpperCase()}</span>
    </div>
    <div class="finding-explain">${escapeHtml(f.explanation)}</div>
    <div class="transcript">${transcriptHtml}</div>`;
  document.getElementById("v1FindingsList").appendChild(card);
}

function v1RenderReport(report) {
  document.getElementById("v1Report").style.display = "block";
  document.getElementById("v1StatFail").textContent = report.total_failures;
  document.getElementById("v1StatUnexp").textContent = report.unexpected_failures;
  document.getElementById("v1StatPass").textContent = report.passed;
  document.getElementById("v1StatTotal").textContent = report.total_personas;
  const list = document.getElementById("v1FindingsList");
  Array.from(list.children).sort((a, b) => {
    const sa = a.querySelector(".sev-badge").className.split("sev-")[1];
    const sb = b.querySelector(".sev-badge").className.split("sev-")[1];
    return severityRank(sa) - severityRank(sb);
  }).forEach(c => list.appendChild(c));
  const ki = document.getElementById("v1Insight");
  if (report.unexpected_failures > 0) {
    ki.innerHTML = `<div class="ki-title">THE CORE VALUE</div><p><strong>${report.unexpected_failures} of ${report.total_failures} failures were ones you likely never wrote a test for.</strong> The Wind Tunnel found them by simulating customers you didn't think to try.</p>`;
    ki.style.display = "block";
  } else if (report.total_failures === 0) {
    ki.innerHTML = `<div class="ki-title">CLEAN RUN</div><p>Your agent passed all ${report.total_personas} synthetic customers. Try more personas or higher-risk domains.</p>`;
    ki.style.display = "block";
  }
}

function v1SetStatus(msg) { document.getElementById("v1StatusText").textContent = msg; }
function v1SetProgress(pct) { document.getElementById("v1ProgressFill").style.width = pct + "%"; }
function v1ShowError(msg) { const b = document.getElementById("v1ErrorBox"); b.textContent = msg; b.style.display = "block"; }
function v1HideError() { document.getElementById("v1ErrorBox").style.display = "none"; }

// ─── V2: RAI SAFETY ───────────────────────────────────────────────────────
let currentRaiMode = "endpoint";

function setRaiMode(mode) {
  currentRaiMode = mode;
  document.getElementById("raiModeEndpoint").classList.toggle("active", mode === "endpoint");
  document.getElementById("raiModePrompt").classList.toggle("active", mode === "prompt");
  document.getElementById("raiEndpointFields").style.display = mode === "endpoint" ? "block" : "none";
  document.getElementById("raiPromptFields").style.display = mode === "prompt" ? "block" : "none";
}

document.getElementById("numProbes").addEventListener("input", e => {
  document.getElementById("numProbesVal").textContent = e.target.value;
});
document.getElementById("raiTurns").addEventListener("input", e => {
  document.getElementById("raiTurnsVal").textContent = e.target.value;
});

// Category checkbox styling
document.querySelectorAll(".cat-item input[type=checkbox]").forEach(cb => {
  cb.addEventListener("change", () => {
    cb.closest(".cat-item").classList.toggle("checked", cb.checked);
  });
});

function getSelectedCategories() {
  return Array.from(document.querySelectorAll(".cat-item input[type=checkbox]:checked")).map(cb => cb.value);
}

async function runRaiTest() {
  const apiKey = document.getElementById("apiKey").value.trim().replace(/\s+/g, "");
  const domain = document.getElementById("raiDomain").value.trim();
  const numProbes = parseInt(document.getElementById("numProbes").value);
  const turns = parseInt(document.getElementById("raiTurns").value);
  const categories = getSelectedCategories();

  if (!apiKey) { raiShowError("Please enter your Anthropic API key."); return; }
  if (!domain) { raiShowError("Please enter a test domain."); return; }
  if (categories.length === 0) { raiShowError("Please select at least one RAI category."); return; }

  const agent = { mode: currentRaiMode, reply_field: "reply" };
  if (currentRaiMode === "endpoint") {
    agent.endpoint_url = document.getElementById("raiEndpointUrl").value.trim();
    agent.reply_field = document.getElementById("raiReplyField").value.trim() || "reply";
    if (!agent.endpoint_url) { raiShowError("Please enter your agent endpoint URL, or switch to Prompt mode."); return; }
  } else {
    agent.system_prompt = document.getElementById("raiSystemPrompt").value.trim();
    if (!agent.system_prompt) { raiShowError("Please paste your agent's system prompt."); return; }
  }

  raiHideError();
  document.getElementById("raiReport").style.display = "none";
  document.getElementById("raiFindingsList").innerHTML = "";
  document.getElementById("raiProgress").style.display = "block";
  document.getElementById("raiRunBtn").disabled = true;
  raiSetStatus("Connecting...");
  raiSetProgress(0);

  const payload = { api_key: apiKey, domain, agent, num_probes: numProbes, turns_per_conversation: turns, categories };

  try {
    const resp = await fetch(`${BACKEND_URL}/run-rai-test`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    if (!resp.ok) throw new Error(`Server returned ${resp.status}. Is the backend running?`);

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let report = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop();
      for (const part of parts) {
        const lines = part.split("\n");
        let event = "message", data = "";
        for (const line of lines) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          if (line.startsWith("data:")) data += line.slice(5).trim();
        }
        if (!data) continue;
        const parsed = JSON.parse(data);
        if (event === "status") {
          raiSetStatus(parsed.message);
          if (parsed.current && parsed.total) raiSetProgress((parsed.current / parsed.total) * 100);
          if (parsed.done) { document.getElementById("raiSpinner").style.display = "none"; raiSetProgress(100); }
        } else if (event === "rai_conversation") {
          raiAppendFinding(parsed.result);
          document.getElementById("raiReport").style.display = "block";
        } else if (event === "rai_report") {
          report = parsed.report;
        } else if (event === "error") {
          raiShowError(parsed.message);
        }
      }
    }
    if (report) raiRenderReport(report);
  } catch (err) {
    raiShowError(err.message || "Something went wrong.");
  } finally {
    document.getElementById("raiRunBtn").disabled = false;
  }
}

function raiAppendFinding(result) {
  const f = result.finding;
  const card = document.createElement("div");
  card.className = "finding-card";
  card.onclick = () => { const t = card.querySelector(".transcript"); t.style.display = t.style.display === "block" ? "none" : "block"; };
  const transcriptHtml = result.transcript.map(t => `
    <div class="turn turn-${t.role === "user" ? "user" : "agent"}">
      <div class="turn-role">${t.role === "user" ? "ATTACKER" : "AGENT"}</div>
      <div class="turn-content">${escapeHtml(t.content)}</div>
    </div>`).join("");
  const catClass = `cat-${f.category}`;
  card.innerHTML = `
    <div class="finding-head">
      <span class="cat-pill ${catClass}">${(f.category || "").replace(/_/g, " ").toUpperCase()}</span>
      <span class="finding-persona">${escapeHtml(f.probe_name)}</span>
      <span class="finding-title">${escapeHtml(f.title)}</span>
      ${f.attack_succeeded ? '<span class="attack-tag">ATTACK SUCCEEDED</span>' : ''}
      <span class="sev-badge sev-${f.severity}">${(f.severity || "").toUpperCase()}</span>
    </div>
    <div class="finding-explain">${escapeHtml(f.explanation)}</div>
    <div class="transcript">${transcriptHtml}</div>`;
  document.getElementById("raiFindingsList").appendChild(card);
}

function raiRenderReport(report) {
  document.getElementById("raiReport").style.display = "block";
  document.getElementById("raiStatFail").textContent = report.total_failures;
  document.getElementById("raiStatPass").textContent = report.passed;
  document.getElementById("raiStatTotal").textContent = report.total_probes;
  const list = document.getElementById("raiFindingsList");
  Array.from(list.children).sort((a, b) => {
    const sa = a.querySelector(".sev-badge").className.split("sev-")[1];
    const sb = b.querySelector(".sev-badge").className.split("sev-")[1];
    return severityRank(sa) - severityRank(sb);
  }).forEach(c => list.appendChild(c));
  const ki = document.getElementById("raiInsight");
  if (report.total_failures > 0) {
    ki.innerHTML = `<div class="ki-title">SAFETY SUMMARY</div><p><strong>${report.total_failures} safety boundary failure${report.total_failures > 1 ? 's' : ''} found across ${report.categories_tested.length} categories.</strong> These are real attack vectors that succeeded against your agent in a controlled environment — before they could succeed in production.</p>`;
    ki.style.display = "block";
  } else {
    ki.innerHTML = `<div class="ki-title">SAFETY BOUNDARIES HELD</div><p>Your agent resisted all ${report.total_probes} adversarial safety probes. Consider increasing probe count or adding more aggressive attack variations.</p>`;
    ki.style.display = "block";
  }
}

function raiSetStatus(msg) { document.getElementById("raiStatusText").textContent = msg; }
function raiSetProgress(pct) { document.getElementById("raiProgressFill").style.width = pct + "%"; }
function raiShowError(msg) { const b = document.getElementById("raiErrorBox"); b.textContent = msg; b.style.display = "block"; }
function raiHideError() { document.getElementById("raiErrorBox").style.display = "none"; }
