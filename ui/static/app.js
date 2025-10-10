async function postJSON(url, data) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data || {}),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}

function el(id) { return document.getElementById(id); }

// Busy overlay helpers
function setBusy(on, msg) {
  const overlay = el("busy");
  const msgEl = el("busyMsg");
  const btnIndex = el("btnIndex");
  const btnAsk = el("btnAsk");
  if (!overlay) return;
  overlay.style.display = on ? "flex" : "none";
  if (msgEl && msg) msgEl.innerText = msg;
  if (btnIndex) btnIndex.disabled = !!on;
  if (btnAsk) btnAsk.disabled = !!on;
}

function renderPlan(plan) {
  el("planTargets").innerHTML = `<div><b>Selected targets:</b> ${
    (plan.targets || []).map(c => `<span class="badge">${c}</span>`).join("") || "n/a"
  }</div>`;
  el("planComponents").innerHTML = `<div><b>Top components:</b><br/>${
    (plan.components_ranking || []).map(e => `- ${e.component} (score=${e.score})`).join("<br/>") || "n/a"
  }</div>`;
  el("planValues").innerHTML = `<div><b>Values files:</b><br/>${
    (plan.values_files || []).map(v => `- ${v.component} | ${v.profile || "common"} | ${v.path}`).join("<br/>") || "n/a"
  }</div>`;
  el("planTop").innerHTML = `<div><b>Top retrieved:</b><br/>${
    (plan.top_results || []).map(r => `- (${r.type}) ${r.component} | ${r.path} | s=${(r.score||0).toFixed(3)}`).join("<br/>") || "n/a"
  }</div>`;
  el("signals").innerHTML = `<div><b>Signals:</b><pre>${plan.signals_summary || "n/a"}</pre></div>`;
}

function renderLLM(llm, answer) {
  el("llmStatus").innerText = llm.ok ? `OK (${llm.model}, ${llm.took_s || 0}s)` : `ERROR: ${llm.error || "unknown"}`;
  el("answer").innerText = answer || "";
}

function renderIntents(intents) {
  el("intents").innerText = intents && intents.length ? JSON.stringify(intents, null, 2) : "n/a";
}

function renderFinalYaml(finalYaml, patchesDir) {
  el("finalYaml").innerText = finalYaml || "# no YAML produced";
  el("patchDir").innerText = patchesDir ? `Patches dir: ${patchesDir}` : "";
}

async function onIndex() {
  const payload = {
    root: el("root").value || ".",
    k8s: el("k8s").value || "k8s",
    embed_model: el("embedModel").value || "all-MiniLM-L6-v2",
  };
  try {
    setBusy(true, "Building index... please wait");
    const r = await postJSON("/api/index", payload);
    alert(`Index built: ${JSON.stringify(r.info)}`);
  } catch (e) {
    alert(`Index error: ${e.message}`);
  } finally {
    setBusy(false);
  }
}

async function onAsk() {
  const payload = {
    root: el("root").value || ".",
    k8s: el("k8s").value || "k8s",
    embed_model: el("embedModel").value || "all-MiniLM-L6-v2",
    model: el("llmModel").value || "llama3.1:latest",
    query: el("query").value || "",
    types: el("types").value || "values,manifest",
    k: Number(el("k").value || 10),
    top_components: Number(el("topComponents").value || 1),
    min_score: el("minScore").value === "" ? null : Number(el("minScore").value),
    prompt_file: "prompts.yaml",
  };
  try {
    setBusy(true, "Planning and querying LLM... please wait");
    el("llmStatus").innerText = "Waiting for LLM response...";
    const r = await postJSON("/api/ask", payload);
    renderPlan(r.plan || {});
    renderLLM(r.llm || {}, r.answer);
    renderIntents(r.intents || []);
    renderFinalYaml(r.final_yaml, r.patches_dir);
  } catch (e) {
    alert(`Ask error: ${e.message}`);
  } finally {
    setBusy(false);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  el("btnIndex").addEventListener("click", onIndex);
  el("btnAsk").addEventListener("click", onAsk);
  el("copyYaml").addEventListener("click", () => {
    const txt = el("finalYaml").innerText;
    navigator.clipboard.writeText(txt).then(() => alert("YAML copied"));
  });
});
