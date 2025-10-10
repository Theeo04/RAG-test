import os, re, json, time, tempfile
from typing import Any, Dict, List
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import ollama
# Add project root to sys.path so imports work when running from ui/
import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Import existing modules
from rag_backend import index_k8s, have_index_fn, hybrid_search_fn, load_index_fn, extract_keys_fn
from plan_module import do_plan, DEFAULT_MODEL_NAME as DEFAULT_EMBED_MODEL
from prompt_loader import load_prompts, render_user_prompt
from patch_module import patch_from_plan

app = Flask(__name__, static_folder="static", static_url_path="/")
CORS(app)

def _extract_intents(text: str) -> List[Dict[str, Any]]:
    m = re.search(r"\{[\s\S]*?\"intents\"\s*:\s*\[[\s\S]*?\][\s\S]*?\}", text)
    if not m:
        m = re.search(r"```json([\s\S]*?)```", text, re.IGNORECASE)
        if not m:
            return []
        blob = m.group(1)
    else:
        blob = m.group(0)
    try:
        data = json.loads(blob)
        intents = data.get("intents") if isinstance(data, dict) else None
        return intents if isinstance(intents, list) else []
    except Exception:
        return []

@app.route("/")
def root():
    return send_from_directory(app.static_folder, "index.html")

@app.post("/api/index")
def api_index():
    body = request.get_json(force=True) or {}
    root = body.get("root") or "."
    k8s = body.get("k8s") or "k8s"
    embed_model = body.get("embed_model") or DEFAULT_EMBED_MODEL
    info = index_k8s(root, k8s_subdir=k8s, model_name=embed_model)
    return jsonify({"ok": True, "info": info})

@app.post("/api/ask")
def api_ask():
    body = request.get_json(force=True) or {}
    root = body.get("root") or "."
    # Normalize root to absolute for consistent path handling
    root = os.path.abspath(root)
    k8s = body.get("k8s") or "k8s"
    query = body.get("query") or ""
    types = (body.get("types") or ["values", "manifest"])
    if isinstance(types, str):
        types = [t.strip() for t in types.split(",") if t.strip()]
    k = int(body.get("k") or 10)
    model = body.get("model") or os.environ.get("OLLAMA_MODEL", "llama3.1:latest")
    embed_model = body.get("embed_model") or DEFAULT_EMBED_MODEL
    top_components = int(body.get("top_components") or 1)
    min_score = body.get("min_score")
    min_score = float(min_score) if (min_score is not None and str(min_score) != "") else None
    prompt_file = body.get("prompt_file") or "prompts.yaml"
    # Resolve prompts file relative to root if not absolute
    if not os.path.isabs(prompt_file):
        prompt_file = os.path.join(root, prompt_file)
    patches_dir = body.get("patches_dir") or tempfile.mkdtemp(prefix="rag_ui_patches_")

    # Ensure index
    if not have_index_fn(root):
        index_k8s(root, k8s_subdir=k8s, model_name=embed_model)

    # Plan
    plan = do_plan(
        root=root,
        query=query,
        types=types,
        k=k,
        have_index_fn=have_index_fn,
        hybrid_search_fn=lambda *a, **kw: hybrid_search_fn(*a, **kw),
        load_index_fn=load_index_fn,
        extract_keys_fn=extract_keys_fn,
        model_name=embed_model,
        weights=None,
        top_components=top_components,
        min_score=min_score,
    )

    # Render prompt with neutral signals
    prompts = load_prompts(prompt_file)
    rendered = render_user_prompt(query, plan, prompts)

    # LLM call
    llm_ok = False; answer = ""; error = None; took = 0.0
    t0 = time.time()
    try:
        resp = ollama.chat(model=model, messages=[
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": rendered["text"]}
        ])
        answer = resp["message"]["content"]
        llm_ok = True
        took = round(time.time() - t0, 3)
    except Exception as e:
        error = str(e)

    intents = _extract_intents(answer) if llm_ok else []
    patch_info = None
    final_yaml = None
    if intents:
        try:
            patch_info = patch_from_plan(plan, intents, patches_dir)
            final_yaml = patch_info.get("final_single_yaml") or patch_info.get("final_merged_yaml")
        except Exception as e:
            error = f"Patch error: {e}"

    # Compact plan summary for UI
    comps = [{"component": e["component"], "score": e["score"]} for e in (plan.get("components_ranking") or [])[:10]]
    targets = [t["component"] for t in (plan.get("targets") or []) if t and t.get("component")]
    values_files = [{"path": vf.get("path"), "component": vf.get("component"), "profile": vf.get("profile")} for vf in plan.get("values_files") or []]
    top_results = [{"type": r.get("type"), "component": r.get("component"), "path": r.get("path"), "score": r.get("score")} for r in (plan.get("top_results") or [])[:15]]

    return jsonify({
        "ok": True,
        "plan": {
            "components_ranking": comps,
            "targets": targets,
            "values_files": values_files,
            "top_results": top_results,
            "signals_summary": rendered.get("signals_summary"),
        },
        "llm": {"ok": llm_ok, "model": model, "took_s": took, "error": error},
        "answer": answer,
        "intents": intents,
        "patch": patch_info or {},
        "final_yaml": final_yaml,
        "patches_dir": patches_dir,
    })

if __name__ == "__main__":
    # Run: python ui/app.py, then open http://localhost:5000
    app.run(host="0.0.0.0", port=5000, debug=True)
