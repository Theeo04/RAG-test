import os, argparse, json, re, sys, time
from typing import List, Dict, Any
from rag_backend import index_k8s, have_index_fn, hybrid_search_fn, load_index_fn, extract_keys_fn
from plan_module import do_plan
from patch_module import patch_from_plan
import ollama
from plan_module import DEFAULT_MODEL_NAME as DEFAULT_EMBED_MODEL  # proper embedding model
from prompt_loader import load_prompts, render_user_prompt

DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:latest")

def _dbg(enabled: bool, *args):
    if enabled:
        print("[DEBUG]", *args)

def _pretty_res(plan: Dict[str, Any]) -> str:
    lines = []
    lines.append("Top components:")
    for r in (plan.get("components_ranking") or [])[:5]:
        lines.append(f"- {r['component']} (score={r['score']})")
    tgt = (plan.get("target") or {}).get("component", "n/a")
    lines.append("Target component: " + tgt)
    lines.append("Values files:")
    for vf in plan.get("values_files") or []:
        lines.append(f"- {vf['path']} (component={vf.get('component')}, profile={vf.get('profile')}, keys={vf.get('keys_count')})")
    return "\n".join(lines)

def _format_table(rows):
    return "\n".join(rows) if rows else "n/a"

def _context_blocks(plan: Dict[str, Any]) -> dict:
    # Build compact tables for prompt substitution
    targets = plan.get("targets") or ([plan.get("target")] if plan.get("target") else [])
    targets_rows = [f"- {t.get('component')}" for t in targets if t and t.get("component")]
    vfs = plan.get("values_files") or []
    values_rows = [
        f"- {vf.get('component')} | {vf.get('profile') or 'common'} | {vf.get('path')}"
        for vf in vfs
    ][:10]
    top = plan.get("top_results") or []
    top_rows = [
        f"- ({r.get('type')}) {r.get('component')} | {os.path.basename(r.get('path',''))} | score={round(r.get('score',0),3)}"
        for r in top[:15]
    ]
    # Context lines as before
    ctx_lines = ["Context:"]
    for r in top[:10]:
        ctx_lines.append(f"- [{r.get('type')}] {r.get('component')} :: {r.get('path')} (score={r.get('score')})")
    return {
        "context": "\n".join(ctx_lines),
        "targets_table": _format_table(targets_rows),
        "values_table": _format_table(values_rows),
        "top_results_table": _format_table(top_rows),
    }

def _build_user_prompt(query: str, plan: Dict[str, Any], templates: dict) -> str:
    blocks = _context_blocks(plan)
    return templates["user"].format(
        query=query,
        context=blocks["context"],
        targets_table=blocks["targets_table"],
        values_table=blocks["values_table"],
        top_results_table=blocks["top_results_table"],
    )

def _extract_intents(text: str) -> List[Dict[str, Any]]:
    # Look for a JSON block with "intents": [...]
    m = re.search(r"\{[\s\S]*?\"intents\"\s*:\s*\[[\s\S]*?\][\s\S]*?\}", text)
    if not m:
        # try fenced json
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

def _gen_answer(query: str, prompt: str, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    """
    Returns {"ok":bool,"content":str,"model":str,"took_s":float,"error":str|None}
    """
    try_models = [model]
    if model.lower() == "lama3.1:latest":
        try_models.append("llama3.1:latest")
    last_err = None
    for m in try_models:
        t0 = time.time()
        try:
            resp = ollama.chat(model=m, messages=[
                {"role": "system", "content": "Be concise and accurate."},
                {"role": "user", "content": prompt}
            ])
            took = round(time.time() - t0, 3)
            return {"ok": True, "content": resp["message"]["content"], "model": m, "took_s": took, "error": None}
        except Exception as e:
            last_err = str(e)
    return {"ok": False, "content": "", "model": try_models[-1], "took_s": 0.0, "error": last_err}

def _load_intents_override(arg: str) -> List[Dict[str, Any]]:
    """
    Accepts a JSON string or a file prefixed with @ (e.g., @/path/to/intents.json).
    Returns a list of intents or [].
    """
    if not arg:
        return []
    data = None
    try:
        s = arg
        if s.startswith("@"):
            with open(s[1:], "r", encoding="utf-8") as f:
                s = f.read()
        data = json.loads(s)
        if isinstance(data, dict) and isinstance(data.get("intents"), list):
            return data["intents"]
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []

def cmd_index(args):
    _dbg(args.verbose, f"Indexing root={args.root} k8s={args.k8s} embed_model={args.embed_model}")
    info = index_k8s(args.root, k8s_subdir=args.k8s, model_name=args.embed_model)
    print(json.dumps(info, indent=2))
    if args.verbose:
        idx_json = os.path.join(args.root, ".rag_index", "index.json")
        if os.path.exists(idx_json):
            with open(idx_json, "r", encoding="utf-8") as f:
                meta = json.load(f)
            items = meta.get("items", [])
            by_type: Dict[str, int] = {}
            by_comp: Dict[str, int] = {}
            for it in items:
                by_type[it["type"]] = by_type.get(it["type"], 0) + 1
                by_comp[it["component"]] = by_comp.get(it["component"], 0) + 1
            print("\n[DEBUG] Index summary")
            print(f"- Embedding model: {meta.get('model_name')}")
            print(f"- Types: {by_type}")
            print(f"- Top components: " + ", ".join([f"{k}={v}" for k, v in sorted(by_comp.items(), key=lambda kv: -kv[1])[:10]]))
        else:
            _dbg(True, "Index metadata not found at", idx_json)

def cmd_ask(args):
    root = args.root
    _dbg(args.verbose, f"Ask params: query={args.query!r}, k8s={args.k8s}, types={args.types}, k={args.k}, llm={args.model}, embed={args.embed_model}")
    if not have_index_fn(root):
        _dbg(args.verbose, "Index missing; building...")
        index_k8s(root, k8s_subdir=args.k8s, model_name=args.embed_model)
    else:
        _dbg(args.verbose, "Index present")

    _dbg(args.verbose, "Planning...")
    plan = do_plan(
        root=root,
        query=args.query,
        types=args.types.split(","),
        k=args.k,
        have_index_fn=have_index_fn,
        hybrid_search_fn=lambda *a, **kw: hybrid_search_fn(*a, **kw),
        load_index_fn=load_index_fn,
        extract_keys_fn=extract_keys_fn,
        model_name=args.embed_model,
        weights=None,
        top_components=args.top_components,
        min_score=args.min_score,
    )

    # Print summaries
    print(_pretty_res(plan))
    # Selected targets
    targets = plan.get("targets") or ([plan.get("target")] if plan.get("target") else [])
    targets_list = [t["component"] for t in targets if t and t.get("component")]
    print("Selected targets:", ", ".join(targets_list) if targets_list else "n/a")

    # Debug: hybrid search stats and top results
    if args.verbose:
        stats = plan.get("stats") or {}
        print(f"[DEBUG] Hybrid search stats: {stats}")
        top_results = plan.get("top_results") or []
        if top_results:
            print("[DEBUG] Top results:")
            for r in top_results:
                print(f"  - ({r.get('type')}) {r.get('component')} :: {r.get('path')} score={r.get('score')}")
        # Components ranking details
        comp_rank = plan.get("components_ranking") or []
        if comp_rank:
            print("[DEBUG] Components ranking (top 10):")
            for e in comp_rank[:10]:
                sig = e.get("signals") or {}
                print(f"  - {e['component']}: final={e['score']} sem={sig.get('semantic_norm')} sea={sig.get('search_norm')} aff={sig.get('affinity_norm')}")

        # Values files overview
        vfs = plan.get("values_files") or []
        if vfs:
            print("[DEBUG] Values files selected:")
            for vf in vfs:
                print(f"  - {vf.get('component')} | {vf.get('profile')} | {vf.get('path')} | keys={vf.get('keys_count')}")

    # Load prompt templates and render (neutral, no hardcoded priorities)
    prompts = load_prompts(args.prompt_file)
    rendered = render_user_prompt(args.query, plan, prompts)
    user_prompt = rendered["text"]
    print("[DEBUG] Signals summary:")
    print(rendered["signals_summary"])

    print(f"\nSending LLM request to model '{args.model}' ...")
    ans = ollama.chat(model=args.model, messages=[
        {"role": "system", "content": prompts["system"]},
        {"role": "user", "content": user_prompt}
    ]) if True else None

    # Normalize LLM status
    ok = False; answer = ""; took = 0.0; used_model = args.model; err = None
    try:
        answer = ans["message"]["content"]
        ok = True
    except Exception as e:
        err = str(e)
    if ok:
        print("LLM responded OK.")
    else:
        print(f"LLM request failed: {err or 'unknown error'}")

    print("\n--- LLM answer ---\n")
    print(answer if ok else f"LLM error: {err or 'unknown error'}")

    # Patching + final output
    if args.patches_dir:
        intents = []
        if args.intents_json:
            intents = _load_intents_override(args.intents_json)
            if intents:
                print(f"\nUsing intents from --intents-json override: {json.dumps(intents)}")
        if not intents and ok:
            intents = _extract_intents(answer)

        if intents:
            print(f"\nDetected intents: {json.dumps(intents)}")
            patch_info = patch_from_plan(plan, intents, args.patches_dir)
            final_single = patch_info.get("final_single_yaml") or patch_info.get("final_merged_yaml")
            is_multi = patch_info.get("final_yaml_is_multidoc")
            # Explanation + references
            print("\n=== Explanation ===")
            print(f"- Applied {len(intents)} intent(s) across {len(patch_info.get('patches', []))} values file(s).")
            print("- Keys touched:")
            for it in intents:
                print(f"  - {it.get('op')} {it.get('key')} {('= '+str(it.get('value'))) if it.get('op')=='set' else ''}")
            print("\n=== References ===")
            refs = []
            for vf in plan.get("values_files") or []:
                refs.append(f"- values: {vf.get('path')} (component={vf.get('component')}, profile={vf.get('profile')})")
            for r in (plan.get("top_results") or [])[:10]:
                refs.append(f"- {r.get('type')} manifest: {r.get('path')} (component={r.get('component')})")
            print("\n".join(refs) if refs else "n/a")

            # Final YAML
            print("\n=== Final YAML (copy-paste) ===")
            if is_multi:
                print("# Note: multiple values files affected; this is a multi-document YAML")
            print(final_single)
        else:
            print("\nNo intents detected in model output. Skipping patch generation.")

def main():
    ap = argparse.ArgumentParser(description="RAG over /k8s with Llama3.1 via Ollama")
    ap.add_argument("--root", default=".", help="Repository root (parent of /k8s)")
    sub = ap.add_subparsers(required=True)

    ap_index = sub.add_parser("index", help="Build index from /k8s")
    ap_index.add_argument("--k8s", default="k8s", help="k8s folder relative to root")
    ap_index.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL,
        help="Embedding model for SentenceTransformers (e.g., all-MiniLM-L6-v2)"
    )
    ap_index.add_argument("--verbose", action="store_true", help="Verbose debug output")
    ap_index.set_defaults(func=cmd_index)

    ap_ask = sub.add_parser("ask", help="Plan + retrieve + ask LLM")
    ap_ask.add_argument("query", help="User question")
    ap_ask.add_argument("--k8s", default="k8s", help="k8s folder relative to root")
    ap_ask.add_argument("--types", default="values,manifest", help="Comma-separated: values,manifest")
    ap_ask.add_argument("-k", type=int, default=10, help="Top results")
    ap_ask.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model (e.g., llama3.1:latest)")
    ap_ask.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL,
        help="Embedding model for SentenceTransformers (e.g., all-MiniLM-L6-v2)"
    )
    ap_ask.add_argument("--patches-dir", default=None, help="If set, generate patch YAMLs when intents are found")
    ap_ask.add_argument("--top-components", type=int, default=1, help="Include top-N components for context/patching")
    ap_ask.add_argument("--min-score", type=float, default=None, help="Include components with fused score >= threshold")
    ap_ask.add_argument("--verbose", action="store_true", help="Verbose debug output")
    ap_ask.add_argument("--prompt-file", default="prompts.yaml", help="Path to a YAML with {system,user} templates")
    ap_ask.add_argument("--intents-json", default=None, help="JSON string or @/path/to/file with intents")
    ap_ask.set_defaults(func=cmd_ask)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
