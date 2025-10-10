import os, yaml, re
from typing import Dict, Any, List

DEFAULT_PROMPTS = {
    "system": "You are a Kubernetes expert. Be concise and accurate.",
    "user": (
        "Use only the provided context. Return precise, actionable guidance.\n"
        "- First, provide a short explanation of what should be changed and why.\n"
        "- Then, propose intents JSON in the format: {{\"intents\":[{{...}}]}}\n"
        "- Finally, mention key files and components used as references.\n"
        "Signals summary:\n{signals_summary}\n"
        "\nUser question:\n{query}\n\nContext:\n{context}\n\nTargets:\n{targets_table}\n\nValues:\n{values_table}\n\nTop results:\n{top_results_table}\n"
    ),
    "templates": {}
}

def load_prompts(path: str | None) -> Dict[str, Any]:
    if not path:
        return DEFAULT_PROMPTS
    p = os.path.abspath(path)
    if not os.path.isfile(p):
        return DEFAULT_PROMPTS
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        out: Dict[str, Any] = dict(DEFAULT_PROMPTS)
        if isinstance(data, dict):
            if isinstance(data.get("system"), str):
                out["system"] = data["system"]
            if isinstance(data.get("user"), str):
                out["user"] = data["user"]
            if isinstance(data.get("templates"), dict):
                out["templates"] = data["templates"]
        return out
    except Exception:
        return DEFAULT_PROMPTS

def _format_table(rows: List[str]) -> str:
    return "\n".join(rows) if rows else "n/a"

def _collect_neutral_signals(plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a neutral summary (no priorities). Let the LLM infer logic.
    """
    comps = [t.get("component") for t in (plan.get("targets") or []) if t and t.get("component")]
    vfs = plan.get("values_files") or []
    mfs = plan.get("manifest_files") or []
    keys_sample = []
    profiles = []
    for vf in vfs[:8]:
        profiles.append(vf.get("profile") or "common")
        ks = (vf.get("candidate_keys") or [])[:20]
        keys_sample.extend(ks)
    kinds = []
    images = []
    for mf in mfs:
        kinds.extend(mf.get("kinds") or [])
        images.extend(mf.get("images") or [])
    for r in plan.get("top_results") or []:
        kinds.extend(r.get("k8s_kinds") or [])
        images.extend(r.get("k8s_images") or [])
    return {
        "components": sorted(set(comps)),
        "profiles": sorted(set([p for p in profiles if p])),
        "kinds": sorted(set([str(k) for k in kinds if k])),
        "images_count": len(set(images)),
        "keys_count": len(keys_sample),
        "keys_example": keys_sample[:30],
    }

def _build_blocks(plan: Dict[str, Any]) -> Dict[str, str]:
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
    ctx_lines = ["Context:"]
    for r in top[:10]:
        ctx_lines.append(f"- [{r.get('type')}] {r.get('component')} :: {r.get('path')} (score={r.get('score')})")
    return {
        "context": "\n".join(ctx_lines),
        "targets_table": _format_table(targets_rows),
        "values_table": _format_table(values_rows),
        "top_results_table": _format_table(top_rows),
    }

def _prepare_template(tmpl: str, allowed: List[str]) -> str:
    """
    Escape all braces, then un-escape only placeholders from 'allowed'.
    This allows literal JSON like {"intents":[...]} in templates.
    """
    escaped = tmpl.replace("{", "{{").replace("}", "}}")
    for name in allowed:
        escaped = escaped.replace("{{" + name + "}}", "{" + name + "}")
    return escaped

def render_user_prompt(query: str, plan: Dict[str, Any], prompts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns:
      {"text": str, "signals_summary": str, "signals": Dict[str,Any]}
    """
    sig = _collect_neutral_signals(plan)
    sig_lines = [
        f"- components: {', '.join(sig['components']) or 'n/a'}",
        f"- profiles: {', '.join(sig['profiles']) or 'n/a'}",
        f"- kinds: {', '.join(sig['kinds']) or 'n/a'}",
        f"- images_count: {sig['images_count']}",
        f"- keys_count: {sig['keys_count']}",
    ]
    if sig["keys_example"]:
        sig_lines.append("- keys_example: " + ", ".join(sig["keys_example"]))
    signals_summary = "\n".join(sig_lines)

    blocks = _build_blocks(plan)
    allowed = ["query", "context", "targets_table", "values_table", "top_results_table", "signals_summary"]
    safe_user = _prepare_template(prompts["user"], allowed)
    text = safe_user.format(
        query=query,
        context=blocks["context"],
        targets_table=blocks["targets_table"],
        values_table=blocks["values_table"],
        top_results_table=blocks["top_results_table"],
        signals_summary=signals_summary,
    )
    return {"text": text, "signals_summary": signals_summary, "signals": sig}
