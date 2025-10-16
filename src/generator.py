import os
import re
from typing import Dict, List, Tuple
import yaml  # NEW
from .llm.ollama_client import OllamaClient  # NEW
from .context import compose_context  # NEW
import time  # NEW
import requests  # NEW

GEN_SYS_PROMPT = """You are a Kubernetes Helm values/common.yaml generator.
- Follow the schema and conventions from the provided context.
- Preserve idioms: image fields, global.app, probes, configMaps, volumes, initContainers, etc.
- Keep templating (Go templates) as in examples if applicable.
- Output only valid YAML for values/common.yaml (no comments unless present in content blocks).
- IMPORTANT: Only include sections explicitly requested by the user. Do NOT add defaults or extra configurations unless strictly required by the request."""

def _format_context(retrieved: List[Tuple[float, Dict, str]], schema_summary: Dict[str, Dict]) -> str:
    parts = []
    for score, meta, text in retrieved:
        parts.append(f"# score={score:.3f} file={meta.get('file')} section={meta.get('section')}\n{text}")
    parts.append("\n# schema-summary\n")
    for f, sch in schema_summary.items():
        parts.append(f"# file: {f}")
        for top, children in sch.items():
            ck = ", ".join(children[:8])
            parts.append(f"# {top}: {ck}")
    return "\n".join(parts)

def _ollama_mode() -> str:  # NEW
    """
    Controls which Ollama API to use: chat | generate | auto (try chat, then generate).
    """
    return os.getenv("OLLAMA_MODE", "auto").lower()

def _detect_allowed_keys(requirement: str) -> Dict[str, any]:
    """
    Tighter heuristics: pick nginx image correctly; prefer readiness=/ready and metrics=/metrics;
    extract configMap file path.
    """
    txt = requirement.lower()
    allowed = set()

    # metrics
    metrics = None
    if "metrics" in txt or "monitoring" in txt:
        path = None
        m1 = re.search(r'metrics[^/\n\r]*?(?:path|at)\s*(/[-/a-z0-9._]+)', requirement, re.IGNORECASE)
        if m1:
            path = m1.group(1)
        if not path:
            m2 = re.search(r'/(metrics[^\s,;]*)', requirement, re.IGNORECASE)
            if m2:
                path = "/" + m2.group(1).lstrip("/")
        port = None
        mport = re.search(r'(?:on\s+)?port\s+(\d+)', requirement, re.IGNORECASE)
        if mport:
            port = int(mport.group(1))
        metrics = {"path": path, "port": port}
        allowed.add("global")

    # hosts
    hosts = None
    if re.search(r'\bhost[s]?\b', txt):
        cands: List[str] = []
        for m in re.finditer(r'\bhost[s]?\b[^A-Za-z0-9._-]*([a-z0-9.-]+\.[a-z]{2,})', requirement, re.IGNORECASE):
            cands.append(m.group(1))
        def ok(h: str) -> bool:
            return not h.lower().endswith((".yaml", ".yml")) and "." in h
        hosts = [h.rstrip(".") for h in cands if ok(h)]
        hosts = list(dict.fromkeys(hosts)) or None
        if hosts:
            allowed.add("global")

    # readiness
    readiness_path = None
    if re.search(r'\breadiness\b|\bprobe\b', txt, re.IGNORECASE):
        r1 = re.search(r'readiness[^/\n\r]*?(?:path|at|url)?[: ]*\s*(/[-/a-z0-9._]+)', requirement, re.IGNORECASE)
        if r1:
            readiness_path = r1.group(1)
        else:
            r2 = re.search(r'/(ready[^\s,;]*)', requirement, re.IGNORECASE)
            if r2:
                readiness_path = "/" + r2.group(1).lstrip("/")
        allowed.add("probes")

    # image
    image_repo = None
    image_tag = None
    if "image" in txt or "nginx" in txt:
        m = re.search(r'image(?:\s+|:)\s*([a-z0-9/_\-.]+)(?::([a-z0-9._\-]+))?', requirement, re.IGNORECASE)
        cand = m.group(1) if m else None
        tag = m.group(2) if m else None
        # accept only plausible image identifiers; else fall back to nginx if present
        def plausible(repo: str) -> bool:
            return "/" in repo or "." in repo or repo in {"nginx", "busybox", "alpine"}
        if cand and plausible(cand.lower()):
            image_repo = cand
            image_tag = tag
        elif "nginx" in txt:
            image_repo = "nginx"
        if image_repo:
            allowed.add("image")

    # configMap path (e.g., /etc/nginx/conf.d/metrics.conf)
    configmap_path = None
    if "configmap" in txt:
        p = re.search(r'(/[-/a-z0-9._]+\.conf)', requirement, re.IGNORECASE) or re.search(r'(/[-/a-z0-9._]+)', requirement, re.IGNORECASE)
        if p:
            configmap_path = p.group(1)
        allowed.add("configMaps")

    # other sections
    for key, pat in [
        ("volumes", r'\bvolume[s]?\b|\bmount\b'),
        ("initContainers", r'\binit[- ]?container[s]?\b'),
        ("resources", r'\bresource[s]?\b|\bcpu\b|\bmemory\b'),
        ("pdb", r'\bpdb\b|\bpod disruption budget\b'),
        ("secrets", r'\bsecret[s]?\b'),
        ("sidecars", r'\bsidecar[s]?\b'),
    ]:
        if re.search(pat, txt):
            allowed.add(key)

    return {
        "allowed": allowed,
        "metrics": metrics,
        "hosts": hosts,
        "readiness_path": readiness_path,
        "image_repo": image_repo,
        "image_tag": image_tag,
        "configmap_path": configmap_path,  # NEW
    }

_KEY_RE = re.compile(r'^([A-Za-z0-9._-]+):(?:\s|$)')

def _filter_yaml_top_sections(yaml_text: str, allowed: set[str]) -> str:
    """
    Remove any top-level sections that are not in 'allowed'.
    """
    lines = yaml_text.splitlines()
    out = []
    skip = False
    for i, line in enumerate(lines):
        # new top-level?
        if line and not line.startswith((" ", "\t")) and not line.lstrip().startswith("#"):
            m = _KEY_RE.match(line)
            if m:
                key = m.group(1)
                skip = key not in allowed
        if not skip:
            out.append(line)
    # strip trailing blank lines
    return "\n".join(out).strip()

def _schema_union(schema_summary: Dict[str, Dict]) -> Dict[str, set]:  # NEW
    """
    Union of schema across files: top-level -> set(second-level keys)
    """
    union: Dict[str, set] = {}
    for _, outline in (schema_summary or {}).items():
        for top, children in outline.items():
            if top not in union:
                union[top] = set()
            for c in children:
                union[top].add(c)
    return union

def _canonical_paths(union: Dict[str, set]) -> Dict[str, str]:  # NEW
    """
    Best-effort canonical paths for common features based on observed schema.
    """
    paths = {}
    # metrics
    if "global" in union and "app" in union["global"]:
        paths["metrics"] = "global.app.monitoring"
    elif "global" in union:
        paths["metrics"] = "global.monitoring"
    else:
        paths["metrics"] = "global.app.monitoring"  # default
    # ingresses
    if "global" in union and "ingresses" in union["global"]:
        paths["ingresses"] = "global.ingresses.main.hosts"
    else:
        paths["ingresses"] = "global.ingresses.main.hosts"
    # image
    paths["image"] = "image"
    # readiness
    paths["readiness"] = "probes.readiness"
    return paths

def _build_llm_prompt(ctx: str, requirement: str, req_info: Dict, union: Dict[str, set], paths: Dict[str, str]) -> str:  # NEW
    allowed_tops = sorted(list(req_info.get("allowed", set())))
    schema_lines = []
    for top in sorted(union.keys()):
        children = ", ".join(sorted(list(union[top]))[:12])
        schema_lines.append(f"- {top}: {children}")
    rules = [
        "Only include top-level sections explicitly requested by the user.",
        "Follow the canonical paths for requested items:",
        f"- metrics -> {paths['metrics']} (keys: activate, path, port)",
        f"- hosts -> {paths['ingresses']}",
        f"- image -> {paths['image']} (keys: repository, tag if present)",
        f"- readiness -> {paths['readiness']} (keys: activate, path)",
        "Output pure YAML with no code fences and no prose."
    ]
    req_bits = []
    if req_info.get("metrics"):
        m = req_info["metrics"]
        req_bits.append(f"metrics: path={m.get('path') or ''} port={m.get('port') if m.get('port') is not None else ''}")
    if req_info.get("hosts"):
        req_bits.append(f"hosts: {', '.join(req_info['hosts'])}")
    if req_info.get("image_repo"):
        tag = f":{req_info['image_tag']}" if req_info.get('image_tag') else ""
        req_bits.append(f"image: {req_info['image_repo']}{tag}")
    if req_info.get("readiness_path"):
        req_bits.append(f"readiness: path={req_info['readiness_path']}")

    return (
        f"Context:\n{ctx}\n\n"
        f"Observed schema (union):\n" + "\n".join(schema_lines) + "\n\n"
        f"Allowed top-level keys: {', '.join(allowed_tops) if allowed_tops else '(none)'}\n"
        f"Constraints:\n- " + "\n- ".join(rules) + "\n\n"
        f"User requirement:\n{requirement}\n\n"
        f"Extracted intents: " + (", ".join(req_bits) if req_bits else "(none)") + "\n\n"
        f"Produce only YAML for values/common.yaml matching the above."
    )

def _first_yaml_block(text: str) -> str | None:  # NEW
    """
    Find the first block that looks like YAML starting at a top-level key (e.g., "global:").
    """
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        s = line.lstrip()
        # ignore markdown/prose lines until a plausible yaml key appears
        if _KEY_RE.match(s):
            start = i
            break
    if start is None:
        return None
    block = "\n".join(lines[start:]).strip()
    return block or None

def _extract_yaml(text: str) -> str | None:
    """
    Extract YAML from a response. Prefer fenced code blocks if present.
    Then fallback to the first YAML-like block starting at a top-level key.
    """
    if not text:
        return None
    m = re.search(r"```(?:yaml|yml)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    block = _first_yaml_block(text)
    if block:
        return block
    return text.strip()

def _has_helm_templates(text: str) -> bool:  # NEW
    """
    Best-effort detection of Helm/Go templates.
    """
    if not text:
        return False
    return ("{{" in text and "}}" in text)

def _parse_yaml(text: str, verbose: bool = False) -> Dict | None:  # CHANGED
    """
    Lenient YAML parsing with Helm template sanitization and multi-document support.
    - Comments out standalone template-only lines.
    - Quotes inline {{ ... }} occurrences.
    - Tries yaml.safe_load_all and merges dict docs (last-wins).
    """
    def _strip_fences(src: str) -> str:
        if not src:
            return src
        m = re.search(r"```(?:yaml|yml)?\s*(.*?)```", src, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return src.strip()

    def _sanitize(src: str) -> str:
        tmpl_only = re.compile(r'^\s*\{\{[-\s]?.*?[-\s]?\}\}\s*$')
        out_lines: List[str] = []
        for ln in (src or "").splitlines():
            if tmpl_only.match(ln):
                out_lines.append("# " + ln)
            else:
                out_lines.append(ln)
        src2 = "\n".join(out_lines)
        # Quote unquoted inline templates
        src2 = re.sub(r'(?<!")(\{\{[^}]+}})(?!")', r'"\1"', src2)
        return src2

    def _merge_docs(docs: List[Dict]) -> Dict | None:
        merged: Dict = {}
        for d in docs:
            if isinstance(d, dict):
                merged.update(d)
        return merged or None

    try:
        clean = _strip_fences(text)
        # Try multi-doc first
        docs = list(yaml.safe_load_all(clean))
        if docs:
            merged = _merge_docs(docs)
            if merged is not None:
                return merged
        # Fallback single-doc
        doc = yaml.safe_load(clean)
        return doc if isinstance(doc, dict) else None
    except Exception as e:
        if verbose:
            preview = (text[:300] + "...") if len(text) > 300 else text
            print(f"[gen] yaml parse error: {e}\n[gen] yaml preview:\n{preview}")
        # Retry with sanitization
        try:
            clean = _strip_fences(text)
            sanitized = _sanitize(clean)
            if verbose:
                print("[gen] sanitized YAML for retry:\n" + (sanitized[:1000] + ("..." if len(sanitized) > 1000 else "")))
            docs = list(yaml.safe_load_all(sanitized))
            if docs:
                merged = _merge_docs(docs)
                if merged is not None:
                    return merged
            doc = yaml.safe_load(sanitized)
            return doc if isinstance(doc, dict) else None
        except Exception as e2:
            if verbose:
                print(f"[gen] retry parse failed: {e2}")
            return None

def _set_path(doc: Dict, dotted: str, value) -> None:  # NEW
    cur = doc
    parts = dotted.split(".")
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value

def _ensure_required_parts(doc: Dict, req_info: Dict, paths: Dict[str, str], verbose: bool = False) -> None:  # NEW
    # metrics
    m = req_info.get("metrics")
    if m and (m.get("path") or m.get("port") is not None):
        mon_path = paths["metrics"]
        node = {}
        node["activate"] = True
        if m.get("path"):
            node["path"] = m["path"]
        if m.get("port") is not None:
            node["port"] = m["port"]
        _set_path(doc, mon_path, node)
        if verbose:
            print(f"[gen] ensured metrics at {mon_path}")
    # hosts/ingresses
    hosts = req_info.get("hosts") or []
    if hosts:
        _set_path(doc, paths["ingresses"], hosts)
        if verbose:
            print(f"[gen] ensured hosts at {paths['ingresses']}: {hosts}")
    # image
    repo = req_info.get("image_repo")
    if "image" in req_info.get("allowed", set()) and repo:
        img = {"repository": repo}
        if req_info.get("image_tag"):
            img["tag"] = req_info["image_tag"]
        _set_path(doc, paths["image"], img)
        if verbose:
            print(f"[gen] ensured image at {paths['image']}: {repo}")

def _prune_to_allowed_and_schema(doc: Dict, allowed: set[str], union: Dict[str, set], verbose: bool = False) -> Dict:  # NEW
    """
    Keep only allowed top-level keys and known child keys from the union schema.
    """
    if not isinstance(doc, dict):
        return {}
    pruned = {}
    for top, val in doc.items():
        if allowed and top not in allowed:
            continue
        if top not in union:
            # keep unknown top if explicitly allowed; else drop
            if allowed and top in allowed:
                pruned[top] = val
            continue
        if isinstance(val, dict):
            allowed_children = union.get(top, set())
            pruned[top] = {}
            for k, v in val.items():
                if not allowed_children or k in allowed_children or not isinstance(v, dict):
                    pruned[top][k] = v
            # drop empty dicts
            if pruned[top] == {}:
                del pruned[top]
        else:
            pruned[top] = val
    if verbose:
        kept = ", ".join(pruned.keys())
        print(f"[gen] pruned top-level keys to: {kept if kept else '(none)'}")
    return pruned

def _dump_yaml(doc: Dict) -> str:  # NEW
    return yaml.safe_dump(doc, sort_keys=False)

def _ollama_generate(prompt: str, ctx: str, verbose: bool = False) -> str | None:  # CHANGED
    """
    Use shared OllamaClient (short connect/read timeouts, preflight, retries).
    """
    client = OllamaClient()
    prefer = os.getenv("OLLAMA_MODE", "generate").lower()
    verbose_flag = bool(verbose) or os.getenv("RAG_DEBUG") == "1" or os.getenv("GEN_VERBOSE") == "1"
    return client.complete(GEN_SYS_PROMPT, prompt, prefer=prefer, temperature=0.2, verbose=verbose_flag)

def _minimal_template(req_info: Dict, paths: Dict[str, str]) -> str:  # NEW
    """
    Minimal YAML strictly from requested intents; does not depend on external modules.
    """
    allowed = req_info.get("allowed", set())
    m = req_info.get("metrics") or {}
    hosts = req_info.get("hosts") or []
    readiness_path = req_info.get("readiness_path")
    repo = req_info.get("image_repo")
    tag = req_info.get("image_tag")
    doc: Dict[str, any] = {}
    # image
    if "image" in allowed and repo:
        node = {"repository": repo}
        if tag:
            node["tag"] = tag
        _set_path(doc, paths.get("image", "image"), node)
    # metrics
    if "global" in allowed and (m.get("path") or (m.get("port") is not None)):
        mon = {"activate": True}
        if m.get("path"):
            mon["path"] = m["path"]
        if m.get("port") is not None:
            mon["port"] = m["port"]
        _set_path(doc, paths.get("metrics", "global.app.monitoring"), mon)
    # ingresses
    if "global" in allowed and hosts:
        _set_path(doc, paths.get("ingresses", "global.ingresses.main.hosts"), hosts)
    # probes
    if "probes" in allowed and readiness_path:
        _set_path(doc, paths.get("readiness", "probes.readiness"), {"activate": True, "path": readiness_path})
    # configMaps (optional; emit if requested and path present in req_info)
    if "configMaps" in allowed and req_info.get("configmap_path"):
        fpath = str(req_info["configmap_path"])
        if "/" in fpath:
            d, fn = fpath.rsplit("/", 1)
        else:
            d, fn = ("/etc", fpath)
        name = "metrics" if "metrics" in fn.lower() else "config"
        doc.setdefault("configMaps", {})
        doc["configMaps"][name] = {
            "triggerReload": True,
            "mountPath": d or "/etc",
            "subPath": fn or "config.conf",
            "content": {
                (fn or "config.conf"): "# generated by minimal template; replace with real content"
            }
        }
    return _dump_yaml(doc).strip()

def generate_yaml(requirement: str,
                  retrieved: List[Tuple[float, Dict, str]],
                  schema_summary: Dict[str, Dict],
                  verbose: bool = False) -> str:
    # Use the first schema as representative for fallback
    any_schema = next(iter(schema_summary.values())) if schema_summary else {}

    # Detect allowed keys/features from requirement
    req_info = _detect_allowed_keys(requirement)
    allowed = req_info["allowed"]
    if verbose:
        print(f"[gen] allowed keys: {sorted(list(allowed))}")
        if req_info.get("metrics"):
            print(f"[gen] metrics: {req_info['metrics']}")
        if req_info.get("hosts"):
            print(f"[gen] hosts: {req_info['hosts']}")
        if req_info.get("readiness_path"):
            print(f"[gen] readiness_path: {req_info['readiness_path']}")
        if req_info.get("image_repo"):
            tag = req_info.get("image_tag")
            print(f"[gen] image: {req_info['image_repo']}{(':'+tag) if tag else ''}")

    # Build schema union and canonical paths
    union = _schema_union(schema_summary)
    paths = _canonical_paths(union)

    # Build a compact, intent-focused context to keep prompts fast
    ctx = compose_context(retrieved, schema_summary, req_info, verbose=verbose)  # NEW

    # LLM-first prompt
    llm_prompt = _build_llm_prompt(
        ctx=ctx,  # CHANGED
        requirement=requirement,
        req_info=req_info,
        union=union,
        paths=paths
    )

    if verbose:  # NEW
        lines = llm_prompt.count("\n") + 1
        print(f"[gen] prompt size: {len(llm_prompt)} chars, {lines} lines")

    # Robust retry with backoff and timing logs
    max_retries = max(1, int(os.getenv("OLLAMA_MAX_RETRIES", "3")))  # NEW
    delay = 1.0  # seconds
    out = None
    for attempt in range(1, max_retries + 1):
        t0 = time.monotonic()
        try:
            out = _ollama_generate(llm_prompt, "", verbose=verbose)
            elapsed = time.monotonic() - t0
            if out:
                if verbose:
                    print(f"[gen] llm attempt {attempt}/{max_retries} ok in {elapsed:.2f}s")
                break
            else:
                if verbose:
                    print(f"[gen] llm attempt {attempt}/{max_retries} empty response in {elapsed:.2f}s")
        except requests.exceptions.Timeout as e:
            elapsed = time.monotonic() - t0
            if verbose:
                print(f"[gen] llm attempt {attempt}/{max_retries} timeout after {elapsed:.2f}s: {e}")
        except requests.exceptions.RequestException as e:
            elapsed = time.monotonic() - t0
            if verbose:
                print(f"[gen] llm attempt {attempt}/{max_retries} request error after {elapsed:.2f}s: {e}")
        except Exception as e:
            elapsed = time.monotonic() - t0
            if verbose:
                print(f"[gen] llm attempt {attempt}/{max_retries} failed after {elapsed:.2f}s: {e}")
        if attempt < max_retries:
            time.sleep(delay)
            delay *= 2.0

    if out:
        if verbose:
            print("[gen] used: ollama")
            print(f"[gen] raw llm preview: {(out[:300] + '...') if len(out) > 300 else out}")
        raw_yaml = _extract_yaml(out)
        if verbose and raw_yaml and raw_yaml != out:
            print(f"[gen] extracted yaml preview: {(raw_yaml[:300] + '...') if len(raw_yaml) > 300 else raw_yaml}")

        raw_text = (raw_yaml or out or "").strip()

        # LENIENT mode: if Helm templates are present, preserve raw YAML and skip parsing/pruning
        lenient = os.getenv("LENIENT_YAML", "1").lower() not in ("0", "false", "no")
        if lenient and _has_helm_templates(raw_text):
            if verbose:
                print("[gen] lenient mode active and Helm templates detected -> returning raw YAML unchanged")
            return raw_text

        # Strict parse path with sanitization and multi-doc merge
        doc = _parse_yaml(raw_text, verbose=verbose)
        if doc is None:
            if verbose:
                print("[gen] parsing failed; preserving raw YAML (no fallback)")
            return raw_text  # keep full output instead of minimal fallback

        # prune to requested and schema-known keys
        pruned = _prune_to_allowed_and_schema(doc, allowed, union, verbose=verbose)
        # ensure explicitly requested parts exist
        _ensure_required_parts(pruned, req_info, paths, verbose=verbose)
        dumped = _dump_yaml(pruned).strip()
        if verbose:
            print(f"[gen] output length: {len(dumped)} chars")
        return dumped

    # Final fallback path (never crash)
    allow_fb = os.getenv("GEN_ALLOW_FALLBACK", "1").lower() not in ("0", "false", "no")  # NEW
    if verbose:
        print("[gen] LLM generation failed after retries")
        print(f"[gen] fallback allowed: {allow_fb}")
    if allow_fb:
        warn = "[gen] WARNING: using minimal template fallback due to LLM failure"
        print(warn)
        return _minimal_template(req_info, paths)
    # Fallback disabled: still avoid crashing; return a minimal stub
    print("[gen] ERROR: LLM generation failed and fallback is disabled; returning empty YAML")
    return ""
