import os
import re
import time
from typing import Dict, List, Tuple
import requests
from requests.adapters import HTTPAdapter
import yaml
from .context import compose_context

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

def _ollama_mode() -> str:
    # Default to generate (lower overhead than chat)
    return os.getenv("OLLAMA_MODE", "generate").lower()

def _ollama_base_url() -> str:
    return os.getenv("OLLAMA_HOST", os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")

def _ollama_model() -> str:
    # Fast local default; override with OLLAMA_MODEL
    return os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")

def _ollama_options(temperature: float = 0.2) -> Dict:
    # Speed-leaning defaults; all overridable via env
    return {
        "temperature": float(os.getenv("OLLAMA_TEMPERATURE", str(temperature))),
        "top_p": float(os.getenv("OLLAMA_TOP_P", "0.8")),
        "top_k": int(os.getenv("OLLAMA_TOP_K", "20")),
        "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "1536")),
        "num_predict": int(os.getenv("OLLAMA_NUM_PREDICT", "256")),
        "seed": int(os.getenv("OLLAMA_SEED", "-1")),  # -1=random
    }

_SESSION = None  # type: requests.Session | None

def _get_session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        s = requests.Session()
        adapter = HTTPAdapter(pool_connections=4, pool_maxsize=8, max_retries=0)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        _SESSION = s
    return _SESSION

def _ollama_http(endpoint: str, payload: Dict, timeout: Tuple[float, float], verbose: bool = False, kind: str | None = None) -> Dict | None:
    base = _ollama_base_url()

    def endpoints() -> List[str]:
        eps: List[str] = []
        if endpoint:
            eps.append(endpoint)
        if kind == "chat":
            eps += ["/api/chat", "/v1/chat/completions"]
        elif kind == "generate":
            eps += ["/api/generate", "/generate", "/v1/completions"]
        # de-dup while preserving order
        seen, out = set(), []
        for e in eps:
            if e not in seen:
                out.append(e)
                seen.add(e)
        return out

    tried: List[str] = []
    for ep in endpoints():
        tried.append(ep)
        url = f"{base}{ep if ep.startswith('/') else '/' + ep}"
        try:
            resp = _get_session().post(url, json=payload, timeout=timeout)
            if resp.status_code == 404:
                if verbose:
                    print(f"[gen] ollama 404 at {url}, trying fallback...")
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            sc = getattr(e.response, "status_code", None) if getattr(e, "response", None) is not None else None
            if sc == 404:
                if verbose:
                    print(f"[gen] ollama 404 at {url}, trying fallback...")
                continue
            if verbose:
                print(f"[gen] ollama http error: {e}")
            return None
        except requests.exceptions.RequestException as e:
            if verbose:
                print(f"[gen] ollama http error: {e}")
            return None
    if verbose:
        print(f"[gen] ollama: all endpoints failed (tried {', '.join(tried)})")
    return None

# NEW: warm-up helper to load the model with a 1-token generate
def _ollama_warmup(model: str, verbose: bool = False) -> None:
    try:
        opts = _ollama_options()
        opts["num_predict"] = 1
        payload = {
            "model": model,
            "prompt": "ok",
            "stream": False,
            "options": opts,
            "keep_alive": os.getenv("OLLAMA_KEEP_ALIVE", "30m"),
        }
        # allow longer warm-up read timeout for model load
        t_connect = float(os.getenv("OLLAMA_CONNECT_TIMEOUT", "0.2"))
        t_read = float(os.getenv("OLLAMA_WARMUP_TIMEOUT", "180"))
        _ = _ollama_http("/api/generate", payload, (t_connect, t_read), verbose=verbose, kind="generate")
    except Exception as e:
        if verbose:
            print(f"[gen] warmup error (ignored): {e}")

def _detect_allowed_keys(requirement: str) -> Dict[str, any]:
    """
    Tighter heuristics: pick nginx image correctly; prefer readiness=/ready and metrics=/metrics;
    extract configMap file path.
    """
    txt = requirement.lower()
    allowed = set()

    # metrics (prefer explicit /metrics, avoid *.conf)
    metrics = None
    if "metrics" in txt or "monitoring" in txt:
        path = None
        m_explicit = re.search(r'(?<![A-Za-z0-9._/-])/(metrics(?:/[A-Za-z0-9._-]+)*)\b', requirement, re.IGNORECASE)
        m_hint = re.search(r'metrics[^/.,\n\r]*?(?:path|at)\s*(/[-/a-z0-9._]+)', requirement, re.IGNORECASE)
        if m_explicit:
            path = "/" + m_explicit.group(1).lstrip("/")
        elif m_hint:
            path = m_hint.group(1)
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
        p = re.search(r'(/[-/a-z0-9._]+\.conf)\b', requirement, re.IGNORECASE) or re.search(r'(/[-/a-z0-9._]+)\b', requirement, re.IGNORECASE)
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
        "configmap_path": configmap_path,
    }

_KEY_RE = re.compile(r'^([A-Za-z0-9._-]+):(?:\s|$)')

def _filter_yaml_top_sections(yaml_text: str, allowed: set[str]) -> str:
    lines = yaml_text.splitlines()
    out = []
    skip = False
    for line in lines:
        if line and not line.startswith((" ", "\t")) and not line.lstrip().startswith("#"):
            m = _KEY_RE.match(line)
            if m:
                key = m.group(1)
                skip = key not in allowed
        if not skip:
            out.append(line)
    return "\n".join(out).strip()

def _schema_union(schema_summary: Dict[str, Dict]) -> Dict[str, set]:
    union: Dict[str, set] = {}
    for _, outline in (schema_summary or {}).items():
        for top, children in outline.items():
            if top not in union:
                union[top] = set()
            for c in children:
                union[top].add(c)
    return union

def _canonical_paths(union: Dict[str, set]) -> Dict[str, str]:
    paths = {}
    # metrics
    if "global" in union and "app" in union["global"]:
        paths["metrics"] = "global.app.monitoring"
    elif "global" in union:
        paths["metrics"] = "global.monitoring"
    else:
        paths["metrics"] = "global.app.monitoring"
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

def _build_llm_prompt(ctx: str, requirement: str, req_info: Dict, union: Dict[str, set], paths: Dict[str, str]) -> str:
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

def _first_yaml_block(text: str) -> str | None:
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        s = line.lstrip()
        if _KEY_RE.match(s):
            start = i
            break
    if start is None:
        return None
    return "\n".join(lines[start:]).strip() or None

def _extract_yaml(text: str) -> str | None:
    if not text:
        return None
    m = re.search(r"```(?:yaml|yml)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    block = _first_yaml_block(text)
    if block:
        return block
    return text.strip()

def _has_helm_templates(text: str) -> bool:
    return bool(text) and ("{{" in text and "}}" in text)

def _parse_yaml(text: str, verbose: bool = False) -> Dict | None:
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
        docs = list(yaml.safe_load_all(clean))
        if docs:
            merged = _merge_docs(docs)
            if merged is not None:
                return merged
        doc = yaml.safe_load(clean)
        return doc if isinstance(doc, dict) else None
    except Exception as e:
        if verbose:
            preview = (text[:300] + "...") if len(text) > 300 else text
            print(f"[gen] yaml parse error: {e}\n[gen] yaml preview:\n{preview}")
        try:
            clean = _strip_fences(text)
            sanitized = _sanitize(clean)
            if verbose:
                prev = sanitized[:1000] + ("..." if len(sanitized) > 1000 else "")
                print("[gen] sanitized YAML for retry:\n" + prev)
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

def _set_path(doc: Dict, dotted: str, value) -> None:
    cur = doc
    parts = dotted.split(".")
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value

def _ensure_required_parts(doc: Dict, req_info: Dict, paths: Dict[str, str], verbose: bool = False) -> None:
    m = req_info.get("metrics")
    if m and (m.get("path") or m.get("port") is not None):
        mon_path = paths["metrics"]
        node: Dict[str, any] = {"activate": True}
        if m.get("path"):
            node["path"] = m["path"]
        if m.get("port") is not None:
            node["port"] = m["port"]
        _set_path(doc, mon_path, node)
        if verbose:
            print(f"[gen] ensured metrics at {mon_path}")
    hosts = req_info.get("hosts") or []
    if hosts:
        _set_path(doc, paths["ingresses"], hosts)
        if verbose:
            print(f"[gen] ensured hosts at {paths['ingresses']}: {hosts}")
    repo = req_info.get("image_repo")
    if "image" in req_info.get("allowed", set()) and repo:
        img = {"repository": repo}
        if req_info.get("image_tag"):
            img["tag"] = req_info["image_tag"]
        _set_path(doc, paths["image"], img)
        if verbose:
            print(f"[gen] ensured image at {paths['image']}: {repo}")

def _prune_to_allowed_and_schema(doc: Dict, allowed: set[str], union: Dict[str, set], verbose: bool = False) -> Dict:
    if not isinstance(doc, dict):
        return {}
    pruned = {}
    for top, val in doc.items():
        if allowed and top not in allowed:
            continue
        if top not in union:
            if allowed and top in allowed:
                pruned[top] = val
            continue
        if isinstance(val, dict):
            allowed_children = union.get(top, set())
            pruned[top] = {}
            for k, v in val.items():
                if not allowed_children or k in allowed_children or not isinstance(v, dict):
                    pruned[top][k] = v
            if pruned[top] == {}:
                del pruned[top]
        else:
            pruned[top] = val
    if verbose:
        kept = ", ".join(pruned.keys())
        print(f"[gen] pruned top-level keys to: {kept if kept else '(none)'}")
    return pruned

def _dump_yaml(doc: Dict) -> str:
    return yaml.safe_dump(doc, sort_keys=False)

def _ollama_generate(prompt: str, ctx: str, verbose: bool = False) -> str | None:
    """
    Call local Ollama HTTP API directly (fast path) with small timeouts and no streaming.
    Respects OLLAMA_MODE=(chat|generate|auto) and OLLAMA_MODEL.
    """
    mode = _ollama_mode()
    model = _ollama_model()
    opts = _ollama_options(temperature=0.2)
    # CHANGED: increase default read timeout to tolerate cold-start loads
    t_connect = float(os.getenv("OLLAMA_CONNECT_TIMEOUT", "0.2"))
    t_read = float(os.getenv("OLLAMA_READ_TIMEOUT", "120"))
    timeout = (t_connect, t_read)

    def call_chat() -> str | None:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": GEN_SYS_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": opts,
            "keep_alive": os.getenv("OLLAMA_KEEP_ALIVE", "30m"),
        }
        data = _ollama_http("/api/chat", payload, timeout, verbose=verbose, kind="chat")
        if data and "message" in data and isinstance(data["message"], dict):
            return data["message"].get("content")
        if data and "choices" in data and isinstance(data["choices"], list) and data["choices"]:
            msg = data["choices"][0].get("message") or {}
            return msg.get("content")
        return None

    def call_generate() -> str | None:
        payload = {
            "model": model,
            "prompt": f"{GEN_SYS_PROMPT}\n\n{prompt}",
            "stream": False,
            "options": opts,
            "keep_alive": os.getenv("OLLAMA_KEEP_ALIVE", "30m"),
        }
        data = _ollama_http("/api/generate", payload, timeout, verbose=verbose, kind="generate")
        if data and "response" in data:
            return data["response"]
        if data and "choices" in data and isinstance(data["choices"], list) and data["choices"]:
            txt = data["choices"][0].get("text")
            if isinstance(txt, str) and txt:
                return txt
        return None

    def try_once() -> str | None:
        if mode == "chat":
            return call_chat()
        if mode == "generate":
            return call_generate()
        out = call_chat()
        return out or call_generate()

    # First attempt (fast path)
    out = try_once()
    if out:
        return out

    # Warm-up and retry once if first attempt failed (likely cold start)
    if os.getenv("OLLAMA_WARMUP_ON_FAIL", "1").lower() not in ("0", "false", "no"):
        if verbose:
            print("[gen] ollama: warming up model and retrying once...")
        _ollama_warmup(model, verbose=verbose)
        return try_once()

    return None

def _minimal_template(req_info: Dict, paths: Dict[str, str]) -> str:
    allowed = req_info.get("allowed", set())
    m = req_info.get("metrics") or {}
    hosts = req_info.get("hosts") or []
    readiness_path = req_info.get("readiness_path")
    repo = req_info.get("image_repo")
    tag = req_info.get("image_tag")
    doc: Dict[str, any] = {}
    if "image" in allowed and repo:
        node = {"repository": repo}
        if tag:
            node["tag"] = tag
        _set_path(doc, paths.get("image", "image"), node)
    if "global" in allowed and (m.get("path") or (m.get("port") is not None)):
        mon = {"activate": True}
        if m.get("path"):
            mon["path"] = m["path"]
        if m.get("port") is not None:
            mon["port"] = m["port"]
        _set_path(doc, paths.get("metrics", "global.app.monitoring"), mon)
    if "global" in allowed and hosts:
        _set_path(doc, paths.get("ingresses", "global.ingresses.main.hosts"), hosts)
    if "probes" in allowed and readiness_path:
        _set_path(doc, paths.get("readiness", "probes.readiness"), {"activate": True, "path": readiness_path})
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
            "content": {fn or "config.conf": "# generated by minimal template; replace with real content"},
        }
    return _dump_yaml(doc).strip()

def generate_yaml(requirement: str,
                  retrieved: List[Tuple[float, Dict, str]],
                  schema_summary: Dict[str, Dict],
                  verbose: bool = False) -> str:
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

    union = _schema_union(schema_summary)
    paths = _canonical_paths(union)

    ctx = compose_context(retrieved, schema_summary, req_info, verbose=verbose)

    llm_prompt = _build_llm_prompt(
        ctx=ctx,
        requirement=requirement,
        req_info=req_info,
        union=union,
        paths=paths
    )

    if verbose:
        lines = llm_prompt.count("\n") + 1
        print(f"[gen] prompt size: {len(llm_prompt)} chars, {lines} lines")

    # Fast retries (default 1)
    max_retries = max(1, int(os.getenv("OLLAMA_MAX_RETRIES", "1")))
    delay = 0.25
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

        lenient = os.getenv("LENIENT_YAML", "1").lower() not in ("0", "false", "no")
        if lenient and _has_helm_templates(raw_text):
            if verbose:
                print("[gen] lenient mode active and Helm templates detected -> returning raw YAML unchanged")
            return raw_text

        doc = _parse_yaml(raw_text, verbose=verbose)
        if doc is None:
            if verbose:
                print("[gen] parsing failed; preserving raw YAML (no fallback)")
            return raw_text

        pruned = _prune_to_allowed_and_schema(doc, allowed, union, verbose=verbose)
        _ensure_required_parts(pruned, req_info, paths, verbose=verbose)
        dumped = _dump_yaml(pruned).strip()
        if verbose:
            print(f"[gen] output length: {len(dumped)} chars")
        return dumped

    allow_fb = os.getenv("GEN_ALLOW_FALLBACK", "1").lower() not in ("0", "false", "no")
    if verbose:
        print("[gen] LLM generation failed after retries")
        print(f"[gen] fallback allowed: {allow_fb}")
    if allow_fb:
        warn = "[gen] WARNING: using minimal template fallback due to LLM failure"
        print(warn)
        return _minimal_template(req_info, paths)
    print("[gen] ERROR: LLM generation failed and fallback is disabled; returning empty YAML")
    return ""
