import os
import re
import json
import time  # NEW
import requests  # NEW
from typing import Any, Dict, List, Optional, Set
from typing import Tuple  # NEW

from .llm.ollama_client import OllamaClient  # NEW

# -------- Ollama helpers (local, no dependency on generator.py) --------

def _ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

def _ollama_model() -> str:
    # reuse chat model env
    return os.getenv("OLLAMA_CHAT_MODEL", "llama3.1:latest")

def _ollama_mode() -> str:
    return os.getenv("OLLAMA_MODE", "auto").lower()  # chat|generate|auto

def _ollama_timeout() -> int:  # CHANGED
    try:
        return int(os.getenv("OLLAMA_TIMEOUT", "600"))
    except Exception:
        return 600

def _stream_post(url: str, payload: Dict, verbose: bool = False) -> Tuple[str, bool]:  # NEW
    buf: List[str] = []
    done_seen = False
    with requests.post(url, json=payload, timeout=_ollama_timeout(), stream=True) as r:
        if r.status_code != 200:
            if verbose:
                print(f"[intents] http {r.status_code}: {r.text[:300]}")
            r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                try:
                    obj = r.json()
                except Exception:
                    continue
            if "message" in obj and isinstance(obj["message"], dict):
                part = (obj["message"].get("content") or "")
                if part:
                    buf.append(part)
            if "response" in obj:
                part = (obj.get("response") or "")
                if part:
                    buf.append(part)
            if obj.get("done") is True:
                done_seen = True
    return "".join(buf), done_seen

def _llm_complete_chat(system: str, user: str, verbose: bool = False) -> Optional[str]:  # CHANGED
    url = _ollama_base_url().rstrip("/") + "/api/chat"
    payload = {
        "model": _ollama_model(),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": True,
        "options": {"temperature": 0.0},
    }
    tries = 3
    for attempt in range(1, tries + 1):
        try:
            content, done = _stream_post(url, payload, verbose=verbose)
            if content:
                return content
            if verbose:
                print(f"[intents] chat empty content attempt={attempt}")
        except Exception as e:
            if verbose:
                print(f"[intents] chat call failed attempt={attempt}: {e}")
        time.sleep(1.5 ** attempt)
    return None

def _llm_complete_generate(system: str, user: str, verbose: bool = False) -> Optional[str]:  # CHANGED
    url = _ollama_base_url().rstrip("/") + "/api/generate"
    fused = f"{system.strip()}\n\nUser:\n{user.strip()}\n"
    payload = {
        "model": _ollama_model(),
        "prompt": fused,
        "stream": True,
        "options": {"temperature": 0.0},
    }
    tries = 3
    for attempt in range(1, tries + 1):
        try:
            content, done = _stream_post(url, payload, verbose=verbose)
            if content:
                return content
            if verbose:
                print(f"[intents] generate empty content attempt={attempt}")
        except Exception as e:
            if verbose:
                print(f"[intents] generate call failed attempt={attempt}: {e}")
        time.sleep(1.5 ** attempt)
    return None

def _llm_complete(system: str, user: str, verbose: bool = False) -> Optional[str]:  # CHANGED
	# Single entry using the shared client
	client = OllamaClient()
	prefer = os.getenv("OLLAMA_MODE", "auto").lower()
	return client.complete(system, user, prefer=prefer, temperature=0.0, verbose=verbose)

# -------- JSON helpers --------

def _extract_json_block(text: str) -> Optional[str]:
    if not text:
        return None
    # Prefer fenced json/yaml blocks
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fallback: first {...} block
    m = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    return m.group(1).strip() if m else None

def _coerce_req_info(obj: Dict[str, Any]) -> Dict[str, Any]:
    allowed = set(obj.get("allowed") or [])
    # normalize fields
    metrics = obj.get("metrics") or {}
    if not isinstance(metrics, dict):
        metrics = {}
    hosts = obj.get("hosts") or []
    if not isinstance(hosts, list):
        hosts = []
    readiness_path = obj.get("readiness_path")
    image_repo = obj.get("image_repo")
    image_tag = obj.get("image_tag")
    volume_hint = obj.get("volume_hint")
    return {
        "allowed": allowed,
        "metrics": metrics if metrics else None,
        "hosts": hosts if hosts else None,
        "readiness_path": readiness_path or None,
        "image_repo": image_repo or None,
        "image_tag": image_tag or None,
        "volume_hint": volume_hint or None,
    }

# -------- Regex fallback (self-contained) --------

def _regex_extract(requirement: str) -> Dict[str, Any]:
    txt = requirement.lower()
    allowed: Set[str] = set()

    metrics = None
    if "metrics" in txt or "monitoring" in txt:
        path = None
        m1 = re.search(r'metrics[^/\n\r]*\s(at|path[: ]?)\s*(/[-/a-z0-9._]+)', requirement, re.IGNORECASE)
        if m1:
            path = m1.group(2)
        if not path:
            m2 = re.search(r'/(metrics[/-]?[a-z0-9._-]*)', requirement, re.IGNORECASE)
            if m2:
                path = "/" + m2.group(1)
        port = None
        mport = re.search(r'(?:on\s+)?port\s+(\d+)', requirement, re.IGNORECASE)
        if mport:
            port = int(mport.group(1))
        metrics = {"path": path, "port": port}
        allowed.add("global")

    hosts = None
    if re.search(r'\bhost[s]?\b', txt):
        cands: List[str] = []
        for m in re.finditer(r'\bhost[s]?\b[^A-Za-z0-9._-]*([a-z0-9.-]+\.[a-z]{2,})', requirement, re.IGNORECASE):
            cands.append(m.group(1))
        def ok(h: str) -> bool:
            return not h.lower().endswith((".yaml", ".yml")) and "." in h
        cands = [h.rstrip(".") for h in cands if ok(h)]
        hosts = list(dict.fromkeys(cands)) or None
        if hosts:
            allowed.add("global")

    readiness_path = None
    if re.search(r'\breadiness\b|\bprobe\b', txt, re.IGNORECASE):
        rp = re.search(r'/(?:[-/a-z0-9._]+)', requirement, re.IGNORECASE)
        readiness_path = rp.group(0) if rp else None
        allowed.add("probes")

    image_repo = None
    image_tag = None
    if "image" in txt:
        m = re.search(r'image(?:\s+|:)\s*([a-z0-9/_\-.]+)(?::([a-z0-9._\-]+))?', requirement, re.IGNORECASE)
        if m:
            image_repo = m.group(1)
            image_tag = m.group(2)
        allowed.add("image")
    if "nginx" in txt and "image" not in txt:
        image_repo = "nginx"
        allowed.add("image")

    volume_hint = None
    if re.search(r'\bvolume\b|\bmount\b', txt):
        allowed.add("volumes")
        mh = re.search(r'(?:mount(?:ed)?|mountpath)\s*(?:at|:)?\s*(/[-/a-z0-9._]+)', requirement, re.IGNORECASE)
        if mh:
            volume_hint = mh.group(1)

    for key, pat in [
        ("configMaps", r'\bconfigmap\b'),
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
        "volume_hint": volume_hint,
    }

# -------- LLM intents --------

_INTENTS_SYS = """You extract deployment intents from short natural language requests for Helm values/common.yaml.
Return strict JSON only, no prose. Fields:
- allowed: array of top-level sections to include (e.g., ["global","image","volumes","probes","configMaps"])
- metrics: { "path": string|null, "port": number|null } or null
- hosts: array of host strings or null
- readiness_path: string or null
- image_repo: string or null
- image_tag: string or null
- volume_hint: mountPath string or null
Only include keys requested or implied by the user. Do not invent values.
"""

def _llm_extract(requirement: str, union: Optional[Dict[str, Set[str]]], verbose: bool = False) -> Optional[Dict[str, Any]]:
    # Hint schema to bias valid keys
    tops = sorted(list((union or {}).keys()))
    hint = f"Allowed top-level keys (from schema): {', '.join(tops)}\n" if tops else ""
    user = f"{hint}User request:\n{requirement}\n\nRespond with JSON only."
    out = _llm_complete(_INTENTS_SYS, user, verbose=verbose)
    if not out:
        return None
    js = _extract_json_block(out) or out.strip()
    try:
        data = json.loads(js)
        if not isinstance(data, dict):
            return None
        return _coerce_req_info(data)
    except Exception as e:
        if verbose:
            print(f"[intents] parse json failed: {e}")
        return None

def extract_intents(requirement: str,
                    union: Optional[Dict[str, Set[str]]] = None,
                    verbose: bool = False) -> Dict[str, Any]:
    """
    LLM-backed intents with regex fallback, controlled by USE_LLM_INTENTS=1.
    """
    use_llm = os.getenv("USE_LLM_INTENTS", "1").lower() not in ("0", "false", "no")
    if use_llm:
        llm = _llm_extract(requirement, union, verbose=verbose)
        if llm:
            # small normalization: if nginx implied and image missing
            txt = requirement.lower()
            if not llm.get("image_repo") and "nginx" in txt:
                llm["image_repo"] = "nginx"
                llm["allowed"].add("image")
            return llm
        if verbose:
            print("[intents] LLM intents unavailable, using regex fallback")
    return _regex_extract(requirement)
