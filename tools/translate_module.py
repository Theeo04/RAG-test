# tools/translate_module.py
# NL -> {intents (values), manifests (k8s)} folosind planul (values_files + manifest_files).
# Integrare LLM: Ollama (/api/chat). Fallback: reguli generale, data-driven (fără bias de domeniu).
# Guardrails: acceptă doar chei din plan (sau allow_new_keys=True). La manifeste, doar Kinds din plan
# (altfel merg în "suggested_new_kinds").

from __future__ import annotations
import os, re, json, time
from typing import List, Dict, Any, Tuple, Set, Optional
import requests

# data-driven matching (fără liste hardcodate de domeniu)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# ============================= Colectare din plan =============================
def _collect_from_plan(plan: Dict[str, Any]) -> Tuple[Set[str], Dict[str, str], Set[str]]:
    """
    Returnează:
      - candidate_keys: set cu toate cheile permise din values_files
      - key2file: map key -> primul fișier values unde apare (hint routing)
      - allowed_kinds: set cu K8s kinds detectate deja în componentă (din manifest_files)
    """
    keys: Set[str] = set()
    key2file: Dict[str, str] = {}

    for vf in plan.get("values_files", []) or []:
        vpath = vf.get("path")
        for k in vf.get("candidate_keys", []) or []:
            if not k:
                continue
            keys.add(k)
            key2file.setdefault(k, vpath)

    kinds: Set[str] = set()
    for mf in plan.get("manifest_files", []) or []:
        for k in mf.get("kinds", []) or []:
            kinds.add(str(k))
    return keys, key2file, kinds

# ============================= Normalizări valori =============================
_CPU_NUMBER = re.compile(r"^\d+(\.\d+)?$")   # 0.5, 1, 2.0
_CPU_MILLI  = re.compile(r"^\d+m$")          # 500m
_MEM_IEC    = re.compile(r"^\d+(Ei|Pi|Ti|Gi|Mi|Ki)$", re.IGNORECASE)  # 256Mi etc.
_NUMBER     = re.compile(r"^\d+$")

def _normalize_cpu_value(s: str) -> str:
    """
    Acceptă 0.5 / 1 / 500m -> normalizează la:
      - 0.5 => 500m
      - 1   => 1
      - 2.5 => 2500m
      - 500m => 500m
    """
    s = str(s).strip().lower()
    if _CPU_MILLI.fullmatch(s):
        return s
    if _CPU_NUMBER.fullmatch(s):
        v = float(s)
        if 0.0 < v < 1.0:
            return f"{int(round(v * 1000))}m"
        if v.is_integer():
            return str(int(v))
        return f"{int(round(v * 1000))}m"
    return s

def _normalize_mem_value(s: str) -> str:
    """
    Acceptă 256Mi / 1Gi / 1024 (-> 1024) și lasă unitățile IEC neschimbate.
    """
    t = str(s).strip()
    if _MEM_IEC.fullmatch(t):
        return t
    if _NUMBER.fullmatch(t):
        return t
    m = re.match(r"^(\d+)([KMGTE]i)$", t, re.IGNORECASE)
    if m:
        return f"{m.group(1)}{m.group(2)}"
    return t

# ============================= Utilitare intenții =============================
def _dedup_keep_last(objs: List[Dict[str, Any]], by: Tuple[str, str]=("op","key")) -> List[Dict[str, Any]]:
    seen: Dict[Tuple[str,str], Dict[str,Any]] = {}
    for it in objs:
        k = (str(it.get(by[0]) or ""), str(it.get(by[1]) or ""))
        seen[k] = it
    return list(seen.values())

def _maybe_pick_key(candidates: Set[str], hint: str) -> Optional[str]:
    """
    Rapid lexical: alege o cheie plauzibilă ce conține toate token-urile din 'hint'.
    Preferă potriviri mai scurte/stabile.
    """
    toks = [t for t in re.split(r"[^a-z0-9]+", hint.lower()) if t]
    if not toks:
        return None
    hits = []
    for k in candidates:
        lk = k.lower()
        if all(t in lk for t in toks):
            hits.append(k)
    if not hits:
        return None
    hits.sort(key=lambda x: (len(x), x))
    return hits[0]

# ============================= Data-driven matching (semantic) =============================
def _normalize_key_for_text(k: str) -> str:
    """
    'global.ingresses.main.tls.certificate' -> 'global ingresses main tls certificate'
    camelCase -> camel Case; bun pentru n-grame de caractere.
    """
    k = re.sub(r"([a-z])([A-Z])", r"\1 \2", k)  # camelCase -> camel Case
    k = k.replace("_", " ").replace("-", " ").replace(".", " ")
    k = re.sub(r"\s+", " ", k).strip().lower()
    return k

def _best_key_semantic(query: str, candidates: Set[str], must_endwith: Optional[str] = None) -> Optional[str]:
    """
    Alege cea mai apropiată cheie din 'candidates' față de 'query' pe baza TF-IDF de n-grame de caractere.
    Dacă 'must_endwith' este dat (ex: '.enabled'), filtrează întâi candidații.
    """
    cand_list = sorted(candidates)
    if must_endwith:
        cand_list = [k for k in cand_list if k.endswith(must_endwith)]
    if not cand_list:
        return None

    corpus = [_normalize_key_for_text(k) for k in cand_list]
    q_txt  = _normalize_key_for_text(query)

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=1)
    X = vec.fit_transform(corpus + [q_txt])  # [N + 1, D]
    sims = linear_kernel(X[-1], X[:-1]).ravel()  # cos similarity
    idx = int(sims.argmax())
    return cand_list[idx]

def _polarity_from_query(query_lc: str) -> Optional[int]:
    """
    +1 = ON/ENABLE, -1 = OFF/DISABLE, None = necunoscut.
    Cuvintele vin din ENV (customizabile fără cod):
      RAG_BOOL_ON="enable,on,true,activate,start,up,yes,activeaza,porni,pornește"
      RAG_BOOL_OFF="disable,off,false,deactivate,stop,down,no,dezactiveaza,opreste,opri"
    """
    on_words  = set(os.getenv("RAG_BOOL_ON",  "enable,on,true,activate,start,up,yes,activeaza,porni,pornește").lower().split(","))
    off_words = set(os.getenv("RAG_BOOL_OFF", "disable,off,false,deactivate,stop,down,no,dezactiveaza,opreste,opri").lower().split(","))
    tokens = set(t for t in re.split(r"[^a-z0-9]+", query_lc) if t)
    if tokens & on_words:
        return +1
    if tokens & off_words:
        return -1
    return None

# ============================= Reguli fallback general (data-driven) =============================
_IMG_RE = re.compile(r'\b([a-z0-9][a-z0-9\-_.\/]*):(\d[\w.\-]*)\b', re.IGNORECASE)   # nginx:1.29.1
_ENV_PAIR_RE = re.compile(r'\b([A-Z0-9_]{2,})\s*=\s*([^\s,;]+)')  # FOO=bar
_HOST_RE = re.compile(r'\b([a-z0-9.-]+\.[a-z]{2,})\b', re.IGNORECASE)

def _rule_based(query: str,
                candidates: Set[str],
                allowed_kinds: Set[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    """
    Produce:
      - intents (values)
      - manifests (scaffold/sugestii)
      - notes
    Fără cuvinte-cheie de domeniu; se bazează pe patternuri uzuale și potrivire data-driven.
    """
    q = query.strip()
    ql = q.lower()
    intents: List[Dict[str, Any]] = []
    manifests: List[Dict[str, Any]] = []
    notes: List[str] = []

    # --- Detectare "create <workload>" generic (din ENV configurabil) ---
    create_words = set(os.getenv("RAG_CREATE_WORDS", "creeaza,creaza,creați,create,adauga,adaugă,add,genereaza,generează").lower().split(","))
    creating = any(w and w in ql for w in create_words)

    kinds_words = [w.strip().lower() for w in os.getenv("RAG_WORKLOAD_KINDS", "deployment,statefulset,daemonset,job,cronjob").split(",") if w.strip()]
    kind_word = None
    for kw in kinds_words:
        if re.search(rf"\b{re.escape(kw)}\b", ql):
            kind_word = kw.capitalize() if kw != "cronjob" else "CronJob"
            break
    target_kind = kind_word or ("Deployment" if creating else "")

    name_match = re.search(r"\b(?:named|nume|name)\s*[:=]?\s*([a-z0-9-_.]+)", ql)
    image_match = re.search(r"\bimage\s*[:=]\s*([a-z0-9./:_-]+)", q, flags=re.IGNORECASE)
    init_hint = ("initcontainer" in ql) or ("init container" in ql) or ("init-cont" in ql)

    default_image = os.getenv("RAG_DEFAULT_IMAGE", "nginx:latest")
    # fallback semantic: dacă se menționează „nginx” dar nu există image, presupunem default_image
    if "nginx" in ql and not image_match:
        image_match = re.search(r"(nginx)(?::([a-z0-9._-]+))?", default_image, flags=re.IGNORECASE)

    if creating and target_kind:
        if allowed_kinds and target_kind not in allowed_kinds:
            notes.append(f"kind '{target_kind}' nu există în componentă; marcat ca suggested_new_kinds")
            manifests.append({
                "op": "suggest",
                "kind": target_kind,
                "name": (name_match.group(1) if name_match else "app"),
                "params": {
                    "image": (image_match.group(0) if image_match else default_image),
                    "withInitContainer": bool(init_hint),
                    "replicas": 1
                },
                "reason": "kind-not-present"
            })
        else:
            manifests.append({
                "op": "create",
                "kind": target_kind,
                "name": (name_match.group(1) if name_match else "app"),
                "params": {
                    "image": (image_match.group(0) if image_match else default_image),
                    "withInitContainer": bool(init_hint),
                    "replicas": 1
                }
            })

    # --- Replicas (dinamically pick from candidates) ---
    m_rep = re.search(r"\breplicas?\s*(?:=|:)?\s*(\d+)\b", ql)
    if m_rep:
        val = m_rep.group(1)
        # prefer keys that contain 'replica' anywhere
        replica_keys = [k for k in candidates if "replica" in k.lower()]
        pick = _best_key_semantic("replicas", set(replica_keys)) if replica_keys else _maybe_pick_key(candidates, "replica")
        if pick and pick in candidates:
            intents.append({"op": "set", "key": pick, "value": val})
        else:
            # fallback configurable
            fallback_replica_keys = [k.strip() for k in os.getenv("RAG_REPLICA_KEYS", "replicaCount,replicas").split(",") if k.strip()]
            for rk in fallback_replica_keys:
                intents.append({"op": "set", "key": rk, "value": val})
            notes.append("no replica keys in plan; used fallback RAG_REPLICA_KEYS")

    # --- CPU (dinamically: all matching *.cpu in candidates) ---
    if "cpu" in ql:
        m_cpu = re.search(r"\bcpu[^0-9a-zA-Z]{0,3}(\d+m|\d+(?:\.\d+)?)", ql)
        if not m_cpu:
            m_cpu = re.search(r"\b(\d+m)\b", ql)
        if m_cpu:
            cpu_val = _normalize_cpu_value(m_cpu.group(1))
            cpu_keys = [k for k in candidates if k.lower().endswith(".cpu")]
            req_cpu = [k for k in cpu_keys if ".requests." in k.lower()]
            lim_cpu = [k for k in cpu_keys if ".limits." in k.lower()]
            chosen = req_cpu + lim_cpu if (req_cpu or lim_cpu) else cpu_keys
            if chosen:
                for ck in sorted(set(chosen)):
                    intents.append({"op": "set", "key": ck, "value": cpu_val})
            else:
                # fallback configurable
                fallback_cpu_keys = [k.strip() for k in os.getenv("RAG_CPU_KEYS", "resources.requests.cpu,resources.limits.cpu").split(",") if k.strip()]
                for ck in fallback_cpu_keys:
                    intents.append({"op": "set", "key": ck, "value": cpu_val})
                notes.append("no CPU keys in plan; used fallback RAG_CPU_KEYS")

    # --- Memory (dinamically: all matching *.memory in candidates) ---
    if "mem" in ql or "memory" in ql:
        m_mem = re.search(r"\b(\d+(?:Ki|Mi|Gi|Ti|Pi|Ei))\b", q)
        if not m_mem:
            m_mem = re.search(r"\bmemory[^0-9a-zA-Z]{0,3}(\d+(?:Ki|Mi|Gi|Ti|Pi|Ei))\b", q, re.IGNORECASE)
        if m_mem:
            mem_val = _normalize_mem_value(m_mem.group(1))
            mem_keys = [k for k in candidates if k.lower().endswith(".memory")]
            req_mem = [k for k in mem_keys if ".requests." in k.lower()]
            lim_mem = [k for k in mem_keys if ".limits." in k.lower()]
            chosen = req_mem + lim_mem if (req_mem or lim_mem) else mem_keys
            if chosen:
                for mk in sorted(set(chosen)):
                    intents.append({"op": "set", "key": mk, "value": mem_val})
            else:
                # fallback configurable
                fallback_mem_keys = [k.strip() for k in os.getenv("RAG_MEM_KEYS", "resources.requests.memory,resources.limits.memory").split(",") if k.strip()]
                for mk in fallback_mem_keys:
                    intents.append({"op": "set", "key": mk, "value": mem_val})
                notes.append("no memory keys in plan; used fallback RAG_MEM_KEYS")

    # --- Image repo:tag (generic) ---
    m_img = image_match or _IMG_RE.search(q)
    if m_img:
        full = m_img.group(0)
        repo, tag = (full, None)
        if ":" in full:
            repo, tag = full.split(":", 1)
        key_repo = _maybe_pick_key(candidates, "image repository") or _maybe_pick_key(candidates, "image")
        key_tag  = _maybe_pick_key(candidates, "image tag") or _maybe_pick_key(candidates, "tag")
        if key_repo and key_repo in candidates:
            intents.append({"op": "set", "key": key_repo, "value": repo})
        if tag and key_tag and key_tag in candidates:
            intents.append({"op": "set", "key": key_tag, "value": tag})

    # --- Env pairs (generic) ---
    env_pairs = list(_ENV_PAIR_RE.findall(q))
    if env_pairs:
        env_key = _maybe_pick_key(candidates, "env") or _maybe_pick_key(candidates, "environment")
        if env_key and env_key in candidates:
            val = ",".join([f"{k}={v}" for k, v in env_pairs])
            intents.append({"op": "set", "key": env_key, "value": val})

    # --- Ingress host (generic) ---
    if any(w in ql for w in ["ingress", "host", "hostname"]):
        hm = _HOST_RE.search(q)
        if hm:
            host = hm.group(1)
            host_key = _maybe_pick_key(candidates, "hosts") or _maybe_pick_key(candidates, "ingress hosts")
            if host_key and host_key in candidates:
                intents.append({"op": "set", "key": host_key, "value": host})

    # --- Enable/Disable generic (DATA-DRIVEN) ---
    pol = _polarity_from_query(ql)
    if pol is not None:
        best_enabled = _best_key_semantic(q, candidates, must_endwith=".enabled")
        if best_enabled:
            intents.append({
                "op": "enable" if pol > 0 else "disable",
                "key": best_enabled
            })

    intents = _dedup_keep_last(intents)
    manifests = _dedup_keep_last(manifests, by=("op","kind"))
    notes.append("generic fallback applied (data-driven; no domain-specific keywords)")

    return intents, manifests, notes

# ============================= Integrare LLM (Ollama) =============================
class _LLMUnavailable(Exception):
    pass

def _extract_json(text: str) -> str:
    text = text.strip()
    try:
        json.loads(text); return text
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found")
    cand = m.group(0).strip()
    json.loads(cand)
    return cand

def _call_ollama(prompt: str) -> str:
    model = os.environ.get("OLLAMA_MODEL", "llama3.1")
    url   = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return ONLY a single valid JSON object. No prose."},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {"temperature": 0}
    }
    last_err = None
    for _ in range(2):
        try:
            r = requests.post(url, headers={"Content-Type":"application/json"}, data=json.dumps(body), timeout=60)
            r.raise_for_status()
            data = r.json()
            content = (data.get("message") or {}).get("content", "")
            js = _extract_json(content)
            json.loads(js)  # validate
            return js
        except Exception as e:
            last_err = e
            time.sleep(0.8)
    raise _LLMUnavailable(f"Ollama call failed: {last_err}")

def _llm_prompt(query: str, candidates: Set[str], allowed_kinds: Set[str]) -> str:
    """
    Prompt compact, strict JSON, fără bias, fără cuvinte-cheie hardcodate.
    """
    allowed_keys_list = sorted(list(candidates))[:300]
    allowed_kinds_list = sorted(list(allowed_kinds)) if allowed_kinds else []

    lines = [
        "You convert natural-language requests into configuration actions.",
        "Return ONLY valid JSON with this schema:",
        "{",
        '  "intents": [ {"op":"enable|disable|set","key":"<ALLOWED_KEY>","value":optional-string} ],',
        '  "manifests": [ {"op":"create|patch|suggest","kind":"<ALLOWED_KIND>","name":string,"params":object} ],',
        '  "notes": []',
        "}",
        f"- Keys must be chosen ONLY from this allow-list: {allowed_keys_list}",
    ]

    if allowed_kinds_list:
        lines.append(f"- Kinds must be chosen ONLY from this allow-list: {allowed_kinds_list}")
    else:
        lines.append("- If you propose a new kind not in allow-list, use op='suggest' with minimal params.")

    lines += [
        "- If CPU or memory is mentioned, set both requests and limits when possible.",
        "- Romanian input possible. Keep units as strings (e.g., '500m', '256Mi').",
        "- Prefer minimal, deterministic output. Do not include explanations.",
        f"User request: {query}",
        "Return ONLY the JSON object."
    ]

    return "\n".join(lines)

def build_llm_prompt(query: str, plan: Dict[str, Any]) -> str:
    """
    Public helper: builds the exact LLM prompt using the allow-lists from the current plan.
    Useful for debugging or for a human-readable view of what will be sent to the LLM.
    """
    candidates, _, allowed_kinds = _collect_from_plan(plan)
    return _llm_prompt(query, candidates, allowed_kinds)

# ============================= API principal =============================
def translate_intents(query: str,
                      plan: Dict[str, Any],
                      provider: str = "rule",
                      allow_new_keys: bool = False) -> Dict[str, Any]:
    """
    Returnează:
    {
      "intents":   [ {"op":"enable|disable|set","key":"...","value":"...?"}, ... ],
      "manifests": [ {"op":"create|patch|suggest","kind":"Deployment","name":"...","params":{...}}, ... ],
      "rejected":  [ {...} ],
      "notes":     [ ... ],
      "key_hint":  { key -> values_path },
      "allowed_kinds": [ ... ],
      "suggested_new_kinds": [ ... ]
    }
    """
    candidates, key2file, allowed_kinds = _collect_from_plan(plan)
    notes: List[str] = [f"{len(candidates)} candidate keys from plan",
                        f"{len(allowed_kinds)} allowed kinds from manifests"]

    intents_raw: List[Dict[str, Any]] = []
    manifests_raw: List[Dict[str, Any]] = []

    if provider == "llm":
        try:
            js = _call_ollama(_llm_prompt(query, candidates, allowed_kinds))
            data = json.loads(js)
            intents_raw   = data.get("intents", []) or []
            manifests_raw = data.get("manifests", []) or []
            if data.get("notes"):
                notes += list(map(str, data["notes"]))
        except _LLMUnavailable as e:
            notes.append(f"LLM unavailable → rule-based fallback ({e})")
            intents_raw, manifests_raw, extra = _rule_based(query, candidates, allowed_kinds)
            notes += extra
        except Exception as e:
            notes.append(f"LLM error → rule-based fallback ({e})")
            intents_raw, manifests_raw, extra = _rule_based(query, candidates, allowed_kinds)
            notes += extra
    else:
        intents_raw, manifests_raw, extra = _rule_based(query, candidates, allowed_kinds)
        notes += extra

    # --- guardrails & normalizări (values) ---
    intents: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for it in intents_raw:
        op  = (it.get("op")  or "").strip().lower()
        key = (it.get("key") or "").strip()
        if not op or not key:
            continue

        # normalizează valori uzuale
        if op == "set":
            if key.endswith(".cpu"):
                v = it.get("value")
                if v is None or str(v).strip() == "":
                    it["reason"] = "missing value"
                    rejected.append(it); continue
                it["value"] = _normalize_cpu_value(str(v))
            elif key.endswith(".memory"):
                v = it.get("value")
                if v is None or str(v).strip() == "":
                    it["reason"] = "missing value"
                    rejected.append(it); continue
                it["value"] = _normalize_mem_value(str(v))

        # guardrail chei permise
        in_candidates = (key in candidates) or any(key.startswith(p + ".") for p in candidates)
        if not in_candidates and not allow_new_keys:
            it["_new_key"] = True
            it["reason"] = "not in candidates"
            rejected.append(it); continue

        intents.append({"op": op if op in ("enable","disable","set") else "set",
                        "key": key,
                        **({"value": it["value"]} if "value" in it else {})})

    intents = _dedup_keep_last(intents)

    # --- guardrails (manifests) ---
    manifests: List[Dict[str, Any]] = []
    suggested_new_kinds: Set[str] = set()

    for mf in manifests_raw:
        op   = (mf.get("op")   or "").strip().lower()
        kind = (mf.get("kind") or "").strip()
        name = (mf.get("name") or "").strip() or "app"
        params = mf.get("params") or {}

        if not kind:
            continue

        if allowed_kinds and kind not in allowed_kinds:
            suggested_new_kinds.add(kind)
            manifests.append({
                "op": "suggest",
                "kind": kind,
                "name": name,
                "params": params,
                "reason": "kind-not-in-allowed-list"
            })
            continue

        if op not in ("create","patch","suggest"):
            op = "patch"

        manifests.append({"op": op, "kind": kind, "name": name, "params": params})

    manifests = _dedup_keep_last(manifests, by=("op","kind"))

    return {
        "intents": intents,
        "manifests": manifests,
        "rejected": rejected,
        "notes": notes,
        "key_hint": key2file,
        "allowed_kinds": sorted(list(allowed_kinds)),
        "suggested_new_kinds": sorted(list(suggested_new_kinds)) if suggested_new_kinds else []
    }
