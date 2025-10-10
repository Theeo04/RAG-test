import os, re, json, glob, yaml, pathlib, time
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi  # NEW

# Reuse encoder shape/model from plan_module
from plan_module import DEFAULT_MODEL_NAME as ENC_MODEL_NAME
from plan_module import _encode as encode_texts  # normalized float32 embeddings

INDEX_DIR_NAME = ".rag_index"
VEC_FILE = "embeddings.npy"
TFIDF_FILE = "tfidf.joblib"
VECTORIZER_FILE = "vectorizer.joblib"
INDEX_JSON = "index.json"

_HELM_INLINE_RE = re.compile(r"\{\{[\s\S]*?\}\}")
_HELM_CTRL_LINE_RE = re.compile(r"^\s*\{\{[-#]?[\s\S]*?[-#]?\}\}\s*$")

def _strip_helm_templates(text: str) -> str:
    """
    Make Helm-templated YAML parseable:
    - drop control-only lines like '{{- if ...}}' / '{{- end }}'
    - replace inline '{{ ... }}' with a neutral placeholder '0'
    """
    lines = []
    for ln in (text or "").splitlines():
        if _HELM_CTRL_LINE_RE.match(ln):
            continue
        lines.append(_HELM_INLINE_RE.sub("0", ln))
    return "\n".join(lines)

def _flatten_yaml(obj, prefix=""):
    keys = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            kstr = f"{prefix}.{k}" if prefix else str(k)
            keys.extend(_flatten_yaml(v, kstr))
    elif isinstance(obj, list):
        # use [] suffix to denote list
        kstr = f"{prefix}[]" if prefix else "[]"
        keys.append(kstr)
        for i, v in enumerate(obj):
            keys.extend(_flatten_yaml(v, f"{prefix}[{i}]"))
    else:
        if prefix:
            keys.append(prefix)
    return keys

def _detect_profile_from_name(fname: str) -> Optional[str]:
    # values.yaml -> common; values-XXX.yaml -> XXX
    m = re.match(r"^values(?:[-_\.](?P<p>[^\.]+))?\.ya?ml$", os.path.basename(fname), re.IGNORECASE)
    if not m:
        return None
    p = m.group("p")
    return (p or "common").lower()

def _is_manifest_doc(doc: Any) -> bool:
    return isinstance(doc, dict) and ("kind" in doc) and ("apiVersion" in doc)

def _read_yaml_file(path: str) -> List[Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        # first try as-is
        try:
            docs = list(yaml.safe_load_all(raw)) or []
            return docs
        except Exception:
            # try sanitized Helm templates
            cleaned = _strip_helm_templates(raw)
            try:
                docs = list(yaml.safe_load_all(cleaned)) or []
                return docs
            except Exception:
                return []
    except Exception:
        return []

def _component_for_path(k8s_root: str, path: str) -> str:
    rel = os.path.relpath(path, k8s_root)
    parts = rel.split(os.sep)
    return parts[0] if len(parts) > 1 else "k8s"

def _extract_manifest_signals(docs: List[Any]) -> Tuple[List[str], List[str]]:
    kinds = []
    images = []
    for d in docs:
        if not _is_manifest_doc(d):
            continue
        kind = d.get("kind")
        if kind:
            kinds.append(str(kind))
        # containers images
        spec = (d.get("spec") or {})
        tpl = ((spec.get("template") or {}).get("spec") or {})
        for field in ("initContainers", "containers"):
            for c in tpl.get(field, []) or []:
                img = c.get("image")
                if img:
                    images.append(str(img))
    return sorted(set(kinds)), sorted(set(images))

def _extract_yaml_keys(docs: List[Any]) -> List[str]:
    keys = []
    for d in docs:
        # treat values-like docs (no kind/apiVersion)
        if isinstance(d, dict) and not _is_manifest_doc(d):
            keys.extend(_flatten_yaml(d))
    # normalize to dotted keys (strip indices)
    norm = []
    for k in keys:
        k = re.sub(r"\[\d+\]", "[]", k)
        k = k.replace("[].", ".")
        norm.append(k)
    uniq = sorted(set([kk for kk in norm if kk]))
    return uniq[:10000]

def _extract_keys_from_text(path: str) -> List[str]:
    """
    Fallback: derive dotted keys by indentation from raw text (Helm-tolerant).
    This is approximate but good enough to surface candidate keys.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
    except Exception:
        return []
    raw = _strip_helm_templates(raw)
    keys: List[str] = []  # FIX: was List[str> which is invalid
    stack: List[Tuple[int, str]] = []  # (indent, key)
    for line in raw.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        # skip pure list item with only '-' (no key)
        if re.match(r"^\s*-\s*$", line):
            continue
        # indent and key:
        m = re.match(r"^(\s*)(-?\s*)([A-Za-z0-9_.-]+)\s*:\s*(.*)?$", line)
        if not m:
            continue
        indent = len(m.group(1).replace("\t", "  "))
        key = m.group(3)
        # pop deeper indents
        while stack and stack[-1][0] >= indent:
            stack.pop()
        stack.append((indent, key))
        dotted = ".".join([k for _, k in stack])
        keys.append(dotted)
    return sorted(set(keys))[:10000]

def _file_type_from_docs(docs: List[Any], path: str) -> str:
    # if any manifest-like doc present -> manifest; else values
    for d in docs:
        if _is_manifest_doc(d):
            return "manifest"
    # heuristic: if name matches values* -> values
    if re.search(r"values.*\.ya?ml$", os.path.basename(path), re.IGNORECASE):
        return "values"
    return "values"

def _values_profile_for_path(path: str) -> Optional[str]:
    prof = _detect_profile_from_name(path)
    if prof:
        return prof
    # Also infer profile from values/<name>.yaml
    base = os.path.basename(path)
    parent = os.path.basename(os.path.dirname(path))
    if parent.lower() == "values":
        name, _ = os.path.splitext(base)
        return (name or "").lower() or None
    return None

def _gather_files(k8s_root: str) -> List[str]:
    return [p for p in glob.glob(os.path.join(k8s_root, "**", "*.y*ml"), recursive=True) if os.path.isfile(p)]

def _text_for_item(item: Dict[str, Any]) -> str:
    # compose text used in retrieval (add signals for values to improve match)
    parts = [f"path: {item['path']}", f"component: {item['component']}", f"type: {item['type']}"]
    if item["type"] == "values":
        if item.get("profile"):
            parts.append(f"profile: {item['profile']}")
            # field-aware lexical token
            parts.append(f"profile:{item['profile']}")
        keys = item.get("yaml_keys") or []
        if keys:
            parts.append("keys: " + " ".join(keys))
            # add prefixed key: tokens (truncated)
            key_tokens = [f"key:{k}" for k in keys[:300]]
            parts.append(" ".join(key_tokens))
        # include raw content too
        try:
            with open(item["path"], "r", encoding="utf-8") as f:
                raw = f.read()
                parts.append(raw)
        except Exception:
            pass
    else:
        kinds = item.get("k8s_kinds") or []
        imgs = item.get("k8s_images") or []
        if kinds:
            parts.append("kinds: " + " ".join(kinds))
            parts.append(" ".join(f"kind:{k}" for k in kinds))  # field-aware tokens
        if imgs:
            parts.append("images: " + " ".join(imgs))
            parts.append(" ".join(f"image:{im}" for im in imgs))  # field-aware tokens
        try:
            with open(item["path"], "r", encoding="utf-8") as f:
                parts.append(f.read())
        except Exception:
            pass
    return "\n".join(parts)

def index_k8s(root: str, k8s_subdir: str = "k8s", model_name: Optional[str] = None) -> Dict[str, Any]:
    t0 = time.time()
    k8s_root = os.path.join(root, k8s_subdir)
    if not os.path.isdir(k8s_root):
        raise RuntimeError(f"k8s folder not found: {k8s_root}")

    files = _gather_files(k8s_root)
    items: List[Dict[str, Any]] = []
    for p in files:
        docs = _read_yaml_file(p)
        t = _file_type_from_docs(docs, p)
        comp = _component_for_path(k8s_root, p)
        entry: Dict[str, Any] = {"component": comp, "type": t, "path": p}
        if t == "values":
            entry["profile"] = _values_profile_for_path(p)
            entry["yaml_keys"] = _extract_yaml_keys(docs)
        else:
            kinds, imgs = _extract_manifest_signals(docs)
            entry["k8s_kinds"] = kinds
            entry["k8s_images"] = imgs
        items.append(entry)

    texts = [_text_for_item(it) for it in items]

    # Resolve embedding model (fallback if an Ollama tag was passed)
    enc_model = (model_name or ENC_MODEL_NAME)
    if ":" in enc_model:
        enc_model = ENC_MODEL_NAME

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), lowercase=True)
    tfidf_mat = vectorizer.fit_transform(texts)
    # Embeddings (normalized)
    emb = encode_texts(texts, enc_model)

    # Persist
    idx_dir = os.path.join(root, INDEX_DIR_NAME)
    pathlib.Path(idx_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(idx_dir, VECTORIZER_FILE))
    joblib.dump(tfidf_mat, os.path.join(idx_dir, TFIDF_FILE))
    np.save(os.path.join(idx_dir, VEC_FILE), emb)
    with open(os.path.join(idx_dir, INDEX_JSON), "w", encoding="utf-8") as f:
        json.dump({
            "k8s_root": k8s_root,
            "model_name": enc_model,  # store the actual embedding model used
            "items": items,
            "created_at": time.time(),
            "counts": {"files": len(files), "items": len(items)}
        }, f, indent=2)

    return {"files_indexed": len(files), "items": len(items), "took_s": round(time.time() - t0, 3)}

def have_index_fn(root: str) -> bool:
    idx_dir = os.path.join(root, INDEX_DIR_NAME)
    return all(os.path.exists(os.path.join(idx_dir, f)) for f in [INDEX_JSON, VECTORIZER_FILE, TFIDF_FILE, VEC_FILE])

def load_index_fn(root: str):
    # Returns a 5-tuple to remain compatible with plan_module expectations.
    idx_dir = os.path.join(root, INDEX_DIR_NAME)
    with open(os.path.join(idx_dir, INDEX_JSON), "r", encoding="utf-8") as f:
        meta_json = json.load(f)
    vectorizer = joblib.load(os.path.join(idx_dir, VECTORIZER_FILE))
    tfidf_mat = joblib.load(os.path.join(idx_dir, TFIDF_FILE))
    emb = np.load(os.path.join(idx_dir, VEC_FILE))
    # plan_module only needs meta list
    meta = meta_json.get("items", [])
    stats = {"files": meta_json.get("counts", {}).get("files", 0),
             "items": meta_json.get("counts", {}).get("items", 0)}
    return (vectorizer, tfidf_mat, emb, meta, stats)

def _minmax_norm(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    mn, mx = float(arr.min()), float(arr.max())
    if mx <= mn:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

def _tokenize(text: str) -> List[str]:
    """
    Simple tokenizer that keeps field prefixes and common k8s dotted keys.
    Splits on whitespace, lowercases.
    """
    return [t.lower() for t in re.split(r"\s+", text) if t and not t.isspace()]

def _intent_flags(query: str) -> Dict[str, bool]:
    q = (query or "").lower()
    return {
        "replicas": bool(re.search(r"\breplicas?\b|\bscale\b", q)),
        "cpu": "cpu" in q or "cores" in q,
        "memory": "memory" in q or "mem" in q or "ram" in q,
        "ingress": "ingress" in q or "host" in q or "domain" in q,
        "image": "image" in q or "tag" in q or "version" in q,
    }

def _field_boost(it: Dict[str, Any], flags: Dict[str, bool]) -> float:
    """
    Small additive boost based on item signals matching intent.
    Max ~0.1
    """
    b = 0.0
    t = it.get("type")
    if t == "manifest":
        kinds = [k.lower() for k in (it.get("k8s_kinds") or [])]
        if flags["ingress"] and ("ingress" in kinds):
            b += 0.05
        if flags["replicas"] and any(k in kinds for k in ("deployment", "statefulset")):
            b += 0.03
    if t == "values":
        keys = [str(k).lower() for k in (it.get("yaml_keys") or [])]
        if flags["cpu"] and any(k.endswith(".cpu") for k in keys):
            b += 0.04
        if flags["memory"] and any(k.endswith(".memory") for k in keys):
            b += 0.04
        if flags["replicas"] and any("replica" in k for k in keys):
            b += 0.03
    if flags["image"]:
        if t == "manifest" and (it.get("k8s_images") or []):
            b += 0.04
    return b

def hybrid_search_fn(root: str,
                     query: str,
                     k: int,
                     *,
                     types: Optional[List[str]] = None,
                     component: Optional[str] = None,
                     profile: Optional[str] = None,
                     max_per_file: int = 2,
                     max_per_component: Optional[int] = 5) -> Dict[str, Any]:
    vectorizer, tfidf_mat, emb_mat, meta, _ = load_index_fn(root)
    idx_dir = os.path.join(root, INDEX_DIR_NAME)
    with open(os.path.join(idx_dir, INDEX_JSON), "r", encoding="utf-8") as f:
        meta_json = json.load(f)
    items = meta_json["items"]
    if not items:
        return {"results": [], "stats": {"searched": 0, "kept": 0}}  # guard empty index

    enc_model = meta_json.get("model_name") or ENC_MODEL_NAME  # use stored embedding model

    # Compose texts for lexical + BM25
    texts = [_text_for_item(it) for it in items]
    corpus_tokens = [_tokenize(t) for t in texts]
    bm25 = BM25Okapi(corpus_tokens)

    # Compose query reps
    q_emb = encode_texts([query], enc_model)[0]  # normalized
    q_tfidf = vectorizer.transform([query])
    q_tokens = _tokenize(query)

    # Scores
    sem = emb_mat @ q_emb  # (N,)
    lex_tfidf = (tfidf_mat @ q_tfidf.T).toarray().ravel()
    lex_bm25 = np.array(bm25.get_scores(q_tokens), dtype=np.float32)

    sem_n = _minmax_norm(sem)
    tfidf_n = _minmax_norm(lex_tfidf)
    bm25_n = _minmax_norm(lex_bm25)
    # lexical fusion
    lex_n = 0.5 * tfidf_n + 0.5 * bm25_n

    # Intent-aware weighting
    flags = _intent_flags(query)
    alpha = 0.5
    if flags["replicas"] or flags["image"]:
        alpha = 0.35
    elif flags["cpu"] or flags["memory"] or flags["ingress"]:
        alpha = 0.4

    base_combo = alpha * sem_n + (1 - alpha) * lex_n

    # Field-aware boosts
    boosts = np.zeros_like(base_combo)
    for i, it in enumerate(items):
        boosts[i] = _field_boost(it, flags)
    combo = base_combo + boosts

    # Pre-filter and gather candidate indices
    cand_idxs: List[int] = []
    for i, it in enumerate(items):
        if types and it["type"] not in types:
            continue
        if component and it["component"] != component:
            continue
        if profile and it.get("profile") != profile:
            continue
        cand_idxs.append(i)

    # Sort by score descending
    cand_idxs.sort(key=lambda i: -float(combo[i]))

    # MMR diversification using embeddings
    mmr_lambda = 0.7
    selected: List[int] = []
    seen_per_file: Dict[str, int] = defaultdict(int)
    seen_per_comp: Dict[str, int] = defaultdict(int)

    def ok_limits(idx: int) -> bool:
        it = items[idx]
        if seen_per_file[it["path"]] >= max_per_file:
            return False
        if max_per_component is not None and seen_per_comp[it["component"]] >= max_per_component:
            return False
        return True

    # seed with top-1 passing limits
    for idx in cand_idxs:
        if ok_limits(idx):
            selected.append(idx)
            it0 = items[idx]
            seen_per_file[it0["path"]] += 1
            seen_per_comp[it0["component"]] += 1
            break

    # iterate to fill up to k
    while len(selected) < k:
        best_idx = None
        best_val = -1e9
        for idx in cand_idxs:
            if idx in selected:
                continue
            if not ok_limits(idx):
                continue
            # relevance
            rel = float(combo[idx])
            # diversity penalty: max sim to any selected
            if selected:
                sims = emb_mat[selected] @ emb_mat[idx]
                max_sim = float(np.max(sims))
            else:
                max_sim = 0.0
            val = mmr_lambda * rel - (1 - mmr_lambda) * max_sim
            if val > best_val:
                best_val = val
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
        itb = items[best_idx]
        seen_per_file[itb["path"]] += 1
        seen_per_comp[itb["component"]] += 1

    # Build results
    results = []
    for idx in selected[:k]:
        it = items[idx]
        results.append({
            "idx": idx,
            "path": it["path"],
            "component": it["component"],
            "type": it["type"],
            "profile": it.get("profile"),
            "score": float(combo[idx]),
            "k8s_kinds": it.get("k8s_kinds"),
            "k8s_images": it.get("k8s_images"),  # fix typo
        })

    return {"results": results, "stats": {"searched": len(cand_idxs), "kept": len(results)}}

def extract_keys_fn(path: str) -> List[str]:
    docs = _read_yaml_file(path)
    keys = _extract_yaml_keys(docs)
    if not keys:
        # fallback to text-based extraction for Helm-templated values files
        keys = _extract_keys_from_text(path)
    return keys
