# tools/plan_module.py
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from functools import lru_cache
import numpy as np
import re

# Folosim același model ca indexarea (poți schimba prin env sau argument, dacă vrei)
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

# ---- Helpers pentru values files ----
def _all_values_files_for_component(component: str,
                                    meta: List[Dict[str, Any]],
                                    extract_keys_fn) -> List[Dict[str, Any]]:
    """Returnează lista unică de values files pentru componentă, cu candidate_keys."""
    seen = set()
    out = []
    for r in meta:
        if r["component"] == component and r["type"] == "values":
            if r["path"] in seen:
                continue
            seen.add(r["path"])
            keys = r.get("yaml_keys") or extract_keys_fn(r["path"])
            out.append({
                "path": r["path"],
                "profile": r.get("profile"),
                "candidate_keys": keys
            })
    out.sort(key=lambda v: (0 if (v.get("profile") or "").lower() == "common" else 1,
                            v.get("profile") or "zzz", v["path"]))
    return out

# ---- Construim "documente" pe componentă (semantica) ----
def _component_docs(meta: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Pentru fiecare componentă:
    - Adună chei din values (yaml_keys)
    - Adună profile names (values profile)
    - Adună K8s kinds din manifeste
    Rezultatul e un text compact pentru embedding.
    """
    docs_keys = defaultdict(list)
    docs_profiles = defaultdict(list)
    docs_kinds = defaultdict(list)

    for r in meta:
        c = r["component"]
        t = r["type"]
        if t == "values":
            ks = r.get("yaml_keys") or []
            if ks:
                # decupăm prefixe utile (ex: initContainers, resources.*, istio.* etc.)
                # păstrăm întregul key; modelul MiniLM se descurcă rezonabil cu tokeni scurți
                docs_keys[c].extend(ks)
            prof = (r.get("profile") or "").strip()
            if prof:
                docs_profiles[c].append(prof)
        elif t == "manifest":
            kinds = r.get("k8s_kinds") or []
            docs_kinds[c].extend(kinds)

    docs = {}
    for c in set(list(docs_keys.keys()) + list(docs_profiles.keys()) + list(docs_kinds.keys())):
        parts = []
        if docs_profiles[c]:
            parts.append("profiles: " + " ".join(sorted(set(docs_profiles[c]))))
        if docs_kinds[c]:
            parts.append("kinds: " + " ".join(sorted(set(docs_kinds[c]))))
        if docs_keys[c]:
            # tăiem dacă e mult: 4000 tokeni ar fi overkill — păstrăm primele 1500 de caractere utilitare
            keys_text = " ".join(sorted(set(docs_keys[c])))
            if len(keys_text) > 8000:
                keys_text = keys_text[:8000]
            parts.append("keys: " + keys_text)
        # fallback minim să nu fie empty
        if not parts:
            parts = ["(no-meta)"]
        docs[c] = "\n".join(parts)
    return docs

# ---- Încărcare model + vectorizare (cu cache) ----
@lru_cache(maxsize=1)
def _load_encoder(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, device="cpu")

def _encode(texts: List[str], model_name: str) -> np.ndarray:
    model = _load_encoder(model_name)
    vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    return vecs.astype(np.float32)

# ---- Scor semantic pe componente ----
def _semantic_rank_components(query: str, meta: List[Dict[str, Any]], model_name: str) -> Dict[str, float]:
    docs = _component_docs(meta)
    if not docs:
        return {}
    comps = sorted(docs.keys())
    query_vec = _encode([query], model_name)[0]  # (d,)
    doc_vecs = _encode([docs[c] for c in comps], model_name)  # (n,d)
    # cosine (deoarece sunt normalizate, e dot product)
    sims = doc_vecs @ query_vec  # (n,)
    return {c: float(s) for c, s in zip(comps, sims)}

# ---- Scor de căutare (din hybrid_search) agregat pe componentă ----
def _search_rank_components(results: List[Dict[str, Any]]) -> Dict[str, float]:
    agg = defaultdict(float)
    for r in results:
        agg[r["component"]] += float(r.get("score", 0.0))
    return dict(agg)

def _minmax_norm(d: Dict[str, float]) -> Dict[str, float]:
    if not d:
        return {}
    vals = list(d.values())
    mn, mx = min(vals), max(vals)
    if mx <= mn:
        return {k: 0.0 for k in d.keys()}
    return {k: (v - mn) / (mx - mn) for k, v in d.items()}

def _component_key_features(meta: List[Dict[str, Any]]) -> Dict[str, Dict[str, bool]]:
    """
    Pentru fiecare componentă, detectează prezența unor categorii de chei:
    - has_cpu: există chei care se termină cu .cpu
    - has_mem: există chei care se termină cu .memory
    - has_replica: există chei care conțin 'replica'
    """
    feats = defaultdict(lambda: {"has_cpu": False, "has_mem": False, "has_replica": False})
    for r in meta:
        if r.get("type") != "values":
            continue
        c = r["component"]
        for k in r.get("yaml_keys") or []:
            kl = str(k).lower()
            if kl.endswith(".cpu"):
                feats[c]["has_cpu"] = True
            if kl.endswith(".memory"):
                feats[c]["has_mem"] = True
            if "replica" in kl:
                feats[c]["has_replica"] = True
    return feats

def _affinity_flags_from_query(q: str) -> Dict[str, bool]:
    ql = (q or "").lower()
    return {
        "need_cpu": ("cpu" in ql),
        "need_mem": ("mem" in ql) or ("memory" in ql),
        "need_replica": bool(re.search(r"\breplicas?\b", ql)),
    }

def _affinity_score_components(query: str, meta: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Scor simplu: +1 per categorie de cheie necesară de query și prezentă în componentă.
    """
    flags = _affinity_flags_from_query(query)
    feats = _component_key_features(meta)
    scores: Dict[str, float] = {}
    for c, f in feats.items():
        s = 0.0
        if flags["need_cpu"] and f["has_cpu"]:
            s += 1.0
        if flags["need_mem"] and f["has_mem"]:
            s += 1.0
        if flags["need_replica"] and f["has_replica"]:
            s += 1.0
        scores[c] = s
    return scores

def _fuse_scores(sem: Dict[str, float],
                 search: Dict[str, float],
                 affinity: Dict[str, float],
                 w_sem: float = 0.7,
                 w_search: float = 0.3,
                 w_aff: float = 0.0) -> Dict[str, Dict[str, float]]:
    """
    Întoarce map component -> {"semantic": x, "search": y, "affinity": a, "final": z}
    Normalizăm fiecare canal separat în [0,1], apoi combinăm liniar.
    """
    comps = sorted(set(list(sem.keys()) + list(search.keys()) + list(affinity.keys())))
    sem_n = _minmax_norm(sem)
    sea_n = _minmax_norm(search)
    aff_n = _minmax_norm(affinity)
    out = {}
    for c in comps:
        s1 = sem_n.get(c, 0.0)
        s2 = sea_n.get(c, 0.0)
        s3 = aff_n.get(c, 0.0)
        z = w_sem * s1 + w_search * s2 + w_aff * s3
        out[c] = {"semantic": s1, "search": s2, "affinity": s3, "final": z}
    return out

def _rank_components(signals: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    """
    Sortează descrescător după 'final', apoi 'semantic', apoi alfabetic.
    """
    ordered = sorted(
        signals.items(),
        key=lambda kv: (-kv[1]["final"], -kv[1]["semantic"], kv[0])
    )
    return [
        {"component": c,
         "score": round(sig["final"], 6),
         "count": None,  # nu mai folosim count-ul brut din search
         "signals": {
             "semantic_norm": round(sig["semantic"], 6),
             "search_norm": round(sig["search"], 6),
             "affinity_norm": round(sig.get("affinity", 0.0), 6),
         }}
        for c, sig in ordered
    ]

# ---- API principal ----
def do_plan(root: str,
            query: str,
            types: List[str],
            k: int,
            have_index_fn,
            hybrid_search_fn,
            load_index_fn,
            extract_keys_fn,
            *,
            model_name: str = DEFAULT_MODEL_NAME,
            weights: Optional[Dict[str, float]] = None) -> dict:
    """
    Planner cu fuziune semantică + search, fără cuvinte-cheie hardcodate.

    weights = {"semantic": 0.7, "search": 0.3} (implicit)
    """
    if not have_index_fn(root):
        raise RuntimeError("Index missing")

    # 1) căutare hibridă existentă — pentru context și semnalul "search"
    out_search = hybrid_search_fn(
        root, query, k=max(k, 20),
        types=types, component=None, profile=None, max_per_file=2
    )
    results = out_search.get("results", [])

    # 2) încărcăm meta (pt. docs pe component și lista values files)
    _, _, _, meta, _ = load_index_fn(root)

    # dacă nu avem deloc rezultate și nici meta, ieșim grațios
    if not meta:
        return {
            "plan_version": "semantic-fusion-v1",
            "query": query,
            "filters": {"types": types, "k": k},
            "components_ranking": [],
            "target": None,
            "values_files": [],
            "top_results": [],
            "stats": out_search.get("stats", {})
        }

    # 3) scor semantic pe TOATE componentele din meta
    sem_scores = _semantic_rank_components(query, meta, model_name=model_name)

    # 4) scorul de căutare agregat pe componentă din rezultatele top-N
    sea_scores = _search_rank_components(results)

    # 4.1) scor de "afinitate" în funcție de cheile existente în componentă vs cerințele query-ului
    aff_scores = _affinity_score_components(query, meta)

    # 5) fuziune
    w = weights or {"semantic": 0.7, "search": 0.3, "affinity": 0.0}
    fused = _fuse_scores(
        sem_scores, sea_scores, aff_scores,
        w_sem=w.get("semantic", 0.7),
        w_search=w.get("search", 0.3),
        w_aff=w.get("affinity", 0.0)
    )
    comp_rank = _rank_components(fused)

    if not comp_rank:
        return {
            "plan_version": "semantic-fusion-v1",
            "query": query,
            "filters": {"types": types, "k": k},
            "components_ranking": [],
            "target": None,
            "values_files": [],
            "top_results": results[:k],
            "stats": out_search.get("stats", {})
        }

    target_comp = comp_rank[0]["component"]
    values_files = _all_values_files_for_component(target_comp, meta, extract_keys_fn)

    # eșantion de chei pentru UX
    for vf in values_files:
        ks = vf.get("candidate_keys") or []
        vf["keys_count"] = len(ks)
        vf["keys_sample"] = ks[:20]

    return {
        "plan_version": "semantic-fusion-v1",
        "query": query,
        "filters": {"types": types, "k": k},
        "components_ranking": comp_rank,
        "target": {"component": target_comp},
        "values_files": values_files,
        "top_results": results[:k],
        "stats": out_search.get("stats", {})
    }
