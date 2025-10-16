import os
from typing import Dict, List, Tuple, Set
import numpy as np

from .embeddings import embed_query, embed_texts
from .store import LocalVectorStore
from .intents import extract_intents

def _expand_query(q: str) -> str:
    # Prefer LLM intents; fallback to simple keywords
    try:
        req = extract_intents(q, union=None, verbose=False)
        extras: List[str] = []
        allowed = req.get("allowed", set())
        if req.get("metrics"):
            extras.append("global app monitoring path port metrics")
        if req.get("hosts"):
            extras.append("global ingresses main hosts")
        if "volumes" in allowed or req.get("volume_hint"):
            extras.append("volumes mountPath emptyDir")
        if req.get("image_repo"):
            extras.append(f"image repository {req['image_repo']}")
        if not extras:
            raise RuntimeError("no extras")
        return f"{q} || " + " ; ".join(extras)
    except Exception:
        # Heuristic fallback
        qt = q.lower()
        extras: List[str] = []
        if "metrics" in qt or "monitoring" in qt:
            extras.append("global app monitoring path port")
        if "host" in qt or "ingress" in qt:
            extras.append("global ingresses main hosts")
        if "volume" in qt or "mount" in qt:
            extras.append("volumes mountPath emptyDir")
        if "nginx" in qt:
            extras.append("image repository nginx")
        return q if not extras else f"{q} || " + " ; ".join(extras)

def _keywords_from_intents(req: Dict) -> Set[str]:
    kws: Set[str] = set()
    allowed = req.get("allowed") or set()
    for k in allowed:
        kws.add(str(k).lower())
    if req.get("metrics"):
        kws.update({"metrics", "monitoring", "path", "port"})
    if req.get("hosts"):
        kws.update({"ingresses", "hosts", "global"})
    if req.get("volume_hint") or ("volumes" in allowed):
        kws.update({"volumes", "mountpath", "emptydir"})
    if req.get("image_repo"):
        kws.update({"image", "repository"})
    # common extras
    kws.update({"probes", "readiness", "configmaps", "secrets"})
    return kws

def _trim_chunk(text: str, max_lines: int, max_chars: int) -> str:
    if not text:
        return text
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    out = "\n".join(lines)
    if len(out) > max_chars:
        out = out[:max_chars]
    return out

def _mmr_select(qvec: np.ndarray, dvecs: np.ndarray, k: int, lam: float = 0.7) -> List[int]:
    if dvecs.shape[0] == 0:
        return []
    # cosine sims (vecs are already normalized in embed_texts)
    sims = (dvecs @ qvec.reshape(-1, 1)).flatten()
    selected: List[int] = []
    candidates = list(range(dvecs.shape[0]))
    # pick best first
    best = int(np.argmax(sims))
    selected.append(best)
    candidates.remove(best)
    while len(selected) < min(k, dvecs.shape[0]) and candidates:
        max_score = -1e9
        max_idx = candidates[0]
        for idx in candidates:
            sim_to_query = sims[idx]
            sim_to_selected = 0.0
            for s in selected:
                sim_to_selected = max(sim_to_selected, float(dvecs[idx] @ dvecs[s]))
            score = lam * sim_to_query - (1.0 - lam) * sim_to_selected
            if score > max_score:
                max_score = score
                max_idx = idx
        selected.append(max_idx)
        candidates.remove(max_idx)
    return selected

def retrieve(store: LocalVectorStore, query: str, k: int = 5) -> List[Tuple[float, Dict, str]]:
    # Embed the expanded query
    expanded = _expand_query(query)
    qvec = embed_query(expanded)

    # Candidate pool size (multiplier) and knobs
    mult = max(2, int(os.getenv("RET_INITIAL_MULT", "4")))
    initial_k = max(k, mult * k)
    lam = float(os.getenv("RET_MMR_LAMBDA", "0.7"))
    kw_w = float(os.getenv("RET_KW_WEIGHT", "0.05"))
    sch_w = float(os.getenv("RET_SCHEMA_WEIGHT", "0.02"))
    max_lines = int(os.getenv("RET_MAX_LINES", "250"))
    max_chars = int(os.getenv("RET_MAX_CHARS", "12000"))

    # Initial dense retrieval
    raw = store.search(qvec, k=initial_k)

    # Keyword boosts
    intents = extract_intents(query, union=None, verbose=False)
    kws = _keywords_from_intents(intents)
    def boost(score: float, meta: Dict, text: str) -> float:
        lower = text.lower()
        hit = sum(1 for kw in kws if kw in lower)
        schema_bonus = sch_w if str(meta.get("section")) == "__schema__" else 0.0
        return float(score) + kw_w * hit + schema_bonus

    boosted = [(boost(s, m, t), m, t) for (s, m, t) in raw]

    # Re-rank with MMR for diversity
    texts = [t for _, _, t in boosted]
    if texts:
        dvecs = embed_texts(texts)  # (n, d), normalized
        idxs = _mmr_select(qvec, dvecs, k=k, lam=lam)
        mmr_selected = [boosted[i] for i in idxs]
    else:
        mmr_selected = boosted[:k]

    # Ensure per-file schema chunk if available
    files = {}
    for s, m, t in mmr_selected:
        files.setdefault(m.get("file"), []).append((s, m, t))
    # Look for schema candidates in the pool for missing files
    for file in list(files.keys()):
        if not any(str(m.get("section")) == "__schema__" for _, m, _ in files[file]):
            # find best schema chunk for this file in boosted pool
            best_sch = None
            for s, m, t in boosted:
                if m.get("file") == file and str(m.get("section")) == "__schema__":
                    best_sch = (s, m, t) if best_sch is None or s > best_sch[0] else best_sch
            if best_sch:
                mmr_selected.append(best_sch)

    # Trim chunks for prompt budget
    final: List[Tuple[float, Dict, str]] = []
    for s, m, t in mmr_selected[: max(k, len(mmr_selected))]:
        final.append((float(s), m, _trim_chunk(t, max_lines=max_lines, max_chars=max_chars)))

    # Sort desc by score and keep top-k
    final.sort(key=lambda x: x[0], reverse=True)
    return final[:k]
