#!/usr/bin/env python3
# ragctl.py — Retrieval „prod-ready”: persistent (FAISS + TF-IDF), hibrid, filtre, JSON stabil + PLAN.
import argparse, os, sys, json, time, glob, pathlib, re, hashlib

# from plan_module import do_plan as plan_build
from tools.plan_module import do_plan as plan_do
from tools.patch_module import patch_from_plan
from tools.translate_module import translate_intents
from tools.translate_module import build_llm_prompt
from tools.config import RAGConfig

from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

import numpy as np
import pickle
import yaml

from sentence_transformers import SentenceTransformer
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# ---------------------------- Config implicite ----------------------------
# REMOVED hardcoded defaults; all come from RAGConfig loaded per --root

# ---------------------------- Utilitare IO ----------------------------
def read_text(p: str) -> str:
    return pathlib.Path(p).read_text(encoding="utf-8", errors="ignore")

def write_text(p: str, content: str):
    pathlib.Path(p).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(p).write_text(content, encoding="utf-8")

def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def list_files(root: str, cfg: RAGConfig) -> List[str]:
    patterns = cfg.file_globs
    files: List[str] = []
    for pat in patterns:
        files += glob.glob(os.path.join(root, pat), recursive=True)
    return sorted(set(files))

# ---------------------------- Clasificare + parsing ----------------------------
KIND_RE = re.compile(r"^\s*kind:\s*([A-Za-z0-9]+)\s*$", re.MULTILINE)
HELM_BLOCK_COMMENT = re.compile(r"{{-?/\*.*?\*/-?}}", re.DOTALL)
HELM_INLINE = re.compile(r"{{-?\s*.*?\s*-?}}")
KEY_LINE = re.compile(r"^(\s*)([A-Za-z0-9_.-]+):\s*(?:[^>|].*)?$")

def classify(root: str, path: str, cfg: RAGConfig) -> Dict[str, Optional[str]]:
    # component as first segment relative to root
    try:
        rel = pathlib.Path(path).resolve().relative_to(pathlib.Path(root).resolve())
        parts = str(rel).replace("\\", "/").split("/")
        comp = parts[0] if parts and parts[0] else pathlib.Path(path).parent.name
    except Exception:
        comp = pathlib.Path(path).parent.name

    lower = path.lower()
    t = "config"; profile = None

    def _match(globs: List[str]) -> bool:
        import fnmatch
        relp = str(pathlib.Path(path))
        return any(fnmatch.fnmatch(relp, os.path.join(root, g)) or fnmatch.fnmatch(relp, g) for g in globs)

    if _match(cfg.readme_globs):
        t = "readme"
    elif _match(cfg.values_globs):
        t = "values"; profile = pathlib.Path(path).stem
    elif _match(cfg.manifest_globs):
        t = "manifest"
    return {"component": comp, "type": t, "profile": profile}

def sanitize_helm_yaml(text: str) -> str:
    # scoate comentarii templating + înlocuiește {{ ... }} cu placeholder sigur
    text = HELM_BLOCK_COMMENT.sub("", text)
    text = HELM_INLINE.sub('"__TEMPLATE__"', text)
    # linii care sunt DOAR directive {{ if ... }} — elimină
    lines=[]
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("{{") and s.endswith("}}") and ":" not in ln:
            continue
        lines.append(ln)
    return "\n".join(lines)

def flatten_yaml_keys(data: Any, prefix: str = "") -> List[str]:
    keys: List[str] = []
    if isinstance(data, dict):
        for k, v in data.items():
            pre = f"{prefix}.{k}" if prefix else k
            keys += flatten_yaml_keys(v, pre)
    elif isinstance(data, list):
        for i, v in enumerate(data):
            pre = f"{prefix}[{i}]"
            keys += flatten_yaml_keys(v, pre)
    else:
        if prefix:
            keys.append(prefix)
    return keys

def clean_idx_notation(k: str) -> str:
    return re.sub(r"\[\d+\]", "[]", k)

def extract_yaml_keys_from_file(path: str) -> List[str]:
    raw = read_text(path)
    # 1) direct
    try:
        data = yaml.safe_load(raw)
        if data is not None:
            return sorted(set(clean_idx_notation(k) for k in flatten_yaml_keys(data)))
    except Exception:
        pass
    # 2) cu sanitizare Helm
    try:
        sanitized = sanitize_helm_yaml(raw)
        data = yaml.safe_load(sanitized)
        if data is not None:
            return sorted(set(clean_idx_notation(k) for k in flatten_yaml_keys(data)))
    except Exception:
        pass
    # 3) fallback: prin indentare
    keys=set(); stack=[]
    for ln in sanitize_helm_yaml(raw).splitlines():
        m = KEY_LINE.match(ln)
        if not m: continue
        indent, key = len(m.group(1).expandtabs(2)), m.group(2)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        stack.append((indent, key))
        dotted = ".".join(k for _, k in stack)
        keys.add(dotted)
    return sorted(keys)

def extract_kinds(text: str) -> List[str]:
    return sorted(set(KIND_RE.findall(text)))

def chunk_text(text: str, n: Optional[int] = None) -> List[str]:
    if n is None or n <= 0 or len(text) <= n:
        return [text]
    return [text[i : i + n] for i in range(0, len(text), n)]

# ---------------------------- Construire corpus ----------------------------
def build_records(root: str, cfg: RAGConfig) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Returnează lista de fragmente (records) + hash-uri per fișier."""
    records: List[Dict[str, Any]] = []
    file_hashes: Dict[str, str] = {}
    for path in list_files(root, cfg):
        cls = classify(root, path, cfg)
        text = read_text(path)

        # meta suplimentare
        yaml_keys = extract_yaml_keys_from_file(path) if cls["type"] == "values" else []
        kinds = extract_kinds(text) if cls["type"] in ("manifest",) else []

        file_id = f"{path}"  # grupare pe fișier

        # chunking
        for i, part in enumerate(chunk_text(text, n=cfg.chunk_size)):
            bm25_text = sanitize_helm_yaml(part) if cls["type"] in ("values", "manifest") else part
            records.append(
                {
                    "path": path,
                    "file_id": file_id,
                    "component": cls["component"],
                    "type": cls["type"],
                    "profile": cls["profile"],
                    "k8s_kinds": kinds if i == 0 else [],
                    "yaml_keys": yaml_keys if cls["type"] == "values" and i == 0 else [],
                    "text": part,
                    "bm25_text": bm25_text,
                }
            )

        # hash pe fișier (pt. snapshot)
        try:
            file_hashes[path] = sha256_file(path)
        except Exception:
            file_hashes[path] = ""
    return records, file_hashes


# ---------------------------- Persistență index ----------------------------
def rag_dir(root: str) -> pathlib.Path:
    d = pathlib.Path(root) / ".rag"
    d.mkdir(parents=True, exist_ok=True)
    return d

def save_index(root: str, faiss_index, ids: np.ndarray, tfidf_pack: dict,
               meta: List[Dict[str, Any]], snapshot: dict):
    d = rag_dir(root)
    faiss.write_index(faiss_index, str(d / "index.faiss"))
    np.save(str(d / "ids.npy"), ids)
    with open(d / "tfidf.pkl", "wb") as f:
        pickle.dump(tfidf_pack, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(d / "meta.jsonl", "w", encoding="utf-8") as f:
        for r in meta:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_text(d / "snapshot.json", json.dumps(snapshot, ensure_ascii=False, indent=2))

def save_key_catalog(root: str, file_keys: Dict[str, set], key_idf: Dict[str, float], N: int):
    d = rag_dir(root)
    # file_keys.jsonl: pe fiecare linie file_id + lista cheilor
    with open(d / "file_keys.jsonl", "w", encoding="utf-8") as f:
        for fid, keys in sorted(file_keys.items()):
            f.write(json.dumps({"file_id": fid, "keys": sorted(keys)}, ensure_ascii=False) + "\n")
    # key_stats.json: N, df, idf (df ca dict simplu)
    # calculăm df din idf dacă nu l-ai păstrat separat
    df = {}
    for k, idf_val in key_idf.items():
        # idf = ln((N+1)/(df+1))  => df ≈ (N+1)/exp(idf) - 1
        try:
            df[k] = max(0, int(round((N + 1) / np.exp(idf_val) - 1)))
        except Exception:
            df[k] = 0
    write_text(d / "key_stats.json", json.dumps({"N": N, "df": df, "idf": key_idf}, ensure_ascii=False, indent=2))

def _load_tfidf_pack(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        # Dacă indexul vechi a fost salvat cu joblib, oferim mesaj clar
        raise RuntimeError(
            "Nu pot încărca tfidf.pkl cu pickle. "
            "Probabil a fost creat cu joblib. Șterge folderul .rag și reindexează: "
            "`rm -rf k3s/.rag && python3 ragctl.py index --root k8s`"
        ) from e

def load_index(root: str):
    d = rag_dir(root)
    fi = faiss.read_index(str(d / "index.faiss"))
    ids = np.load(str(d / "ids.npy"))
    tfidf_pack = _load_tfidf_pack(d / "tfidf.pkl")
    meta: List[Dict[str, Any]] = []
    with open(d / "meta.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    snapshot = json.loads(read_text(d / "snapshot.json"))
    return fi, ids, tfidf_pack, meta, snapshot

def have_index(root: str) -> bool:
    d = rag_dir(root)
    needed = ["index.faiss", "ids.npy", "tfidf.pkl", "meta.jsonl", "snapshot.json",
              "file_keys.jsonl", "key_stats.json"]
    return all((d / n).exists() for n in needed)


# ---------------------------- Indexare ----------------------------
def do_index(root: str, cfg: RAGConfig) -> dict:
    t0 = time.perf_counter()
    records, file_hashes = build_records(root, cfg)

    # ---------------- Dense (FAISS) ----------------
    t_dense0 = time.perf_counter()
    model = SentenceTransformer(cfg.model_name, device="cpu")
    texts_dense = [r["text"] for r in records]
    vecs = model.encode(
        texts_dense,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=False,
    )
    dim = vecs.shape[1]
    index = faiss.IndexHNSWFlat(dim, cfg.hnsw_m)
    index.hnsw.efConstruction = cfg.hnsw_ef
    index.add(vecs.astype(np.float32))
    dense_ms = int((time.perf_counter() - t_dense0) * 1000)

    # ---------------- TF-IDF (BM25-like) ----------------
    t_bm0 = time.perf_counter()
    texts_bm25 = [r["bm25_text"] for r in records]
    vectorizer = TfidfVectorizer(ngram_range=cfg.tfidf_ngram, min_df=cfg.tfidf_min_df)
    X = vectorizer.fit_transform(texts_bm25)
    X = normalize(X, norm="l2", copy=False)
    bm25_ms = int((time.perf_counter() - t_bm0) * 1000)

    # ---------------- Catalog chei + IDF ----------------
    # agregă cheile din toate fișierele values, pe file_id
    from collections import defaultdict, Counter
    file_keys: Dict[str, set] = defaultdict(set)
    for r in records:
        if r["type"] == "values" and r.get("yaml_keys"):
            file_keys[r["file_id"]].update(r["yaml_keys"])

    # document frequency (df) pe chei
    df_counter = Counter()
    for fid, keys in file_keys.items():
        for k in set(keys):
            df_counter[k] += 1
    N = max(1, len(file_keys))  # nr. „documente” = nr. fișiere values cu chei
    key_idf = {k: float(np.log((N + 1) / (df_counter[k] + 1))) for k in df_counter}  # +1 smoothing

    # ---------------- Persistă index + cataloage ----------------
    ids = np.arange(len(records))
    snapshot = {
        "model": cfg.model_name,
        "chunk_size": cfg.chunk_size,
        "files": file_hashes,
        "created_at": time.time(),
        "counts": {"records": len(records), "files_with_keys": N, "unique_keys": len(key_idf)},
    }
    save_index(root, index, ids, {"vectorizer": vectorizer, "matrix": X, "ids": ids}, records, snapshot)
    save_key_catalog(root, file_keys, key_idf, N)

    total_ms = int((time.perf_counter() - t0) * 1000)
    return {
        "dense_ms": dense_ms,
        "bm25_ms": bm25_ms,
        "total_ms": total_ms,
        "records": len(records),
        "files_with_keys": N,
        "unique_keys": len(key_idf),
    }

# ---------------------------- Căutare hibridă ----------------------------
def reciprocal_rank_fusion(ranks: Dict[int, int], k: int) -> float:
    # scor RRF = sum(1/(k + rank_i))
    return sum(1.0 / (k + r) for r in ranks.values())

def hybrid_search(root: str, query: str, k: int, types: List[str], component: Optional[str], profile: Optional[str], max_per_file: int = 2, weights: Optional[Tuple[float,float]] = None, cfg: Optional[RAGConfig] = None) -> dict:
    fi, ids, tfidf_pack, meta, snapshot = load_index(root)
    rrf_k = (cfg.rrf_k if cfg else 60)
    wts = weights or ((cfg.dense_weight, cfg.bm25_weight) if cfg else (0.6, 0.4))

    # encode query
    t_dense0 = time.perf_counter()
    model = SentenceTransformer(snapshot.get("model", (cfg.model_name if cfg else "all-MiniLM-L6-v2")), device="cpu")
    q_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    # dense topN
    topN = min((cfg.search_topN if cfg else 200), len(ids))
    D_dense, I_dense = fi.search(q_vec.astype(np.float32), topN)
    dense_ms = int((time.perf_counter() - t_dense0) * 1000)

    # bm25
    t_bm0 = time.perf_counter()
    vec = tfidf_pack["vectorizer"].transform([query])
    vec = normalize(vec, norm="l2", copy=False)
    sims = (tfidf_pack["matrix"] @ vec.T).toarray().ravel()  # cosine on L2-normalized tf-idf
    topM = min((cfg.search_topM if cfg else 200), len(ids))
    I_bm = np.argsort(-sims)[:topM]
    bm25_ms = int((time.perf_counter() - t_bm0) * 1000)

    # rank maps
    ranks_dense = {int(int_id): rank for rank, int_id in enumerate(I_dense[0])}
    ranks_bm = {int(int_id): rank for rank, int_id in enumerate(I_bm)}

    # candidați unioni
    cand = set(list(ranks_dense.keys()) + list(ranks_bm.keys()))

    # fuziune RRF + ponderi (convertim RRF în două componente)
    fused_scores = {}
    for idx in cand:
        rd = ranks_dense.get(idx, 10_000)
        rb = ranks_bm.get(idx, 10_000)
        s = wts[0] * (1.0 / (rrf_k + rd)) + wts[1] * (1.0 / (rrf_k + rb))
        fused_scores[idx] = s

    # aplică filtre (types/component/profile)
    def passes_filters(r: Dict[str, Any]) -> bool:
        if types and r["type"] not in types: return False
        if component and r["component"] != component: return False
        if profile and (r.get("profile") != profile): return False
        return True

    # ordonează, grupează pe fișier
    ordered = sorted([i for i in cand], key=lambda i: (-fused_scores[i], i))
    results = []
    seen_by_file = defaultdict(int)
    for idx in ordered:
        r = meta[int(idx)]
        if not passes_filters(r): continue
        fid = r["file_id"]
        if seen_by_file[fid] >= max_per_file: continue
        seen_by_file[fid] += 1
        snippet = r["text"].replace("\n", " ")
        if len(snippet) > 220: snippet = snippet[:220] + "…"
        results.append({
            "score": round(float(fused_scores[idx]), 6),
            "path": r["path"],
            "component": r["component"],
            "type": r["type"],
            "profile": r.get("profile"),
            "file_id": fid,
            "k8s_kinds": r.get("k8s_kinds") or [],
            "yaml_keys": r.get("yaml_keys") or [],
            "snippet": snippet
        })
        if len(results) >= k:
            break

    stats = {
        "dense_ms": dense_ms,
        "bm25_ms": bm25_ms,
        "merge_ms": 0,  # simplificat
        "records": len(meta),
        "model": snapshot.get("model"),
    }
    grouping = [{"file_id": fid, "count": c} for fid, c in seen_by_file.items()]

    return {
        "query": query,
        "filters": {"types": types or None, "component": component, "profile": profile, "k": k},
        "results": results,
        "grouping": grouping,
        "stats": stats,
    }

# ---------------------------- PLAN (targetare + values files) ----------------------------
def _component_ranking_from_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    agg = defaultdict(float); cnt = defaultdict(int)
    for r in results:
        c = r["component"]
        agg[c] += float(r["score"])
        cnt[c] += 1
    ordered = sorted(agg.items(), key=lambda kv: (-kv[1], -cnt[kv[0]], kv[0]))
    return [{"component": c, "score": agg[c], "count": cnt[c]} for c, _ in ordered]

def _all_values_files_for_component(root: str, component: str, meta: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for r in meta:
        if r["component"] == component and r["type"] == "values":
            if r["path"] in seen: 
                continue
            seen.add(r["path"])
            keys = r.get("yaml_keys") or extract_yaml_keys_from_file(r["path"])
            out.append({
                "path": r["path"],
                "profile": r.get("profile"),
                "candidate_keys": keys
            })
    out.sort(key=lambda v: (0 if (v.get("profile") or "").lower() == "common" else 1, v.get("profile") or "zzz", v["path"]))
    return out

def do_plan(root: str, query: str, types: List[str], k: int) -> dict:
    if not have_index(root):
        raise RuntimeError("Index missing")
    # rulăm căutarea hibridă pentru context
    out_search = hybrid_search(root, query, k=max(k, 20), types=types, component=None, profile=None, max_per_file=2)
    results = out_search["results"]

    # încărcăm meta pentru a lista toate values files din componenta țintă
    _, _, _, meta, _ = load_index(root)

    if not results:
        return {
            "query": query,
            "filters": {"types": types, "k": k},
            "components_ranking": [],
            "target": None,
            "values_files": [],
            "top_results": [],
            "stats": out_search.get("stats", {})
        }

    comp_rank = _component_ranking_from_results(results)
    target_comp = comp_rank[0]["component"]
    values_files = _all_values_files_for_component(root, target_comp, meta)

    # esantion chei
    for vf in values_files:
        ks = vf["candidate_keys"]
        vf["keys_count"] = len(ks)
        vf["keys_sample"] = ks[:20]

    return {
        "query": query,
        "filters": {"types": types, "k": k},
        "components_ranking": comp_rank,
        "target": {"component": target_comp},
        "values_files": values_files,
        "top_results": results[:k],
        "stats": out_search.get("stats", {})
    }

import textwrap
try:
    import yaml
except Exception:
    yaml = None  # fallback fără YAML dacă lipsește pyyaml

def _render_manifest_stub(mf: dict) -> str:
    """Generează un stub YAML pentru create/suggest (Deployment/Job/etc)."""
    kind = mf.get("kind", "Deployment")
    name = mf.get("name", os.getenv("RAG_DEFAULT_NAME", "app"))
    p = mf.get("params", {}) or {}
    image = p.get("image", os.getenv("RAG_DEFAULT_IMAGE", "nginx:latest"))
    replicas = int(p.get("replicas", os.getenv("RAG_DEFAULT_REPLICAS", 1)))
    with_init = bool(p.get("withInitContainer"))

    doc = {
        "apiVersion": "apps/v1" if kind in ("Deployment","StatefulSet","DaemonSet") else "v1",
        "kind": kind,
        "metadata": {"name": name},
        "spec": {}
    }

    if kind in ("Deployment","StatefulSet","DaemonSet"):
        doc["spec"]["replicas"] = replicas
        podspec = {
            "containers": [{
                "name": name,
                "image": image,
            }]
        }
        if with_init:
            podspec["initContainers"] = [{
                "name": "init",
                "image": image,
                "command": ["sh","-c","echo init && sleep 1"]
            }]
        doc["spec"]["selector"] = {"matchLabels": {"app": name}}
        doc["spec"]["template"] = {
            "metadata": {"labels": {"app": name}},
            "spec": podspec
        }
    # alte Kinds pot fi adăugate ușor aici (Job, CronJob etc.)

    if yaml:
        return yaml.safe_dump(doc, sort_keys=False).strip()
    # fallback fără pyyaml:
    return textwrap.dedent(f"""\
      kind: {doc['kind']}
      apiVersion: {doc['apiVersion']}
      metadata:
        name: {name}
      spec: {{ ... }}
    """).rstrip()

def _pretty_print_translate(plan: dict, tr: dict):
    tgt = (plan.get("target") or {}).get("component", "-")
    print(f"\nComponentă țintă: {tgt} (planner={plan.get('plan_version')})")
    if tr.get("notes"):
        print("Note:")
        for n in tr["notes"][:5]:
            print(f"  • {n}")

    if tr.get("intents"):
        print("\nIntenții (values):")
        for it in tr["intents"]:
            if it["op"] == "set":
                print(f"  - set {it['key']} = {it['value']}")
            else:
                print(f"  - {it['op']} {it['key']}")
    if tr.get("rejected"):
        print("\nRespins (guardrails):")
        for it in tr["rejected"]:
            print(f"  - {it.get('key','?')} ({it.get('reason','')})")

    if tr.get("manifests"):
        print("\nManifeste propuse:")
        for mf in tr["manifests"]:
            line = f"  - {mf.get('op','patch')} {mf.get('kind','?')} name={mf.get('name','app')}"
            reason = mf.get("reason")
            if reason: line += f" [{reason}]"
            print(line)
        # arată un YAML de exemplu pentru primul „create/suggest”
        mf0 = next((m for m in tr["manifests"] if m.get("op") in ("create","suggest")), None)
        if mf0:
            print("\nExemplu YAML:")
            print(_render_manifest_stub(mf0))

def _pretty_print_autopatch(payload: dict):
    print(f"\nComponentă: {payload.get('component','-')} (planner={payload.get('plan_version')})")
    tr = payload.get("translate", {})
    if tr.get("intents"):
        print("Intenții acceptate:")
        for it in tr["intents"]:
            if it["op"] == "set":
                print(f"  - set {it['key']} = {it['value']}")
            else:
                print(f"  - {it['op']} {it['key']}")
    if tr.get("rejected"):
        print("\nRespins (guardrails):")
        for it in tr["rejected"]:
            print(f"  - {it.get('key','?')} ({it.get('reason','')})")

    outp = payload.get("patch_result", {})
    patches = outp.get("patches", [])
    if patches:
        print("\nPatch-uri generate:")
        for p in patches:
            print(f"  - {p['patch_yaml_path']} -> {p['values_path']} (profile={p.get('profile')})")
    cmds = outp.get("apply_commands", [])
    if cmds:
        print("\nComenzi yq (merge determinist):")
        for c in cmds:
            print(c)

def _parse_env_pairs(s: str) -> List[Dict[str, str]]:
    items = []
    if not s:
        return items
    for pair in re.split(r"[,\s]+", str(s).strip()):
        if not pair or "=" not in pair:
            continue
        k, v = pair.split("=", 1)
        if k:
            items.append({"name": k, "value": v})
    return items

def _last_intent_value(intents: List[Dict[str, Any]], pred) -> Optional[str]:
    val = None
    for it in intents:
        if it.get("op") == "set" and pred(str(it.get("key","")).lower()):
            val = str(it.get("value",""))
    return val

def _build_final_manifest(mf: dict, intents: List[Dict[str, Any]]) -> str:
    """
    Synthesize a final K8s manifest from the suggested manifest and intents:
    - image.repository + image.tag or full 'image'
    - replicas from replicaCount/replicas
    - resources.{requests,limits}.{cpu,memory}
    - env pairs if present in intents (any key containing 'env')
    """
    kind = mf.get("kind", "Deployment")
    name = mf.get("name", os.getenv("RAG_DEFAULT_NAME", "app"))
    p = (mf.get("params") or {}).copy()

    # extract from intents
    img_repo = _last_intent_value(intents, lambda k: k.endswith("image.repository"))
    img_tag  = _last_intent_value(intents, lambda k: k.endswith("image.tag"))
    img_full = _last_intent_value(intents, lambda k: k.endswith(".image") or k == "image")
    if img_repo:
        p["image"] = f"{img_repo}:{img_tag or 'latest'}"
    elif img_full:
        p["image"] = img_full
    image = p.get("image", os.getenv("RAG_DEFAULT_IMAGE", "nginx:latest"))

    rep_from_intents = _last_intent_value(intents, lambda k: k.endswith("replicas") or k.endswith("replicacount"))
    replicas = int(rep_from_intents) if (rep_from_intents and rep_from_intents.isdigit()) else int(p.get("replicas", os.getenv("RAG_DEFAULT_REPLICAS", 1)))

    req_cpu = _last_intent_value(intents, lambda k: k.endswith("resources.requests.cpu"))
    lim_cpu = _last_intent_value(intents, lambda k: k.endswith("resources.limits.cpu"))
    req_mem = _last_intent_value(intents, lambda k: k.endswith("resources.requests.memory"))
    lim_mem = _last_intent_value(intents, lambda k: k.endswith("resources.limits.memory"))

    env_val = _last_intent_value(intents, lambda k: "env" in k and not k.endswith(".enabled"))
    env_list = _parse_env_pairs(env_val) if env_val else []

    with_init = bool(p.get("withInitContainer"))

    # base manifest
    doc = {
        "apiVersion": "apps/v1" if kind in ("Deployment","StatefulSet","DaemonSet") else "v1",
        "kind": kind,
        "metadata": {"name": name},
        "spec": {}
    }

    # podspec and containers
    if kind in ("Deployment","StatefulSet","DaemonSet"):
        doc["spec"]["replicas"] = replicas
        podspec = {
            "containers": [{
                "name": name,
                "image": image,
            }]
        }
        if env_list:
            podspec["containers"][0]["env"] = env_list

        # resources
        resources = {}
        if req_cpu or req_mem:
            resources.setdefault("requests", {})
            if req_cpu:
                resources["requests"]["cpu"] = req_cpu
            if req_mem:
                resources["requests"]["memory"] = req_mem
        if lim_cpu or lim_mem:
            resources.setdefault("limits", {})
            if lim_cpu:
                resources["limits"]["cpu"] = lim_cpu
            if lim_mem:
                resources["limits"]["memory"] = lim_mem
        if resources:
            podspec["containers"][0]["resources"] = resources

        if with_init:
            podspec["initContainers"] = [{
                "name": "init",
                "image": image,
                "command": ["sh","-c","echo init && sleep 1"]
            }]

        doc["spec"]["selector"] = {"matchLabels": {"app": name}}
        doc["spec"]["template"] = {
            "metadata": {"labels": {"app": name}},
            "spec": podspec
        }
    else:
        # other kinds can be extended as needed
        doc["spec"] = {}

    return yaml.safe_dump(doc, sort_keys=False).strip()

# ---------------------------- CLI ----------------------------
def main():
    ap = argparse.ArgumentParser(description="ragctl — Retrieval persistent (FAISS+TFIDF), hibrid, filtre, JSON stabil + PLAN")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_idx = sub.add_parser("index", help="Construiește/actualizează indexul (.rag/)")
    ap_idx.add_argument("--root", required=True, help="rădăcina datasetului (ex: k8s)")
    ap_idx.add_argument("--force", action="store_true", help="ignoră snapshot-ul și reindexează tot")

    ap_s = sub.add_parser("search", help="Căutare hibridă (dense+BM25) cu filtre, JSON stabil")
    ap_s.add_argument("--root", required=True)
    ap_s.add_argument("--q", required=True, help="interogare în limbaj natural")
    ap_s.add_argument("--types", default=os.getenv("RAG_DEFAULT_TYPES", "values,manifest"), help="CSV: values,manifest,config,readme")
    ap_s.add_argument("--component", help="filtru pe componentă (subfolder)")
    ap_s.add_argument("--profile", help="filtru pe profil values")
    ap_s.add_argument("--k", type=int, default=int(os.getenv("RAG_SEARCH_K", "8")), help="număr rezultate")
    ap_s.add_argument("--max-per-file", type=int, default=int(os.getenv("RAG_MAX_PER_FILE", "2")), help="max fragmente/fișier în top-k")
    ap_s.add_argument("--weights", default=os.getenv("RAG_SEARCH_WEIGHTS", ""), help="ponderi dense,bm25 (ex: 0.6,0.4)")
    ap_s.add_argument("--json", action="store_true", help="afișează exact JSON-ul contractului")

    ap_p = sub.add_parser("plan", help="Alege componenta țintă și listează values files + chei candidate")
    ap_p.add_argument("--root", required=True)
    ap_p.add_argument("--q", required=True)
    ap_p.add_argument("--types", default=os.getenv("RAG_DEFAULT_TYPES", "values,manifest"))
    ap_p.add_argument("--k", type=int, default=int(os.getenv("RAG_PLAN_K", "8")))
    ap_p.add_argument("--json", action="store_true")

    ap_patch = sub.add_parser("patch", help="Generează patch-uri YAML rutate pe fișierele values (folosind plan).")
    ap_patch.add_argument("--root", required=True)
    ap_patch.add_argument("--q", default="", help="cerință în limbaj natural (pentru plan/targetare)")
    ap_patch.add_argument("--types", default=os.getenv("RAG_DEFAULT_TYPES", "values,manifest"), help="pentru plan")
    ap_patch.add_argument("--k", type=int, default=int(os.getenv("RAG_PLAN_K", "20")), help="pentru plan")
    ap_patch.add_argument("--out", dest="out_dir", default=os.getenv("RAG_PATCH_OUT", "out"), help="folder output patch-uri")
    # intenții directe (fără LLM):
    ap_patch.add_argument("--set", dest="sets", action="append", default=[], help="key=value (repetabil)")
    ap_patch.add_argument("--enable", dest="enables", action="append", default=[], help="key (repetabil)")
    ap_patch.add_argument("--disable", dest="disables", action="append", default=[], help="key (repetabil)")
    ap_patch.add_argument("--json", action="store_true")

    # translate
    ap_tr = sub.add_parser("translate", help="NL -> intenții (folosind plan + rule/LLM)")
    ap_tr.add_argument("--root", required=True)
    ap_tr.add_argument("--q", required=True)
    ap_tr.add_argument("--types", default=os.getenv("RAG_DEFAULT_TYPES", "values,manifest"))
    ap_tr.add_argument("--k", type=int, default=int(os.getenv("RAG_TRANSLATE_K", "20")))
    ap_tr.add_argument("--provider", choices=["rule", "llm"], default=os.getenv("RAG_TRANSLATE_PROVIDER", "rule"))
    ap_tr.add_argument("--allow-new-keys", action="store_true")
    ap_tr.add_argument("--json", action="store_true")
    ap_tr.add_argument("--show-prompt", action="store_true", help="Print the generated LLM prompt (no call) and exit")
    # NEW flags to render final YAML
    ap_tr.add_argument("--render-final", action="store_true", help="Render final manifest YAML from intents + suggestion")
    ap_tr.add_argument("--yaml-only", action="store_true", help="Print only the YAML when used with --render-final")

    # autopatch = plan → translate → patch (într-o singură comandă)
    ap_auto = sub.add_parser("autopatch", help="Plan → Translate → Patch (end-to-end)")
    ap_auto.add_argument("--root", required=True)
    ap_auto.add_argument("--q", required=True)
    ap_auto.add_argument("--types", default=os.getenv("RAG_DEFAULT_TYPES", "values,manifest"))
    ap_auto.add_argument("--k", type=int, default=int(os.getenv("RAG_AUTOPATCH_K", "20")))
    ap_auto.add_argument("--provider", choices=["rule", "llm"], default=os.getenv("RAG_TRANSLATE_PROVIDER", "rule"))
    ap_auto.add_argument("--allow-new-keys", action="store_true")
    ap_auto.add_argument("--out", dest="out_dir", default=os.getenv("RAG_PATCH_OUT", "out"))
    ap_auto.add_argument("--json", action="store_true")
    ap_auto.add_argument("--show-prompt", action="store_true", help="Print the generated LLM prompt (no call) and exit")
    # NEW flags to render final YAML
    ap_auto.add_argument("--render-final", action="store_true", help="Render final manifest YAML from intents + suggestion")
    ap_auto.add_argument("--yaml-only", action="store_true", help="Print only the YAML when used with --render-final")

    args = ap.parse_args()
    cfg = RAGConfig.from_root(args.root)

    # --- INDEX ---
    if args.cmd == "index":
        if have_index(args.root) and not args.force:
            try:
                _, _, _, _, snap = load_index(args.root)
                _, current_hashes = build_records(args.root, cfg)
                if (
                    snap.get("files") == current_hashes
                    and snap.get("model") == cfg.model_name
                    and snap.get("chunk_size") == cfg.chunk_size
                ):
                    print("Indexul este deja la zi.")
                    return
            except Exception:
                pass
        stats = do_index(args.root, cfg)
        print(json.dumps({"status": "ok", "stats": stats}, ensure_ascii=False, indent=2))
        return

    # --- SEARCH ---
    if args.cmd == "search":
        if not have_index(args.root):
            print("Nu există index. Rulează întâi: python ragctl.py index --root k8s", file=sys.stderr)
            sys.exit(2)
        types = [t.strip() for t in args.types.split(",") if t.strip()] if args.types else [t.strip() for t in cfg.default_types.split(",")]
        if args.weights:
            w_parts = [float(x) for x in args.weights.split(",")]
            w = (w_parts[0], w_parts[1])
        else:
            w = cfg.weights_tuple()
        out = hybrid_search(
            args.root, args.q, args.k, types, args.component, args.profile, args.max_per_file, w, cfg
        )
        if args.json:
            print(json.dumps(out, ensure_ascii=False, indent=2))
        else:
            print(f"Query: {out['query']}")
            print(f"Filtre: {out['filters']}")
            print("Rezultate:")
            for i, r in enumerate(out["results"], 1):
                tag = f"[{r['component']}|{r['type']}" + (f"|{r['profile']}" if r['profile'] else "") + "]"
                print(f"{i}. {r['score']:.4f} {tag} {r['path']}\n   {r['snippet']}")
            print("Stats:", out["stats"])
        return

    # --- PLAN ---
    if args.cmd == "plan":
        if not have_index(args.root):
            print("Nu există index. Rulează întâi: python ragctl.py index --root k8s", file=sys.stderr)
            sys.exit(2)

        types = [t.strip() for t in args.types.split(",") if t.strip()] if args.types else [t.strip() for t in cfg.default_types.split(",")]

        out = plan_do(
            root=args.root,
            query=args.q,
            types=types,
            k=args.k,
            have_index_fn=have_index,
            hybrid_search_fn=lambda *a, **kw: hybrid_search(*a, **{**kw, "cfg": cfg}),
            load_index_fn=load_index,
            extract_keys_fn=extract_yaml_keys_from_file,
            model_name=cfg.model_name,
            weights={
                "semantic": cfg.plan_w_sem,
                "search":   cfg.plan_w_search,
                "affinity": cfg.plan_w_affinity,
            },
        )

        if args.json:
            print(json.dumps(out, ensure_ascii=False, indent=2))
        else:
            tgt = out["target"]["component"] if out.get("target") else "-"
            print(f"Plan version: {out.get('plan_version')}")
            print(f"Target component: {tgt}")
            print("Values files:")
            for vf in out.get("values_files", []):
                print(f" - {vf['path']} (profile={vf.get('profile')}, keys={vf.get('keys_count')})")
        return

    # --- PATCH ---
    if args.cmd == "patch":
        if not have_index(args.root):
            print("Nu există index. Rulează întâi: python ragctl.py index --root k8s", file=sys.stderr)
            sys.exit(2)

        types = [t.strip() for t in args.types.split(",") if t.strip()] if args.types else [t.strip() for t in cfg.default_types.split(",")]

        # 1) PLAN (folosește plan_do)
        plan = plan_do(
            root=args.root,
            query=args.q or "default",
            types=types,
            k=args.k,
            have_index_fn=have_index,
            hybrid_search_fn=lambda *a, **kw: hybrid_search(*a, **{**kw, "cfg": cfg}),
            load_index_fn=load_index,
            extract_keys_fn=extract_yaml_keys_from_file,
            model_name=cfg.model_name,
            weights={
                "semantic": cfg.plan_w_sem,
                "search":   cfg.plan_w_search,
                "affinity": cfg.plan_w_affinity,
            },
        )

        # sanity: target component
        target_comp = (plan.get("target") or {}).get("component")
        if not target_comp:
            print("Plannerul nu a putut alege o componentă-țintă (niciun rezultat relevant).", file=sys.stderr)
            sys.exit(3)

        # 2) intenții din CLI
        intents = []
        for kv in args.sets:
            if "=" not in kv:
                print(f"--set invalid: {kv} (folosește key=value)", file=sys.stderr)
                sys.exit(3)
            k, v = kv.split("=", 1)
            intents.append({"op": "set", "key": k.strip(), "value": v.strip()})
        for k in args.enables:
            intents.append({"op": "enable", "key": k.strip()})
        for k in args.disables:
            intents.append({"op": "disable", "key": k.strip()})

        if not intents:
            print("Nicio modificare specificată; folosește --set/--enable/--disable.", file=sys.stderr)
            sys.exit(3)

        # 3) patch-uri
        out = patch_from_plan(plan, intents, args.out_dir)

        # 4) output
        if args.json:
            print(json.dumps({
                "plan_version": plan.get("plan_version"),
                "component": target_comp,
                "intents": intents,
                "patch_result": out
            }, ensure_ascii=False, indent=2))
        else:
            print(f"Component: {target_comp} (planner={plan.get('plan_version')})")
            for p in out["patches"]:
                print(f"- {p['patch_yaml_path']} -> {p['values_path']} (profile={p['profile']})")
            if out["apply_commands"]:
                print("\nComenzi yq (merge determinist):")
                for c in out["apply_commands"]:
                    print(c)
        return

    # --- TRANSLATE ---
    if args.cmd == "translate":
        if not have_index(args.root):
            print("Nu există index. Rulează întâi: python ragctl.py index --root k8s", file=sys.stderr)
            sys.exit(2)

        types = [t.strip() for t in args.types.split(",") if t.strip()] if args.types else [t.strip() for t in cfg.default_types.split(",")]

        # 1) PLAN
        plan = plan_do(
            root=args.root,
            query=args.q,
            types=types,
            k=args.k,
            have_index_fn=have_index,
            hybrid_search_fn=lambda *a, **kw: hybrid_search(*a, **{**kw, "cfg": cfg}),
            load_index_fn=load_index,
            extract_keys_fn=extract_yaml_keys_from_file,
            model_name=cfg.model_name,
            weights={
                "semantic": cfg.plan_w_sem,
                "search":   cfg.plan_w_search,
                "affinity": cfg.plan_w_affinity,
            },
        )

        # If requested, print the LLM prompt and exit (human-readable)
        if args.provider == "llm" and getattr(args, "show_prompt", False):
            prompt = build_llm_prompt(args.q, plan)
            print(prompt)
            return

        target_comp = (plan.get("target") or {}).get("component")
        if not target_comp:
            print("Plannerul nu a putut alege o componentă-țintă (niciun rezultat relevant).", file=sys.stderr)
            sys.exit(3)

        # 2) NL -> intenții
        out = translate_intents(args.q, plan, provider=args.provider, allow_new_keys=args.allow_new_keys)

        # Optionally render final manifest YAML
        if getattr(args, "render_final", False):
            mf0 = next((m for m in out.get("manifests", []) if m.get("op") in ("create","suggest")), None)
            if mf0:
                final_yaml = _build_final_manifest(mf0, out.get("intents", []))
                if args.yaml_only:
                    print(final_yaml)
                    return
                else:
                    print("\nFinal YAML:")
                    print(final_yaml)

        # 3) output
        if args.json:
            print(json.dumps({"plan_version": plan.get("plan_version"),
                            "plan_target": plan.get("target"),
                            "translate": out}, ensure_ascii=False, indent=2))
        else:
            _pretty_print_translate(plan, out)

    # --- AUTOPATCH ---
    if args.cmd == "autopatch":
        if not have_index(args.root):
            print("Nu există index. Rulează întâi: python ragctl.py index --root k8s", file=sys.stderr)
            sys.exit(2)

        types = [t.strip() for t in args.types.split(",") if t.strip()] if args.types else [t.strip() for t in cfg.default_types.split(",")]

        # 1) PLAN
        plan = plan_do(
            root=args.root,
            query=args.q,
            types=types,
            k=args.k,
            have_index_fn=have_index,
            hybrid_search_fn=lambda *a, **kw: hybrid_search(*a, **{**kw, "cfg": cfg}),
            load_index_fn=load_index,
            extract_keys_fn=extract_yaml_keys_from_file,
            model_name=cfg.model_name,
            weights={
                "semantic": cfg.plan_w_sem,
                "search":   cfg.plan_w_search,
                "affinity": cfg.plan_w_affinity,
            },
        )

        # If requested, print the LLM prompt and exit early
        if args.provider == "llm" and getattr(args, "show_prompt", False):
            prompt = build_llm_prompt(args.q, plan)
            print(prompt)
            return

        target_comp = (plan.get("target") or {}).get("component")
        if not target_comp:
            print("Plannerul nu a putut alege o componentă-țintă (niciun rezultat relevant).", file=sys.stderr)
            sys.exit(3)

        # 2) NL -> intenții
        tr = translate_intents(args.q, plan, provider=args.provider, allow_new_keys=args.allow_new_keys)

        # Optionally render final manifest YAML
        if getattr(args, "render_final", False):
            mf0 = next((m for m in tr.get("manifests", []) if m.get("op") in ("create","suggest")), None)
            if mf0:
                final_yaml = _build_final_manifest(mf0, tr.get("intents", []))
                if args.yaml_only:
                    print(final_yaml)
                    return
                else:
                    print("\nFinal YAML:")
                    print(final_yaml)

        # 3) patch DOAR din intențiile acceptate
        outp = patch_from_plan(plan, tr["intents"], args.out_dir)

        payload = {
            "plan_version": plan.get("plan_version"),
            "component": target_comp,
            "translate": tr,
            "patch_result": outp
        }

        # 4) output
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            _pretty_print_autopatch(payload)

if __name__ == "__main__":  
    main()