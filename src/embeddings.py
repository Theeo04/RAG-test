import os
import numpy as np
from typing import List
from tqdm import tqdm
import requests

_ST_MODEL = None

def _dbg(msg: str) -> None:
    if os.getenv("RAG_DEBUG") == "1":
        print(f"[embed] {msg}")

def _init_st():
    global _ST_MODEL
    if _ST_MODEL is not None:
        return _ST_MODEL
    # Force CPU by default; allow override via ST_DEVICE
    device = os.getenv("ST_DEVICE", "cpu").lower()
    if device == "cpu":
        # Hide CUDA devices so torch doesn't try to use an incompatible GPU
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from sentence_transformers import SentenceTransformer
    try:
        _ST_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    except Exception:
        # Fallback to CPU if requested device fails
        _ST_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    return _ST_MODEL

def _ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

def _ollama_embed_model() -> str:
    return os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

def _ollama_embed_timeout() -> int:
    try:
        return int(os.getenv("OLLAMA_EMBED_TIMEOUT", "20"))
    except Exception:
        return 20

def _ollama_model_installed(model: str) -> bool:
    """
    Check if the embedding model is already installed in Ollama to avoid long auto-pulls.
    """
    try:
        url = _ollama_base_url().rstrip("/") + "/api/tags"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json() or {}
        models = [m.get("name") for m in (data.get("models") or [])]
        ok = any(str(m or "").split(":")[0] == str(model).split(":")[0] for m in models)
        _dbg(f"ollama tags contains {model}: {ok}")
        return ok
    except Exception as e:
        _dbg(f"ollama tags check failed: {e}")
        return False

def _ollama_embeddings(texts: List[str]) -> np.ndarray | None:
    """
    Use Ollama /api/embeddings with a local embedding model (e.g., nomic-embed-text).
    Returns normalized embeddings or None on failure.
    """
    # Allow disabling via env
    if os.getenv("OLLAMA_EMBED_DISABLE", "0").lower() in ("1", "true", "yes"):
        _dbg("ollama embeddings disabled by env")
        return None

    url = _ollama_base_url().rstrip("/") + "/api/embeddings"
    model = _ollama_embed_model()

    # Fast-skip if model not installed (to avoid long model pulls)
    if not _ollama_model_installed(model):
        _dbg(f"ollama embed model not installed: {model} (skipping)")
        return None

    out = []
    timeout = _ollama_embed_timeout()
    try:
        sess = requests.Session()
        for t in tqdm(texts, desc="embedding (ollama)", disable=len(texts) < 5):
            r = sess.post(url, json={"model": model, "input": t}, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            vec = np.array(data["embedding"], dtype=np.float32)
            n = np.linalg.norm(vec) + 1e-9  # normalize to cosine
            out.append(vec / n)
        _dbg(f"used ollama model={model}, n={len(out)}")
        return np.vstack(out)
    except Exception as e:
        _dbg(f"ollama embeddings failed: {e}")
        return None

def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Returns (n, d) embeddings.
    Backends can be ordered via EMBED_BACKENDS env (comma list): e.g., "st" or "ollama,st".
    Default order: ollama,st
    """
    order = [s.strip().lower() for s in os.getenv("EMBED_BACKENDS", "ollama,st").split(",") if s.strip()]
    if not order:
        order = ["ollama", "st"]

    # Try Ollama if requested
    if "ollama" in order:
        vecs = _ollama_embeddings(texts)
        if vecs is not None:
            return vecs

    # Fallback to sentence-transformers
    if "st" in order:
        model = _init_st()
        _dbg("used sentence-transformers: all-MiniLM-L6-v2")
        vecs = model.encode(texts, normalize_embeddings=True)
        return np.array(vecs, dtype=np.float32)

    # If no backend succeeded
    raise RuntimeError("No embedding backend available (check EMBED_BACKENDS/OLLAMA settings)")

def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]
