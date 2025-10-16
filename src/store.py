import os
import pickle
import numpy as np
from typing import Any, Dict, List, Tuple

class LocalVectorStore:
    def __init__(self, path: str):
        self.path = path
        self.texts: List[str] = []
        self.metas: List[Dict[str, Any]] = []
        self.vecs: np.ndarray | None = None

    def add(self, texts: List[str], metas: List[Dict[str, Any]], vecs: np.ndarray):
        if self.vecs is None:
            self.vecs = vecs
        else:
            self.vecs = np.vstack([self.vecs, vecs])
        self.texts.extend(texts)
        self.metas.extend(metas)

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump({
                "texts": self.texts,
                "metas": self.metas,
                "vecs": self.vecs
            }, f)

    @classmethod
    def load(cls, path: str) -> "LocalVectorStore":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        inst = cls(path)
        inst.texts = obj["texts"]
        inst.metas = obj["metas"]
        inst.vecs = obj["vecs"]
        return inst

    def search(self, qvec: np.ndarray, k: int = 5) -> List[Tuple[float, Dict[str, Any], str]]:
        if self.vecs is None or len(self.texts) == 0:
            return []
        # cosine similarity (vecs expected normalized if using ST; OpenAI not normalized -> normalize)
        A = self.vecs
        q = qvec
        if not np.isclose(np.linalg.norm(q), 1.0, atol=1e-3):
            q = q / (np.linalg.norm(q) + 1e-9)
            A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
        sims = (A @ q).flatten()
        idx = np.argsort(-sims)[:k]
        return [(float(sims[i]), self.metas[i], self.texts[i]) for i in idx]
