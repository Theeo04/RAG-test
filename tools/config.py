from __future__ import annotations
import os
import pathlib
from typing import List, Tuple, Dict, Any, Optional

try:
    import yaml
except Exception:
    yaml = None  # no PyYAML, env-only config

def _csv_list(s: Optional[str]) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

def _csv_tuple_int(s: Optional[str], default: Tuple[int, int]) -> Tuple[int, int]:
    if not s:
        return default
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 2:
        return default
    try:
        return int(parts[0]), int(parts[1])
    except Exception:
        return default

class RAGConfig:
    def __init__(self, root: str, data: Dict[str, Any]):
        self.root = root

        # model + indexing
        self.model_name: str = str(data.get("model_name") or os.getenv("RAG_ENCODER", "all-MiniLM-L6-v2"))
        self.chunk_size: int = int(data.get("chunk_size") or os.getenv("RAG_CHUNK_SIZE", "1000"))
        self.rrf_k: int = int(data.get("rrf_k") or os.getenv("RAG_RRF_K", "60"))
        self.dense_weight: float = float(data.get("dense_weight") or os.getenv("RAG_DENSE_WEIGHT", "0.6"))
        self.bm25_weight: float = float(data.get("bm25_weight") or os.getenv("RAG_BM25_WEIGHT", "0.4"))

        # file discovery and classification
        self.file_globs: List[str] = data.get("file_globs") or _csv_list(os.getenv("RAG_FILE_PATTERNS")) or [
            "**/*.yaml", "**/*.yml", "**/*.md", "**/*.txt", "**/*.conf", "**/*.py"
        ]
        self.values_globs: List[str] = data.get("values_globs") or _csv_list(os.getenv("RAG_VALUES_GLOBS")) or [
            "**/values/*.yaml", "**/values/*.yml"
        ]
        self.manifest_globs: List[str] = data.get("manifest_globs") or _csv_list(os.getenv("RAG_MANIFEST_GLOBS")) or [
            "**/*.yaml", "**/*.yml"
        ]
        self.readme_globs: List[str] = data.get("readme_globs") or _csv_list(os.getenv("RAG_README_GLOBS")) or [
            "**/README.md", "**/readme.md"
        ]

        # component extraction (first path segment under root)
        self.component_from_root: bool = bool(data.get("component_from_root", True))

        # FAISS HNSW and TF-IDF settings
        self.hnsw_m: int = int(data.get("hnsw_m") or os.getenv("RAG_HNSW_M", "32"))
        self.hnsw_ef: int = int(data.get("hnsw_ef") or os.getenv("RAG_HNSW_EF", "200"))
        self.tfidf_ngram: Tuple[int, int] = data.get("tfidf_ngram") or _csv_tuple_int(os.getenv("RAG_TFIDF_NGRAM"), (1, 2))
        self.tfidf_min_df: int = int(data.get("tfidf_min_df") or os.getenv("RAG_TFIDF_MIN_DF", "1"))

        # search
        self.search_topN: int = int(data.get("search_topN") or os.getenv("RAG_SEARCH_TOPN", "200"))
        self.search_topM: int = int(data.get("search_topM") or os.getenv("RAG_SEARCH_TOPM", "200"))

        # planner weights
        self.plan_w_sem: float = float(data.get("plan_w_sem") or os.getenv("PLAN_W_SEM", "0.85"))
        self.plan_w_search: float = float(data.get("plan_w_search") or os.getenv("PLAN_W_SEARCH", "0.15"))
        self.plan_w_affinity: float = float(data.get("plan_w_affinity") or os.getenv("PLAN_W_AFFINITY", "0.2"))

        # defaults for CLI (types, k, etc.) — optional
        self.default_types: str = str(data.get("default_types") or os.getenv("RAG_DEFAULT_TYPES", "values,manifest"))

        # translate defaults
        self.default_image: str = str(data.get("default_image") or os.getenv("RAG_DEFAULT_IMAGE", "nginx:latest"))
        self.default_replicas: int = int(data.get("default_replicas") or os.getenv("RAG_DEFAULT_REPLICAS", "1"))

    @classmethod
    def from_root(cls, root: str) -> "RAGConfig":
        root = str(root)
        conf_path = os.getenv("RAG_CONFIG") or str(pathlib.Path(root) / ".rag" / "config.yaml")
        data: Dict[str, Any] = {}
        if yaml and pathlib.Path(conf_path).exists():
            try:
                with open(conf_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
            except Exception:
                data = {}
        return cls(root=root, data=data)

    def weights_tuple(self) -> Tuple[float, float]:
        return (self.dense_weight, self.bm25_weight)
