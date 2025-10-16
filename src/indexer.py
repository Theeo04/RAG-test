import glob
import os
from typing import Dict, List, Optional, Any
from .parser import extract_sections, schema_outline
from .chunker import make_chunks
from .embeddings import embed_texts
from .store import LocalVectorStore

def discover_common_yaml(root: str) -> List[str]:
    pattern = os.path.join(root, "k8s", "**", "values", "common.yaml")
    return sorted(glob.glob(pattern, recursive=True))

def _dbg(msg: str):
    if os.getenv("RAG_DEBUG") == "1":
        print(f"[index] {msg}")

def _include_schema_flag(default: bool = True) -> bool:
    """
    Env-driven toggle to include schema chunks in the vector index.
    INDEX_INCLUDE_SCHEMA=0|false|no to exclude.
    """
    val = os.getenv("INDEX_INCLUDE_SCHEMA", "1" if default else "0").lower()
    return val not in ("0", "false", "no")

def build_index(root: str, out_path: str, include_schema: Optional[bool] = None) -> Dict:
    files = discover_common_yaml(root)
    _dbg(f"discovered files: {len(files)}")
    texts, metas = [], []
    schemas: Dict[str, Dict] = {}

    # resolve inclusion policy
    include_schema = _include_schema_flag(True) if include_schema is None else include_schema

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            content = f.read()
        sections = extract_sections(content)
        chunks = make_chunks(fp, content, sections)
        texts.extend([c["text"] for c in chunks])
        metas.extend([c["metadata"] for c in chunks])

        # always collect schema outline for separate JSON context
        schemas[fp] = schema_outline(content)

        # Optionally add a schema chunk into the vector index, but:
        # - don't include FILE/SECTION headers in the embedded text
        # - make it clearly marked as schema
        if include_schema:
            sch = schemas[fp]
            sch_lines = [f"- {top}: {', '.join(children[:12])}" for top, children in sch.items()]
            texts.append("SCHEMA CHUNK:\n" + "\n".join(sch_lines))
            metas.append({"file": fp, "section": "__schema__"})
    _dbg(f"total chunks: {len(texts)}")

    vecs = embed_texts(texts)
    store = LocalVectorStore(out_path)
    store.add(texts, metas, vecs)
    store.save()
    _dbg(f"index saved: {out_path}")
    return {"files_indexed": len(files), "chunks": len(texts), "schemas": schemas}

class _FilteredVectorStore:
    """
    Lightweight wrapper to exclude schema chunks at query time without rebuilding.
    """
    def __init__(self, base: Any, include_schema: bool = True) -> None:
        self._base = base
        self._include_schema = include_schema

    def search(self, qvec, k: int):
        # request a bit more to compensate for filtering
        raw = self._base.search(qvec, k=max(k * 2, k + 4))
        if self._include_schema:
            return raw[:k]
        filtered = [(s, m, t) for (s, m, t) in raw if str(m.get("section")) != "__schema__"]
        return filtered[:k]

def load_index(out_path: str, include_schema: Optional[bool] = None):
    """
    Load the index; optionally exclude schema chunks from retrieval via a wrapper.
    Use INDEX_INCLUDE_SCHEMA=0|false|no to disable schema at query time.
    """
    base = LocalVectorStore.load(out_path)
    include_schema = _include_schema_flag(True) if include_schema is None else include_schema
    if include_schema:
        return base
    return _FilteredVectorStore(base, include_schema=False)
