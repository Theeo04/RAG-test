import argparse
import json
import os
from typing import Dict

from .indexer import build_index, load_index
from .retriever import retrieve
from .parser import schema_outline
from .generator import generate_yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INDEX_PATH = os.path.join(ROOT, "data", "index.pkl")
SCHEMA_PATH = os.path.join(ROOT, "data", "schemas.json")

def _save_schemas(schemas: Dict):
    os.makedirs(os.path.dirname(SCHEMA_PATH), exist_ok=True)
    with open(SCHEMA_PATH, "w", encoding="utf-8") as f:
        json.dump(schemas, f, indent=2)

def ensure_index():
    if not os.path.exists(INDEX_PATH):
        print("Index not found. Building index...")
        res = build_index(ROOT, INDEX_PATH)
        _save_schemas(res.get("schemas", {}))

def load_or_build_schemas() -> Dict:
    if os.path.exists(SCHEMA_PATH):
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    from .indexer import discover_common_yaml
    schemas: Dict = {}
    for fp in discover_common_yaml(ROOT):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                schemas[fp] = schema_outline(f.read())
        except Exception:
            continue
    _save_schemas(schemas)
    return schemas

def cmd_ingest(args):
    res = build_index(ROOT, INDEX_PATH)
    _save_schemas(res.get("schemas", {}))
    print(json.dumps({k: v for k, v in res.items() if k != "schemas"}, indent=2))

def cmd_query(args):
    ensure_index()
    store = load_index(INDEX_PATH)
    if getattr(args, "verbose", False):
        print(f"[query] using index: {INDEX_PATH}")
        print(f"[query] k={args.k}")
    hits = retrieve(store, args.q, k=args.k)
    if getattr(args, "verbose", False):
        print(f"[query] hits: {len(hits)}")
    for score, meta, text in hits:
        print(f"score={score:.3f} file={meta.get('file')} section={meta.get('section')}")
        print("-" * 80)
        print(text)
        print("=" * 80)

def cmd_generate(args):
    ensure_index()
    # propagate fallback preference to generator without changing its signature
    os.environ["GEN_ALLOW_FALLBACK"] = "1" if getattr(args, "allow_fallback", False) else "0"
    store = load_index(INDEX_PATH)
    if getattr(args, "verbose", False):
        print(f"[gen] using index: {INDEX_PATH}")
        print(f"[gen] k={args.k}")
    hits = retrieve(store, args.q, k=args.k)
    if getattr(args, "verbose", False):
        print(f"[gen] retrieved {len(hits)} chunks")
    schemas: Dict = load_or_build_schemas()
    if getattr(args, "verbose", False):
        print(f"[gen] schemas loaded: {len(schemas)} files")
    yaml_text = generate_yaml(args.q, hits, schemas, verbose=getattr(args, "verbose", False))

    # Always show the generated YAML in console
    print(yaml_text)

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(yaml_text)
        print(f"Wrote {args.out}")

def main():
    p = argparse.ArgumentParser(prog="final-hack-rag")
    sub = p.add_subparsers(required=True)

    p_ing = sub.add_parser("ingest", help="index k8s/**/values/common.yaml")
    p_ing.add_argument("-v", "--verbose", action="store_true")
    p_ing.set_defaults(func=cmd_ingest)

    p_q = sub.add_parser("query", help="retrieve chunks")
    p_q.add_argument("--q", required=True, help="query text")
    p_q.add_argument("-k", type=int, default=5)
    p_q.add_argument("-v", "--verbose", action="store_true")
    p_q.set_defaults(func=cmd_query)

    p_gen = sub.add_parser("generate", help="generate a new values/common.yaml")
    p_gen.add_argument("--q", required=True, help="generation requirement")
    p_gen.add_argument("-k", type=int, default=6)
    p_gen.add_argument("--out", help="output file")
    p_gen.add_argument("--allow-fallback", action="store_true", default=True, help="allow minimal fallback on LLM failure")  # NEW
    p_gen.add_argument("-v", "--verbose", action="store_true")
    p_gen.set_defaults(func=cmd_generate)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
