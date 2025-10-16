# final-hack-rag

A tool that turns short requests into Helm `values/common.yaml`.  
It learns from real examples, keeps Helm templates (like `{{ ... }}`), and fills in missing pieces when needed.

## What it does

- Finds and reads real `values/common.yaml` files.
- Pulls the most useful examples for your request.
- Builds a compact context for the LLM (schema first, examples second).
- Detects your intent (image, hosts, metrics, volumes, probes, configMaps).
- Asks the LLM to write YAML.
- Parses YAML carefully and keeps Helm templates intact.
- Adds only what you asked for, plus what is required to be valid.
- Falls back to a safe structure if context or output is weak.

## How it works (flow)

1) Scan and index examples
- What: Scan the repo for `k8s/**/values/common.yaml`, split them into chunks (per section + whole file), and build a small “schema outline” (top keys and their children) for each file.
- Why: The LLM learns style and structure from real data; schema outlines make it easier to follow chart conventions.
- Tech:
  - Python (glob, file IO)
  - src/parser.py + src/chunker.py to extract sections and create chunks
  - src/indexer.py to build the index and schema summary (JSON)

2) Create embeddings
- What: Turn each chunk into a numeric vector so we can search by meaning, not just words.
- Why: Finds relevant examples even if the wording is different.
- Tech:
  - Ollama /api/embeddings (default) with model like nomic-embed-text
  - Fallback: SentenceTransformers (all-MiniLM-L6-v2) on CPU
  - NumPy for normalization
  - tqdm for progress bars

3) Store vectors
- What: Save all vectors and their metadata in a local file for fast retrieval.
- Why: Avoid re-computing embeddings every time.
- Tech:
  - src/store.py LocalVectorStore (pickled index on disk)
  - data/index.pkl and data/schemas.json

4) Retrieve good context
- What: Search the index for the most useful chunks for the user request.
- Why: Give the LLM the right examples and schema, not random text.
- Tech:
  - Cosine similarity on normalized vectors
  - MMR diversification to reduce duplicates (src/retriever.py)
  - Keyword and schema boosts (intents guide which keys matter)
  - Ensure at least one per-file “__schema__” chunk
  - Trim long chunks to fit prompt budget
  - Env knobs: RET_INITIAL_MULT, RET_MMR_LAMBDA, RET_MAX_LINES/CHARS

5) Compose a compact context
- What: Build a small, readable context: schema first, then a few focused example lines that match the request (hosts, probes, volumes, etc.).
- Why: Short, targeted context helps the LLM stay on style and avoid noise.
- Tech:
  - src/context.py: intent-aware compression, dedupe, strict budgets
  - Env knobs: CTX_MAX_SCHEMA, CTX_MAX_EXAMPLES, CTX_MAX_LINES/CHARS

6) Understand user intent
- What: Extract a small JSON of what to generate: allowed top-level keys, image repo:tag, hosts, metrics, volume mount, probe path.
- Why: Keep the output focused and structurally correct.
- Tech:
  - Ollama chat/generate via a small JSON-only prompt (src/intents.py)
  - Fallback: regex-based extractor when LLM is off
  - Env: USE_LLM_INTENTS

7) Ask the LLM to write YAML
- What: Send a system prompt + the composed context + the user spec.
- Why: The LLM mirrors the style and schema from examples.
- Tech:
  - Ollama /api/chat (preferred) or /api/generate (auto fallback)
  - Models: llama3.1 variants (configurable)
  - Env: OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL, OLLAMA_MODE

8) Keep YAML valid and preserve Helm templates
- What: If YAML contains Helm templates (`{{ ... }}`) and lenient mode is on, return it as-is. Otherwise, parse it safely.
- Why: Helm templates can confuse parsers; we prefer to keep them intact.
- Tech:
  - src/utils/yaml_utils.py + PyYAML
  - Lenient parse: comment out standalone template-only lines, quote inline `{{ ... }}`, support multi-document YAML and merge dicts
  - Env: LENIENT_YAML

9) Post-process and fill required parts
- What: Keep only requested sections, follow schema rules, and ensure required nodes exist (e.g., hosts under `global.ingresses`, minimal `volumes`).
- Why: Produce minimal, correct YAML that fits the chart.
- Tech:
  - Pruning and “ensure required parts” in src/generator.py
  - Canonical paths: metrics, ingresses, image, readiness

10) Self-heal when context is weak
- What: If retrieval is weak or output is thin, synthesize a small schema-guided skeleton and merge in user intents.
- Why: Always produce usable YAML, even if the input is poor.
- Tech:
  - src/fallbacks.py provides minimal/skeleton fallbacks
  - Merge paths for image, ingresses, metrics, probes, volumes
  - Env: RAG_FALLBACK

## Main parts

- src/indexer.py: builds the vector index and schema.
- src/retriever.py: finds and ranks good example chunks.
- src/context.py: assembles a small, relevant context for the LLM.
- src/intents.py: extracts user intent (LLM + regex fallback).
- src/generator.py: prompts the LLM, parses YAML, applies schema rules, and falls back if needed.
- src/utils/yaml_utils.py: YAML helpers, lenient parsing with Helm template support.
- src/embeddings.py: embeddings via Ollama first, SentenceTransformers as fallback.
- src/app.py: command-line interface.

## Setup

Requirements
- Python 3.10+
- Ollama running (local or remote)

Install
- pip install -r requirements.txt

Pull models (examples)
- ollama pull llama3.1:8b-instruct-q4_K_M
- ollama pull nomic-embed-text

Environment
- OLLAMA_BASE_URL=http://popp-llm01.server.lan:11434
- OLLAMA_CHAT_MODEL=llama3.1:8b-instruct-q4_K_M
- OLLAMA_EMBED_MODEL=nomic-embed-text
- OLLAMA_MODE=auto              # chat | generate | auto
- OLLAMA_TIMEOUT=600
- LENIENT_YAML=1                # keep raw YAML if it has Helm templates
- USE_LLM_INTENTS=1
- RAG_DEBUG=0
- ST_DEVICE=cpu

Retrieval/context knobs
- RET_INITIAL_MULT=4
- RET_MMR_LAMBDA=0.7
- RET_KW_WEIGHT=0.05
- RET_SCHEMA_WEIGHT=0.02
- RET_MAX_LINES=250
- RET_MAX_CHARS=12000
- CTX_MAX_SCHEMA=3
- CTX_MAX_EXAMPLES=6
- CTX_MAX_LINES=350
- CTX_MAX_CHARS=12000

## Usage

Build index
- python3 -m src.app ingest -v

See retrieved examples
- python3 -m src.app query --q "ingress + configMap + probes" -k 6 -v

Generate YAML
- python3 -m src.app generate --q "Create a values/common.yaml for a small apache-benchmark image which runs an echo script and sleeps, exposed by an ingress and with a file copied via a configMap" --out ./generated/common.yaml --verbose

Ollama generate (example)
- curl "$OLLAMA_BASE_URL/api/generate" -H 'Content-Type: application/json' -d '{
    "model": "'"$OLLAMA_CHAT_MODEL"'",
    "prompt": "Ce film este pe primul loc in TOP 250 IMDB?",
    "stream": false
  }'

## Why it’s reliable

- Diverse retrieval: fewer duplicates, more useful lines.
- Intent-aware context: only what the prompt needs.
- Lenient YAML parser: keeps `{{ ... }}` intact and merges multi-docs.
- Template-preserving mode: skip parsing and return raw YAML when needed.
- Schema pruning + “ensure required parts”: valid and minimal.
- Skeleton fallback: fills gaps when context or output is weak.

## Tips and troubleshooting

- Bad endpoint or model? Check OLLAMA_BASE_URL and OLLAMA_CHAT_MODEL. Try OLLAMA_MODE=generate.
- Parse errors? Set LENIENT_YAML=1 to return raw YAML as-is.
- Empty results? Rebuild the index: python3 -m src.app ingest.
- No embeddings? It falls back to SentenceTransformers on CPU.

## Conclusion

This tool turns short specs into Helm-friendly `values/common.yaml`.  
It learns from your examples, keeps Helm templates safe, and stays robust even when context is thin.
