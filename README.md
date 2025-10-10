RAG for Kubernetes (/k8s) with Llama3.1 via Ollama

Prereqs
- Python 3.10+
- Ollama installed and llama3.1 model pulled:
  - ollama pull llama3.1:latest

Install
- python -m venv .venv && source .venv/bin/activate
- pip install -r requirements.txt

Index
- python rag_cli.py index --root /path/to/repo --k8s k8s
  - Uses SentenceTransformers embedding model: all-MiniLM-L6-v2 (default)
  - Add --verbose to print index summary.

Ask
- python rag_cli.py ask "Increase CPU for api component to 200m and set replicas to 3" --root /path/to/repo --k8s k8s --patches-dir /tmp/patches
  - Uses Ollama LLM: llama3.1:latest (default)
  - On success prints "Final YAML (copy-paste)" that merges patches into target values files.
  - Add --verbose to print hybrid-search stats, components ranking, selected targets, and values files.

Troubleshooting
- Error like "No sentence-transformers model found with name sentence-transformers/llama3.1:latest":
  - You passed an Ollama tag as the embedding model. Use --embed-model all-MiniLM-L6-v2 (default) or another HF embedding model.

Notes
- The planner uses semantic + lexical fusion to select the target component and values files.
- If the model outputs an intents JSON block, patch YAMLs are created in --patches-dir and a merged YAML is printed for copy-paste.
- Edit weights or embedding model in plan_module if needed.

## Implementation roadmap (retrieval, planning, patching)

This section lists small, incremental tasks you can implement quickly. Each item points to the exact place to change.

### Retrieval quality
- Intent-aware weighting (quick win)
  - Where: rag_backend.hybrid_search_fn
  - Task: Detect query intent (replicas/cpu/memory/ingress/image) and reweight combo = α*sem + (1-α)*lex (e.g., replicas → boost lex).
  - How: Simple regex map in hybrid_search_fn before combo calc.
- BM25 alongside TF-IDF
  - Where: rag_backend.hybrid_search_fn
  - Task: Add a second vectorizer (e.g., rank_bm25 via rank_bm25 package) or sklearn’s Tfidf with binary + idf tweak; fuse with sem score.
- MMR diversify top results
  - Where: rag_backend.hybrid_search_fn
  - Task: After sorting by combo, greedily pick results penalizing similarity by path/component to reduce duplicates.
- Field-aware boosts
  - Where: rag_backend._text_for_item
  - Task: Prefix tokens like kind:Deployment, image:repo/name, profile:prod to improve lexical hits; optionally boost kinds/images via feature weights when scoring.

### Planning
- Multi-component selection threshold (already supported via --top-components, --min-score)
  - Where: plan_module.do_plan
  - Task: Tune defaults: top_components=2 or min_score=0.5 for broader planning; expose via CLI defaults if helpful.
- Synonyms/normalization
  - Where: plan_module._affinity_flags_from_query
  - Task: Extend regex/synonym list: “scale”, “pods”, “hpa”, “limits/requests”, “ram” → map to cpu/memory/replicas flags.
- Profile affinity
  - Where: plan_module._component_docs and do_plan
  - Task: Parse prod/dev/stage hints in query and prefer matching profiles when ranking values files (extra bonus in _all_values_files_for_component).

### Patching
- Quantity validation
  - Where: patch_module.parse_simple_value and build_patches_from_intents
  - Task: Add validation for CPU (e.g., ^\d+m$|^\d+(\.\d+)?$) and memory (e.g., ^\d+(Mi|Gi)$). Reject/normalize invalid values with a clear error.
- Safety checks
  - Where: patch_module.build_patches_from_intents
  - Task: Guardrails (e.g., replicas >= 1). If violated, append a warning in the return notes and skip that intent.
- Conflict detection
  - Where: patch_module.build_patches_from_intents
  - Task: Detect duplicate/contradictory intents for same key across files; consolidate or report conflicts in notes.

### Observability and UX
- JSON output option
  - Where: rag_cli.py (cmd_ask)
  - Task: Add --out json|text. When json, print plan + LLM status + patch plan as a single JSON for automation.
- Retries/backoff for LLM
  - Where: rag_cli._gen_answer
  - Task: Add simple retry with exponential backoff and jitter; log each attempt when --verbose.
- Streaming
  - Where: rag_cli._gen_answer
  - Task: Use ollama.chat with stream=True; print tokens as they arrive; keep final content aggregation for intents parsing.

### Testing (quick to add)
- Unit tests for core utils
  - dotted_to_nested, deep_merge, parse_simple_value: ensure lists/dicts merge correctly.
  - Where: tests/test_patch_utils.py
- Planning fusion tests
  - Where: tests/test_planner.py
  - Task: Feed synthetic meta and verify component ranking order under different weights.
- Retrieval smoke test
  - Where: tests/test_retrieval.py
  - Task: Build a tiny /k8s fixture and assert hybrid_search_fn returns expected kinds/values hits.

### Small tasks you can do in minutes
- Add intents-only prompt mode
  - Where: rag_cli._build_user_prompt
  - Task: Offer a flag to request strict JSON with only intents for safer patch generation.
- Increase keys_sample
  - Where: plan_module.do_plan (values_files processing)
  - Task: Make keys_sample configurable via CLI (e.g., --keys-sample 30).
- Print manifest kinds/images in verbose
  - Where: rag_cli (verbose section)
  - Task: Show kinds/images for top_results to help reasoning.

Tip: Implement these behind flags so defaults remain stable. Start with intent-aware weighting and validation; they give the highest quality and safety per line of code.
