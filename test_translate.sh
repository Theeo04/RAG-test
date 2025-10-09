#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-k8s}"
Q="${2:-creeaza un deployment de nginx cu initContainer; seteaza replicas=2; limiteaza CPU la 750m si memoria la 512Mi}"
MODE="${3:-run}"  # use "prompt" to show LLM prompt, "yaml" for final manifest only

# Optional: set Ollama if you want --provider llm
export OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434/api/chat}"
export OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.1}"

# Dynamic behavior can also be tuned via env:
export RAG_DEFAULT_IMAGE="${RAG_DEFAULT_IMAGE:-nginx:latest}"
export RAG_CREATE_WORDS="${RAG_CREATE_WORDS:-creeaza,creaza,creați,create,adauga,adaugă,add,genereaza,generează}"
export RAG_WORKLOAD_KINDS="${RAG_WORKLOAD_KINDS:-deployment,statefulset,daemonset,job,cronjob}"

# 1) Build index
python3 ragctl.py index --root "$ROOT"

# 2) Translate, show prompt, or render YAML
if [[ "$MODE" == "prompt" ]]; then
  echo "----- LLM Prompt (human-readable) -----"
  python3 ragctl.py translate \
    --root "$ROOT" \
    --q "$Q" \
    --types values,manifest --k 12 \
    --provider llm \
    --allow-new-keys \
    --show-prompt
  echo "---------------------------------------"
elif [[ "$MODE" == "yaml" ]]; then
  python3 ragctl.py translate \
    --root "$ROOT" \
    --q "$Q" \
    --types values,manifest --k 12 \
    --provider llm \
    --allow-new-keys \
    --render-final \
    --yaml-only
else
  python3 ragctl.py translate \
    --root "$ROOT" \
    --q "$Q" \
    --types values,manifest --k 12 \
    --provider llm \
    --allow-new-keys \
    --render-final
fi
