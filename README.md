# RAG-IONOS-hackathon

## 1. Setup venv

```
# from your repo root
python3 -m venv .venv
# activate:
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows (PowerShell/CMD)
python -m pip install --upgrade pip wheel setuptools

pip install -r requirements.txt
```

## Commands:

```
# index:
python ragctl.py index --root k8s

python ragctl.py plan --root k8s --q "..." --types values,manifest --k 8 --json
```

Traducere + patch automat (cu Ollama):

```
export OLLAMA_MODEL=llama3.1
mkdir -p out
python ragctl.py autopatch --root k8s \
  --q "activeaza istio si limiteaza CPU la 500m" \
  --types values,manifest --k 8 \
  --provider llm --out out --json | tee out/autopatch.json
```