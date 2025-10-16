import os
import re
from typing import Dict, List, Tuple, Set

def _intent_keywords(req_info: Dict) -> Set[str]:
	kw: Set[str] = set()
	allowed = req_info.get("allowed") or set()
	for k in allowed:
		kw.add(str(k).lower())
	if req_info.get("metrics"):
		kw.update({"metrics", "monitoring", "path", "port"})
	if req_info.get("hosts"):
		kw.update({"ingresses", "hosts", "global"})
	if req_info.get("volume_hint") or ("volumes" in allowed):
		kw.update({"volumes", "mountpath", "emptydir"})
	if req_info.get("image_repo"):
		kw.update({"image", "repository"})
	# common extras frequently relevant in Helm values
	kw.update({"probes", "readiness", "liveness", "configmaps", "secrets"})
	return kw

def _compress_chunk(text: str, kws: Set[str], min_lines: int = 40, max_lines: int = 220) -> str:
	if not text:
		return text
	lines = text.splitlines()
	# always keep the header lines if present
	header = []
	body = []
	for i, ln in enumerate(lines[:3]):
		if ln.startswith(("FILE:", "SECTION:", "# file:", "# score=")):
			header.append(ln)
	# keyword-focused filtering (case-insensitive)
	kws_l = {k.lower() for k in kws}
	for ln in lines:
		low = ln.lower()
		if any(k in low for k in kws_l):
			body.append(ln)
	# if too sparse, fallback to the first N lines
	if len(body) < min_lines:
		body = lines[:min(max_lines, len(lines))]
	# final cap
	if len(body) > max_lines:
		body = body[:max_lines]
	return "\n".join(header + body).strip()

def _budget() -> Tuple[int, int]:
	max_chars = int(os.getenv("CTX_MAX_CHARS", os.getenv("RET_MAX_CHARS", "12000")))
	max_lines = int(os.getenv("CTX_MAX_LINES", os.getenv("RET_MAX_LINES", "350")))
	return max_chars, max_lines

def _schema_cap() -> int:
	return int(os.getenv("CTX_MAX_SCHEMA", "3"))

def _example_cap() -> int:
	return int(os.getenv("CTX_MAX_EXAMPLES", "6"))

def _dedupe(parts: List[str]) -> List[str]:
	seen = set()
	out: List[str] = []
	for p in parts:
		key = re.sub(r"\s+", " ", (p or "")).strip().lower()
		if key and key not in seen:
			seen.add(key)
			out.append(p)
	return out

def compose_context(retrieved: List[Tuple[float, Dict, str]],
                    schema_summary: Dict[str, Dict],
                    req_info: Dict,
                    verbose: bool = False) -> str:
	kws = _intent_keywords(req_info)
	max_chars, max_lines = _budget()
	max_schema = _schema_cap()
	max_examples = _example_cap()

	# Separate schema vs examples; keep best per file+section for examples
	schema_chunks: List[str] = []
	examples: List[Tuple[float, str, str, str]] = []  # (score, file, section, text)

	for score, meta, text in retrieved:
		section = str(meta.get("section"))
		file = str(meta.get("file"))
		if section == "__schema__":
			schema_chunks.append(text)
		else:
			examples.append((float(score), file, section, text))

	# Take top schema chunks up to cap; if none, synthesize from schema_summary
	schema_chunks = schema_chunks[:max_schema]
	if not schema_chunks and schema_summary:
		# synthesize a compact schema summary
		for f, outline in list(schema_summary.items())[:max_schema]:
			lines = [f"# file: {f}"]
			for top, children in outline.items():
				lines.append(f"- {top}: {', '.join(children[:12])}")
			schema_chunks.append("\n".join(lines))

	# For examples: keep only top per (file, section)
	per_key_best: Dict[Tuple[str, str], Tuple[float, str]] = {}
	for score, file, section, text in examples:
		key = (file, section)
		if key not in per_key_best or score > per_key_best[key][0]:
			per_key_best[key] = (score, text)
	# Re-rank by score and compress around intents
	best_examples = sorted(((s, t) for (_, (s, t)) in per_key_best.items()), key=lambda x: x[0], reverse=True)
	example_texts = []
	for s, t in best_examples[:max_examples * 2]:  # take a few extra before compression/dedupe
		example_texts.append(_compress_chunk(t, kws, min_lines=40, max_lines=220))
	example_texts = example_texts[:max_examples]

	# Compose final with budget
	parts: List[str] = []
	# Header for traceability
	parts.append("# schema-context")
	parts.extend(schema_chunks)
	parts.append("\n# exemplar-context")
	parts.extend(example_texts)
	parts = _dedupe([p for p in parts if p and p.strip()])

	# Enforce global budget
	out_lines: List[str] = []
	total_chars = 0
	for p in parts:
		for ln in p.splitlines():
			if len(out_lines) >= max_lines or total_chars >= max_chars:
				if verbose:
					print(f"[ctx] budget reached lines={len(out_lines)} chars={total_chars}")
				return "\n".join(out_lines).strip()
			out_lines.append(ln)
			total_chars += len(ln) + 1
	return "\n".join(out_lines).strip()
