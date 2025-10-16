import re
import yaml
from typing import Dict, Optional, List

END_MARKER = "---END---"
KEY_RE = re.compile(r'^([A-Za-z0-9._-]+):(?:\s|$)')

def append_end_marker_rule(prompt: str) -> str:
	return prompt + "\n\nImportant: Output only YAML and end your output with the exact marker on a new line:\n" + END_MARKER + "\nDo not include the marker anywhere else."

def strip_end_marker(text: str) -> str:
	return text.replace(END_MARKER, "").strip() if text else text

def first_yaml_block(text: str) -> Optional[str]:
	lines = text.splitlines()
	for i, line in enumerate(lines):
		s = line.lstrip()
		if KEY_RE.match(s):
			block = "\n".join(lines[i:]).strip()
			return block or None
	return None

def extract_yaml(text: str) -> Optional[str]:
	if not text:
		return None
	m = re.search(r"```(?:yaml|yml)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
	if m:
		return m.group(1).strip()
	block = first_yaml_block(text)
	return block or text.strip()

def filter_yaml_top_sections(yaml_text: str, allowed: set[str]) -> str:
	lines = yaml_text.splitlines()
	out: List[str] = []
	skip = False
	for line in lines:
		if line and not line.startswith((" ", "\t")) and not line.lstrip().startswith("#"):
			m = KEY_RE.match(line)
			if m:
				key = m.group(1)
				skip = key not in allowed
		if not skip:
			out.append(line)
	return "\n".join(out).strip()

def looks_incomplete_yaml(text: str) -> bool:
	if not text:
		return True
	if END_MARKER not in text:
		lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
		if not lines:
			return True
		last = lines[-1]
		if last.endswith(":") or last.endswith("-") or last.endswith("..."):
			return True
	return False

def parse_yaml(text: str, verbose: bool = False) -> Optional[Dict]:
	def _strip_fences_and_marker(src: str) -> str:
		if not src:
			return src
		# Remove code fences if any slipped through
		m = re.search(r"```(?:yaml|yml)?\s*(.*?)```", src, re.DOTALL | re.IGNORECASE)
		if m:
			src = m.group(1)
		# Drop end marker and YAML doc end token
		src = src.replace(END_MARKER, "")
		# Remove leading YAML doc separators that can create multi-docs
		# We'll still parse with load_all, but this reduces noise
		return src.strip()

	def _sanitize(src: str) -> str:
		# 1) Comment out standalone Helm template lines (invalid scalars)
		tmpl_only = re.compile(r'^\s*\{\{[-\s]?.*?[-\s]?\}\}\s*$')
		lines = []
		for ln in (src or "").splitlines():
			if tmpl_only.match(ln):
				lines.append("# " + ln)
			else:
				lines.append(ln)
		src2 = "\n".join(lines)
		# 2) Quote unquoted {{ ... }} so YAML treats them as strings
		src2 = re.sub(r'(?<!")(\{\{[^}]+}})(?!")', r'"\1"', src2)
		return src2

	def _merge_docs(docs: List[Dict]) -> Optional[Dict]:
		if not docs:
			return None
		out: Dict = {}
		for d in docs:
			if isinstance(d, dict):
				out.update(d)
		return out or None

	try:
		clean = _strip_fences_and_marker(text)
		# First try parsing as multiple documents and merge dicts
		all_docs = list(yaml.safe_load_all(clean))
		if all_docs:
			merged = _merge_docs(all_docs)
			if merged is not None:
				return merged
		# Fallback to single-doc
		doc = yaml.safe_load(clean)
		return doc if isinstance(doc, dict) else None
	except Exception as e:
		if verbose:
			preview = (text[:300] + "...") if len(text) > 300 else text
			print(f"[yaml] parse error: {e}\n[yaml] preview:\n{preview}")
		try:
			clean = _strip_fences_and_marker(text)
			sanitized = _sanitize(clean)
			# Retry with multi-doc merging after sanitization
			all_docs = list(yaml.safe_load_all(sanitized))
			if all_docs:
				merged = _merge_docs(all_docs)
				if merged is not None:
					return merged
			doc = yaml.safe_load(sanitized)
			return doc if isinstance(doc, dict) else None
		except Exception as e2:
			if verbose:
				print(f"[yaml] retry parse failed: {e2}")
			return None
