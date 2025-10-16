from typing import Dict, List, Tuple, Set

from .utils.yaml_utils import KEY_RE

def schema_union(schema_summary: Dict[str, Dict]) -> Dict[str, Set[str]]:
	union: Dict[str, Set[str]] = {}
	for _, outline in (schema_summary or {}).items():
		for top, children in outline.items():
			union.setdefault(top, set()).update(children)
	return union

def canonical_paths(union: Dict[str, Set[str]]) -> Dict[str, str]:
	paths = {}
	if "global" in union and "app" in union["global"]:
		paths["metrics"] = "global.app.monitoring"
	elif "global" in union:
		paths["metrics"] = "global.monitoring"
	else:
		paths["metrics"] = "global.app.monitoring"
	paths["ingresses"] = "global.ingresses.main.hosts"
	paths["image"] = "image"
	paths["readiness"] = "probes.readiness"
	return paths

def collect_context_top_keys(retrieved: List[Tuple[float, Dict, str]]) -> Set[str]:
	tops: Set[str] = set()
	for _, _, text in retrieved:
		for line in text.splitlines():
			if not line or line.lstrip().startswith("#") or line.startswith((" ", "\t")):
				continue
			m = KEY_RE.match(line)
			if m:
				k = m.group(1)
				if k not in {"FILE", "SECTION"}:
					tops.add(k)
	return tops

def format_context(retrieved: List[Tuple[float, Dict, str]], schema_summary: Dict[str, Dict]) -> str:
	parts: List[str] = []
	for score, meta, text in retrieved:
		parts.append(f"# score={score:.3f} file={meta.get('file')} section={meta.get('section')}\n{text}")
	parts.append("\n# schema-summary\n")
	for f, sch in schema_summary.items():
		parts.append(f"# file: {f}")
		for top, children in sch.items():
			ck = ", ".join(children[:8])
			parts.append(f"# {top}: {ck}")
	return "\n".join(parts)
