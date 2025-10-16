import re
from typing import Dict, List, Tuple

TopSections = Dict[str, Tuple[int, int]]  # name -> (start_line, end_line)

KEY_RE = re.compile(r'^([A-Za-z0-9._-]+):(?:\s|$)')
INDENTED_KEY_RE = re.compile(r'^(\s+)([A-Za-z0-9._-]+):(?:\s|$)')

def split_top_level_sections(text: str) -> TopSections:
    """
    Split by top-level keys (no leading whitespace).
    Returns mapping of section -> (start_line, end_line_exclusive).
    """
    lines = text.splitlines()
    sections: TopSections = {}
    order: List[str] = []
    for i, line in enumerate(lines):
        if not line or line.lstrip().startswith("#"):
            continue
        if line.startswith(" ") or line.startswith("\t"):
            continue
        m = KEY_RE.match(line)
        if m:
            sec = m.group(1)
            sections[sec] = (i, len(lines))  # temp end
            order.append(sec)
    # fix ends
    for idx, sec in enumerate(order):
        start, _ = sections[sec]
        end = sections[order[idx + 1]][0] if idx + 1 < len(order) else len(lines)
        sections[sec] = (start, end)
    return sections

def extract_sections(text: str) -> Dict[str, str]:
    lines = text.splitlines()
    spans = split_top_level_sections(text)
    return {k: "\n".join(lines[s:e]).rstrip() for k, (s, e) in spans.items()}

def schema_outline(text: str) -> Dict[str, List[str]]:
    """
    Build a compact schema: top-level key -> list of frequent second-level keys.
    Uses indentation heuristics (no strict YAML parsing).
    """
    lines = text.splitlines()
    sections = split_top_level_sections(text)
    outline: Dict[str, List[str]] = {}
    for sec, (s, e) in sections.items():
        child_keys = []
        base_indent = None
        for i in range(s + 1, e):
            line = lines[i]
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            m = INDENTED_KEY_RE.match(line)
            if not m:
                continue
            indent, key = m.group(1), m.group(2)
            if base_indent is None:
                base_indent = indent
            if len(indent) == len(base_indent):
                child_keys.append(key)
        # dedupe while preserving order
        seen = set()
        uniq = []
        for k in child_keys:
            if k not in seen:
                seen.add(k)
                uniq.append(k)
        outline[sec] = uniq
    return outline
