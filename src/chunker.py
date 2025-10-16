from typing import Dict, List
import hashlib
import re

def _normalize(text: str) -> str:
    # Strip trailing spaces, collapse multiple blank lines, preserve indentation
    lines = [(ln.rstrip()) for ln in (text or "").splitlines()]
    out_lines: List[str] = []
    blank = False
    for ln in lines:
        if ln == "":
            if not blank:
                out_lines.append("")
            blank = True
        else:
            out_lines.append(ln)
            blank = False
    return "\n".join(out_lines).strip("\n")

def _sliding_windows(s: str, max_chars: int, overlap: int) -> List[tuple]:
    if not s:
        return []
    n = len(s)
    if n <= max_chars:
        return [(0, n)]

    wins: List[tuple] = []
    start = 0
    last_end = -1
    while start < n:
        # ensure forward progress
        if start <= last_end:
            start = last_end + 1
        end = min(start + max_chars, n)

        # try to end on a line boundary if possible
        if end < n:
            rel = s[start:end]
            pos = rel.rfind("\n")
            if pos != -1 and pos > 50:  # avoid ultra tiny tail if boundary too close
                end = start + pos

        if end <= start:
            # fallback to force progress
            end = min(start + max_chars, n)

        wins.append((start, end))
        last_end = end
        if end >= n:
            break

        # next start with overlap; align to a line start if possible
        next_start = max(0, end - overlap)
        align = s.rfind("\n", 0, next_start)
        if align != -1:
            next_start = align + 1
        start = next_start

    return wins

def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def make_chunks(filepath: str, full_text: str, sections: Dict[str, str], max_chars: int = 3000, overlap: int = 200) -> List[Dict]:
    """
    Returns a list of chunks with text and metadata.
    Each chunk: { "text": str, "metadata": { "file": str, "section": str, "span": {"start": int, "end": int}, "role": "section"|"full", "content_hash": str } }
    - No FILE:/SECTION: headers are included in the text; those stay only in metadata.
    """
    chunks: List[Dict] = []
    seen_hashes = set()

    for sec, body in sections.items():
        # Normalize body for better embeddings and stable offsets
        normalized = _normalize(body or "")
        if not normalized:
            continue

        for start, end in _sliding_windows(normalized, max_chars=max_chars, overlap=overlap):
            text = normalized[start:end]
            # Drop tiny chunks post-normalization
            if len(text) < 50:
                continue
            h = _hash_text(text)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            chunks.append({
                "text": text,
                "metadata": {
                    "file": filepath,
                    "section": sec,
                    "span": {"start": start, "end": end},
                    "role": "section",
                    "content_hash": h
                }
            })

    # Whole-file chunk with raw content (no headers)
    full_hash = _hash_text(full_text or "")
    chunks.append({
        "text": full_text,
        "metadata": {
            "file": filepath,
            "section": "__full__",
            "span": {"start": 0, "end": len(full_text or "")},
            "role": "full",
            "content_hash": full_hash
        }
    })

    return chunks
