from __future__ import annotations
from pathlib import Path
from typing import Iterable, Any

from langchain_core.documents import Document

from src.schemas import SourceRef
from src.rag.output_cleaner import clean_rag_output

import re

_SOURCES_HEADER_RE = re.compile(r"^\s*sources\s*:\s*$", re.IGNORECASE)

def normalize_page(page: Any) -> int | str:
    if page is None:
        return "n/a"
    
    if isinstance(page, int):
        return page
    
    if isinstance(page, float) and page.is_integer():
        return int(page)
    
    try:
        import numpy as np
        if isinstance(page, np.integer):
            return int(page)
    except Exception:
        pass
    
    s = str(page).strip()
    
    return s if s else "n/a"

def format_docs_for_prompt(
    docs: Iterable[Document],
    *, max_chars_per_chunk:int
) -> str:
    parts: list[str] = []
      
    for d in docs:
        meta = d.metadata or {}
        src = Path(meta.get("source", "unknown")).name
        page = normalize_page(meta.get("page"))
        
        parts.append(f"[{src}, page {page}]\n{d.page_content[:max_chars_per_chunk]}")

    return "\n\n".join(parts)

def collect_sources(docs: Iterable[Document]) -> list[SourceRef]:
    seen: set[tuple[str, int | str]] = set()
    out: list[SourceRef] = []

    for d in docs:
        meta = d.metadata or {}
        filename = Path(meta.get("source", "unknown")).name
        page = normalize_page(meta.get("page"))

        key = (filename, page)
        if key in seen:
            continue

        seen.add(key)
        out.append(SourceRef(filename, page))

    out.sort(key=lambda s: (s.filename, str(s.page)))
    
    return out

def normalize_answer_for_cli(text: str) -> str:
    cleaned = clean_rag_output(text).text
    out = []
    
    for ln in cleaned.splitlines():
        if _SOURCES_HEADER_RE.match(ln.strip()):
            break
        out.append(ln.rstrip())
    return "\n".join(out).strip()
