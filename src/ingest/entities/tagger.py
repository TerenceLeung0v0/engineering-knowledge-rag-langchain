from __future__ import annotations
from typing import Iterable

from langchain_core.documents import Document

from src.ingest.entities.schemas import EntityRegistry

import re

def _normalize_text(x: object) -> str:
    return str(x or "").strip()

def _count_hits(
    patterns: Iterable[re.Pattern[str]],
    text: str
) -> int:
    hits = 0
    for pattern in patterns:
        if pattern.search(text):
            hits += 1
    return hits

def tag_entities_for_docs(
    docs: Iterable[Document],
    *, registry: EntityRegistry,
) -> list[Document]:
    out: list[Document] = []

    for doc in docs:
        meta = dict(doc.metadata or {})
        text = _normalize_text(doc.page_content)

        found: list[str] = []
        for key, spec in registry.entities.items():
            if not spec.doc_aliases:
                continue
            hits = _count_hits(spec.doc_aliases, text)
            if hits >= spec.doc_min_hits:
                found.append(key)

        meta["entities"] = sorted(set(found))
        out.append(Document(
            page_content=doc.page_content,
            metadata=meta
        ))

    return out
