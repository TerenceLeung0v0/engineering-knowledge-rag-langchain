from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

def _normalize_page(v: Any) -> int | None:
    if v is None:
        return None

    if isinstance(v, int):
        return v

    if isinstance(v, str):
        s = v.strip()
        if s.isdigit():
            return int(s)

    try:
        return int(v)
    except Exception:
        return None

@dataclass(frozen=True)
class DocMetadataTemplate:
    default_source: str = "unknown"
    default_tags: tuple[str, ...] = ()

    def apply(
        self,
        doc: Document,
        *,
        source: str | None = None,
        doc_id: str | None = None
    ) -> Document:
        """
        Document.metadata will contain:
        - source (str): Origin of the doc (default: 'unknown').
        - doc_id (str): Unique identifier.
        - title (str): Document title (default: "").
        - tags (list[str]): List of strings for categorization.
        - entities (list): Named entities extracted (default: []).
        """
        meta: dict[str, Any] = dict(doc.metadata or {})

        meta.setdefault("source", source or self.default_source)
        meta.setdefault("doc_id", doc_id or meta.get("doc_id") or meta.get("source") or self.default_source)
        meta.setdefault("title", meta.get("title") or "")
        meta["page"] = _normalize_page(meta.get("page"))
        meta.setdefault("section", meta.get("section") or "")

        if "tags" not in meta:
            meta["tags"] = list(self.default_tags)
        elif isinstance(meta["tags"], tuple):
            meta["tags"] = list(meta["tags"])
        elif not isinstance(meta["tags"], list):
            meta["tags"] = [str(meta["tags"])]

        meta.setdefault("entities", [])

        return Document(
            page_content=doc.page_content,
            metadata=meta
        )
